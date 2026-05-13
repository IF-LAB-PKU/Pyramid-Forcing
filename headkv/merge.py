"""MergeStrategy — spatiotemporal block merging for middle anchors.

Merge blocks are defined by:
- temporal block length = patch_size ** 2 frames
- spatial block size = patch_size x patch_size tokens within each frame

Each completed temporal block contributes one merged token per spatial patch.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import torch

from .base import CollectedAnchor


@dataclass
class _MergeBlock:
    start_t: int
    end_t: int
    median_t: int
    seen_slots: list[bool]
    complete_count: int = 0
    group_ids_frame: torch.Tensor | None = None
    output_pos: torch.Tensor | None = None
    tokens_per_group: torch.Tensor | None = None
    sum_k: torch.Tensor | None = None
    sum_v: torch.Tensor | None = None
    merged_anchor: CollectedAnchor | None = None


class MergeStrategy:
    """Patch-merge middle strategy.

    Args:
        patch_size: Spatial patch edge length `s`; temporal block length is `s * s`.
        capacity: Number of completed temporal blocks to retain.
            -1 means unlimited.
        dynamic_rope: Kept for config compatibility. Merge now always uses block-median time.
    """

    def __init__(self, patch_size: int = 2, capacity: int = 1, dynamic_rope: bool = True):
        self.patch_size = max(1, int(patch_size))
        self.capacity = -1 if int(capacity) < 0 else max(1, int(capacity))
        self.dynamic_rope = bool(dynamic_rope)
        self.block_frames = self.patch_size * self.patch_size
        # per (batch*head) -> OrderedDict[block_id -> _MergeBlock]
        self._blocks: list[OrderedDict[int, _MergeBlock]] = []
        self._complete_block_ids: list[deque[int]] = []
        self._complete_block_sets: list[set[int]] = []
        # Cached grid metadata. Frame layout is identical across blocks of one run,
        # so patch_cols / num_groups only need to be derived once (one .item() sync
        # at warmup) and can be reused sync-free during steady-state (required for
        # CUDA Graph capture).
        self._patch_cols: int | None = None
        self._num_groups: int | None = None

    def reset(self, num_seq: int) -> None:
        self._blocks = [OrderedDict() for _ in range(num_seq)]
        self._complete_block_ids = [deque() for _ in range(num_seq)]
        self._complete_block_sets = [set() for _ in range(num_seq)]

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return

        blocks = self._blocks[idx]
        complete_ids = self._complete_block_ids[idx]
        complete_set = self._complete_block_sets[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_vals = pos_seq[::frame_seqlen, 0].long().tolist()

        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            t_val = int(t_vals[frame_idx])
            block_id = t_val // self.block_frames
            start_t = block_id * self.block_frames
            end_t = start_t + self.block_frames - 1
            block = blocks.get(block_id)
            if block is None:
                block = _MergeBlock(
                    start_t=start_t,
                    end_t=end_t,
                    median_t=(start_t + end_t) // 2,
                    seen_slots=[False] * self.block_frames,
                )
                blocks[block_id] = block
            local_idx = t_val - start_t
            if local_idx < 0 or local_idx >= self.block_frames:
                continue

            if block.seen_slots[local_idx]:
                raise ValueError(
                    f"Duplicate merge frame slot for seq={idx}, block={block_id}, t={t_val}."
                )

            frame_k = k_seq[start:end]
            frame_v = v_seq[start:end]
            frame_pos = pos_seq[start:end]

            if block.group_ids_frame is None:
                group_ids_frame, output_pos = self._build_patch_groups(frame_pos, t_value=block.median_t)
                num_groups = int(output_pos.shape[0])
                block.group_ids_frame = group_ids_frame
                block.output_pos = output_pos
                # `torch.bincount` is implicitly synchronizing (reads input.max() to size
                # output). Replace with scatter_add_ which is sync-free — required for
                # CUDA Graph capture of the steady-state forward path.
                counts_frame = torch.zeros(
                    num_groups, device=group_ids_frame.device, dtype=frame_k.dtype,
                )
                counts_frame.scatter_add_(
                    0, group_ids_frame, torch.ones_like(group_ids_frame, dtype=frame_k.dtype),
                )
                block.tokens_per_group = counts_frame.clamp_min_(1).unsqueeze(1) * self.block_frames
                block.sum_k = frame_k.new_zeros((num_groups, frame_k.shape[-1]))
                block.sum_v = frame_v.new_zeros((num_groups, frame_v.shape[-1]))
            elif frame_k.shape[0] != block.group_ids_frame.shape[0]:
                raise ValueError(
                    f"Inconsistent merge frame shape for seq={idx}, block={block_id}, t={t_val}: "
                    f"expected {block.group_ids_frame.shape[0]} tokens, got {frame_k.shape[0]}."
                )

            block.sum_k.index_add_(0, block.group_ids_frame, frame_k)
            block.sum_v.index_add_(0, block.group_ids_frame, frame_v)
            block.seen_slots[local_idx] = True
            block.complete_count += 1

            if block.complete_count == self.block_frames:
                self._finalize_block(block)
                if block_id not in complete_set:
                    complete_ids.append(block_id)
                    complete_set.add(block_id)
        if self.capacity > 0:
            while len(complete_ids) > self.capacity:
                drop_id = complete_ids.popleft()
                complete_set.discard(drop_id)
                blocks.pop(drop_id, None)

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        blocks = self._blocks[idx]

        for block in blocks.values():
            if block.merged_anchor is None:
                continue
            if block.start_t <= sink_max_t:
                continue
            if block.end_t >= recent_min_t:
                continue
            result.append(block.merged_anchor)
        return result

    def _finalize_block(self, block: _MergeBlock) -> None:
        if (
            block.sum_k is None
            or block.sum_v is None
            or block.output_pos is None
            or block.tokens_per_group is None
        ):
            raise RuntimeError("Cannot finalize merge block without accumulated state.")

        merged_k = block.sum_k / block.tokens_per_group
        merged_v = block.sum_v / block.tokens_per_group
        block.merged_anchor = CollectedAnchor(
            kind="merge",
            t=block.median_t,
            dynamic_rope=False,
            k=merged_k,
            v=merged_v,
            pos=block.output_pos,
            token_count=int(block.output_pos.shape[0]),
        )
        block.group_ids_frame = None
        block.output_pos = None
        block.tokens_per_group = None
        block.sum_k = None
        block.sum_v = None

    def _build_patch_groups(
        self,
        frame_pos: torch.Tensor,
        t_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = frame_pos[:, 1].to(dtype=torch.long)
        x = frame_pos[:, 2].to(dtype=torch.long)
        patch_y = torch.div(y, self.patch_size, rounding_mode="floor")
        patch_x = torch.div(x, self.patch_size, rounding_mode="floor")
        if self._patch_cols is None:
            # First call only — pays one .item() sync during warmup, then cached.
            self._patch_cols = int(patch_x.max().item()) + 1 if patch_x.numel() > 0 else 1
        patch_cols = self._patch_cols
        merge_group_ids = patch_y * patch_cols + patch_x
        if self._num_groups is None:
            self._num_groups = int(merge_group_ids.max().item()) + 1 if merge_group_ids.numel() > 0 else 0
        num_groups = self._num_groups
        output_pos = torch.zeros((num_groups, 3), dtype=torch.long, device=frame_pos.device)
        output_pos[:, 0] = t_value
        output_pos[:, 1] = torch.div(torch.arange(num_groups, device=frame_pos.device), patch_cols, rounding_mode="floor") * self.patch_size
        output_pos[:, 2] = torch.remainder(torch.arange(num_groups, device=frame_pos.device), patch_cols) * self.patch_size
        return merge_group_ids, output_pos

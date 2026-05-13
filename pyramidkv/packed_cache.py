"""PyramidKVPackedCache: minimal Python wrapper around the C++ plan/pack/update ops.

This is a **vertical slice**: it does not replace ``AdaptiveKVCache``. Strategy
is hard-coded to a simple FIFO recent buffer that caps at
``max_recent_frames``. It exists to validate that ``pyramidkv_plan``,
``pyramidkv_pack`` and ``pyramidkv_update`` interoperate on realistic-shape data
without per-call Python loops, and to give CUDA-Graph-capture code a clean
handle to wrap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import _ops


@dataclass
class _PackedReadout:
    k_flat: torch.Tensor      # [total_tokens, 1, head_dim]
    v_flat: torch.Tensor      # [total_tokens, 1, head_dim]
    cu_seqlens_k: torch.Tensor  # [num_layers * num_heads + 1] int32
    total_tokens: int


class PyramidKVPackedCache:
    """Minimal cache backed by torch.classes.adahead.PyramidKVCacheManager.

    Strategy: FIFO recent only.
        - All incoming frames go to recent_pool.
        - When recent fills, oldest slot is overwritten (ring buffer).
        - sink_pool / middle_pool stay empty (placeholder for future strategies).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        frame_seqlen: int,
        max_recent_frames: int = 4,
        max_sink_frames: int = 1,
        max_middle_frames: int = 1,
        device: str = "cuda:0",
        kv_dtype: str = "bfloat16",
    ):
        if not _ops.available():
            raise RuntimeError("adahead C++ extension not available")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.frame_seqlen = frame_seqlen
        self.max_recent = max_recent_frames
        self.max_sink = max_sink_frames
        self.max_middle = max_middle_frames

        Cls = torch.classes.adahead.PyramidKVCacheManager
        # Plan B — packed_cache is a vertical slice / fixture (single-chunk
        # plan + pack), so 1 chunk is enough. The all-layer pack here uses
        # cu = L*H+1 entries (fits when max_attend_chunks >= L).
        self._mgr = Cls(
            num_layers, num_heads, head_dim, frame_seqlen,
            max_sink_frames, max_middle_frames, max_recent_frames,
            device, kv_dtype,
            num_layers,
        )
        # Tracks the next slot per (layer, head) for FIFO writes.
        # Mirrored on CPU because plan currently reads valid_count via .cpu().
        # In M2 graph capture this will be migrated to a device tensor.
        self._next_slot = [
            [0] * num_heads for _ in range(num_layers)
        ]

    @property
    def manager(self):
        return self._mgr

    def reset(self) -> None:
        self._mgr.reset()
        self._next_slot = [
            [0] * self.num_heads for _ in range(self.num_layers)
        ]

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,   # [frames_in_block, num_heads, frame_seqlen, head_dim]
        new_v: torch.Tensor,
    ) -> None:
        """Append `frames_in_block` frames to recent_pool for the given layer.

        Strategy: FIFO. Slot index = next_slot, then next_slot = (next_slot + 1) % max_recent.
        valid_count grows up to max_recent then saturates.
        """
        assert new_k.dim() == 4 and new_v.dim() == 4, "new_k/new_v must be [frames, H, F, D]"
        frames_in_block, H, F, D = new_k.shape
        assert H == self.num_heads, f"head count mismatch: {H} vs {self.num_heads}"
        assert F == self.frame_seqlen, f"frame_seqlen mismatch: {F} vs {self.frame_seqlen}"
        assert D == self.head_dim, f"head_dim mismatch: {D} vs {self.head_dim}"

        # Build per-(frame, head) write descriptor.
        N = frames_in_block * H
        dst_kind_list = [2] * N                      # all to recent pool
        dst_slot_list: list[int] = []
        src_frame_list: list[int] = []
        src_head_list: list[int] = []

        for f in range(frames_in_block):
            for h in range(H):
                slot_in_pool = self._next_slot[layer_idx][h]
                slot_global = (layer_idx * self.num_heads + h) * self.max_recent + slot_in_pool
                dst_slot_list.append(slot_global)
                src_frame_list.append(f)
                src_head_list.append(h)
                self._next_slot[layer_idx][h] = (slot_in_pool + 1) % self.max_recent

        device = new_k.device
        dst_kind = torch.tensor(dst_kind_list, dtype=torch.int32, device=device)
        dst_slot = torch.tensor(dst_slot_list, dtype=torch.int32, device=device)
        src_frame = torch.tensor(src_frame_list, dtype=torch.int32, device=device)
        src_head = torch.tensor(src_head_list, dtype=torch.int32, device=device)

        # packed_cache doesn't track raw-K pos — pass empty placeholder.
        _empty_pos = torch.empty(0, dtype=torch.int64, device=device)
        torch.ops.adahead.pyramidkv_update(
            self._mgr, new_k, new_v, _empty_pos,
            dst_kind, dst_slot, src_frame, src_head
        )

        # Bump valid_count up to max_recent (saturates).
        vc = self._mgr.valid_count()  # [L, H, 3] int64 on device
        # In-place add then clamp.
        vc[layer_idx, :, 2] = torch.clamp(
            vc[layer_idx, :, 2] + frames_in_block, max=self.max_recent
        )

    def readout(self, current_t: int = 0, pass_kind: int = 0) -> _PackedReadout:
        """Plan + pack: produce flat K/V over all (layer, head) heads.

        Returned tensors are views into the manager's preallocated workspace
        and remain valid until the next readout / reset.
        """
        from . import _ops as _ops_mod
        cu, sk, sg, sl, dst = torch.ops.adahead.pyramidkv_plan(
            self._mgr, int(current_t), int(pass_kind)
        )
        _ops_mod.pyramidkv_pack(self._mgr, sk, sg, sl, dst)
        # total_tokens = cu_seqlens_k[-1]
        total = int(cu[-1].item())
        k_flat = self._mgr.k_flat_out()[:total]
        v_flat = self._mgr.v_flat_out()[:total]
        return _PackedReadout(
            k_flat=k_flat,
            v_flat=v_flat,
            cu_seqlens_k=cu,
            total_tokens=total,
        )

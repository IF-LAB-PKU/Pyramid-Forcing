"""Opt-in compact middle-strategy manager for PyramidKV.

The manager intentionally mirrors the public semantics of the Python middle
strategies while keeping anchor state in compact per-sequence buffers.  The
AdaptiveKVCache integration only enables it for CUDA tensors when the compiled
extension is available; tests can instantiate it with relaxed requirements.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os

import torch

from ._scatter_ext import anchor_store_write_frames, cuda_refresh_available
from .base import CollectedAnchor
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .merge import MergeStrategy
from .stride import StrideStrategy


KIND_NONE = 0
KIND_CYCLIC = 1
KIND_STRIDE = 2
KIND_MERGE = 3
KIND_UNSUPPORTED = -1


@dataclass(frozen=True)
class CppStrategyPolicy:
    kind: int
    sink_frames: int
    recent_frames: int
    dynamic_rope: bool = False
    period: int = 1
    bucket_cap: int = 1
    interval: int = 1
    capacity: int = -1
    patch_size: int = 1
    block_frames: int = 1


@dataclass
class _FrameStore:
    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    t: list[int | None]
    valid: list[bool]


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
    merged: CollectedAnchor | None = None


@dataclass(frozen=True)
class CppAnchorCount:
    """Compact readout metadata for managed middle anchors."""

    token_count: int
    anchor_count: int
    anchor_lengths: tuple[int, ...]
    dynamic_rope: bool
    kind: int


def cpp_strategy_requested() -> bool:
    return os.environ.get("PYRAMIDKV_CPP_STRATEGY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def compile_cpp_strategy_policies(
    compositions_row,
) -> tuple[list[CppStrategyPolicy], list[bool]]:
    """Compile HeadComposition objects into compact per-head policies.

    Returns `(policies, supported_middle_heads)`.  Heads without a middle
    strategy are valid `none` policies but are not counted as supported middle
    heads.  Lag and mixed-middle heads deliberately remain on the Python path.
    """
    policies: list[CppStrategyPolicy] = []
    supported: list[bool] = []
    if compositions_row is None:
        return policies, supported

    for comp in compositions_row:
        strategies = list(getattr(comp, "middle_strategies", ()) or ())
        sink_frames = max(0, int(getattr(comp, "sink_frames", 0)))
        recent_frames = max(1, int(getattr(comp, "recent_frames", 1)))
        if len(strategies) == 0:
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_NONE, sink_frames=sink_frames, recent_frames=recent_frames
                )
            )
            supported.append(False)
            continue
        if len(strategies) != 1 or any(isinstance(s, LagStrategy) for s in strategies):
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_UNSUPPORTED,
                    sink_frames=sink_frames,
                    recent_frames=recent_frames,
                )
            )
            supported.append(False)
            continue

        strategy = strategies[0]
        if isinstance(strategy, CyclicStrategy):
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_CYCLIC,
                    sink_frames=sink_frames,
                    recent_frames=recent_frames,
                    dynamic_rope=bool(strategy.dynamic_rope),
                    period=max(1, int(strategy.period)),
                    bucket_cap=max(1, int(strategy.bucket_cap)),
                )
            )
            supported.append(True)
        elif isinstance(strategy, StrideStrategy):
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_STRIDE,
                    sink_frames=sink_frames,
                    recent_frames=recent_frames,
                    dynamic_rope=bool(strategy.dynamic_rope),
                    interval=max(1, int(strategy.interval)),
                    capacity=int(strategy.capacity),
                )
            )
            supported.append(True)
        elif isinstance(strategy, MergeStrategy):
            patch_size = max(1, int(strategy.patch_size))
            capacity = (
                -1 if int(strategy.capacity) < 0 else max(1, int(strategy.capacity))
            )
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_MERGE,
                    sink_frames=sink_frames,
                    recent_frames=recent_frames,
                    dynamic_rope=False,
                    capacity=capacity,
                    patch_size=patch_size,
                    block_frames=patch_size * patch_size,
                )
            )
            supported.append(True)
        else:
            policies.append(
                CppStrategyPolicy(
                    kind=KIND_UNSUPPORTED,
                    sink_frames=sink_frames,
                    recent_frames=recent_frames,
                )
            )
            supported.append(False)
    return policies, supported


class CppStrategyManager:
    """Compact per-cache manager for cyclic, stride, and merge middle state."""

    def __init__(
        self,
        policies: list[CppStrategyPolicy],
        *,
        num_seq: int,
        num_heads: int,
        head_dim: int,
        require_cuda: bool = True,
        require_extension: bool = True,
    ) -> None:
        self.policies = policies
        self.num_seq = int(num_seq)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.require_cuda = bool(require_cuda)
        self.require_extension = bool(require_extension)
        self._extension_checked = False
        self._extension_ok = False
        self._active = False
        self._frame_seqlen: int | None = None

        self._stores: list[_FrameStore | None] = [None] * self.num_seq
        self._cyclic_orders: list[list[list[int]] | None] = [None] * self.num_seq
        self._cyclic_cursors: list[list[int] | None] = [None] * self.num_seq
        self._stride_maps: list[OrderedDict[int, int] | None] = [None] * self.num_seq
        self._merge_blocks: list[OrderedDict[int, _MergeBlock] | None] = [
            None
        ] * self.num_seq
        self._merge_complete_ids: list[list[int] | None] = [None] * self.num_seq
        self.stats = {
            "cpp_strategy_update_count": 0.0,
            "cpp_strategy_collect_count": 0.0,
            "cpp_strategy_anchor_count": 0.0,
            "cpp_strategy_token_count": 0.0,
            "cpp_strategy_materialize_count": 0.0,
            "cpp_strategy_materialize_token_count": 0.0,
        }

    @property
    def has_supported_middle(self) -> bool:
        return any(
            p.kind in (KIND_CYCLIC, KIND_STRIDE, KIND_MERGE) for p in self.policies
        )

    def head_supported(self, head_idx: int) -> bool:
        return 0 <= head_idx < len(self.policies) and self.policies[head_idx].kind in (
            KIND_CYCLIC,
            KIND_STRIDE,
            KIND_MERGE,
        )

    def reset(self, num_seq: int | None = None) -> None:
        if num_seq is not None and int(num_seq) != self.num_seq:
            self.num_seq = int(num_seq)
        self._frame_seqlen = None
        self._active = False
        self._stores = [None] * self.num_seq
        self._cyclic_orders = [None] * self.num_seq
        self._cyclic_cursors = [None] * self.num_seq
        self._stride_maps = [None] * self.num_seq
        self._merge_blocks = [None] * self.num_seq
        self._merge_complete_ids = [None] * self.num_seq

    def pop_stats(self) -> dict[str, float]:
        stats = dict(self.stats)
        for key in self.stats:
            self.stats[key] = 0.0
        return stats

    def usable_for(self, tensor: torch.Tensor) -> bool:
        if self.require_cuda and not tensor.is_cuda:
            return False
        if self.require_extension:
            if not self._extension_checked:
                self._extension_ok = cuda_refresh_available()
                self._extension_checked = True
            if not self._extension_ok:
                return False
        return True

    def update_all(
        self,
        *,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        pos_flat: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
    ) -> bool:
        if frame_seqlen <= 0 or k_flat.shape[1] < frame_seqlen:
            return False
        if k_flat.shape[1] % frame_seqlen != 0:
            return False
        if not self.usable_for(k_flat):
            return False
        self._frame_seqlen = int(frame_seqlen)
        any_updated = False
        for seq_idx in range(min(self.num_seq, int(k_flat.shape[0]))):
            head_idx = seq_idx % self.num_heads
            if not self.head_supported(head_idx):
                continue
            policy = self.policies[head_idx]
            if policy.kind == KIND_CYCLIC:
                self._update_cyclic(
                    seq_idx,
                    policy,
                    k_flat[seq_idx],
                    v_flat[seq_idx],
                    pos_flat[seq_idx],
                    frame_seqlen,
                    current_t,
                )
            elif policy.kind == KIND_STRIDE:
                self._update_stride(
                    seq_idx,
                    policy,
                    k_flat[seq_idx],
                    v_flat[seq_idx],
                    pos_flat[seq_idx],
                    frame_seqlen,
                    current_t,
                )
            elif policy.kind == KIND_MERGE:
                self._update_merge(
                    seq_idx,
                    policy,
                    k_flat[seq_idx],
                    v_flat[seq_idx],
                    pos_flat[seq_idx],
                    frame_seqlen,
                    current_t,
                )
            any_updated = True
        if any_updated:
            self._active = True
            self.stats["cpp_strategy_update_count"] += 1.0
        return any_updated

    def collect(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor] | None:
        if not self.head_supported(head_idx):
            return None
        if not self._active:
            return None
        policy = self.policies[head_idx]
        if policy.kind in (KIND_CYCLIC, KIND_STRIDE) and self._stores[seq_idx] is None:
            anchors: list[CollectedAnchor] = []
        elif policy.kind == KIND_CYCLIC:
            anchors = self._collect_cyclic(
                seq_idx, policy, current_t, recent_min_t, sink_max_t
            )
        elif policy.kind == KIND_STRIDE:
            anchors = self._collect_stride(seq_idx, policy, recent_min_t, sink_max_t)
        elif policy.kind == KIND_MERGE:
            anchors = self._collect_merge(seq_idx, recent_min_t, sink_max_t)
        else:
            return None
        self.stats["cpp_strategy_collect_count"] += 1.0
        self.stats["cpp_strategy_anchor_count"] += float(len(anchors))
        self.stats["cpp_strategy_token_count"] += float(
            sum(a.token_count for a in anchors)
        )
        return anchors

    def count_anchors(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> CppAnchorCount | None:
        """Count managed anchors without building CollectedAnchor objects."""
        if not self.head_supported(head_idx):
            return None
        policy = self.policies[head_idx]
        if not self._active:
            self.stats["cpp_strategy_collect_count"] += 1.0
            return CppAnchorCount(
                token_count=0,
                anchor_count=0,
                anchor_lengths=(),
                dynamic_rope=bool(policy.dynamic_rope),
                kind=int(policy.kind),
            )
        refs = self._select_anchor_refs(
            seq_idx=seq_idx,
            policy=policy,
            current_t=current_t,
            recent_min_t=recent_min_t,
            sink_max_t=sink_max_t,
        )
        if refs is None:
            return None
        lengths = tuple(int(ref[4]) for ref in refs)
        token_count = int(sum(lengths))
        self.stats["cpp_strategy_collect_count"] += 1.0
        self.stats["cpp_strategy_anchor_count"] += float(len(lengths))
        self.stats["cpp_strategy_token_count"] += float(token_count)
        return CppAnchorCount(
            token_count=token_count,
            anchor_count=len(lengths),
            anchor_lengths=lengths,
            dynamic_rope=bool(policy.dynamic_rope),
            kind=int(policy.kind),
        )

    def materialize_anchors(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
        out_pos: torch.Tensor,
        out_frame_ids: torch.Tensor,
        offset: int,
        dynamic_rope_t: int | None,
        capture_physical: bool,
    ) -> int:
        """Write managed anchors directly into readout workspaces.

        K is written in raw, pre-RoPE form.  The caller runs the normal batched
        RoPE pass after all static/dynamic/anchor segments have been packed.
        """
        if not self.head_supported(head_idx):
            raise RuntimeError(f"Head {head_idx} is not managed by CppStrategyManager.")
        if not self._active:
            raise RuntimeError("CppStrategyManager is inactive.")
        policy = self.policies[head_idx]
        refs = self._select_anchor_refs(
            seq_idx=seq_idx,
            policy=policy,
            current_t=current_t,
            recent_min_t=recent_min_t,
            sink_max_t=sink_max_t,
        )
        if refs is None:
            raise RuntimeError("CppStrategyManager could not select anchors.")

        write_offset = int(offset)
        for kind, payload, _t_val, dyn_rope, length in refs:
            n = int(length)
            if n <= 0:
                continue
            if kind == KIND_CYCLIC or kind == KIND_STRIDE:
                store = self._stores[seq_idx]
                if store is None:
                    raise RuntimeError("Missing frame store for managed anchor.")
                slot = int(payload)
                anchor_k = store.k[slot]
                anchor_v = store.v[slot]
                anchor_pos = store.pos[slot]
            elif kind == KIND_MERGE:
                block = payload
                if block.merged is None:
                    raise RuntimeError("Missing finalized merge anchor.")
                anchor_k = block.merged.k
                anchor_v = block.merged.v
                anchor_pos = block.merged.pos
                if anchor_k is None or anchor_v is None or anchor_pos is None:
                    raise RuntimeError("Malformed finalized merge anchor.")
            else:
                raise RuntimeError(f"Unsupported managed anchor kind: {kind}")

            end = write_offset + n
            out_k[write_offset:end].copy_(anchor_k)
            out_v[write_offset:end].copy_(anchor_v)
            out_pos[write_offset:end].copy_(anchor_pos)
            if dyn_rope and dynamic_rope_t is not None:
                out_pos[write_offset:end, 0] = int(dynamic_rope_t)
            if capture_physical:
                out_frame_ids[write_offset:end].copy_(
                    anchor_pos[:, 0].to(dtype=torch.long)
                )
            else:
                out_frame_ids[write_offset:end].copy_(
                    out_pos[write_offset:end, 0].to(dtype=torch.long)
                )
            write_offset = end

        token_count = write_offset - int(offset)
        self.stats["cpp_strategy_materialize_count"] += 1.0
        self.stats["cpp_strategy_materialize_token_count"] += float(token_count)
        return token_count

    def _select_anchor_refs(
        self,
        *,
        seq_idx: int,
        policy: CppStrategyPolicy,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[tuple[int, object, int, bool, int]] | None:
        """Return compact refs as (kind, payload, t, dynamic_rope, token_count)."""
        if policy.kind == KIND_CYCLIC:
            store = self._stores[seq_idx]
            orders = self._cyclic_orders[seq_idx]
            if store is None or orders is None:
                return []
            phase = int(current_t) % policy.period
            refs: list[tuple[int, object, int, bool, int]] = []
            for slot in orders[phase]:
                t_val = store.t[slot]
                if (
                    t_val is None
                    or not store.valid[slot]
                    or t_val <= sink_max_t
                    or t_val >= recent_min_t
                ):
                    continue
                refs.append(
                    (
                        KIND_CYCLIC,
                        int(slot),
                        int(t_val),
                        bool(policy.dynamic_rope),
                        int(store.k.shape[1]),
                    )
                )
            return refs

        if policy.kind == KIND_STRIDE:
            store = self._stores[seq_idx]
            mapping = self._stride_maps[seq_idx]
            if store is None or mapping is None:
                return []
            refs = []
            for t_val, slot in sorted(mapping.items()):
                if (
                    t_val <= sink_max_t
                    or t_val >= recent_min_t
                    or not store.valid[slot]
                ):
                    continue
                refs.append(
                    (
                        KIND_STRIDE,
                        int(slot),
                        int(t_val),
                        bool(policy.dynamic_rope),
                        int(store.k.shape[1]),
                    )
                )
            return refs

        if policy.kind == KIND_MERGE:
            blocks = self._merge_blocks[seq_idx]
            if blocks is None:
                return []
            refs = []
            for block in blocks.values():
                if (
                    block.merged is None
                    or block.start_t <= sink_max_t
                    or block.end_t >= recent_min_t
                ):
                    continue
                refs.append(
                    (
                        KIND_MERGE,
                        block,
                        int(block.median_t),
                        False,
                        int(block.merged.token_count),
                    )
                )
            return refs

        return None

    def _ensure_store(
        self,
        seq_idx: int,
        *,
        slots: int,
        frame_seqlen: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
    ) -> _FrameStore:
        store = self._stores[seq_idx]
        slots = max(1, int(slots))
        if (
            store is None
            or store.k.shape[0] < slots
            or store.k.shape[1] != frame_seqlen
            or store.k.device != k_seq.device
            or store.k.dtype != k_seq.dtype
        ):
            new_slots = slots if store is None else max(slots, store.k.shape[0] * 2)
            new_store = _FrameStore(
                k=torch.empty(
                    (new_slots, frame_seqlen, self.head_dim),
                    device=k_seq.device,
                    dtype=k_seq.dtype,
                ),
                v=torch.empty(
                    (new_slots, frame_seqlen, self.head_dim),
                    device=v_seq.device,
                    dtype=v_seq.dtype,
                ),
                pos=torch.empty(
                    (new_slots, frame_seqlen, 3),
                    device=pos_seq.device,
                    dtype=torch.long,
                ),
                t=[None] * new_slots,
                valid=[False] * new_slots,
            )
            if store is not None:
                old = store.k.shape[0]
                new_store.k[:old] = store.k
                new_store.v[:old] = store.v
                new_store.pos[:old] = store.pos
                new_store.t[:old] = store.t
                new_store.valid[:old] = store.valid
            self._stores[seq_idx] = new_store
            store = new_store
        return store

    def _write_frames(
        self,
        store: _FrameStore,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        frames: list[tuple[int, int, int]],
    ) -> None:
        if not frames:
            return
        if self.require_extension:
            desc = torch.tensor(
                [(f, slot) for f, slot, _t in frames],
                dtype=torch.long,
                device=k_seq.device,
            )
            anchor_store_write_frames(
                k_seq.contiguous(),
                v_seq.contiguous(),
                pos_seq.contiguous(),
                desc,
                store.k,
                store.v,
                store.pos,
            )
        else:
            for frame_idx, slot, _t in frames:
                start = frame_idx * frame_seqlen
                end = start + frame_seqlen
                store.k[slot].copy_(k_seq[start:end])
                store.v[slot].copy_(v_seq[start:end])
                store.pos[slot].copy_(pos_seq[start:end])
        for _frame_idx, slot, t_val in frames:
            store.t[slot] = int(t_val)
            store.valid[slot] = True

    def _update_cyclic(
        self,
        seq_idx: int,
        policy: CppStrategyPolicy,
        k_seq,
        v_seq,
        pos_seq,
        frame_seqlen,
        current_t,
    ) -> None:
        slots = policy.period * policy.bucket_cap
        store = self._ensure_store(
            seq_idx,
            slots=slots,
            frame_seqlen=frame_seqlen,
            k_seq=k_seq,
            v_seq=v_seq,
            pos_seq=pos_seq,
        )
        if self._cyclic_orders[seq_idx] is None:
            self._cyclic_orders[seq_idx] = [[] for _ in range(policy.period)]
            self._cyclic_cursors[seq_idx] = [0 for _ in range(policy.period)]
        orders = self._cyclic_orders[seq_idx]
        cursors = self._cyclic_cursors[seq_idx]
        assert orders is not None and cursors is not None
        frames: list[tuple[int, int, int]] = []
        for frame_idx in range(k_seq.shape[0] // frame_seqlen):
            t_val = int(current_t) + frame_idx
            phase = t_val % policy.period
            cursor = cursors[phase]
            slot = phase * policy.bucket_cap + cursor
            cursors[phase] = (cursor + 1) % policy.bucket_cap
            if slot in orders[phase]:
                orders[phase].remove(slot)
            orders[phase].append(slot)
            while len(orders[phase]) > policy.bucket_cap:
                old_slot = orders[phase].pop(0)
                store.valid[old_slot] = False
                store.t[old_slot] = None
            frames.append((frame_idx, slot, t_val))
        self._write_frames(store, k_seq, v_seq, pos_seq, frame_seqlen, frames)

    def _collect_cyclic(
        self,
        seq_idx: int,
        policy: CppStrategyPolicy,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        store = self._stores[seq_idx]
        orders = self._cyclic_orders[seq_idx]
        if store is None or orders is None:
            return []
        phase = int(current_t) % policy.period
        result: list[CollectedAnchor] = []
        for slot in orders[phase]:
            t_val = store.t[slot]
            if (
                t_val is None
                or not store.valid[slot]
                or t_val <= sink_max_t
                or t_val >= recent_min_t
            ):
                continue
            k = store.k[slot]
            result.append(
                CollectedAnchor(
                    kind="frame",
                    t=int(t_val),
                    dynamic_rope=policy.dynamic_rope,
                    k=k,
                    v=store.v[slot],
                    pos=store.pos[slot],
                    token_count=int(k.shape[0]),
                    source_kind="cpp_strategy",
                )
            )
        return result

    def _update_stride(
        self,
        seq_idx: int,
        policy: CppStrategyPolicy,
        k_seq,
        v_seq,
        pos_seq,
        frame_seqlen,
        current_t,
    ) -> None:
        capacity = int(policy.capacity)
        initial_slots = (
            capacity if capacity > 0 else max(16, k_seq.shape[0] // frame_seqlen)
        )
        store = self._ensure_store(
            seq_idx,
            slots=initial_slots,
            frame_seqlen=frame_seqlen,
            k_seq=k_seq,
            v_seq=v_seq,
            pos_seq=pos_seq,
        )
        if self._stride_maps[seq_idx] is None:
            self._stride_maps[seq_idx] = OrderedDict()
        mapping = self._stride_maps[seq_idx]
        assert mapping is not None
        frames: list[tuple[int, int, int]] = []
        for frame_idx in range(k_seq.shape[0] // frame_seqlen):
            t_val = int(current_t) + frame_idx
            if t_val % policy.interval != 0:
                continue
            preferred_slot = None
            if t_val in mapping:
                preferred_slot = mapping.pop(t_val)
            elif capacity > 0 and len(mapping) >= capacity:
                if mapping:
                    _old_t, preferred_slot = mapping.popitem(last=False)
            if preferred_slot is None:
                used_slots = set(mapping.values())
                free_slots = [
                    i
                    for i, valid in enumerate(store.valid)
                    if not valid and i not in used_slots
                ]
                if not free_slots:
                    store = self._ensure_store(
                        seq_idx,
                        slots=store.k.shape[0] + max(1, store.k.shape[0]),
                        frame_seqlen=frame_seqlen,
                        k_seq=k_seq,
                        v_seq=v_seq,
                        pos_seq=pos_seq,
                    )
                    used_slots = set(mapping.values())
                    free_slots = [
                        i
                        for i, valid in enumerate(store.valid)
                        if not valid and i not in used_slots
                    ]
                preferred_slot = free_slots.pop(0)
            store.valid[preferred_slot] = False
            frames.append((frame_idx, preferred_slot, t_val))
            mapping[t_val] = preferred_slot
        self._write_frames(store, k_seq, v_seq, pos_seq, frame_seqlen, frames)

    def _collect_stride(
        self,
        seq_idx: int,
        policy: CppStrategyPolicy,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        store = self._stores[seq_idx]
        mapping = self._stride_maps[seq_idx]
        if store is None or mapping is None:
            return []
        result: list[CollectedAnchor] = []
        for t_val, slot in sorted(mapping.items()):
            if t_val <= sink_max_t or t_val >= recent_min_t or not store.valid[slot]:
                continue
            k = store.k[slot]
            result.append(
                CollectedAnchor(
                    kind="frame",
                    t=int(t_val),
                    dynamic_rope=policy.dynamic_rope,
                    k=k,
                    v=store.v[slot],
                    pos=store.pos[slot],
                    token_count=int(k.shape[0]),
                    source_kind="cpp_strategy",
                )
            )
        return result

    def _update_merge(
        self,
        seq_idx: int,
        policy: CppStrategyPolicy,
        k_seq,
        v_seq,
        pos_seq,
        frame_seqlen,
        current_t,
    ) -> None:
        if self._merge_blocks[seq_idx] is None:
            self._merge_blocks[seq_idx] = OrderedDict()
            self._merge_complete_ids[seq_idx] = []
        blocks = self._merge_blocks[seq_idx]
        complete_ids = self._merge_complete_ids[seq_idx]
        assert blocks is not None and complete_ids is not None
        for frame_idx in range(k_seq.shape[0] // frame_seqlen):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            t_val = int(current_t) + frame_idx
            block_id = t_val // policy.block_frames
            start_t = block_id * policy.block_frames
            block = blocks.get(block_id)
            if block is None:
                block = _MergeBlock(
                    start_t=start_t,
                    end_t=start_t + policy.block_frames - 1,
                    median_t=(start_t + start_t + policy.block_frames - 1) // 2,
                    seen_slots=[False] * policy.block_frames,
                )
                blocks[block_id] = block
            local_idx = t_val - start_t
            if block.seen_slots[local_idx]:
                raise ValueError(
                    f"Duplicate merge frame slot for seq={seq_idx}, block={block_id}, t={t_val}."
                )
            frame_k = k_seq[start:end]
            frame_v = v_seq[start:end]
            frame_pos = pos_seq[start:end]
            if block.group_ids_frame is None:
                group_ids, output_pos = self._build_patch_groups(
                    frame_pos, policy.patch_size, block.median_t
                )
                num_groups = int(output_pos.shape[0])
                counts = torch.bincount(group_ids, minlength=num_groups).to(
                    device=frame_k.device, dtype=frame_k.dtype
                )
                block.group_ids_frame = group_ids
                block.output_pos = output_pos
                block.tokens_per_group = (
                    counts.clamp_min_(1).unsqueeze(1) * policy.block_frames
                )
                block.sum_k = frame_k.new_zeros((num_groups, frame_k.shape[-1]))
                block.sum_v = frame_v.new_zeros((num_groups, frame_v.shape[-1]))
            elif frame_k.shape[0] != block.group_ids_frame.shape[0]:
                raise ValueError(
                    f"Inconsistent merge frame shape for seq={seq_idx}, block={block_id}, t={t_val}: "
                    f"expected {block.group_ids_frame.shape[0]} tokens, got {frame_k.shape[0]}."
                )
            assert (
                block.group_ids_frame is not None
                and block.sum_k is not None
                and block.sum_v is not None
            )
            block.sum_k.index_add_(0, block.group_ids_frame, frame_k)
            block.sum_v.index_add_(0, block.group_ids_frame, frame_v)
            block.seen_slots[local_idx] = True
            block.complete_count += 1
            if block.complete_count == policy.block_frames:
                self._finalize_merge_block(block)
                if block_id not in complete_ids:
                    complete_ids.append(block_id)
        if policy.capacity > 0:
            while len(complete_ids) > policy.capacity:
                drop_id = complete_ids.pop(0)
                blocks.pop(drop_id, None)

    def _collect_merge(
        self, seq_idx: int, recent_min_t: int, sink_max_t: int
    ) -> list[CollectedAnchor]:
        blocks = self._merge_blocks[seq_idx]
        if blocks is None:
            return []
        result: list[CollectedAnchor] = []
        for block in blocks.values():
            if (
                block.merged is None
                or block.start_t <= sink_max_t
                or block.end_t >= recent_min_t
            ):
                continue
            result.append(block.merged)
        return result

    @staticmethod
    def _build_patch_groups(
        frame_pos: torch.Tensor, patch_size: int, t_value: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = frame_pos[:, 1].to(dtype=torch.long)
        x = frame_pos[:, 2].to(dtype=torch.long)
        patch_y = torch.div(y, patch_size, rounding_mode="floor")
        patch_x = torch.div(x, patch_size, rounding_mode="floor")
        patch_cols = int(patch_x.max().item()) + 1 if patch_x.numel() > 0 else 1
        group_ids = patch_y * patch_cols + patch_x
        num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
        output_pos = torch.zeros(
            (num_groups, 3), dtype=torch.long, device=frame_pos.device
        )
        output_pos[:, 0] = int(t_value)
        arange = torch.arange(num_groups, device=frame_pos.device)
        output_pos[:, 1] = (
            torch.div(arange, patch_cols, rounding_mode="floor") * patch_size
        )
        output_pos[:, 2] = torch.remainder(arange, patch_cols) * patch_size
        return group_ids, output_pos

    @staticmethod
    def _finalize_merge_block(block: _MergeBlock) -> None:
        if (
            block.sum_k is None
            or block.sum_v is None
            or block.output_pos is None
            or block.tokens_per_group is None
        ):
            raise RuntimeError("Cannot finalize merge block without accumulated state.")
        merged_k = block.sum_k / block.tokens_per_group
        merged_v = block.sum_v / block.tokens_per_group
        block.merged = CollectedAnchor(
            kind="merge",
            t=block.median_t,
            dynamic_rope=False,
            k=merged_k,
            v=merged_v,
            pos=block.output_pos,
            token_count=int(block.output_pos.shape[0]),
            source_kind="cpp_strategy",
        )
        block.group_ids_frame = None
        block.output_pos = None
        block.tokens_per_group = None
        block.sum_k = None
        block.sum_v = None

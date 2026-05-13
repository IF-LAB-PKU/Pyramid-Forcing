"""Python reference for mega_state_update kernel.

Mirrors the C++ logic in pyramidkv/csrc/anchor_store_update.cu byte-for-byte so
we can:
  1. Run a CPU-only correctness check against the existing strategy classes
     (cyclic.py / stride.py / lag.py / recent.py)
  2. Use as the oracle when the CUDA kernel ships in the next commit
  3. Validate edge cases (FIFO eviction, cyclic ring wrap, noisy-pass skip)
     without needing a GPU build

This is NOT the production path — it's a verification fixture. The real work
goes through torch.ops.adahead.mega_state_update once the binding lands.

Mirrors the field layout in pyramidkv/csrc/anchor_store.cuh:PerHeadState. Field
names match exactly so future maintenance is mechanical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Strategy kinds — must match enum StrategyKind in anchor_store.cuh
# ---------------------------------------------------------------------------
SK_RECENT = 0
SK_CYCLIC = 1
SK_STRIDE = 2
SK_LAG = 3
SK_MERGE = 4

# Compile-time bounds — must match the C++ constants
MAX_PHASE = 6
MAX_BUCKET = 4
MAX_T_KEYED = 24
MAX_LAG_OFFSETS = 4
MAX_MERGE_BLOCKS = 6
MAX_MERGE_ACTIVE = 2
MAX_MERGE_BLOCK_FRAMES = 4

# Destination kind encoding — must match scatter_copy_bind.cpp / pyramidkv_update.cu
# 0/1/2/-1 are pre-existing for sink/middle/recent/skip — used by the existing
# pyramidkv_update_kernel data-movement path. The MERGE_ACCUM kind is NEW: it
# tells the upcoming mega_merge_accum_kernel to scatter-add into the merge
# accumulator pool instead of doing a flat slot copy.
DST_KIND_SINK = 0
DST_KIND_MIDDLE = 1
DST_KIND_RECENT = 2
DST_KIND_SKIP = -1
DST_KIND_MERGE_ACCUM = 5  # frame contributes to an accumulating merge block


class MergeDuplicateError(ValueError):
    """Raised when a merge block sees the same local frame slot twice.

    Mirrors the ValueError at pyramidkv/merge.py:108 so callers can pin down
    a regression to the same boundary the Python implementation flagged.
    """


@dataclass
class PerHeadState:
    """Mirror of the C++ PerHeadState struct (POD)."""

    # ===== shared metadata =====
    kind: int = SK_RECENT
    # M5 step 5: per-head sink capacity. MegaCache.update routes per-head
    # (osc heads with sink_capacity=1 send frames 1,2 to recent, not sink).
    # mega_plan also dedups middle/merge anchors with t < sink_capacity.
    sink_capacity: int = 0
    period: int = 0
    bucket_cap: int = 0
    interval: int = 0
    capacity: int = 0
    patch_size: int = 0
    block_frames: int = 0

    lag_offsets: list[int] = field(default_factory=lambda: [0] * MAX_LAG_OFFSETS)
    lag_offset_count: int = 0

    # ===== Cyclic state =====
    cyclic_slot: list[int] = field(
        default_factory=lambda: [-1] * (MAX_PHASE * MAX_BUCKET)
    )
    cyclic_t: list[int] = field(
        default_factory=lambda: [0] * (MAX_PHASE * MAX_BUCKET)
    )
    cyclic_cursor: list[int] = field(default_factory=lambda: [0] * MAX_PHASE)

    # ===== Stride / Lag shared FIFO (ring buffer) =====
    tkey_slot: list[int] = field(default_factory=lambda: [-1] * MAX_T_KEYED)
    tkey_t: list[int] = field(default_factory=lambda: [0] * MAX_T_KEYED)
    tkey_count: int = 0   # saturates at capacity
    tkey_head: int = 0    # next-write slot, wraps mod capacity

    # ===== Merge state =====
    merge_completed_slot: list[int] = field(
        default_factory=lambda: [-1] * MAX_MERGE_BLOCKS
    )
    merge_completed_block_id: list[int] = field(
        default_factory=lambda: [-1] * MAX_MERGE_BLOCKS
    )
    merge_completed_count: int = 0
    merge_active_block_id: list[int] = field(
        default_factory=lambda: [-1] * MAX_MERGE_ACTIVE
    )
    merge_active_seen: list[list[bool]] = field(
        default_factory=lambda: [[False] * MAX_MERGE_BLOCK_FRAMES for _ in range(MAX_MERGE_ACTIVE)]
    )
    merge_active_complete_count: list[int] = field(
        default_factory=lambda: [0] * MAX_MERGE_ACTIVE
    )

    # Cached invariants
    cached_num_groups: int = 0


def _update_cyclic(state: PerHeadState, t_val: int) -> tuple[int, int]:
    """Returns (kind, slot). Matches anchor_store_update.cu:update_cyclic."""
    phase = t_val % state.period
    cursor = state.cyclic_cursor[phase]
    slot = phase * state.bucket_cap + cursor
    state.cyclic_slot[slot] = slot
    state.cyclic_t[slot] = t_val
    state.cyclic_cursor[phase] = (cursor + 1) % state.bucket_cap
    return DST_KIND_MIDDLE, slot


def _update_stride(state: PerHeadState, t_val: int) -> tuple[int, int]:
    if t_val % state.interval != 0:
        return DST_KIND_SKIP, -1
    slot = state.tkey_head
    state.tkey_head = (slot + 1) % state.capacity
    if state.tkey_count < state.capacity:
        state.tkey_count += 1
    state.tkey_slot[slot] = slot
    state.tkey_t[slot] = t_val
    return DST_KIND_MIDDLE, slot


def _update_lag(state: PerHeadState, t_val: int) -> tuple[int, int]:
    slot = state.tkey_head
    state.tkey_head = (slot + 1) % state.capacity
    if state.tkey_count < state.capacity:
        state.tkey_count += 1
    state.tkey_slot[slot] = slot
    state.tkey_t[slot] = t_val
    return DST_KIND_MIDDLE, slot


def _update_merge(
    state: PerHeadState, t_val: int, head_idx: int = 0
) -> tuple[int, int, int, int, int]:
    """Returns (kind, accum_slot, local_idx, finalize_completed_idx, is_new_block).

    Mirrors pyramidkv/merge.py:65 update() per-frame logic:
      block_id = t_val // block_frames
      local_idx = t_val % block_frames
      find_or_alloc active accumulator slot for block_id
      raise on duplicate local_idx
      mark seen, increment complete_count
      if complete_count == block_frames: finalize (move to completed FIFO)

    State mutated:
      - merge_active_block_id[s] / merge_active_seen[s][i] / merge_active_complete_count[s]
      - merge_completed_slot[i] / merge_completed_block_id[i] / merge_completed_count
        (on finalize)

    Merge accumulator signals:
      - is_new_block = 1 iff we just allocated a fresh accumulator slot (the
        accum_slot was -1 before this frame). The merge_accum kernel uses
        this signal to ZERO the accumulator buffer before its first add and
        to capture the frame's group_ids / pos / tokens_per_group.
      - finalize_completed_idx >= 0 when this frame triggered finalize; the
        merge_accum kernel uses it to divide-and-copy the accumulator into
        merge_*_pool[completed_idx] and clear the buffer.
    """
    if state.block_frames <= 0:
        return DST_KIND_SKIP, -1, -1, -1, 0
    block_id = t_val // state.block_frames
    local_idx = t_val % state.block_frames

    # Find existing active slot for this block_id, or allocate a new one.
    accum_slot = -1
    for s in range(MAX_MERGE_ACTIVE):
        if state.merge_active_block_id[s] == block_id:
            accum_slot = s
            break
    is_new_block = 0
    if accum_slot == -1:
        for s in range(MAX_MERGE_ACTIVE):
            if state.merge_active_block_id[s] == -1:
                accum_slot = s
                break
        if accum_slot == -1:
            # All active slots in use; oldest active gets evicted.
            accum_slot = 0
            state.merge_active_block_id[0] = -1
            state.merge_active_seen[0] = [False] * MAX_MERGE_BLOCK_FRAMES
            state.merge_active_complete_count[0] = 0
        state.merge_active_block_id[accum_slot] = block_id
        state.merge_active_seen[accum_slot] = [False] * MAX_MERGE_BLOCK_FRAMES
        state.merge_active_complete_count[accum_slot] = 0
        is_new_block = 1

    # Duplicate detection — matches Python merge.py:108 ValueError
    if state.merge_active_seen[accum_slot][local_idx]:
        raise MergeDuplicateError(
            f"Duplicate merge frame slot for head={head_idx}, "
            f"block={block_id}, t={t_val}, local_idx={local_idx}"
        )

    state.merge_active_seen[accum_slot][local_idx] = True
    state.merge_active_complete_count[accum_slot] += 1

    finalize_completed_idx = -1
    if state.merge_active_complete_count[accum_slot] >= state.block_frames:
        start_t = block_id * state.block_frames
        end_t = start_t + state.block_frames - 1

        if state.merge_completed_count < state.capacity:
            completed_idx = state.merge_completed_count
            state.merge_completed_count += 1
        else:
            for i in range(1, state.capacity):
                state.merge_completed_slot[i - 1] = state.merge_completed_slot[i]
                state.merge_completed_block_id[i - 1] = state.merge_completed_block_id[i]
            completed_idx = state.capacity - 1
        state.merge_completed_slot[completed_idx] = completed_idx
        state.merge_completed_block_id[completed_idx] = block_id

        state.merge_active_block_id[accum_slot] = -1
        state.merge_active_seen[accum_slot] = [False] * MAX_MERGE_BLOCK_FRAMES
        state.merge_active_complete_count[accum_slot] = 0
        finalize_completed_idx = completed_idx

    return (DST_KIND_MERGE_ACCUM, accum_slot, local_idx,
            finalize_completed_idx, is_new_block)


def mega_state_update_ref(
    states: list[PerHeadState],
    new_t_vals: list[int],
    pass_kind: int,
) -> tuple[
    list[int], list[int], list[int], list[int],
    list[int], list[int], list[int], list[int],
]:
    """Python reference for the mega_state_update kernel.

    Args:
        states: list of PerHeadState, length H. Mutated in place.
        new_t_vals: list of frame t values, length frames_in_block.
        pass_kind: 0=noisy (state untouched, all SKIP), 1=clean.

    Returns 8-tuple, all length H*F head-major:
        desc_dst_kind, desc_dst_slot, desc_src_frame, desc_src_head,
        desc_merge_accum_slot, desc_merge_local_idx,
        desc_merge_finalize_completed_idx, desc_merge_is_new_block

      For non-merge frames the merge_* fields are -1/0. For merge frames:
        - desc_dst_kind  = DST_KIND_MERGE_ACCUM (5)
        - desc_dst_slot  = -1 (no flat-pool destination)
        - desc_merge_accum_slot = which active accumulator (0..MMA-1)
        - desc_merge_local_idx  = which frame within the block (0..block_frames-1)
        - desc_merge_finalize_completed_idx = completed_idx when this frame
          finishes the block (triggers divide+copy in accum kernel); -1 else
        - desc_merge_is_new_block = 1 when this frame allocated a fresh
          active accumulator slot (triggers zero-fill in accum kernel); 0 else

    Raises:
        MergeDuplicateError: matches Python merge.py:108 ValueError.
    """
    H = len(states)
    F = len(new_t_vals)
    N = H * F

    desc_dst_kind = [DST_KIND_SKIP] * N
    desc_dst_slot = [-1] * N
    desc_src_frame = [0] * N
    desc_src_head = [0] * N
    desc_merge_accum_slot = [-1] * N
    desc_merge_local_idx = [-1] * N
    desc_merge_finalize_completed_idx = [-1] * N
    desc_merge_is_new_block = [0] * N

    if pass_kind != 1:
        for h in range(H):
            base = h * F
            for f in range(F):
                desc_src_frame[base + f] = f
                desc_src_head[base + f] = h
        return (
            desc_dst_kind, desc_dst_slot, desc_src_frame, desc_src_head,
            desc_merge_accum_slot, desc_merge_local_idx,
            desc_merge_finalize_completed_idx, desc_merge_is_new_block,
        )

    for h in range(H):
        state = states[h]
        base = h * F
        for f in range(F):
            t_val = new_t_vals[f]
            accum_slot = -1
            local_idx = -1
            finalize_idx = -1
            is_new = 0
            if state.kind == SK_RECENT:
                kind, slot = DST_KIND_SKIP, -1
            elif state.kind == SK_CYCLIC:
                kind, slot = _update_cyclic(state, t_val)
            elif state.kind == SK_STRIDE:
                kind, slot = _update_stride(state, t_val)
            elif state.kind == SK_LAG:
                kind, slot = _update_lag(state, t_val)
            elif state.kind == SK_MERGE:
                (kind, accum_slot, local_idx,
                 finalize_idx, is_new) = _update_merge(state, t_val, head_idx=h)
                slot = -1
            else:
                kind, slot = DST_KIND_SKIP, -1
            desc_dst_kind[base + f] = kind
            desc_dst_slot[base + f] = slot
            desc_src_frame[base + f] = f
            desc_src_head[base + f] = h
            desc_merge_accum_slot[base + f] = accum_slot
            desc_merge_local_idx[base + f] = local_idx
            desc_merge_finalize_completed_idx[base + f] = finalize_idx
            desc_merge_is_new_block[base + f] = is_new

    return (
        desc_dst_kind, desc_dst_slot, desc_src_frame, desc_src_head,
        desc_merge_accum_slot, desc_merge_local_idx,
        desc_merge_finalize_completed_idx, desc_merge_is_new_block,
    )


# ---------------------------------------------------------------------------
# Builders for tests
# ---------------------------------------------------------------------------
def make_cyclic(period: int, bucket_cap: int) -> PerHeadState:
    return PerHeadState(kind=SK_CYCLIC, period=period, bucket_cap=bucket_cap)


def make_stride(interval: int, capacity: int) -> PerHeadState:
    return PerHeadState(kind=SK_STRIDE, interval=interval, capacity=capacity)


def make_lag(history_frames: int, offsets: list[int] | None = None) -> PerHeadState:
    s = PerHeadState(kind=SK_LAG, capacity=history_frames)
    if offsets is not None:
        for i, off in enumerate(offsets[:MAX_LAG_OFFSETS]):
            s.lag_offsets[i] = off
        s.lag_offset_count = min(len(offsets), MAX_LAG_OFFSETS)
    return s


def make_recent() -> PerHeadState:
    return PerHeadState(kind=SK_RECENT)


def make_merge(patch_size: int = 2, capacity: int = 6) -> PerHeadState:
    return PerHeadState(
        kind=SK_MERGE,
        patch_size=patch_size,
        capacity=capacity,
        block_frames=patch_size * patch_size,
    )

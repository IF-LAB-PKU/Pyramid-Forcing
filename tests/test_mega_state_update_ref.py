"""Tests for the Python reference of mega_state_update kernel.

Validates that the reference implementation matches the existing Python
strategy classes (cyclic.py / stride.py / lag.py / recent.py) for the
state-mutation behavior we'll port to CUDA.

These tests are CPU-only — no GPU build needed. Once the CUDA binding
lands, a parallel test file (test_mega_state_update_cuda.py) will run
the same scenarios on the kernel and assert identical descriptor output.
"""
from __future__ import annotations

import pytest

from pyramidkv._mega_state_ref import (
    DST_KIND_MERGE_ACCUM,
    DST_KIND_MIDDLE,
    DST_KIND_SKIP,
    SK_CYCLIC,
    SK_LAG,
    SK_MERGE,
    SK_RECENT,
    SK_STRIDE,
    MAX_MERGE_ACTIVE,
    MAX_MERGE_BLOCK_FRAMES,
    MergeDuplicateError,
    make_cyclic,
    make_lag,
    make_merge,
    make_recent,
    make_stride,
    mega_state_update_ref,
)


class TestMegaStateUpdateBasics:
    def test_recent_emits_skip_for_all_frames(self):
        states = [make_recent()]
        kinds, slots, frames, heads, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[10, 11, 12], pass_kind=1
        )
        assert kinds == [DST_KIND_SKIP] * 3
        assert slots == [-1] * 3
        assert frames == [0, 1, 2]
        assert heads == [0, 0, 0]

    def test_noisy_pass_skips_all_kinds(self):
        states = [make_cyclic(period=6, bucket_cap=3)]
        # Even though strategy is cyclic, noisy pass (pass_kind=0) must skip.
        kinds, slots, _, _, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2], pass_kind=0
        )
        assert kinds == [DST_KIND_SKIP] * 3
        assert slots == [-1] * 3
        # State must remain untouched
        assert states[0].cyclic_cursor == [0] * 6
        assert all(s == -1 for s in states[0].cyclic_slot)


class TestCyclicStrategy:
    def test_phase_assignment(self):
        # period=6, bucket_cap=3 → 18 slots, t=0..5 hit 6 distinct phases
        states = [make_cyclic(period=6, bucket_cap=3)]
        kinds, slots, _, _, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3, 4, 5], pass_kind=1
        )
        assert kinds == [DST_KIND_MIDDLE] * 6
        # Each frame goes to its own phase, cursor=0 → slot = phase * 3
        assert slots == [0, 3, 6, 9, 12, 15]
        # All cursors should advance from 0 to 1
        assert states[0].cyclic_cursor == [1] * 6

    def test_ring_wrap_within_phase(self):
        # Same phase 4 times (period=2, bucket_cap=2)
        # t=0,2,4,6 all hit phase 0; cursor cycles 0→1→0→1
        states = [make_cyclic(period=2, bucket_cap=2)]
        kinds, slots, _, _, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 2, 4, 6], pass_kind=1
        )
        assert kinds == [DST_KIND_MIDDLE] * 4
        # phase=0, cursor=0,1,0,1 → slots 0,1,0,1
        assert slots == [0, 1, 0, 1]
        # Final cursor for phase 0 = 0 (after wrap)
        assert states[0].cyclic_cursor[0] == 0

    def test_t_recorded_at_overwrite(self):
        states = [make_cyclic(period=2, bucket_cap=2)]
        mega_state_update_ref(states, new_t_vals=[0, 2, 4], pass_kind=1)
        # Slot 0 was written at t=0 then overwritten at t=4
        assert states[0].cyclic_t[0] == 4
        assert states[0].cyclic_t[1] == 2


class TestStrideStrategy:
    def test_only_t_divisible_by_interval(self):
        # interval=3, capacity=4 — only t=0,3,6 are stored (skip 1,2,4,5)
        states = [make_stride(interval=3, capacity=4)]
        kinds, slots, _, _, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3, 4, 5, 6], pass_kind=1
        )
        assert kinds == [
            DST_KIND_MIDDLE,  # t=0
            DST_KIND_SKIP,    # t=1
            DST_KIND_SKIP,    # t=2
            DST_KIND_MIDDLE,  # t=3
            DST_KIND_SKIP,    # t=4
            DST_KIND_SKIP,    # t=5
            DST_KIND_MIDDLE,  # t=6
        ]
        # Each kept frame appended at the next free slot
        assert states[0].tkey_count == 3
        assert states[0].tkey_t[:3] == [0, 3, 6]

    def test_fifo_eviction_over_capacity(self):
        # interval=1 (keep all), capacity=3 — 5 frames in, oldest 2 evicted.
        # Ring-buffer FIFO: t=10,11,12 → slots 0,1,2 then head wraps to 0;
        # t=13 lands at slot 0, t=14 at slot 1. So tkey_t = [13, 14, 12].
        states = [make_stride(interval=1, capacity=3)]
        mega_state_update_ref(states, new_t_vals=[10, 11, 12, 13, 14], pass_kind=1)
        assert states[0].tkey_count == 3
        assert sorted(states[0].tkey_t[:3]) == [12, 13, 14]
        # Verify exact ring order: head ends at 2 (next-write slot).
        assert states[0].tkey_head == 2
        assert states[0].tkey_t[0] == 13
        assert states[0].tkey_t[1] == 14
        assert states[0].tkey_t[2] == 12


class TestLagStrategy:
    def test_no_filter_all_frames_stored(self):
        states = [make_lag(history_frames=4)]
        kinds, slots, _, _, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[5, 6, 7], pass_kind=1
        )
        assert kinds == [DST_KIND_MIDDLE] * 3
        assert states[0].tkey_count == 3
        assert states[0].tkey_t[:3] == [5, 6, 7]

    def test_history_fifo_eviction(self):
        states = [make_lag(history_frames=2)]
        mega_state_update_ref(states, new_t_vals=[1, 2, 3, 4], pass_kind=1)
        # capacity=2, only last 2 remain
        assert states[0].tkey_count == 2
        assert states[0].tkey_t[:2] == [3, 4]


class TestMultiHead:
    def test_pyramid_forcing_combo(self):
        # Mirrors the round-robin (osc=cyclic, sta=stride, sparse=recent)
        # head assignment in pyramid-forcing.yaml. 12 heads total.
        states = []
        for h in range(12):
            label_idx = h % 3
            if label_idx == 0:
                # osc: cyclic period=6 cap=3
                states.append(make_cyclic(period=6, bucket_cap=3))
            elif label_idx == 1:
                # sta: stride interval=6 cap=4
                states.append(make_stride(interval=6, capacity=4))
            else:
                # sparse: recent (no middle)
                states.append(make_recent())

        # 3 frames per block, t starting at 6 (after some warmup blocks)
        kinds, slots, frames, heads, _, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[6, 7, 8], pass_kind=1
        )

        # Verify head ordering in descriptor: h0_f0..h0_f2, h1_f0..h1_f2, ...
        assert heads == sum(([h] * 3 for h in range(12)), [])
        assert frames == [0, 1, 2] * 12

        # Cyclic heads (h % 3 == 0): all 3 frames go to middle
        for h in [0, 3, 6, 9]:
            base = h * 3
            assert kinds[base:base + 3] == [DST_KIND_MIDDLE] * 3
            # slots = (t % 6) * 3 + 0 = 18, 21, 24 → wait, max slot = 17
            # phase = 6 % 6 = 0, 7 % 6 = 1, 8 % 6 = 2
            # slot = phase * bucket_cap + cursor = 0,3,6
            assert slots[base:base + 3] == [0, 3, 6]
        # Stride heads (h % 3 == 1): only t=6 (divisible by 6) kept
        for h in [1, 4, 7, 10]:
            base = h * 3
            assert kinds[base:base + 3] == [
                DST_KIND_MIDDLE, DST_KIND_SKIP, DST_KIND_SKIP
            ]
            assert slots[base] == 0  # first stride entry
        # Recent heads (h % 3 == 2): all skip
        for h in [2, 5, 8, 11]:
            base = h * 3
            assert kinds[base:base + 3] == [DST_KIND_SKIP] * 3


class TestStateInvariants:
    def test_cyclic_cursor_stays_in_bucket_cap_range(self):
        # 100 frames, period=3, bucket_cap=2 — cursors should never exceed 1
        states = [make_cyclic(period=3, bucket_cap=2)]
        mega_state_update_ref(
            states, new_t_vals=list(range(100)), pass_kind=1
        )
        for c in states[0].cyclic_cursor:
            assert 0 <= c < 2

    def test_stride_count_capped_at_capacity(self):
        states = [make_stride(interval=1, capacity=4)]
        mega_state_update_ref(
            states, new_t_vals=list(range(20)), pass_kind=1
        )
        assert states[0].tkey_count == 4

    def test_lag_count_capped_at_capacity(self):
        states = [make_lag(history_frames=5)]
        mega_state_update_ref(
            states, new_t_vals=list(range(20)), pass_kind=1
        )
        assert states[0].tkey_count == 5


class TestMergeStrategy:
    """Day 3 — merge accumulator + finalize state machine.

    Mirrors pyramidkv/merge.py:65 update() behavior; the actual K/V
    accumulation kernel lands in Day 4 (mega_merge_accum.cu).
    """

    def test_first_frame_allocates_active_slot(self):
        # patch_size=2 → block_frames=4. t=0 is the first frame of block 0.
        states = [make_merge(patch_size=2, capacity=6)]
        kinds, slots, _, _, accum_slots, local_idxs, _, _ = mega_state_update_ref(
            states, new_t_vals=[0], pass_kind=1
        )
        assert kinds == [DST_KIND_MERGE_ACCUM]
        assert slots == [-1]                 # merge has no flat-pool slot
        assert accum_slots == [0]            # first active slot
        assert local_idxs == [0]             # first frame in block 0
        # State: active slot 0 holds block_id=0 with seen[0]=True
        assert states[0].merge_active_block_id[0] == 0
        assert states[0].merge_active_seen[0][0] is True
        assert states[0].merge_active_complete_count[0] == 1
        assert states[0].merge_completed_count == 0  # not yet finalized

    def test_block_finalizes_after_all_frames(self):
        # Send 4 frames at t=0..3 → block 0 should complete and move to FIFO
        states = [make_merge(patch_size=2, capacity=6)]
        kinds, _, _, _, accum_slots, local_idxs, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3], pass_kind=1
        )
        assert kinds == [DST_KIND_MERGE_ACCUM] * 4
        assert accum_slots == [0, 0, 0, 0]   # all stay in same active slot
        assert local_idxs == [0, 1, 2, 3]
        # After 4th frame, block 0 finalizes:
        # - active slot reset
        # - completed slot 0 holds block_id=0
        assert states[0].merge_active_block_id[0] == -1
        assert states[0].merge_active_complete_count[0] == 0
        assert states[0].merge_completed_count == 1
        assert states[0].merge_completed_block_id[0] == 0

    def test_two_blocks_finalize_in_sequence(self):
        states = [make_merge(patch_size=2, capacity=6)]
        # Block 0: t=0..3, block 1: t=4..7
        mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3, 4, 5, 6, 7], pass_kind=1
        )
        assert states[0].merge_completed_count == 2
        assert states[0].merge_completed_block_id[:2] == [0, 1]
        # No active blocks remain
        assert states[0].merge_active_block_id == [-1, -1]

    def test_capacity_evicts_oldest_completed_fifo(self):
        # capacity=2, send 3 complete blocks → block 0 should be evicted
        states = [make_merge(patch_size=2, capacity=2)]
        # Three blocks: t=0..3 (block 0), t=4..7 (block 1), t=8..11 (block 2)
        mega_state_update_ref(
            states, new_t_vals=list(range(12)), pass_kind=1
        )
        assert states[0].merge_completed_count == 2
        # FIFO: block 0 evicted, block 1 + 2 retained at indices 0, 1
        assert states[0].merge_completed_block_id[:2] == [1, 2]

    def test_duplicate_frame_raises(self):
        # Send t=0 twice — should raise on the second occurrence.
        states = [make_merge(patch_size=2, capacity=6)]
        with pytest.raises(MergeDuplicateError):
            mega_state_update_ref(states, new_t_vals=[0, 0], pass_kind=1)

    def test_out_of_order_frames_within_block(self):
        # t=2,0,3,1 — same block, different arrival order. Should still
        # finalize correctly (block 0 complete = all 4 local_idx seen).
        states = [make_merge(patch_size=2, capacity=6)]
        kinds, _, _, _, accum_slots, local_idxs, _, _ = mega_state_update_ref(
            states, new_t_vals=[2, 0, 3, 1], pass_kind=1
        )
        assert kinds == [DST_KIND_MERGE_ACCUM] * 4
        assert local_idxs == [2, 0, 3, 1]    # local_idx = t % block_frames
        assert accum_slots == [0, 0, 0, 0]
        assert states[0].merge_completed_count == 1
        assert states[0].merge_completed_block_id[0] == 0

    def test_two_active_blocks_simultaneously(self):
        # Frames from block 0 and block 1 interleaved (2 active accumulators).
        # t=0 (b0_f0), t=4 (b1_f0), t=1 (b0_f1), t=5 (b1_f1)
        # — neither completes yet, but each has its own accumulator slot
        states = [make_merge(patch_size=2, capacity=6)]
        kinds, _, _, _, accum_slots, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 4, 1, 5], pass_kind=1
        )
        assert kinds == [DST_KIND_MERGE_ACCUM] * 4
        # Block 0 lands in accum slot 0, block 1 in accum slot 1
        assert accum_slots == [0, 1, 0, 1]
        # Both blocks still accumulating (count=2 each, not yet 4)
        assert states[0].merge_active_block_id[:MAX_MERGE_ACTIVE] == [0, 1]
        assert states[0].merge_active_complete_count[:2] == [2, 2]
        assert states[0].merge_completed_count == 0

    def test_noisy_pass_skips_merge(self):
        states = [make_merge(patch_size=2, capacity=6)]
        kinds, _, _, _, accum_slots, _, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3], pass_kind=0
        )
        assert kinds == [DST_KIND_SKIP] * 4
        assert accum_slots == [-1] * 4
        # State must remain untouched
        assert states[0].merge_active_block_id == [-1, -1]
        assert states[0].merge_completed_count == 0


class TestMergeCombo:
    """Multi-head with merge mixed in (sparse heads in pyramid-forcing)."""

    def test_pyramid_forcing_with_merge(self):
        # 4 osc (cyclic) + 4 sta (stride) + 4 sparse (merge), 12 heads
        states = []
        for h in range(12):
            label = h % 3
            if label == 0:
                states.append(make_cyclic(period=6, bucket_cap=3))
            elif label == 1:
                states.append(make_stride(interval=6, capacity=4))
            else:
                states.append(make_merge(patch_size=2, capacity=6))

        # Send a full block: t=0,1,2,3 (block 0 complete)
        kinds, slots, _, _, accum_slots, local_idxs, _, _ = mega_state_update_ref(
            states, new_t_vals=[0, 1, 2, 3], pass_kind=1
        )

        # Cyclic heads (h % 3 == 0): all 4 frames go to middle (different phases)
        for h in [0, 3, 6, 9]:
            base = h * 4
            assert kinds[base:base + 4] == [DST_KIND_MIDDLE] * 4
        # Stride heads (h % 3 == 1, interval=6): only t=0 kept
        for h in [1, 4, 7, 10]:
            base = h * 4
            assert kinds[base:base + 4] == [
                DST_KIND_MIDDLE, DST_KIND_SKIP, DST_KIND_SKIP, DST_KIND_SKIP
            ]
        # Merge heads (h % 3 == 2): all 4 frames accum, then finalize
        for h in [2, 5, 8, 11]:
            base = h * 4
            assert kinds[base:base + 4] == [DST_KIND_MERGE_ACCUM] * 4
            assert accum_slots[base:base + 4] == [0, 0, 0, 0]
            assert local_idxs[base:base + 4] == [0, 1, 2, 3]
            # After 4 frames, block 0 finalized
            assert states[h].merge_completed_count == 1

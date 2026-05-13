"""End-to-end CUDA kernel ↔ Python reference equivalence for mega_state_update.

Each test case constructs identical input states (one list for ref, a deep
copy for CUDA), runs both the Python reference (``mega_state_update_ref``)
and the CUDA kernel via the wrapper, then asserts:
  1. The four descriptor outputs match exactly (atol=0 — these are int32
     opcodes, no rounding tolerance).
  2. The mutated post-state matches every primitive field.

Day 4 scope: cyclic / stride / lag / recent. Merge cases stay on the Python
ref only; the kernel will return SKIP for SK_MERGE until Day 5.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover — skipped on CPU runners
    pytest.skip("CUDA required for mega_state_update kernel tests",
                allow_module_level=True)

from headkv import _mega_state_ops as ops_mod
from headkv import _mega_state_ref as ref


def _run_both(states, new_t_vals, pass_kind):
    """Run the Python ref and the CUDA kernel on identical inputs.

    Returns ((ref_desc, ref_states), (cuda_desc, cuda_states)) where each
    *_desc is an 8-tuple of descriptor lists matching the kernel signature.
    """
    ref_states = copy.deepcopy(states)
    cuda_states = copy.deepcopy(states)

    ref_out = ref.mega_state_update_ref(
        ref_states, list(new_t_vals), pass_kind
    )
    ref_desc = ref_out  # 8-tuple

    cuda_out = ops_mod.mega_state_update_cuda(
        cuda_states, list(new_t_vals), pass_kind
    )
    # last element is the mutated dataclass list; the first 8 are descriptors.
    cuda_desc = cuda_out[:8]
    mutated = cuda_out[8]

    return (ref_desc, ref_states), (cuda_desc, mutated)


def _assert_descriptors_match(ref_desc, cuda_desc):
    names = ("desc_dst_kind", "desc_dst_slot",
             "desc_src_frame", "desc_src_head",
             "desc_merge_accum_slot", "desc_merge_local_idx",
             "desc_merge_finalize_completed_idx", "desc_merge_is_new_block")
    for name, r, c in zip(names, ref_desc, cuda_desc):
        assert r == c, f"{name} mismatch:\n  ref={r}\n  cuda={c}"


_PRIMITIVE_FIELDS = (
    "kind", "period", "bucket_cap", "interval", "capacity",
    "patch_size", "block_frames", "lag_offset_count",
    "tkey_count", "merge_completed_count", "cached_num_groups",
)


def _assert_states_match(ref_states, cuda_states):
    assert len(ref_states) == len(cuda_states)
    for h, (a, b) in enumerate(zip(ref_states, cuda_states)):
        for fname in _PRIMITIVE_FIELDS:
            av, bv = getattr(a, fname), getattr(b, fname)
            assert av == bv, (
                f"head {h}: field '{fname}' differs (ref={av} cuda={bv})"
            )
        # Variable-length list / array fields.
        for fname in (
            "lag_offsets",
            "cyclic_slot", "cyclic_t", "cyclic_cursor",
            "tkey_slot", "tkey_t",
            "merge_completed_slot", "merge_completed_block_id",
            "merge_active_block_id",
            "merge_active_complete_count",
        ):
            av, bv = getattr(a, fname), getattr(b, fname)
            assert list(av) == list(bv), (
                f"head {h}: list field '{fname}' differs\n"
                f"  ref={av}\n  cuda={bv}"
            )


# ---------------------------------------------------------------------------
# Sanity: ABI probe
# ---------------------------------------------------------------------------
def test_abi_size_matches_cpp():
    """numpy structured dtype itemsize must equal sizeof(PerHeadState)."""
    from headkv import _ops  # triggers JIT load
    cpp_size = int(_ops.ops().mega_state_perhead_size())
    py_size = ops_mod.PER_HEAD_STATE_DTYPE.itemsize
    assert cpp_size == py_size, (
        f"struct ABI drift: cpp={cpp_size} bytes, numpy={py_size} bytes"
    )


# ---------------------------------------------------------------------------
# Cyclic
# ---------------------------------------------------------------------------
def test_cyclic_basic_phase_advance():
    states = [ref.make_cyclic(period=6, bucket_cap=3)]
    new_t_vals = [12, 13, 14]  # phases 0, 1, 2
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


def test_cyclic_ring_wrap_within_phase():
    """Same phase hit 4× exercises the bucket_cap=3 ring wrap."""
    states = [ref.make_cyclic(period=6, bucket_cap=3)]
    new_t_vals = [0, 6, 12, 18]  # phase 0 four times
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------
def test_stride_skips_when_t_mod_interval_nonzero():
    states = [ref.make_stride(interval=6, capacity=4)]
    new_t_vals = [3, 6, 9, 12]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


def test_stride_fifo_eviction_at_capacity():
    states = [ref.make_stride(interval=6, capacity=4)]
    # Six valid t's (0,6,12,18,24,30) — last 2 evict the first 2.
    new_t_vals = [0, 6, 12, 18, 24, 30]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


# ---------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------
def test_lag_appends_every_frame_until_capacity():
    states = [ref.make_lag(history_frames=4)]
    new_t_vals = [10, 11, 12, 13]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


def test_lag_fifo_eviction_when_history_exceeded():
    states = [ref.make_lag(history_frames=3)]
    new_t_vals = [10, 11, 12, 13, 14]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


# ---------------------------------------------------------------------------
# Recent
# ---------------------------------------------------------------------------
def test_recent_emits_skip_and_does_not_mutate():
    states = [ref.make_recent()]
    new_t_vals = [5, 6, 7]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


# ---------------------------------------------------------------------------
# Mixed multi-head + pass_kind=0 (noisy) negative path
# ---------------------------------------------------------------------------
def test_pyramid_forcing_g6_layer_mix_clean_pass():
    """One layer's worth: 4 cyclic heads + 4 stride + 2 lag + 2 recent."""
    states = (
        [ref.make_cyclic(period=6, bucket_cap=3) for _ in range(4)]
        + [ref.make_stride(interval=6, capacity=4) for _ in range(4)]
        + [ref.make_lag(history_frames=4) for _ in range(2)]
        + [ref.make_recent() for _ in range(2)]
    )
    new_t_vals = [12, 13, 14]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)


def test_noisy_pass_emits_all_skip_and_state_unchanged():
    states = [
        ref.make_cyclic(period=6, bucket_cap=3),
        ref.make_stride(interval=6, capacity=4),
        ref.make_lag(history_frames=4),
        ref.make_recent(),
    ]
    new_t_vals = [12, 13, 14]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=0)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    # Extra: every kind must be SKIP for noisy passes.
    assert all(k == ref.DST_KIND_SKIP for k in cd[0])


# ---------------------------------------------------------------------------
# Merge — Day 5a: kernel now mirrors Python ref _update_merge
# ---------------------------------------------------------------------------
def test_merge_first_frame_emits_accum_descriptor():
    """Single frame in a brand-new block: kind=MERGE_ACCUM, accum_slot=0,
    local_idx=0; no completed block yet. New-block signal is 1."""
    states = [ref.make_merge(patch_size=2, capacity=6)]
    new_t_vals = [0]  # block 0, local 0
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    assert cd[0] == [ref.DST_KIND_MERGE_ACCUM]
    assert cd[4] == [0]   # accum_slot
    assert cd[5] == [0]   # local_idx
    assert cd[6] == [-1]  # finalize: no block done yet
    assert cd[7] == [1]   # is_new_block: just allocated


def test_merge_block_finalizes_after_all_frames():
    """patch_size=2 → block_frames=4. Feed 4 frames (block 0). Block must
    finalize: merge_completed_count==1, active slot reset to -1.
    new_block=1 on first frame only; finalize_completed_idx=0 on last."""
    states = [ref.make_merge(patch_size=2, capacity=6)]
    new_t_vals = [0, 1, 2, 3]  # block 0, local 0..3
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    assert all(k == ref.DST_KIND_MERGE_ACCUM for k in cd[0])
    assert cs[0].merge_completed_count == 1
    assert cs[0].merge_active_block_id[0] == -1
    # new_block signal: only on the first frame of the block
    assert cd[7] == [1, 0, 0, 0]
    # finalize signal: only on the 4th frame, with completed_idx = 0
    assert cd[6] == [-1, -1, -1, 0]


def test_merge_two_blocks_finalize_in_sequence():
    states = [ref.make_merge(patch_size=2, capacity=6)]
    new_t_vals = list(range(8))  # block 0 (t=0..3) + block 1 (t=4..7)
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    assert cs[0].merge_completed_count == 2
    # new_block fires on frame 0 (block 0 alloc) and frame 4 (block 1 alloc)
    assert cd[7] == [1, 0, 0, 0, 1, 0, 0, 0]
    # finalize fires on frame 3 (block 0 → idx 0) and frame 7 (block 1 → idx 1)
    assert cd[6] == [-1, -1, -1, 0, -1, -1, -1, 1]


def test_merge_capacity_evicts_oldest_completed_fifo():
    """capacity=2 with 3 completed blocks → first one evicted."""
    states = [ref.make_merge(patch_size=2, capacity=2)]
    new_t_vals = list(range(12))  # 3 blocks of 4 frames
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    assert cs[0].merge_completed_count == 2
    # Block ids: [1, 2] after evicting block 0.
    assert cs[0].merge_completed_block_id[0] == 1
    assert cs[0].merge_completed_block_id[1] == 2


def test_merge_two_active_blocks_simultaneously():
    """Two distinct block_ids active at once → both occupy active slots."""
    states = [ref.make_merge(patch_size=2, capacity=6)]
    new_t_vals = [0, 4, 1, 5]  # interleaved frames from blocks 0 and 1
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    # Both active slots should be occupied (one for block 0, one for block 1).
    active_blocks = sorted(cs[0].merge_active_block_id)
    assert active_blocks == [0, 1]


def test_merge_noisy_pass_skips_all_with_minus_one_descriptors():
    states = [ref.make_merge(patch_size=2, capacity=6)]
    new_t_vals = [0, 1, 2, 3]
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=0)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)
    assert all(k == ref.DST_KIND_SKIP for k in cd[0])
    assert all(a == -1 for a in cd[4])
    assert all(li == -1 for li in cd[5])
    assert all(fi == -1 for fi in cd[6])
    assert all(nb == 0 for nb in cd[7])


def test_pyramid_forcing_g6_layer_with_merge_head():
    """Realistic mix: 4 cyclic + 4 stride + 2 merge + 2 recent."""
    states = (
        [ref.make_cyclic(period=6, bucket_cap=3) for _ in range(4)]
        + [ref.make_stride(interval=6, capacity=4) for _ in range(4)]
        + [ref.make_merge(patch_size=2, capacity=6) for _ in range(2)]
        + [ref.make_recent() for _ in range(2)]
    )
    new_t_vals = [12, 13, 14]  # block 3 (12//4=3, 13//4=3, 14//4=3); locals 0,1,2
    (rd, rs), (cd, cs) = _run_both(states, new_t_vals, pass_kind=1)
    _assert_descriptors_match(rd, cd)
    _assert_states_match(rs, cs)

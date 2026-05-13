"""M1 Day 5b — verify PyramidKVCacheManager allocates merge accumulator pools.

The Day 5b commit only adds buffers + accessors (no kernel work). These
tests pin down:
  1. Shapes match the design contract from anchor_store.cuh:
     merge_*_pool sized [L, H, MMB, MTPA, *], accumulators by [L, H, MMA, ...]
  2. Dtypes are correct: bf16 for K/V data pools, fp32 for accumulators
     (intentional — bf16 sum-of-N drift makes the divide-finalize unstable),
     int32 for token counts and group_ids, int64 for positions.
  3. reset() clears the gating tensors (merge_token_count,
     merge_accum_num_groups) so the next prompt sees empty merge state.
"""
from __future__ import annotations

import pytest
import torch


def _try_load_or_skip():
    try:
        from pyramidkv import _ops
    except Exception as exc:
        pytest.skip(f"pyramidkv._ops import failed: {exc}")
        return None
    if not _ops._ensure_loaded():
        pytest.skip("Extension failed to load")
    return _ops


def _make_manager():
    _try_load_or_skip()
    Cls = torch.classes.adahead.PyramidKVCacheManager
    return Cls(
        2,    # num_layers
        4,    # num_heads
        16,   # head_dim
        8,    # frame_seqlen (intentionally small; merge_pos uses real value)
        2,    # max_sink
        4,    # max_middle
        2,    # max_recent
        "cuda:0",
        "bfloat16",
        2,    # max_attend_chunks (Plan B; small for test)
    )


@pytest.mark.gpu
class TestMergePools:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_compile_time_bounds_are_exposed(self):
        mgr = _make_manager()
        # Bounds match anchor_store.cuh constants.
        assert int(mgr.max_merge_blocks()) == 6
        assert int(mgr.max_merge_active()) == 2
        assert int(mgr.max_merge_tokens_per_anchor()) == 420

    def test_merge_kv_pool_shapes_and_dtype(self):
        mgr = _make_manager()
        L, H, D = 2, 4, 16
        MMB = int(mgr.max_merge_blocks())
        MTPA = int(mgr.max_merge_tokens_per_anchor())

        for getter, name in (
            (mgr.merge_k_pool, "merge_k_pool"),
            (mgr.merge_v_pool, "merge_v_pool"),
        ):
            t = getter()
            assert tuple(t.shape) == (L, H, MMB, MTPA, D), name
            assert t.dtype == torch.bfloat16, name
            assert t.is_cuda, name

        pos = mgr.merge_pos_pool()
        assert tuple(pos.shape) == (L, H, MMB, MTPA, 3)
        assert pos.dtype == torch.int64

        cnt = mgr.merge_token_count()
        assert tuple(cnt.shape) == (L, H, MMB)
        assert cnt.dtype == torch.int32

    def test_merge_accumulator_shapes_and_dtype(self):
        mgr = _make_manager()
        L, H, D, F = 2, 4, 16, 8
        MMA = int(mgr.max_merge_active())
        MTPA = int(mgr.max_merge_tokens_per_anchor())

        # K/V accumulators are fp32 (numerical stability under sum-of-N).
        for getter, name in (
            (mgr.merge_accum_sum_k, "merge_accum_sum_k"),
            (mgr.merge_accum_sum_v, "merge_accum_sum_v"),
        ):
            t = getter()
            assert tuple(t.shape) == (L, H, MMA, MTPA, D), name
            assert t.dtype == torch.float32, name

        # Positions captured at first frame of a block — full accumulator size.
        pos = mgr.merge_accum_pos()
        assert tuple(pos.shape) == (L, H, MMA, MTPA, 3)
        assert pos.dtype == torch.int64

        # group_ids is per-token (frame_seqlen wide, NOT MTPA wide).
        gids = mgr.merge_accum_group_ids()
        assert tuple(gids.shape) == (L, H, MMA, F)
        assert gids.dtype == torch.int32

        tpg = mgr.merge_accum_tokens_per_group()
        assert tuple(tpg.shape) == (L, H, MMA, MTPA)
        assert tpg.dtype == torch.float32

        ngroups = mgr.merge_accum_num_groups()
        assert tuple(ngroups.shape) == (L, H, MMA)
        assert ngroups.dtype == torch.int32

    def test_reset_clears_merge_gating_tensors(self):
        """reset() must wipe merge_token_count + merge_accum_num_groups —
        these gate 'is anchor populated'. Data pools may stay dirty."""
        mgr = _make_manager()

        # Plant non-zero state.
        cnt = mgr.merge_token_count()
        cnt[0, 0, 0] = 123
        cnt[1, 2, 3] = 45

        ngroups = mgr.merge_accum_num_groups()
        ngroups[0, 1, 0] = 78
        ngroups[1, 3, 1] = 9

        mgr.reset()

        assert int(mgr.merge_token_count().sum()) == 0
        assert int(mgr.merge_accum_num_groups().sum()) == 0

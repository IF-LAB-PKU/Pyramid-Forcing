"""M1.3 test: HeadKVCacheManager allocation + reset + set_strategy.

Verifies:
1. Constructor allocates all pools / state tensors with correct shapes/dtypes.
2. reset() clears ring-buffer state but preserves pool buffers.
3. set_strategy() writes the kind+params slot for one (layer, head) without
   touching others.
4. The custom class is reachable via torch.classes.adahead.HeadKVCacheManager
   (which is what M2 graph capture will rely on).
"""
from __future__ import annotations

import pytest
import torch


def _try_load_or_skip():
    try:
        from headkv import _ops
    except Exception as exc:
        pytest.skip(f"headkv._ops import failed: {exc}")
        return None
    if not _ops._ensure_loaded():
        pytest.skip("Extension failed to load")
    return _ops


def _make_manager(device="cuda:0", dtype="bfloat16"):
    _try_load_or_skip()
    Cls = torch.classes.adahead.HeadKVCacheManager
    return Cls(
        2,    # num_layers (small for test)
        4,    # num_heads
        16,   # head_dim
        8,    # frame_seqlen
        2,    # max_sink_frames
        4,    # max_middle_frames
        2,    # max_recent_frames
        device,
        dtype,
        2,    # max_attend_chunks (Plan B; small for test)
    )


@pytest.mark.gpu
class TestHeadKVCacheManager:
    def test_constructor_allocates_expected_shapes(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr = _make_manager()

        L, H, D, F = 2, 4, 16, 8
        sink, mid, rec = 2, 4, 2
        total = sink + mid + rec
        # Plan B — pack workspace sized for one attend call: chunks × H ×
        # (max_total + max_merge_blocks) × F. _make_manager uses chunks=2.
        max_attend_chunks = 2
        max_merge_blocks = mgr.max_merge_blocks()
        pack_tokens = max_attend_chunks * H * (total + max_merge_blocks) * F

        # Pool shapes
        assert tuple(mgr.sink_pool().shape) == (L, H, sink, F, D)
        assert tuple(mgr.middle_pool().shape) == (L, H, mid, F, D)
        assert tuple(mgr.recent_pool().shape) == (L, H, rec, F, D)

        # Ring state shapes
        assert tuple(mgr.head_t_table().shape) == (L, H, total)
        assert tuple(mgr.write_idx().shape) == (L, H, 3)
        assert tuple(mgr.valid_count().shape) == (L, H, 3)
        assert tuple(mgr.strategy().shape) == (L, H, 8)

        # Output workspace
        assert tuple(mgr.k_flat_out().shape) == (pack_tokens, 1, D)
        assert tuple(mgr.v_flat_out().shape) == (pack_tokens, 1, D)
        assert tuple(mgr.cu_seqlens_k().shape) == (max_attend_chunks * H + 1,)
        assert tuple(mgr.pos_flat_out().shape) == (pack_tokens, 3)

        # Dtypes
        assert mgr.sink_pool().dtype == torch.bfloat16
        assert mgr.head_t_table().dtype == torch.int64
        assert mgr.cu_seqlens_k().dtype == torch.int32
        assert mgr.strategy().dtype == torch.int8

        # Devices
        assert mgr.sink_pool().is_cuda
        assert mgr.cu_seqlens_k().is_cuda

    def test_initial_ring_state(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr = _make_manager()
        # head_t_table should start as -1 (empty slot sentinel)
        assert (mgr.head_t_table() == -1).all().item()
        # ring counters at zero
        assert (mgr.write_idx() == 0).all().item()
        assert (mgr.valid_count() == 0).all().item()

    def test_set_strategy_writes_slot(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr = _make_manager()

        params = torch.tensor([6, 3, 4], dtype=torch.int8)  # period, cap, recent
        # TORCH_LIBRARY custom class methods take positional args only.
        mgr.set_strategy(0, 2, 1, params)

        s = mgr.strategy().cpu()
        # Slot for (0, 2)
        assert int(s[0, 2, 0]) == 1   # kind
        assert int(s[0, 2, 1]) == 6
        assert int(s[0, 2, 2]) == 3
        assert int(s[0, 2, 3]) == 4
        # Other heads untouched
        assert int(s[0, 0, 0]) == 0
        assert int(s[1, 2, 0]) == 0

    def test_reset_clears_ring_state_but_keeps_strategy(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr = _make_manager()

        params = torch.tensor([6], dtype=torch.int8)
        mgr.set_strategy(0, 0, 2, params)

        # Manually mark some ring state as if a forward had run.
        # (Direct in-place edits via the returned views.)
        wi = mgr.write_idx()
        vc = mgr.valid_count()
        ht = mgr.head_t_table()
        wi.fill_(7)
        vc.fill_(3)
        ht.fill_(99)
        cu = mgr.cu_seqlens_k()
        cu.fill_(11)

        mgr.reset()

        assert (mgr.write_idx() == 0).all().item()
        assert (mgr.valid_count() == 0).all().item()
        assert (mgr.head_t_table() == -1).all().item()
        assert (mgr.cu_seqlens_k() == 0).all().item()
        # Strategy slot survives reset.
        assert int(mgr.strategy().cpu()[0, 0, 0]) == 2

    def test_set_strategy_validates_indices(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr = _make_manager()
        params = torch.tensor([1], dtype=torch.int8)
        with pytest.raises(Exception):
            mgr.set_strategy(99, 0, 0, params)
        with pytest.raises(Exception):
            mgr.set_strategy(0, 99, 0, params)

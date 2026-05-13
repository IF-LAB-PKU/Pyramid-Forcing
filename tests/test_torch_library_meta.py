"""M1.2 test: TORCH_LIBRARY registration + meta kernels for adahead ops.

Verifies:
1. The JIT extension loads and registers the ``adahead`` namespace.
2. ``torch.ops.adahead.scatter_copy`` etc. are callable on CUDA tensors and
   match the legacy ``_scatter_ext`` PYBIND11 path bit-for-bit.
3. Meta kernels exist and don't crash when the op is dispatched on
   ``meta`` device tensors — required for torch.compile / export / vmap and
   for CUDA Graph capture lookahead.

Marked as gpu since most checks need real CUDA (we still also check meta
dispatch works for shape-only flows).
"""
from __future__ import annotations

import pytest
import torch


def _try_load_or_skip():
    try:
        from pyramidkv import _ops
    except Exception as exc:  # pragma: no cover - import error
        pytest.skip(f"pyramidkv._ops import failed: {exc}")
        return None
    if not _ops.available():
        pytest.skip("CUDA scatter extension not available")
    return _ops


class TestTorchLibraryRegistration:
    def test_adahead_namespace_registered(self):
        ops = _try_load_or_skip()
        ns = ops.ops()
        assert hasattr(ns, "scatter_copy")
        assert hasattr(ns, "apply_pos_override")
        assert hasattr(ns, "anchor_store_write_frames")
        assert hasattr(ns, "refresh_readout_layout")

    def test_scatter_copy_torch_ops_matches_pybind(self):
        """torch.ops.adahead.scatter_copy must produce the same output as
        the legacy _scatter_ext.scatter_copy on identical inputs.
        """
        ops = _try_load_or_skip()
        from pyramidkv import _scatter_ext
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")

        # Build 3 source tensors on CUDA, descriptor tensors, and dst.
        head_dim = 16
        seg_lengths = [4, 7, 5]
        src_tensors = [
            torch.randn(L, head_dim, device=device, dtype=torch.float32).contiguous()
            for L in seg_lengths
        ]
        total = sum(seg_lengths)
        offsets = torch.tensor([0, 4, 11], dtype=torch.int64, device=device)
        lengths = torch.tensor(seg_lengths, dtype=torch.int64, device=device)
        src_ptrs = torch.tensor(
            [t.data_ptr() for t in src_tensors], dtype=torch.int64, device=device
        )

        # Path 1: torch.ops
        dst_a = torch.zeros(total, head_dim, device=device, dtype=torch.float32)
        ops.scatter_copy(src_ptrs, lengths, offsets, dst_a, head_dim)
        torch.cuda.synchronize()

        # Path 2: legacy _scatter_ext (PYBIND11)
        dst_b = torch.zeros(total, head_dim, device=device, dtype=torch.float32)
        _scatter_ext.scatter_copy(src_ptrs, lengths, offsets, dst_b, head_dim)
        torch.cuda.synchronize()

        torch.testing.assert_close(dst_a, dst_b, atol=0, rtol=0)

        # Reference: manual cat
        dst_ref = torch.cat(src_tensors, dim=0)
        torch.testing.assert_close(dst_a, dst_ref, atol=0, rtol=0)

    def test_apply_pos_override_torch_ops(self):
        ops = _try_load_or_skip()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")

        # 12 rows, write t=99 to rows [2, 5) and t=42 to rows [7, 10)
        pos_a = torch.zeros(12, 3, dtype=torch.int64, device=device)
        pos_a[:, 0] = torch.arange(12, device=device)
        pos_b = pos_a.clone()

        starts = torch.tensor([2, 7], dtype=torch.int64, device=device)
        ends = torch.tensor([5, 10], dtype=torch.int64, device=device)
        vals = torch.tensor([99, 42], dtype=torch.int64, device=device)

        ops.apply_pos_override(pos_a, starts, ends, vals)

        from pyramidkv import _scatter_ext
        _scatter_ext.apply_pos_override(pos_b, starts, ends, vals)
        torch.cuda.synchronize()

        torch.testing.assert_close(pos_a, pos_b, atol=0, rtol=0)
        # Sanity: rows 2..5 and 7..10 got overridden
        assert int(pos_a[2, 0]) == 99
        assert int(pos_a[4, 0]) == 99
        assert int(pos_a[7, 0]) == 42
        assert int(pos_a[5, 0]) == 5  # untouched

    def test_meta_dispatch_does_not_crash(self):
        """Calling the op on meta-device tensors should not crash even
        though the kernel does no work — this is the path that
        torch.compile / export rely on. We can run this without a GPU.
        """
        try:
            from pyramidkv import _ops
        except Exception as exc:
            pytest.skip(f"pyramidkv._ops import failed: {exc}")
        if not _ops._ensure_loaded():
            pytest.skip("Extension failed to load")
        ns = torch.ops.adahead

        meta = torch.device("meta")
        src_ptrs = torch.empty(3, dtype=torch.int64, device=meta)
        lengths = torch.empty(3, dtype=torch.int64, device=meta)
        offsets = torch.empty(3, dtype=torch.int64, device=meta)
        dst = torch.empty(16, 8, dtype=torch.float32, device=meta)
        # Should not raise.
        ns.scatter_copy(src_ptrs, lengths, offsets, dst, 8)

        pos = torch.empty(16, 3, dtype=torch.int64, device=meta)
        starts = torch.empty(2, dtype=torch.int64, device=meta)
        ends = torch.empty(2, dtype=torch.int64, device=meta)
        vals = torch.empty(2, dtype=torch.int64, device=meta)
        ns.apply_pos_override(pos, starts, ends, vals)

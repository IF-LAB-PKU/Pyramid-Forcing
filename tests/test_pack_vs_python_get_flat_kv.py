"""M1.6 part 2 integration test: C++ plan+pack matches Python get_flat_kv.

Proves the C++ readout path (torch.ops.adahead.headkv_plan +
headkv_pack) produces bit-equivalent output to the existing Python
HeadKVCache.get_flat_kv() given the same logical cache state. This
bridges the unit tests and a full pipeline integration: it exercises
the kernel on production-shape data (12 heads, frame_seqlen=1560)
without yet touching adaptive_cache.py.

The test populates dynamic_k[i] / static_k[i] with synthetic frames,
mirrors that state into a HeadKVCacheManager's pools, then compares
the V outputs (pre-RoPE on K is already applied in the cache, but V
goes through unchanged in both paths so it's the safe comparison).
"""
from __future__ import annotations

import pytest
import torch
from headkv import _ops as _ops_mod


def _try_load_or_skip():
    try:
        from headkv import _ops
    except Exception as exc:
        pytest.skip(f"headkv._ops import failed: {exc}")
        return None
    if not _ops._ensure_loaded():
        pytest.skip("Extension failed to load")
    return _ops


@pytest.mark.gpu
class TestPackVsPythonGetFlatKV:
    def test_recent_only_matches(self):
        """Each head has dynamic_k[i] = N frames (no static); compare Python
        torch.cat path against C++ plan+pack on the same logical state."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()

        # Production-aligned dims (1 layer cache; cross-layer global manager
        # in production has L=30 — the kernel is layer-agnostic).
        H, D, F = 12, 128, 1560  # 12 heads, head_dim=128, frame_seqlen=1560
        max_recent = 4
        L = 1
        device = "cuda:0"

        # Build C++ manager.
        Cls = torch.classes.adahead.HeadKVCacheManager
        mgr = Cls(L, H, D, F, 1, 1, max_recent, device, "bfloat16", L)

        # Synthetic per-head dynamic state: each head has `n_frames_per_head`
        # frames; we pick the same N for all heads to keep the test simple.
        n_frames = max_recent  # fully populated
        torch.manual_seed(11)
        # Per head: contiguous [n_frames * F, D]; mirrors AdaptiveKVCache.dynamic_k[i] shape.
        dynamic_v_list = []  # python ref state
        for h in range(H):
            v = torch.randn(n_frames * F, D, dtype=torch.bfloat16, device=device)
            dynamic_v_list.append(v)
            # Mirror into recent_v_pool[0, h, :n_frames] which is shape [n_frames, F, D]
            mgr.recent_v_pool()[0, h, :n_frames].copy_(v.reshape(n_frames, F, D))

        vc = mgr.valid_count()
        vc[0, :, 0] = 0           # sink empty
        vc[0, :, 1] = 0           # middle empty
        vc[0, :, 2] = n_frames    # recent

        # Python reference: torch.cat over heads (head-major order, like
        # HeadKVCache.get_flat_kv does)
        ref_v = torch.cat(dynamic_v_list, dim=0)

        # C++: plan + pack
        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.headkv_plan(mgr, 0, 0)
        _ops_mod.headkv_pack(mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()
        total = int(cu[-1].item())
        out_v = mgr.v_flat_out()[:total, 0, :]

        torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)

        # Verify cu_seqlens_k matches the Python lengths.
        cu_cpu = cu.cpu()
        for h in range(H):
            assert int(cu_cpu[h + 1] - cu_cpu[h]) == n_frames * F

    def test_sink_plus_recent_matches(self):
        """Each head has both static_k (sink) AND dynamic_k (recent)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()

        # Smaller dims for speed. Keeping H=12 to mirror production head count.
        H, D, F = 12, 32, 64
        max_sink = 2
        max_recent = 3
        L = 1
        device = "cuda:0"

        Cls = torch.classes.adahead.HeadKVCacheManager
        mgr = Cls(L, H, D, F, max_sink, 1, max_recent, device, "bfloat16", L)

        torch.manual_seed(42)
        ref_chunks = []  # head-major: static_v[h] + dynamic_v[h] for each h
        for h in range(H):
            sn_v = torch.randn(max_sink * F, D, dtype=torch.bfloat16, device=device)
            dn_v = torch.randn(max_recent * F, D, dtype=torch.bfloat16, device=device)
            mgr.sink_v_pool()[0, h, :max_sink].copy_(sn_v.reshape(max_sink, F, D))
            mgr.recent_v_pool()[0, h, :max_recent].copy_(dn_v.reshape(max_recent, F, D))
            # Python ref: static then dynamic (mirrors HeadKVCache.get_flat_kv)
            ref_chunks.append(sn_v)
            ref_chunks.append(dn_v)

        vc = mgr.valid_count()
        vc[0, :, 0] = max_sink
        vc[0, :, 1] = 0
        vc[0, :, 2] = max_recent

        ref_v = torch.cat(ref_chunks, dim=0)

        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.headkv_plan(mgr, 0, 0)
        _ops_mod.headkv_pack(mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()
        total = int(cu[-1].item())
        out_v = mgr.v_flat_out()[:total, 0, :]

        torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)

        # Per-head total = (max_sink + max_recent) * F
        cu_cpu = cu.cpu()
        for h in range(H):
            assert int(cu_cpu[h + 1] - cu_cpu[h]) == (max_sink + max_recent) * F

    def test_partial_sink_recent_matches(self):
        """Some heads have only recent, others have sink+recent (mixed)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()

        H, D, F = 4, 16, 32
        L = 1
        device = "cuda:0"

        Cls = torch.classes.adahead.HeadKVCacheManager
        mgr = Cls(L, H, D, F, 2, 1, 2, device, "bfloat16", L)

        # Per-head varying valid counts.
        sink_counts = [2, 0, 1, 0]
        recent_counts = [1, 2, 2, 2]

        torch.manual_seed(0)
        ref_chunks = []
        for h in range(H):
            if sink_counts[h] > 0:
                sn = torch.randn(sink_counts[h] * F, D, dtype=torch.bfloat16, device=device)
                mgr.sink_v_pool()[0, h, :sink_counts[h]].copy_(sn.reshape(sink_counts[h], F, D))
                ref_chunks.append(sn)
            if recent_counts[h] > 0:
                rc = torch.randn(recent_counts[h] * F, D, dtype=torch.bfloat16, device=device)
                mgr.recent_v_pool()[0, h, :recent_counts[h]].copy_(rc.reshape(recent_counts[h], F, D))
                ref_chunks.append(rc)

        vc = mgr.valid_count()
        for h in range(H):
            vc[0, h, 0] = sink_counts[h]
            vc[0, h, 1] = 0
            vc[0, h, 2] = recent_counts[h]

        ref_v = torch.cat(ref_chunks, dim=0)

        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.headkv_plan(mgr, 0, 0)
        _ops_mod.headkv_pack(mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()
        total = int(cu[-1].item())
        out_v = mgr.v_flat_out()[:total, 0, :]

        torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)

        cu_cpu = cu.cpu()
        for h in range(H):
            expected = (sink_counts[h] + recent_counts[h]) * F
            assert int(cu_cpu[h + 1] - cu_cpu[h]) == expected

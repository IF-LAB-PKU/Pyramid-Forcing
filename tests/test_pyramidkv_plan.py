"""M1.4 part 2 test: pyramidkv_plan + pyramidkv_pack end-to-end.

Manually populate manager pools + valid_count, call pyramidkv_plan to build
descriptor tensors, run pyramidkv_pack, verify the flat output equals the
expected concatenation.

This pins down the plan stage's contract: each (layer, head) emits up to
3 segments (sink, middle, recent), inactive kinds get src_kind=-1, and
the dst layout is sink→middle→recent within each head, head-major.
"""
from __future__ import annotations

import pytest
import torch
from pyramidkv import _ops as _ops_mod


def _try_load_or_skip():
    try:
        from pyramidkv import _ops
    except Exception as exc:
        pytest.skip(f"pyramidkv._ops import failed: {exc}")
        return None
    if not _ops._ensure_loaded():
        pytest.skip("Extension failed to load")
    return _ops


@pytest.mark.gpu
class TestPyramidKVPlan:
    def test_plan_then_pack_roundtrip(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        Cls = torch.classes.adahead.PyramidKVCacheManager
        L, H, D, F, ms, mm, mr = 2, 4, 16, 8, 2, 4, 2
        # Plan B — old pyramidkv_plan packs all L layers in one call.
        mgr = Cls(L, H, D, F, ms, mm, mr, "cuda:0", "bfloat16", L)

        # Populate pools with deterministic data.
        torch.manual_seed(42)
        mgr.sink_k_pool().copy_(torch.randn_like(mgr.sink_k_pool()))
        mgr.sink_v_pool().copy_(torch.randn_like(mgr.sink_v_pool()))
        mgr.middle_k_pool().copy_(torch.randn_like(mgr.middle_k_pool()))
        mgr.middle_v_pool().copy_(torch.randn_like(mgr.middle_v_pool()))
        mgr.recent_k_pool().copy_(torch.randn_like(mgr.recent_k_pool()))
        mgr.recent_v_pool().copy_(torch.randn_like(mgr.recent_v_pool()))

        # Mark all slots valid (max ms/mm/mr per kind per head).
        vc = mgr.valid_count()
        for kind, n in enumerate([ms, mm, mr]):
            vc[..., kind] = n

        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.pyramidkv_plan(mgr, 0, 0)
        torch.cuda.synchronize()

        # Sanity on plan output shapes.
        N = L * H
        assert tuple(cu.shape) == (N + 1,)
        assert tuple(sk.shape) == (N * 3,)
        assert tuple(sg.shape) == (N * 3,)
        assert tuple(sl.shape) == (N * 3,)
        assert tuple(dst.shape) == (N * 3,)
        assert cu.dtype == torch.int32
        assert sk.dtype == torch.int32

        # Per (layer, head): seg lengths sum to (ms + mm + mr) * F.
        cu_cpu = cu.cpu()
        per_head_total = (ms + mm + mr) * F
        for i in range(N):
            head_len = int(cu_cpu[i + 1] - cu_cpu[i])
            assert head_len == per_head_total, f"head {i}: got {head_len}, want {per_head_total}"

        # Run pack.
        _ops_mod.pyramidkv_pack(mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()

        # Build the reference concatenation. Plan iterates kinds in Python
        # pack order: sink → recent → middle (matches AdaptiveKVCache pack).
        ref_chunks = []
        for l in range(L):
            for h in range(H):
                ref_chunks.append(mgr.sink_k_pool()[l, h, :ms].reshape(ms * F, D))
                ref_chunks.append(mgr.recent_k_pool()[l, h, :mr].reshape(mr * F, D))
                ref_chunks.append(mgr.middle_k_pool()[l, h, :mm].reshape(mm * F, D))
        ref_k = torch.cat(ref_chunks, dim=0)

        total_tokens = N * per_head_total
        out_k = mgr.k_flat_out()[:total_tokens, 0, :]
        torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)

    def test_plan_inactive_kinds(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        Cls = torch.classes.adahead.PyramidKVCacheManager
        L, H, D, F, ms, mm, mr = 1, 2, 8, 4, 2, 2, 2
        mgr = Cls(L, H, D, F, ms, mm, mr, "cuda:0", "bfloat16", L)

        # Only middle has data; sink/recent empty.
        mgr.middle_k_pool().copy_(torch.randn_like(mgr.middle_k_pool()))
        mgr.middle_v_pool().copy_(torch.randn_like(mgr.middle_v_pool()))
        vc = mgr.valid_count()
        vc[..., 0] = 0          # sink empty
        vc[..., 1] = mm         # middle full
        vc[..., 2] = 0          # recent empty

        ns = torch.ops.adahead
        cu, sk, sg, sl, dst = ns.pyramidkv_plan(mgr, 0, 0)
        torch.cuda.synchronize()

        sk_cpu = sk.cpu()
        # Plan iterates kinds in Python pack order: sink, recent, middle.
        # With only middle populated, descriptor positions are:
        #   slot 0 (sink) = -1, slot 1 (recent) = -1, slot 2 (middle) = 1.
        N = L * H
        for i in range(N):
            assert int(sk_cpu[i * 3 + 0]) == -1
            assert int(sk_cpu[i * 3 + 1]) == -1
            assert int(sk_cpu[i * 3 + 2]) == 1

        # Per-head total length = mm * F.
        cu_cpu = cu.cpu()
        for i in range(N):
            assert int(cu_cpu[i + 1] - cu_cpu[i]) == mm * F

        _ops_mod.pyramidkv_pack(mgr, sk, sg, sl, dst)
        torch.cuda.synchronize()

        ref_chunks = []
        for l in range(L):
            for h in range(H):
                ref_chunks.append(mgr.middle_k_pool()[l, h, :mm].reshape(mm * F, D))
        ref_k = torch.cat(ref_chunks, dim=0)
        total = N * mm * F
        out_k = mgr.k_flat_out()[:total, 0, :]
        torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)

"""M1.4 test: pyramidkv_pack kernel correctness.

Bypasses the (yet-to-be-written) plan stage by hand-crafting the
descriptor tensors that plan() would produce, populating the manager's
sink/middle/recent pools with synthetic data, running the pack op, and
comparing the resulting k_flat_out / v_flat_out against a manual
torch.cat reference.

This pins down the pack kernel's contract before M1.5 (update kernels)
or M1.6 (Python pipeline cutover) lands.
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


def _make_manager(L=2, H=4, D=16, F=8, ms=2, mm=4, mr=2):
    _try_load_or_skip()
    Cls = torch.classes.adahead.PyramidKVCacheManager
    # Plan B — old pyramidkv_plan does cross-layer single-shot pack, so the
    # pack workspace needs L "chunks" worth of capacity.
    return Cls(L, H, D, F, ms, mm, mr, "cuda:0", "bfloat16", L), L, H, D, F, ms, mm, mr


@pytest.mark.gpu
class TestPyramidKVPack:
    def test_pack_single_segment_per_kind(self):
        """Each (layer, head) has one segment per kind, all slots filled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr, L, H, D, F, ms, mm, mr = _make_manager()

        # Populate K/V pools with deterministic data.
        # sink_k_pool[layer, head, slot, frame_token, dim]
        torch.manual_seed(0)
        sink_k = mgr.sink_k_pool()
        sink_v = mgr.sink_v_pool()
        middle_k = mgr.middle_k_pool()
        middle_v = mgr.middle_v_pool()
        recent_k = mgr.recent_k_pool()
        recent_v = mgr.recent_v_pool()

        sink_k.copy_(torch.randn_like(sink_k))
        sink_v.copy_(torch.randn_like(sink_v))
        middle_k.copy_(torch.randn_like(middle_k))
        middle_v.copy_(torch.randn_like(middle_v))
        recent_k.copy_(torch.randn_like(recent_k))
        recent_v.copy_(torch.randn_like(recent_v))

        # Build descriptors as if every slot is filled.
        # For each (layer, head), 3 segments: sink (ms slots), middle (mm), recent (mr).
        N = L * H
        max_seg = 3
        device = "cuda:0"

        src_kind_list = []
        src_slot_global_list = []
        seg_lengths_list = []
        dst_token_offsets_list = []

        running_offset = 0
        # Reference: build the expected concatenation in Python.
        ref_k_chunks = []
        ref_v_chunks = []

        for l in range(L):
            for h in range(H):
                for kind, (n_slots, k_pool, v_pool) in enumerate([
                    (ms, sink_k, sink_v),
                    (mm, middle_k, middle_v),
                    (mr, recent_k, recent_v),
                ]):
                    # slot_global indexes into pool flattened as [L*H, max_slots, F, D]
                    head_idx = l * H + h
                    slot_global_first = head_idx * n_slots + 0  # start at slot 0
                    seg_len_tokens = n_slots * F
                    src_kind_list.append(kind)
                    src_slot_global_list.append(slot_global_first)
                    seg_lengths_list.append(seg_len_tokens)
                    dst_token_offsets_list.append(running_offset)
                    running_offset += seg_len_tokens

                    # Reference: this segment's data is pool[l, h, 0:n_slots, :, :]
                    seg_ref = k_pool[l, h, :n_slots].reshape(seg_len_tokens, D)
                    ref_k_chunks.append(seg_ref)
                    seg_ref_v = v_pool[l, h, :n_slots].reshape(seg_len_tokens, D)
                    ref_v_chunks.append(seg_ref_v)

        src_kind_t = torch.tensor(src_kind_list, dtype=torch.int32, device=device)
        src_slot_global_t = torch.tensor(src_slot_global_list, dtype=torch.int32, device=device)
        seg_lengths_t = torch.tensor(seg_lengths_list, dtype=torch.int32, device=device)
        dst_token_offsets_t = torch.tensor(dst_token_offsets_list, dtype=torch.int32, device=device)

        # Run pack
        ns = torch.ops.adahead
        _ops_mod.pyramidkv_pack(mgr, src_kind_t, src_slot_global_t, seg_lengths_t, dst_token_offsets_t)
        torch.cuda.synchronize()

        ref_k = torch.cat(ref_k_chunks, dim=0)
        ref_v = torch.cat(ref_v_chunks, dim=0)
        out_k = mgr.k_flat_out()[:running_offset, 0, :]
        out_v = mgr.v_flat_out()[:running_offset, 0, :]

        torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)
        torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)

    def test_pack_inactive_segments_skipped(self):
        """Segments with src_kind = -1 must not write to dst."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr, L, H, D, F, ms, mm, mr = _make_manager()

        # Pre-fill the destination workspace with a sentinel.
        sentinel = torch.bfloat16(torch.tensor(7.5)).item() if hasattr(torch.bfloat16, "__call__") else 7.5
        mgr.k_flat_out().fill_(sentinel)
        mgr.v_flat_out().fill_(sentinel)

        device = "cuda:0"
        # Single inactive segment.
        src_kind_t = torch.tensor([-1], dtype=torch.int32, device=device)
        src_slot_global_t = torch.tensor([0], dtype=torch.int32, device=device)
        seg_lengths_t = torch.tensor([F], dtype=torch.int32, device=device)
        dst_token_offsets_t = torch.tensor([0], dtype=torch.int32, device=device)

        ns = torch.ops.adahead
        _ops_mod.pyramidkv_pack(mgr, src_kind_t, src_slot_global_t, seg_lengths_t, dst_token_offsets_t)
        torch.cuda.synchronize()

        # Sentinel must be intact.
        assert (mgr.k_flat_out() == sentinel).all().item()
        assert (mgr.v_flat_out() == sentinel).all().item()

    def test_pack_zero_length_segment_skipped(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mgr, L, H, D, F, ms, mm, mr = _make_manager()
        mgr.k_flat_out().fill_(3.0)
        mgr.v_flat_out().fill_(3.0)
        device = "cuda:0"
        src_kind_t = torch.tensor([0], dtype=torch.int32, device=device)
        src_slot_global_t = torch.tensor([0], dtype=torch.int32, device=device)
        seg_lengths_t = torch.tensor([0], dtype=torch.int32, device=device)
        dst_token_offsets_t = torch.tensor([0], dtype=torch.int32, device=device)
        ns = torch.ops.adahead
        _ops_mod.pyramidkv_pack(mgr, src_kind_t, src_slot_global_t, seg_lengths_t, dst_token_offsets_t)
        torch.cuda.synchronize()
        assert (mgr.k_flat_out() == 3.0).all().item()

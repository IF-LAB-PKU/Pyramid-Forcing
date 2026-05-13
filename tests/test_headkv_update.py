"""M1.5 test: headkv_update kernel correctness.

Each (frame_in_block, head) -> pool slot write is verified by populating
the source new_k/new_v with deterministic data, calling headkv_update,
and checking the manager's pool slots match the corresponding source
chunks.

Companion to test_headkv_pack: pack reads pools, update writes them.
After populating with update + reading with plan + pack, the round-trip
should yield exactly the input new_k/new_v concatenated head-major.
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


@pytest.mark.gpu
class TestHeadKVUpdate:
    def test_update_single_layer_to_recent(self):
        """Write 3 frames into recent_pool for layer 0, verify slot contents."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        Cls = torch.classes.adahead.HeadKVCacheManager
        L, H, D, F, ms, mm, mr = 1, 4, 16, 8, 2, 2, 4
        mgr = Cls(L, H, D, F, ms, mm, mr, "cuda:0", "bfloat16", L)

        device = "cuda:0"
        frames_in_block = 3

        torch.manual_seed(0)
        # new_k layout: [frames_in_block, H, F, D] flattened
        new_k = torch.randn(frames_in_block, H, F, D, dtype=torch.bfloat16, device=device)
        new_v = torch.randn(frames_in_block, H, F, D, dtype=torch.bfloat16, device=device)

        # Each frame's head writes to recent[layer=0, head=h, slot=frame_idx].
        # Build descriptor:
        N = frames_in_block * H
        dst_kind_list = []
        dst_slot_list = []
        src_frame_list = []
        src_head_list = []
        for f in range(frames_in_block):
            for h in range(H):
                dst_kind_list.append(2)  # recent pool
                # recent_pool layout: [L, H, mr, F, D]; slot_global = (l*H + h)*mr + slot
                slot_global = (0 * H + h) * mr + f
                dst_slot_list.append(slot_global)
                src_frame_list.append(f)
                src_head_list.append(h)

        dst_kind_t = torch.tensor(dst_kind_list, dtype=torch.int32, device=device)
        dst_slot_t = torch.tensor(dst_slot_list, dtype=torch.int32, device=device)
        src_frame_t = torch.tensor(src_frame_list, dtype=torch.int32, device=device)
        src_head_t = torch.tensor(src_head_list, dtype=torch.int32, device=device)

        _empty_pos = torch.empty(0, dtype=torch.int64, device=device)
        torch.ops.adahead.headkv_update(
            mgr, new_k, new_v, _empty_pos,
            dst_kind_t, dst_slot_t, src_frame_t, src_head_t
        )
        torch.cuda.synchronize()

        # Verify: recent_k_pool[0, h, f] == new_k[f, h]
        recent_k = mgr.recent_k_pool()
        recent_v = mgr.recent_v_pool()
        for f in range(frames_in_block):
            for h in range(H):
                torch.testing.assert_close(
                    recent_k[0, h, f], new_k[f, h], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    recent_v[0, h, f], new_v[f, h], atol=0, rtol=0
                )
        # Slots beyond frames_in_block should still be zero.
        if mr > frames_in_block:
            assert (recent_k[0, :, frames_in_block:] == 0).all().item()

    def test_update_inactive_skip(self):
        """dst_kind = -1 must not write."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        Cls = torch.classes.adahead.HeadKVCacheManager
        mgr = Cls(1, 2, 8, 4, 1, 2, 1, "cuda:0", "bfloat16", 1)
        device = "cuda:0"

        new_k = torch.full((1, 2, 4, 8), 7.0, dtype=torch.bfloat16, device=device)
        new_v = torch.full_like(new_k, 9.0)

        dst_kind_t = torch.tensor([-1], dtype=torch.int32, device=device)
        dst_slot_t = torch.tensor([0], dtype=torch.int32, device=device)
        src_frame_t = torch.tensor([0], dtype=torch.int32, device=device)
        src_head_t = torch.tensor([0], dtype=torch.int32, device=device)

        _empty_pos = torch.empty(0, dtype=torch.int64, device=device)
        torch.ops.adahead.headkv_update(
            mgr, new_k, new_v, _empty_pos,
            dst_kind_t, dst_slot_t, src_frame_t, src_head_t
        )
        torch.cuda.synchronize()

        # All pools should still be zero.
        assert (mgr.sink_k_pool() == 0).all().item()
        assert (mgr.recent_k_pool() == 0).all().item()
        assert (mgr.middle_k_pool() == 0).all().item()

    def test_update_then_plan_then_pack_roundtrip(self):
        """End-to-end: update writes new K/V, plan+pack reads back, equal to manual cat."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        _try_load_or_skip()
        Cls = torch.classes.adahead.HeadKVCacheManager
        L, H, D, F, ms, mm, mr = 1, 3, 8, 4, 1, 1, 2
        mgr = Cls(L, H, D, F, ms, mm, mr, "cuda:0", "bfloat16", L)
        device = "cuda:0"

        frames_in_block = 2  # ms + mr - mm = 2 (first goes to sink, second to recent)
        torch.manual_seed(123)
        new_k = torch.randn(frames_in_block, H, F, D, dtype=torch.bfloat16, device=device)
        new_v = torch.randn(frames_in_block, H, F, D, dtype=torch.bfloat16, device=device)

        # Write frame 0 → sink slot 0; frame 1 → recent slot 0; for every head.
        dst_kind, dst_slot, src_frame, src_head = [], [], [], []
        for h in range(H):
            # sink: kind=0, slot_global = (0*H + h) * ms + 0
            dst_kind.append(0)
            dst_slot.append((0 * H + h) * ms + 0)
            src_frame.append(0)
            src_head.append(h)
            # recent: kind=2, slot_global = (0*H + h) * mr + 0
            dst_kind.append(2)
            dst_slot.append((0 * H + h) * mr + 0)
            src_frame.append(1)
            src_head.append(h)

        dst_kind_t = torch.tensor(dst_kind, dtype=torch.int32, device=device)
        dst_slot_t = torch.tensor(dst_slot, dtype=torch.int32, device=device)
        src_frame_t = torch.tensor(src_frame, dtype=torch.int32, device=device)
        src_head_t = torch.tensor(src_head, dtype=torch.int32, device=device)

        _empty_pos = torch.empty(0, dtype=torch.int64, device=device)
        torch.ops.adahead.headkv_update(
            mgr, new_k, new_v, _empty_pos,
            dst_kind_t, dst_slot_t, src_frame_t, src_head_t
        )

        # Plan needs valid_count to know how many slots are used.
        vc = mgr.valid_count()
        vc[..., 0] = 1  # sink: 1 slot
        vc[..., 1] = 0  # middle: 0
        vc[..., 2] = 1  # recent: 1

        cu, sk, sg, sl, dst_off = torch.ops.adahead.headkv_plan(mgr, 0, 1)
        torch.ops.adahead.headkv_pack(mgr, sk, sg, sl, dst_off)
        torch.cuda.synchronize()

        # Expected: per head h, output is new_k[0, h] (sink) followed by new_k[1, h] (recent)
        ref_chunks = []
        for h in range(H):
            ref_chunks.append(new_k[0, h])  # sink frame
            ref_chunks.append(new_k[1, h])  # recent frame
        ref_k = torch.cat(ref_chunks, dim=0).reshape(-1, D)

        total = H * 2 * F
        out_k = mgr.k_flat_out()[:total, 0, :]
        torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)

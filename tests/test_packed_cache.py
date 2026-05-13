"""M1.6 part 1 test: HeadKVPackedCache vertical slice.

Validates that the (plan, pack, update) ops cohere into a working
FIFO recent-only KV cache. Compares against a Python reference.
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
class TestHeadKVPackedCache:
    def test_fifo_recent_roundtrip(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from headkv.packed_cache import HeadKVPackedCache

        L, H, D, F = 2, 3, 8, 4
        max_recent = 4
        cache = HeadKVPackedCache(
            num_layers=L, num_heads=H, head_dim=D, frame_seqlen=F,
            max_recent_frames=max_recent,
        )

        device = "cuda:0"
        torch.manual_seed(7)

        # Drive 3 update steps, each writing 2 frames into each layer.
        ref_history = [[] for _ in range(L)]  # per-layer list of frames as [H, F, D]
        for step in range(3):
            for l in range(L):
                new_k = torch.randn(2, H, F, D, dtype=torch.bfloat16, device=device)
                new_v = torch.randn_like(new_k)
                cache.update(l, new_k, new_v)
                # Reference: keep last max_recent frames per layer.
                for f in range(2):
                    ref_history[l].append((new_k[f].clone(), new_v[f].clone()))

        torch.cuda.synchronize()

        # Reference output: for each (layer, head), concat last max_recent frames in order.
        ref_chunks_k = []
        ref_chunks_v = []
        # Match plan layout: layer-major, head-major, then concat slots in slot index order.
        # Our FIFO writes slots 0,1,2,3,4,5; with max_recent=4 we'd have ring slots 0..3
        # but the slot SEQUENCE in memory is: new content overwrites slot 0, 1, 2... so
        # the final pool layout has slots [0..3] containing the LAST 4 frames in WRITE order.
        # Plan reads slots [0..valid_count) in sequence, so the order is:
        # If we wrote 6 frames total, slots ended up as:
        #   slot 0 = frame 4, slot 1 = frame 5, slot 2 = frame 2, slot 3 = frame 3
        # We need to reproduce this exact ordering.

        # Build slot index → frame for each (layer, head):
        for l in range(L):
            n_writes = len(ref_history[l])
            n_kept = min(n_writes, max_recent)
            slot_to_frame = [0] * max_recent
            for write_idx in range(n_writes):
                slot = write_idx % max_recent
                slot_to_frame[slot] = write_idx
            # Plan currently emits slots 0..valid_count in order.
            valid_slots = min(n_writes, max_recent)
            for h in range(H):
                for slot in range(valid_slots):
                    frame_idx = slot_to_frame[slot]
                    k_chunk, v_chunk = ref_history[l][frame_idx]
                    # k_chunk shape: [H, F, D]; we want chunk for head h as [F, D]
                    ref_chunks_k.append(k_chunk[h])
                    ref_chunks_v.append(v_chunk[h])

        ref_k = torch.cat(ref_chunks_k, dim=0)  # [total_tokens, D]
        ref_v = torch.cat(ref_chunks_v, dim=0)

        ro = cache.readout(current_t=0, pass_kind=0)
        out_k = ro.k_flat[:, 0, :]
        out_v = ro.v_flat[:, 0, :]

        torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)
        torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)

    def test_under_capacity(self):
        """Fewer writes than max_recent: only valid slots returned."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from headkv.packed_cache import HeadKVPackedCache

        L, H, D, F = 1, 2, 4, 4
        cache = HeadKVPackedCache(
            num_layers=L, num_heads=H, head_dim=D, frame_seqlen=F,
            max_recent_frames=4,
        )
        device = "cuda:0"
        torch.manual_seed(0)
        new_k = torch.randn(2, H, F, D, dtype=torch.bfloat16, device=device)
        new_v = torch.randn_like(new_k)
        cache.update(0, new_k, new_v)

        ro = cache.readout()
        # 2 frames × H heads × F tokens = 2*2*4 = 16 tokens total
        assert ro.total_tokens == 2 * H * F
        # Each head should have its 2 frames in order.
        for h in range(H):
            head_start = h * 2 * F
            head_end = head_start + 2 * F
            chunk = ro.k_flat[head_start:head_end, 0, :]
            ref = torch.cat([new_k[0, h], new_k[1, h]], dim=0)
            torch.testing.assert_close(chunk, ref, atol=0, rtol=0)

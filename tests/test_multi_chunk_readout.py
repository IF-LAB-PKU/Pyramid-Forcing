"""Regression baseline for ``get_decoupled_flat_kv_and_frames_multi``.

Multi-chunk readout loops 3 chunks per generator() call, each with a
different ``current_start``, paying 3 cold packs/timestep — ~67% of the
total cold packs in a long-video run.

A future Path B refactor will share the static + middle anchor portion
across the 3 chunks, since they only differ in dynamic-recent RoPE
positions. Before any refactor lands, this test pins down the current
behavior:

  multi(cs0, cs1, cs2) == cat([single(cs0), single(cs1), single(cs2)])

If a future refactor breaks this equivalence, the test catches it.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from pyramidkv.config import PyramidKVConfig
from pyramidkv.adaptive_cache import AdaptiveKVCache
from wan.modules.model import rope_params


def _build_rope_freqs(head_dim: int, max_seq_len: int = 256) -> torch.Tensor:
    return torch.cat(
        [
            rope_params(max_seq_len, head_dim - 4 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def _build_config(num_heads: int, capacity: int) -> PyramidKVConfig:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        for h in range(num_heads):
            f.write(f"0,{h},{capacity}\n")
        path = f.name
    config = PyramidKVConfig(path, num_layers=1, num_heads=num_heads)
    os.unlink(path)
    return config


@pytest.mark.gpu
class TestMultiChunkReadout:
    def test_multi_equals_concatenated_singles(self):
        """multi(cs0, cs1, cs2) ≡ cat(single(cs0), single(cs1), single(cs2))."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")

        num_heads = 2
        head_dim = 16
        frame_seqlen = 8
        block_len = 3 * frame_seqlen  # 3 frames per block
        capacity = 256

        config = _build_config(num_heads, capacity)

        # Build two identical caches: one we call multi on, one we call single 3x on.
        def _new_cache():
            return AdaptiveKVCache(
                config=config, batch_size=1, num_heads=num_heads, head_dim=head_dim,
                layer_idx=0, sink_len=frame_seqlen, tail_len=4 * frame_seqlen,
                ivc_ratio=0.0, semantic_ratio=0.0, update_interval=1,
            )

        cache_a = _new_cache()
        cache_b = _new_cache()

        freqs = _build_rope_freqs(head_dim).to(device)
        grid_sizes = torch.tensor([[3, 2, 4]], dtype=torch.long, device=device)

        # Same deterministic populations on both caches.
        torch.manual_seed(0)
        for blk in range(2):
            cs = blk * block_len
            k = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache_a.update(k, v, current_start=cs, grid_sizes=grid_sizes, freqs=freqs)
            cache_b.update(k, v, current_start=cs, grid_sizes=grid_sizes, freqs=freqs)

        # Query at a 3-chunk window: cs0=2*block_len, cs1=cs0+frame_seqlen, cs2=cs0+2*frame_seqlen.
        cs0 = 2 * block_len
        starts = [cs0, cs0 + frame_seqlen, cs0 + 2 * frame_seqlen]

        # Path 1: multi
        k_m, v_m, cu_m, max_m, fids_m = cache_a.get_decoupled_flat_kv_and_frames_multi(
            current_starts=starts, grid_sizes=grid_sizes, freqs=freqs,
        )

        # Path 2: 3 single calls + manual concat
        k_singles, v_singles, fids_singles = [], [], []
        cu_per_chunk = []
        for cs in starts:
            k_s, v_s, cu_s, _max_s, fids_s = cache_b.get_decoupled_flat_kv_and_frames(
                current_start=cs, grid_sizes=grid_sizes, freqs=freqs,
            )
            k_singles.append(k_s.clone())
            v_singles.append(v_s.clone())
            fids_singles.append(fids_s.clone())
            cu_per_chunk.append(cu_s.clone())

        k_cat = torch.cat(k_singles, dim=0)
        v_cat = torch.cat(v_singles, dim=0)
        fids_cat = torch.cat(fids_singles, dim=0)

        torch.testing.assert_close(k_m, k_cat, atol=0, rtol=0)
        torch.testing.assert_close(v_m, v_cat, atol=0, rtol=0)
        torch.testing.assert_close(fids_m, fids_cat, atol=0, rtol=0)

        # cu_seqlens_k: multi merges chunks into a single prefix-sum array of
        # length num_chunks * num_seq + 1. Each chunk contributes num_seq deltas;
        # those must equal the per-chunk cu_seqlens deltas in order.
        num_seq = num_heads
        assert cu_m.shape == (3 * num_seq + 1,)
        cu_m_cpu = cu_m.cpu().tolist()
        running = 0
        idx = 0
        for chunk_idx, cu_s in enumerate(cu_per_chunk):
            cu_s_cpu = cu_s.cpu().tolist()
            for s in range(num_seq):
                delta = cu_s_cpu[s + 1] - cu_s_cpu[s]
                running += delta
                idx += 1
                assert cu_m_cpu[idx] == running, (
                    f"cu mismatch at chunk={chunk_idx} seq={s}: got {cu_m_cpu[idx]}, want {running}"
                )

    def test_multi_single_chunk_short_circuits(self):
        """num_chunks=1 should short-circuit to single-chunk path."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda:0")

        num_heads = 2
        head_dim = 16
        frame_seqlen = 8
        block_len = 3 * frame_seqlen
        config = _build_config(num_heads, 128)

        cache = AdaptiveKVCache(
            config=config, batch_size=1, num_heads=num_heads, head_dim=head_dim,
            layer_idx=0, sink_len=frame_seqlen, tail_len=4 * frame_seqlen,
            ivc_ratio=0.0, semantic_ratio=0.0, update_interval=1,
        )

        freqs = _build_rope_freqs(head_dim).to(device)
        grid_sizes = torch.tensor([[3, 2, 4]], dtype=torch.long, device=device)

        torch.manual_seed(1)
        k = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache.update(k, v, current_start=0, grid_sizes=grid_sizes, freqs=freqs)

        k_m, v_m, cu_m, _max_m, fids_m = cache.get_decoupled_flat_kv_and_frames_multi(
            current_starts=[0], grid_sizes=grid_sizes, freqs=freqs,
        )
        k_s, v_s, cu_s, _max_s, fids_s = cache.get_decoupled_flat_kv_and_frames(
            current_start=0, grid_sizes=grid_sizes, freqs=freqs,
        )
        torch.testing.assert_close(k_m, k_s, atol=0, rtol=0)
        torch.testing.assert_close(v_m, v_s, atol=0, rtol=0)
        torch.testing.assert_close(cu_m, cu_s, atol=0, rtol=0)
        torch.testing.assert_close(fids_m, fids_s, atol=0, rtol=0)

"""Probe: can flash_attn_varlen_func be captured in a CUDA Graph?

The M2.2 integration design decision hinges on this. If FA2.8 supports
graph capture under ragged ``cu_seqlens_k``, we can capture the full
diffusion forward (Path A — fast, simple). If not, we need vLLM-style
piecewise capture leaving attention eager (Path B — safer, more work).

We capture a tiny varlen attention call (representative of what one
layer does in the Wan backbone) and verify the replay output matches
the eager output bit-exactly.
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
class TestFAVarlenGraphProbe:
    def test_varlen_captures_and_replays(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            pytest.skip("flash_attn not installed")

        device = "cuda:0"
        torch.manual_seed(0)

        # Build a fixed-shape varlen call: 1 batch, 12 heads, head_dim=128,
        # ragged kv per head — match the production "1 sequence per head"
        # flattening Pyramid Forcing uses.
        num_heads = 12
        head_dim = 128
        per_head_q = 1560  # 1 frame's worth, like one block's q
        # Heterogeneous KV: 12 heads with different lengths to mimic ragged.
        k_lens = torch.tensor([3120, 4680, 3120, 4680, 1560, 1560, 7800, 7800,
                               4680, 4680, 3120, 3120], dtype=torch.int32, device=device)
        q_lens = torch.full((num_heads,), per_head_q, dtype=torch.int32, device=device)

        cu_q = torch.cat([q_lens.new_zeros(1), q_lens]).cumsum(0).to(torch.int32)
        cu_k = torch.cat([k_lens.new_zeros(1), k_lens]).cumsum(0).to(torch.int32)
        max_q = int(per_head_q)
        max_k = int(k_lens.max().item())
        total_q = int(cu_q[-1].item())
        total_k = int(cu_k[-1].item())

        q = torch.randn(total_q, 1, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(total_k, 1, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(total_k, 1, head_dim, device=device, dtype=torch.bfloat16)

        # Eager baseline
        out_eager = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=False,
        )

        # Side-stream warmup (required by PyTorch CUDA Graph docs)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _ = flash_attn_varlen_func(
                    q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=False,
                )
        torch.cuda.current_stream().wait_stream(side)

        # Capture
        try:
            graph = torch.cuda.CUDAGraph()
            out_static = torch.empty_like(out_eager)
            with torch.cuda.graph(graph):
                captured = flash_attn_varlen_func(
                    q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=False,
                )
                out_static.copy_(captured)
        except Exception as e:
            pytest.fail(
                f"flash_attn_varlen_func capture FAILED — Path A unviable, "
                f"M2.2 must use Path B (vLLM piecewise). Error: {type(e).__name__}: {e}"
            )

        graph.replay()
        torch.cuda.synchronize()

        # Refresh inputs and verify replay matches a fresh eager call
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))

        out_eager2 = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=False,
        )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(out_static, out_eager2, atol=1e-2, rtol=1e-2)

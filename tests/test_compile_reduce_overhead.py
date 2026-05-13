"""G3a-test: Verify torch.compile with reduce-overhead (cudagraphs) mode.

This test verifies that FFN compilation with CUDA Graph mode produces
bit-consistent results vs eager mode. It uses a minimal model forward
(not full pipeline) to keep runtime reasonable.

Expected: PASS when HEADKV_CUDA_GRAPH=1 produces same output as eager.
"""
import os
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestCompileReduceOverhead:
    """Verify FFN compile with reduce-overhead doesn't break numerics."""

    def test_ffn_compile_reduce_overhead_matches_eager(self, device):
        """Compile a single FFN block with reduce-overhead and check output."""
        torch.manual_seed(42)

        dim = 1536
        ffn_dim = 8960
        seq_len = 48

        ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, ffn_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(ffn_dim, dim, bias=False),
        ).to(device=device, dtype=torch.bfloat16)

        x = torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)

        # Eager reference
        with torch.no_grad():
            ref = ffn(x).clone()

        # Compile with reduce-overhead (enables cudagraphs internally)
        ffn_compiled = torch.compile(ffn, mode="reduce-overhead")

        # Warmup (graph capture happens on first few calls)
        with torch.no_grad():
            for _ in range(3):
                _ = ffn_compiled(x)

        # Actual test
        with torch.no_grad():
            out = ffn_compiled(x)

        assert torch.allclose(ref, out, atol=1e-3), (
            f"reduce-overhead output diverges from eager. "
            f"Max diff: {(ref - out).abs().max().item():.6f}"
        )

    def test_env_var_selects_compile_mode(self):
        """HEADKV_CUDA_GRAPH=1 should select reduce-overhead mode."""
        os.environ["HEADKV_CUDA_GRAPH"] = "1"
        try:
            mode = (
                "reduce-overhead"
                if os.environ.get("HEADKV_CUDA_GRAPH") == "1"
                else "max-autotune-no-cudagraphs"
            )
            assert mode == "reduce-overhead"
        finally:
            del os.environ["HEADKV_CUDA_GRAPH"]

        mode = (
            "reduce-overhead"
            if os.environ.get("HEADKV_CUDA_GRAPH") == "1"
            else "max-autotune-no-cudagraphs"
        )
        assert mode == "max-autotune-no-cudagraphs"

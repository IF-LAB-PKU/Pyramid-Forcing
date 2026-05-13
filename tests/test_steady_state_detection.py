"""G2-test: Steady-state detection for CUDA Graph capture eligibility.

Verifies that _steady_state_reached becomes True after enough blocks
have been processed (all cyclic buckets full, stride windows full,
dynamic recent windows full). After that point, per-head KV lengths
must not change.

Expected: FAIL on current code (_steady_state_reached doesn't exist).
After G2-impl: PASS.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_cache_many_blocks(device, num_blocks=10):
    """Build a cache with cyclic+stride strategies and feed it many blocks."""
    from headkv.config import HeadKVConfig
    from headkv.adaptive_cache import AdaptiveKVCache
    from wan.modules.model import rope_params
    import tempfile, os

    num_layers = 1
    num_heads = 4
    head_dim = 128
    batch_size = 1
    frame_seqlen = 16
    num_frame_per_block = 3
    block_len = num_frame_per_block * frame_seqlen

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for h in range(num_heads):
            label = -1 if h % 3 == 0 else (1 if h % 3 == 1 else 2)
            f.write(f"0,{h},{label}\n")
        config_path = f.name

    config = HeadKVConfig(config_path, num_layers=num_layers, num_heads=num_heads)
    os.unlink(config_path)

    cache = AdaptiveKVCache(
        config=config,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_idx=0,
        sink_len=1 * frame_seqlen,
        tail_len=4 * frame_seqlen,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=1,
    )

    freqs = torch.cat([
        rope_params(512, head_dim - 4 * (head_dim // 6)),
        rope_params(512, 2 * (head_dim // 6)),
        rope_params(512, 2 * (head_dim // 6)),
    ], dim=1).to(device)

    grid_sizes = torch.tensor([[num_frame_per_block, 4, 4]], dtype=torch.long, device=device)

    return cache, freqs, grid_sizes, block_len, num_blocks


class TestSteadyStateDetection:
    """Verify _steady_state_reached flag behavior."""

    def test_becomes_true_after_warmup(self, device):
        """After enough blocks, _steady_state_reached should be True."""
        cache, freqs, grid_sizes, block_len, num_blocks = _make_cache_many_blocks(device)

        for blk in range(num_blocks):
            current_start = blk * block_len
            k = torch.randn(1, block_len, 4, 128, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes, freqs=freqs)
            cache.get_decoupled_flat_kv_and_frames(
                current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
            )

        assert hasattr(cache, "_steady_state_reached"), \
            "_steady_state_reached attribute missing from AdaptiveKVCache"
        assert cache._steady_state_reached, \
            f"Expected steady state after {num_blocks} blocks"

    def test_shapes_stable_after_steady(self, device):
        """Once steady, per-head KV lengths must not change between readouts."""
        cache, freqs, grid_sizes, block_len, num_blocks = _make_cache_many_blocks(device)

        prev_cu = None
        stable_since = None

        for blk in range(num_blocks):
            current_start = blk * block_len
            k = torch.randn(1, block_len, 4, 128, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes, freqs=freqs)
            _, _, cu, _, _ = cache.get_decoupled_flat_kv_and_frames(
                current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
            )

            cu_cpu = cu.cpu()
            if prev_cu is not None and torch.equal(cu_cpu, prev_cu):
                if stable_since is None:
                    stable_since = blk
            else:
                stable_since = None
            prev_cu = cu_cpu

        assert stable_since is not None, "cu_seqlens_k never stabilized"
        assert stable_since < num_blocks - 1, \
            f"Stabilized too late (block {stable_since}/{num_blocks})"

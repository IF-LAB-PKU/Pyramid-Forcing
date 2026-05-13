"""M6-test: Verify fast-path for noisy overwrite is bit-exact vs slow path.

The fast-path bypasses the 360-iter Python loop in update() when all
invariants hold (steady state, noisy mode, overwrite, sufficient capacity).
It must produce exactly the same _dyn_store_* state as the slow path.

Expected: FAIL on current code (no _try_fast_path_noisy_overwrite method).
After M6-impl: PASS.
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


def _build_cache(device):
    from pyramidkv.config import PyramidKVConfig
    from pyramidkv.adaptive_cache import AdaptiveKVCache
    from wan.modules.model import rope_params
    import tempfile

    num_heads = 4
    head_dim = 128
    frame_seqlen = 16
    num_frame_per_block = 3
    block_len = num_frame_per_block * frame_seqlen

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for h in range(num_heads):
            label = -1 if h % 3 == 0 else (1 if h % 3 == 1 else 2)
            f.write(f"0,{h},{label}\n")
        config_path = f.name

    config = PyramidKVConfig(config_path, num_layers=1, num_heads=num_heads)
    os.unlink(config_path)

    cache = AdaptiveKVCache(
        config=config, batch_size=1, num_heads=num_heads, head_dim=head_dim,
        layer_idx=0, sink_len=1 * frame_seqlen, tail_len=4 * frame_seqlen,
        ivc_ratio=0.0, semantic_ratio=0.0, update_interval=1,
    )
    freqs = torch.cat([
        rope_params(512, head_dim - 4 * (head_dim // 6)),
        rope_params(512, 2 * (head_dim // 6)),
        rope_params(512, 2 * (head_dim // 6)),
    ], dim=1).to(device)
    grid_sizes = torch.tensor([[num_frame_per_block, 4, 4]], dtype=torch.long, device=device)
    return cache, freqs, grid_sizes, block_len, num_heads, head_dim


def _warmup_to_steady(cache, freqs, grid_sizes, block_len, num_heads, head_dim, device, num_blocks=8):
    """Run enough blocks to reach steady state."""
    for blk in range(num_blocks):
        current_start = blk * block_len
        k = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
                     cache_update_mode="clean")
        cache.get_decoupled_flat_kv_and_frames(
            current_start=current_start, grid_sizes=grid_sizes, freqs=freqs)


class TestUpdateFastPath:
    """Verify fast-path equivalence with slow path."""

    def test_fast_path_method_exists(self, device):
        """_try_fast_path_noisy_overwrite must exist on AdaptiveKVCache."""
        from pyramidkv.adaptive_cache import AdaptiveKVCache
        assert hasattr(AdaptiveKVCache, "_try_fast_path_noisy_overwrite"), (
            "_try_fast_path_noisy_overwrite method missing — M6-impl not applied"
        )

    def test_fast_path_bit_exact_vs_slow(self, device):
        """Fast-path output must be bit-exact with slow-path on steady noisy overwrite."""
        torch.manual_seed(42)
        cache_fast, freqs, grid_sizes, block_len, num_heads, head_dim = _build_cache(device)
        cache_slow, _, _, _, _, _ = _build_cache(device)

        # Warmup both identically to reach steady state
        torch.manual_seed(42)
        _warmup_to_steady(cache_fast, freqs, grid_sizes, block_len, num_heads, head_dim, device)
        torch.manual_seed(42)
        _warmup_to_steady(cache_slow, freqs, grid_sizes, block_len, num_heads, head_dim, device)

        assert cache_fast._steady_state_reached, "fast cache did not reach steady state"
        assert cache_slow._steady_state_reached, "slow cache did not reach steady state"

        # Now do a noisy overwrite call
        # current_start = previous block (re-entering same block that's now stored)
        # overwrite mode: current_end <= global_end_index
        prev_start = (8 - 1) * block_len  # last block we appended
        torch.manual_seed(100)
        k_in = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v_in = torch.randn_like(k_in)

        # Slow path: force via env var
        os.environ["PYRAMIDKV_DISABLE_M6_FASTPATH"] = "1"
        try:
            cache_slow.update(k_in.clone(), v_in.clone(),
                              current_start=prev_start, grid_sizes=grid_sizes,
                              freqs=freqs, cache_update_mode="noisy")
        finally:
            del os.environ["PYRAMIDKV_DISABLE_M6_FASTPATH"]

        # Fast path
        cache_fast.update(k_in.clone(), v_in.clone(),
                          current_start=prev_start, grid_sizes=grid_sizes,
                          freqs=freqs, cache_update_mode="noisy")

        # Compare all dynamic stores bit-exact
        n = num_heads
        for i in range(n):
            sk_f = cache_fast._dyn_store_k[i]
            sk_s = cache_slow._dyn_store_k[i]
            sv_f = cache_fast._dyn_store_v[i]
            sv_s = cache_slow._dyn_store_v[i]
            assert sk_f is not None and sk_s is not None
            assert sk_f.shape == sk_s.shape, f"head {i} K shape diff"
            assert torch.equal(sk_f, sk_s), f"head {i} K content diff"
            assert torch.equal(sv_f, sv_s), f"head {i} V content diff"
            assert cache_fast._dyn_store_start[i] == cache_slow._dyn_store_start[i]
            assert cache_fast._dyn_store_len[i] == cache_slow._dyn_store_len[i]

    def test_fast_path_disabled_before_steady(self, device):
        """Fast-path must NOT be taken before steady state (first block)."""
        cache, freqs, grid_sizes, block_len, num_heads, head_dim = _build_cache(device)
        assert not cache._steady_state_reached

        k = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        # First call: not steady → fast path rejects → slow path executes
        # (We just verify it doesn't crash and cache still works)
        cache.update(k, v, current_start=0, grid_sizes=grid_sizes, freqs=freqs,
                     cache_update_mode="clean")
        # Sanity: dynamic stores populated
        assert cache._dyn_store_k[0] is not None

    def test_fast_path_bit_exact_clean(self, device):
        """Clean-pass fast-path output must match slow path bit-exact.

        Clean pass may additionally update middle-strategy anchors
        (cyclic/lag/stride) in osc mode. The fast-path inlines those
        calls via composition.update_all; must match.
        """
        torch.manual_seed(42)
        cache_fast, freqs, grid_sizes, block_len, num_heads, head_dim = _build_cache(device)
        cache_slow, _, _, _, _, _ = _build_cache(device)

        torch.manual_seed(42)
        _warmup_to_steady(cache_fast, freqs, grid_sizes, block_len, num_heads, head_dim, device)
        torch.manual_seed(42)
        _warmup_to_steady(cache_slow, freqs, grid_sizes, block_len, num_heads, head_dim, device)

        assert cache_fast._steady_state_reached
        assert cache_slow._steady_state_reached

        prev_start = (8 - 1) * block_len
        torch.manual_seed(200)
        k_in = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v_in = torch.randn_like(k_in)

        os.environ["PYRAMIDKV_DISABLE_M6_FASTPATH"] = "1"
        try:
            cache_slow.update(k_in.clone(), v_in.clone(),
                              current_start=prev_start, grid_sizes=grid_sizes,
                              freqs=freqs, cache_update_mode="clean")
        finally:
            del os.environ["PYRAMIDKV_DISABLE_M6_FASTPATH"]

        cache_fast.update(k_in.clone(), v_in.clone(),
                          current_start=prev_start, grid_sizes=grid_sizes,
                          freqs=freqs, cache_update_mode="clean")

        for i in range(num_heads):
            sk_f = cache_fast._dyn_store_k[i]
            sk_s = cache_slow._dyn_store_k[i]
            assert sk_f is not None and sk_s is not None
            assert sk_f.shape == sk_s.shape, f"head {i} K shape diff"
            assert torch.equal(sk_f, sk_s), f"head {i} K content diff"
            assert torch.equal(cache_fast._dyn_store_v[i], cache_slow._dyn_store_v[i]), \
                f"head {i} V content diff"
            assert cache_fast._dyn_store_start[i] == cache_slow._dyn_store_start[i]
            assert cache_fast._dyn_store_len[i] == cache_slow._dyn_store_len[i]

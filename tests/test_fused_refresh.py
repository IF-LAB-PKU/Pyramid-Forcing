"""M8-test: Fused refresh CUDA kernel bit-exact equivalence.

The fused kernel replaces the 360-iter Python loop in
_refresh_cached_readout_mutable_segments with a single CUDA launch.
It must produce bit-identical _ws_k/_ws_v/_ws_rope_pos/_ws_frame_ids
state vs the Python reference path.

Expected: FAIL on current code (no fused_refresh_tail kernel yet).
After M8-impl: PASS.
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


class TestFusedRefreshAvailable:
    def test_fused_refresh_module_exists(self):
        """_scatter_ext should expose fused_refresh_available() + fused_refresh_tail()."""
        from pyramidkv import _scatter_ext
        assert hasattr(_scatter_ext, "fused_refresh_available"), \
            "fused_refresh_available() missing — M8-impl not applied"
        assert hasattr(_scatter_ext, "fused_refresh_tail"), \
            "fused_refresh_tail() missing — M8-impl not applied"

    def test_env_toggle(self):
        """PYRAMIDKV_FUSED_REFRESH=1 should opt-in the fused path."""
        from pyramidkv import _scatter_ext
        # With toggle off, available() returns False (opt-in only)
        os.environ.pop("PYRAMIDKV_FUSED_REFRESH", None)
        assert not _scatter_ext.fused_refresh_available(), \
            "fused_refresh should be opt-in (default off)"


class TestFusedRefreshEquivalence:
    """Bit-exact equivalence of fused kernel vs Python refresh loop."""

    def _build_cache(self, device):
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
        config.frame_seq_length = frame_seqlen
        os.unlink(config_path)

        cache = AdaptiveKVCache(
            config=config, batch_size=1, num_heads=num_heads, head_dim=head_dim,
            layer_idx=0, sink_len=1 * frame_seqlen, tail_len=4 * frame_seqlen,
            ivc_ratio=0.0, semantic_ratio=0.0, update_interval=1,
            sink_grid_decoupling=True,
        )
        freqs = torch.cat([
            rope_params(512, head_dim - 4 * (head_dim // 6)),
            rope_params(512, 2 * (head_dim // 6)),
            rope_params(512, 2 * (head_dim // 6)),
        ], dim=1).to(device)
        grid_sizes = torch.tensor([[num_frame_per_block, 4, 4]], dtype=torch.long, device=device)
        return cache, freqs, grid_sizes, block_len, num_heads, head_dim

    def _warm_to_steady(self, cache, freqs, grid_sizes, block_len, num_heads, head_dim, device, n=8):
        for blk in range(n):
            current_start = blk * block_len
            k = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                         freqs=freqs, cache_update_mode="clean")
            cache.get_decoupled_flat_kv_and_frames(
                current_start=current_start, grid_sizes=grid_sizes, freqs=freqs)

    def test_fused_refresh_matches_python(self, device):
        """Fused kernel output must be bit-exact with Python refresh."""
        torch.manual_seed(42)
        cache_fused, freqs, grid_sizes, block_len, num_heads, head_dim = self._build_cache(device)
        cache_python, _, _, _, _, _ = self._build_cache(device)

        torch.manual_seed(42)
        self._warm_to_steady(cache_fused, freqs, grid_sizes, block_len, num_heads, head_dim, device)
        torch.manual_seed(42)
        self._warm_to_steady(cache_python, freqs, grid_sizes, block_len, num_heads, head_dim, device)

        assert cache_fused._steady_state_reached
        assert cache_python._steady_state_reached

        # Trigger noisy overwrite on same block → refresh-dirty path
        last_start = 7 * block_len
        torch.manual_seed(100)
        k_noise = torch.randn(1, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v_noise = torch.randn_like(k_noise)

        # Python path
        os.environ.pop("PYRAMIDKV_FUSED_REFRESH", None)
        cache_python.update(k_noise.clone(), v_noise.clone(),
                            current_start=last_start, grid_sizes=grid_sizes,
                            freqs=freqs, cache_update_mode="noisy")
        out_py = cache_python.get_decoupled_flat_kv_and_frames(
            current_start=last_start, grid_sizes=grid_sizes, freqs=freqs)

        # Fused path
        os.environ["PYRAMIDKV_FUSED_REFRESH"] = "1"
        try:
            cache_fused.update(k_noise.clone(), v_noise.clone(),
                               current_start=last_start, grid_sizes=grid_sizes,
                               freqs=freqs, cache_update_mode="noisy")
            out_fu = cache_fused.get_decoupled_flat_kv_and_frames(
                current_start=last_start, grid_sizes=grid_sizes, freqs=freqs)
        finally:
            os.environ.pop("PYRAMIDKV_FUSED_REFRESH", None)

        k_py, v_py, cu_py, maxlen_py, fids_py = out_py
        k_fu, v_fu, cu_fu, maxlen_fu, fids_fu = out_fu

        assert torch.equal(cu_py, cu_fu), "cu_seqlens differ"
        assert maxlen_py == maxlen_fu, "max_seqlen differs"
        assert torch.equal(k_py, k_fu), \
            f"K content diff. Max abs: {(k_py.float() - k_fu.float()).abs().max().item()}"
        assert torch.equal(v_py, v_fu), "V content diff"
        assert torch.equal(fids_py, fids_fu), "frame_ids diff"

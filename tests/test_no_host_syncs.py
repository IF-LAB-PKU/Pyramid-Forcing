"""G1-test: Detect implicit host-device synchronizations in the hot path.

Uses torch.cuda.set_sync_debug_mode('error') to make any implicit sync
(e.g. .item(), int(tensor), torch.any(gpu_tensor)) raise RuntimeError.

Expected: FAIL on current code (multiple .item() calls in update/readout).
After G1-impl: PASS.
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


def _make_cache_and_inputs(device, num_blocks=3):
    """Build a minimal AdaptiveKVCache that exercises the hot path."""
    from pyramidkv.config import PyramidKVConfig
    from pyramidkv.adaptive_cache import AdaptiveKVCache
    from wan.modules.model import rope_params
    import tempfile, os

    num_layers = 1
    num_heads = 4
    head_dim = 128
    batch_size = 1
    frame_seqlen = 16  # small for speed
    num_frame_per_block = 3
    block_len = num_frame_per_block * frame_seqlen

    # Build minimal config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for h in range(num_heads):
            label = -1 if h % 3 == 0 else (1 if h % 3 == 1 else 2)
            f.write(f"0,{h},{label}\n")
        config_path = f.name

    config = PyramidKVConfig(config_path, num_layers=num_layers, num_heads=num_heads)
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

    blocks_k = []
    blocks_v = []
    for blk in range(num_blocks):
        k = torch.randn(batch_size, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        blocks_k.append(k)
        blocks_v.append(v)

    return cache, freqs, grid_sizes, blocks_k, blocks_v, block_len


class TestNoHostSyncs:
    """Assert zero implicit host-device syncs in the update+readout hot path."""

    def test_update_readout_no_sync(self, device):
        """Run update + get_decoupled_flat_kv_and_frames under sync_debug_mode('error').

        Any .item() / int(tensor) / torch.any(gpu_tensor) will raise.
        """
        cache, freqs, grid_sizes, blocks_k, blocks_v, block_len = _make_cache_and_inputs(device)

        # Warm up: first block without sync detection (init paths may legitimately sync)
        cache.update(
            blocks_k[0], blocks_v[0],
            current_start=0,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        cache.get_decoupled_flat_kv_and_frames(
            current_start=0, grid_sizes=grid_sizes, freqs=freqs,
        )

        # Now enable strict sync detection for subsequent blocks
        torch.cuda.set_sync_debug_mode("error")
        try:
            for blk_idx in range(1, len(blocks_k)):
                current_start = blk_idx * block_len
                cache.update(
                    blocks_k[blk_idx], blocks_v[blk_idx],
                    current_start=current_start,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
                cache.get_decoupled_flat_kv_and_frames(
                    current_start=current_start,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_update_readout_no_sync_realistic_config(self, device):
        """Same test but with IVC + semantic selection + sink_grid_decoupling.

        This is the config actually used by pyramid-forcing inference.
        Catches deep .item() calls in _capture_sink_if_needed,
        _update_cyclic_anchors, etc. that the minimal config misses.
        """
        from pyramidkv.config import PyramidKVConfig
        from pyramidkv.adaptive_cache import AdaptiveKVCache
        from wan.modules.model import rope_params
        import tempfile

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

        config = PyramidKVConfig(config_path, num_layers=num_layers, num_heads=num_heads)
        config.frame_seq_length = frame_seqlen
        os.unlink(config_path)

        cache = AdaptiveKVCache(
            config=config,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=0,
            sink_len=1 * frame_seqlen,
            tail_len=4 * frame_seqlen,
            ivc_ratio=0.3,
            semantic_ratio=0.3,
            update_interval=1,
            sink_grid_decoupling=True,
        )

        freqs = torch.cat([
            rope_params(512, head_dim - 4 * (head_dim // 6)),
            rope_params(512, 2 * (head_dim // 6)),
            rope_params(512, 2 * (head_dim // 6)),
        ], dim=1).to(device)
        grid_sizes = torch.tensor([[num_frame_per_block, 4, 4]], dtype=torch.long, device=device)

        # Prime prompt_v for semantic selector
        prompt_v = torch.randn(num_heads, head_dim, device=device, dtype=torch.bfloat16)
        cache.set_prompt_values(prompt_v)

        # Block 0 warmup (init paths sync legitimately)
        k0 = torch.randn(batch_size, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v0 = torch.randn_like(k0)
        cache.update(k0, v0, current_start=0, grid_sizes=grid_sizes, freqs=freqs,
                     cache_update_mode="clean")
        cache.get_decoupled_flat_kv_and_frames(
            current_start=0, grid_sizes=grid_sizes, freqs=freqs,
        )

        # Sync detection enabled for subsequent blocks
        torch.cuda.set_sync_debug_mode("error")
        try:
            for blk_idx in range(1, 3):
                current_start = blk_idx * block_len
                k = torch.randn(batch_size, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
                v = torch.randn_like(k)
                # Test both noisy and clean passes
                cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                             freqs=freqs, cache_update_mode="noisy")
                cache.get_decoupled_flat_kv_and_frames(
                    current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
                )
                cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                             freqs=freqs, cache_update_mode="clean")
                cache.get_decoupled_flat_kv_and_frames(
                    current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
                )
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_update_readout_no_sync_pyramid_forcing(self, device):
        """Sync detection on pyramid-forcing-style compositions
        (cyclic + stride + merge all enabled, 12 heads, frame_seqlen=1560).

        Mirrors the production hot path under ``configs/pyramid-forcing.yaml``:
        - osc heads (-1): sink1 + cyclic(period=6, cap=3) + recent4
        - sta+ heads (1): sink3 + stride(interval=6, cap=4) + recent4
        - sta- heads (2): sink3 + merge(patch=2, cap=6) + recent4

        Catches per-block syncs in composition.update_all() / merge.update() /
        do_clean_anchors path that the smaller realistic_config test misses.
        """
        from pyramidkv.config import PyramidKVConfig
        from pyramidkv.adaptive_cache import AdaptiveKVCache
        from pyramidkv.factory import build_compositions
        from wan.modules.model import rope_params
        import tempfile

        num_layers = 1
        num_heads = 12
        head_dim = 128
        batch_size = 1
        frame_seqlen = 1560  # production frame_seqlen (240*416/4)
        num_frame_per_block = 3
        block_len = num_frame_per_block * frame_seqlen
        capacity = 32760

        # Round-robin labels: -1, 1, 2 (4 heads each)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Wide format: 1 row per layer, num_heads columns per row
            for _ in range(num_layers):
                row = ",".join(str([-1, 1, 2][h % 3]) for h in range(num_heads))
                f.write(row + "\n")
            config_path = f.name

        config = PyramidKVConfig(config_path, num_layers=num_layers, num_heads=num_heads)
        config.frame_seq_length = frame_seqlen
        os.unlink(config_path)

        # Override compositions with pyramid-forcing.yaml-aligned settings.
        capacity_tensor = torch.full((num_layers, num_heads), capacity, dtype=torch.int32)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            row = ",".join(str([-1, 1, 2][h % 3]) for h in range(num_heads))
            f.write(row + "\n")
            csv_path2 = f.name
        config.compositions = build_compositions(
            num_layers=num_layers,
            num_heads=num_heads,
            capacities=capacity_tensor,
            csv_path=csv_path2,
            cyclic_enabled=True, cyclic_period=6, cyclic_bucket_cap=3,
            cyclic_dynamic_rope=True, cyclic_osc_only=True,
            stride_enabled=True, stride_interval=6, stride_capacity=4, stride_dynamic_rope=True,
            merge_enabled=True, merge_patch_size=2, merge_capacity=6, merge_dynamic_rope=True,
            recent_frames=4, stable_recent_frames=4,
            osc_sink_frames=1, stable_sink_frames=3,
            label_sink_frames_map={"-1": 1, "1": 3, "2": 3},
            label_recent_frames_map={"-1": 4, "1": 4, "2": 4},
            label_phase_bucket_map={"-1": 3, "1": 0, "2": 0},
            label_stride_enabled_map={"1": True, "2": False},
            label_stride_interval_map={"1": 6},
            label_merge_enabled_map={"-1": False, "1": False, "2": True},
            label_merge_patch_size_map={"2": 2},
            label_merge_capacity_map={"2": 6},
        )
        os.unlink(csv_path2)

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
            sink_grid_decoupling=True,
            use_osc_frame_mode=True,
            phase_period=6,
            phase_bucket_capacity_frames=3,
            local_tail_frames=4,
            label_phase_bucket_map={"-1": 3, "1": 0, "2": 0},
            label_stride_enabled_map={"1": True, "2": False},
            label_sink_frames_map={"-1": 1, "1": 3, "2": 3},
            label_recent_frames_map={"-1": 4, "1": 4, "2": 4},
        )

        freqs = torch.cat([
            rope_params(2048, head_dim - 4 * (head_dim // 6)),
            rope_params(2048, 2 * (head_dim // 6)),
            rope_params(2048, 2 * (head_dim // 6)),
        ], dim=1).to(device)
        # 240/2=120 along H, 416/2=208 along W → 1560 tokens per frame * 3 frames per block.
        grid_sizes = torch.tensor([[num_frame_per_block, 120, 13]], dtype=torch.long, device=device)
        # 120*13 = 1560 ✓

        # Block 0 warmup with both noisy and clean to seed all anchor stores
        for blk_idx in range(2):
            current_start = blk_idx * block_len
            k = torch.randn(batch_size, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                         freqs=freqs, cache_update_mode="noisy")
            cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                         freqs=freqs, cache_update_mode="clean")
            cache.get_decoupled_flat_kv_and_frames(
                current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
            )

        # Sync detection enabled for blocks 2..5 (covers 6-frame periodicity ≥ 1 cycle)
        torch.cuda.set_sync_debug_mode("error")
        try:
            for blk_idx in range(2, 6):
                current_start = blk_idx * block_len
                k = torch.randn(batch_size, block_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
                v = torch.randn_like(k)
                cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                             freqs=freqs, cache_update_mode="noisy")
                cache.get_decoupled_flat_kv_and_frames(
                    current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
                )
                cache.update(k, v, current_start=current_start, grid_sizes=grid_sizes,
                             freqs=freqs, cache_update_mode="clean")
                cache.get_decoupled_flat_kv_and_frames(
                    current_start=current_start, grid_sizes=grid_sizes, freqs=freqs,
                )
        finally:
            torch.cuda.set_sync_debug_mode("default")

import os
import tempfile

import pytest
import torch

from headkv.adaptive_cache import AdaptiveKVCache
from headkv.base import HeadComposition
from headkv.config import HeadKVConfig
from headkv.cyclic import CyclicStrategy
from headkv.lag import LagStrategy
from headkv.stride import StrideStrategy
from wan.modules.model import rope_params


def _build_config(num_heads: int, capacities: list[int]) -> HeadKVConfig:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        for head_idx, cap in enumerate(capacities):
            f.write(f"0,{head_idx},{cap}\n")
        path = f.name
    config = HeadKVConfig(path, num_layers=1, num_heads=num_heads)
    os.unlink(path)
    return config


def _build_rope_freqs(head_dim: int, max_seq_len: int = 64) -> torch.Tensor:
    return torch.cat(
        [
            rope_params(max_seq_len, head_dim - 4 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def _make_tokens(start: int, length: int, num_heads: int, head_dim: int) -> torch.Tensor:
    vals = torch.arange(start, start + length, dtype=torch.float32)
    return vals.view(1, length, 1, 1).expand(1, length, num_heads, head_dim)


def _build_cache() -> tuple[AdaptiveKVCache, torch.Tensor, torch.Tensor]:
    num_heads = 2
    head_dim = 4
    config = _build_config(num_heads=num_heads, capacities=[16, 16])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_idx=0,
        sink_len=2,
        tail_len=16,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        update_interval=100,
    )
    freqs = _build_rope_freqs(head_dim)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)
    return cache, freqs, grid_sizes


def _build_composed_cache(
    *,
    device: torch.device | str = "cpu",
    capture_frame_id_mode: str = "mapped",
) -> tuple[AdaptiveKVCache, torch.Tensor]:
    num_heads = 1
    head_dim = 4
    config = _build_config(num_heads=num_heads, capacities=[16])
    config.compositions = [[
        HeadComposition(
            name="L0_H0_osc",
            label=-1,
            sink_frames=1,
            recent_frames=2,
            middle_strategies=[
                CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
                LagStrategy(offsets=[2], history_frames=8, dynamic_rope=False),
                StrideStrategy(interval=3, capacity=4, dynamic_rope=True),
            ],
        )
    ]]
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_idx=0,
        sink_len=0,
        tail_len=16,
        sink_grid_decoupling=True,
        use_osc_frame_mode=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        update_interval=100,
        capture_frame_id_mode=capture_frame_id_mode,
    )
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    return cache, grid_sizes


def _build_cpp_refresh_cache(
    *,
    device: torch.device | str = "cpu",
    capture_frame_id_mode: str = "mapped",
) -> tuple[AdaptiveKVCache, torch.Tensor]:
    num_heads = 1
    head_dim = 4
    config = _build_config(num_heads=num_heads, capacities=[16])
    config.compositions = [[
        HeadComposition(
            name="L0_H0_osc",
            label=-1,
            sink_frames=1,
            recent_frames=2,
            middle_strategies=[CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True)],
        )
    ]]
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_idx=0,
        sink_len=0,
        tail_len=16,
        sink_grid_decoupling=True,
        use_osc_frame_mode=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        update_interval=100,
        capture_frame_id_mode=capture_frame_id_mode,
    )
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    return cache, grid_sizes


def test_readout_layout_profile_counts_cold_reuse_and_refresh():
    cache, freqs, grid_sizes = _build_cache()
    cache.set_profile_enabled(True)

    k0 = _make_tokens(0, 4, num_heads=2, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    stats = cache.pop_profile_stats()
    assert stats["cold_pack_count"] == 1.0
    assert stats["refresh_pack_count"] == 0.0
    assert stats["layout_reuse_count"] == 0.0
    assert stats["anchor_store_update_count"] == 0.0
    assert stats["anchor_store_collect_count"] == 0.0
    assert stats["anchor_store_fallback_count"] == 0.0
    assert stats["anchor_store_anchor_count"] == 0.0
    assert stats["anchor_store_token_count"] == 0.0
    assert stats["readout_total_len"] > 0.0
    assert stats["readout_max_seqlen"] > 0.0

    cache.set_profile_enabled(True)
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    stats = cache.pop_profile_stats()
    assert stats["cold_pack_count"] == 0.0
    assert stats["layout_reuse_count"] == 1.0

    cache.set_profile_enabled(True)
    k1 = _make_tokens(100, 4, num_heads=2, head_dim=4)
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="noisy")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    stats = cache.pop_profile_stats()
    assert stats["cold_pack_count"] == 0.0
    assert stats["refresh_pack_count"] == 1.0


def test_contiguous_anchor_store_cpu_opt_in_falls_back_and_counts(monkeypatch):
    monkeypatch.setenv("HEADKV_CONTIG_ANCHOR_STORE", "1")
    cache, grid_sizes = _build_composed_cache()
    cache.set_profile_enabled(True)

    k0 = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, cache_update_mode="clean")
    stats = cache.pop_profile_stats()
    assert stats["anchor_store_update_count"] == 0.0
    assert stats["anchor_store_collect_count"] == 0.0
    assert stats["anchor_store_anchor_count"] == 0.0
    assert stats["anchor_store_token_count"] == 0.0
    assert stats["anchor_store_fallback_count"] > 0.0


def test_cpp_strategy_cpu_opt_in_falls_back_and_counts(monkeypatch):
    monkeypatch.setenv("HEADKV_CPP_STRATEGY", "1")
    cache, grid_sizes = _build_composed_cache()
    cache.set_profile_enabled(True)

    k0 = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, cache_update_mode="clean")
    stats = cache.pop_profile_stats()
    assert stats["cpp_strategy_update_count"] == 0.0
    assert stats["cpp_strategy_collect_count"] == 0.0
    assert stats["cpp_strategy_anchor_count"] == 0.0
    assert stats["cpp_strategy_token_count"] == 0.0
    assert stats["cpp_strategy_fallback_count"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpp_strategy_cuda_refresh_readout_counts_refresh_and_materialize(monkeypatch):
    device = torch.device("cuda")
    freqs = _build_rope_freqs(head_dim=4).to(device)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    inputs = [_make_tokens(t * 100, 4, num_heads=1, head_dim=4).to(device) for t in range(10)]

    monkeypatch.setenv("HEADKV_CPP_STRATEGY", "1")
    monkeypatch.setenv("HEADKV_CUDA_REFRESH", "1")
    cache, _ = _build_cpp_refresh_cache(device=device)
    cache.set_profile_enabled(True)

    for frame_idx, k in enumerate(inputs[:8]):
        cache.update(
            k,
            k.clone(),
            current_start=frame_idx * 4,
            grid_sizes=grid_sizes,
            cache_update_mode="clean",
        )

    cache.update(
        inputs[8],
        inputs[8].clone(),
        current_start=32,
        grid_sizes=grid_sizes,
        cache_update_mode="clean",
    )
    cache.get_decoupled_flat_kv_and_frames(current_start=32, grid_sizes=grid_sizes, freqs=freqs)
    cache.pop_profile_stats()

    cache.update(
        inputs[9],
        inputs[9].clone(),
        current_start=36,
        grid_sizes=grid_sizes,
        cache_update_mode="clean",
    )
    cache.get_decoupled_flat_kv_and_frames(current_start=36, grid_sizes=grid_sizes, freqs=freqs)
    stats = cache.pop_profile_stats()

    assert stats["cold_pack_count"] == 0.0
    assert stats["refresh_pack_count"] == 1.0
    assert stats["cuda_refresh_count"] == 1.0
    assert stats["cpp_strategy_fallback_count"] == 0.0
    assert stats["cpp_strategy_materialize_count"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frame_mode", ["mapped", "physical"])
def test_contiguous_anchor_store_cuda_readout_matches_default(monkeypatch, frame_mode):
    device = torch.device("cuda")
    freqs = _build_rope_freqs(head_dim=4).to(device)

    monkeypatch.delenv("HEADKV_CONTIG_ANCHOR_STORE", raising=False)
    baseline, baseline_grid = _build_composed_cache(device=device, capture_frame_id_mode=frame_mode)
    inputs = [_make_tokens(t * 100, 4, num_heads=1, head_dim=4).to(device) for t in range(5)]
    for frame_idx, k in enumerate(inputs):
        baseline.update(
            k,
            k.clone(),
            current_start=frame_idx * 4,
            grid_sizes=baseline_grid,
            cache_update_mode="clean",
        )
    expected = baseline.get_decoupled_flat_kv_and_frames(
        current_start=16,
        grid_sizes=baseline_grid,
        freqs=freqs,
    )

    monkeypatch.setenv("HEADKV_CONTIG_ANCHOR_STORE", "1")
    opt_in, opt_in_grid = _build_composed_cache(device=device, capture_frame_id_mode=frame_mode)
    opt_in.set_profile_enabled(True)
    for frame_idx, k in enumerate(inputs):
        opt_in.update(
            k,
            k.clone(),
            current_start=frame_idx * 4,
            grid_sizes=opt_in_grid,
            cache_update_mode="clean",
        )
    actual = opt_in.get_decoupled_flat_kv_and_frames(
        current_start=16,
        grid_sizes=opt_in_grid,
        freqs=freqs,
    )
    stats = opt_in.pop_profile_stats()

    for got, want in zip(actual[:3], expected[:3]):
        assert torch.equal(got, want)
    assert actual[3] == expected[3]
    assert torch.equal(actual[4], expected[4])
    assert stats["anchor_store_fallback_count"] == 0.0
    assert stats["anchor_store_update_count"] > 0.0
    assert stats["anchor_store_anchor_count"] > 0.0
    assert stats["anchor_store_token_count"] > 0.0

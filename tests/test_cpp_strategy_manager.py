import os
import tempfile

import pytest
import torch

from headkv.adaptive_cache import AdaptiveKVCache
from headkv.base import HeadComposition
from headkv.config import HeadKVConfig
from headkv.cpp_strategy import CppStrategyManager, compile_cpp_strategy_policies
from headkv.cyclic import CyclicStrategy
from headkv.merge import MergeStrategy
from headkv.stride import StrideStrategy
from wan.modules.model import rope_params


def _frame_batch(t_values: list[int], *, frame_seqlen: int = 4, head_dim: int = 4):
    k_parts = []
    v_parts = []
    p_parts = []
    for t in t_values:
        base = float(t * 100)
        vals = torch.arange(
            base, base + frame_seqlen * head_dim, dtype=torch.float32
        ).view(frame_seqlen, head_dim)
        pos = torch.zeros(frame_seqlen, 3, dtype=torch.long)
        pos[:, 0] = int(t)
        pos[:, 1] = torch.arange(frame_seqlen, dtype=torch.long)
        k_parts.append(vals)
        v_parts.append(vals + 1000.0)
        p_parts.append(pos)
    return (
        torch.cat(k_parts, dim=0),
        torch.cat(v_parts, dim=0),
        torch.cat(p_parts, dim=0),
    )


def _cache_tokens(start: int, length: int, *, num_heads: int = 1, head_dim: int = 4):
    vals = torch.arange(start, start + length, dtype=torch.float32)
    return vals.view(1, length, 1, 1).expand(1, length, num_heads, head_dim)


def _manager_for(strategy, *, recent_frames: int = 2, head_dim: int = 4):
    comp = HeadComposition(
        name="L0_H0",
        label=1,
        sink_frames=1,
        recent_frames=recent_frames,
        middle_strategies=[strategy],
    )
    policies, supported = compile_cpp_strategy_policies([[comp][0]])
    assert supported == [True]
    manager = CppStrategyManager(
        policies,
        num_seq=1,
        num_heads=1,
        head_dim=head_dim,
        require_cuda=False,
        require_extension=False,
    )
    return comp, manager


def _update_both(comp, manager, t_values, *, frame_seqlen: int = 4, head_dim: int = 4):
    k, v, pos = _frame_batch(t_values, frame_seqlen=frame_seqlen, head_dim=head_dim)
    comp.update_all(0, k, v, pos, frame_seqlen, t_values[0])
    ok = manager.update_all(
        k_flat=k.unsqueeze(0),
        v_flat=v.unsqueeze(0),
        pos_flat=pos.unsqueeze(0),
        frame_seqlen=frame_seqlen,
        current_t=t_values[0],
    )
    assert ok


def _assert_same_anchors(expected, actual):
    assert [a.t for a in actual] == [a.t for a in expected]
    assert [a.dynamic_rope for a in actual] == [a.dynamic_rope for a in expected]
    assert [a.token_count for a in actual] == [a.token_count for a in expected]
    for got, want in zip(actual, expected):
        assert torch.equal(got.k, want.k)
        assert torch.equal(got.v, want.v)
        assert torch.equal(got.pos, want.pos)


def test_cpp_strategy_manager_inactive_count_returns_empty_anchor_count():
    _comp, manager = _manager_for(
        CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True), recent_frames=2
    )

    count = manager.count_anchors(
        seq_idx=0,
        head_idx=0,
        current_t=0,
        recent_min_t=0,
        sink_max_t=-1,
    )
    stats = manager.pop_stats()

    assert count is not None
    assert count.token_count == 0
    assert count.anchor_count == 0
    assert count.anchor_lengths == ()
    assert count.dynamic_rope is True
    assert count.kind == manager.policies[0].kind
    assert stats["cpp_strategy_collect_count"] == 1.0
    assert stats["cpp_strategy_anchor_count"] == 0.0
    assert stats["cpp_strategy_token_count"] == 0.0


def _build_config_with_strategy(strategy) -> HeadKVConfig:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        f.write("0,0,32\n")
        path = f.name
    config = HeadKVConfig(path, num_layers=1, num_heads=1)
    os.unlink(path)
    config.compositions = [
        [
            HeadComposition(
                name="L0_H0",
                label=-1,
                sink_frames=1,
                recent_frames=2,
                middle_strategies=[strategy],
            )
        ]
    ]
    return config


def _build_cache_with_strategy(strategy, *, device: torch.device):
    return AdaptiveKVCache(
        config=_build_config_with_strategy(strategy),
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=0,
        tail_len=16,
        sink_grid_decoupling=True,
        use_osc_frame_mode=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        update_interval=100,
    )


def _rope_freqs(head_dim: int, *, device: torch.device):
    return torch.cat(
        [
            rope_params(128, head_dim - 4 * (head_dim // 6)),
            rope_params(128, 2 * (head_dim // 6)),
            rope_params(128, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)


def test_cpp_strategy_manager_cyclic_matches_python_collect():
    comp, manager = _manager_for(
        CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True), recent_frames=2
    )
    comp.reset_all(1)
    _update_both(comp, manager, [0, 1, 2, 3, 4])

    expected = comp.collect_all(0, current_t=4, recent_min_t=3, sink_max_t=0)
    actual = manager.collect(
        seq_idx=0, head_idx=0, current_t=4, recent_min_t=3, sink_max_t=0
    )
    _assert_same_anchors(expected, actual)


def test_cpp_strategy_manager_stride_matches_python_capacity_and_sort():
    comp, manager = _manager_for(
        StrideStrategy(interval=2, capacity=2, dynamic_rope=False), recent_frames=1
    )
    comp.reset_all(1)
    _update_both(comp, manager, [0, 1, 2, 3, 4, 5, 6])

    expected = comp.collect_all(0, current_t=6, recent_min_t=6, sink_max_t=0)
    actual = manager.collect(
        seq_idx=0, head_idx=0, current_t=6, recent_min_t=6, sink_max_t=0
    )
    _assert_same_anchors(expected, actual)
    assert [a.t for a in actual] == sorted(a.t for a in actual)


def test_cpp_strategy_manager_merge_matches_python_completed_blocks():
    comp, manager = _manager_for(
        MergeStrategy(patch_size=2, capacity=1), recent_frames=1
    )
    comp.reset_all(1)

    def grid_frame_batch(t_values):
        k_parts = []
        v_parts = []
        p_parts = []
        for t in t_values:
            k = torch.arange(t * 100, t * 100 + 16, dtype=torch.float32).view(4, 4)
            v = k + 1000.0
            pos = torch.tensor(
                [[t, 0, 0], [t, 0, 1], [t, 1, 0], [t, 1, 1]],
                dtype=torch.long,
            )
            k_parts.append(k)
            v_parts.append(v)
            p_parts.append(pos)
        return torch.cat(k_parts), torch.cat(v_parts), torch.cat(p_parts)

    k, v, pos = grid_frame_batch([0, 1, 2, 3, 4, 5, 6, 7])
    comp.update_all(0, k, v, pos, 4, 0)
    assert manager.update_all(
        k_flat=k.unsqueeze(0),
        v_flat=v.unsqueeze(0),
        pos_flat=pos.unsqueeze(0),
        frame_seqlen=4,
        current_t=0,
    )

    expected = comp.collect_all(0, current_t=8, recent_min_t=8, sink_max_t=0)
    actual = manager.collect(
        seq_idx=0, head_idx=0, current_t=8, recent_min_t=8, sink_max_t=0
    )
    _assert_same_anchors(expected, actual)


def test_cpp_strategy_manager_merge_duplicate_frame_matches_python_error():
    comp, manager = _manager_for(
        MergeStrategy(patch_size=2, capacity=1), recent_frames=1
    )
    comp.reset_all(1)
    k, v, pos = _frame_batch([0], frame_seqlen=4, head_dim=4)
    comp.update_all(0, k, v, pos, 4, 0)
    assert manager.update_all(
        k_flat=k.unsqueeze(0),
        v_flat=v.unsqueeze(0),
        pos_flat=pos.unsqueeze(0),
        frame_seqlen=4,
        current_t=0,
    )

    with pytest.raises(ValueError):
        comp.update_all(0, k, v, pos, 4, 0)
    with pytest.raises(ValueError):
        manager.update_all(
            k_flat=k.unsqueeze(0),
            v_flat=v.unsqueeze(0),
            pos_flat=pos.unsqueeze(0),
            frame_seqlen=4,
            current_t=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpp_strategy_cuda_empty_readout_before_clean_update_does_not_fallback(
    monkeypatch,
):
    device = torch.device("cuda")
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    freqs = _rope_freqs(4, device=device)

    monkeypatch.setenv("HEADKV_CPP_STRATEGY", "1")
    cache = _build_cache_with_strategy(
        CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
        device=device,
    )
    cache.set_profile_enabled(True)

    def _collect_all_must_not_run(*_args, **_kwargs):
        raise AssertionError(
            "HEADKV_CPP_STRATEGY empty readout called Python collect_all"
        )

    monkeypatch.setattr(HeadComposition, "collect_all", _collect_all_must_not_run)
    k_flat, v_flat, cu_seqlens, max_seqlen, frame_ids = (
        cache.get_decoupled_flat_kv_and_frames(
            current_start=0,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
    )
    stats = cache.pop_profile_stats()

    assert k_flat.numel() == 0
    assert v_flat.numel() == 0
    assert frame_ids.numel() == 0
    assert cu_seqlens.tolist() == [0, 0]
    assert max_seqlen == 0
    assert stats["cpp_strategy_collect_count"] == 1.0
    assert stats["cpp_strategy_anchor_count"] == 0.0
    assert stats["cpp_strategy_token_count"] == 0.0
    assert stats["cpp_strategy_materialize_count"] == 0.0
    assert stats["cpp_strategy_fallback_count"] == 0.0
    assert stats["cpp_strategy_fallback_inactive_count"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "baseline_strategy,cpp_strategy",
    [
        (
            CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
            CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
        ),
        (
            StrideStrategy(interval=2, capacity=3, dynamic_rope=True),
            StrideStrategy(interval=2, capacity=3, dynamic_rope=True),
        ),
        (
            MergeStrategy(patch_size=2, capacity=1),
            MergeStrategy(patch_size=2, capacity=1),
        ),
    ],
)
def test_cpp_strategy_cache_readout_matches_python_cuda(
    monkeypatch, baseline_strategy, cpp_strategy
):
    device = torch.device("cuda")
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    freqs = _rope_freqs(4, device=device)

    monkeypatch.delenv("HEADKV_CPP_STRATEGY", raising=False)
    baseline = _build_cache_with_strategy(baseline_strategy, device=device)
    inputs = [
        _cache_tokens(t * 100, 4, num_heads=1, head_dim=4).to(device) for t in range(8)
    ]
    for frame_idx, k in enumerate(inputs):
        baseline.update(
            k,
            k.clone(),
            current_start=frame_idx * 4,
            grid_sizes=grid_sizes,
            cache_update_mode="clean",
        )
    expected = baseline.get_decoupled_flat_kv_and_frames(
        current_start=32,
        grid_sizes=grid_sizes,
        freqs=freqs,
    )

    monkeypatch.setenv("HEADKV_CPP_STRATEGY", "1")
    opt_in = _build_cache_with_strategy(cpp_strategy, device=device)
    opt_in.set_profile_enabled(True)
    for frame_idx, k in enumerate(inputs):
        opt_in.update(
            k,
            k.clone(),
            current_start=frame_idx * 4,
            grid_sizes=grid_sizes,
            cache_update_mode="clean",
        )

    def _collect_all_must_not_run(*_args, **_kwargs):
        raise AssertionError("HEADKV_CPP_STRATEGY readout called Python collect_all")

    monkeypatch.setattr(HeadComposition, "collect_all", _collect_all_must_not_run)
    actual = opt_in.get_decoupled_flat_kv_and_frames(
        current_start=32,
        grid_sizes=grid_sizes,
        freqs=freqs,
    )
    stats = opt_in.pop_profile_stats()

    for got, want in zip(actual[:3], expected[:3]):
        assert torch.equal(got, want)
    assert actual[3] == expected[3]
    assert torch.equal(actual[4], expected[4])
    assert stats["cpp_strategy_update_count"] > 0.0
    assert stats["cpp_strategy_collect_count"] > 0.0
    if stats["cpp_strategy_token_count"] > 0.0:
        assert stats["cpp_strategy_materialize_count"] > 0.0
        assert stats["cpp_strategy_materialize_token_count"] > 0.0
    assert stats["cpp_strategy_fallback_count"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpp_strategy_cache_refresh_matches_python_cuda(monkeypatch):
    device = torch.device("cuda")
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long, device=device)
    freqs = _rope_freqs(4, device=device)
    inputs = [
        _cache_tokens(t * 100, 4, num_heads=1, head_dim=4).to(device) for t in range(10)
    ]

    def _run_two_block_readout(cache):
        out = None
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
        cache.get_decoupled_flat_kv_and_frames(
            current_start=32,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        cache.pop_profile_stats()
        cache.update(
            inputs[9],
            inputs[9].clone(),
            current_start=36,
            grid_sizes=grid_sizes,
            cache_update_mode="clean",
        )
        out = cache.get_decoupled_flat_kv_and_frames(
            current_start=36,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        stats = cache.pop_profile_stats()
        return out, stats

    monkeypatch.delenv("HEADKV_CPP_STRATEGY", raising=False)
    monkeypatch.delenv("HEADKV_CUDA_REFRESH", raising=False)
    baseline = _build_cache_with_strategy(
        CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
        device=device,
    )
    baseline.set_profile_enabled(True)
    expected, baseline_stats = _run_two_block_readout(baseline)

    monkeypatch.setenv("HEADKV_CPP_STRATEGY", "1")
    monkeypatch.setenv("HEADKV_CUDA_REFRESH", "1")
    opt_in = _build_cache_with_strategy(
        CyclicStrategy(period=2, bucket_cap=2, dynamic_rope=True),
        device=device,
    )
    opt_in.set_profile_enabled(True)
    actual, stats = _run_two_block_readout(opt_in)

    for got, want in zip(actual[:3], expected[:3]):
        assert torch.equal(got, want)
    assert actual[3] == expected[3]
    assert torch.equal(actual[4], expected[4])
    assert baseline_stats["cold_pack_count"] > 0.0
    assert stats["cold_pack_count"] == 0.0
    assert stats["refresh_pack_count"] > 0.0
    assert stats["cpp_strategy_materialize_count"] > 0.0
    assert stats["cpp_strategy_fallback_count"] == 0.0

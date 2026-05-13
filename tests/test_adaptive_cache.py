import os
import tempfile

import pytest
import torch

from pyramidkv.config import PyramidKVConfig
from pyramidkv import HeadComposition
from pyramidkv.merge import MergeStrategy
from wan.modules.model import rope_params
from pyramidkv.adaptive_cache import (
    AdaptiveKVCache,
    SemanticValueSelector,
    ThreeDIVCSelector,
)
from pyramidkv.rope import apply_rope_to_flat_k


def _build_config(num_layers, num_heads, capacities):
    entries = [(0, head_idx, cap) for head_idx, cap in enumerate(capacities)]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        for layer_idx, head_idx, cap in entries:
            f.write(f"{layer_idx},{head_idx},{cap}\n")
        path = f.name

    config = PyramidKVConfig(path, num_layers=num_layers, num_heads=num_heads)
    os.unlink(path)
    return config


def _set_layer_labels(config: PyramidKVConfig, labels):
    for head_idx, label in enumerate(labels):
        config.label_map[0, head_idx] = int(label)


def _build_rope_freqs(head_dim: int, max_seq_len: int = 128) -> torch.Tensor:
    return torch.cat(
        [
            rope_params(max_seq_len, head_dim - 4 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
            rope_params(max_seq_len, 2 * (head_dim // 6)),
        ],
        dim=1,
    )


def _make_tokens(start, length, num_heads, head_dim):
    vals = torch.arange(start, start + length, dtype=torch.float32)
    return vals.view(1, length, 1, 1).expand(1, length, num_heads, head_dim)


class _CountingAdaptiveKVCache(AdaptiveKVCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_cache_calls = 0

    def update_cache(self, *args, **kwargs):
        self.update_cache_calls += 1
        return super().update_cache(*args, **kwargs)


class _CollectCountingAdaptiveKVCache(AdaptiveKVCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.middle_collect_calls = 0

    def _collect_middle_cache(self, *args, **kwargs):
        self.middle_collect_calls += 1
        return super()._collect_middle_cache(*args, **kwargs)


def test_ivc_mask_topk_ratio():
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
        ],
        dtype=torch.long,
    )
    d_model = 128
    freqs = _build_rope_freqs(d_model)
    mask = ThreeDIVCSelector.get_ivc_mask(pos_3d=pos, d_model=d_model, freqs=freqs, ratio=0.25)
    assert mask.dtype == torch.bool
    assert mask.shape == (12,)
    assert int(mask.sum().item()) == 3


def test_semantic_mask_headwise_ratio():
    kv_v = torch.randn(2, 10, 4)
    prompt_v = torch.randn(2, 4)
    mask = SemanticValueSelector.get_semantic_mask(
        kv_v=kv_v,
        prompt_v=prompt_v,
        ratio=0.2,
        seed_ratio=0.1,
    )
    assert mask.shape == (2, 10)
    assert mask.dtype == torch.bool
    assert int(mask[0].sum().item()) == 2
    assert int(mask[1].sum().item()) == 2


def test_adaptive_cache_ragged_compaction_and_pos_ids():
    config = _build_config(num_layers=1, num_heads=2, capacities=[6, 10])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=2,
        ivc_ratio=0.3,
        semantic_ratio=0.3,
        update_interval=1,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[2, 2, 2]], dtype=torch.long)

    k1 = _make_tokens(0, 8, num_heads=2, head_dim=4)
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, prompt_v=torch.randn(2, 4))

    k2 = _make_tokens(8, 8, num_heads=2, head_dim=4)
    cache.update(k2, k2.clone(), current_start=8, grid_sizes=grid_sizes, freqs=freqs, prompt_v=torch.randn(2, 4))

    k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_ids = cache.get_flat_kv_and_pos()
    assert cu_seqlens_k.shape[0] == 3  # B * H + 1
    lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    assert lens[0] <= 6
    assert lens[1] <= 10
    assert max_seqlen_k == max(lens)
    assert k_flat.shape[0] == sum(lens)
    assert v_flat.shape[0] == sum(lens)
    assert pos_ids.shape == (sum(lens), 3)

    roped = cache.apply_rope_to_flat_k(k_flat, pos_ids, freqs=freqs)
    assert roped.shape == k_flat.shape


def test_dynamic_buffer_compaction_preserves_suffix_without_overlap_error():
    config = _build_config(num_layers=1, num_heads=1, capacities=[16])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=0,
        tail_len=16,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=1,
    )

    initial_k = torch.arange(24, dtype=torch.float32).view(6, 4)
    initial_v = initial_k + 100.0
    initial_pos = torch.stack(
        [
            torch.arange(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
        ],
        dim=1,
    )
    cache._set_dynamic_store(0, initial_k, initial_v, initial_pos, reserve_extra=4)
    cache._keep_dynamic_suffix(0, keep_len=4)

    append_k = torch.arange(20, dtype=torch.float32).view(5, 4) + 1000.0
    append_v = append_k + 100.0
    append_pos = torch.stack(
        [
            torch.arange(6, 11, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
            torch.zeros(5, dtype=torch.long),
        ],
        dim=1,
    )

    cache._append_dynamic(0, append_k, append_v, append_pos)

    expected_k = torch.cat([initial_k[-4:], append_k], dim=0)
    expected_v = torch.cat([initial_v[-4:], append_v], dim=0)
    expected_pos = torch.cat([initial_pos[-4:], append_pos], dim=0)
    assert torch.equal(cache.dynamic_k[0], expected_k)
    assert torch.equal(cache.dynamic_v[0], expected_v)
    assert torch.equal(cache.dynamic_pos[0], expected_pos)


def test_sink_grid_decoupling_syncs_sink_time_to_current_step():
    config = _build_config(num_layers=1, num_heads=1, capacities=[8])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=4,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=1,
    )
    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[1, 1, 5]], dtype=torch.long)  # frame_seqlen=5

    k = _make_tokens(0, 5, num_heads=1, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    assert cache.static_k[0].shape[0] == 2
    assert cache.dynamic_k[0].shape[0] == 3

    k_flat_now, _, cu_now, _ = cache.get_decoupled_flat_kv(
        current_start=10,  # sync_t=2
        grid_sizes=grid_sizes,
        freqs=freqs,
    )
    assert cu_now.tolist() == [0, 5]
    k_flat_now_sink = k_flat_now[:2].clone()  # clone before next call (workspace reuse)

    k_flat_later, _, cu_later, _ = cache.get_decoupled_flat_kv(
        current_start=15,  # sync_t=3
        grid_sizes=grid_sizes,
        freqs=freqs,
    )
    assert cu_later.tolist() == [0, 5]
    # sink segment should change with sync_t
    assert not torch.allclose(k_flat_now_sink, k_flat_later[:2])


def test_sink_grid_decoupling_does_not_create_second_sink_in_dynamic():
    config = _build_config(num_layers=1, num_heads=1, capacities=[6])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=2,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k1 = _make_tokens(0, 4, num_heads=1, head_dim=4)  # static=[0,1], dynamic=[2,3]
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs)

    k2 = _make_tokens(4, 4, num_heads=1, head_dim=4)  # merged dynamic=[2,3,4,5,6,7], dyn_cap=4
    cache.update(k2, k2.clone(), current_start=4, grid_sizes=grid_sizes, freqs=freqs)

    dyn = cache.dynamic_k[0][:, 0]
    # Without a second sink in dynamic, compaction should keep the most recent 4 tokens.
    assert dyn.shape[0] == 4
    assert int(dyn.min().item()) >= 4
    assert set(dyn.tolist()) == {4.0, 5.0, 6.0, 7.0}


def test_cache_update_mode_noisy_skips_reselection():
    config = _build_config(num_layers=1, num_heads=1, capacities=[20])
    cache = _CountingAdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=1,
        tail_len=1,
        ivc_ratio=0.5,
        semantic_ratio=0.5,
        update_interval=1,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="noisy")
    assert cache.update_cache_calls == 0

    cache.update(k, k.clone(), current_start=4, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    assert cache.update_cache_calls > 0


def test_sink_decoupling_applies_only_to_reduced_capacity_heads():
    # head0 small capacity (oscillating), head1 large capacity (stable)
    config = _build_config(num_layers=1, num_heads=2, capacities=[4, 8])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=2,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k = _make_tokens(0, 4, num_heads=2, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs)

    # head0 decoupled -> static sink captured
    assert cache.static_k[0] is not None
    assert cache.static_k[0].shape[0] == 2
    # head1 stable -> no decoupled static sink capture
    assert cache.static_k[1] is None


def test_decoupled_sink_tokens_caps_static_bucket():
    config = _build_config(num_layers=1, num_heads=1, capacities=[16])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=8,
        tail_len=4,
        sink_grid_decoupling=True,
        decoupled_sink_tokens=3,
        ivc_ratio=0.5,
        semantic_ratio=0.5,
        update_interval=1,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 8]], dtype=torch.long)

    k = _make_tokens(0, 8, num_heads=1, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, prompt_v=torch.randn(1, 4))
    assert cache.static_k[0] is not None
    assert cache.static_k[0].shape[0] == 3


def test_sink_decoupling_time_lag_and_spatial_lock():
    config = _build_config(num_layers=1, num_heads=1, capacities=[8])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=4,
        tail_len=4,
        sink_grid_decoupling=True,
        decoupled_sink_time_lag=2,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)  # frame_seqlen=4

    k = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    assert cache.static_pos[0] is not None
    stat_pos = cache.static_pos[0].clone()

    # current_start=20 -> t_current=5, lag=2 => sync_t=3
    k_flat, _, _, _ = cache.get_decoupled_flat_kv(
        current_start=20,
        grid_sizes=grid_sizes,
        freqs=freqs,
    )
    expected_pos = stat_pos.clone()
    expected_pos[:, 0] = 3
    expected = cache.apply_rope_to_flat_k(cache.static_k[0], expected_pos, freqs=freqs)
    assert torch.allclose(k_flat[: expected.shape[0]], expected)


def test_trajectory_scores_capture_temporal_change_on_same_spatial_track():
    # Tokens 0/2 are same (y=0,x=0) across t=0,1 with large change.
    # Tokens 1/3 are same (y=0,x=1) across t=0,1 with tiny change.
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=torch.long,
    )
    v = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],   # large motion on (0,0) trajectory
            [0.1, 0.0, 0.0, 0.0],    # tiny motion on (0,1) trajectory
        ],
        dtype=torch.float32,
    )
    scores = AdaptiveKVCache.get_trajectory_scores(pos_seg=pos, v_seg=v)
    assert scores.shape == (4,)
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]


def test_history_frame_quota_preserves_temporal_coverage():
    config = _build_config(num_layers=1, num_heads=1, capacities=[100])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=0,
        tail_len=0,  # push all tokens into candidate/history branch
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        history_frame_quota=1,
        update_interval=1,
    )

    # 3 time frames, 2 tokens per frame -> total 6
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [2, 0, 0],
            [2, 0, 1],
        ],
        dtype=torch.long,
    )
    k = torch.randn(6, 4)
    v = torch.randn(6, 4)
    k_out, v_out, p_out = cache.update_cache(
        k_seq=k,
        v_seq=v,
        pos_seq=pos,
        budget=3,
        freqs=None,
        prompt_head=None,
        apply_selection=False,
        sink_len=0,
    )
    assert k_out.shape[0] == 3
    assert v_out.shape[0] == 3
    # One token retained for each time frame.
    assert set(p_out[:, 0].tolist()) == {0, 1, 2}


def test_osc_frame_mode_keeps_first_sink_and_local_tail_4_frames():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,  # 1 frame
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[6, 1, 2]], dtype=torch.long)  # 6 frames, 2 tokens/frame
    k = _make_tokens(0, 12, num_heads=1, head_dim=4)

    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    assert cache.static_k[0] is not None
    assert cache.static_k[0].shape[0] == 2
    assert cache.dynamic_k[0] is not None
    assert cache.dynamic_k[0].shape[0] == 8  # 4 local frames * 2 tokens/frame
    assert cache.tail_len == 8
    # Dynamic region should keep the most recent 4 frames.
    assert set(cache.dynamic_pos[0][:, 0].tolist()) == {2, 3, 4, 5}


def test_osc_frame_mode_phase_bucket_overwrites_by_mod6():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=256)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)  # 1 frame/update, 2 tokens/frame

    for t in range(13):
        k = _make_tokens(t * 2, 2, num_heads=1, head_dim=4)
        cache.update(k, k.clone(), current_start=t * 2, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    for phase in range(6):
        bucket = cache.cyclic_buckets[0][phase]
        assert len(bucket) == 1
        _, _, _, t_val = bucket[0]
        expected_t = max(t for t in range(13) if t % 6 == phase)
        assert t_val == expected_t


def test_osc_frame_mode_phase_bucket_skips_noisy_updates():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)
    k = _make_tokens(0, 2, num_heads=1, head_dim=4)

    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="noisy")
    assert all(len(bucket) == 0 for bucket in cache.cyclic_buckets[0])

    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    assert any(len(bucket) > 0 for bucket in cache.cyclic_buckets[0])


def test_osc_frame_mode_phase_bucket_only_for_osc_heads():
    config = _build_config(num_layers=1, num_heads=2, capacities=[32, 32])
    _set_layer_labels(config, [-1, 1])  # head0 oscillating, head1 stable
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[2, 1, 2]], dtype=torch.long)
    k = _make_tokens(0, 4, num_heads=2, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    assert any(len(bucket) > 0 for bucket in cache.cyclic_buckets[0])
    assert all(len(bucket) == 0 for bucket in cache.cyclic_buckets[1])


def test_osc_frame_mode_decoupled_reads_current_phase_anchor_only():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=256)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)

    for t in range(8):
        k = _make_tokens(t * 2, 2, num_heads=1, head_dim=4)
        cache.update(k, k.clone(), current_start=t * 2, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    # Query at t=12 (phase 0): include current-phase anchor (from t=6), but not all phases.
    k_flat, _, cu, _ = cache.get_decoupled_flat_kv(current_start=24, grid_sizes=grid_sizes, freqs=freqs)
    assert cu.tolist() == [0, 12]  # static(2) + local tail(8) + one phase anchor frame(2)

    bucket = cache.cyclic_buckets[0][0]
    assert len(bucket) == 1
    anchor_k, _, anchor_pos, anchor_t = bucket[0]
    assert anchor_t == 6
    expected_pos = anchor_pos.clone()
    expected_pos[:, 0] = cache._map_sink_time(12)
    expected_k = cache.apply_rope_to_flat_k(anchor_k, expected_pos, freqs=freqs)
    assert torch.allclose(k_flat[-2:], expected_k)


def test_osc_frame_mode_phase_anchor_dynamic_rope_time_sync_spatial_lock():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        decoupled_sink_time_lag=0,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=1,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
        phase_sink_dynamic_rope=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=256)
    grid_sizes = torch.tensor([[1, 2, 1]], dtype=torch.long)  # 1 frame/update, y has two positions

    # First frame initializes sink; second frame at t=6 populates phase-0 anchor.
    k0 = _make_tokens(0, 2, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    k6 = _make_tokens(12, 2, num_heads=1, head_dim=4)
    cache.update(k6, k6.clone(), current_start=12, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    k_flat, _, cu, _ = cache.get_decoupled_flat_kv(current_start=24, grid_sizes=grid_sizes, freqs=freqs)  # sync_t=12
    assert cu.tolist() == [0, 6]  # sink(2) + local(2) + phase anchor(2)

    bucket = cache.cyclic_buckets[0][0]
    assert len(bucket) == 1
    anchor_k, _, anchor_pos, _ = bucket[0]
    expected_pos = anchor_pos.clone()
    expected_pos[:, 0] = 12
    # spatial lock: y/x should stay unchanged
    assert torch.equal(expected_pos[:, 1:], anchor_pos[:, 1:])
    expected_k = cache.apply_rope_to_flat_k(anchor_k, expected_pos, freqs=freqs)
    assert torch.allclose(k_flat[-2:], expected_k)


def test_osc_lag_mode_uses_relative_t_minus_6_and_compresses_to_6_frames():
    config = _build_config(num_layers=1, num_heads=1, capacities=[64])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,  # 1 frame sink
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=0,  # disable mod-phase bucket
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
        use_osc_lag_mode=True,
        osc_lag_offsets_frames=[6],
        osc_lag_history_frames=21,
        osc_lag_dynamic_rope=False,
    )
    freqs = _build_rope_freqs(4, max_seq_len=256)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)  # 1 frame/update, 2 tokens/frame

    for t in range(8):
        k = _make_tokens(t * 2, 2, num_heads=1, head_dim=4)
        cache.update(k, k.clone(), current_start=t * 2, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    # Query at t=8 -> local tail keeps t=4..7; lag branch should add t-6=2.
    k_flat, _, cu, _ = cache.get_decoupled_flat_kv(current_start=16, grid_sizes=grid_sizes, freqs=freqs)
    assert cu.tolist() == [0, 12]  # sink(2) + local4(8) + lag1(2) = 6 frames total
    assert all(len(bucket) == 0 for bucket in cache.cyclic_buckets[0])

    anchor = cache._find_anchor_by_t(cache.lag_anchor_frames[0], 2)
    assert anchor is not None
    anchor_k, _, anchor_pos, _ = anchor
    expected_k = cache.apply_rope_to_flat_k(anchor_k, anchor_pos, freqs=freqs)
    assert torch.allclose(k_flat[-2:], expected_k)


def test_osc_lag_mode_dynamic_rope_uses_mapped_t_minus_6():
    config = _build_config(num_layers=1, num_heads=1, capacities=[96])
    _set_layer_labels(config, [-1])
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=96,
        sink_grid_decoupling=True,
        sink_time_mapping_mode="window_clamp",
        sink_time_clamp_min=18,
        sink_time_clamp_max=21,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_period=6,
        phase_bucket_capacity_frames=0,
        local_tail_frames=4,
        phase_sink_for_osc_only=True,
        use_osc_lag_mode=True,
        osc_lag_offsets_frames=[6],
        osc_lag_history_frames=21,
        osc_lag_dynamic_rope=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=512)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)

    for t in range(30):
        k = _make_tokens(t * 2, 2, num_heads=1, head_dim=4)
        cache.update(k, k.clone(), current_start=t * 2, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    # Query at raw t=30: sink mapped time is 9 (window_clamp with max=21), lag frame is raw t=24.
    k_flat, _, cu, _ = cache.get_decoupled_flat_kv(current_start=60, grid_sizes=grid_sizes, freqs=freqs)
    assert cu.tolist() == [0, 12]

    anchor = cache._find_anchor_by_t(cache.lag_anchor_frames[0], 24)
    assert anchor is not None
    anchor_k, _, anchor_pos, _ = anchor
    expected_pos = anchor_pos.clone()
    expected_pos[:, 0] = 3  # mapped lag time = sync_t(9) - 6
    expected_k = cache.apply_rope_to_flat_k(anchor_k, expected_pos, freqs=freqs)
    assert torch.allclose(k_flat[-2:], expected_k)


def test_disable_first_sink_for_osc_heads_only():
    config = _build_config(num_layers=1, num_heads=2, capacities=[64, 64])
    _set_layer_labels(config, [-1, 1])  # head0 oscillating, head1 stable
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=64,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        phase_bucket_capacity_frames=0,
        local_tail_frames=4,
        disable_first_sink_for_osc_heads=True,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 2]], dtype=torch.long)
    k = _make_tokens(0, 2, num_heads=2, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    # Oscillating head does not keep first-frame sink.
    assert cache.static_k[0] is None
    # Stable head still keeps first-frame sink.
    assert cache.static_k[1] is not None
    assert cache.static_k[1].shape[0] == 2


def test_stable_head_policy_can_be_disabled():
    config = _build_config(num_layers=1, num_heads=1, capacities=[128])
    _set_layer_labels(config, [1])  # stable
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=0,
        tail_len=128,
        sink_grid_decoupling=False,
        use_osc_frame_mode=True,
        local_tail_frames=10,
        use_stable_head_policies=False,
    )
    cache.compositions_row = [HeadComposition(name="L0_H0_stride", label=1, sink_frames=0, recent_frames=4, policy_type="stride", capacity=128)]
    cache.policies_row = cache.compositions_row
    freqs = _build_rope_freqs(4, max_seq_len=256)
    grid_sizes = torch.tensor([[10, 1, 2]], dtype=torch.long)
    k = _make_tokens(0, 20, num_heads=1, head_dim=4)

    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    t_vals = cache.dynamic_pos[0][:, 0].tolist()
    assert cache.dynamic_k[0].shape[0] == 20
    assert sorted(set(t_vals)) == list(range(10))


def test_per_head_sink_frames_supports_sta_sink3_and_osc_sink1():
    config = _build_config(num_layers=1, num_heads=2, capacities=[64, 64])
    _set_layer_labels(config, [-1, 1])  # head0=osc, head1=stable
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        sink_len=6,  # fallback only
        tail_len=64,
        sink_grid_decoupling=True,
        use_osc_frame_mode=True,
        local_tail_frames=4,
        use_stable_head_policies=True,
        stable_sink_frames=3,
        osc_sink_frames=1,
    )
    cache.compositions_row = [
        HeadComposition(name="L0_H0_osc", label=-1, sink_frames=1, recent_frames=4, policy_type="osc", capacity=64),
        HeadComposition(name="L0_H1_stride", label=1, sink_frames=3, recent_frames=4, policy_type="stride", capacity=64),
    ]
    cache.policies_row = cache.compositions_row
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[3, 1, 2]], dtype=torch.long)  # 3 frames, 2 tokens/frame
    k = _make_tokens(0, 6, num_heads=2, head_dim=4)
    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")

    # oscillating head uses sink1 => 2 tokens
    assert cache.static_k[0] is not None
    assert cache.static_k[0].shape[0] == 2
    # stable head uses sink3 => 6 tokens
    assert cache.static_k[1] is not None
    assert cache.static_k[1].shape[0] == 6


def test_merge_strategy_uses_block_median_time_and_single_rope_pass():
    config = _build_config(num_layers=1, num_heads=1, capacities=[128])
    _set_layer_labels(config, [1])
    merge = MergeStrategy(patch_size=2, capacity=2, dynamic_rope=True)
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=0,
        tail_len=128,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=True,
        local_tail_frames=1,
    )
    cache.compositions_row = [
        HeadComposition(
            name="L0_H0_merge",
            label=1,
            sink_frames=0,
            recent_frames=1,
            middle_strategies=[merge],
            policy_type="merge",
            capacity=128,
        )
    ]
    cache.policies_row = cache.compositions_row
    cache.compositions_row[0].reset_all(num_seq=1)

    freqs = _build_rope_freqs(4, max_seq_len=128)
    grid_sizes = torch.tensor([[4, 2, 1]], dtype=torch.long)  # 4 frames, 2 tokens/frame
    k = _make_tokens(0, 8, num_heads=1, head_dim=4)

    cache.update(k, k.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    k_flat, _, cu, _ = cache.get_decoupled_flat_kv(current_start=8, grid_sizes=grid_sizes, freqs=freqs)
    assert cu.tolist() == [0, 3]  # recent frame(2) + one merged token

    collected = merge.collect(0, current_t=4, recent_min_t=4, sink_max_t=-1)
    assert len(collected) == 1
    anchor = collected[0]
    assert anchor.t == 1
    assert anchor.dynamic_rope is False
    assert anchor.pos.tolist() == [[1, 0, 0]]
    expected_k = cache.apply_rope_to_flat_k(anchor.k, anchor.pos, freqs=freqs)
    assert torch.allclose(k_flat[-1:], expected_k)


def test_decoupled_readout_reuses_workspace_for_same_block_noisy_overwrite():
    config = _build_config(num_layers=1, num_heads=1, capacities=[16])
    cache = _CollectCountingAdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=16,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k0 = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    first = cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    first_k = first[0].clone()
    first_collect_calls = cache.middle_collect_calls

    k1 = _make_tokens(100, 4, num_heads=1, head_dim=4)
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="noisy")
    second = cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)

    assert first_collect_calls == 1
    assert cache.middle_collect_calls == first_collect_calls
    assert not cache._readout_cache_tail_dirty
    assert not torch.allclose(first_k, second[0])


def test_decoupled_readout_reuses_workspace_for_same_block_clean_overwrite_without_middle_update():
    config = _build_config(num_layers=1, num_heads=1, capacities=[16])
    cache = _CollectCountingAdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=16,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        use_osc_frame_mode=False,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k0 = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    first_collect_calls = cache.middle_collect_calls

    k1 = _make_tokens(200, 4, num_heads=1, head_dim=4)
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)

    assert first_collect_calls == 1
    assert cache.middle_collect_calls == first_collect_calls


def test_decoupled_readout_cache_can_be_disabled():
    config = _build_config(num_layers=1, num_heads=1, capacities=[16])
    cache = _CollectCountingAdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=4,
        layer_idx=0,
        sink_len=2,
        tail_len=16,
        sink_grid_decoupling=True,
        ivc_ratio=0.0,
        semantic_ratio=0.0,
        trajectory_ratio=0.0,
        readout_cache_enabled=False,
        update_interval=100,
    )
    freqs = _build_rope_freqs(4, max_seq_len=64)
    grid_sizes = torch.tensor([[1, 1, 4]], dtype=torch.long)

    k0 = _make_tokens(0, 4, num_heads=1, head_dim=4)
    cache.update(k0, k0.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="clean")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)
    first_collect_calls = cache.middle_collect_calls

    k1 = _make_tokens(100, 4, num_heads=1, head_dim=4)
    cache.update(k1, k1.clone(), current_start=0, grid_sizes=grid_sizes, freqs=freqs, cache_update_mode="noisy")
    cache.get_decoupled_flat_kv_and_frames(current_start=0, grid_sizes=grid_sizes, freqs=freqs)

    assert first_collect_calls == 1
    assert cache.middle_collect_calls == first_collect_calls + 1

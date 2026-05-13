from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from pyramidkv.adaptive_cache import AdaptiveKVCache


def _make_cache() -> AdaptiveKVCache:
    cache = AdaptiveKVCache.__new__(AdaptiveKVCache)
    cache.head_labels = [-1, 1, 2, 7]
    cache.osc_head_flags = [True, False, False, False]
    cache.af_head_groups = ["A", "B", "C", ""]
    cache.label_recent_frames_map = {"-1": 9}
    cache.label_phase_bucket_map = {"-1": 0}
    cache.label_lag_offsets_map = {"-1": [], "7": [3, 9]}
    cache.label_sink_frames_map = {"-1": 5, "7": 4}
    cache.label_stride_enabled_map = {"1": False, "2": True}
    cache.use_af_head_policies = True
    cache.af_recent_frames_map = {"B": 6}
    cache.af_phase_bucket_map = {"B": 2}
    cache.af_lag_offsets_map = {"C": [6]}
    cache.af_sink_frames_map = {"C": 3}
    cache.af_stride_enabled_map = {"C": True}
    cache.local_tail_frames = 4
    cache.phase_bucket_capacity_frames = 1
    cache.use_osc_lag_mode = True
    cache.osc_lag_offsets_frames = [6]
    cache.use_stable_head_policies = True
    cache.stable_sink_frames = 2
    cache.osc_sink_frames = 1
    cache.stable_recent_frames = 7
    cache.policies_row = [
        SimpleNamespace(policy_type="osc"),
        SimpleNamespace(policy_type="stride"),
        SimpleNamespace(policy_type="recent_only"),
        SimpleNamespace(policy_type="recent_only"),
    ]
    return cache


def test_recent_phase_lag_maps_use_per_label_then_af_then_global():
    cache = _make_cache()
    assert cache._head_recent_frames(0) == 9
    assert cache._head_recent_frames(1) == 6
    assert cache._head_recent_frames(3) == 7

    assert cache._head_phase_bucket_capacity(0) == 0
    assert cache._head_phase_bucket_capacity(1) == 2
    assert cache._head_phase_bucket_capacity(3) == 1

    assert cache._head_lag_offsets(0) == []
    assert cache._head_lag_offsets(2) == [6]
    assert cache._head_lag_offsets(3) == [3, 9]


def test_sink_and_stride_kind_maps_use_per_label_then_af_then_global():
    cache = _make_cache()
    assert cache._head_sink_frames(0) == 5
    assert cache._head_sink_frames(2) == 3
    assert cache._head_sink_frames(1) == 2

    assert cache._stable_strategy_kind(0) is None
    assert cache._stable_strategy_kind(1) == "recent_only"
    assert cache._stable_strategy_kind(2) == "stride"
    assert cache._stable_strategy_kind(3) == "recent_only"


def test_cache_map_builders_accept_omegaconf_configs():
    cache = AdaptiveKVCache.__new__(AdaptiveKVCache)
    cfg = OmegaConf.create(
        {
            "label_recent": {"-1": 18},
            "label_phase": {"-1": 0},
            "label_lag": {"-1": [3, 6]},
            "label_sink": {"1": 3},
            "label_stride": {"2": False},
            "af_recent": {"A": 5},
            "af_phase": {"D": 2},
            "af_lag": {"D": [6, 12]},
            "af_sink": {"E": 2},
            "af_stride": {"F": True},
        }
    )

    assert cache._build_label_recent_frames_map(cfg.label_recent) == {"-1": 18}
    assert cache._build_label_phase_bucket_map(cfg.label_phase) == {"-1": 0}
    assert cache._build_label_lag_offsets_map(cfg.label_lag) == {"-1": [3, 6]}
    assert cache._build_label_sink_frames_map(cfg.label_sink) == {"1": 3}
    assert cache._build_label_stride_enabled_map(cfg.label_stride) == {"2": False}
    assert cache._build_af_recent_frames_map(cfg.af_recent)["A"] == 5
    assert cache._build_af_phase_bucket_map(cfg.af_phase)["D"] == 2
    assert cache._build_af_lag_offsets_map(cfg.af_lag)["D"] == [6, 12]
    assert cache._build_af_sink_frames_map(cfg.af_sink) == {"E": 2}
    assert cache._build_af_stride_enabled_map(cfg.af_stride) == {"F": True}

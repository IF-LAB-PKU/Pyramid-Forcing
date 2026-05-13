from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from headkv.adaptive_cache import AdaptiveKVCache


def _base_cache() -> AdaptiveKVCache:
    cache = AdaptiveKVCache.__new__(AdaptiveKVCache)
    cache.head_labels = [-1, 1, 2, 1, 1, 1]
    cache.osc_head_flags = [True, False, False, False, False, False]
    cache.af_head_groups = ["A", "B", "C", "D", "E", "F"]
    cache.local_tail_frames = 4
    cache.phase_bucket_capacity_frames = 1
    cache.use_osc_lag_mode = True
    cache.osc_lag_offsets_frames = [6]
    cache.use_stable_head_policies = True
    cache.stable_sink_frames = 2
    cache.osc_sink_frames = 1
    cache.stable_recent_frames = 7
    cache.use_af_head_policies = True
    cache.policies_row = [
        SimpleNamespace(policy_type="osc"),
        SimpleNamespace(policy_type="recent_only"),
        SimpleNamespace(policy_type="recent_only"),
        SimpleNamespace(policy_type="recent_only"),
        SimpleNamespace(policy_type="recent_only"),
        SimpleNamespace(policy_type="recent_only"),
    ]
    return cache


def test_af_runtime_helpers_accept_omegaconf_and_drive_semantics():
    cache = _base_cache()
    cfg = OmegaConf.create(
        {
            "af_recent": {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3},
            "af_phase": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
            "af_lag": {"A": [], "B": [3], "C": [6], "D": [9], "E": [12], "F": [15]},
            "af_sink": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
            "af_stride": {"A": False, "B": True, "C": False, "D": True, "E": False, "F": True},
        }
    )
    cache.label_recent_frames_map = {}
    cache.label_phase_bucket_map = {}
    cache.label_lag_offsets_map = {}
    cache.label_sink_frames_map = {}
    cache.label_stride_enabled_map = {}
    cache.af_recent_frames_map = cache._build_af_recent_frames_map(cfg.af_recent)
    cache.af_phase_bucket_map = cache._build_af_phase_bucket_map(cfg.af_phase)
    cache.af_lag_offsets_map = cache._build_af_lag_offsets_map(cfg.af_lag)
    cache.af_sink_frames_map = cache._build_af_sink_frames_map(cfg.af_sink)
    cache.af_stride_enabled_map = cache._build_af_stride_enabled_map(cfg.af_stride)

    assert cache._head_recent_frames(1) == 7
    assert cache._head_phase_bucket_capacity(3) == 3
    assert cache._head_lag_offsets(4) == [12]
    assert cache._head_sink_frames(5) == 6
    assert cache._stable_strategy_kind(1) == "stride"
    assert cache._stable_strategy_kind(2) == "recent_only"
    assert cache._stable_strategy_kind(3) == "stride"


def test_per_label_runtime_helpers_loaded_from_omegaconf_override_af_and_global():
    cache = _base_cache()
    af_cfg = OmegaConf.create(
        {
            "recent": {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3},
            "phase": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
            "lag": {"A": [], "B": [3], "C": [6], "D": [9], "E": [12], "F": [15]},
            "sink": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
            "stride": {"A": False, "B": True, "C": False, "D": True, "E": False, "F": True},
        }
    )
    label_cfg = OmegaConf.create(
        {
            "recent": {"1": 11, "2": 9},
            "phase": {"1": 0, "-1": 0},
            "lag": {"1": [21]},
            "sink": {"1": 5, "2": 4},
            "stride": {"1": False, "2": True},
        }
    )
    cache.af_recent_frames_map = cache._build_af_recent_frames_map(af_cfg.recent)
    cache.af_phase_bucket_map = cache._build_af_phase_bucket_map(af_cfg.phase)
    cache.af_lag_offsets_map = cache._build_af_lag_offsets_map(af_cfg.lag)
    cache.af_sink_frames_map = cache._build_af_sink_frames_map(af_cfg.sink)
    cache.af_stride_enabled_map = cache._build_af_stride_enabled_map(af_cfg.stride)
    cache.label_recent_frames_map = cache._build_label_recent_frames_map(label_cfg.recent)
    cache.label_phase_bucket_map = cache._build_label_phase_bucket_map(label_cfg.phase)
    cache.label_lag_offsets_map = cache._build_label_lag_offsets_map(label_cfg.lag)
    cache.label_sink_frames_map = cache._build_label_sink_frames_map(label_cfg.sink)
    cache.label_stride_enabled_map = cache._build_label_stride_enabled_map(label_cfg.stride)

    assert cache._head_recent_frames(1) == 11
    assert cache._head_phase_bucket_capacity(1) == 0
    assert cache._head_lag_offsets(1) == [21]
    assert cache._head_sink_frames(1) == 5
    assert cache._stable_strategy_kind(1) == "recent_only"

    assert cache._head_recent_frames(2) == 9
    assert cache._head_sink_frames(2) == 4
    assert cache._stable_strategy_kind(2) == "stride"


def test_runtime_helpers_match_between_omegaconf_and_plain_dict_inputs():
    cache_a = _base_cache()
    cache_b = _base_cache()
    cfg = OmegaConf.create(
        {
            "label_recent": {"-1": 18, "1": 11},
            "label_phase": {"-1": 0},
            "label_lag": {"1": [12]},
            "label_sink": {"-1": 3, "1": 5},
            "label_stride": {"1": False, "2": True},
            "af_recent": {"A": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3},
            "af_phase": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
            "af_lag": {"A": [], "B": [3], "C": [6], "D": [9], "E": [12], "F": [15]},
            "af_sink": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
            "af_stride": {"A": False, "B": True, "C": False, "D": True, "E": False, "F": True},
        }
    )
    plain = OmegaConf.to_container(cfg, resolve=True)

    for cache, source in ((cache_a, cfg), (cache_b, plain)):
        cache.af_recent_frames_map = cache._build_af_recent_frames_map(source["af_recent"])
        cache.af_phase_bucket_map = cache._build_af_phase_bucket_map(source["af_phase"])
        cache.af_lag_offsets_map = cache._build_af_lag_offsets_map(source["af_lag"])
        cache.af_sink_frames_map = cache._build_af_sink_frames_map(source["af_sink"])
        cache.af_stride_enabled_map = cache._build_af_stride_enabled_map(source["af_stride"])
        cache.label_recent_frames_map = cache._build_label_recent_frames_map(source["label_recent"])
        cache.label_phase_bucket_map = cache._build_label_phase_bucket_map(source["label_phase"])
        cache.label_lag_offsets_map = cache._build_label_lag_offsets_map(source["label_lag"])
        cache.label_sink_frames_map = cache._build_label_sink_frames_map(source["label_sink"])
        cache.label_stride_enabled_map = cache._build_label_stride_enabled_map(source["label_stride"])

    for head_idx in range(len(cache_a.head_labels)):
        assert cache_a._head_recent_frames(head_idx) == cache_b._head_recent_frames(head_idx)
        assert cache_a._head_phase_bucket_capacity(head_idx) == cache_b._head_phase_bucket_capacity(head_idx)
        assert cache_a._head_lag_offsets(head_idx) == cache_b._head_lag_offsets(head_idx)
        assert cache_a._head_sink_frames(head_idx) == cache_b._head_sink_frames(head_idx)
        assert cache_a._stable_strategy_kind(head_idx) == cache_b._stable_strategy_kind(head_idx)

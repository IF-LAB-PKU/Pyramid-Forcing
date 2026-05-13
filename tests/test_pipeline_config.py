"""Tests for pipeline/pyramidkv_config.py — PyramidKVPipelineConfig.from_args()."""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

PYRAMIDKV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "pyramidkv_config.py"
_spec = importlib.util.spec_from_file_location("pyramidkv_config_module", PYRAMIDKV_CONFIG_PATH)
_pyramidkv_config_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _pyramidkv_config_module
_spec.loader.exec_module(_pyramidkv_config_module)
PyramidKVPipelineConfig = _pyramidkv_config_module.PyramidKVPipelineConfig


class TestFromArgsMinimal:
    def test_from_args_minimal(self):
        args = SimpleNamespace()
        cfg = PyramidKVPipelineConfig.from_args(args)
        assert cfg.use_pyramidkv is False
        assert cfg.pyramidkv_config_path is None
        assert cfg.cyclic_period == 6
        assert cfg.pyramidkv_recent_frames == 4
        assert cfg.pyramidkv_prompt_v_cache_enabled is False


class TestFromArgsExtractsAllFields:
    def test_from_args_extracts_all_fields(self):
        args = SimpleNamespace(
            use_pyramidkv=True,
            pyramidkv_config_path="/some/path.csv",
            pyramidkv_default_capacity=16384,
            pyramidkv_strategy_factor=2,
            pyramidkv_code_map={"-1": 8192, "1": 16384},
            pyramidkv_context_len=780,
            i2v=True,
            use_adaptive_pyramidkv=True,
            pyramidkv_policy_csv_path="/some/policy.csv",
            pyramidkv_drop_heads_csv_path="/some/drop.csv",
            pyramidkv_soft_ablate_csv_path="/some/soft.csv",
            pyramidkv_soft_ablate_region="middle",
            pyramidkv_soft_ablate_scale=0.5,
            pyramidkv_dynamic_rope_mode="window_clamp",
            pyramidkv_sink_tokens=780,
            pyramidkv_recent_frames=3,
            pyramidkv_lag_offsets=[6, 12],
            pyramidkv_lag_history=15,
            pyramidkv_disable_osc_sink=True,
            pyramidkv_stable_policy_enabled=False,
            pyramidkv_stable_sink_frames=3,
            pyramidkv_osc_sink_frames=1,
            pyramidkv_stable_recent_frames=5,
            pyramidkv_frame_seq_length=780,
            pyramidkv_capture_frame_id_mode="raw",
            pyramidkv_af_policy_enabled=True,
            pyramidkv_af_csv_path="/some/af.csv",
            pyramidkv_af_group_dir="/some/dir",
            pyramidkv_af_manifest_path="/some/manifest.csv",
            pyramidkv_af_recent_frames_map={"A": 4},
            pyramidkv_af_phase_bucket_map={"B": 3},
            pyramidkv_af_lag_offsets_map={"C": []},
            pyramidkv_af_sink_frames_map={"D": 1},
            pyramidkv_af_stride_enabled_map={"E": False},
            pyramidkv_label_sink_frames_map={"-1": 1, "7": 2},
            pyramidkv_label_recent_frames_map={"1": 6},
            pyramidkv_label_stride_enabled_map={"2": False},
            pyramidkv_label_phase_bucket_map={"-1": 0},
            pyramidkv_label_lag_offsets_map={"7": [3, 6]},
            pyramidkv_label_merge_enabled_map={"1": True},
            pyramidkv_label_merge_patch_size_map={"1": 2},
            pyramidkv_label_merge_capacity_map={"1": 3},
            pyramidkv_dynamic_capacity=7800,
            ivc_ratio=0.2,
            semantic_ratio=0.2,
            trajectory_ratio=0.1,
            trajectory_weight=0.5,
            history_frame_quota=3,
            history_quota_ivc_ratio=0.05,
            post_train_stabilize_t=10,
            post_train_trajectory_scale=2.0,
            post_train_history_ivc_ratio=0.1,
            update_interval=2,
            semantic_seed_ratio=0.02,
            sink_grid_decoupling=True,
            decoupled_sink_tokens=780,
            decoupled_sink_time_lag=3,
            sink_time_clamp_min=15,
            sink_time_clamp_max=18,
            history_time_mapping_mode="relative",
            history_relative_t_max=15,
            history_time_soft_factor=0.3,
            pyramidkv_readout_cache_enabled=False,
            pyramidkv_prompt_v_cache_enabled=False,
            cyclic_enabled=True,
            cyclic_period=8,
            cyclic_bucket_cap=2,
            cyclic_osc_only=False,
            cyclic_dynamic_rope=False,
            lag_enabled=True,
            lag_dynamic_rope=True,
            stride_enabled=True,
            stride_interval=8,
            stride_dynamic_rope=False,
            merge_enabled=True,
            merge_patch_size=2,
            merge_capacity=3,
            merge_dynamic_rope=False,
        )
        cfg = PyramidKVPipelineConfig.from_args(args, frame_seq_length=780)
        assert cfg.use_pyramidkv is True
        assert cfg.pyramidkv_config_path == "/some/path.csv"
        assert cfg.pyramidkv_default_capacity == 16384
        assert cfg.pyramidkv_strategy_factor == 2
        assert cfg.pyramidkv_is_i2v is True
        assert cfg.use_adaptive_pyramidkv is True
        assert cfg.cyclic_enabled is True
        assert cfg.cyclic_period == 8
        assert cfg.lag_enabled is True
        assert cfg.stride_enabled is True
        assert cfg.stride_interval == 8
        assert cfg.pyramidkv_af_policy_enabled is True
        assert cfg.pyramidkv_dynamic_capacity == 7800
        assert cfg.pyramidkv_label_sink_frames_map == {"-1": 1, "7": 2}
        assert cfg.pyramidkv_label_recent_frames_map == {"1": 6}
        assert cfg.pyramidkv_label_stride_enabled_map == {"2": False}
        assert cfg.pyramidkv_label_phase_bucket_map == {"-1": 0}
        assert cfg.pyramidkv_label_lag_offsets_map == {"7": [3, 6]}
        assert cfg.pyramidkv_label_merge_enabled_map == {"1": True}
        assert cfg.pyramidkv_label_merge_patch_size_map == {"1": 2}
        assert cfg.pyramidkv_label_merge_capacity_map == {"1": 3}
        assert cfg.merge_enabled is True
        assert cfg.merge_patch_size == 2
        assert cfg.merge_capacity == 3
        assert cfg.merge_dynamic_rope is False
        assert cfg.pyramidkv_readout_cache_enabled is False
        assert cfg.pyramidkv_prompt_v_cache_enabled is False


class TestLagDefaults:
    def test_lag_enabled_defaults_from_offsets(self):
        args = SimpleNamespace(pyramidkv_lag_offsets=[6])
        cfg = PyramidKVPipelineConfig.from_args(args)
        assert cfg.lag_enabled is True

    def test_lag_disabled_when_offsets_empty(self):
        args = SimpleNamespace(pyramidkv_lag_offsets=[])
        cfg = PyramidKVPipelineConfig.from_args(args)
        assert cfg.lag_enabled is False


class TestFrameSeqLengthPropagation:
    def test_frame_seq_length_propagation(self):
        args = SimpleNamespace()
        cfg = PyramidKVPipelineConfig.from_args(args, frame_seq_length=780)
        assert cfg.pyramidkv_frame_seq_length == 780
        assert cfg.pyramidkv_context_len == 780
        assert cfg.pyramidkv_sink_tokens == 780
        assert cfg.pyramidkv_dynamic_capacity == 4 * 780


class TestCyclicEnabledDefault:
    def test_cyclic_enabled_defaults_to_adaptive(self):
        args = SimpleNamespace(use_adaptive_pyramidkv=True)
        cfg = PyramidKVPipelineConfig.from_args(args)
        assert cfg.cyclic_enabled is True

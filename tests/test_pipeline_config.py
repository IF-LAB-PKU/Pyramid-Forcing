"""Tests for pipeline/headkv_config.py — HeadKVPipelineConfig.from_args()."""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

HEADKV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "headkv_config.py"
_spec = importlib.util.spec_from_file_location("headkv_config_module", HEADKV_CONFIG_PATH)
_headkv_config_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _headkv_config_module
_spec.loader.exec_module(_headkv_config_module)
HeadKVPipelineConfig = _headkv_config_module.HeadKVPipelineConfig


class TestFromArgsMinimal:
    def test_from_args_minimal(self):
        args = SimpleNamespace()
        cfg = HeadKVPipelineConfig.from_args(args)
        assert cfg.use_headkv is False
        assert cfg.headkv_config_path is None
        assert cfg.cyclic_period == 6
        assert cfg.headkv_recent_frames == 4
        assert cfg.headkv_prompt_v_cache_enabled is False


class TestFromArgsExtractsAllFields:
    def test_from_args_extracts_all_fields(self):
        args = SimpleNamespace(
            use_headkv=True,
            headkv_config_path="/some/path.csv",
            headkv_default_capacity=16384,
            headkv_strategy_factor=2,
            headkv_code_map={"-1": 8192, "1": 16384},
            headkv_context_len=780,
            i2v=True,
            use_adaptive_headkv=True,
            headkv_policy_csv_path="/some/policy.csv",
            headkv_drop_heads_csv_path="/some/drop.csv",
            headkv_soft_ablate_csv_path="/some/soft.csv",
            headkv_soft_ablate_region="middle",
            headkv_soft_ablate_scale=0.5,
            headkv_dynamic_rope_mode="window_clamp",
            headkv_sink_tokens=780,
            headkv_recent_frames=3,
            headkv_lag_offsets=[6, 12],
            headkv_lag_history=15,
            headkv_disable_osc_sink=True,
            headkv_stable_policy_enabled=False,
            headkv_stable_sink_frames=3,
            headkv_osc_sink_frames=1,
            headkv_stable_recent_frames=5,
            headkv_frame_seq_length=780,
            headkv_capture_frame_id_mode="raw",
            headkv_af_policy_enabled=True,
            headkv_af_csv_path="/some/af.csv",
            headkv_af_group_dir="/some/dir",
            headkv_af_manifest_path="/some/manifest.csv",
            headkv_af_recent_frames_map={"A": 4},
            headkv_af_phase_bucket_map={"B": 3},
            headkv_af_lag_offsets_map={"C": []},
            headkv_af_sink_frames_map={"D": 1},
            headkv_af_stride_enabled_map={"E": False},
            headkv_label_sink_frames_map={"-1": 1, "7": 2},
            headkv_label_recent_frames_map={"1": 6},
            headkv_label_stride_enabled_map={"2": False},
            headkv_label_phase_bucket_map={"-1": 0},
            headkv_label_lag_offsets_map={"7": [3, 6]},
            headkv_label_merge_enabled_map={"1": True},
            headkv_label_merge_patch_size_map={"1": 2},
            headkv_label_merge_capacity_map={"1": 3},
            headkv_dynamic_capacity=7800,
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
            headkv_readout_cache_enabled=False,
            headkv_prompt_v_cache_enabled=False,
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
        cfg = HeadKVPipelineConfig.from_args(args, frame_seq_length=780)
        assert cfg.use_headkv is True
        assert cfg.headkv_config_path == "/some/path.csv"
        assert cfg.headkv_default_capacity == 16384
        assert cfg.headkv_strategy_factor == 2
        assert cfg.headkv_is_i2v is True
        assert cfg.use_adaptive_headkv is True
        assert cfg.cyclic_enabled is True
        assert cfg.cyclic_period == 8
        assert cfg.lag_enabled is True
        assert cfg.stride_enabled is True
        assert cfg.stride_interval == 8
        assert cfg.headkv_af_policy_enabled is True
        assert cfg.headkv_dynamic_capacity == 7800
        assert cfg.headkv_label_sink_frames_map == {"-1": 1, "7": 2}
        assert cfg.headkv_label_recent_frames_map == {"1": 6}
        assert cfg.headkv_label_stride_enabled_map == {"2": False}
        assert cfg.headkv_label_phase_bucket_map == {"-1": 0}
        assert cfg.headkv_label_lag_offsets_map == {"7": [3, 6]}
        assert cfg.headkv_label_merge_enabled_map == {"1": True}
        assert cfg.headkv_label_merge_patch_size_map == {"1": 2}
        assert cfg.headkv_label_merge_capacity_map == {"1": 3}
        assert cfg.merge_enabled is True
        assert cfg.merge_patch_size == 2
        assert cfg.merge_capacity == 3
        assert cfg.merge_dynamic_rope is False
        assert cfg.headkv_readout_cache_enabled is False
        assert cfg.headkv_prompt_v_cache_enabled is False


class TestLagDefaults:
    def test_lag_enabled_defaults_from_offsets(self):
        args = SimpleNamespace(headkv_lag_offsets=[6])
        cfg = HeadKVPipelineConfig.from_args(args)
        assert cfg.lag_enabled is True

    def test_lag_disabled_when_offsets_empty(self):
        args = SimpleNamespace(headkv_lag_offsets=[])
        cfg = HeadKVPipelineConfig.from_args(args)
        assert cfg.lag_enabled is False


class TestFrameSeqLengthPropagation:
    def test_frame_seq_length_propagation(self):
        args = SimpleNamespace()
        cfg = HeadKVPipelineConfig.from_args(args, frame_seq_length=780)
        assert cfg.headkv_frame_seq_length == 780
        assert cfg.headkv_context_len == 780
        assert cfg.headkv_sink_tokens == 780
        assert cfg.headkv_dynamic_capacity == 4 * 780


class TestCyclicEnabledDefault:
    def test_cyclic_enabled_defaults_to_adaptive(self):
        args = SimpleNamespace(use_adaptive_headkv=True)
        cfg = HeadKVPipelineConfig.from_args(args)
        assert cfg.cyclic_enabled is True

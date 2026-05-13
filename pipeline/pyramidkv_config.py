from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PyramidKVPipelineConfig:
    """Unified configuration for PyramidKV pipeline parameters."""

    # --- Basic PyramidKV settings ---
    use_pyramidkv: bool = False
    pyramidkv_config_path: Optional[str] = None
    pyramidkv_default_capacity: Optional[int] = None
    pyramidkv_strategy_factor: int = 3
    pyramidkv_code_map: Optional[dict] = None
    pyramidkv_context_len: int = 1560
    pyramidkv_is_i2v: bool = False
    use_adaptive_pyramidkv: bool = False
    pyramidkv_policy_csv_path: Optional[str] = None
    pyramidkv_drop_heads_csv_path: Optional[str] = None
    pyramidkv_soft_ablate_csv_path: Optional[str] = None
    pyramidkv_soft_ablate_region: str = "none"
    pyramidkv_soft_ablate_scale: float = 1.0
    pyramidkv_dynamic_rope_mode: str = "lag"
    pyramidkv_sink_tokens: int = 1560
    pyramidkv_recent_frames: int = 4
    pyramidkv_lag_offsets: list = field(default_factory=lambda: [6])
    pyramidkv_lag_history: int = 21
    pyramidkv_disable_osc_sink: bool = False
    pyramidkv_stable_policy_enabled: bool = True
    pyramidkv_stable_sink_frames: Optional[int] = None
    pyramidkv_osc_sink_frames: Optional[int] = None
    pyramidkv_stable_recent_frames: Optional[int] = None
    pyramidkv_frame_seq_length: int = 1560
    pyramidkv_capture_frame_id_mode: str = "mapped"

    # --- A-F taxonomy ---
    pyramidkv_af_policy_enabled: bool = False
    pyramidkv_af_csv_path: Optional[str] = None
    pyramidkv_af_group_dir: Optional[str] = None
    pyramidkv_af_manifest_path: Optional[str] = None
    pyramidkv_af_recent_frames_map: Optional[dict] = None
    pyramidkv_af_phase_bucket_map: Optional[dict] = None
    pyramidkv_af_lag_offsets_map: Optional[dict] = None
    pyramidkv_af_sink_frames_map: Optional[dict] = None
    pyramidkv_af_stride_enabled_map: Optional[dict] = None

    # --- Per-label policy maps ---
    pyramidkv_label_sink_frames_map: Optional[dict] = None
    pyramidkv_label_recent_frames_map: Optional[dict] = None
    pyramidkv_label_stride_enabled_map: Optional[dict] = None
    pyramidkv_label_stride_interval_map: Optional[dict] = None
    pyramidkv_label_phase_bucket_map: Optional[dict] = None
    pyramidkv_label_lag_offsets_map: Optional[dict] = None
    pyramidkv_label_merge_enabled_map: Optional[dict] = None
    pyramidkv_label_merge_patch_size_map: Optional[dict] = None
    pyramidkv_label_merge_capacity_map: Optional[dict] = None

    # --- Adaptive cache params ---
    pyramidkv_dynamic_capacity: int = 6240  # was tail_len; 4 * 1560
    ivc_ratio: float = 0.1
    semantic_ratio: float = 0.1
    trajectory_ratio: float = 0.0
    trajectory_weight: float = 0.0
    history_frame_quota: int = 0
    history_quota_ivc_ratio: float = 0.0
    post_train_stabilize_t: int = -1
    post_train_trajectory_scale: float = 1.0
    post_train_history_ivc_ratio: float = -1.0
    update_interval: int = 1
    semantic_seed_ratio: float = 0.01
    sink_grid_decoupling: bool = False
    decoupled_sink_tokens: int = 0
    decoupled_sink_time_lag: int = 0
    sink_time_clamp_min: int = 18
    sink_time_clamp_max: int = 21
    history_time_mapping_mode: str = "none"
    history_relative_t_max: int = 21
    history_time_soft_factor: float = 0.5
    pyramidkv_readout_cache_enabled: bool = True
    pyramidkv_prompt_v_cache_enabled: bool = False
    # --- Cyclic (was phase/osc_frame) ---
    cyclic_enabled: bool = False  # was use_osc_frame_mode
    cyclic_period: int = 6  # was phase_period
    cyclic_bucket_cap: int = 1  # was phase_bucket_capacity_frames
    cyclic_osc_only: bool = True  # was phase_sink_for_osc_only
    cyclic_dynamic_rope: bool = True  # was phase_sink_dynamic_rope

    # --- Lag ---
    lag_enabled: bool = False  # was use_osc_lag_mode
    lag_dynamic_rope: bool = False  # was osc_lag_dynamic_rope

    # --- Stride ---
    stride_enabled: bool = False
    stride_interval: int = 6  # every k-th frame
    stride_capacity: int = -1  # max stride anchors per head; -1 = unlimited
    stride_dynamic_rope: bool = True

    # --- Merge ---
    merge_enabled: bool = False
    merge_patch_size: int = 2
    merge_capacity: int = 1
    merge_dynamic_rope: bool = True

    @classmethod
    def from_args(cls, args, frame_seq_length: int = 1560) -> "PyramidKVPipelineConfig":
        """Build config from an OmegaConf/argparse namespace."""

        use_adaptive = getattr(args, "use_adaptive_pyramidkv", False)
        lag_offsets = getattr(args, "pyramidkv_lag_offsets", [6])
        lag_enabled_default = len(lag_offsets) > 0 if lag_offsets else False

        return cls(
            use_pyramidkv=getattr(args, "use_pyramidkv", False),
            pyramidkv_config_path=getattr(args, "pyramidkv_config_path", None),
            pyramidkv_default_capacity=getattr(args, "pyramidkv_default_capacity", None),
            pyramidkv_strategy_factor=getattr(args, "pyramidkv_strategy_factor", 3),
            pyramidkv_code_map=getattr(args, "pyramidkv_code_map", None),
            pyramidkv_context_len=getattr(args, "pyramidkv_context_len", frame_seq_length),
            pyramidkv_is_i2v=getattr(args, "i2v", False),
            use_adaptive_pyramidkv=use_adaptive,
            pyramidkv_policy_csv_path=getattr(args, "pyramidkv_policy_csv_path", None),
            pyramidkv_drop_heads_csv_path=getattr(args, "pyramidkv_drop_heads_csv_path", None),
            pyramidkv_soft_ablate_csv_path=getattr(args, "pyramidkv_soft_ablate_csv_path", None),
            pyramidkv_soft_ablate_region=getattr(args, "pyramidkv_soft_ablate_region", "none"),
            pyramidkv_soft_ablate_scale=float(getattr(args, "pyramidkv_soft_ablate_scale", 1.0)),
            pyramidkv_dynamic_rope_mode=getattr(args, "pyramidkv_dynamic_rope_mode", "lag"),
            pyramidkv_sink_tokens=getattr(args, "pyramidkv_sink_tokens", frame_seq_length),
            pyramidkv_recent_frames=getattr(args, "pyramidkv_recent_frames", 4),
            pyramidkv_lag_offsets=lag_offsets,
            pyramidkv_lag_history=getattr(args, "pyramidkv_lag_history", 21),
            pyramidkv_disable_osc_sink=getattr(args, "pyramidkv_disable_osc_sink", False),
            pyramidkv_stable_policy_enabled=getattr(args, "pyramidkv_stable_policy_enabled", True),
            pyramidkv_stable_sink_frames=getattr(args, "pyramidkv_stable_sink_frames", None),
            pyramidkv_osc_sink_frames=getattr(args, "pyramidkv_osc_sink_frames", None),
            pyramidkv_stable_recent_frames=getattr(args, "pyramidkv_stable_recent_frames", None),
            pyramidkv_frame_seq_length=int(getattr(args, "pyramidkv_frame_seq_length", frame_seq_length)),
            pyramidkv_capture_frame_id_mode=getattr(args, "pyramidkv_capture_frame_id_mode", "mapped"),
            pyramidkv_af_policy_enabled=bool(getattr(args, "pyramidkv_af_policy_enabled", False)),
            pyramidkv_af_csv_path=getattr(args, "pyramidkv_af_csv_path", None),
            pyramidkv_af_group_dir=getattr(args, "pyramidkv_af_group_dir", None),
            pyramidkv_af_manifest_path=getattr(args, "pyramidkv_af_manifest_path", None),
            pyramidkv_af_recent_frames_map=getattr(args, "pyramidkv_af_recent_frames_map", None),
            pyramidkv_af_phase_bucket_map=getattr(args, "pyramidkv_af_phase_bucket_map", None),
            pyramidkv_af_lag_offsets_map=getattr(args, "pyramidkv_af_lag_offsets_map", None),
            pyramidkv_af_sink_frames_map=getattr(args, "pyramidkv_af_sink_frames_map", None),
            pyramidkv_af_stride_enabled_map=getattr(args, "pyramidkv_af_stride_enabled_map", None),
            pyramidkv_label_sink_frames_map=getattr(args, "pyramidkv_label_sink_frames_map", None),
            pyramidkv_label_recent_frames_map=getattr(args, "pyramidkv_label_recent_frames_map", None),
            pyramidkv_label_stride_enabled_map=getattr(args, "pyramidkv_label_stride_enabled_map", None),
            pyramidkv_label_stride_interval_map=getattr(args, "pyramidkv_label_stride_interval_map", None),
            pyramidkv_label_phase_bucket_map=getattr(args, "pyramidkv_label_phase_bucket_map", None),
            pyramidkv_label_lag_offsets_map=getattr(args, "pyramidkv_label_lag_offsets_map", None),
            pyramidkv_label_merge_enabled_map=getattr(args, "pyramidkv_label_merge_enabled_map", None),
            pyramidkv_label_merge_patch_size_map=getattr(args, "pyramidkv_label_merge_patch_size_map", None),
            pyramidkv_label_merge_capacity_map=getattr(args, "pyramidkv_label_merge_capacity_map", None),
            pyramidkv_dynamic_capacity=getattr(args, "pyramidkv_dynamic_capacity", 4 * frame_seq_length),
            ivc_ratio=getattr(args, "ivc_ratio", 0.1),
            semantic_ratio=getattr(args, "semantic_ratio", 0.1),
            trajectory_ratio=getattr(args, "trajectory_ratio", 0.0),
            trajectory_weight=getattr(args, "trajectory_weight", 0.0),
            history_frame_quota=getattr(args, "history_frame_quota", 0),
            history_quota_ivc_ratio=getattr(args, "history_quota_ivc_ratio", 0.0),
            post_train_stabilize_t=getattr(args, "post_train_stabilize_t", -1),
            post_train_trajectory_scale=getattr(args, "post_train_trajectory_scale", 1.0),
            post_train_history_ivc_ratio=getattr(args, "post_train_history_ivc_ratio", -1.0),
            update_interval=getattr(args, "update_interval", 1),
            semantic_seed_ratio=getattr(args, "semantic_seed_ratio", 0.01),
            sink_grid_decoupling=getattr(args, "sink_grid_decoupling", False),
            decoupled_sink_tokens=getattr(args, "decoupled_sink_tokens", 0),
            decoupled_sink_time_lag=getattr(args, "decoupled_sink_time_lag", 0),
            sink_time_clamp_min=getattr(args, "sink_time_clamp_min", 18),
            sink_time_clamp_max=getattr(args, "sink_time_clamp_max", 21),
            history_time_mapping_mode=getattr(args, "history_time_mapping_mode", "none"),
            history_relative_t_max=getattr(args, "history_relative_t_max", 21),
            history_time_soft_factor=getattr(args, "history_time_soft_factor", 0.5),
            pyramidkv_readout_cache_enabled=bool(getattr(args, "pyramidkv_readout_cache_enabled", True)),
            pyramidkv_prompt_v_cache_enabled=bool(getattr(args, "pyramidkv_prompt_v_cache_enabled", False)),
            cyclic_enabled=getattr(args, "cyclic_enabled", use_adaptive),
            cyclic_period=getattr(args, "cyclic_period", 6),
            cyclic_bucket_cap=getattr(args, "cyclic_bucket_cap", 1),
            cyclic_osc_only=getattr(args, "cyclic_osc_only", True),
            cyclic_dynamic_rope=getattr(args, "cyclic_dynamic_rope", True),
            lag_enabled=getattr(args, "lag_enabled", lag_enabled_default),
            lag_dynamic_rope=getattr(args, "lag_dynamic_rope", False),
            stride_enabled=getattr(args, "stride_enabled", False),
            stride_interval=getattr(args, "stride_interval", 6),
            stride_capacity=getattr(args, "stride_capacity", -1),
            stride_dynamic_rope=getattr(args, "stride_dynamic_rope", True),
            merge_enabled=getattr(args, "merge_enabled", False),
            merge_patch_size=getattr(args, "merge_patch_size", 2),
            merge_capacity=getattr(args, "merge_capacity", 1),
            merge_dynamic_rope=getattr(args, "merge_dynamic_rope", True),
        )

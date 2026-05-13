import os
import torch
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from ._scatter_ext import (
    scatter_available,
    scatter_copy,
    apply_pos_override,
    cuda_refresh_available,
    refresh_readout_layout,
)

from .cache import PyramidKVCache
from .selectors import (
    _topk_mask,
    _normalize_scores,
    ThreeDIVCSelector,
    SemanticValueSelector,
)
from .rope import (
    apply_rope_to_flat_k,
    map_dynamic_pos_time,
    map_sink_time,
)
from .cpp_strategy import (
    CppStrategyManager,
    compile_cpp_strategy_policies,
    cpp_strategy_requested,
)


def _as_long_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.long else tensor.to(dtype=torch.long)


@dataclass
class _ReadoutSegment:
    kind: str
    seq_idx: int
    offset: int
    length: int
    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    dynamic_rope_t: int | None
    dynamic_time_map: bool
    frame_ids_physical: bool


@dataclass
class _CppReadoutSegment:
    seq_idx: int
    head_idx: int
    offset: int
    length: int
    sync_t_raw: int
    tail_min_t: int
    sink_max_t: int
    dynamic_rope_t: int | None
    frame_ids_physical: bool


@dataclass
class _ReadoutSpec:
    segments: list[_ReadoutSegment]
    cpp_segments: list[_CppReadoutSegment]
    lengths: list[int]
    cu_cpu: list[int]
    total_len: int
    max_seqlen: int
    static_specs: list[tuple[int, int] | None]
    tail_specs: list[tuple[int, int] | None]
    shape_key: tuple
    anchor_shape_key: tuple


_CUDA_REFRESH_FLAG_DYNAMIC_TIME_MAP = 1
_CUDA_REFRESH_FLAG_FRAME_IDS_PHYSICAL = 2
_CUDA_REFRESH_FLAG_DYNAMIC_ROPE = 4


class AdaptiveKVCache(PyramidKVCache):
    """Per-head heterogeneous KV cache for Pyramid Forcing inference.

    Extends :class:`PyramidKVCache` with the full Pyramid Forcing policy stack:

      * **Composition** — per-head ``[sink + middle + recent]`` strategies
        loaded from ``config.compositions`` (see :class:`HeadComposition`).
        Middle slots are picked by pluggable strategies (cyclic, stride,
        lag, recent, merge) per head, so two heads in the same layer can
        retain completely different frames.
      * **Sink-grid decoupling** — when enabled, splits the static sink
        from the dynamic recent window and applies dynamic RoPE remapping
        so anchor positions stay within the model's trained range.
      * **Double-pass semantics** — supports Self-Forcing's noisy/clean
        cache update modes: noisy iterations overwrite the same tentative
        slot, only the clean pass commits to permanent storage.
      * **Workspace-pinned packing** — flat K/V/pos are written into
        pre-allocated workspace buffers (no Python allocations on hot
        path); an optional C++ shadow validator can mirror the same
        state into ``PyramidKVCacheManager`` for kernel parity testing.

    The class is instantiated per layer (and per batch element) by
    :class:`CausalInferencePipeline`. Construction reads its per-head
    capacities, labels, and compositions from the supplied
    :class:`PyramidKVConfig`; the rest of the constructor args toggle
    selector/RoPE/cache experiments.
    """

    post_prune_rope = True

    def __init__(
        self,
        config,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        is_i2v: bool = False,
        context_len: int = 0,
        sink_len: int = 0,
        tail_len: int = 32760,
        ivc_ratio: float = 0.1,
        semantic_ratio: float = 0.1,
        update_interval: int = 1,
        seed_ratio: float = 0.01,
        trajectory_ratio: float = 0.0,
        trajectory_weight: float = 0.0,
        history_frame_quota: int = 0,
        history_quota_ivc_ratio: float = 0.0,
        post_train_stabilize_t: int = -1,
        post_train_trajectory_scale: float = 1.0,
        post_train_history_ivc_ratio: float = -1.0,
        prune_sink: bool = False,
        prune_tail: bool = False,
        aggressive_all: bool = False,
        sink_grid_decoupling: bool = False,
        decoupled_sink_tokens: int = 0,
        decoupled_sink_time_lag: int = 0,
        sink_time_mapping_mode: str = "lag",
        sink_time_clamp_min: int = 18,
        sink_time_clamp_max: int = 21,
        history_time_mapping_mode: str = "none",
        history_relative_t_max: int = 21,
        history_time_soft_factor: float = 0.5,
        osc_full_kv_retention: bool = False,
        periodic_peak_mask: bool = False,
        periodic_peak_period: int = 6,
        periodic_peak_offsets: list[int] | None = None,
        periodic_peak_start_t: int = 6,
        periodic_peak_only_oscillating: bool = True,
        use_osc_frame_mode: bool = False,
        phase_period: int = 6,
        phase_bucket_capacity_frames: int = 1,
        local_tail_frames: int = 4,
        phase_sink_for_osc_only: bool = True,
        phase_sink_dynamic_rope: bool = True,
        use_osc_lag_mode: bool = False,
        osc_lag_offsets_frames: list[int] | None = None,
        osc_lag_history_frames: int = 21,
        osc_lag_dynamic_rope: bool = False,
        disable_first_sink_for_osc_heads: bool = False,
        use_stable_head_policies: bool = True,
        stable_sink_frames: int | None = None,
        osc_sink_frames: int | None = None,
        stable_recent_frames: int | None = None,
        use_af_head_policies: bool = False,
        af_recent_frames_map: dict | None = None,
        af_phase_bucket_map: dict | None = None,
        af_lag_offsets_map: dict | None = None,
        af_sink_frames_map: dict | None = None,
        af_stride_enabled_map: dict | None = None,
        label_recent_frames_map: dict | None = None,
        label_phase_bucket_map: dict | None = None,
        label_lag_offsets_map: dict | None = None,
        label_sink_frames_map: dict | None = None,
        label_stride_enabled_map: dict | None = None,
        capture_frame_id_mode: str = "mapped",
        readout_cache_enabled: bool = True,
        prompt_value_cache_enabled: bool = False,
    ):
        super().__init__(
            config=config,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            is_i2v=is_i2v,
            context_len=context_len,
            prompt_value_cache_enabled=prompt_value_cache_enabled,
        )
        self.sink_len = max(0, int(sink_len))
        self.tail_len = max(0, int(tail_len))
        self.ivc_ratio = float(ivc_ratio)
        self.semantic_ratio = float(semantic_ratio)
        self.update_interval = max(1, int(update_interval))
        self.seed_ratio = float(seed_ratio)
        self.trajectory_ratio = float(trajectory_ratio)
        self.trajectory_weight = float(trajectory_weight)
        self.history_frame_quota = max(0, int(history_frame_quota))
        self.history_quota_ivc_ratio = max(0.0, min(1.0, float(history_quota_ivc_ratio)))
        self.post_train_stabilize_t = int(post_train_stabilize_t)
        self.post_train_trajectory_scale = max(0.0, float(post_train_trajectory_scale))
        self.post_train_history_ivc_ratio = float(post_train_history_ivc_ratio)
        self.prune_sink = bool(prune_sink)
        self.prune_tail = bool(prune_tail)
        self.aggressive_all = bool(aggressive_all)
        self.sink_grid_decoupling = bool(sink_grid_decoupling)
        self.decoupled_sink_tokens = max(0, int(decoupled_sink_tokens))
        self.decoupled_sink_time_lag = max(0, int(decoupled_sink_time_lag))
        self.sink_time_mapping_mode = str(sink_time_mapping_mode)
        self.sink_time_clamp_min = max(0, int(sink_time_clamp_min))
        self.sink_time_clamp_max = max(self.sink_time_clamp_min, int(sink_time_clamp_max))
        self.history_time_mapping_mode = str(history_time_mapping_mode)
        self.history_relative_t_max = max(0, int(history_relative_t_max))
        self.history_time_soft_factor = max(0.0, min(1.0, float(history_time_soft_factor)))
        self.osc_full_kv_retention = bool(osc_full_kv_retention)
        self.periodic_peak_mask = bool(periodic_peak_mask)
        self.periodic_peak_period = max(1, int(periodic_peak_period))
        offs = [0, 1] if periodic_peak_offsets is None else periodic_peak_offsets
        normalized_offs = []
        for o in offs:
            try:
                normalized_offs.append(int(o) % self.periodic_peak_period)
            except (TypeError, ValueError):
                continue
        self.periodic_peak_offsets = sorted(set(normalized_offs)) if normalized_offs else [0, 1]
        self.periodic_peak_start_t = max(0, int(periodic_peak_start_t))
        self.periodic_peak_only_oscillating = bool(periodic_peak_only_oscillating)
        self.use_osc_frame_mode = bool(use_osc_frame_mode)
        self.phase_period = max(1, int(phase_period))
        self.phase_bucket_capacity_frames = max(0, int(phase_bucket_capacity_frames))
        self.local_tail_frames = max(1, int(local_tail_frames))
        self.phase_sink_for_osc_only = bool(phase_sink_for_osc_only)
        self.phase_sink_dynamic_rope = bool(phase_sink_dynamic_rope)
        lag_offs = [6] if osc_lag_offsets_frames is None else osc_lag_offsets_frames
        normalized_lag_offs = []
        for off in lag_offs:
            try:
                off_int = int(off)
            except (TypeError, ValueError):
                continue
            if off_int > 0:
                normalized_lag_offs.append(off_int)
        self.osc_lag_offsets_frames = sorted(set(normalized_lag_offs))
        self.use_osc_lag_mode = bool(use_osc_lag_mode and len(self.osc_lag_offsets_frames) > 0)
        self.osc_lag_history_frames = max(1, int(osc_lag_history_frames))
        if self.use_osc_lag_mode:
            self.osc_lag_history_frames = max(self.osc_lag_history_frames, max(self.osc_lag_offsets_frames) + 1)
        self.osc_lag_dynamic_rope = bool(osc_lag_dynamic_rope)
        self.disable_first_sink_for_osc_heads = bool(disable_first_sink_for_osc_heads)
        self.use_stable_head_policies = bool(use_stable_head_policies)
        self.stable_sink_frames = (
            None if stable_sink_frames is None else max(1, int(stable_sink_frames))
        )
        self.osc_sink_frames = (
            None if osc_sink_frames is None else max(1, int(osc_sink_frames))
        )
        self.stable_recent_frames = (
            None if stable_recent_frames is None else max(1, int(stable_recent_frames))
        )
        self.use_af_head_policies = bool(use_af_head_policies)
        # For attention visualization only:
        # - mapped: report RoPE-mapped time ids
        # - physical: report original physical frame ids
        mode = str(capture_frame_id_mode).strip().lower()
        if mode not in {"mapped", "physical"}:
            mode = "mapped"
        self.capture_frame_id_mode = mode
        self.readout_cache_enabled = bool(readout_cache_enabled)
        self._base_tail_len = self.tail_len
        max_cap = max(self.capacities) if self.capacities else 0
        min_cap = min(self.capacities) if self.capacities else 0
        self.max_capacity = max_cap
        self.head_labels = (
            config.get_layer_labels(layer_idx)
            if hasattr(config, "get_layer_labels")
            else [1] * self.num_heads
        )
        self.osc_head_flags = [int(lbl) == -1 for lbl in self.head_labels]
        self.af_head_groups = list(getattr(self, "af_group_row", [""] * self.num_heads))
        self.af_recent_frames_map = self._build_af_recent_frames_map(af_recent_frames_map)
        self.af_phase_bucket_map = self._build_af_phase_bucket_map(af_phase_bucket_map)
        self.af_lag_offsets_map = self._build_af_lag_offsets_map(af_lag_offsets_map)
        self.af_sink_frames_map = self._build_af_sink_frames_map(af_sink_frames_map)
        self.af_stride_enabled_map = self._build_af_stride_enabled_map(af_stride_enabled_map)
        self.label_recent_frames_map = self._build_label_recent_frames_map(label_recent_frames_map)
        self.label_phase_bucket_map = self._build_label_phase_bucket_map(label_phase_bucket_map)
        self.label_lag_offsets_map = self._build_label_lag_offsets_map(label_lag_offsets_map)
        self.label_sink_frames_map = self._build_label_sink_frames_map(label_sink_frames_map)
        self.label_stride_enabled_map = self._build_label_stride_enabled_map(label_stride_enabled_map)
        if self.sink_grid_decoupling and min_cap < max_cap:
            # In class-aware configs (e.g. -1 oscillating, 1 stable), reduced-capacity heads
            # are treated as oscillating heads and receive sink-grid decoupling.
            if any(self.osc_head_flags):
                self.decouple_head_flags = self.osc_head_flags.copy()
            else:
                self.decouple_head_flags = [cap < max_cap for cap in self.capacities]
        else:
            # If all capacities are equal, keep behavior backward compatible.
            self.decouple_head_flags = [self.sink_grid_decoupling] * self.num_heads

        self.static_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self.dynamic_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self.update_step = 0
        self.prompt_v: torch.Tensor | None = None
        self.last_flat_pos_ids: torch.Tensor | None = None
        # Workspace buffers for get_decoupled_flat_kv_and_frames (B2 optimization)
        self._ws_k_raw: torch.Tensor | None = None
        self._ws_k: torch.Tensor | None = None
        self._ws_v: torch.Tensor | None = None
        self._ws_frame_ids: torch.Tensor | None = None
        self._ws_cu_seqlens: torch.Tensor | None = None
        self._ws_rope_pos: torch.Tensor | None = None
        self._readout_cache_valid = False
        self._readout_cache_current_start: int | None = None
        self._readout_cache_sync_t_raw: int | None = None
        self._readout_cache_total_len = 0
        self._readout_cache_max_seqlen = 0
        self._readout_cache_frame_seqlen = 0
        self._readout_cache_tail_dirty = False
        self._readout_tail_write_through_valid = False
        self._readout_cache_shape_key: tuple | None = None
        self._last_readout_shape_key: tuple | None = None
        self._last_readout_anchor_shape_key: tuple | None = None
        self._readout_static_specs: list[tuple[int, int] | None] = [None] * (batch_size * num_heads)
        self._readout_tail_specs: list[tuple[int, int] | None] = [None] * (batch_size * num_heads)
        self._cuda_refresh_disabled = False
        self._cuda_refresh_desc_capacity = 0
        self._cuda_refresh_desc_key: tuple | None = None
        self._cuda_refresh_src_ptrs_k: torch.Tensor | None = None
        self._cuda_refresh_src_ptrs_v: torch.Tensor | None = None
        self._cuda_refresh_src_ptrs_pos: torch.Tensor | None = None
        self._cuda_refresh_offsets: torch.Tensor | None = None
        self._cuda_refresh_lengths: torch.Tensor | None = None
        self._cuda_refresh_flags: torch.Tensor | None = None
        self._cuda_refresh_dynamic_rope_t: torch.Tensor | None = None
        self._cuda_refresh_override_starts: torch.Tensor | None = None
        self._cuda_refresh_override_ends: torch.Tensor | None = None
        self._cuda_refresh_override_vals: torch.Tensor | None = None
        self._current_block_token_len: list[int] = [0] * (batch_size * num_heads)
        # Cached grid/frame info (avoids GPU→CPU sync on every call)
        self._grid_fhw: list[tuple[int, int, int]] | None = None
        self._frame_seqlen: int | None = self.frame_seq_length
        # Steady-state detection for CUDA Graph eligibility (G2)
        self._steady_state_reached: bool = False
        self._prev_cu_seqlens: tuple[int, ...] | None = None
        self._dyn_store_k: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_v: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_start: list[int] = [0] * (batch_size * num_heads)
        self._dyn_store_len: list[int] = [0] * (batch_size * num_heads)
        self._profile_enabled = False
        self._profile_stats = {
            "update_ms": 0.0,
            "collect_ms": 0.0,
            "pack_ms": 0.0,
            "rope_ms": 0.0,
            "cold_pack_count": 0.0,
            "refresh_pack_count": 0.0,
            "layout_reuse_count": 0.0,
            "layout_shape_changed_count": 0.0,
            "layout_anchor_changed_count": 0.0,
            "cuda_refresh_count": 0.0,
            "cuda_refresh_fallback_count": 0.0,
            "cuda_refresh_segment_count": 0.0,
            "cuda_refresh_total_len": 0.0,
            "readout_total_len": 0.0,
            "readout_max_seqlen": 0.0,
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
            "anchor_store_anchor_count": 0.0,
            "anchor_store_token_count": 0.0,
            "cpp_strategy_update_count": 0.0,
            "cpp_strategy_collect_count": 0.0,
            "cpp_strategy_fallback_count": 0.0,
            "cpp_strategy_fallback_cpu_count": 0.0,
            "cpp_strategy_fallback_extension_count": 0.0,
            "cpp_strategy_fallback_policy_count": 0.0,
            "cpp_strategy_fallback_shape_count": 0.0,
            "cpp_strategy_fallback_inactive_count": 0.0,
            "cpp_strategy_anchor_count": 0.0,
            "cpp_strategy_token_count": 0.0,
            "cpp_strategy_materialize_count": 0.0,
            "cpp_strategy_materialize_token_count": 0.0,
        }
        self.ivc_selector = ThreeDIVCSelector()
        self.semantic_selector = SemanticValueSelector()
        self._init_cyclic_anchor_storage()
        self._init_cpp_strategy_manager()
        # Initialize compositions' middle strategies if available
        if self.compositions_row is not None:
            num_seq = batch_size * num_heads
            for comp in self.compositions_row:
                if comp.has_middle:
                    comp.reset_all(num_seq)

        # Optional C++ shadow validator gated by PYRAMIDKV_USE_CPP_PACK=1. Lazy-
        # built on first update() once _frame_seqlen is known. Default off.
        self._shadow = None

    def _ensure_shadow(self) -> None:
        """Lazy-build the C++ shadow manager once frame_seqlen is known.

        No-op when PYRAMIDKV_USE_CPP_PACK is unset or when batch_size != 1
        (the shadow only supports B=1)."""
        if self._shadow is not None:
            return
        if self.batch_size != 1:
            return
        if self._frame_seqlen is None or self._frame_seqlen <= 0:
            return
        from pyramidkv._cpp_shadow import maybe_attach_shadow
        # Per-head sink/recent counts vary by composition (osc=1 sink, stable=3
        # sink, etc.) — pool capacity must fit the LARGEST or stable heads
        # silently truncate, leading to missing frames in cpp pack output.
        comp_sink = 0
        comp_recent = 0
        if self.compositions_row is not None:
            for comp in self.compositions_row:
                comp_sink = max(comp_sink, int(comp.sink_frames))
                comp_recent = max(comp_recent, int(comp.recent_frames))
        max_sink = max(1, comp_sink, int(self.sink_len // self._frame_seqlen)) + 1
        max_recent = max(1, comp_recent, int(self._base_tail_len // self._frame_seqlen)) + 1
        device = self._dyn_store_k[0].device if self._dyn_store_k and self._dyn_store_k[0] is not None else torch.device("cuda:0")
        self._shadow = maybe_attach_shadow(
            layer_idx=self.layer_idx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            frame_seqlen=self._frame_seqlen,
            max_sink=max_sink,
            max_recent=max_recent,
            device=device,
        )

    def _mirror_to_shadow(self) -> None:
        """Push current Python state into the shadow manager pools (if active)."""
        self._ensure_shadow()
        if self._shadow is None:
            return
        self._shadow.mirror_after_update(
            static_k=self.static_k, dynamic_k=self.dynamic_k,
            static_v=self.static_v, dynamic_v=self.dynamic_v,
        )

    def _shadow_swap_v(self, v_flat_slice: torch.Tensor, spec) -> None:
        """When PYRAMIDKV_USE_CPP_PACK_OUTPUT=1, replace v_flat with C++ pack V.

        Mirrors middle anchors (sink+recent already mirrored via update())
        then runs C++ plan+pack and copies the V output back into v_flat.
        K remains Python (RoPE unchanged). This is the perf-target path:
        skips the Python V scatter logic entirely on the next call.
        """
        if self._shadow is None:
            return
        from pyramidkv._cpp_shadow import cpp_pack_output_enabled
        if not cpp_pack_output_enabled():
            return
        self._shadow.mirror_middle_from_spec_and_vflat(spec, v_flat_slice)
        cpp_v = self._shadow.cpp_pack_v()
        # cpp_v shape may differ from python by partial merge anchor; use min.
        n = min(cpp_v.shape[0], v_flat_slice.shape[0])
        v_flat_slice[:n].copy_(cpp_v[:n])

    def _shadow_assert_v(self, v_flat_slice: torch.Tensor, spec=None) -> None:
        """Run cpp pack and compare V; mirror middle anchors from spec first.

        When ``spec`` is provided, its middle segments are mirrored into the
        manager middle pools so the C++ pack output includes them. With sink+
        recent already mirrored at update time, this completes the per-head
        layout and the V output should bit-match Python's pack.

        Without ``spec`` (legacy callers), the assert is skipped on configs
        that use middle strategies to avoid spurious mismatches.
        """
        if self._shadow is None:
            return
        if spec is not None:
            # Unified middle mirror: covers both spec.segments (kind=middle)
            # and spec.cpp_segments (anchors written by _cpp_strategy_manager
            # into v_flat at runtime). Reads cpp_segment data from the just-
            # populated v_flat slice — sink/recent already came from update().
            self._shadow.mirror_middle_from_spec_and_vflat(spec, v_flat_slice)
        elif self.compositions_row is not None:
            for comp in self.compositions_row:
                if comp.has_middle:
                    return
        try:
            self._shadow.assert_v_matches(v_flat_slice)
        except AssertionError as exc:
            # Warn only on the FIRST mismatch per layer (not every block * pass);
            # repeat mismatches are captured in _shadow.mismatch_count for telemetry.
            if self._shadow.mismatch_count == 1:
                import warnings
                warnings.warn(f"[PYRAMIDKV_SHADOW] V mismatch on layer {self.layer_idx}: {exc}")

    def set_profile_enabled(self, enabled: bool) -> None:
        self._profile_enabled = bool(enabled)
        if not enabled:
            self.reset_profile_stats()

    def reset_profile_stats(self) -> None:
        self._profile_stats = {
            "update_ms": 0.0,
            "collect_ms": 0.0,
            "pack_ms": 0.0,
            "rope_ms": 0.0,
            "cold_pack_count": 0.0,
            "refresh_pack_count": 0.0,
            "layout_reuse_count": 0.0,
            "layout_shape_changed_count": 0.0,
            "layout_anchor_changed_count": 0.0,
            "cuda_refresh_count": 0.0,
            "cuda_refresh_fallback_count": 0.0,
            "cuda_refresh_segment_count": 0.0,
            "cuda_refresh_total_len": 0.0,
            "readout_total_len": 0.0,
            "readout_max_seqlen": 0.0,
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
            "anchor_store_anchor_count": 0.0,
            "anchor_store_token_count": 0.0,
            "cpp_strategy_update_count": 0.0,
            "cpp_strategy_collect_count": 0.0,
            "cpp_strategy_fallback_count": 0.0,
            "cpp_strategy_fallback_cpu_count": 0.0,
            "cpp_strategy_fallback_extension_count": 0.0,
            "cpp_strategy_fallback_policy_count": 0.0,
            "cpp_strategy_fallback_shape_count": 0.0,
            "cpp_strategy_fallback_inactive_count": 0.0,
            "cpp_strategy_anchor_count": 0.0,
            "cpp_strategy_token_count": 0.0,
            "cpp_strategy_materialize_count": 0.0,
            "cpp_strategy_materialize_token_count": 0.0,
        }

    def pop_profile_stats(self) -> dict[str, float]:
        self._drain_anchor_store_strategy_stats()
        self._drain_cpp_strategy_stats()
        stats = dict(self._profile_stats)
        self.reset_profile_stats()
        return stats

    def _record_profile(self, key: str, start_time: float) -> None:
        if not self._profile_enabled:
            return
        self._profile_stats[key] += (perf_counter() - start_time) * 1000.0

    def _drain_anchor_store_strategy_stats(self) -> None:
        if self.compositions_row is None:
            return
        for comp in self.compositions_row:
            for strategy in getattr(comp, "middle_strategies", ()):
                pop_stats = getattr(strategy, "pop_anchor_store_stats", None)
                if pop_stats is None:
                    continue
                for key, value in pop_stats().items():
                    if key in self._profile_stats:
                        self._profile_stats[key] += float(value)

    def _init_cpp_strategy_manager(self) -> None:
        self._cpp_strategy_requested = cpp_strategy_requested()
        self._cpp_strategy_manager: CppStrategyManager | None = None
        self._cpp_strategy_supported_heads: list[bool] = [False] * self.num_heads
        self._cpp_strategy_has_middle = False
        self._cpp_strategy_has_unsupported_middle = False
        if not self._cpp_strategy_requested or self.compositions_row is None:
            return
        policies, supported = compile_cpp_strategy_policies(self.compositions_row)
        self._cpp_strategy_supported_heads = supported[: self.num_heads] + [False] * max(0, self.num_heads - len(supported))
        self._cpp_strategy_has_middle = any(
            bool(getattr(comp, "has_middle", False)) for comp in self.compositions_row
        )
        self._cpp_strategy_has_unsupported_middle = any(
            bool(getattr(comp, "has_middle", False))
            and not self._cpp_strategy_supported_heads[idx]
            for idx, comp in enumerate(self.compositions_row[: self.num_heads])
        )
        if any(self._cpp_strategy_supported_heads):
            self._cpp_strategy_manager = CppStrategyManager(
                policies,
                num_seq=self.batch_size * self.num_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                require_cuda=True,
                require_extension=True,
            )

    def _drain_cpp_strategy_stats(self) -> None:
        manager = self._cpp_strategy_manager
        if manager is None:
            return
        for key, value in manager.pop_stats().items():
            if key in self._profile_stats:
                self._profile_stats[key] += float(value)

    def _record_cpp_strategy_fallback(self, reason: str) -> None:
        self._profile_stats["cpp_strategy_fallback_count"] += 1.0
        reason_key = f"cpp_strategy_fallback_{reason}_count"
        if reason_key in self._profile_stats:
            self._profile_stats[reason_key] += 1.0

    def _cpp_strategy_head_supported(self, head_idx: int) -> bool:
        return (
            self._cpp_strategy_requested
            and 0 <= head_idx < len(self._cpp_strategy_supported_heads)
            and bool(self._cpp_strategy_supported_heads[head_idx])
            and self._cpp_strategy_manager is not None
        )

    def _try_cpp_strategy_update_all(
        self,
        *,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        pos_flat: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
    ) -> bool:
        manager = self._cpp_strategy_manager
        if not self._cpp_strategy_requested or manager is None or not manager.has_supported_middle:
            if self._cpp_strategy_requested and self._cpp_strategy_has_middle:
                self._record_cpp_strategy_fallback("policy")
            return False
        if not k_flat.is_cuda or not v_flat.is_cuda or not pos_flat.is_cuda:
            self._record_cpp_strategy_fallback("cpu")
            return False
        if frame_seqlen <= 0 or k_flat.shape[1] < frame_seqlen or k_flat.shape[1] % frame_seqlen != 0:
            self._record_cpp_strategy_fallback("shape")
            return False
        if not manager.usable_for(k_flat):
            self._record_cpp_strategy_fallback("extension")
            return False
        try:
            updated = manager.update_all(
                k_flat=k_flat,
                v_flat=v_flat,
                pos_flat=pos_flat,
                frame_seqlen=frame_seqlen,
                current_t=current_t,
            )
        except ValueError:
            raise
        except Exception:
            self._record_cpp_strategy_fallback("extension")
            return False
        if not updated:
            self._record_cpp_strategy_fallback("inactive")
        if self._cpp_strategy_has_unsupported_middle:
            self._record_cpp_strategy_fallback("policy")
        return updated

    def _try_cpp_strategy_collect(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        sync_t_raw: int,
        tail_min_t: int,
        sink_max_t: int,
    ) -> list | None:
        manager = self._cpp_strategy_manager
        if not self._cpp_strategy_requested:
            return None
        if manager is None or not self._cpp_strategy_head_supported(head_idx):
            self._record_cpp_strategy_fallback("policy")
            return None
        anchors = manager.collect(
            seq_idx=seq_idx,
            head_idx=head_idx,
            current_t=sync_t_raw,
            recent_min_t=tail_min_t,
            sink_max_t=sink_max_t,
        )
        if anchors is None:
            self._record_cpp_strategy_fallback("inactive")
        return anchors

    def _try_cpp_strategy_count(
        self,
        *,
        seq_idx: int,
        head_idx: int,
        sync_t_raw: int,
        tail_min_t: int,
        sink_max_t: int,
    ):
        manager = self._cpp_strategy_manager
        if not self._cpp_strategy_requested:
            return None
        if manager is None or not self._cpp_strategy_head_supported(head_idx):
            self._record_cpp_strategy_fallback("policy")
            return None
        count = manager.count_anchors(
            seq_idx=seq_idx,
            head_idx=head_idx,
            current_t=sync_t_raw,
            recent_min_t=tail_min_t,
            sink_max_t=sink_max_t,
        )
        if count is None:
            self._record_cpp_strategy_fallback("inactive")
        return count

    def _init_cyclic_anchor_storage(self) -> None:
        num_seq = self.batch_size * self.num_heads
        # per (batch, head) -> per phase bucket -> deque of frame-level anchors
        self.cyclic_buckets: list[list[deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]]] = [
            [deque() for _ in range(self.phase_period)] for _ in range(num_seq)
        ]
        # per (batch, head) -> OrderedDict[t -> frame-level anchor]
        self.lag_anchor_frames: list[OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]] = [
            OrderedDict() for _ in range(num_seq)
        ]

    def _sync_dynamic_views(self, idx: int) -> None:
        store_k = self._dyn_store_k[idx]
        if store_k is None:
            self.dynamic_k[idx] = None
            self.dynamic_v[idx] = None
            self.dynamic_pos[idx] = None
            return

        start = self._dyn_store_start[idx]
        end = start + self._dyn_store_len[idx]
        self.dynamic_k[idx] = self._dyn_store_k[idx][start:end]
        self.dynamic_v[idx] = self._dyn_store_v[idx][start:end]
        self.dynamic_pos[idx] = self._dyn_store_pos[idx][start:end]

    def _set_dynamic_store(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        reserve_extra: int = 0,
    ) -> None:
        reserve = max(0, int(reserve_extra))
        length = int(k_seq.shape[0])
        capacity = max(length, length + reserve)
        self._dyn_store_k[idx] = torch.empty((capacity, self.head_dim), device=k_seq.device, dtype=k_seq.dtype)
        self._dyn_store_v[idx] = torch.empty((capacity, self.head_dim), device=v_seq.device, dtype=v_seq.dtype)
        self._dyn_store_pos[idx] = torch.empty((capacity, 3), device=pos_seq.device, dtype=pos_seq.dtype)
        if length > 0:
            self._dyn_store_k[idx][:length] = k_seq
            self._dyn_store_v[idx][:length] = v_seq
            self._dyn_store_pos[idx][:length] = pos_seq
        self._dyn_store_start[idx] = 0
        self._dyn_store_len[idx] = length
        self._sync_dynamic_views(idx)

    def _set_dynamic_empty(
        self,
        idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        empty_k = torch.empty((0, self.head_dim), device=device, dtype=dtype)
        empty_v = torch.empty((0, self.head_dim), device=device, dtype=dtype)
        empty_pos = torch.empty((0, 3), device=device, dtype=torch.long)
        self._set_dynamic_store(idx, empty_k, empty_v, empty_pos, reserve_extra=0)

    def _ensure_dynamic_capacity(
        self,
        idx: int,
        append_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        store_k = self._dyn_store_k[idx]
        if store_k is None:
            alloc = max(append_len + max(16, append_len // 2), append_len)
            self._dyn_store_k[idx] = torch.empty((alloc, self.head_dim), device=device, dtype=dtype)
            self._dyn_store_v[idx] = torch.empty((alloc, self.head_dim), device=device, dtype=dtype)
            self._dyn_store_pos[idx] = torch.empty((alloc, 3), device=device, dtype=torch.long)
            self._dyn_store_start[idx] = 0
            self._dyn_store_len[idx] = 0
            return

        start = self._dyn_store_start[idx]
        length = self._dyn_store_len[idx]
        end = start + length
        if store_k.shape[0] - end >= append_len:
            return

        total_free = store_k.shape[0] - length
        if start > 0 and total_free >= append_len:
            if length > 0:
                self._dyn_store_k[idx][:length] = self._dyn_store_k[idx][start:end].clone()
                self._dyn_store_v[idx][:length] = self._dyn_store_v[idx][start:end].clone()
                self._dyn_store_pos[idx][:length] = self._dyn_store_pos[idx][start:end].clone()
            self._dyn_store_start[idx] = 0
            return

        new_capacity = max(length + append_len, int(store_k.shape[0] * 1.5), length + append_len + 64)
        new_k = torch.empty((new_capacity, self.head_dim), device=device, dtype=dtype)
        new_v = torch.empty((new_capacity, self.head_dim), device=device, dtype=dtype)
        new_pos = torch.empty((new_capacity, 3), device=device, dtype=torch.long)
        if length > 0:
            new_k[:length] = self._dyn_store_k[idx][start:end]
            new_v[:length] = self._dyn_store_v[idx][start:end]
            new_pos[:length] = self._dyn_store_pos[idx][start:end]
        self._dyn_store_k[idx] = new_k
        self._dyn_store_v[idx] = new_v
        self._dyn_store_pos[idx] = new_pos
        self._dyn_store_start[idx] = 0

    def _append_dynamic(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
    ) -> None:
        append_len = int(k_seq.shape[0])
        if append_len <= 0:
            self._sync_dynamic_views(idx)
            return
        self._ensure_dynamic_capacity(idx, append_len, device=k_seq.device, dtype=k_seq.dtype)
        start = self._dyn_store_start[idx]
        length = self._dyn_store_len[idx]
        end = start + length
        self._dyn_store_k[idx][end:end + append_len] = k_seq
        self._dyn_store_v[idx][end:end + append_len] = v_seq
        self._dyn_store_pos[idx][end:end + append_len] = pos_seq
        self._dyn_store_len[idx] = length + append_len
        self._sync_dynamic_views(idx)

    def _overwrite_dynamic_tail(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
    ) -> None:
        overwrite_len = int(k_seq.shape[0])
        if overwrite_len <= 0:
            self._sync_dynamic_views(idx)
            return
        if self._dyn_store_k[idx] is None or self._dyn_store_len[idx] < overwrite_len:
            self._set_dynamic_store(idx, k_seq, v_seq, pos_seq, reserve_extra=max(16, overwrite_len // 2))
            return
        start = self._dyn_store_start[idx]
        end = start + self._dyn_store_len[idx]
        tail_start = end - overwrite_len
        self._dyn_store_k[idx][tail_start:end] = k_seq
        self._dyn_store_v[idx][tail_start:end] = v_seq
        self._dyn_store_pos[idx][tail_start:end] = pos_seq
        self._sync_dynamic_views(idx)

    def _keep_dynamic_suffix(self, idx: int, keep_len: int) -> None:
        if self._dyn_store_k[idx] is None:
            return
        keep = max(0, int(keep_len))
        length = self._dyn_store_len[idx]
        if keep >= length:
            self._sync_dynamic_views(idx)
            return
        self._dyn_store_start[idx] += length - keep
        self._dyn_store_len[idx] = keep
        self._sync_dynamic_views(idx)

    @staticmethod
    def _normalize_af_group_key(key: object) -> str:
        raw = str(key).strip().upper()
        if not raw:
            return ""
        if raw in {"A", "B", "C", "D", "E", "F"}:
            return raw
        if raw.startswith(("A_", "B_", "C_", "D_", "E_", "F_")):
            return raw[0]
        return ""

    @staticmethod
    def _normalize_label_key(key: object) -> str:
        raw = str(key).strip()
        if not raw:
            return ""
        try:
            return str(int(raw))
        except (TypeError, ValueError):
            return raw

    @staticmethod
    def _map_items(user_map: Mapping | None):
        if not isinstance(user_map, Mapping):
            return ()
        return user_map.items()

    @staticmethod
    def _as_sequence(value: object) -> list[object]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _build_label_recent_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_phase_bucket_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(0, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_lag_offsets_map(self, user_map: Mapping | None) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            vals = self._as_sequence(val)
            offs: list[int] = []
            for item in vals:
                try:
                    off = int(item)
                except (TypeError, ValueError):
                    continue
                if off > 0:
                    offs.append(off)
            out[label] = sorted(set(offs))
        return out

    def _build_label_sink_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_stride_enabled_map(self, user_map: Mapping | None) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            out[label] = bool(val)
        return out

    def _build_af_recent_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out = {"A": 4, "B": 3, "C": 4, "D": 3, "E": 2, "F": 5}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_phase_bucket_map(self, user_map: Mapping | None) -> dict[str, int]:
        out = {"A": 0, "B": 1, "C": 1, "D": 1, "E": 0, "F": 0}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(0, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_lag_offsets_map(self, user_map: Mapping | None) -> dict[str, list[int]]:
        out = {"A": [], "B": [], "C": [], "D": [6], "E": [], "F": []}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            offs = []
            vals = self._as_sequence(val)
            for item in vals:
                try:
                    off = int(item)
                except (TypeError, ValueError):
                    continue
                if off > 0:
                    offs.append(off)
            out[group] = sorted(set(offs))
        return out

    def _build_af_sink_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_stride_enabled_map(self, user_map: Mapping | None) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            out[group] = bool(val)
        return out

    def _af_group(self, head_idx: int) -> str:
        if head_idx < 0 or head_idx >= len(self.af_head_groups):
            return ""
        return self._normalize_af_group_key(self.af_head_groups[head_idx])

    def _head_label_key(self, head_idx: int) -> str:
        if head_idx < 0 or head_idx >= len(self.head_labels):
            return ""
        return self._normalize_label_key(self.head_labels[head_idx])

    def _head_sink_frames(self, head_idx: int) -> int | None:
        label = self._head_label_key(head_idx)
        if label in self.label_sink_frames_map:
            return max(1, int(self.label_sink_frames_map[label]))
        if self.use_af_head_policies and self.af_sink_frames_map:
            group = self._af_group(head_idx)
            if group in self.af_sink_frames_map:
                return max(1, int(self.af_sink_frames_map[group]))
        if self.osc_head_flags[head_idx]:
            if self.osc_sink_frames is not None:
                return max(1, int(self.osc_sink_frames))
            return None
        if self.stable_sink_frames is not None:
            return max(1, int(self.stable_sink_frames))
        return None

    def _has_explicit_recent_override(self, head_idx: int) -> bool:
        label = self._head_label_key(head_idx)
        if label in self.label_recent_frames_map:
            return True
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_recent_frames_map:
                return True
        return self.stable_recent_frames is not None

    def _head_recent_frames(self, head_idx: int) -> int:
        label = self._head_label_key(head_idx)
        if label in self.label_recent_frames_map:
            return max(1, int(self.label_recent_frames_map[label]))
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_recent_frames_map:
                return max(1, int(self.af_recent_frames_map[group]))
        if not self.osc_head_flags[head_idx] and self.stable_recent_frames is not None:
            return max(1, int(self.stable_recent_frames))
        return self.local_tail_frames

    def _head_phase_bucket_capacity(self, head_idx: int) -> int:
        label = self._head_label_key(head_idx)
        if label in self.label_phase_bucket_map:
            return max(0, int(self.label_phase_bucket_map[label]))
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_phase_bucket_map:
                return max(0, int(self.af_phase_bucket_map[group]))
        return self.phase_bucket_capacity_frames

    def _head_lag_offsets(self, head_idx: int) -> list[int]:
        label = self._head_label_key(head_idx)
        if label in self.label_lag_offsets_map:
            return list(self.label_lag_offsets_map[label])
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_lag_offsets_map:
                return list(self.af_lag_offsets_map[group])
            return []
        if not self.use_osc_lag_mode:
            return []
        return list(self.osc_lag_offsets_frames)

    def _is_phase_sink_head(self, head_idx: int) -> bool:
        if self.use_af_head_policies:
            return self._head_phase_bucket_capacity(head_idx) > 0
        if not self.phase_sink_for_osc_only:
            return True
        return self.osc_head_flags[head_idx]

    def _update_cyclic_anchors(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        t_start: int | None = None,
    ) -> None:
        if not self.use_osc_frame_mode:
            return
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return
        head_idx = idx % self.num_heads
        bucket_cap = self._head_phase_bucket_capacity(head_idx)
        if bucket_cap <= 0:
            return
        if not self._is_phase_sink_head(head_idx):
            return

        num_frames = k_seq.shape[0] // frame_seqlen
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            frame_pos = pos_seq[start:end]
            if frame_pos.shape[0] != frame_seqlen:
                continue
            t_val = int(t_start + frame_idx) if t_start is not None else frame_idx
            phase = t_val % self.phase_period
            bucket = self.cyclic_buckets[idx][phase]
            bucket.append(
                (
                    k_seq[start:end].clone(),
                    v_seq[start:end].clone(),
                    frame_pos.clone(),
                    t_val,
                )
            )
            while len(bucket) > bucket_cap:
                bucket.popleft()

    def _update_lag_anchors(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        t_start: int | None = None,
    ) -> None:
        if not self.use_osc_lag_mode and not self.use_af_head_policies:
            return
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return
        head_idx = idx % self.num_heads
        if not self._is_phase_sink_head(head_idx):
            return
        lag_offsets = self._head_lag_offsets(head_idx)
        if len(lag_offsets) == 0:
            return

        anchors = self.lag_anchor_frames[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            frame_pos = pos_seq[start:end]
            if frame_pos.shape[0] != frame_seqlen:
                continue
            t_val = int(t_start + frame_idx) if t_start is not None else frame_idx
            if t_val in anchors:
                del anchors[t_val]
            anchors[t_val] = (
                k_seq[start:end].clone(),
                v_seq[start:end].clone(),
                frame_pos.clone(),
                t_val,
            )
            while len(anchors) > self.osc_lag_history_frames:
                anchors.popitem(last=False)

    @staticmethod
    def _find_anchor_by_t(
        anchors,
        target_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        if isinstance(anchors, Mapping):
            return anchors.get(target_t)
        for item in reversed(anchors):
            if item[3] == target_t:
                return item
        return None

    def _stable_strategy_kind(self, head_idx: int) -> str | None:
        label = self._head_label_key(head_idx)
        if label in self.label_stride_enabled_map:
            if self.osc_head_flags[head_idx]:
                return None
            return "stride" if self.label_stride_enabled_map[label] else "recent_only"
        # AF per-group stride override
        if self.use_af_head_policies and self.af_stride_enabled_map:
            group = self._af_group(head_idx)
            if group and self.af_stride_enabled_map.get(group, False):
                return "stride"
        if not self.use_stable_head_policies:
            return None
        if self.osc_head_flags[head_idx]:
            return None
        if getattr(self, "policies_row", None) is None or head_idx >= len(self.policies_row):
            return None
        impl = self.policies_row[head_idx]
        kind = getattr(impl, "policy_type", None)
        if kind in {"stride", "recent_only"}:
            return kind
        return None

    def _stride_frame_ids(self, head_idx: int, kind: str, num_frames: int) -> list[int]:
        if num_frames <= 0:
            return []

        # Optional explicit stable-tail policy:
        # - stride: source-like periodic keep (start at frame idx 3, every phase_period)
        #            + recent K
        # - recent_only: recent K only
        # This supports patterns like "sink3 + every6 (before recent) + recent4".
        if self._has_explicit_recent_override(head_idx):
            keep: list[int] = []
            recent = min(num_frames, self._head_recent_frames(head_idx))
            recent_start = max(0, num_frames - recent)
            keep.extend(range(recent_start, num_frames))
            if kind == "stride":
                step = max(1, int(self.phase_period))
                f_idx = 0
                while f_idx < recent_start:
                    keep.append(f_idx)
                    f_idx += step
            return sorted(set(keep))

        # Backward-compatible default stable policy behavior.
        keep: list[int] = []
        head_frames = min(3, num_frames)
        for f_idx in range(head_frames):
            keep.append(f_idx)
        last = num_frames - 1
        if kind == "stride" and num_frames > 4:
            f_idx = 3
            while f_idx < last:
                keep.append(f_idx)
                f_idx += 6
        if last not in keep:
            keep.append(last)
        return sorted(set(keep))

    def _apply_stable_strategy(
        self,
        head_idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kind = self._stable_strategy_kind(head_idx)
        if kind is None:
            return k_seq, v_seq, pos_seq
        if frame_seqlen <= 0 or pos_seq.shape[0] < frame_seqlen:
            return k_seq, v_seq, pos_seq
        # _build_pos_ids lays out tokens frame-major (rows [k*frame_seqlen,
        # (k+1)*frame_seqlen) share time t_k), and compaction preserves
        # frame-major layout. So num_frames is derivable from shape alone —
        # no need for torch.unique (which is a host-device sync).
        if pos_seq.shape[0] % frame_seqlen != 0:
            return k_seq, v_seq, pos_seq
        num_frames = pos_seq.shape[0] // frame_seqlen
        if num_frames <= 1:
            return k_seq, v_seq, pos_seq

        keep_f = self._stride_frame_ids(head_idx, kind, num_frames=num_frames)
        if not keep_f or len(keep_f) >= num_frames:
            return k_seq, v_seq, pos_seq

        # Direct frame-major indexing: rows = keep_f[i]*frame_seqlen + [0..frame_seqlen)
        # Build keep_f_tensor sync-free via pinned memory + non_blocking H2D
        # (CUDA Graph capture requires zero host syncs in the forward path).
        device = pos_seq.device
        keep_f_cpu = torch.tensor(keep_f, dtype=torch.long).pin_memory()
        keep_f_tensor = torch.empty(len(keep_f), dtype=torch.long, device=device)
        keep_f_tensor.copy_(keep_f_cpu, non_blocking=True)
        offsets = torch.arange(frame_seqlen, dtype=torch.long, device=device)
        keep_idx = (keep_f_tensor.unsqueeze(1) * frame_seqlen + offsets.unsqueeze(0)).flatten()
        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

    def _map_sink_time(self, sync_t_raw: int) -> int:
        return map_sink_time(
            sync_t_raw,
            sink_time_mapping_mode=self.sink_time_mapping_mode,
            sink_time_clamp_min=self.sink_time_clamp_min,
            sink_time_clamp_max=self.sink_time_clamp_max,
            decoupled_sink_time_lag=self.decoupled_sink_time_lag,
        )

    def _map_dynamic_pos_time(self, dyn_pos: torch.Tensor, current_t: int) -> torch.Tensor:
        return map_dynamic_pos_time(
            dyn_pos,
            current_t=current_t,
            history_time_mapping_mode=self.history_time_mapping_mode,
            history_relative_t_max=self.history_relative_t_max,
            history_time_soft_factor=self.history_time_soft_factor,
        )

    def _capture_sink_if_needed(
        self,
        idx: int,
        head_idx: int,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        p_in: torch.Tensor,
        current_start: int | None,
        overwrite: bool,
        freqs: torch.Tensor | None = None,
        prompt_head: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For T2V decoupling mode, keep the first sink_len tokens in a static bucket (raw, unrotated).
        The sink can be refreshed on overwrite at the same start position (e.g. clean pass).
        """
        if self.is_i2v:
            return k_in, v_in, p_in
        if not self.sink_grid_decoupling:
            return k_in, v_in, p_in
        head_idx = idx % self.num_heads
        if not self.decouple_head_flags[head_idx]:
            return k_in, v_in, p_in
        frame_seqlen = self._frame_seqlen or 0

        sink_len = self.sink_len
        if frame_seqlen > 0:
            sink_frames = self._head_sink_frames(head_idx)
            if sink_frames is not None:
                sink_len = sink_frames * frame_seqlen

        if sink_len <= 0:
            return k_in, v_in, p_in
        if current_start is None or current_start != 0:
            return k_in, v_in, p_in
        if self.disable_first_sink_for_osc_heads and self.osc_head_flags[head_idx]:
            return k_in, v_in, p_in

        take = min(sink_len, k_in.shape[0])
        if take <= 0:
            return k_in, v_in, p_in

        sink_k = k_in[:take]
        sink_v = v_in[:take]
        sink_p = p_in[:take]
        if self.decoupled_sink_tokens > 0 and take > self.decoupled_sink_tokens:
            budget = self.decoupled_sink_tokens
            if freqs is not None:
                sink_mask = self._ranked_select(
                    pos_seg=sink_p,
                    v_seg=sink_v,
                    budget=budget,
                    freqs=freqs,
                    prompt_head=prompt_head,
                    apply_selection=True,
                )
                select_idx = torch.nonzero(sink_mask, as_tuple=False).squeeze(1).sort().values
            else:
                select_idx = torch.linspace(
                    0,
                    take - 1,
                    steps=budget,
                    device=k_in.device,
                ).round().to(torch.long)
            sink_k = sink_k[select_idx]
            sink_v = sink_v[select_idx]
            sink_p = sink_p[select_idx]

        should_write_static = (self.static_k[idx] is None) or overwrite
        if should_write_static:
            self.static_k[idx] = sink_k.clone()
            self.static_v[idx] = sink_v.clone()
            self.static_pos[idx] = sink_p.clone()

        return k_in[take:], v_in[take:], p_in[take:]

    def _periodic_peak_local_mask(self, t_vals: torch.Tensor) -> torch.Tensor:
        if not self.periodic_peak_mask or t_vals.numel() == 0:
            return torch.zeros_like(t_vals, dtype=torch.bool)
        valid = t_vals >= self.periodic_peak_start_t
        rel = (t_vals - self.periodic_peak_start_t).remainder(self.periodic_peak_period)
        mask = torch.zeros_like(valid, dtype=torch.bool)
        for off in self.periodic_peak_offsets:
            mask |= (rel == off)
        return valid & mask

    def set_prompt_values(self, prompt_v: torch.Tensor | None) -> None:
        self.prompt_v = prompt_v

    def _effective_selection_params(self, pos_seg: torch.Tensor, current_t: int | None = None) -> tuple[float, float, float]:
        traj_ratio = self.trajectory_ratio
        traj_weight = self.trajectory_weight
        quota_ivc_ratio = self.history_quota_ivc_ratio
        if pos_seg.numel() == 0:
            return traj_ratio, traj_weight, quota_ivc_ratio

        if self.post_train_stabilize_t >= 0:
            if current_t is None:
                current_t = int(pos_seg[:, 0].max().item())
            if current_t >= self.post_train_stabilize_t:
                traj_ratio = traj_ratio * self.post_train_trajectory_scale
                traj_weight = traj_weight * self.post_train_trajectory_scale
                if self.post_train_history_ivc_ratio >= 0.0:
                    quota_ivc_ratio = max(quota_ivc_ratio, min(1.0, self.post_train_history_ivc_ratio))
        return traj_ratio, traj_weight, quota_ivc_ratio

    def reset(self):
        super().reset()
        self.static_pos = [None] * (self.batch_size * self.num_heads)
        self.dynamic_pos = [None] * (self.batch_size * self.num_heads)
        self.update_step = 0
        self.last_flat_pos_ids = None
        # Clear workspace buffers
        self._ws_k_raw = None
        self._ws_k = None
        self._ws_v = None
        self._ws_frame_ids = None
        self._ws_cu_seqlens = None
        self._ws_rope_pos = None
        num_seq = self.batch_size * self.num_heads
        self._current_block_token_len = [0] * num_seq
        self._dyn_store_k = [None] * num_seq
        self._dyn_store_v = [None] * num_seq
        self._dyn_store_pos = [None] * num_seq
        self._dyn_store_start = [0] * num_seq
        self._dyn_store_len = [0] * num_seq
        self._invalidate_readout_cache()
        self.reset_profile_stats()
        self._init_cyclic_anchor_storage()
        if self._cpp_strategy_manager is not None:
            self._cpp_strategy_manager.reset(self.batch_size * self.num_heads)
        # Reset compositions' middle strategies if available
        if self.compositions_row is not None:
            num_seq = self.batch_size * self.num_heads
            for comp in self.compositions_row:
                if comp.has_middle:
                    comp.reset_all(num_seq)

    def _update_steady_state(self, cu_seqlens_k: torch.Tensor) -> None:
        """Steady-state detection via pure-Python _dyn_store_len comparison.

        _dyn_store_len is a list[int] (pure Python) maintained by update(),
        so this check performs zero GPU→CPU sync. Steady state = two
        consecutive readouts observe identical per-head dynamic lengths.

        cu_seqlens_k is kept in the signature for API compatibility but
        is NOT read (would sync). The _dyn_store_len snapshot is a
        strict superset of information in cu_seqlens_k for our purposes.
        """
        if self._steady_state_reached:
            return
        # Take a Python-level snapshot (fast, no sync)
        current = tuple(self._dyn_store_len)
        if self._prev_cu_seqlens is not None and current == self._prev_cu_seqlens:
            self._steady_state_reached = True
        self._prev_cu_seqlens = current

    def _build_pos_ids(self, grid_sizes: torch.Tensor, seq_len: int, current_start: int, device: torch.device) -> torch.Tensor:
        pos = torch.zeros((self.batch_size, seq_len, 3), dtype=torch.long, device=device)
        if self._grid_fhw is None:
            self._grid_fhw = [tuple(int(x) for x in grid_sizes[b].tolist()) for b in range(self.batch_size)]
        for b in range(self.batch_size):
            f, h, w = self._grid_fhw[b]
            frame_seqlen = max(1, h * w)
            start_frame = current_start // frame_seqlen
            n_valid = min(seq_len, f * h * w)
            idx = torch.arange(n_valid, device=device, dtype=torch.long)
            t = idx // frame_seqlen + start_frame
            y = (idx % frame_seqlen) // max(1, w)
            x = idx % max(1, w)
            pos[b, :n_valid] = torch.stack([t, y, x], dim=-1)
        return pos

    def _segment_indices(
        self,
        length: int,
        device: torch.device,
        sink_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective_sink_len = self.sink_len if sink_len is None else max(0, int(sink_len))
        sink_end = min(effective_sink_len, length)
        tail_start = max(sink_end, length - self.tail_len)

        sink_idx = torch.arange(0, sink_end, device=device, dtype=torch.long)
        history_idx = torch.arange(sink_end, tail_start, device=device, dtype=torch.long)
        tail_idx = torch.arange(tail_start, length, device=device, dtype=torch.long)

        if self.aggressive_all:
            mandatory = torch.empty(0, device=device, dtype=torch.long)
            candidate = torch.arange(0, length, device=device, dtype=torch.long)
            return mandatory, candidate

        mandatory_parts = []
        candidate_parts = [history_idx]
        if self.prune_sink:
            candidate_parts.append(sink_idx)
        else:
            mandatory_parts.append(sink_idx)
        if self.prune_tail:
            candidate_parts.append(tail_idx)
        else:
            mandatory_parts.append(tail_idx)

        mandatory = torch.cat(mandatory_parts) if mandatory_parts else torch.empty(0, device=device, dtype=torch.long)
        candidate = torch.cat(candidate_parts) if candidate_parts else torch.empty(0, device=device, dtype=torch.long)
        return mandatory, candidate

    def _ranked_select(
        self,
        pos_seg: torch.Tensor,
        v_seg: torch.Tensor,
        budget: int,
        freqs: torch.Tensor | None,
        prompt_head: torch.Tensor | None,
        apply_selection: bool,
    ) -> torch.Tensor:
        n = v_seg.shape[0]
        select = torch.zeros(n, dtype=torch.bool, device=v_seg.device)
        if n == 0 or budget <= 0:
            return select

        if not apply_selection:
            select[-min(n, budget):] = True
            return select

        traj_ratio_eff, traj_weight_eff, _ = self._effective_selection_params(pos_seg)
        ivc_scores = None
        sem_scores = None
        traj_scores = None

        if self.ivc_ratio > 0 and freqs is not None:
            ivc_scores = self.ivc_selector.get_ivc_scores(pos_seg, d_model=self.head_dim, freqs=freqs)
            k_ivc = max(1, int(round(n * self.ivc_ratio)))
            select |= _topk_mask(ivc_scores, k=k_ivc)

        if self.semantic_ratio > 0:
            sem_scores = self.semantic_selector.get_semantic_scores(
                v_seg,
                prompt_v=prompt_head,
                seed_ratio=self.seed_ratio,
            )
            k_sem = max(1, int(round(n * self.semantic_ratio)))
            select |= _topk_mask(sem_scores, k=k_sem)

        if traj_ratio_eff > 0:
            traj_scores = self.get_trajectory_scores(pos_seg=pos_seg, v_seg=v_seg)
            k_traj = max(1, int(round(n * traj_ratio_eff)))
            select |= _topk_mask(traj_scores, k=k_traj)

        combined = torch.zeros(n, dtype=torch.float32, device=v_seg.device)
        if ivc_scores is not None:
            combined = combined + _normalize_scores(ivc_scores) * max(self.ivc_ratio, 1e-6)
        if sem_scores is not None:
            combined = combined + _normalize_scores(sem_scores) * max(self.semantic_ratio, 1e-6)
        if traj_scores is not None:
            traj_w = traj_weight_eff if traj_weight_eff > 0 else traj_ratio_eff
            combined = combined + _normalize_scores(traj_scores) * max(traj_w, 1e-6)
        if torch.all(combined == 0):
            combined = torch.arange(n, device=v_seg.device, dtype=torch.float32)

        num_selected = int(select.sum().item())
        if num_selected > budget:
            keep_idx = torch.topk(combined.masked_fill(~select, float("-inf")), k=budget, largest=True, sorted=False).indices
            select = torch.zeros_like(select)
            select[keep_idx] = True
            return select

        if num_selected < budget:
            remainder = ~select
            fill_scores = combined.masked_fill(~remainder, float("-inf"))
            add_k = min(int(remainder.sum().item()), budget - num_selected)
            if add_k > 0:
                add_idx = torch.topk(fill_scores, k=add_k, largest=True, sorted=False).indices
                select[add_idx] = True
        return select

    @staticmethod
    def get_trajectory_scores(pos_seg: torch.Tensor, v_seg: torch.Tensor) -> torch.Tensor:
        """
        Compute motion saliency by tracking value-vector changes along the same spatial lattice (y, x)
        over time t. Higher score means stronger temporal change for that trajectory.
        """
        n = v_seg.shape[0]
        if n <= 1:
            return torch.zeros(n, dtype=torch.float32, device=v_seg.device)
        if pos_seg.ndim != 2 or pos_seg.shape[1] != 3:
            raise ValueError(f"pos_seg must be [N,3], got {tuple(pos_seg.shape)}")

        pos = _as_long_tensor(pos_seg)
        y = pos[:, 1]
        x = pos[:, 2]
        t = pos[:, 0]
        max_x = int(x.max().item()) + 1 if x.numel() > 0 else 1
        yx_id = y * max(1, max_x) + x

        # Lexicographic order by (yx, t)
        sort_key = yx_id * max(1, int(t.max().item()) + 1) + t.clamp(min=0)
        perm = torch.argsort(sort_key)

        yx_s = yx_id[perm]
        v_s = v_seg[perm].float()
        scores_s = torch.zeros(n, dtype=torch.float32, device=v_seg.device)

        dv = (v_s[1:] - v_s[:-1]).norm(dim=-1)
        same_track = yx_s[1:] == yx_s[:-1]
        dv = dv * same_track.float()

        # Assign motion evidence to both endpoints of each temporal edge.
        scores_s[1:] = torch.maximum(scores_s[1:], dv)
        scores_s[:-1] = torch.maximum(scores_s[:-1], dv)

        scores = torch.zeros_like(scores_s)
        scores[perm] = scores_s
        return scores

    def update_cache(
        self,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        budget: int,
        freqs: torch.Tensor | None,
        prompt_head: torch.Tensor | None,
        apply_selection: bool,
        sink_len: int | None = None,
        head_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if budget <= 0:
            return (
                k_seq.new_empty((0, self.head_dim)),
                v_seq.new_empty((0, self.head_dim)),
                pos_seq.new_empty((0, 3), dtype=torch.long),
            )

        length = k_seq.shape[0]
        if length <= budget and not apply_selection:
            return k_seq, v_seq, pos_seq

        mandatory, candidate = self._segment_indices(length=length, device=k_seq.device, sink_len=sink_len)

        if mandatory.shape[0] >= budget:
            keep_idx = mandatory.sort().values[-budget:]
            return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        remain_budget = budget - mandatory.shape[0]
        if candidate.numel() == 0:
            keep_idx = mandatory.sort().values
            return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        if self.periodic_peak_mask:
            is_osc = True if head_idx is None else self.osc_head_flags[head_idx]
            should_apply = (not self.periodic_peak_only_oscillating) or is_osc
            if should_apply:
                cand_pos = pos_seq[candidate]
                peak_local_mask = self._periodic_peak_local_mask(_as_long_tensor(cand_pos[:, 0]))
                if torch.any(peak_local_mask):
                    peak_global = candidate[peak_local_mask]
                    mandatory = torch.unique(torch.cat([mandatory, peak_global], dim=0), sorted=False)
                    if mandatory.shape[0] >= budget:
                        keep_idx = mandatory.sort().values[-budget:]
                        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]
                    remain_budget = budget - mandatory.shape[0]
                    candidate = candidate[~peak_local_mask]
                    if candidate.numel() == 0:
                        keep_idx = mandatory.sort().values
                        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        if self.history_frame_quota > 0:
            # Guarantee temporal coverage: each history frame contributes at least a small
            # number of tokens before global ranking.
            cand_pos = pos_seq[candidate]
            cand_v = v_seq[candidate]
            traj_scores = self.get_trajectory_scores(pos_seg=cand_pos, v_seg=cand_v)
            quota_scores = traj_scores
            _, _, quota_ivc_ratio_eff = self._effective_selection_params(cand_pos)
            if freqs is not None and quota_ivc_ratio_eff > 0:
                ivc_scores = self.ivc_selector.get_ivc_scores(cand_pos, d_model=self.head_dim, freqs=freqs)
                ivc_norm = _normalize_scores(ivc_scores)
                traj_norm = _normalize_scores(traj_scores)
                mix = quota_ivc_ratio_eff
                quota_scores = mix * ivc_norm + (1.0 - mix) * traj_norm
            picked_chunks = []
            for t_val in torch.unique(cand_pos[:, 0], sorted=True):
                local = torch.nonzero(cand_pos[:, 0] == t_val, as_tuple=False).squeeze(1)
                if local.numel() == 0:
                    continue
                k_keep = min(self.history_frame_quota, int(local.numel()))
                top_local = local[torch.topk(quota_scores[local], k=k_keep, largest=True, sorted=False).indices]
                picked_chunks.append(top_local)
            if picked_chunks:
                picked_local = torch.unique(torch.cat(picked_chunks, dim=0), sorted=False)
                picked_global = candidate[picked_local]
                mandatory = torch.unique(torch.cat([mandatory, picked_global], dim=0), sorted=False)
                if mandatory.shape[0] >= budget:
                    keep_idx = mandatory.sort().values[-budget:]
                    return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]
                remain_budget = budget - mandatory.shape[0]
                keep_mask = torch.ones(candidate.shape[0], dtype=torch.bool, device=candidate.device)
                keep_mask[picked_local] = False
                candidate = candidate[keep_mask]
                if candidate.numel() == 0:
                    keep_idx = mandatory.sort().values
                    return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        candidate_pos = pos_seq[candidate]
        candidate_v = v_seq[candidate]
        local_mask = self._ranked_select(
            pos_seg=candidate_pos,
            v_seg=candidate_v,
            budget=remain_budget,
            freqs=freqs,
            prompt_head=prompt_head,
            apply_selection=apply_selection,
        )
        selected = candidate[local_mask]
        keep_idx = torch.cat([mandatory, selected]).sort().values
        if keep_idx.shape[0] > budget:
            keep_idx = keep_idx[-budget:]
        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

    @torch.no_grad()
    def _try_write_through_readout_tail(
        self,
        *,
        new_k_flat: torch.Tensor,
        new_v_flat: torch.Tensor,
        pos_flat: torch.Tensor,
        l_in: int,
        current_start: int | None,
    ) -> bool:
        if os.environ.get("PYRAMIDKV_WRITE_THROUGH_READOUT") != "1":
            return False
        if not self._readout_cache_valid or current_start is None:
            return False
        if self._readout_cache_current_start != int(current_start):
            return False
        if self._ws_k_raw is None or self._ws_v is None or self._ws_rope_pos is None:
            return False
        if l_in <= 0:
            return False
        n = self.batch_size * self.num_heads
        if new_k_flat.shape[0] != n or new_k_flat.shape[1] != l_in:
            return False
        for i in range(n):
            spec = self._readout_tail_specs[i]
            if spec is None or int(spec[1]) != int(l_in):
                return False
        for i in range(n):
            offset, tail_len = self._readout_tail_specs[i]
            self._ws_k_raw[offset:offset + tail_len] = new_k_flat[i]
            self._ws_v[offset:offset + tail_len] = new_v_flat[i]
            self._ws_rope_pos[offset:offset + tail_len] = pos_flat[i]
        self._readout_tail_write_through_valid = True
        return True

    @torch.no_grad()
    def _try_fast_path_noisy_overwrite(
        self,
        new_k_flat: torch.Tensor,
        new_v_flat: torch.Tensor,
        pos_flat: torch.Tensor,
        l_in: int,
        current_start: int | None,
        current_end: int | None,
        cache_update_mode: str,
        frame_seqlen: int = 0,
        frame_start_t: int = 0,
    ) -> bool:
        """M6 fast-path: bypass the 360-iter Python loop in update().

        Returns True if the fast-path successfully executed (caller should
        return immediately). Returns False if any invariant fails and the
        slow path must run.

        Handles both noisy and clean pass in steady state. Clean adds
        anchor-strategy updates (cyclic/stride/lag) via composition.update_all
        — the tight loop inlines only those calls that are actually
        configured for this cache.

        Invariants required (all must hold):
        1. PYRAMIDKV_DISABLE_M6_FASTPATH=1 not set
        2. cache_update_mode in {"noisy", "clean"}
        3. self._steady_state_reached is True (from G2)
        4. current_end is not None and overwrite_current_block (every batch)
        5. every _dyn_store_* head has len >= l_in (room to overwrite tail)
        6. l_in > 0
        7. For clean+osc mode: no i2v sink capture (context_len already captured)
        """
        if os.environ.get("PYRAMIDKV_DISABLE_M6_FASTPATH") == "1":
            return False
        if cache_update_mode not in ("noisy", "clean"):
            return False
        if not self._steady_state_reached:
            return False
        if l_in <= 0:
            return False
        if current_end is None:
            return False
        # Invariant: overwrite for every batch (current_end <= global_end_index)
        gei = self.global_end_index
        for b_idx in range(self.batch_size):
            if current_end > gei[b_idx]:
                return False

        store_k = self._dyn_store_k
        store_v = self._dyn_store_v
        store_pos = self._dyn_store_pos
        starts = self._dyn_store_start
        lens = self._dyn_store_len
        n = self.batch_size * self.num_heads
        # Invariant: every head has at least l_in rows stored
        for i in range(n):
            if store_k[i] is None or lens[i] < l_in:
                return False

        # Clean pass additionally updates middle strategies (osc mode only).
        # Guard: clean-path needs the same structural conditions as slow path
        # (use_osc_frame_mode + composition.has_middle). If osc is off, clean
        # behaves like noisy for this fast-path's purposes (only tail write).
        do_clean_anchors = (
            cache_update_mode == "clean"
            and self.use_osc_frame_mode
            and frame_seqlen > 0
            and l_in >= frame_seqlen
            and (l_in % frame_seqlen == 0)
        )

        # All invariants hold. Execute tight inline write loop.
        # Views self.dynamic_k[i] = _dyn_store_k[i][start:start+len] — bounds
        # unchanged by an in-place overwrite, so no _sync_dynamic_views needed.
        h = self.num_heads
        compositions = self.compositions_row if do_clean_anchors else None
        cpp_strategy_updated = False
        if do_clean_anchors:
            cpp_strategy_updated = self._try_cpp_strategy_update_all(
                k_flat=new_k_flat,
                v_flat=new_v_flat,
                pos_flat=pos_flat,
                frame_seqlen=frame_seqlen,
                current_t=frame_start_t,
            )
        for i in range(n):
            if compositions is not None:
                head_idx = i % h
                if head_idx < len(compositions):
                    comp = compositions[head_idx]
                    if (
                        comp is not None
                        and comp.has_middle
                        and not (cpp_strategy_updated and self._cpp_strategy_head_supported(head_idx))
                    ):
                        comp.update_all(
                            idx=i,
                            k_seq=new_k_flat[i],
                            v_seq=new_v_flat[i],
                            pos_seq=pos_flat[i],
                            frame_seqlen=frame_seqlen,
                            current_t=frame_start_t,
                        )
            end = starts[i] + lens[i]
            tail = end - l_in
            store_k[i][tail:end] = new_k_flat[i]
            store_v[i][tail:end] = new_v_flat[i]
            store_pos[i][tail:end] = pos_flat[i]
        self._try_write_through_readout_tail(
            new_k_flat=new_k_flat,
            new_v_flat=new_v_flat,
            pos_flat=pos_flat,
            l_in=l_in,
            current_start=current_start,
        )
        return True

    @torch.no_grad()
    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        current_start: int | None = None,
        grid_sizes: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        prompt_v: torch.Tensor | None = None,
        **kwargs,
    ):
        update_start = perf_counter()
        if prompt_v is not None:
            self.prompt_v = prompt_v
        cache_update_mode = kwargs.get("cache_update_mode", "default")

        b, l_new, h, d = new_k.shape
        assert b == self.batch_size and h == self.num_heads and d == self.head_dim
        current_end = None
        if current_start is not None:
            current_end = current_start + l_new
        overwrite_current_block = False
        if current_end is not None:
            overwrite_current_block = all(
                current_end <= int(self.global_end_index[batch_idx])
                for batch_idx in range(self.batch_size)
            )

        if grid_sizes is None:
            result = super().update(new_k, new_v, current_start=current_start)
            self._record_profile("update_ms", update_start)
            return result

        new_k_flat = new_k.transpose(1, 2).reshape(b * h, l_new, d)
        new_v_flat = new_v.transpose(1, 2).reshape(b * h, l_new, d)
        pos_b = self._build_pos_ids(grid_sizes=grid_sizes, seq_len=l_new, current_start=current_start or 0, device=new_k.device)
        pos_flat = pos_b.unsqueeze(1).expand(b, h, l_new, 3).reshape(b * h, l_new, 3)
        if self._frame_seqlen is None:
            frame_tokens = _as_long_tensor(grid_sizes[:, 1] * grid_sizes[:, 2])
            if torch.any(frame_tokens <= 0):
                raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
            if torch.unique(frame_tokens).numel() != 1:
                raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
            self._frame_seqlen = int(frame_tokens[0].item())
        frame_seqlen = self._frame_seqlen
        frame_start_t = 0 if frame_seqlen <= 0 else int((current_start or 0) // frame_seqlen)

        # M6: Fast-path for steady-state noisy/clean overwrite.
        # When all heads already have dyn_store populated with at least l_new rows
        # and we're re-entering the same block that was appended before, the entire
        # 360-iter Python loop reduces to a tight per-head tail write. Skip all
        # helper-method overhead and _sync_dynamic_views (slice bounds unchanged).
        # Clean pass also inlines composition.update_all for middle strategies.
        if self._try_fast_path_noisy_overwrite(
            new_k_flat=new_k_flat, new_v_flat=new_v_flat, pos_flat=pos_flat,
            l_in=l_new, current_start=current_start, current_end=current_end, cache_update_mode=cache_update_mode,
            frame_seqlen=frame_seqlen, frame_start_t=frame_start_t,
        ):
            if current_end is not None:
                for batch_idx in range(self.batch_size):
                    self.global_end_index[batch_idx] = current_end
            # Fast-path mirrors slow-path readout-cache handling:
            # - noisy: structure preserved → keep cache, dirty-tail refresh
            # - clean in osc mode: anchors (cyclic/lag) may have shifted, so
            #   readout cache tensors for anchor regions are stale → invalidate.
            # - clean in non-osc mode: structure preserved, safe to keep cache.
            clean_osc = (cache_update_mode == "clean" and self.use_osc_frame_mode)
            if self._readout_cache_valid and current_start is not None and \
                    self._readout_cache_current_start == int(current_start) and not clean_osc:
                self._readout_cache_tail_dirty = True
            else:
                self._invalidate_readout_cache()
            self._mirror_to_shadow()
            self._record_profile("update_ms", update_start)
            return

        if self.use_osc_frame_mode:
            # Keep the dynamic region as a short local neighborhood for smoother transitions.
            self.tail_len = self.local_tail_frames * frame_seqlen
        else:
            self.tail_len = self._base_tail_len

        prompt_per_head = self.semantic_selector.prepare_prompt_values(
            self.prompt_v,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        self.update_step += 1
        should_reselect = (self.update_step % self.update_interval == 0)
        structural_change = False
        cpp_strategy_updated = False
        if self.use_osc_frame_mode and cache_update_mode in {"default", "clean"}:
            cpp_strategy_updated = self._try_cpp_strategy_update_all(
                k_flat=new_k_flat,
                v_flat=new_v_flat,
                pos_flat=pos_flat,
                frame_seqlen=frame_seqlen,
                current_t=frame_start_t,
            )

        for i in range(b * h):
            batch_idx = i // h
            head_idx = i % h
            full_cap = self.capacities[head_idx]
            if self.osc_full_kv_retention and self.osc_head_flags[head_idx]:
                full_cap = self.max_capacity
            prompt_head = None
            if prompt_per_head is not None:
                prompt_head = prompt_per_head[head_idx]

            if self.use_osc_frame_mode and cache_update_mode in {"default", "clean"}:
                composition = (
                    self.compositions_row[head_idx]
                    if self.compositions_row is not None and head_idx < len(self.compositions_row)
                    else None
                )
                if composition is not None and composition.has_middle:
                    if not (cpp_strategy_updated and self._cpp_strategy_head_supported(head_idx)):
                        composition.update_all(
                            idx=i,
                            k_seq=new_k_flat[i],
                            v_seq=new_v_flat[i],
                            pos_seq=pos_flat[i],
                            frame_seqlen=frame_seqlen,
                            current_t=frame_start_t,
                        )
                else:
                    self._update_cyclic_anchors(
                        idx=i,
                        k_seq=new_k_flat[i],
                        v_seq=new_v_flat[i],
                        pos_seq=pos_flat[i],
                        frame_seqlen=frame_seqlen,
                        t_start=frame_start_t,
                    )
                    self._update_lag_anchors(
                        idx=i,
                        k_seq=new_k_flat[i],
                        v_seq=new_v_flat[i],
                        pos_seq=pos_flat[i],
                        frame_seqlen=frame_seqlen,
                        t_start=frame_start_t,
                    )

            if self.is_i2v and self.static_k[i] is None:
                if l_new >= self.context_len:
                    self.static_k[i] = new_k_flat[i, :self.context_len].clone()
                    self.static_v[i] = new_v_flat[i, :self.context_len].clone()
                    self.static_pos[i] = pos_flat[i, :self.context_len].clone()
                    if l_new > self.context_len:
                        self._set_dynamic_store(
                            i,
                            new_k_flat[i, self.context_len:],
                            new_v_flat[i, self.context_len:],
                            pos_flat[i, self.context_len:],
                            reserve_extra=max(16, (l_new - self.context_len) // 2),
                        )
                    else:
                        self._set_dynamic_empty(i, device=new_k.device, dtype=new_k.dtype)
                    self._current_block_token_len[i] = 0
                    structural_change = True
                    continue

            curr_dyn_k = self.dynamic_k[i]
            curr_dyn_v = self.dynamic_v[i]
            curr_dyn_pos = self.dynamic_pos[i]
            overwrite = False
            if current_end is not None:
                overwrite = current_end <= int(self.global_end_index[batch_idx])

            incoming_k = new_k_flat[i]
            incoming_v = new_v_flat[i]
            incoming_p = pos_flat[i]
            incoming_k, incoming_v, incoming_p = self._capture_sink_if_needed(
                idx=i,
                head_idx=head_idx,
                k_in=incoming_k,
                v_in=incoming_v,
                p_in=incoming_p,
                current_start=current_start,
                overwrite=overwrite,
                freqs=freqs,
                prompt_head=prompt_head,
            )
            l_in = incoming_k.shape[0]
            self._current_block_token_len[i] = int(l_in)

            if curr_dyn_k is None:
                self._set_dynamic_store(
                    i,
                    incoming_k,
                    incoming_v,
                    incoming_p,
                    reserve_extra=max(16, l_in // 2),
                )
                structural_change = True
            elif overwrite:
                if l_in == 0:
                    self._sync_dynamic_views(i)
                elif curr_dyn_k.shape[0] >= l_in:
                    self._overwrite_dynamic_tail(i, incoming_k, incoming_v, incoming_p)
                else:
                    self._set_dynamic_store(
                        i,
                        incoming_k,
                        incoming_v,
                        incoming_p,
                        reserve_extra=max(16, l_in // 2),
                    )
                    structural_change = True
            else:
                self._append_dynamic(i, incoming_k, incoming_v, incoming_p)
                structural_change = True

            k_merged = self.dynamic_k[i]
            v_merged = self.dynamic_v[i]
            p_merged = self.dynamic_pos[i]

            stable_kind = self._stable_strategy_kind(head_idx)
            dyn_cap = full_cap
            if self.use_osc_frame_mode and not self.is_i2v:
                # Oscillating heads keep short local recent tail; stable heads can keep wider history
                # and let stable policies downsample at frame level.
                if stable_kind is None:
                    dyn_cap = min(full_cap, self._head_recent_frames(head_idx) * frame_seqlen)
                else:
                    dyn_cap = full_cap
            elif self.is_i2v:
                dyn_cap = max(0, full_cap - self.context_len)
            elif self.sink_grid_decoupling and self.static_k[i] is not None and self.decouple_head_flags[head_idx]:
                dyn_cap = max(0, full_cap - int(self.static_k[i].shape[0]))

            if dyn_cap <= 0:
                self._set_dynamic_empty(i, device=new_k.device, dtype=new_k.dtype)
                structural_change = True
                continue

            if self.use_osc_frame_mode:
                if k_merged.shape[0] > dyn_cap:
                    self._keep_dynamic_suffix(i, dyn_cap)
                    k_merged = self.dynamic_k[i]
                    v_merged = self.dynamic_v[i]
                    p_merged = self.dynamic_pos[i]
                    structural_change = True
                if stable_kind is not None:
                    k_merged, v_merged, p_merged = self._apply_stable_strategy(
                        head_idx=head_idx,
                        k_seq=k_merged,
                        v_seq=v_merged,
                        pos_seq=p_merged,
                        frame_seqlen=frame_seqlen,
                    )
                    reserve_extra = min(max(16, l_in), max(0, dyn_cap - int(k_merged.shape[0])))
                    self._set_dynamic_store(i, k_merged, v_merged, p_merged, reserve_extra=reserve_extra)
                    structural_change = True
            else:
                needs_compaction = (k_merged.shape[0] > dyn_cap)
                allow_reselect = cache_update_mode in {"default", "clean"}
                apply_selection = allow_reselect and should_reselect and freqs is not None
                segment_sink_len = self.sink_len
                if self.sink_grid_decoupling and self.static_k[i] is not None and self.decouple_head_flags[head_idx]:
                    # In decoupling mode, sink tokens are fully externalized in static_k/static_v.
                    # Dynamic history should not re-introduce a sink-protected prefix.
                    segment_sink_len = 0
                if needs_compaction or apply_selection:
                    k_merged, v_merged, p_merged = self.update_cache(
                        k_seq=k_merged,
                        v_seq=v_merged,
                        pos_seq=p_merged,
                        budget=dyn_cap,
                        freqs=freqs,
                        prompt_head=prompt_head,
                        apply_selection=apply_selection,
                        sink_len=segment_sink_len,
                        head_idx=head_idx,
                    )
                    reserve_extra = min(max(16, l_in), max(0, dyn_cap - int(k_merged.shape[0])))
                    self._set_dynamic_store(i, k_merged, v_merged, p_merged, reserve_extra=reserve_extra)
                    structural_change = True
                elif k_merged.shape[0] > dyn_cap:
                    self._keep_dynamic_suffix(i, dyn_cap)
                    structural_change = True

        if current_end is not None:
            for batch_idx in range(self.batch_size):
                self.global_end_index[batch_idx] = current_end
        can_reuse_same_block = (
            self.readout_cache_enabled
            and current_start is not None
            and overwrite_current_block
            and self._readout_cache_valid
            and self._readout_cache_current_start == int(current_start)
            and not structural_change
            and (cache_update_mode == "noisy" or (cache_update_mode == "clean" and not self.use_osc_frame_mode))
        )
        if can_reuse_same_block:
            self._readout_cache_tail_dirty = True
        else:
            self._invalidate_readout_cache()
        self._mirror_to_shadow()
        self._record_profile("update_ms", update_start)

    def get_flat_kv_and_pos(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        total_k = []
        total_v = []
        total_pos = []
        lengths = []

        for i in range(self.batch_size * self.num_heads):
            k_parts = []
            v_parts = []
            p_parts = []

            if self.static_k[i] is not None:
                k_parts.append(self.static_k[i])
                v_parts.append(self.static_v[i])
                p_parts.append(self.static_pos[i])
            if self.dynamic_k[i] is not None:
                k_parts.append(self.dynamic_k[i])
                v_parts.append(self.dynamic_v[i])
                p_parts.append(self.dynamic_pos[i])

            if len(k_parts) == 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                t_k = torch.empty(0, self.head_dim, device=device)
                t_v = torch.empty(0, self.head_dim, device=device)
                t_pos = torch.empty(0, 3, dtype=torch.long, device=device)
            elif len(k_parts) == 1:
                t_k = k_parts[0]
                t_v = v_parts[0]
                t_pos = p_parts[0]
            else:
                t_k = torch.cat(k_parts, dim=0)
                t_v = torch.cat(v_parts, dim=0)
                t_pos = torch.cat(p_parts, dim=0)

            total_k.append(t_k)
            total_v.append(t_v)
            total_pos.append(t_pos)
            lengths.append(t_k.shape[0])

        k_flat = torch.cat(total_k, dim=0)
        v_flat = torch.cat(total_v, dim=0)
        pos_flat = torch.cat(total_pos, dim=0)
        cu_seqlens_k = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32).to(k_flat.device, non_blocking=True)
        max_seqlen_k = max(lengths) if lengths else 0
        self.last_flat_pos_ids = pos_flat
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_flat

    def get_flat_kv(self, **kwargs):
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, _ = self.get_flat_kv_and_pos()
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k

    def _ensure_workspace(
        self,
        total_len: int,
        num_seq: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return workspace views for flattened decoupled readout.

        Allocates with 20% headroom on first call or when capacity is insufficient.
        Otherwise returns sliced views of existing buffers (cu_seqlens is zero-filled).
        """
        alloc_len = int(total_len * 1.2) + 64  # 20% headroom
        cu_len = num_seq + 1

        need_alloc = (
            self._ws_k_raw is None
            or self._ws_k_raw.shape[0] < total_len
            or self._ws_k_raw.device != device
            or self._ws_k_raw.dtype != dtype
            or self._ws_k is None
            or self._ws_k.shape[0] < total_len
            or self._ws_k.device != device
            or self._ws_k.dtype != dtype
        )
        if need_alloc:
            self._ws_k_raw = torch.empty(alloc_len, self.head_dim, device=device, dtype=dtype)
            self._ws_k = torch.empty(alloc_len, self.head_dim, device=device, dtype=dtype)
            self._ws_v = torch.empty(alloc_len, self.head_dim, device=device, dtype=dtype)
            self._ws_frame_ids = torch.empty(alloc_len, dtype=torch.long, device=device)
            self._ws_rope_pos = torch.empty(alloc_len, 3, dtype=torch.long, device=device)

        if self._ws_cu_seqlens is None or self._ws_cu_seqlens.shape[0] < cu_len or self._ws_cu_seqlens.device != device:
            self._ws_cu_seqlens = torch.zeros(cu_len, dtype=torch.int32, device=device)
        else:
            self._ws_cu_seqlens[:cu_len].zero_()

        return (
            self._ws_k_raw[:total_len],
            self._ws_k[:total_len],
            self._ws_v[:total_len],
            self._ws_frame_ids[:total_len],
            self._ws_cu_seqlens[:cu_len],
            self._ws_rope_pos[:total_len],
        )

    def _invalidate_readout_cache(self) -> None:
        self._readout_cache_valid = False
        self._readout_cache_current_start = None
        self._readout_cache_sync_t_raw = None
        self._readout_cache_total_len = 0
        self._readout_cache_max_seqlen = 0
        self._readout_cache_frame_seqlen = 0
        self._readout_cache_tail_dirty = False
        self._readout_tail_write_through_valid = False
        self._readout_cache_shape_key = None
        num_seq = self.batch_size * self.num_heads
        self._readout_static_specs = [None] * num_seq
        self._readout_tail_specs = [None] * num_seq

    def _cache_readout_layout(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        frame_seqlen: int,
        total_len: int,
        max_seqlen_k: int,
        shape_key: tuple | None = None,
    ) -> None:
        self._readout_cache_valid = True
        self._readout_cache_current_start = int(current_start)
        self._readout_cache_sync_t_raw = int(sync_t_raw)
        self._readout_cache_total_len = int(total_len)
        self._readout_cache_max_seqlen = int(max_seqlen_k)
        self._readout_cache_frame_seqlen = int(frame_seqlen)
        self._readout_cache_shape_key = shape_key
        self._readout_cache_tail_dirty = False
        self._readout_tail_write_through_valid = False
        self._profile_stats["readout_total_len"] = float(total_len)
        self._profile_stats["readout_max_seqlen"] = float(max_seqlen_k)

    def _can_reuse_readout_cache(self, current_start: int, sync_t_raw: int, frame_seqlen: int) -> bool:
        if not self.readout_cache_enabled:
            return False
        if not self._readout_cache_valid:
            return False
        if (
            self._ws_k_raw is None
            or self._ws_k is None
            or self._ws_v is None
            or self._ws_frame_ids is None
            or self._ws_cu_seqlens is None
        ):
            return False
        return (
            self._readout_cache_current_start == int(current_start)
            and self._readout_cache_sync_t_raw == int(sync_t_raw)
            and self._readout_cache_frame_seqlen == int(frame_seqlen)
        )

    def _can_reuse_readout_cache_for_rope_only(
        self, current_start: int, frame_seqlen: int
    ) -> bool:
        """Path A+B: cache layout (K_RAW + V) is valid for the same readout
        shape, but cs differs from the cached state — only RoPE needs refresh.

        Triggered when multi-chunk readout queries chunks 1, 2 after chunk 0
        cached the layout. Cached K_RAW source data is identical across chunks;
        only ws_rope_pos / ws_k (post-RoPE) need re-derivation for the new
        sync_t / sync_t_raw.
        """
        if not self.readout_cache_enabled:
            return False
        if not self._readout_cache_valid:
            return False
        if (
            self._ws_k_raw is None
            or self._ws_k is None
            or self._ws_v is None
            or self._ws_rope_pos is None
            or self._ws_frame_ids is None
            or self._ws_cu_seqlens is None
        ):
            return False
        if self._readout_cache_frame_seqlen != int(frame_seqlen):
            return False
        # Only allow rope-only refresh within the same denoising block. We
        # detect "same block" via the cached shape_key — if the cold pack just
        # ran for a chunk in this multi-call, the shape will be stable across
        # the remaining chunks. The cache's exact (cs, sync_t_raw) won't match
        # this chunk, but the underlying KV state is identical.
        if self._readout_cache_shape_key is None:
            return False
        return True

    def _cached_readout_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        total_len = self._readout_cache_total_len
        num_seq = self.batch_size * self.num_heads
        return (
            self._ws_k[:total_len],
            self._ws_v[:total_len],
            self._ws_cu_seqlens[: num_seq + 1],
            self._readout_cache_max_seqlen,
            self._ws_frame_ids[:total_len],
        )

    def _refresh_rope_only_for_cs(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        sync_t: int,
        frame_seqlen: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor] | None:
        """Path A+B: K_RAW + V are cached and shape-stable across chunks within
        a multi-readout call. Only sync_t/sync_t_raw differ per chunk, so the
        rope_pos / frame_ids / k_flat (post-RoPE) need refresh — but the
        scatter_copy work for K_RAW/V can be skipped.

        Returns None if the cached shape doesn't match (caller falls back to
        cold pack). Returns the same view tuple as _cached_readout_views() on
        success.

        Conservative: requires no cpp_segments (active C++ strategy manager).
        Production pyramid-forcing config has cpp_segments empty by default.
        """
        # Build a fresh spec for the new sync_t / sync_t_raw. This pays the
        # spec-build cost (~150us per call) but gates everything below.
        spec = self._build_readout_spec(sync_t_raw=sync_t_raw, sync_t=sync_t)

        # Shape mismatch → caller must do a full cold pack to rebuild K_RAW/V.
        if spec.shape_key != self._readout_cache_shape_key:
            return None

        # Bail on configs that use the active C++ strategy manager. Those
        # heads write into the workspace via materialize_anchors; the
        # rope-only path doesn't currently rewrite cpp_segments output.
        if spec.cpp_segments:
            return None

        capture_physical = self.capture_frame_id_mode == "physical"
        rope_pos_flat = self._ws_rope_pos
        frame_ids_flat = self._ws_frame_ids
        k_raw = self._ws_k_raw
        k_flat = self._ws_k

        # Step 1: refresh rope_pos from segment source pos (overwriting any
        # previous chunk's dynamic_rope_t / time-mapping mutations).
        for seg in spec.segments:
            start = seg.offset
            end = start + seg.length
            rope_pos_flat[start:end].copy_(seg.pos)
            if seg.dynamic_rope_t is not None:
                rope_pos_flat[start:end, 0] = int(seg.dynamic_rope_t)

        # Step 2: dynamic_time_map per segment + frame_ids refresh.
        rope_start = perf_counter()
        for seg in spec.segments:
            start = seg.offset
            end = start + seg.length
            if seg.dynamic_time_map:
                map_dynamic_pos_time(
                    rope_pos_flat[start:end],
                    current_t=sync_t_raw,
                    history_time_mapping_mode=self.history_time_mapping_mode,
                    history_relative_t_max=self.history_relative_t_max,
                    history_time_soft_factor=self.history_time_soft_factor,
                    inplace=True,
                )
            if capture_physical or (seg.dynamic_rope_t is None and not seg.dynamic_time_map):
                frame_ids_flat[start:end] = _as_long_tensor(seg.pos[:, 0])
            else:
                frame_ids_flat[start:end] = _as_long_tensor(rope_pos_flat[start:end, 0])

        # Step 3: re-apply RoPE on the (still cached) K_RAW.
        apply_rope_to_flat_k(
            k_raw[:spec.total_len],
            rope_pos_flat[:spec.total_len],
            freqs=freqs,
            freq_parts=freq_parts,
            out=k_flat[:spec.total_len],
        )
        self._record_profile("rope_ms", rope_start)

        # Step 4: update cache state for the new chunk.
        self._readout_static_specs = spec.static_specs
        self._readout_tail_specs = spec.tail_specs
        # Path A+B v2 (PYRAMIDKV_PATH_AB_V2): preserve _readout_cache_current_start
        # and _readout_cache_sync_t_raw at the values set by the prior cold pack
        # (= block-anchor cs). Subsequent denoising step's update() with cs =
        # block-anchor will then match and avoid invalidation. shape_key /
        # total_len / max_seqlen are guaranteed identical (we just verified
        # shape match above) so refreshing them is a no-op. The flags
        # `_readout_cache_tail_dirty` and `_readout_tail_write_through_valid`
        # need a reset since we've re-derived the post-RoPE k_flat from K_RAW.
        if os.environ.get("PYRAMIDKV_PATH_AB_V2", "1") != "0":
            self._readout_cache_total_len = int(spec.total_len)
            self._readout_cache_max_seqlen = int(spec.max_seqlen)
            self._readout_cache_frame_seqlen = int(frame_seqlen)
            self._readout_cache_shape_key = spec.shape_key
            self._readout_cache_tail_dirty = False
            self._readout_tail_write_through_valid = False
        else:
            self._cache_readout_layout(
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                frame_seqlen=frame_seqlen,
                total_len=spec.total_len,
                max_seqlen_k=spec.max_seqlen,
                shape_key=spec.shape_key,
            )

        self._profile_stats["rope_only_refresh_count"] = (
            self._profile_stats.get("rope_only_refresh_count", 0.0) + 1.0
        )
        return self._cached_readout_views()

    def _materialize_cpp_readout_segments(
        self,
        spec: _ReadoutSpec,
        *,
        k_raw: torch.Tensor,
        v_flat: torch.Tensor,
        rope_pos_flat: torch.Tensor,
        frame_ids_flat: torch.Tensor,
    ) -> None:
        if not spec.cpp_segments:
            return
        manager = self._cpp_strategy_manager
        for cpp_seg in spec.cpp_segments:
            if manager is None:
                raise RuntimeError("C++ strategy readout segment requires an active manager.")
            written = manager.materialize_anchors(
                seq_idx=cpp_seg.seq_idx,
                head_idx=cpp_seg.head_idx,
                current_t=cpp_seg.sync_t_raw,
                recent_min_t=cpp_seg.tail_min_t,
                sink_max_t=cpp_seg.sink_max_t,
                out_k=k_raw,
                out_v=v_flat,
                out_pos=rope_pos_flat,
                out_frame_ids=frame_ids_flat,
                offset=cpp_seg.offset,
                dynamic_rope_t=cpp_seg.dynamic_rope_t,
                capture_physical=cpp_seg.frame_ids_physical,
            )
            if written != cpp_seg.length:
                raise RuntimeError(
                    f"C++ strategy materialized {written} tokens for seq={cpp_seg.seq_idx}, "
                    f"expected {cpp_seg.length}."
                )

    def _cuda_refresh_opt_in(self) -> bool:
        return os.environ.get("PYRAMIDKV_CUDA_REFRESH", "0").strip().lower() in ("1", "true", "yes", "on")

    def _ensure_cuda_refresh_descriptor_buffers(self, n_seg: int, device: torch.device) -> None:
        if (
            self._cuda_refresh_desc_capacity >= n_seg
            and self._cuda_refresh_src_ptrs_k is not None
            and self._cuda_refresh_src_ptrs_k.device == device
        ):
            return
        capacity = max(16, int(n_seg * 1.25) + 8)
        self._cuda_refresh_desc_capacity = capacity
        self._cuda_refresh_desc_key = None
        self._cuda_refresh_src_ptrs_k = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_src_ptrs_v = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_src_ptrs_pos = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_offsets = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_lengths = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_flags = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_dynamic_rope_t = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_override_starts = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_override_ends = torch.empty(capacity, dtype=torch.int64, device=device)
        self._cuda_refresh_override_vals = torch.empty(capacity, dtype=torch.int64, device=device)

    @staticmethod
    def _copy_values_to_buffer(buffer: torch.Tensor, values: Sequence[int]) -> None:
        if not values:
            return
        buffer[: len(values)].copy_(torch.as_tensor(values, dtype=buffer.dtype), non_blocking=True)

    @staticmethod
    def _copy_data_ptrs_to_buffer(buffer: torch.Tensor, tensors: Sequence[torch.Tensor]) -> None:
        if not tensors:
            return
        ptrs = [int(t.data_ptr()) for t in tensors]
        buffer[: len(ptrs)].copy_(torch.as_tensor(ptrs, dtype=torch.int64), non_blocking=True)

    def _prepare_cuda_refresh_descriptor_buffers(
        self,
        *,
        src_k: Sequence[torch.Tensor],
        src_v: Sequence[torch.Tensor],
        src_pos: Sequence[torch.Tensor],
        offsets: Sequence[int],
        lengths: Sequence[int],
        flags: Sequence[int],
        dynamic_rope_t: Sequence[int],
        desc_key: tuple,
        device: torch.device,
    ) -> None:
        n_seg = len(src_k)
        self._ensure_cuda_refresh_descriptor_buffers(n_seg, device)
        assert self._cuda_refresh_src_ptrs_k is not None
        assert self._cuda_refresh_src_ptrs_v is not None
        assert self._cuda_refresh_src_ptrs_pos is not None
        assert self._cuda_refresh_offsets is not None
        assert self._cuda_refresh_lengths is not None
        assert self._cuda_refresh_flags is not None
        assert self._cuda_refresh_dynamic_rope_t is not None
        self._copy_data_ptrs_to_buffer(self._cuda_refresh_src_ptrs_k, src_k)
        self._copy_data_ptrs_to_buffer(self._cuda_refresh_src_ptrs_v, src_v)
        self._copy_data_ptrs_to_buffer(self._cuda_refresh_src_ptrs_pos, src_pos)
        if desc_key != self._cuda_refresh_desc_key:
            self._copy_values_to_buffer(self._cuda_refresh_offsets, offsets)
            self._copy_values_to_buffer(self._cuda_refresh_lengths, lengths)
            self._copy_values_to_buffer(self._cuda_refresh_flags, flags)
            self._copy_values_to_buffer(self._cuda_refresh_dynamic_rope_t, dynamic_rope_t)
            self._cuda_refresh_desc_key = desc_key

    def _try_cuda_refresh_cached_readout(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        sync_t: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        if self._cuda_refresh_disabled:
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return False
        if (
            not self._readout_cache_valid
            or self._ws_k_raw is None
            or self._ws_k is None
            or self._ws_v is None
            or self._ws_rope_pos is None
            or self._ws_frame_ids is None
            or not self._ws_k_raw.is_cuda
        ):
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return False
        if not cuda_refresh_available():
            self._cuda_refresh_disabled = True
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return False

        capture_physical = self.capture_frame_id_mode == "physical"
        rewrite_static = int(current_start) == 0
        num_seq = self.batch_size * self.num_heads
        src_k: list[torch.Tensor] = []
        src_v: list[torch.Tensor] = []
        src_pos: list[torch.Tensor] = []
        offsets: list[int] = []
        lengths: list[int] = []
        flags: list[int] = []
        dynamic_rope_t: list[int] = []
        desc_parts = []

        if rewrite_static:
            for i in range(num_seq):
                static_spec = self._readout_static_specs[i]
                if static_spec is None:
                    continue
                stat_k = self.static_k[i]
                stat_v = self.static_v[i]
                stat_pos = self.static_pos[i]
                offset, n_s = static_spec
                if stat_k is None or stat_v is None or stat_pos is None or stat_k.shape[0] != n_s:
                    self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                    return False
                head_idx = i % self.num_heads
                has_dynamic_rope = bool(self.decouple_head_flags[head_idx])
                flag = 0
                if capture_physical or not has_dynamic_rope:
                    flag |= _CUDA_REFRESH_FLAG_FRAME_IDS_PHYSICAL
                if has_dynamic_rope:
                    flag |= _CUDA_REFRESH_FLAG_DYNAMIC_ROPE
                src_k.append(stat_k)
                src_v.append(stat_v)
                src_pos.append(stat_pos)
                offsets.append(int(offset))
                lengths.append(int(n_s))
                flags.append(flag)
                dynamic_rope_t.append(int(sync_t) if has_dynamic_rope else 0)
                desc_parts.append(("static", i, int(offset), int(n_s), flag, int(sync_t) if has_dynamic_rope else 0))

        time_map = self.history_time_mapping_mode != "none"
        for i in range(num_seq):
            tail_spec = self._readout_tail_specs[i]
            if tail_spec is None:
                continue
            dyn_k = self.dynamic_k[i]
            dyn_v = self.dynamic_v[i]
            dyn_pos = self.dynamic_pos[i]
            offset, tail_len = tail_spec
            if dyn_k is None or dyn_v is None or dyn_pos is None or dyn_k.shape[0] < tail_len:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return False
            dyn_k_tail = dyn_k[-tail_len:]
            dyn_v_tail = dyn_v[-tail_len:]
            dyn_pos_tail = dyn_pos[-tail_len:]
            flag = 0
            if time_map:
                flag |= _CUDA_REFRESH_FLAG_DYNAMIC_TIME_MAP
            if capture_physical:
                flag |= _CUDA_REFRESH_FLAG_FRAME_IDS_PHYSICAL
            src_k.append(dyn_k_tail)
            src_v.append(dyn_v_tail)
            src_pos.append(dyn_pos_tail)
            offsets.append(int(offset))
            lengths.append(int(tail_len))
            flags.append(flag)
            dynamic_rope_t.append(0)
            desc_parts.append(("tail", i, int(offset), int(tail_len), flag, 0))

        n_seg = len(src_k)
        if n_seg == 0:
            self._readout_cache_tail_dirty = False
            self._readout_tail_write_through_valid = False
            self._profile_stats["cuda_refresh_count"] += 1.0
            self._profile_stats["readout_total_len"] = float(self._readout_cache_total_len)
            self._profile_stats["readout_max_seqlen"] = float(self._readout_cache_max_seqlen)
            return True

        dst_device = self._ws_k_raw.device
        dst_dtype = self._ws_k_raw.dtype
        for tensors in (src_k, src_v, src_pos):
            for tensor in tensors:
                if not tensor.is_cuda or tensor.device != dst_device or not tensor.is_contiguous():
                    self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                    return False
        for tensor in src_k:
            if tensor.dtype != dst_dtype or tensor.ndim != 2 or tensor.shape[1] != self.head_dim:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return False
        for tensor in src_v:
            if tensor.dtype != dst_dtype or tensor.ndim != 2 or tensor.shape[1] != self.head_dim:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return False
        for tensor in src_pos:
            if tensor.dtype != torch.long or tensor.ndim != 2 or tensor.shape[1] != 3:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return False

        self._ensure_cuda_refresh_descriptor_buffers(n_seg, dst_device)
        assert self._cuda_refresh_src_ptrs_k is not None
        assert self._cuda_refresh_src_ptrs_v is not None
        assert self._cuda_refresh_src_ptrs_pos is not None
        assert self._cuda_refresh_offsets is not None
        assert self._cuda_refresh_lengths is not None
        assert self._cuda_refresh_flags is not None
        assert self._cuda_refresh_dynamic_rope_t is not None

        desc_key = tuple(desc_parts)
        if desc_key != self._cuda_refresh_desc_key:
            self._cuda_refresh_offsets[:n_seg].copy_(
                torch.tensor(offsets, dtype=torch.int64, device=dst_device), non_blocking=True
            )
            self._cuda_refresh_lengths[:n_seg].copy_(
                torch.tensor(lengths, dtype=torch.int64, device=dst_device), non_blocking=True
            )
            self._cuda_refresh_flags[:n_seg].copy_(
                torch.tensor(flags, dtype=torch.int64, device=dst_device), non_blocking=True
            )
            self._cuda_refresh_dynamic_rope_t[:n_seg].copy_(
                torch.tensor(dynamic_rope_t, dtype=torch.int64, device=dst_device), non_blocking=True
            )
            self._cuda_refresh_desc_key = desc_key

        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()
        total_len = self._readout_cache_total_len
        try:
            refresh_readout_layout(
                src_k,
                src_v,
                src_pos,
                self._cuda_refresh_src_ptrs_k[:n_seg],
                self._cuda_refresh_src_ptrs_v[:n_seg],
                self._cuda_refresh_src_ptrs_pos[:n_seg],
                self._cuda_refresh_offsets[:n_seg],
                self._cuda_refresh_lengths[:n_seg],
                self._cuda_refresh_flags[:n_seg],
                self._cuda_refresh_dynamic_rope_t[:n_seg],
                self._ws_k_raw[:total_len],
                self._ws_v[:total_len],
                self._ws_rope_pos[:total_len],
                self._ws_frame_ids[:total_len],
                self.head_dim,
                int(sync_t_raw),
                self.history_time_mapping_mode,
                int(self.history_relative_t_max),
                float(self.history_time_soft_factor),
            )
            rope_start = perf_counter()
            apply_rope_to_flat_k(
                self._ws_k_raw[:total_len],
                self._ws_rope_pos[:total_len],
                freqs=freqs,
                freq_parts=freq_parts,
                out=self._ws_k[:total_len],
            )
            self._record_profile("rope_ms", rope_start)
        except Exception:
            self._cuda_refresh_disabled = True
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return False

        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._profile_stats["refresh_pack_count"] += 1.0
        self._profile_stats["cuda_refresh_count"] += 1.0
        self._profile_stats["cuda_refresh_segment_count"] += float(n_seg)
        self._profile_stats["cuda_refresh_total_len"] += float(sum(lengths))
        self._profile_stats["readout_total_len"] = float(total_len)
        self._profile_stats["readout_max_seqlen"] = float(self._readout_cache_max_seqlen)
        self._readout_cache_tail_dirty = False
        self._readout_tail_write_through_valid = False
        return True

    def _try_cuda_refresh_readout_spec(
        self,
        spec: _ReadoutSpec,
        *,
        current_start: int,
        sync_t_raw: int,
        frame_seqlen: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor] | None:
        if not self.readout_cache_enabled or not self._cuda_refresh_opt_in():
            return None
        if self._cuda_refresh_disabled:
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return None
        if not freqs.is_cuda or not cuda_refresh_available():
            self._cuda_refresh_disabled = True
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return None

        num_seq = self.batch_size * self.num_heads
        device = freqs.device
        dtype = torch.bfloat16
        for seg in spec.segments:
            if seg.length > 0:
                dtype = seg.k.dtype
                break

        k_raw, k_flat, v_flat, frame_ids_flat, cu_seqlens_k, rope_pos_flat = self._ensure_workspace(
            spec.total_len, num_seq, device, dtype,
        )
        self._copy_values_to_buffer(cu_seqlens_k[: num_seq + 1], spec.cu_cpu)

        src_k: list[torch.Tensor] = []
        src_v: list[torch.Tensor] = []
        src_pos: list[torch.Tensor] = []
        offsets: list[int] = []
        lengths: list[int] = []
        flags: list[int] = []
        dynamic_rope_t: list[int] = []
        desc_parts = []
        for seg in spec.segments:
            if seg.length <= 0:
                continue
            flag = 0
            if seg.dynamic_time_map:
                flag |= _CUDA_REFRESH_FLAG_DYNAMIC_TIME_MAP
            if seg.frame_ids_physical:
                flag |= _CUDA_REFRESH_FLAG_FRAME_IDS_PHYSICAL
            if seg.dynamic_rope_t is not None:
                flag |= _CUDA_REFRESH_FLAG_DYNAMIC_ROPE
            src_k.append(seg.k)
            src_v.append(seg.v)
            src_pos.append(seg.pos)
            offsets.append(int(seg.offset))
            lengths.append(int(seg.length))
            flags.append(flag)
            dynamic_rope = int(seg.dynamic_rope_t) if seg.dynamic_rope_t is not None else 0
            dynamic_rope_t.append(dynamic_rope)
            desc_parts.append((seg.kind, seg.seq_idx, int(seg.offset), int(seg.length), flag, dynamic_rope))

        n_seg = len(src_k)
        if n_seg == 0:
            self._profile_stats["cuda_refresh_count"] += 1.0
            self._profile_stats["readout_total_len"] = float(spec.total_len)
            self._profile_stats["readout_max_seqlen"] = float(spec.max_seqlen)
            self._readout_static_specs = spec.static_specs
            self._readout_tail_specs = spec.tail_specs
            self._cache_readout_layout(
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                frame_seqlen=frame_seqlen,
                total_len=spec.total_len,
                max_seqlen_k=spec.max_seqlen,
                shape_key=spec.shape_key,
            )
            return k_flat, v_flat, cu_seqlens_k, spec.max_seqlen, frame_ids_flat

        for tensors in (src_k, src_v, src_pos):
            for tensor in tensors:
                if not tensor.is_cuda or tensor.device != device or not tensor.is_contiguous():
                    self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                    return None
        for tensor in src_k:
            if tensor.dtype != dtype or tensor.ndim != 2 or tensor.shape[1] != self.head_dim:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return None
        for tensor in src_v:
            if tensor.dtype != dtype or tensor.ndim != 2 or tensor.shape[1] != self.head_dim:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return None
        for tensor in src_pos:
            if tensor.dtype != torch.long or tensor.ndim != 2 or tensor.shape[1] != 3:
                self._profile_stats["cuda_refresh_fallback_count"] += 1.0
                return None

        if spec.total_len == 0:
            self._profile_stats["cuda_refresh_count"] += 1.0
            self._profile_stats["readout_total_len"] = float(spec.total_len)
            self._profile_stats["readout_max_seqlen"] = float(spec.max_seqlen)
            self._readout_static_specs = spec.static_specs
            self._readout_tail_specs = spec.tail_specs
            self._cache_readout_layout(
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                frame_seqlen=frame_seqlen,
                total_len=spec.total_len,
                max_seqlen_k=spec.max_seqlen,
                shape_key=spec.shape_key,
            )
            return k_flat, v_flat, cu_seqlens_k, spec.max_seqlen, frame_ids_flat

        desc_key = ("spec", spec.shape_key, tuple(desc_parts))
        self._prepare_cuda_refresh_descriptor_buffers(
            src_k=src_k,
            src_v=src_v,
            src_pos=src_pos,
            offsets=offsets,
            lengths=lengths,
            flags=flags,
            dynamic_rope_t=dynamic_rope_t,
            desc_key=desc_key,
            device=device,
        )

        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()
        try:
            if n_seg > 0:
                refresh_readout_layout(
                    src_k,
                    src_v,
                    src_pos,
                    self._cuda_refresh_src_ptrs_k[:n_seg],
                    self._cuda_refresh_src_ptrs_v[:n_seg],
                    self._cuda_refresh_src_ptrs_pos[:n_seg],
                    self._cuda_refresh_offsets[:n_seg],
                    self._cuda_refresh_lengths[:n_seg],
                    self._cuda_refresh_flags[:n_seg],
                    self._cuda_refresh_dynamic_rope_t[:n_seg],
                    k_raw[:spec.total_len],
                    v_flat[:spec.total_len],
                    rope_pos_flat[:spec.total_len],
                    frame_ids_flat[:spec.total_len],
                    self.head_dim,
                    int(sync_t_raw),
                    self.history_time_mapping_mode,
                    int(self.history_relative_t_max),
                    float(self.history_time_soft_factor),
                )
            self._materialize_cpp_readout_segments(
                spec,
                k_raw=k_raw[:spec.total_len],
                v_flat=v_flat[:spec.total_len],
                rope_pos_flat=rope_pos_flat[:spec.total_len],
                frame_ids_flat=frame_ids_flat[:spec.total_len],
            )
            rope_start = perf_counter()
            apply_rope_to_flat_k(
                k_raw[:spec.total_len],
                rope_pos_flat[:spec.total_len],
                freqs=freqs,
                freq_parts=freq_parts,
                out=k_flat[:spec.total_len],
            )
            self._record_profile("rope_ms", rope_start)
        except Exception:
            self._cuda_refresh_disabled = True
            self._profile_stats["cuda_refresh_fallback_count"] += 1.0
            return None

        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._profile_stats["refresh_pack_count"] += 1.0
        self._profile_stats["cuda_refresh_count"] += 1.0
        self._profile_stats["cuda_refresh_segment_count"] += float(n_seg)
        self._profile_stats["cuda_refresh_total_len"] += float(sum(lengths))
        self._profile_stats["readout_total_len"] = float(spec.total_len)
        self._profile_stats["readout_max_seqlen"] = float(spec.max_seqlen)
        self._readout_static_specs = spec.static_specs
        self._readout_tail_specs = spec.tail_specs
        self._cache_readout_layout(
            current_start=current_start,
            sync_t_raw=sync_t_raw,
            frame_seqlen=frame_seqlen,
            total_len=spec.total_len,
            max_seqlen_k=spec.max_seqlen,
            shape_key=spec.shape_key,
        )
        return k_flat, v_flat, cu_seqlens_k, spec.max_seqlen, frame_ids_flat

    def _refresh_cached_readout_mutable_segments(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        sync_t: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        if not self._readout_cache_valid or self._ws_k_raw is None or self._ws_rope_pos is None:
            return False

        if self._cuda_refresh_opt_in() and self._ws_k_raw.is_cuda and freqs.is_cuda:
            if self._try_cuda_refresh_cached_readout(
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                sync_t=sync_t,
                freqs=freqs,
                freq_parts=freq_parts,
            ):
                return True

        capture_physical = self.capture_frame_id_mode == "physical"
        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()
        rewrite_static = int(current_start) == 0
        num_seq = self.batch_size * self.num_heads
        tail_write_through = self._readout_tail_write_through_valid

        # M7: Steady-state fast-path for refresh-dirty.
        # When current_start > 0 (not first block), static rewrite is skipped.
        # All heads just need tail content refresh + time mapping + RoPE.
        # Use local aliases and skip null checks / profiling per-iter.
        if (
            not rewrite_static
            and self._steady_state_reached
            and os.environ.get("PYRAMIDKV_DISABLE_M6_FASTPATH") != "1"
        ):
            ws_k_raw = self._ws_k_raw
            ws_k = self._ws_k
            ws_v = self._ws_v
            ws_rope_pos = self._ws_rope_pos
            ws_frame_ids = self._ws_frame_ids
            dyn_k_list = self.dynamic_k
            dyn_v_list = self.dynamic_v
            dyn_pos_list = self.dynamic_pos
            tail_specs = self._readout_tail_specs
            do_time_map = self.history_time_mapping_mode != "none"
            time_mode = self.history_time_mapping_mode
            t_max = self.history_relative_t_max
            t_soft = self.history_time_soft_factor

            for i in range(num_seq):
                spec = tail_specs[i]
                if spec is None:
                    continue
                offset, tail_len = spec
                dk = dyn_k_list[i]
                dv = dyn_v_list[i]
                dp = dyn_pos_list[i]
                if dk is None or dk.shape[0] < tail_len:
                    return False
                rp = ws_rope_pos[offset:offset + tail_len]
                if not tail_write_through:
                    ws_k_raw[offset:offset + tail_len] = dk[-tail_len:]
                    ws_v[offset:offset + tail_len] = dv[-tail_len:]
                    rp.copy_(dp[-tail_len:])
                if do_time_map:
                    map_dynamic_pos_time(
                        rp, current_t=sync_t_raw,
                        history_time_mapping_mode=time_mode,
                        history_relative_t_max=t_max,
                        history_time_soft_factor=t_soft,
                        inplace=True,
                    )
                if capture_physical:
                    ws_frame_ids[offset:offset + tail_len] = _as_long_tensor(dp[-tail_len:, 0])
                else:
                    ws_frame_ids[offset:offset + tail_len] = _as_long_tensor(rp[:, 0])
                apply_rope_to_flat_k(
                    ws_k_raw[offset:offset + tail_len],
                    rp,
                    freqs=freqs,
                    freq_parts=freq_parts,
                    out=ws_k[offset:offset + tail_len],
                )

            rope_delta = (perf_counter() - pack_start) * 1000.0 * 0.5
            self._profile_stats["rope_ms"] += rope_delta
            self._profile_stats["pack_ms"] += max(0.0, (perf_counter() - pack_start) * 1000.0 - rope_delta)
            self._profile_stats["refresh_pack_count"] += 1.0
            self._readout_cache_tail_dirty = False
            self._readout_tail_write_through_valid = False
            return True

        # Original slow path (handles rewrite_static + non-steady cases)
        for i in range(num_seq):
            head_idx = i % self.num_heads
            static_spec = self._readout_static_specs[i]
            if rewrite_static and static_spec is not None:
                stat_k = self.static_k[i]
                stat_v = self.static_v[i]
                stat_pos = self.static_pos[i]
                offset, n_s = static_spec
                if (
                    stat_k is None
                    or stat_v is None
                    or stat_pos is None
                    or stat_k.shape[0] != n_s
                ):
                    return False
                self._ws_k_raw[offset:offset + n_s] = stat_k
                self._ws_v[offset:offset + n_s] = stat_v
                rope_pos = self._ws_rope_pos[offset:offset + n_s]
                rope_pos.copy_(stat_pos)
                if self.decouple_head_flags[head_idx]:
                    rope_pos[:, 0] = sync_t
                    if capture_physical:
                        self._ws_frame_ids[offset:offset + n_s] = _as_long_tensor(stat_pos[:, 0])
                    else:
                        self._ws_frame_ids[offset:offset + n_s] = sync_t
                else:
                    self._ws_frame_ids[offset:offset + n_s] = _as_long_tensor(stat_pos[:, 0])
                rope_start = perf_counter()
                apply_rope_to_flat_k(
                    self._ws_k_raw[offset:offset + n_s],
                    rope_pos,
                    freqs=freqs,
                    freq_parts=freq_parts,
                    out=self._ws_k[offset:offset + n_s],
                )
                self._record_profile("rope_ms", rope_start)

            tail_spec = self._readout_tail_specs[i]
            if tail_spec is None:
                continue
            dyn_k = self.dynamic_k[i]
            dyn_v = self.dynamic_v[i]
            dyn_pos = self.dynamic_pos[i]
            offset, tail_len = tail_spec
            if (
                dyn_k is None
                or dyn_v is None
                or dyn_pos is None
                or dyn_k.shape[0] < tail_len
            ):
                return False
            dyn_k_tail = dyn_k[-tail_len:]
            dyn_v_tail = dyn_v[-tail_len:]
            dyn_pos_tail = dyn_pos[-tail_len:]

            rope_pos = self._ws_rope_pos[offset:offset + tail_len]
            if not tail_write_through:
                self._ws_k_raw[offset:offset + tail_len] = dyn_k_tail
                self._ws_v[offset:offset + tail_len] = dyn_v_tail
                rope_pos.copy_(dyn_pos_tail)
            if self.history_time_mapping_mode != "none":
                map_dynamic_pos_time(
                    rope_pos,
                    current_t=sync_t_raw,
                    history_time_mapping_mode=self.history_time_mapping_mode,
                    history_relative_t_max=self.history_relative_t_max,
                    history_time_soft_factor=self.history_time_soft_factor,
                    inplace=True,
                )
            if capture_physical:
                self._ws_frame_ids[offset:offset + tail_len] = _as_long_tensor(dyn_pos_tail[:, 0])
            else:
                self._ws_frame_ids[offset:offset + tail_len] = _as_long_tensor(rope_pos[:, 0])
            rope_start = perf_counter()
            apply_rope_to_flat_k(
                self._ws_k_raw[offset:offset + tail_len],
                rope_pos,
                freqs=freqs,
                freq_parts=freq_parts,
                out=self._ws_k[offset:offset + tail_len],
            )
            self._record_profile("rope_ms", rope_start)

        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._profile_stats["refresh_pack_count"] += 1.0
        self._readout_cache_tail_dirty = False
        self._readout_tail_write_through_valid = False
        return True

    def get_decoupled_flat_kv(
        self,
        current_start: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, _ = self.get_decoupled_flat_kv_and_frames(
            current_start=current_start,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k

    def _collect_middle_cache(
        self,
        seq_idx: int,
        head_idx: int,
        sync_t_raw: int,
        has_stat: bool,
    ) -> tuple[tuple[str, object], int]:
        n = 0
        composition = (
            self.compositions_row[head_idx]
            if self.compositions_row is not None and head_idx < len(self.compositions_row)
            else None
        )
        collect_start = perf_counter()
        if composition is not None and composition.has_middle:
            tail_min_t = sync_t_raw - composition.recent_frames + 1
            sink_max_t = 0 if has_stat else -1
            cpp_collected = self._try_cpp_strategy_collect(
                seq_idx=seq_idx,
                head_idx=head_idx,
                sync_t_raw=sync_t_raw,
                tail_min_t=tail_min_t,
                sink_max_t=sink_max_t,
            )
            if cpp_collected is not None:
                self._record_profile("collect_ms", collect_start)
                return ("comp", cpp_collected), sum(int(anchor.token_count) for anchor in cpp_collected)
            collected = composition.collect_all(seq_idx, sync_t_raw, tail_min_t, sink_max_t)
            self._record_profile("collect_ms", collect_start)
            for anchor in collected:
                n += int(anchor.token_count)
                if getattr(anchor, "source_kind", "tensor") == "anchor_store":
                    self._profile_stats["anchor_store_anchor_count"] += 1.0
                    self._profile_stats["anchor_store_token_count"] += float(anchor.token_count)
            return ("comp", collected), n

        inline = []
        if self.use_osc_frame_mode and self._is_phase_sink_head(head_idx):
            phase_idx = sync_t_raw % self.phase_period
            tail_min_t_cyc = sync_t_raw - self._head_recent_frames(head_idx) + 1
            for anchor in self.cyclic_buckets[seq_idx][phase_idx]:
                anchor_t = anchor[3]
                if has_stat and anchor_t == 0:
                    continue
                if anchor_t >= tail_min_t_cyc:
                    continue
                inline.append(("cyc", anchor))
                n += anchor[0].shape[0]
        if self._is_phase_sink_head(head_idx):
            lag_offsets = self._head_lag_offsets(head_idx)
            if len(lag_offsets) > 0:
                tail_min_t_lag = sync_t_raw - self._head_recent_frames(head_idx) + 1
            for lag in lag_offsets:
                target_t = sync_t_raw - lag
                if target_t < 0:
                    continue
                if has_stat and target_t == 0:
                    continue
                if target_t >= tail_min_t_lag:
                    continue
                anchor = self._find_anchor_by_t(self.lag_anchor_frames[seq_idx], target_t)
                if anchor is None:
                    continue
                inline.append(("lag", lag, anchor))
                n += anchor[0].shape[0]
        self._record_profile("collect_ms", collect_start)
        return ("inline", inline), n

    def _write_anchor_segment(
        self,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
        out_frame_ids: torch.Tensor,
        offset: int,
        *,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        anchor_pos: torch.Tensor,
        effective_t: int | None,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        capture_physical: bool,
    ) -> int:
        n = anchor_k.shape[0]
        out_v[offset:offset + n] = anchor_v
        if effective_t is None:
            rope_pos = anchor_pos
            mapped_t = _as_long_tensor(anchor_pos[:, 0])
        else:
            rope_pos = anchor_pos.clone()
            rope_pos[:, 0] = int(effective_t)
            mapped_t = _as_long_tensor(rope_pos[:, 0])
        rope_start = perf_counter()
        out_k[offset:offset + n] = apply_rope_to_flat_k(
            anchor_k,
            rope_pos,
            freqs=freqs,
            freq_parts=freq_parts,
        )
        self._record_profile("rope_ms", rope_start)
        if capture_physical:
            out_frame_ids[offset:offset + n] = _as_long_tensor(anchor_pos[:, 0])
        else:
            out_frame_ids[offset:offset + n] = mapped_t
        return offset + n

    def _build_readout_spec(
        self,
        *,
        sync_t_raw: int,
        sync_t: int,
    ) -> _ReadoutSpec:
        num_seq = self.batch_size * self.num_heads
        capture_physical = self.capture_frame_id_mode == "physical"
        has_dynamic_time_mapping = self.history_time_mapping_mode != "none"

        segments: list[_ReadoutSegment] = []
        cpp_segments: list[_CppReadoutSegment] = []
        lengths = [0] * num_seq
        cu_cpu = [0] * (num_seq + 1)
        static_specs: list[tuple[int, int] | None] = [None] * num_seq
        tail_specs: list[tuple[int, int] | None] = [None] * num_seq
        shape_parts = []
        anchor_shape_parts = []

        offset = 0
        for i in range(num_seq):
            head_idx = i % self.num_heads
            seq_start = offset
            stat_k = self.static_k[i]
            has_stat = stat_k is not None and stat_k.shape[0] > 0

            if has_stat:
                stat_v = self.static_v[i]
                stat_pos = self.static_pos[i]
                n_s = int(stat_k.shape[0])
                static_specs[i] = (offset, n_s)
                dynamic_rope_t = sync_t if self.decouple_head_flags[head_idx] else None
                segments.append(
                    _ReadoutSegment(
                        kind="static",
                        seq_idx=i,
                        offset=offset,
                        length=n_s,
                        k=stat_k,
                        v=stat_v,
                        pos=stat_pos,
                        dynamic_rope_t=dynamic_rope_t,
                        dynamic_time_map=False,
                        frame_ids_physical=capture_physical or dynamic_rope_t is None,
                    )
                )
                shape_parts.append((i, "static", n_s, dynamic_rope_t is not None, False, capture_physical))
                offset += n_s

            dyn_k = self.dynamic_k[i]
            if dyn_k is not None and dyn_k.shape[0] > 0:
                dyn_v = self.dynamic_v[i]
                dyn_pos = self.dynamic_pos[i]
                n_d = int(dyn_k.shape[0])
                dyn_offset = offset
                segments.append(
                    _ReadoutSegment(
                        kind="dynamic",
                        seq_idx=i,
                        offset=offset,
                        length=n_d,
                        k=dyn_k,
                        v=dyn_v,
                        pos=dyn_pos,
                        dynamic_rope_t=None,
                        dynamic_time_map=has_dynamic_time_mapping,
                        frame_ids_physical=capture_physical,
                    )
                )
                shape_parts.append((i, "dynamic", n_d, False, has_dynamic_time_mapping, capture_physical))
                tail_len = min(max(0, int(self._current_block_token_len[i])), n_d)
                if tail_len > 0:
                    tail_specs[i] = (dyn_offset + n_d - tail_len, tail_len)
                offset += n_d

            composition = (
                self.compositions_row[head_idx]
                if self.compositions_row is not None and head_idx < len(self.compositions_row)
                else None
            )
            managed_count = None
            if composition is not None and composition.has_middle and self._cpp_strategy_head_supported(head_idx):
                tail_min_t = sync_t_raw - composition.recent_frames + 1
                sink_max_t = 0 if has_stat else -1
                managed_count = self._try_cpp_strategy_count(
                    seq_idx=i,
                    head_idx=head_idx,
                    sync_t_raw=sync_t_raw,
                    tail_min_t=tail_min_t,
                    sink_max_t=sink_max_t,
                )
                if managed_count is not None:
                    if managed_count.token_count > 0:
                        dynamic_rope_t = sync_t if managed_count.dynamic_rope else None
                        cpp_segments.append(
                            _CppReadoutSegment(
                                seq_idx=i,
                                head_idx=head_idx,
                                offset=offset,
                                length=int(managed_count.token_count),
                                sync_t_raw=sync_t_raw,
                                tail_min_t=tail_min_t,
                                sink_max_t=sink_max_t,
                                dynamic_rope_t=dynamic_rope_t,
                                frame_ids_physical=capture_physical,
                            )
                        )
                        for anchor_idx, n_a in enumerate(managed_count.anchor_lengths):
                            part = (
                                i,
                                "anchor",
                                "cpp",
                                int(managed_count.kind),
                                anchor_idx,
                                int(n_a),
                                dynamic_rope_t is not None,
                                capture_physical,
                            )
                            shape_parts.append(part)
                            anchor_shape_parts.append(part)
                        offset += int(managed_count.token_count)

            if managed_count is not None:
                lengths[i] = offset - seq_start
                cu_cpu[i + 1] = offset
                continue

            anchor_cache, _anchor_n = self._collect_middle_cache(i, head_idx, sync_t_raw, has_stat)
            anchor_type, anchor_data = anchor_cache
            if anchor_type == "comp":
                for anchor_idx, anchor in enumerate(anchor_data):
                    n_a = int(anchor.k.shape[0])
                    dynamic_rope_t = sync_t if anchor.dynamic_rope else None
                    segments.append(
                        _ReadoutSegment(
                            kind="anchor",
                            seq_idx=i,
                            offset=offset,
                            length=n_a,
                            k=anchor.k,
                            v=anchor.v,
                            pos=anchor.pos,
                            dynamic_rope_t=dynamic_rope_t,
                            dynamic_time_map=False,
                            frame_ids_physical=capture_physical,
                        )
                    )
                    part = (i, "anchor", "comp", anchor_idx, n_a, dynamic_rope_t is not None, capture_physical)
                    shape_parts.append(part)
                    anchor_shape_parts.append(part)
                    offset += n_a
            else:
                for anchor_idx, anchor_info in enumerate(anchor_data):
                    if anchor_info[0] == "cyc":
                        anchor_k, anchor_v, anchor_pos, _anchor_t = anchor_info[1]
                        n_a = int(anchor_k.shape[0])
                        dynamic_rope_t = sync_t if self.phase_sink_dynamic_rope else None
                        anchor_kind = "cyc"
                    else:
                        lag = anchor_info[1]
                        anchor_k, anchor_v, anchor_pos, _ = anchor_info[2]
                        n_a = int(anchor_k.shape[0])
                        dynamic_rope_t = max(0, sync_t - lag) if self.osc_lag_dynamic_rope else None
                        anchor_kind = "lag"
                    segments.append(
                        _ReadoutSegment(
                            kind="anchor",
                            seq_idx=i,
                            offset=offset,
                            length=n_a,
                            k=anchor_k,
                            v=anchor_v,
                            pos=anchor_pos,
                            dynamic_rope_t=dynamic_rope_t,
                            dynamic_time_map=False,
                            frame_ids_physical=capture_physical,
                        )
                    )
                    part = (i, "anchor", anchor_kind, anchor_idx, n_a, dynamic_rope_t is not None, capture_physical)
                    shape_parts.append(part)
                    anchor_shape_parts.append(part)
                    offset += n_a

            lengths[i] = offset - seq_start
            cu_cpu[i + 1] = offset

        return _ReadoutSpec(
            segments=segments,
            cpp_segments=cpp_segments,
            lengths=lengths,
            cu_cpu=cu_cpu,
            total_len=offset,
            max_seqlen=max(lengths) if lengths else 0,
            static_specs=static_specs,
            tail_specs=tail_specs,
            shape_key=tuple(shape_parts),
            anchor_shape_key=tuple(anchor_shape_parts),
        )

    def _record_readout_shape_stats(self, spec: _ReadoutSpec) -> None:
        if self._last_readout_shape_key is not None and spec.shape_key != self._last_readout_shape_key:
            self._profile_stats["layout_shape_changed_count"] += 1.0
        if (
            self._last_readout_anchor_shape_key is not None
            and spec.anchor_shape_key != self._last_readout_anchor_shape_key
        ):
            self._profile_stats["layout_anchor_changed_count"] += 1.0
        self._last_readout_shape_key = spec.shape_key
        self._last_readout_anchor_shape_key = spec.anchor_shape_key

    def _materialize_readout_spec(
        self,
        spec: _ReadoutSpec,
        *,
        current_start: int,
        sync_t_raw: int,
        frame_seqlen: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        num_seq = self.batch_size * self.num_heads
        device = freqs.device
        dtype = torch.bfloat16
        for seg in spec.segments:
            if seg.length > 0:
                dtype = seg.k.dtype
                break

        k_raw, k_flat, v_flat, frame_ids_flat, cu_seqlens_k, rope_pos_flat = self._ensure_workspace(
            spec.total_len, num_seq, device, dtype,
        )
        self._copy_values_to_buffer(cu_seqlens_k[: num_seq + 1], spec.cu_cpu)

        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()

        if spec.total_len > 0:
            if spec.segments:
                src_k_list: list[torch.Tensor] = []
                src_v_list: list[torch.Tensor] = []
                src_pos_list: list[torch.Tensor] = []
                offsets: list[int] = []
                lengths: list[int] = []
                flags: list[int] = []
                dynamic_rope_t: list[int] = []
                override_starts: list[int] = []
                override_ends: list[int] = []
                override_vals: list[int] = []
                desc_parts = []
                for seg in spec.segments:
                    src_k_list.append(seg.k)
                    src_v_list.append(seg.v)
                    src_pos_list.append(seg.pos)
                    offsets.append(int(seg.offset))
                    lengths.append(int(seg.length))
                    flag = 0
                    if seg.dynamic_time_map:
                        flag |= _CUDA_REFRESH_FLAG_DYNAMIC_TIME_MAP
                    if seg.frame_ids_physical:
                        flag |= _CUDA_REFRESH_FLAG_FRAME_IDS_PHYSICAL
                    if seg.dynamic_rope_t is not None:
                        flag |= _CUDA_REFRESH_FLAG_DYNAMIC_ROPE
                    flags.append(flag)
                    rope_t = int(seg.dynamic_rope_t) if seg.dynamic_rope_t is not None else 0
                    dynamic_rope_t.append(rope_t)
                    desc_parts.append((seg.kind, seg.seq_idx, int(seg.offset), int(seg.length), flag, rope_t))
                    if seg.dynamic_rope_t is not None:
                        override_starts.append(int(seg.offset))
                        override_ends.append(int(seg.offset + seg.length))
                        override_vals.append(rope_t)

                if os.environ.get("PYRAMIDKV_PTR_DEBUG"):
                    for _dbg_i, _dbg_t in enumerate(src_k_list):
                        _dbg_st = _dbg_t.untyped_storage()
                        print(
                            f"[B{current_start}] seg{_dbg_i:02d} "
                            f"storage={_dbg_st.data_ptr():#x} "
                            f"data={_dbg_t.data_ptr():#x} "
                            f"shape={tuple(_dbg_t.shape)} "
                            f"contig={_dbg_t.is_contiguous()} "
                            f"is_view={_dbg_t._base is not None}",
                            flush=True,
                        )

                if scatter_available() and k_raw.is_cuda:
                    pinned_k = [t if t.is_contiguous() else t.contiguous() for t in src_k_list]
                    pinned_v = [t if t.is_contiguous() else t.contiguous() for t in src_v_list]
                    pinned_p = [t if t.is_contiguous() else t.contiguous() for t in src_pos_list]
                    desc_key = ("spec", spec.shape_key, tuple(desc_parts))
                    self._prepare_cuda_refresh_descriptor_buffers(
                        src_k=pinned_k,
                        src_v=pinned_v,
                        src_pos=pinned_p,
                        offsets=offsets,
                        lengths=lengths,
                        flags=flags,
                        dynamic_rope_t=dynamic_rope_t,
                        desc_key=desc_key,
                        device=k_raw.device,
                    )
                    n_seg = len(pinned_k)
                    scatter_copy(
                        self._cuda_refresh_src_ptrs_k[:n_seg],
                        self._cuda_refresh_lengths[:n_seg],
                        self._cuda_refresh_offsets[:n_seg],
                        k_raw[:spec.total_len],
                        self.head_dim,
                    )
                    scatter_copy(
                        self._cuda_refresh_src_ptrs_v[:n_seg],
                        self._cuda_refresh_lengths[:n_seg],
                        self._cuda_refresh_offsets[:n_seg],
                        v_flat[:spec.total_len],
                        self.head_dim,
                    )
                    scatter_copy(
                        self._cuda_refresh_src_ptrs_pos[:n_seg],
                        self._cuda_refresh_lengths[:n_seg],
                        self._cuda_refresh_offsets[:n_seg],
                        rope_pos_flat[:spec.total_len],
                        3,
                    )
                    if override_starts:
                        n_override = len(override_starts)
                        self._copy_values_to_buffer(self._cuda_refresh_override_starts, override_starts)
                        self._copy_values_to_buffer(self._cuda_refresh_override_ends, override_ends)
                        self._copy_values_to_buffer(self._cuda_refresh_override_vals, override_vals)
                        apply_pos_override(
                            rope_pos_flat[:spec.total_len],
                            self._cuda_refresh_override_starts[:n_override],
                            self._cuda_refresh_override_ends[:n_override],
                            self._cuda_refresh_override_vals[:n_override],
                        )
                    del pinned_k, pinned_v, pinned_p
                else:
                    for seg in spec.segments:
                        start = seg.offset
                        end = start + seg.length
                        k_raw[start:end].copy_(seg.k)
                        v_flat[start:end].copy_(seg.v)
                        rope_pos_flat[start:end].copy_(seg.pos)
                        if seg.dynamic_rope_t is not None:
                            rope_pos_flat[start:end, 0] = int(seg.dynamic_rope_t)

            self._materialize_cpp_readout_segments(
                spec,
                k_raw=k_raw,
                v_flat=v_flat,
                rope_pos_flat=rope_pos_flat,
                frame_ids_flat=frame_ids_flat,
            )

            for seg in spec.segments:
                start = seg.offset
                end = start + seg.length
                if seg.dynamic_time_map:
                    map_dynamic_pos_time(
                        rope_pos_flat[start:end],
                        current_t=sync_t_raw,
                        history_time_mapping_mode=self.history_time_mapping_mode,
                        history_relative_t_max=self.history_relative_t_max,
                        history_time_soft_factor=self.history_time_soft_factor,
                        inplace=True,
                    )
                if seg.frame_ids_physical:
                    frame_ids_flat[start:end] = _as_long_tensor(seg.pos[:, 0])
                else:
                    frame_ids_flat[start:end] = _as_long_tensor(rope_pos_flat[start:end, 0])

            rope_start = perf_counter()
            apply_rope_to_flat_k(
                k_raw[:spec.total_len],
                rope_pos_flat[:spec.total_len],
                freqs=freqs,
                freq_parts=freq_parts,
                out=k_flat[:spec.total_len],
            )
            self._record_profile("rope_ms", rope_start)

        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._profile_stats["cold_pack_count"] += 1.0
        self._profile_stats["readout_total_len"] = float(spec.total_len)
        self._profile_stats["readout_max_seqlen"] = float(spec.max_seqlen)
        self._readout_static_specs = spec.static_specs
        self._readout_tail_specs = spec.tail_specs
        self._cache_readout_layout(
            current_start=current_start,
            sync_t_raw=sync_t_raw,
            frame_seqlen=frame_seqlen,
            total_len=spec.total_len,
            max_seqlen_k=spec.max_seqlen,
            shape_key=spec.shape_key,
        )
        if spec.total_len > 0:
            self._shadow_swap_v(v_flat[:spec.total_len], spec)
            self._shadow_assert_v(v_flat[:spec.total_len], spec=spec)
        return k_flat, v_flat, cu_seqlens_k, spec.max_seqlen, frame_ids_flat

    def get_decoupled_flat_kv_and_frames(
        self,
        current_start: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Build flattened KV for sink-grid decoupling:
        - static sink is rotated to the current query frame index (time-synchronized),
        - dynamic history is rotated with its own saved position ids.

        Two-pass implementation:
          1. compute per-head token counts (no tensor allocation),
          2. write K/V/pos directly into pre-allocated workspace buffers, then
             apply RoPE in a single batched call (no clone/cat/scatter).
        """
        if not self.sink_grid_decoupling:
            k_flat, v_flat, cu_seqlens_k, max_seqlen_k = self.get_flat_kv()
            frame_ids = (
                _as_long_tensor(self.last_flat_pos_ids[:, 0])
                if self.last_flat_pos_ids is not None
                else torch.empty(0, dtype=torch.long, device=k_flat.device)
            )
            self._update_steady_state(cu_seqlens_k)
            return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, frame_ids

        if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
            raise ValueError(f"grid_sizes must be [B,3], got {tuple(grid_sizes.shape)}")
        if self._frame_seqlen is None:
            frame_tokens = _as_long_tensor(grid_sizes[:, 1] * grid_sizes[:, 2])
            if torch.any(frame_tokens <= 0):
                raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
            if torch.unique(frame_tokens).numel() != 1:
                raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
            self._frame_seqlen = int(frame_tokens[0].item())
        frame_seqlen = self._frame_seqlen
        sync_t_raw = 0 if frame_seqlen <= 0 else int(current_start // frame_seqlen)
        sync_t = self._map_sink_time(sync_t_raw)

        # Pre-split freqs once for all heads (avoids ~150k redundant .split() calls)
        c = self.head_dim // 2
        split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
        freq_parts = tuple(freqs.split(split_sizes, dim=1))

        if self._can_reuse_readout_cache(current_start=current_start, sync_t_raw=sync_t_raw, frame_seqlen=frame_seqlen):
            if self._readout_cache_tail_dirty:
                if self._refresh_cached_readout_mutable_segments(
                    current_start=current_start,
                    sync_t_raw=sync_t_raw,
                    sync_t=sync_t,
                    freqs=freqs,
                    freq_parts=freq_parts,
                ):
                    return self._cached_readout_views()
                self._invalidate_readout_cache()
            else:
                self._profile_stats["layout_reuse_count"] += 1.0
                return self._cached_readout_views()

        # Path A+B: cs differs but layout still valid (multi-chunk readout
        # within one timestep). Cached K_RAW/V are reusable; only RoPE and
        # frame_ids need refresh for the new sync_t / sync_t_raw.
        # PYRAMIDKV_PATH_AB=0 disables (for A/B benchmarking).
        if os.environ.get("PYRAMIDKV_PATH_AB", "1") != "0" and self._can_reuse_readout_cache_for_rope_only(
            current_start=current_start, frame_seqlen=frame_seqlen,
        ):
            refreshed = self._refresh_rope_only_for_cs(
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                sync_t=sync_t,
                frame_seqlen=frame_seqlen,
                freqs=freqs,
                freq_parts=freq_parts,
            )
            if refreshed is not None:
                return refreshed

        spec = self._build_readout_spec(sync_t_raw=sync_t_raw, sync_t=sync_t)
        shape_matches_previous = (
            self._last_readout_shape_key is not None
            and spec.shape_key == self._last_readout_shape_key
        )
        self._record_readout_shape_stats(spec)
        if shape_matches_previous:
            refreshed = self._try_cuda_refresh_readout_spec(
                spec,
                current_start=current_start,
                sync_t_raw=sync_t_raw,
                frame_seqlen=frame_seqlen,
                freqs=freqs,
                freq_parts=freq_parts,
            )
            if refreshed is not None:
                if not self._steady_state_reached:
                    self._update_steady_state(refreshed[2])
                return refreshed
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, frame_ids_flat = self._materialize_readout_spec(
            spec,
            current_start=current_start,
            sync_t_raw=sync_t_raw,
            frame_seqlen=frame_seqlen,
            freqs=freqs,
            freq_parts=freq_parts,
        )

        if not self._steady_state_reached:
            self._update_steady_state(cu_seqlens_k)

        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, frame_ids_flat

    def get_decoupled_flat_kv_and_frames_multi(
        self,
        current_starts: list[int],
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Build merged KV for multiple query chunks in one pass.

        Calls get_decoupled_flat_kv_and_frames for each current_start,
        then returns concatenated K/V with a merged cu_seqlens_k.
        The individual results already reside in contiguous workspace
        buffers, so we just cat the views.
        """
        num_chunks = len(current_starts)
        if num_chunks == 1:
            return self.get_decoupled_flat_kv_and_frames(
                current_start=current_starts[0], grid_sizes=grid_sizes, freqs=freqs,
            )

        results = []
        for cs in current_starts:
            k, v, cu, maxlen, fids = self.get_decoupled_flat_kv_and_frames(
                current_start=cs, grid_sizes=grid_sizes, freqs=freqs,
            )
            # Clone all workspace views — subsequent calls overwrite the same buffers
            results.append((k.clone(), v.clone(), cu.clone(), maxlen, fids.clone()))

        num_seq = self.batch_size * self.num_heads

        # Concatenate K/V/frame_ids from all chunks
        k_cat = torch.cat([r[0] for r in results], dim=0)
        v_cat = torch.cat([r[1] for r in results], dim=0)
        frame_ids_cat = torch.cat([r[4] for r in results], dim=0)

        # Build merged cu_seqlens_k
        device = k_cat.device
        merged_cu = torch.zeros(num_chunks * num_seq + 1, dtype=torch.int32, device=device)
        running = 0
        for c_idx, (_, _, cu_c, _, _) in enumerate(results):
            cu_c_cpu = cu_c.cpu()
            for s in range(num_seq):
                running += int(cu_c_cpu[s + 1]) - int(cu_c_cpu[s])
                merged_cu[c_idx * num_seq + s + 1] = running

        max_seqlen_k = max(r[3] for r in results)
        return k_cat, v_cat, merged_cu, max_seqlen_k, frame_ids_cat

    @staticmethod
    def apply_rope_to_flat_k(k_flat: torch.Tensor, pos_3d: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return apply_rope_to_flat_k(k_flat, pos_3d, freqs)

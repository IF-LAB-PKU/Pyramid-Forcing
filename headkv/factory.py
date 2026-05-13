"""Factory for building HeadComposition instances from config parameters.

Replaces the old build_strategies() and build_heads() factories.
"""
from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence

import torch

from .base import HeadComposition, MiddleStrategy
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .merge import MergeStrategy
from .stride import StrideStrategy
from .recent import RecentStrategy

HEAD_LABEL_MAP = {
    -1: "oscillating",
    1: "stable",
    2: "stable_sparse",
}


def _normalize_label_key(key: object) -> str:
    raw = str(key).strip()
    if not raw:
        return ""
    try:
        return str(int(raw))
    except (TypeError, ValueError):
        return raw


def _map_items(user_map: Mapping | None):
    if not isinstance(user_map, Mapping):
        return ()
    return user_map.items()


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _build_int_map(user_map: Mapping | None, *, min_value: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        try:
            out[norm] = max(min_value, int(val))
        except (TypeError, ValueError):
            continue
    return out


def _build_capacity_map(user_map: Mapping | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        try:
            parsed = int(val)
        except (TypeError, ValueError):
            continue
        out[norm] = -1 if parsed < 0 else max(1, parsed)
    return out


def _build_bool_map(user_map: Mapping | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        out[norm] = bool(val)
    return out


def _build_offsets_map(user_map: Mapping | None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key, val in _map_items(user_map):
        norm = _normalize_label_key(key)
        if not norm:
            continue
        vals = _as_sequence(val)
        offsets: list[int] = []
        for item in vals:
            try:
                off = int(item)
            except (TypeError, ValueError):
                continue
            if off > 0:
                offsets.append(off)
        out[norm] = sorted(set(offsets))
    return out


def load_head_labels(
    csv_path: str,
    num_layers: int,
    num_heads: int,
) -> list[list[int]]:
    """Load head classification labels from CSV.

    CSV format: each row is a layer, each column is a head.
    Values: -1 (oscillating), 1 (stable), 2 (stable_sparse).
    """
    labels = [[1] * num_heads for _ in range(num_layers)]
    if not csv_path or not os.path.exists(csv_path):
        return labels
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    for layer_idx in range(min(num_layers, len(rows))):
        row = rows[layer_idx]
        for head_idx in range(min(num_heads, len(row))):
            try:
                labels[layer_idx][head_idx] = int(str(row[head_idx]).strip())
            except ValueError:
                continue
    return labels


def _label_to_policy_type(label: int) -> str:
    if label == -1:
        return "osc"
    if label == 2:
        return "recent_only"
    return "stride"


def _resolve_cyclic_enabled_for_label(
    *,
    label_key: str,
    is_osc: bool,
    cyclic_enabled: bool,
    cyclic_osc_only: bool,
    phase_map: dict[str, int],
    cyclic_bucket_cap: int,
) -> tuple[bool, int]:
    if not cyclic_enabled:
        return False, 0
    if label_key in phase_map:
        head_bucket_cap = phase_map[label_key]
        return head_bucket_cap > 0, head_bucket_cap
    if cyclic_osc_only and not is_osc:
        return False, 0
    if is_osc or not cyclic_osc_only:
        raise ValueError(
            f"Missing explicit cyclic resolution for label {label_key}: "
            "set headkv_label_phase_bucket_map to 0 or a positive capacity."
        )
    return False, 0


def _resolve_lag_offsets_for_label(
    *,
    label_key: str,
    is_osc: bool,
    lag_enabled: bool,
    cyclic_osc_only: bool,
    lag_map: dict[str, list[int]],
    lag_offsets: list[int] | None,
) -> list[int]:
    if not lag_enabled:
        return []
    if label_key in lag_map:
        return lag_map[label_key]
    if cyclic_osc_only and not is_osc:
        return []
    if (lag_offsets or []) and (is_osc or not cyclic_osc_only):
        raise ValueError(
            f"Missing explicit lag resolution for label {label_key}: "
            "set headkv_label_lag_offsets_map to [] or a non-empty list."
        )
    return list(lag_offsets or [])


def _resolve_stride_enabled_for_label(
    *,
    label_key: str,
    is_osc: bool,
    stride_enabled: bool,
    stride_map: dict[str, bool],
) -> bool:
    if is_osc:
        return False
    if label_key in stride_map:
        return stride_map[label_key]
    if stride_enabled:
        raise ValueError(
            f"Missing explicit stride resolution for label {label_key}: "
            "set headkv_label_stride_enabled_map to true or false."
        )
    return False


def _resolve_merge_enabled_for_label(
    *,
    label_key: str,
    merge_enabled: bool,
    merge_map: dict[str, bool],
) -> bool:
    if label_key in merge_map:
        return merge_map[label_key]
    if merge_enabled:
        raise ValueError(
            f"Missing explicit merge resolution for label {label_key}: "
            "set headkv_label_merge_enabled_map to true or false."
        )
    return False


def build_compositions(
    num_layers: int,
    num_heads: int,
    capacities: Sequence[Sequence[int]] | torch.Tensor,
    csv_path: str | None = None,
    *,
    # Cyclic params
    cyclic_enabled: bool = False,
    cyclic_period: int = 6,
    cyclic_bucket_cap: int = 1,
    cyclic_dynamic_rope: bool = True,
    cyclic_osc_only: bool = True,
    # Lag params
    lag_enabled: bool = False,
    lag_offsets: list[int] | None = None,
    lag_history: int = 21,
    lag_dynamic_rope: bool = False,
    # Stride params
    stride_enabled: bool = False,
    stride_interval: int = 6,
    stride_capacity: int = -1,
    stride_dynamic_rope: bool = True,
    # Merge params
    merge_enabled: bool = False,
    merge_patch_size: int = 2,
    merge_capacity: int = 1,
    merge_dynamic_rope: bool = True,
    # Sink/recent params
    osc_sink_frames: int | None = None,
    stable_sink_frames: int | None = None,
    recent_frames: int = 4,
    stable_recent_frames: int | None = None,
    label_sink_frames_map: dict | None = None,
    label_recent_frames_map: dict | None = None,
    label_stride_enabled_map: dict | None = None,
    label_stride_interval_map: dict | None = None,
    label_phase_bucket_map: dict | None = None,
    label_lag_offsets_map: dict | None = None,
    label_merge_enabled_map: dict | None = None,
    label_merge_patch_size_map: dict | None = None,
    label_merge_capacity_map: dict | None = None,
) -> list[list[HeadComposition]]:
    """Build per-layer, per-head HeadComposition instances.

    Returns list[list[HeadComposition]] indexed by [layer][head].
    """
    cap_tensor = (
        capacities
        if isinstance(capacities, torch.Tensor)
        else torch.as_tensor(capacities, dtype=torch.int32)
    )
    labels = load_head_labels(csv_path, num_layers, num_heads) if csv_path else [
        [1] * num_heads for _ in range(num_layers)
    ]
    sink_map = _build_int_map(label_sink_frames_map, min_value=1)
    recent_map = _build_int_map(label_recent_frames_map, min_value=1)
    stride_map = _build_bool_map(label_stride_enabled_map)
    interval_map = _build_int_map(label_stride_interval_map, min_value=1)
    phase_map = _build_int_map(label_phase_bucket_map, min_value=0)
    lag_map = _build_offsets_map(label_lag_offsets_map)
    merge_map = _build_bool_map(label_merge_enabled_map)
    merge_patch_map = _build_int_map(label_merge_patch_size_map, min_value=1)
    merge_capacity_map = _build_capacity_map(label_merge_capacity_map)

    compositions: list[list[HeadComposition]] = []
    for layer_idx in range(num_layers):
        row: list[HeadComposition] = []
        for head_idx in range(num_heads):
            label = labels[layer_idx][head_idx]
            label_key = _normalize_label_key(label)
            cap = int(cap_tensor[layer_idx, head_idx].item())
            is_osc = label == -1
            policy_type = _label_to_policy_type(label)

            # Determine sink/recent for this head
            if label_key in sink_map:
                sink = sink_map[label_key]
            elif is_osc:
                sink = osc_sink_frames if osc_sink_frames is not None else 1
            else:
                sink = stable_sink_frames if stable_sink_frames is not None else 1
            if label_key in recent_map:
                head_recent = recent_map[label_key]
            else:
                head_recent = recent_frames
            if label_key not in recent_map and not is_osc and stable_recent_frames is not None:
                head_recent = stable_recent_frames

            # Build middle strategies
            strategies: list[MiddleStrategy] = []

            use_cyclic, head_bucket_cap = _resolve_cyclic_enabled_for_label(
                label_key=label_key,
                is_osc=is_osc,
                cyclic_enabled=cyclic_enabled,
                cyclic_osc_only=cyclic_osc_only,
                phase_map=phase_map,
                cyclic_bucket_cap=cyclic_bucket_cap,
            )
            head_lag_offsets = _resolve_lag_offsets_for_label(
                label_key=label_key,
                is_osc=is_osc,
                lag_enabled=lag_enabled,
                cyclic_osc_only=cyclic_osc_only,
                lag_map=lag_map,
                lag_offsets=lag_offsets,
            )
            use_stride = _resolve_stride_enabled_for_label(
                label_key=label_key,
                is_osc=is_osc,
                stride_enabled=stride_enabled,
                stride_map=stride_map,
            )
            use_merge = _resolve_merge_enabled_for_label(
                label_key=label_key,
                merge_enabled=merge_enabled,
                merge_map=merge_map,
            )

            active_middle = []
            if use_cyclic:
                active_middle.append("cyclic")
            if use_stride:
                active_middle.append("stride")
            if use_merge:
                active_middle.append("merge")
            if len(active_middle) > 1:
                raise ValueError(
                    f"Middle strategies must be mutually exclusive for label {label_key}, "
                    f"got {active_middle}."
                )

            if use_cyclic:
                strategies.append(CyclicStrategy(
                    period=cyclic_period,
                    bucket_cap=head_bucket_cap,
                    dynamic_rope=cyclic_dynamic_rope,
                ))
                policy_type = "osc"

            if lag_enabled and len(head_lag_offsets) > 0:
                strategies.append(LagStrategy(
                    offsets=head_lag_offsets,
                    history_frames=lag_history,
                    dynamic_rope=lag_dynamic_rope,
                ))

            if use_stride:
                if label_key not in interval_map and stride_interval <= 0:
                    raise ValueError(f"Invalid stride interval for label {label_key}.")
                head_interval = interval_map.get(label_key, stride_interval)
                strategies.append(StrideStrategy(
                    interval=head_interval,
                    capacity=stride_capacity,
                    dynamic_rope=stride_dynamic_rope,
                ))
                policy_type = "stride"

            if use_merge:
                if label_key not in merge_patch_map:
                    raise ValueError(
                        f"Missing explicit merge patch size for label {label_key}: "
                        "set headkv_label_merge_patch_size_map."
                    )
                if label_key not in merge_capacity_map:
                    raise ValueError(
                        f"Missing explicit merge capacity for label {label_key}: "
                        "set headkv_label_merge_capacity_map."
                    )
                strategies.append(MergeStrategy(
                    patch_size=merge_patch_map.get(label_key, merge_patch_size),
                    capacity=merge_capacity_map.get(label_key, merge_capacity),
                    dynamic_rope=merge_dynamic_rope,
                ))
                policy_type = "merge"

            name = f"L{layer_idx}_H{head_idx}_{policy_type}"
            row.append(HeadComposition(
                name=name,
                label=label,
                sink_frames=sink,
                recent_frames=head_recent,
                middle_strategies=strategies,
                policy_type=policy_type,
                capacity=cap,
            ))
        compositions.append(row)
    return compositions

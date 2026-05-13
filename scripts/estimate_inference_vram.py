#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import yaml


def mib(x: float) -> float:
    return x / (1024.0 * 1024.0)


def gib(x: float) -> float:
    return x / (1024.0 * 1024.0 * 1024.0)


def _read_labels(csv_path: Path, num_layers: int, num_heads: int) -> list[list[int]]:
    labels = [[1 for _ in range(num_heads)] for _ in range(num_layers)]
    if not csv_path.exists():
        return labels
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                rows.append([x.strip() for x in row if str(x).strip() != ""])
    if rows and len(rows[0]) > 3:
        for l in range(min(num_layers, len(rows))):
            for h in range(min(num_heads, len(rows[l]))):
                try:
                    labels[l][h] = int(float(rows[l][h]))
                except ValueError:
                    labels[l][h] = 1
        return labels
    return labels


def _build_capacities(
    labels: list[list[int]],
    default_capacity: int,
    code_map: dict[str, int] | None,
) -> list[list[int]]:
    out = []
    cm = {str(k): int(v) for k, v in (code_map or {}).items()}
    for row in labels:
        cap_row = []
        for lab in row:
            cap_row.append(int(cm.get(str(lab), default_capacity)))
        out.append(cap_row)
    return out


def _decouple_flags_for_layer(
    capacities: list[int],
    osc_flags: list[bool],
    sink_grid_decoupling: bool,
) -> list[bool]:
    max_cap = max(capacities) if capacities else 0
    min_cap = min(capacities) if capacities else 0
    if sink_grid_decoupling and min_cap < max_cap:
        if any(osc_flags):
            return osc_flags[:]
        return [c < max_cap for c in capacities]
    return [sink_grid_decoupling for _ in capacities]


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate RadicalPyramidKV inference VRAM footprint.")
    p.add_argument("--config_path", required=True, type=Path)
    p.add_argument("--frames", nargs="+", type=int, default=[72, 240])
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--num_layers", type=int, default=30)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--frame_tokens", type=int, default=1560)
    p.add_argument("--dtype_bytes", type=int, default=2, help="2 for bf16/fp16, 4 for fp32")
    p.add_argument("--pos_bytes", type=int, default=8, help="torch.long is 8 bytes")
    args = p.parse_args()

    with args.config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sink_len = int(cfg.get("sink_len", args.frame_tokens))
    local_tail_frames = int(cfg.get("local_tail_frames", 4))
    sink_grid_decoupling = bool(cfg.get("sink_grid_decoupling", False))
    use_osc_frame_mode = bool(cfg.get("use_osc_frame_mode", False))
    phase_bucket_capacity_frames = int(cfg.get("phase_bucket_capacity_frames", 1))
    phase_sink_for_osc_only = bool(cfg.get("phase_sink_for_osc_only", True))
    use_osc_lag_mode = bool(cfg.get("use_osc_lag_mode", False))
    osc_lag_offsets_frames = [int(x) for x in cfg.get("osc_lag_offsets_frames", [6])]
    osc_lag_history_frames = int(cfg.get("osc_lag_history_frames", 21))
    disable_first_sink_for_osc_heads = bool(cfg.get("disable_first_sink_for_osc_heads", False))

    default_capacity = int(cfg.get("pyramidkv_default_capacity", 32768))
    code_map = cfg.get("pyramidkv_code_map", {"-1": default_capacity, "1": default_capacity})
    label_csv = Path(cfg.get("pyramidkv_config_path", ""))
    if not label_csv.is_absolute():
        label_csv = (Path.cwd() / label_csv).resolve()

    labels = _read_labels(label_csv, args.num_layers, args.num_heads)
    capacities = _build_capacities(labels, default_capacity, code_map)

    total_static_tokens = 0
    total_dynamic_tokens = 0
    total_lag_anchor_tokens = 0
    total_phase_anchor_tokens = 0

    flat_tokens_per_layer = []
    osc_heads_total = 0
    stable_heads_total = 0

    local_tokens = local_tail_frames * args.frame_tokens
    lag_count = len([x for x in osc_lag_offsets_frames if x > 0]) if use_osc_lag_mode else 0

    for l in range(args.num_layers):
        layer_labels = labels[l]
        layer_caps = capacities[l]
        osc_flags = [int(v) == -1 for v in layer_labels]
        decouple_flags = _decouple_flags_for_layer(layer_caps, osc_flags, sink_grid_decoupling)

        layer_flat_tokens = 0
        for h in range(args.num_heads):
            is_osc = osc_flags[h]
            if is_osc:
                osc_heads_total += 1
            else:
                stable_heads_total += 1

            full_cap = int(layer_caps[h])
            dyn_tokens = local_tokens if use_osc_frame_mode else full_cap
            dyn_tokens = min(full_cap, dyn_tokens)
            total_dynamic_tokens += dyn_tokens

            has_static = decouple_flags[h] and sink_len > 0
            static_tokens = 0
            if has_static:
                if not (disable_first_sink_for_osc_heads and is_osc):
                    static_tokens = sink_len
            total_static_tokens += static_tokens

            head_is_phase_sink = (not phase_sink_for_osc_only) or is_osc
            if use_osc_frame_mode and phase_bucket_capacity_frames > 0 and head_is_phase_sink:
                total_phase_anchor_tokens += phase_bucket_capacity_frames * args.frame_tokens

            if use_osc_lag_mode and head_is_phase_sink:
                total_lag_anchor_tokens += osc_lag_history_frames * args.frame_tokens

            extra_frames = 0
            if use_osc_frame_mode and phase_bucket_capacity_frames > 0 and head_is_phase_sink:
                extra_frames += phase_bucket_capacity_frames
            if use_osc_lag_mode and head_is_phase_sink:
                extra_frames += lag_count

            layer_flat_tokens += static_tokens + dyn_tokens + extra_frames * args.frame_tokens

        flat_tokens_per_layer.append(layer_flat_tokens)

    kv_token_bytes = args.head_dim * args.dtype_bytes * 2  # K and V
    pos_token_bytes = 3 * args.pos_bytes

    persistent_kv_bytes = (total_static_tokens + total_dynamic_tokens + total_phase_anchor_tokens + total_lag_anchor_tokens) * kv_token_bytes
    persistent_pos_bytes = (total_phase_anchor_tokens + total_lag_anchor_tokens) * pos_token_bytes
    persistent_total = persistent_kv_bytes + persistent_pos_bytes

    max_layer_flat_tokens = max(flat_tokens_per_layer) if flat_tokens_per_layer else 0
    one_layer_flat_kv_bytes = max_layer_flat_tokens * kv_token_bytes

    print("=== RadicalPyramidKV VRAM Estimate ===")
    print(f"config_path: {args.config_path}")
    print(f"label_csv: {label_csv}")
    print(f"layers x heads: {args.num_layers} x {args.num_heads}")
    print(f"osc_heads_total: {osc_heads_total}, stable_heads_total: {stable_heads_total}")
    print(f"sink_len_tokens: {sink_len}, local_tail_frames: {local_tail_frames}, frame_tokens: {args.frame_tokens}")
    print(f"use_osc_lag_mode: {use_osc_lag_mode}, osc_lag_offsets_frames: {osc_lag_offsets_frames}, lag_history_frames: {osc_lag_history_frames}")
    print("")
    print("Persistent cache (model forward resident):")
    print(f"  static_tokens: {total_static_tokens:,}")
    print(f"  dynamic_tokens: {total_dynamic_tokens:,}")
    print(f"  lag_anchor_tokens: {total_lag_anchor_tokens:,}")
    print(f"  phase_anchor_tokens: {total_phase_anchor_tokens:,}")
    print(f"  persistent_kv: {_fmt(gib(persistent_kv_bytes))} GiB")
    print(f"  persistent_pos: {_fmt(gib(persistent_pos_bytes))} GiB")
    print(f"  persistent_total: {_fmt(gib(persistent_total))} GiB")
    print("")
    print("Per-layer transient flat KV (rough peak increment):")
    print(f"  max_flat_tokens_per_layer: {max_layer_flat_tokens:,}")
    print(f"  one_layer_flat_kv: {_fmt(gib(one_layer_flat_kv_bytes))} GiB")
    print("")
    print("Frame-count dependent tensors (noise + latent output, rough):")
    latent_elems_per_frame = 16 * 60 * 104
    for f in args.frames:
        elems = args.num_samples * f * latent_elems_per_frame
        # sampled_noise + output(latent) approximated
        frame_tensor_bytes = elems * args.dtype_bytes * 2
        print(
            f"  frames={f:>4d}: noise+output ~= {_fmt(mib(frame_tensor_bytes))} MiB "
            f"({ _fmt(gib(frame_tensor_bytes)) } GiB)"
        )

    print("")
    print("Note:")
    print("  - Not including model weights, attention workspace, CUDA allocator fragmentation.")
    print("  - 72 vs 240 peak difference is mainly noise/output tensors; KV cache mostly saturates early.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analytically compute KV token counts per head type from a YAML config.

Zero GPU cost — purely arithmetic based on config parameters.

Usage:
    python scripts/compute_kv_tokens.py configs/pyramid-forcing.yaml
    python scripts/compute_kv_tokens.py configs/pyramid-forcing.yaml --num-frames 120
    python scripts/compute_kv_tokens.py path/to/configs_dir/ --batch  # all yamls in dir
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

from omegaconf import OmegaConf


def compute_kv_tokens(
    cfg,
    num_frames: int = 120,
    frame_seqlen: int = 1560,
    num_frame_per_block: int = 3,
) -> dict:
    """Compute per-head-type KV token counts at steady state.

    Returns dict with keys: osc, stable, stable_sparse, and their breakdowns.
    """
    # --- Parse config ---
    osc_sink = getattr(cfg, "pyramidkv_osc_sink_frames", 1) or 1
    stable_sink = getattr(cfg, "pyramidkv_stable_sink_frames", 3) or 3
    osc_recent = getattr(cfg, "pyramidkv_recent_frames", 4)
    stable_recent = getattr(cfg, "pyramidkv_stable_recent_frames", None)
    if stable_recent is None:
        stable_recent = osc_recent

    cyclic_enabled = getattr(cfg, "cyclic_enabled", False)
    cyclic_period = getattr(cfg, "cyclic_period", 6)
    cyclic_bucket_cap = getattr(cfg, "cyclic_bucket_cap", 1)

    stride_enabled = getattr(cfg, "stride_enabled", False)
    stride_interval = getattr(cfg, "stride_interval", 6)
    stride_capacity = getattr(cfg, "stride_capacity", -1)

    # --- Compute middle token counts ---
    # Cyclic middle: at query time, only 1 bucket is read → bucket_cap frames
    cyclic_middle = cyclic_bucket_cap if cyclic_enabled else 0

    # Stride middle: number of frames stored = min(num_aligned_frames, capacity)
    if stride_enabled:
        # Total aligned frames between sink and end of available frames
        # At steady state with num_frames total: frames 0..(num_frames-1)
        # Sink covers [0, stable_sink-1], recent covers last stable_recent frames
        # Stride stores t where t % interval == 0 and sink < t < recent_min
        available_frames = num_frames - stable_sink - stable_recent
        if available_frames > 0:
            stride_stored = math.ceil(available_frames / stride_interval)
        else:
            stride_stored = 0
        if stride_capacity > 0:
            stride_stored = min(stride_stored, stride_capacity)
        stride_middle = stride_stored
    else:
        stride_middle = 0

    # --- Per-type totals (in frames) ---
    osc_total_frames = osc_sink + cyclic_middle + osc_recent
    stable_total_frames = stable_sink + stride_middle + stable_recent
    sparse_total_frames = stable_sink + stable_recent  # no middle

    # --- Convert to tokens ---
    osc_tokens = osc_total_frames * frame_seqlen
    stable_tokens = stable_total_frames * frame_seqlen
    sparse_tokens = sparse_total_frames * frame_seqlen
    full_attn_tokens = num_frames * frame_seqlen

    # --- Compute savings ---
    results = {
        "osc": {
            "sink_frames": osc_sink,
            "middle_frames": cyclic_middle,
            "middle_type": "cyclic" if cyclic_enabled else "none",
            "recent_frames": osc_recent,
            "total_frames": osc_total_frames,
            "total_tokens": osc_tokens,
            "savings_pct": (1 - osc_tokens / full_attn_tokens) * 100,
        },
        "stable": {
            "sink_frames": stable_sink,
            "middle_frames": stride_middle,
            "middle_type": f"stride(int={stride_interval},cap={stride_capacity})" if stride_enabled else "none",
            "recent_frames": stable_recent,
            "total_frames": stable_total_frames,
            "total_tokens": stable_tokens,
            "savings_pct": (1 - stable_tokens / full_attn_tokens) * 100,
        },
        "sparse": {
            "sink_frames": stable_sink,
            "middle_frames": 0,
            "middle_type": "none",
            "recent_frames": stable_recent,
            "total_frames": sparse_total_frames,
            "total_tokens": sparse_tokens,
            "savings_pct": (1 - sparse_tokens / full_attn_tokens) * 100,
        },
        "full_attn_tokens": full_attn_tokens,
        "num_frames": num_frames,
    }
    return results


def print_results(results: dict, config_name: str = "") -> None:
    """Pretty-print KV token analysis."""
    header = f"KV Token Analysis: {config_name}" if config_name else "KV Token Analysis"
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"  Total frames: {results['num_frames']}")
    print(f"  Full attention: {results['full_attn_tokens']:,} tokens/head")
    print(f"{'='*60}")

    for head_type in ("osc", "stable", "sparse"):
        r = results[head_type]
        print(f"\n  {head_type.upper()} heads:")
        print(f"    Sink:   {r['sink_frames']} frames")
        print(f"    Middle: {r['middle_frames']} frames ({r['middle_type']})")
        print(f"    Recent: {r['recent_frames']} frames")
        print(f"    Total:  {r['total_frames']} frames = {r['total_tokens']:,} tokens")
        print(f"    Saving: {r['savings_pct']:.1f}%")
    print()


def batch_compute(config_dir: str, num_frames: int) -> None:
    """Compute KV tokens for all YAML files in a directory and output CSV."""
    yaml_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith(".yaml") and not f.startswith(".")
    )
    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        return

    default_cfg = OmegaConf.load("configs/default_config.yaml")

    rows = []
    for yf in yaml_files:
        path = os.path.join(config_dir, yf)
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(path))
        results = compute_kv_tokens(cfg, num_frames=num_frames)
        name = os.path.splitext(yf)[0]
        rows.append({
            "config": name,
            "osc_tokens": results["osc"]["total_tokens"],
            "stable_tokens": results["stable"]["total_tokens"],
            "sparse_tokens": results["sparse"]["total_tokens"],
            "osc_frames": results["osc"]["total_frames"],
            "stable_frames": results["stable"]["total_frames"],
            "sparse_frames": results["sparse"]["total_frames"],
            "osc_saving_pct": f"{results['osc']['savings_pct']:.1f}",
            "stable_saving_pct": f"{results['stable']['savings_pct']:.1f}",
            "sparse_saving_pct": f"{results['sparse']['savings_pct']:.1f}",
        })

    # Print CSV
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    # Also save to file
    csv_path = os.path.join(config_dir, "kv_tokens.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved to {csv_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Compute KV token counts from config")
    parser.add_argument("config", help="YAML config file or directory (with --batch)")
    parser.add_argument("--num-frames", type=int, default=120,
                        help="Number of output frames (default: 120)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all YAML files in directory")
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.config):
        batch_compute(args.config, args.num_frames)
    else:
        default_cfg = OmegaConf.load("configs/default_config.yaml")
        cfg = OmegaConf.merge(default_cfg, OmegaConf.load(args.config))
        results = compute_kv_tokens(cfg, num_frames=args.num_frames)
        print_results(results, config_name=os.path.basename(args.config))


if __name__ == "__main__":
    main()

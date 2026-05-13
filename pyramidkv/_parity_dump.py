"""Per-call parity dump for diagnosing AdaptiveKVCache vs MegaCache divergence.

When ``PYRAMIDKV_PARITY_DUMP_DIR`` is set, every call to
``CausalWanSelfAttention.forward`` records a small fingerprint of the
attention output ``x`` (norm, per-head stats, first/last 16 token slices)
to ``$PYRAMIDKV_PARITY_DUMP_DIR/$PYRAMIDKV_PARITY_DUMP_KIND/call_NNNNNN.pt``.

Env vars:
  PYRAMIDKV_PARITY_DUMP_DIR   — base output dir (required to enable)
  PYRAMIDKV_PARITY_DUMP_KIND  — subdir (e.g. "adaptive" or "mega"; default "unknown")
  PYRAMIDKV_PARITY_MAX_CALLS  — stop dumping after N calls (default unlimited).
                             For 24f, one prompt = 960 calls (4 steps × 8 blocks × 30 layers).

Usage:
  # AdaptiveKVCache baseline (24f, one prompt):
  PYRAMIDKV_PARITY_DUMP_DIR=videos/parity \\
  PYRAMIDKV_PARITY_DUMP_KIND=adaptive \\
  PYRAMIDKV_PARITY_MAX_CALLS=960 \\
  python inference.py ... --data_path prompts/parity_one.txt --num_output_frames 24
  # MegaCache run (same seed/config):
  PYRAMIDKV_PARITY_DUMP_DIR=videos/parity \\
  PYRAMIDKV_PARITY_DUMP_KIND=mega \\
  PYRAMIDKV_PARITY_MAX_CALLS=960 \\
  PYRAMIDKV_USE_MEGA_CACHE=1 \\
  python inference.py ... --data_path prompts/parity_one.txt --num_output_frames 24
  # Run twice (once per mega-cache toggle), diff the resulting dump dirs
  # manually to confirm bit-exact parity.

When ``PYRAMIDKV_PARITY_DUMP_DIR`` is unset, all functions are O(1) no-ops.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import torch


_lock = threading.Lock()
_state: dict[str, Any] = {
    "dir": None,
    "counter": 0,
    "max_calls": None,
    "first_dump_announced": False,
    "max_reached_announced": False,
    "initialized": False,
}


def _init() -> None:
    if _state["initialized"]:
        return
    _state["initialized"] = True
    base = os.environ.get("PYRAMIDKV_PARITY_DUMP_DIR")
    if not base:
        return
    kind = os.environ.get("PYRAMIDKV_PARITY_DUMP_KIND", "unknown")
    out = Path(base) / kind
    out.mkdir(parents=True, exist_ok=True)
    _state["dir"] = out
    _state["counter"] = 0
    # Optional cap on number of dumps (per-process). Useful for stopping
    # after 1 prompt's worth of calls (~960 for 24f) so we don't fill
    # disk on full MovieGenVideoBench runs.
    max_calls_env = os.environ.get("PYRAMIDKV_PARITY_MAX_CALLS")
    if max_calls_env:
        try:
            _state["max_calls"] = int(max_calls_env)
        except ValueError:
            _state["max_calls"] = None

    # Write a tiny meta file so you can verify env vars were picked up
    # even before the first dump fires.
    meta = {
        "kind": kind,
        "max_calls": _state["max_calls"],
        "pid": os.getpid(),
    }
    try:
        (out / "_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # Print an ACTIVATION banner to stderr so it shows in inference logs
    # without being swallowed by tqdm.
    sys.stderr.write(
        f"[parity_dump] ENABLED — kind={kind!r} dir={out} "
        f"max_calls={_state['max_calls']}\n"
    )
    sys.stderr.flush()


def enabled() -> bool:
    if not _state["initialized"]:
        _init()
    return _state["dir"] is not None


def dump_x(x: torch.Tensor) -> None:
    """Save a compact fingerprint of the attention output tensor.

    ``x`` shape is ``[B, L_q, H, D]`` (post-self-attn, pre-FFN).
    Both AdaptiveKVCache and MegaCache produce the same shape, so the
    fingerprints are directly comparable.
    """
    if not enabled():
        return
    with _lock:
        counter = _state["counter"]
        max_calls = _state["max_calls"]
        if max_calls is not None and counter >= max_calls:
            if not _state["max_reached_announced"]:
                _state["max_reached_announced"] = True
                sys.stderr.write(
                    f"[parity_dump] reached PYRAMIDKV_PARITY_MAX_CALLS={max_calls}; "
                    "subsequent calls skipped\n"
                )
                sys.stderr.flush()
            return
        _state["counter"] = counter + 1
        out_dir = _state["dir"]

    if out_dir is None:
        return

    if not _state["first_dump_announced"]:
        _state["first_dump_announced"] = True
        sys.stderr.write(
            f"[parity_dump] first dump fired — writing to {out_dir}\n"
        )
        sys.stderr.flush()

    x_f32 = x.detach().to(torch.float32)
    B, L_q, H, D = x_f32.shape
    head_first = min(16, L_q)
    head_last_start = max(0, L_q - 16)

    # torch.norm with a 3-tuple dim is interpreted as matrix norm (requires
    # 2 dims), so use linalg.vector_norm for the multi-dim Frobenius reduce.
    per_head_norm = torch.linalg.vector_norm(x_f32, dim=(0, 1, 3))   # [H]
    per_head_max_abs = x_f32.abs().amax(dim=(0, 1, 3))                # [H]
    payload = {
        "counter": counter,
        "shape": (int(B), int(L_q), int(H), int(D)),
        "norm": float(torch.linalg.vector_norm(x_f32).item()),
        "max_abs": float(x_f32.abs().max().item()),
        "mean": float(x_f32.mean().item()),
        # Per-head stats — helps localize which head label (osc/sta+/sta-)
        # diverges. Index 0..H-1.
        "per_head_norm": per_head_norm.cpu(),
        "per_head_max_abs": per_head_max_abs.cpu(),
        # First and last 16 query positions (B=1 assumed).
        "x_first16": x_f32[0, :head_first].clone().cpu(),         # [F16, H, D]
        "x_last16":  x_f32[0, head_last_start:].clone().cpu(),    # [F16, H, D]
    }

    fpath = out_dir / f"call_{counter:06d}.pt"
    torch.save(payload, fpath)


def reset_counter() -> None:
    """Reset the per-process counter — useful if you want to compare only
    a subset of prompts in a multi-prompt run."""
    _state["counter"] = 0
    _state["max_reached_announced"] = False


def current_counter() -> int:
    return int(_state["counter"])

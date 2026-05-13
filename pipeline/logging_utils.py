from __future__ import annotations

import os
import sys
import time

import torch.distributed as dist


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def get_log_mode() -> str:
    """Return the inference log mode.

    Env var PYRAMIDKV_LOG_MODE selects the mode:
      - "tqdm"   — rich per-timestep progress bar (default when TTY; slow over SSH)
      - "print"  — one plain print per block with elapsed ms (SSH-friendly)
      - "silent" — no output
    Falls back to "tqdm" if TTY is detected, otherwise "print".
    """
    mode = os.environ.get("PYRAMIDKV_LOG_MODE", "").strip().lower()
    if mode in ("tqdm", "print", "silent"):
        return mode
    if not is_main_process():
        return "silent"
    stdout_isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    stderr_isatty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return "tqdm" if (stdout_isatty or stderr_isatty) else "print"


def should_use_timestep_progress(
    *,
    main_process: bool | None = None,
    stdout_isatty: bool | None = None,
    stderr_isatty: bool | None = None,
) -> bool:
    if main_process is None:
        main_process = is_main_process()
    if not main_process:
        return False

    if stdout_isatty is None:
        stdout_isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if stderr_isatty is None:
        stderr_isatty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return bool(stdout_isatty or stderr_isatty)


def format_block_progress(
    block_index: int,
    total_blocks: int,
    block_start_frame: int,
    total_frames: int,
) -> str:
    return f"block {block_index}/{total_blocks} - {block_start_frame}/{total_frames}"


class BlockTimer:
    """Context manager for plain-print block timing (PYRAMIDKV_LOG_MODE=print).

    Records wall-clock entry time, prints one line on exit with elapsed ms.
    """

    def __init__(self, label: str, enabled: bool = True):
        self.label = label
        self.enabled = enabled
        self._t0 = 0.0

    def __enter__(self):
        if self.enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        if self.enabled:
            elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
            print(f"{self.label}  {elapsed_ms:7.1f} ms", flush=True)


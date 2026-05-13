"""M2.1 — GraphedDiffusionRunner skeleton.

Wraps a denoising-block forward in a CUDA Graph capture/replay flow,
following the vLLM "piecewise" pattern: pack/RoPE/FFN go inside the
graph, flash_attn_varlen stays eager (FA2 doesn't capture cleanly under
ragged cu_seqlens; FA3+ does, but we keep the conservative path until
flash-attn ≥ 2.7 is confirmed in the runtime).

Lifecycle
---------
1. ``warmup(sample_inputs)`` — runs the forward 3 times on a side stream
   so cuDNN/cuBLAS algorithm selection stabilizes (NVIDIA's required
   pre-capture step).
2. ``capture_noisy(sample_inputs)`` — first allocates input/output
   tensors that will alias across replays, then records the graph.
3. ``capture_clean(sample_inputs)`` — same, on a separate cudaGraphExec_t
   sharing the same mempool.
4. ``step(x, t, pass_kind)`` — copies inputs into pre-bound tensors and
   replays the appropriate graph; returns a clone of the output (the
   graph buffer is overwritten on the next replay).

The runner is not active by default: callers gate on
``PYRAMIDKV_USE_CUDA_GRAPH=1`` via the env var or a config flag.

This file contains only the skeleton; the actual capture wiring follows once
the C++ pack is folded into the forward path.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import torch


def cuda_graph_requested() -> bool:
    return os.environ.get("PYRAMIDKV_USE_CUDA_GRAPH", "0") == "1"


@dataclass
class _GraphSlot:
    """Bundle of (graph, predefined input tensors, predefined output) for one
    capture configuration. Multiple slots share a mempool but each has its
    own cudaGraphExec_t."""
    graph: torch.cuda.CUDAGraph
    in_x: torch.Tensor
    in_t: torch.Tensor
    out: torch.Tensor


class GraphedDiffusionRunner:
    """Skeleton runner. Real capture wires up in M2.2."""

    def __init__(
        self,
        forward_fn: Callable[..., torch.Tensor],
        warmup_steps: int = 3,
    ):
        self._forward = forward_fn
        self._warmup_steps = max(1, int(warmup_steps))

        # Two pools: long-lived KV-cache state (managed by PyramidKVCacheManager)
        # and short-lived per-step activations. Captured graphs use the short
        # pool; the manager allocates outside this runner so its tensors
        # persist across replays.
        self._mempool_short = torch.cuda.graph_pool_handle()

        # Generic keyed slot map. capture_noisy/capture_clean populate the
        # canonical "noisy"/"clean" keys for backward compatibility, but real
        # diffusion blocks have multiple timesteps so callers can capture more
        # slots via capture_slot(key, x, t) — e.g. (timestep_idx, "noisy").
        self._slots: dict[Hashable, _GraphSlot] = {}
        self._steady = False
        self._block_count = 0

    @property
    def steady(self) -> bool:
        return self._steady

    def warmup_side_stream(self, *sample_inputs) -> None:
        """Run a few eager forwards on a side stream to stabilize algorithm
        selection. Required by PyTorch CUDA Graph docs before the first
        capture; see https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._warmup_steps):
                _ = self._forward(*sample_inputs)
        torch.cuda.current_stream().wait_stream(side)

    def has_slot(self, key: Hashable) -> bool:
        return key in self._slots

    def capture_slot(self, key: Hashable, x: torch.Tensor, t: torch.Tensor, pass_kind: int) -> None:
        """Capture a forward keyed by an arbitrary hashable identifier.

        Use when one block forward has multiple distinct shapes / timesteps —
        e.g. ``key=(timestep_idx, "noisy")`` for each denoising step. The
        underlying graph still shares ``_mempool_short`` with all other slots.
        """
        self._slots[key] = self._capture(x, t, pass_kind)

    def _capture(self, x: torch.Tensor, t: torch.Tensor, pass_kind: int) -> _GraphSlot:
        """Record one CUDA Graph for a (pass_kind) forward.

        Pre-bind the input tensors so subsequent replays just memcpy into them
        (no realloc). The output is captured into a static tensor we own so the
        graph's mempool buffers don't dangle across replays.

        Caller must have already run ``warmup_side_stream`` so cuDNN/cuBLAS
        algorithm selection is stable; capture surfaces any remaining sync
        as ``cudaErrorStreamCaptureUnsupported``.
        """
        in_x = x.detach().clone()
        in_t = t.detach().clone()

        # Eager pre-warm under the same mempool so output shape is concrete and
        # any allocations for the forward live inside the captured pool.
        eager_out = self._forward(in_x, in_t, pass_kind)
        out = torch.empty_like(eager_out)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._mempool_short):
            captured = self._forward(in_x, in_t, pass_kind)
            out.copy_(captured)

        return _GraphSlot(graph=graph, in_x=in_x, in_t=in_t, out=out)

    def capture_noisy(self, x: torch.Tensor, t: torch.Tensor) -> None:
        self.capture_slot("noisy", x, t, pass_kind=0)

    def capture_clean(self, x: torch.Tensor, t: torch.Tensor) -> None:
        self.capture_slot("clean", x, t, pass_kind=1)

    def mark_steady(self) -> None:
        """Caller flips this after all required graphs are captured.

        Until then ``step`` falls through to the eager forward so warmup blocks
        run normally. After steady=True, ``step`` replays the captured graph
        for each requested key.

        Caller must have populated at least the canonical ``"noisy"`` and
        ``"clean"`` slots (the original two-pass design); additional keyed
        slots are optional.
        """
        if "noisy" not in self._slots or "clean" not in self._slots:
            raise RuntimeError(
                "mark_steady requires both noisy + clean graphs to be captured first"
            )
        self._steady = True

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pass_kind: int,
        key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        """Eager forward during warmup; replay the matching captured graph
        once ``mark_steady`` has flipped the state.

        ``key`` selects the slot to replay. If omitted, defaults to the
        canonical ``"noisy"``/``"clean"`` mapping based on ``pass_kind`` so
        existing two-pass callers keep working.

        The output is cloned out of the captured static buffer so the next
        replay can overwrite it without aliasing the value we returned.
        """
        self._block_count += 1
        if not self._steady:
            return self._forward(x, t, pass_kind)
        if key is None:
            key = "noisy" if pass_kind == 0 else "clean"
        slot = self._slots[key]
        slot.in_x.copy_(x, non_blocking=True)
        slot.in_t.copy_(t, non_blocking=True)
        slot.graph.replay()
        return slot.out.clone()

"""MiddleStrategy protocol and HeadComposition class.

Each attention head's KV cache uses a [sink ... middle ... recent] structure.
MiddleStrategy defines the interface for middle-segment strategies (cyclic, lag, stride, merge).
HeadComposition composes sink + middle strategies + recent for one head.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Protocol, runtime_checkable

import torch


@dataclass
class FrameAnchor:
    """One raw frame-level anchor stored by middle strategies."""

    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    t: int


@dataclass
class CollectedAnchor:
    """Payload returned by middle strategies during readout.

    `frame` anchors reuse raw frame KV/pos and can still receive post-prune RoPE.
    `merge` anchors now carry already-aggregated raw KV/pos and use one shared
    block-median time during the normal batched RoPE pass.
    """

    kind: str
    t: int
    dynamic_rope: bool
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    pos: torch.Tensor | None = None
    token_count: int = 0
    source_kind: str = "tensor"


@runtime_checkable
class MiddleStrategy(Protocol):
    """Interface for middle-segment frame selection strategies."""

    dynamic_rope: bool

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        """Store frame anchors from incoming KV tokens."""
        ...

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        """Return anchors for the current query time, excluding sink/recent overlap."""
        ...

    def reset(self, num_seq: int) -> None:
        """Reset internal storage for num_seq sequences (batch * heads)."""
        ...


class HeadComposition:
    """Per-head strategy composition: sink + middle strategies (union) + recent.

    Attributes:
        name: Human-readable identifier (e.g. "L0_H3_osc").
        label: Head class label (-1=oscillating, 1=stable, 2=stable_sparse).
        sink_frames: Number of sink frames to retain.
        recent_frames: Number of recent frames in sliding window.
        middle_strategies: List of MiddleStrategy instances whose anchors are unioned.
        policy_type: Legacy compatibility — "osc", "stride", or "recent_only".
        capacity: Per-head capacity (for legacy PyramidKVCache path).
    """

    def __init__(
        self,
        name: str,
        label: int,
        sink_frames: int,
        recent_frames: int,
        middle_strategies: list[MiddleStrategy] | None = None,
        policy_type: str = "osc",
        capacity: int = 32768,
    ):
        self.name = name
        self.label = label
        self.sink_frames = max(0, int(sink_frames))
        self.recent_frames = max(1, int(recent_frames))
        self.middle_strategies = middle_strategies or []
        self.policy_type = policy_type
        self.capacity = int(capacity)

    @property
    def has_middle(self) -> bool:
        return len(self.middle_strategies) > 0

    def reset_all(self, num_seq: int) -> None:
        for strategy in self.middle_strategies:
            strategy.reset(num_seq)

    def update_all(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
    ) -> None:
        # Precompute t_vals once to share across all strategies (1 GPU→CPU sync)
        t_vals: list[int] | None = None
        if frame_seqlen > 0 and pos_seq.shape[0] >= frame_seqlen and pos_seq.shape[0] % frame_seqlen == 0:
            num_frames = pos_seq.shape[0] // frame_seqlen
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        for strategy in self.middle_strategies:
            strategy.update(idx, k_seq, v_seq, pos_seq, frame_seqlen, current_t, t_vals=t_vals)

    def collect_all(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        """Collect anchors from all middle strategies, deduplicated by t."""
        if len(self.middle_strategies) == 1:
            anchors = self.middle_strategies[0].collect(idx, current_t, recent_min_t, sink_max_t)
            if len(anchors) < 2 or all(prev.t <= curr.t for prev, curr in zip(anchors, anchors[1:])):
                return anchors
            return sorted(anchors, key=operator.attrgetter("t"))

        seen_t: set[int] = set()
        result: list[CollectedAnchor] = []
        for strategy in self.middle_strategies:
            anchors = strategy.collect(idx, current_t, recent_min_t, sink_max_t)
            for anchor in anchors:
                if anchor.t not in seen_t:
                    seen_t.add(anchor.t)
                    result.append(anchor)
        result.sort(key=operator.attrgetter("t"))
        return result

    def __repr__(self) -> str:
        strats = [type(s).__name__ for s in self.middle_strategies]
        return f"<HeadComposition {self.name} label={self.label} middle={strats}>"

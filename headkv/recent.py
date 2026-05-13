"""RecentStrategy — degenerate middle strategy with no anchors.

Used for heads that only need sink + recent (no middle segment).
"""
from __future__ import annotations

import torch

from .base import FrameAnchor


class RecentStrategy:
    """No-op middle strategy: collect() always returns empty list."""

    dynamic_rope: bool = False

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
        pass

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[FrameAnchor]:
        return []

    def reset(self, num_seq: int) -> None:
        pass

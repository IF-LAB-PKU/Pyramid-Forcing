from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


WAN_TEACACHE_COEFFICIENTS: dict[str, list[float]] = {
    "wan2.1_t2v_1.3b": [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
    ],
    "wan2.1_t2v_14b": [
        -5784.54975374,
        5449.50911966,
        -1811.16591783,
        256.27178429,
        -13.02252404,
    ],
    "wan2.1_i2v_480p_14b": [
        -3.02331670e02,
        2.23948934e02,
        -5.25463970e01,
        5.87348440e00,
        -2.01973289e-01,
    ],
    "wan2.1_i2v_720p_14b": [
        -114.36346466,
        65.26524496,
        -18.82220707,
        4.91518089,
        -0.23412683,
    ],
}

WAN_TEACACHE_RET_COEFFICIENTS: dict[str, list[float]] = {
    "wan2.1_t2v_1.3b": [
        -5.21862437e04,
        9.23041404e03,
        -5.28275948e02,
        1.36987616e01,
        -4.99875664e-02,
    ],
    "wan2.1_t2v_14b": [
        -3.03318725e05,
        4.90537029e04,
        -2.65530556e03,
        5.87365115e01,
        -3.15583525e-01,
    ],
    "wan2.1_i2v_480p_14b": [
        2.57151496e05,
        -3.54229917e04,
        1.40286849e03,
        -1.35890334e01,
        1.32517977e-01,
    ],
    "wan2.1_i2v_720p_14b": [
        8.10705460e03,
        2.13393892e03,
        -3.72934672e02,
        1.66203073e01,
        -4.17769401e-02,
    ],
}


def normalize_teacache_variant(value: str | None) -> str:
    if not value:
        return "wan2.1_t2v_1.3b"
    normalized = value.lower().replace("-", "_").replace(".", ".")
    normalized = normalized.replace("wan2_1", "wan2.1")
    normalized = normalized.replace("t2v_1_3b", "t2v_1.3b")
    normalized = normalized.replace("t2v_14_b", "t2v_14b")
    normalized = normalized.replace("i2v_480p_14_b", "i2v_480p_14b")
    normalized = normalized.replace("i2v_720p_14_b", "i2v_720p_14b")
    return normalized


def get_wan_teacache_coefficients(
    variant: str | None,
    *,
    use_ret_steps: bool = False,
) -> list[float]:
    variant_key = normalize_teacache_variant(variant)
    table = WAN_TEACACHE_RET_COEFFICIENTS if use_ret_steps else WAN_TEACACHE_COEFFICIENTS
    if variant_key not in table:
        known = ", ".join(sorted(table))
        raise ValueError(f"Unknown TeaCache variant {variant!r}; expected one of: {known}")
    return list(table[variant_key])


@dataclass
class TeaCacheStats:
    full_calls: int = 0
    skipped_calls: int = 0


class TeaCacheController:
    """Stateful TeaCache decision logic for one denoising block."""

    def __init__(
        self,
        *,
        rel_l1_thresh: float,
        coefficients: Iterable[float],
        max_skip_steps: int = 3,
        enabled: bool = True,
    ) -> None:
        coefficients = list(coefficients)
        if len(coefficients) != 5:
            raise ValueError(f"TeaCache coefficients must have 5 values, got {len(coefficients)}")
        if max_skip_steps < 0:
            raise ValueError("max_skip_steps must be >= 0")
        self.enabled = enabled
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.coefficients = [float(v) for v in coefficients]
        self.max_skip_steps = int(max_skip_steps)
        self.reset()
        self.reset_stats()

    def reset(self) -> None:
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance = 0.0
        self.consecutive_skips = 0

    def reset_stats(self) -> None:
        self.stats = TeaCacheStats()

    def pop_stats(self) -> dict[str, int]:
        stats = {
            "full_calls": self.stats.full_calls,
            "skipped_calls": self.stats.skipped_calls,
        }
        self.reset_stats()
        return stats

    def _rescale(self, value: float) -> float:
        result = 0.0
        for coeff in self.coefficients:
            result = result * value + coeff
        return result

    def should_skip(self, modulated_input: torch.Tensor, *, force_calc: bool = False) -> bool:
        if not self.enabled or force_calc or self.max_skip_steps == 0:
            self.previous_modulated_input = modulated_input.detach().clone()
            self.accumulated_rel_l1_distance = 0.0
            self.consecutive_skips = 0
            return False

        current = modulated_input.detach()
        previous = self.previous_modulated_input
        self.previous_modulated_input = current.clone()
        if previous is None or self.previous_residual is None:
            self.consecutive_skips = 0
            return False

        denominator = previous.abs().mean().clamp_min(1e-6)
        rel_l1 = ((current - previous).abs().mean() / denominator).float().item()
        self.accumulated_rel_l1_distance += self._rescale(rel_l1)
        if (
            self.accumulated_rel_l1_distance < self.rel_l1_thresh
            and self.consecutive_skips < self.max_skip_steps
        ):
            self.consecutive_skips += 1
            self.stats.skipped_calls += 1
            return True

        self.accumulated_rel_l1_distance = 0.0
        self.consecutive_skips = 0
        return False

    def apply_cached_residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.previous_residual is None:
            raise RuntimeError("TeaCache residual is not initialized")
        return x + self.previous_residual.to(device=x.device, dtype=x.dtype)

    def update_residual(self, x_before_blocks: torch.Tensor, x_after_blocks: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.previous_residual = (x_after_blocks - x_before_blocks).detach()
        self.stats.full_calls += 1

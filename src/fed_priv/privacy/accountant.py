"""RDP-based privacy accountant for subsampled Gaussian mechanism."""

from __future__ import annotations

import math
from typing import Iterable


def _rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    """RDP of Gaussian mechanism with sensitivity 1 and noise std = noise_multiplier."""
    if noise_multiplier <= 0:
        return float("inf")
    return alpha / (2.0 * noise_multiplier**2)


def _rdp_subsampled_gaussian(
    alpha: float,
    noise_multiplier: float,
    sample_rate: float,
) -> float:
    """RDP of subsampled Gaussian mechanism (Balle et al. tight bound, simplified)."""
    if sample_rate == 0:
        return 0.0
    if sample_rate == 1.0:
        return _rdp_gaussian(alpha, noise_multiplier)

    # Tight upper bound via log-sum-exp over two terms (simplified stable form).
    log_a = math.log(sample_rate)
    log_1ma = math.log1p(-sample_rate)
    rdp_full = _rdp_gaussian(alpha, noise_multiplier)

    t1 = log_a + (alpha - 1) * rdp_full
    t2 = log_1ma
    log_sum = max(t1, t2) + math.log(math.exp(t1 - max(t1, t2)) + math.exp(t2 - max(t1, t2)))
    return log_sum / (alpha - 1) if alpha > 1 else 0.0


def compute_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: Iterable[float] | None = None,
) -> float:
    """Convert composed RDP to (epsilon, delta)-DP via optimal order alpha."""
    if alphas is None:
        alphas = [1 + x / 10.0 for x in range(1, 80)]

    best_eps = float("inf")
    for alpha in alphas:
        if alpha <= 1.0:
            continue
        rdp = steps * _rdp_subsampled_gaussian(alpha, noise_multiplier, sample_rate)
        eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        best_eps = min(best_eps, eps)
    return best_eps

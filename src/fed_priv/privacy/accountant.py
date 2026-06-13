"""RDP-based privacy accountant for subsampled Gaussian mechanism.

Provides both a stateless `compute_epsilon` function and a stateful
`RenyiAccountant` class for tracking cumulative privacy expenditure
across multiple training phases or compositions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


def _rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    """RDP of Gaussian mechanism with sensitivity 1 and noise std = noise_multiplier.

    Args:
        alpha: Renyi divergence order (> 1).
        noise_multiplier: Ratio of noise std to sensitivity.

    Returns:
        RDP epsilon at order alpha: alpha / (2 * sigma^2).
    """
    if noise_multiplier <= 0:
        return float("inf")
    return alpha / (2.0 * noise_multiplier**2)


def _rdp_subsampled_gaussian(
    alpha: float,
    noise_multiplier: float,
    sample_rate: float,
) -> float:
    """RDP of subsampled Gaussian mechanism (Balle et al. tight bound, simplified).

    Args:
        alpha: Renyi divergence order (> 1).
        noise_multiplier: Ratio of noise std to sensitivity.
        sample_rate: Probability q of each record being included in a batch.

    Returns:
        RDP guarantee for one step of subsampled Gaussian mechanism.
    """
    if sample_rate == 0:
        return 0.0
    if sample_rate == 1.0:
        return _rdp_gaussian(alpha, noise_multiplier)

    log_a = math.log(sample_rate)
    log_1ma = math.log1p(-sample_rate)
    rdp_full = _rdp_gaussian(alpha, noise_multiplier)

    t1 = log_a + (alpha - 1) * rdp_full
    t2 = log_1ma
    log_sum = max(t1, t2) + math.log(math.exp(t1 - max(t1, t2)) + math.exp(t2 - max(t1, t2)))
    return log_sum / (alpha - 1) if alpha > 1 else 0.0


def _rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """Convert a single RDP guarantee to (epsilon, delta)-DP.

    Uses the conversion: epsilon = rdp + log(1/delta) / (alpha - 1)

    Args:
        rdp: RDP guarantee at order alpha.
        alpha: Renyi divergence order.
        delta: Target delta.

    Returns:
        Epsilon for the given delta.
    """
    if alpha <= 1.0:
        return float("inf")
    return rdp + math.log(1.0 / delta) / (alpha - 1.0)


def compute_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: Iterable[float] | None = None,
) -> float:
    """Convert composed RDP to (epsilon, delta)-DP via optimal order alpha.

    Computes the tightest epsilon across a range of Renyi divergence orders
    for T steps of subsampled Gaussian mechanism composed sequentially.

    Args:
        noise_multiplier: Noise scale sigma.
        sample_rate: Subsampling probability q = B/N.
        steps: Number of mechanism applications T.
        delta: Target delta for (epsilon, delta)-DP.
        alphas: Renyi divergence orders to search. Defaults to [1.1, 1.2, ..., 8.9].

    Returns:
        Minimum epsilon across all alpha orders.

    Math:
        epsilon(delta) = min_alpha [T * rdp_q(alpha, sigma) + log(1/delta)/(alpha-1)]
    """
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


@dataclass
class RDPStep:
    """Record of a single RDP composition step.

    Args:
        noise_multiplier: Noise scale for this step.
        sampling_rate: Subsampling probability.
        num_steps: Number of mechanism applications in this phase.
    """

    noise_multiplier: float
    sampling_rate: float
    num_steps: int


class RenyiAccountant:
    """Stateful Renyi Differential Privacy accountant.

    Tracks cumulative privacy expenditure using Renyi Divergence-based
    composition across multiple training phases with potentially different
    noise multipliers and sampling rates.

    The accountant maintains a vector of RDP guarantees across multiple
    alpha orders and converts to (epsilon, delta)-DP using the optimal alpha.

    Args:
        alpha_orders: List of Renyi divergence orders to track. Higher orders
            give tighter bounds for low-noise regimes; lower orders are better
            for high-noise. Default range covers [1.1, 64].

    Math:
        RDP composition: rdp_total(alpha) = sum_t rdp_t(alpha)
        Conversion: epsilon = min_alpha [rdp_total(alpha) + log(1/delta)/(alpha-1)]

    Example:
        >>> accountant = RenyiAccountant()
        >>> accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=1000)
        >>> eps = accountant.get_epsilon(delta=1e-5)
    """

    def __init__(self, alpha_orders: list[float] | None = None) -> None:
        if alpha_orders is None:
            self.alpha_orders = [1.0 + i / 10.0 for i in range(1, 640)]
        else:
            self.alpha_orders = [a for a in alpha_orders if a > 1.0]
        self._privacy_spent_rdp: list[float] = [0.0] * len(self.alpha_orders)
        self._history: list[RDPStep] = []

    @property
    def privacy_spent_rdp(self) -> list[float]:
        """Current RDP guarantee vector across all alpha orders."""
        return list(self._privacy_spent_rdp)

    @property
    def history(self) -> list[RDPStep]:
        """History of accumulated RDP steps."""
        return list(self._history)

    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
    ) -> None:
        """Accumulate privacy cost for a training phase.

        Adds the RDP cost of `steps` applications of the subsampled Gaussian
        mechanism with the given noise and sampling parameters.

        Args:
            noise_multiplier: Noise scale sigma for this phase.
            sampling_rate: Subsampling probability q = B/N.
            steps: Number of gradient steps in this phase.
        """
        self._history.append(RDPStep(
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=steps,
        ))
        for idx, alpha in enumerate(self.alpha_orders):
            rdp_per_step = _rdp_subsampled_gaussian(alpha, noise_multiplier, sampling_rate)
            self._privacy_spent_rdp[idx] += steps * rdp_per_step

        logger.debug(
            "Accumulated %d steps: sigma=%.3f, q=%.4f",
            steps,
            noise_multiplier,
            sampling_rate,
        )

    def get_epsilon(self, delta: float) -> float:
        """Convert accumulated RDP to (epsilon, delta)-DP.

        Finds the tightest epsilon by optimizing over all tracked alpha orders.

        Args:
            delta: Target delta for (epsilon, delta)-DP guarantee.

        Returns:
            Minimum epsilon achievable at the given delta.

        Raises:
            ValueError: If delta is non-positive.
        """
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        best_eps = float("inf")
        for idx, alpha in enumerate(self.alpha_orders):
            eps = _rdp_to_epsilon(self._privacy_spent_rdp[idx], alpha, delta)
            best_eps = min(best_eps, eps)
        return best_eps

    def reset(self) -> None:
        """Reset the accountant to zero privacy expenditure."""
        self._privacy_spent_rdp = [0.0] * len(self.alpha_orders)
        self._history = []

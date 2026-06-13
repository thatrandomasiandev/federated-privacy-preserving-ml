"""Pairwise masking for secure aggregation.

Provides both functional utilities and a stateful SecureAggregator class
for secure model update aggregation using pairwise random masks that
cancel in summation, preventing the server from observing individual updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SecureAggResult:
    """Result container for secure aggregation trials.

    Args:
        max_abs_error: Maximum absolute error between secure and plain sum.
        mean_abs_error: Mean absolute error between secure and plain sum.
        reconstruction_mse: MSE of server-side reconstruction attack.
        n_clients: Number of participating clients.
    """

    max_abs_error: float
    mean_abs_error: float
    reconstruction_mse: float
    n_clients: int


def _pairwise_mask(
    i: int,
    j: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    """Deterministic pseudo-random mask for client pair (i, j) with i < j.

    Args:
        i: First client index (must be < j).
        j: Second client index.
        dim: Dimensionality of the mask vector.
        seed: Base seed for deterministic generation.

    Returns:
        Mask vector of shape (dim,) sampled from N(0, 1).
    """
    rng = np.random.default_rng(seed + i * 10007 + j * 10009)
    return rng.normal(0, 1, size=dim).astype(np.float64)


def secure_aggregate(
    client_updates: list[np.ndarray],
    seed: int = 42,
) -> np.ndarray:
    """Aggregate client updates with pairwise masks that cancel in the sum.

    Client i adds +M_ij for j > i and -M_ji for j < i to its update before sending.
    The masks cancel pairwise in summation, yielding the true aggregate.

    Args:
        client_updates: List of N client update vectors, each of shape (D,).
        seed: Seed for deterministic mask generation.

    Returns:
        Aggregated update vector of shape (D,).
    """
    return _masked_shares(client_updates, seed).sum(axis=0)


def naive_sum(client_updates: list[np.ndarray]) -> np.ndarray:
    """Plain sum without masking (ground truth for server-side aggregation).

    Args:
        client_updates: List of client update vectors.

    Returns:
        Element-wise sum of all updates.
    """
    return np.sum(client_updates, axis=0).astype(np.float64)


def _masked_shares(
    client_updates: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Per-client masked vectors as seen by the aggregation server.

    Args:
        client_updates: List of N client update vectors of shape (D,).
        seed: Base seed for mask generation.

    Returns:
        Array of shape (N, D) with masked client shares.
    """
    n_clients = len(client_updates)
    dim = client_updates[0].shape[0]
    masked = np.zeros((n_clients, dim), dtype=np.float64)
    for i in range(n_clients):
        masked[i] = client_updates[i].astype(np.float64)
        for j in range(n_clients):
            if i == j:
                continue
            if i < j:
                masked[i] += _pairwise_mask(i, j, dim, seed)
            else:
                masked[i] -= _pairwise_mask(j, i, dim, seed)
    return masked


class SecureAggregator:
    """Stateful secure aggregation protocol using pairwise random masks.

    Implements a simplified secure aggregation where each pair of clients
    (i, j) shares a random mask M_ij. Client i adds +M_ij for j > i and
    subtracts M_ji for j < i. When the server sums all masked updates,
    the masks cancel pairwise, recovering the true aggregate.

    This prevents the server from learning individual client updates while
    still computing the correct sum.

    Args:
        n_clients: Number of participating clients.
        seed: Base seed for deterministic mask generation.

    Math:
        For each client i, the masked share is:
            s_i = u_i + sum_{j>i} M_ij - sum_{j<i} M_ji
        The aggregate is:
            sum_i s_i = sum_i u_i (masks cancel: +M_ij from i, -M_ij from j)

    Example:
        >>> agg = SecureAggregator(n_clients=5, seed=42)
        >>> updates = [torch.randn(100) for _ in range(5)]
        >>> result = agg.aggregate(updates)
    """

    def __init__(self, n_clients: int, seed: int = 42) -> None:
        self.n_clients = n_clients
        self.seed = seed

    def _generate_pairwise_mask(self, i: int, j: int, dim: int) -> torch.Tensor:
        """Generate a deterministic pairwise mask for clients (i, j).

        Uses a seeded RNG keyed to the pair indices to produce a reproducible
        mask that both clients can independently compute.

        Args:
            i: First client index (must be < j).
            j: Second client index.
            dim: Dimensionality of the mask vector.

        Returns:
            Torch tensor mask of shape (dim,) from N(0, 1).
        """
        mask_seed = self.seed + i * 10007 + j * 10009
        gen = torch.Generator()
        gen.manual_seed(mask_seed)
        return torch.randn(dim, generator=gen, dtype=torch.float64)

    def aggregate(self, client_updates: list[torch.Tensor]) -> torch.Tensor:
        """Securely aggregate client updates using pairwise masking.

        Each client's update is masked before being sent to the server.
        The server sums the masked updates; pairwise masks cancel to yield
        the true sum of unmasked updates.

        Args:
            client_updates: List of N client update tensors, each of shape (D,).

        Returns:
            Aggregated tensor of shape (D,) equal to sum of original updates.

        Raises:
            ValueError: If number of updates doesn't match n_clients.
        """
        if len(client_updates) != self.n_clients:
            raise ValueError(
                f"Expected {self.n_clients} updates, got {len(client_updates)}"
            )

        dim = client_updates[0].shape[0]
        masked_shares = torch.zeros(self.n_clients, dim, dtype=torch.float64)

        for i in range(self.n_clients):
            masked_shares[i] = client_updates[i].to(torch.float64)
            for j in range(self.n_clients):
                if i == j:
                    continue
                if i < j:
                    masked_shares[i] += self._generate_pairwise_mask(i, j, dim)
                else:
                    masked_shares[i] -= self._generate_pairwise_mask(j, i, dim)

        result = masked_shares.sum(dim=0)
        self._verify_reconstruction(client_updates, result)
        return result

    def _verify_reconstruction(
        self,
        client_updates: list[torch.Tensor],
        reconstructed: torch.Tensor,
        atol: float = 1e-6,
    ) -> bool:
        """Verify that secure aggregation correctly reconstructs the true sum.

        Args:
            client_updates: Original unmasked client updates.
            reconstructed: Result of secure aggregation.
            atol: Absolute tolerance for floating-point comparison.

        Returns:
            True if reconstruction matches within tolerance.
        """
        true_sum = torch.stack([u.to(torch.float64) for u in client_updates]).sum(dim=0)
        is_correct = torch.allclose(reconstructed, true_sum, atol=atol)
        if not is_correct:
            max_err = (reconstructed - true_sum).abs().max().item()
            logger.warning(
                "Secure aggregation reconstruction error: max_abs=%.2e (tol=%.2e)",
                max_err,
                atol,
            )
        return is_correct


def reconstruction_attack_mse(
    client_updates: list[np.ndarray],
    seed: int,
    target_idx: int = 0,
) -> float:
    """Server-side attack using only masked shares.

    Subtracting all other masked shares from the aggregate yields the target
    client's masked vector, not its raw update — residual MSE measures privacy.

    Args:
        client_updates: Original (unmasked) client updates.
        seed: Seed used for masking.
        target_idx: Index of the target client to attack.

    Returns:
        MSE between the attack estimate and the true client update.
    """
    masked = _masked_shares(client_updates, seed)
    aggregated = masked.sum(axis=0)
    estimate = aggregated - masked.sum(axis=0) + masked[target_idx]
    target = client_updates[target_idx]
    return float(np.mean((estimate - target) ** 2))


def run_secure_agg_trial(
    n_clients: int,
    update_dim: int,
    seed: int,
) -> SecureAggResult:
    """Single trial of secure aggregation correctness and attack resistance.

    Args:
        n_clients: Number of simulated clients.
        update_dim: Dimensionality of update vectors.
        seed: Random seed.

    Returns:
        SecureAggResult with error and privacy metrics.
    """
    rng = np.random.default_rng(seed)
    updates = [rng.normal(0, 1, size=update_dim).astype(np.float64) for _ in range(n_clients)]

    true_sum = naive_sum(updates)
    secure_sum = secure_aggregate(updates, seed=seed)
    error = np.abs(secure_sum - true_sum)
    recon_mse = reconstruction_attack_mse(updates, seed=seed, target_idx=0)

    return SecureAggResult(
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        reconstruction_mse=recon_mse,
        n_clients=n_clients,
    )


def run_secure_agg_benchmark(
    n_clients: int,
    update_dim: int,
    n_trials: int,
    seed: int,
) -> SecureAggResult:
    """Average secure aggregation metrics over multiple trials.

    Args:
        n_clients: Number of simulated clients.
        update_dim: Dimensionality of update vectors.
        n_trials: Number of independent trials to average over.
        seed: Base random seed.

    Returns:
        SecureAggResult with averaged metrics.
    """
    trials = [
        run_secure_agg_trial(n_clients, update_dim, seed=seed + t) for t in range(n_trials)
    ]
    return SecureAggResult(
        max_abs_error=float(np.mean([t.max_abs_error for t in trials])),
        mean_abs_error=float(np.mean([t.mean_abs_error for t in trials])),
        reconstruction_mse=float(np.mean([t.reconstruction_mse for t in trials])),
        n_clients=n_clients,
    )

"""Pairwise masking for secure aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SecureAggResult:
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
    """Deterministic pseudo-random mask for client pair (i, j) with i < j."""
    rng = np.random.default_rng(seed + i * 10007 + j * 10009)
    return rng.normal(0, 1, size=dim).astype(np.float64)


def secure_aggregate(
    client_updates: list[np.ndarray],
    seed: int = 42,
) -> np.ndarray:
    """
    Aggregate client updates with pairwise masks that cancel in the sum.

    Client i adds +M_ij for j > i and -M_ji for j < i to its update before sending.
    """
    return _masked_shares(client_updates, seed).sum(axis=0)


def naive_sum(client_updates: list[np.ndarray]) -> np.ndarray:
    """Plain sum without masking (ground truth for server-side aggregation)."""
    return np.sum(client_updates, axis=0).astype(np.float64)


def _masked_shares(
    client_updates: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Per-client masked vectors as seen by the aggregation server."""
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


def reconstruction_attack_mse(
    client_updates: list[np.ndarray],
    seed: int,
    target_idx: int = 0,
) -> float:
    """
    Server-side attack using only masked shares.

    Subtracting all other masked shares from the aggregate yields the target
    client's masked vector, not its raw update — residual MSE measures privacy.
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
    """Single trial of secure aggregation correctness and attack resistance."""
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
    """Average secure aggregation metrics over multiple trials."""
    trials = [
        run_secure_agg_trial(n_clients, update_dim, seed=seed + t) for t in range(n_trials)
    ]
    return SecureAggResult(
        max_abs_error=float(np.mean([t.max_abs_error for t in trials])),
        mean_abs_error=float(np.mean([t.mean_abs_error for t in trials])),
        reconstruction_mse=float(np.mean([t.reconstruction_mse for t in trials])),
        n_clients=n_clients,
    )

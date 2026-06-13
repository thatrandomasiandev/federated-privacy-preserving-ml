"""Synthetic partitioned classification data with known linear ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fed_priv.data.base import ClientPartition, FederatedDataset


@dataclass
class PartitionedDGPConfig:
    """Configuration for federated classification DGP."""

    n_clients: int = 5
    n_samples: int = 2000
    feature_dim: int = 12
    dirichlet_alpha: float = 1.0
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    label_noise: float = 0.05
    feature_shift_scale: float = 0.3
    seed: int = 42


def _logistic_labels(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    rng: np.random.Generator,
    label_noise: float,
) -> np.ndarray:
    logits = np.sum(X.astype(np.float64) * weights.astype(np.float64), axis=1) + bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(X.shape[0]) < probs).astype(np.float32)
    if label_noise > 0:
        flip = rng.random(X.shape[0]) < label_noise
        y = np.where(flip, 1.0 - y, y)
    return y


def _allocate_client_indices(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Dirichlet label-skewed allocation of sample indices to clients."""
    labels = y.astype(int)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for label in (0, 1):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_clients)
        counts = (proportions * len(idx)).astype(int)
        diff = len(idx) - counts.sum()
        counts[np.argmax(counts)] += diff
        start = 0
        for c, count in enumerate(counts):
            end = start + count
            client_indices[c].extend(idx[start:end].tolist())
            start = end
    return [np.array(sorted(indices), dtype=np.int64) for indices in client_indices]


def generate_partitioned_dataset(config: PartitionedDGPConfig) -> FederatedDataset:
    """Generate a federated binary classification dataset with known separator."""
    rng = np.random.default_rng(config.seed)

    weights = rng.normal(0, 1, size=config.feature_dim).astype(np.float32)
    weights /= np.linalg.norm(weights) + 1e-8
    bias = float(rng.normal(0, 0.5))

    n_train = int(config.n_samples * (1 - config.test_ratio - config.val_ratio))
    n_val = int(config.n_samples * config.val_ratio)
    n_test = config.n_samples - n_train - n_val

    X_all = rng.normal(0, 1, size=(config.n_samples, config.feature_dim)).astype(np.float32)
    y_all = _logistic_labels(X_all, weights, bias, rng, config.label_noise)

    perm = rng.permutation(config.n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    allocations = _allocate_client_indices(
        y_train,
        config.n_clients,
        config.dirichlet_alpha,
        rng,
    )

    clients: list[ClientPartition] = []
    for client_id, indices in enumerate(allocations):
        X_local = X_train[indices].copy()
        if config.feature_shift_scale > 0:
            shift = rng.normal(0, config.feature_shift_scale, size=config.feature_dim)
            X_local += shift.astype(np.float32)
        y_local = y_train[indices]
        label_counts = np.bincount(y_local.astype(int), minlength=2)
        label_dist = {
            int(k): float(v / max(len(y_local), 1)) for k, v in enumerate(label_counts)
        }
        clients.append(
            ClientPartition(
                client_id=client_id,
                X=X_local,
                y=y_local,
                label_distribution=label_dist,
            )
        )

    logits_test = np.sum(X_test.astype(np.float64) * weights.astype(np.float64), axis=1) + bias
    y_bayes = (logits_test >= 0.0).astype(np.float32)

    return FederatedDataset(
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        metadata={
            "n_clients": config.n_clients,
            "dirichlet_alpha": config.dirichlet_alpha,
            "feature_dim": config.feature_dim,
            "n_train": n_train,
        },
        ground_truth={
            "weights": weights,
            "bias": bias,
            "bayes_optimal_acc": float(np.mean(y_test == y_bayes)),
        },
    )

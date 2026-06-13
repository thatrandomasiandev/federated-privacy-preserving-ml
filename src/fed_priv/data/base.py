"""Dataset containers for federated and centralized training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClientPartition:
    """Local training data for one federated client."""

    client_id: int
    X: np.ndarray
    y: np.ndarray
    label_distribution: dict[int, float] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


@dataclass
class FederatedDataset:
    """Global dataset partitioned across clients with ground-truth separator."""

    clients: list[ClientPartition]
    X_test: np.ndarray
    y_test: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @property
    def feature_dim(self) -> int:
        return int(self.X_test.shape[1])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    def stacked_train(self) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all client data (centralized baseline)."""
        X = np.vstack([c.X for c in self.clients])
        y = np.concatenate([c.y for c in self.clients])
        return X, y

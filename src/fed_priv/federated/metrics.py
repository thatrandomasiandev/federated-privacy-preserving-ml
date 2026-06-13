"""Evaluation metrics for classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ClassificationMetrics:
    accuracy: float
    loss: float


def evaluate_classifier(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module | None = None,
) -> ClassificationMetrics:
    """Compute accuracy and average BCE loss on a dataset."""
    model.eval()
    criterion = criterion or nn.BCEWithLogitsLoss()
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t)
        loss = float(criterion(logits, y_t).item())
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = float((preds == y_t).float().mean().item())
    return ClassificationMetrics(accuracy=acc, loss=loss)


def label_skew_entropy(clients: list) -> float:
    """Mean entropy of per-client label distributions (higher = more balanced)."""
    entropies = []
    for client in clients:
        probs = np.array([client.label_distribution.get(0, 0.0), client.label_distribution.get(1, 0.0)])
        probs = probs / (probs.sum() + 1e-8)
        ent = -np.sum(probs * np.log(probs + 1e-8))
        entropies.append(float(ent))
    return float(np.mean(entropies))

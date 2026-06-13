"""Differentially private SGD with per-example gradient clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.federated.metrics import evaluate_classifier
from fed_priv.models.mlp import ClassificationMLP
from fed_priv.privacy.accountant import compute_epsilon
from fed_priv.utils.seed import set_torch_seed


@dataclass
class DPTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.01
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.0
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    seed: int = 42


@dataclass
class DPTrainResult:
    private_test_acc: float
    private_val_acc: float
    non_private_test_acc: float
    utility_gap: float
    epsilon: float
    noise_multiplier: float


def _per_sample_grads(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
) -> list[torch.Tensor]:
    """Compute per-example gradients for a mini-batch."""
    per_sample: list[list[torch.Tensor]] = []
    for i in range(xb.shape[0]):
        model.zero_grad()
        loss = criterion(model(xb[i : i + 1]), yb[i : i + 1])
        loss.backward()
        per_sample.append([p.grad.clone().detach() for p in model.parameters() if p.requires_grad])
    return per_sample


def _clip_and_aggregate(
    per_sample_grads: list[list[torch.Tensor]],
    max_norm: float,
) -> list[torch.Tensor]:
    """Clip per-sample gradients and average."""
    clipped: list[list[torch.Tensor]] = []
    for grads in per_sample_grads:
        flat = torch.cat([g.view(-1) for g in grads])
        norm = flat.norm(2)
        scale = min(1.0, max_norm / (norm.item() + 1e-8))
        clipped.append([g * scale for g in grads])

    n = len(clipped)
    aggregated = []
    for param_idx in range(len(clipped[0])):
        stacked = torch.stack([clipped[i][param_idx] for i in range(n)])
        aggregated.append(stacked.mean(dim=0))
    return aggregated


def _add_gaussian_noise(
    grads: list[torch.Tensor],
    noise_multiplier: float,
    max_norm: float,
) -> list[torch.Tensor]:
    """Add calibrated Gaussian noise to clipped averaged gradients."""
    noisy = []
    sigma = noise_multiplier * max_norm
    for g in grads:
        noise = torch.normal(mean=0.0, std=sigma, size=g.shape)
        noisy.append(g + noise)
    return noisy


def _apply_grads(model: nn.Module, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param.add_(-lr * grad)


def _standard_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: DPTrainConfig,
) -> float:
    set_torch_seed(config.seed)
    model = ClassificationMLP(
        in_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
        dropout=config.dropout,
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(config.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return evaluate_classifier(model, X_test, y_test).accuracy


def run_dp_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: DPTrainConfig | None = None,
) -> DPTrainResult:
    """Train with DP-SGD and compare to non-private baseline."""
    cfg = config or DPTrainConfig()
    set_torch_seed(cfg.seed)

    model = ClassificationMLP(
        in_dim=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
    )
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    steps = 0

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            per_sample = _per_sample_grads(model, xb, yb, criterion)
            avg_grads = _clip_and_aggregate(per_sample, cfg.max_grad_norm)
            noisy_grads = _add_gaussian_noise(avg_grads, cfg.noise_multiplier, cfg.max_grad_norm)
            _apply_grads(model, noisy_grads, cfg.lr)
            steps += 1

    sample_rate = min(cfg.batch_size / len(X_train), 1.0)
    epsilon = compute_epsilon(
        noise_multiplier=cfg.noise_multiplier,
        sample_rate=sample_rate,
        steps=steps,
        delta=cfg.delta,
    )

    private_test = evaluate_classifier(model, X_test, y_test)
    private_val = evaluate_classifier(model, X_val, y_val)
    non_private_acc = _standard_train(X_train, y_train, X_val, y_val, X_test, y_test, cfg)

    return DPTrainResult(
        private_test_acc=private_test.accuracy,
        private_val_acc=private_val.accuracy,
        non_private_test_acc=non_private_acc,
        utility_gap=non_private_acc - private_test.accuracy,
        epsilon=epsilon,
        noise_multiplier=cfg.noise_multiplier,
    )

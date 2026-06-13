"""FedAvg client and server training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.data.base import FederatedDataset
from fed_priv.federated.metrics import ClassificationMetrics, evaluate_classifier, label_skew_entropy
from fed_priv.models.mlp import ClassificationMLP, flatten_state_dict, load_flat_state_dict
from fed_priv.utils.seed import set_torch_seed


@dataclass
class FedTrainConfig:
    federated_rounds: int = 30
    local_epochs: int = 2
    centralized_epochs: int = 60
    batch_size: int = 64
    lr: float = 0.01
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.0
    target_accuracy: float = 0.85
    seed: int = 42


@dataclass
class FedTrainResult:
    federated_test_acc: float
    federated_val_acc: float
    centralized_test_acc: float
    centralized_gap: float
    rounds_to_target: int
    label_skew_entropy: float
    round_history: list[float] = field(default_factory=list)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)


def _local_train(
    model: ClassificationMLP,
    X: np.ndarray,
    y: np.ndarray,
    config: FedTrainConfig,
) -> ClassificationMLP:
    loader = _to_loader(X, y, config.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(config.local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def _centralized_train(
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    set_torch_seed(config.seed)
    X_train, y_train = data.stacked_train()
    model = ClassificationMLP(
        in_dim=data.feature_dim,
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
        dropout=config.dropout,
    )
    loader = _to_loader(X_train, y_train, config.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(config.centralized_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def _fedavg_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    global_flat = flatten_state_dict(global_model)
    client_updates: list[torch.Tensor] = []
    client_sizes: list[int] = []

    for client in data.clients:
        local_model = ClassificationMLP(
            in_dim=data.feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        load_flat_state_dict(local_model, global_flat.clone())
        _local_train(local_model, client.X, client.y, config)
        client_updates.append(flatten_state_dict(local_model))
        client_sizes.append(client.n_samples)

    total = sum(client_sizes)
    weights = torch.tensor(
        [s / total for s in client_sizes],
        dtype=torch.float32,
    )
    stacked = torch.stack(client_updates)
    averaged = (stacked * weights.view(-1, 1)).sum(dim=0)
    load_flat_state_dict(global_model, averaged)
    return global_model


def run_federated_experiment(
    data: FederatedDataset,
    config: FedTrainConfig | None = None,
) -> FedTrainResult:
    """Run FedAvg and centralized baselines on a partitioned dataset."""
    cfg = config or FedTrainConfig()
    set_torch_seed(cfg.seed)

    global_model = ClassificationMLP(
        in_dim=data.feature_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
    )

    round_history: list[float] = []
    rounds_to_target = cfg.federated_rounds

    for round_idx in range(cfg.federated_rounds):
        _fedavg_round(global_model, data, cfg)
        val_metrics = evaluate_classifier(global_model, data.X_val, data.y_val)
        round_history.append(val_metrics.accuracy)
        if val_metrics.accuracy >= cfg.target_accuracy and rounds_to_target == cfg.federated_rounds:
            rounds_to_target = round_idx + 1

    fed_test = evaluate_classifier(global_model, data.X_test, data.y_test)
    fed_val = evaluate_classifier(global_model, data.X_val, data.y_val)

    central_model = _centralized_train(data, cfg)
    central_test = evaluate_classifier(central_model, data.X_test, data.y_test)

    return FedTrainResult(
        federated_test_acc=fed_test.accuracy,
        federated_val_acc=fed_val.accuracy,
        centralized_test_acc=central_test.accuracy,
        centralized_gap=central_test.accuracy - fed_test.accuracy,
        rounds_to_target=rounds_to_target,
        label_skew_entropy=label_skew_entropy(data.clients),
        round_history=round_history,
    )

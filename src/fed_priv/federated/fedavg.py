"""FedAvg, FedProx, FedNova, and FedBN federated training algorithms."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.data.base import FederatedDataset
from fed_priv.federated.metrics import ClassificationMetrics, evaluate_classifier, label_skew_entropy
from fed_priv.models.mlp import ClassificationMLP, flatten_state_dict, load_flat_state_dict
from fed_priv.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


class FedAlgorithm(str, Enum):
    """Supported federated learning algorithms."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    FEDBN = "fedbn"


@dataclass
class FedTrainConfig:
    """Configuration for federated training experiments.

    Args:
        federated_rounds: Number of global communication rounds.
        local_epochs: Number of local SGD epochs per client per round.
        centralized_epochs: Epochs for the centralized baseline.
        batch_size: Mini-batch size for local and centralized training.
        lr: Learning rate for SGD.
        weight_decay: L2 regularization coefficient.
        hidden_dim: Hidden layer width for the MLP.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
        target_accuracy: Accuracy threshold for early convergence detection.
        algorithm: Federated algorithm variant to use.
        mu: Proximal penalty coefficient for FedProx (mu/2 * ||w - w_global||^2).
        seed: Random seed for reproducibility.
    """

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
    algorithm: FedAlgorithm = FedAlgorithm.FEDAVG
    mu: float = 0.01
    seed: int = 42


@dataclass
class FedTrainResult:
    """Result container for a federated training experiment.

    Args:
        federated_test_acc: Final test accuracy of the global model.
        federated_val_acc: Final validation accuracy of the global model.
        centralized_test_acc: Test accuracy of the centralized baseline.
        centralized_gap: centralized_test_acc - federated_test_acc.
        rounds_to_target: Round index at which target_accuracy was first achieved.
        label_skew_entropy: Mean per-client label distribution entropy.
        round_history: Per-round validation accuracy trace.
    """

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
    """Standard local training (FedAvg objective).

    Args:
        model: Local model initialized with global parameters.
        X: Client training features.
        y: Client training labels.
        config: Training configuration.

    Returns:
        Trained local model.
    """
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


def _local_train_fedprox(
    model: ClassificationMLP,
    X: np.ndarray,
    y: np.ndarray,
    config: FedTrainConfig,
    global_params: list[torch.Tensor],
) -> ClassificationMLP:
    """FedProx local training with proximal regularization.

    Minimizes: F_i(w) + (mu/2) * ||w - w_global||^2

    Args:
        model: Local model initialized with global parameters.
        X: Client training features.
        y: Client training labels.
        config: Training configuration (uses config.mu for proximal weight).
        global_params: Snapshot of global model parameters for proximal term.

    Returns:
        Trained local model with proximal penalty applied.
    """
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
            prox_term = torch.tensor(0.0)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox_term = prox_term + ((local_p - global_p) ** 2).sum()
            loss = loss + (config.mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()
    return model


def _local_train_fednova(
    model: ClassificationMLP,
    X: np.ndarray,
    y: np.ndarray,
    config: FedTrainConfig,
) -> tuple[ClassificationMLP, int]:
    """FedNova local training tracking the number of local gradient steps.

    Args:
        model: Local model initialized with global parameters.
        X: Client training features.
        y: Client training labels.
        config: Training configuration.

    Returns:
        Tuple of (trained local model, number of local SGD steps taken).
    """
    loader = _to_loader(X, y, config.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    local_steps = 0
    for _ in range(config.local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            local_steps += 1
    return model, local_steps


def _centralized_train(
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Train a centralized baseline model on all client data combined.

    Args:
        data: Federated dataset with stacked_train() method.
        config: Training configuration.

    Returns:
        Trained centralized model.
    """
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


def _is_bn_param(name: str) -> bool:
    """Check if a parameter name belongs to a BatchNorm layer."""
    bn_keywords = ("bn", "batch_norm", "batchnorm", "running_mean", "running_var", "num_batches_tracked")
    return any(kw in name.lower() for kw in bn_keywords)


def _fedavg_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedAvg: broadcast, local train, aggregate.

    Args:
        global_model: Current global model to distribute.
        data: Federated dataset with client partitions.
        config: Training configuration.

    Returns:
        Updated global model after weighted averaging.
    """
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


def _fedprox_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedProx: broadcast, proximal-regularized local train, aggregate.

    The local objective is: min_w F_i(w) + (mu/2) * ||w - w_global||^2

    Args:
        global_model: Current global model to distribute.
        data: Federated dataset with client partitions.
        config: Training configuration (uses config.mu).

    Returns:
        Updated global model after weighted averaging.
    """
    global_flat = flatten_state_dict(global_model)
    global_params = [p.data.clone().detach() for p in global_model.parameters()]
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
        _local_train_fedprox(local_model, client.X, client.y, config, global_params)
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


def _fednova_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedNova: normalize client updates by local steps.

    Each client's normalized pseudo-gradient is:
        d_k = (w_k - w_global) / tau_k
    where tau_k is the number of local SGD steps taken by client k.

    The global model is updated as:
        w_global += tau_eff * sum(p_k * d_k)
    where tau_eff = sum(p_k * tau_k) and p_k are sample-proportion weights.

    Args:
        global_model: Current global model to distribute.
        data: Federated dataset with client partitions.
        config: Training configuration.

    Returns:
        Updated global model after normalized aggregation.
    """
    global_flat = flatten_state_dict(global_model)
    client_deltas: list[torch.Tensor] = []
    client_sizes: list[int] = []
    client_tau: list[int] = []

    for client in data.clients:
        local_model = ClassificationMLP(
            in_dim=data.feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        load_flat_state_dict(local_model, global_flat.clone())
        local_model, tau_k = _local_train_fednova(local_model, client.X, client.y, config)
        local_flat = flatten_state_dict(local_model)
        delta_k = (local_flat - global_flat) / max(tau_k, 1)
        client_deltas.append(delta_k)
        client_sizes.append(client.n_samples)
        client_tau.append(tau_k)

    total = sum(client_sizes)
    weights = torch.tensor(
        [s / total for s in client_sizes],
        dtype=torch.float32,
    )

    tau_eff = sum(w.item() * t for w, t in zip(weights, client_tau))
    stacked = torch.stack(client_deltas)
    weighted_delta = (stacked * weights.view(-1, 1)).sum(dim=0)
    new_flat = global_flat + tau_eff * weighted_delta
    load_flat_state_dict(global_model, new_flat)
    return global_model


def _fedbn_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedBN: aggregate only non-BatchNorm parameters.

    BatchNorm statistics (running_mean, running_var, weight, bias of BN layers)
    remain local to each client. Only non-BN parameters are averaged globally.

    Args:
        global_model: Current global model to distribute.
        data: Federated dataset with client partitions.
        config: Training configuration.

    Returns:
        Updated global model with only non-BN parameters aggregated.
    """
    global_state = global_model.state_dict()
    non_bn_keys = [k for k in global_state.keys() if not _is_bn_param(k)]
    all_keys = list(global_state.keys())

    client_states: list[dict[str, torch.Tensor]] = []
    client_sizes: list[int] = []

    for client in data.clients:
        local_model = ClassificationMLP(
            in_dim=data.feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        local_model.load_state_dict(copy.deepcopy(global_state))
        _local_train(local_model, client.X, client.y, config)
        client_states.append(local_model.state_dict())
        client_sizes.append(client.n_samples)

    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    new_state = copy.deepcopy(global_state)
    for key in non_bn_keys:
        stacked = torch.stack([cs[key].float() for cs in client_states])
        w = torch.tensor(weights, dtype=torch.float32)
        for _ in range(stacked.dim() - 1):
            w = w.unsqueeze(-1)
        new_state[key] = (stacked * w).sum(dim=0)

    global_model.load_state_dict(new_state)
    return global_model


def run_federated_experiment(
    data: FederatedDataset,
    config: FedTrainConfig | None = None,
) -> FedTrainResult:
    """Run a federated learning experiment with the specified algorithm variant.

    Supports FedAvg, FedProx, FedNova, and FedBN. Also trains a centralized
    baseline for comparison.

    Args:
        data: Federated dataset partitioned across clients.
        config: Training configuration. Defaults to FedTrainConfig().

    Returns:
        FedTrainResult with final metrics, convergence info, and round history.
    """
    cfg = config or FedTrainConfig()
    set_torch_seed(cfg.seed)

    global_model = ClassificationMLP(
        in_dim=data.feature_dim,
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
    )

    round_fn = {
        FedAlgorithm.FEDAVG: _fedavg_round,
        FedAlgorithm.FEDPROX: _fedprox_round,
        FedAlgorithm.FEDNOVA: _fednova_round,
        FedAlgorithm.FEDBN: _fedbn_round,
    }[cfg.algorithm]

    round_history: list[float] = []
    rounds_to_target = cfg.federated_rounds

    for round_idx in range(cfg.federated_rounds):
        round_fn(global_model, data, cfg)
        val_metrics = evaluate_classifier(global_model, data.X_val, data.y_val)
        round_history.append(val_metrics.accuracy)
        if val_metrics.accuracy >= cfg.target_accuracy and rounds_to_target == cfg.federated_rounds:
            rounds_to_target = round_idx + 1

    fed_test = evaluate_classifier(global_model, data.X_test, data.y_test)
    fed_val = evaluate_classifier(global_model, data.X_val, data.y_val)

    central_model = _centralized_train(data, cfg)
    central_test = evaluate_classifier(central_model, data.X_test, data.y_test)

    logger.info(
        "Experiment complete: algorithm=%s fed_acc=%.4f central_acc=%.4f gap=%.4f",
        cfg.algorithm.value,
        fed_test.accuracy,
        central_test.accuracy,
        central_test.accuracy - fed_test.accuracy,
    )

    return FedTrainResult(
        federated_test_acc=fed_test.accuracy,
        federated_val_acc=fed_val.accuracy,
        centralized_test_acc=central_test.accuracy,
        centralized_gap=central_test.accuracy - fed_test.accuracy,
        rounds_to_target=rounds_to_target,
        label_skew_entropy=label_skew_entropy(data.clients),
        round_history=round_history,
    )

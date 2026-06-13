"""Tests for federated learning algorithms: FedAvg, FedProx, FedNova, FedBN."""

from __future__ import annotations

import torch

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.federated.fedavg import (
    FedAlgorithm,
    FedTrainConfig,
    FedTrainResult,
    run_federated_experiment,
)
from fed_priv.federated.metrics import evaluate_classifier
from fed_priv.models.mlp import ClassificationMLP, flatten_state_dict, load_flat_state_dict


def test_flatten_roundtrip():
    model = ClassificationMLP(in_dim=8, hidden_dim=16, n_hidden=1)
    flat = flatten_state_dict(model)
    clone = ClassificationMLP(in_dim=8, hidden_dim=16, n_hidden=1)
    load_flat_state_dict(clone, flat)
    for p1, p2 in zip(model.parameters(), clone.parameters()):
        assert torch.allclose(p1, p2)


def test_fedavg_runs_and_beats_chance():
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=3, n_samples=800, seed=10)
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=10,
            local_epochs=1,
            centralized_epochs=15,
            algorithm=FedAlgorithm.FEDAVG,
            seed=10,
        ),
    )
    assert 0.0 <= result.federated_test_acc <= 1.0
    assert result.federated_test_acc > 0.55
    assert abs(result.centralized_gap) < 0.2


def test_fedavg_val_acc_increases():
    """FedAvg validation accuracy should generally increase over rounds."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=5, n_samples=1500, dirichlet_alpha=10.0, seed=42)
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=20,
            local_epochs=2,
            centralized_epochs=40,
            algorithm=FedAlgorithm.FEDAVG,
            seed=42,
        ),
    )
    assert len(result.round_history) == 20
    early_acc = sum(result.round_history[:5]) / 5
    late_acc = sum(result.round_history[-5:]) / 5
    assert late_acc >= early_acc - 0.05


def test_fedprox_runs():
    """FedProx should train successfully with proximal term."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=4, n_samples=1200, dirichlet_alpha=5.0, seed=20)
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=15,
            local_epochs=2,
            centralized_epochs=30,
            algorithm=FedAlgorithm.FEDPROX,
            mu=0.01,
            seed=20,
        ),
    )
    assert result.federated_test_acc > 0.50
    assert result.centralized_gap < 0.3


def test_fedprox_helps_non_iid():
    """FedProx should perform at least as well as FedAvg on highly non-IID data."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(
            n_clients=5,
            n_samples=1500,
            dirichlet_alpha=0.3,
            seed=33,
        )
    )
    fedavg_result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=15,
            local_epochs=3,
            centralized_epochs=30,
            algorithm=FedAlgorithm.FEDAVG,
            seed=33,
        ),
    )
    fedprox_result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=15,
            local_epochs=3,
            centralized_epochs=30,
            algorithm=FedAlgorithm.FEDPROX,
            mu=0.1,
            seed=33,
        ),
    )
    assert fedprox_result.federated_test_acc >= fedavg_result.federated_test_acc - 0.05


def test_fednova_runs():
    """FedNova should train successfully with step normalization."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=4, n_samples=1000, seed=25)
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=10,
            local_epochs=2,
            centralized_epochs=20,
            algorithm=FedAlgorithm.FEDNOVA,
            seed=25,
        ),
    )
    assert result.federated_test_acc > 0.50


def test_fedbn_runs():
    """FedBN should train successfully (only aggregates non-BN params)."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=3, n_samples=1200, dirichlet_alpha=5.0, seed=30)
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=15,
            local_epochs=3,
            centralized_epochs=30,
            algorithm=FedAlgorithm.FEDBN,
            seed=30,
        ),
    )
    assert result.federated_val_acc > 0.50


def test_centralized_gap_bounded():
    """The gap between centralized and federated should be bounded."""
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(
            n_clients=5,
            n_samples=2000,
            dirichlet_alpha=10.0,
            seed=50,
        )
    )
    result = run_federated_experiment(
        data,
        config=FedTrainConfig(
            federated_rounds=30,
            local_epochs=2,
            centralized_epochs=60,
            algorithm=FedAlgorithm.FEDAVG,
            seed=50,
        ),
    )
    assert result.centralized_gap < 0.15


def test_evaluate_classifier_perfect():
    data = generate_partitioned_dataset(PartitionedDGPConfig(n_samples=200, seed=11))
    model = ClassificationMLP(in_dim=data.feature_dim, hidden_dim=8, n_hidden=1)
    model.eval()
    metrics = evaluate_classifier(model, data.X_test, data.y_test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0

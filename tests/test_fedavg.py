"""Tests for FedAvg training."""

import torch

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.federated.fedavg import FedTrainConfig, run_federated_experiment
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
            seed=10,
        ),
    )
    assert 0.0 <= result.federated_test_acc <= 1.0
    assert result.federated_test_acc > 0.55
    assert abs(result.centralized_gap) < 0.2


def test_evaluate_classifier_perfect():
    data = generate_partitioned_dataset(PartitionedDGPConfig(n_samples=200, seed=11))
    model = ClassificationMLP(in_dim=data.feature_dim, hidden_dim=8, n_hidden=1)
    model.eval()
    # Random untrained model should still produce valid metrics
    metrics = evaluate_classifier(model, data.X_test, data.y_test)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0

"""Tests for DP-SGD and privacy accounting."""

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.privacy.accountant import compute_epsilon
from fed_priv.privacy.dp_sgd import DPTrainConfig, run_dp_experiment


def test_epsilon_decreases_with_more_noise():
    eps_low = compute_epsilon(
        noise_multiplier=0.5,
        sample_rate=0.1,
        steps=100,
        delta=1e-5,
    )
    eps_high = compute_epsilon(
        noise_multiplier=2.0,
        sample_rate=0.1,
        steps=100,
        delta=1e-5,
    )
    assert eps_high < eps_low


def test_dp_experiment_runs():
    data = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=1, n_samples=600, seed=20)
    )
    X_train, y_train = data.stacked_train()
    result = run_dp_experiment(
        X_train,
        y_train,
        data.X_val,
        data.y_val,
        data.X_test,
        data.y_test,
        config=DPTrainConfig(epochs=5, noise_multiplier=1.5, seed=20),
    )
    assert result.epsilon > 0
    assert 0.0 <= result.private_test_acc <= 1.0
    assert result.utility_gap >= 0.0

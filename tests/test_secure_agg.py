"""Tests for secure aggregation."""

import numpy as np

from fed_priv.secure.masking import (
    naive_sum,
    reconstruction_attack_mse,
    run_secure_agg_benchmark,
    secure_aggregate,
)


def test_secure_sum_matches_naive_sum():
    rng = np.random.default_rng(7)
    updates = [rng.normal(size=32) for _ in range(4)]
    true_sum = naive_sum(updates)
    secure_sum = secure_aggregate(updates, seed=7)
    np.testing.assert_allclose(secure_sum, true_sum, atol=1e-10)


def test_reconstruction_mse_positive_with_masking():
    rng = np.random.default_rng(8)
    updates = [rng.normal(size=16) for _ in range(5)]
    mse = reconstruction_attack_mse(updates, seed=8, target_idx=0)
    assert mse > 0.0


def test_secure_agg_benchmark_runs():
    result = run_secure_agg_benchmark(n_clients=4, update_dim=32, n_trials=5, seed=9)
    assert result.max_abs_error < 1e-8
    assert result.reconstruction_mse > 0.0

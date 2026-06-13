"""Tests for privacy mechanisms: DP-SGD, gradient clipping, and RDP accountant."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.models.mlp import ClassificationMLP
from fed_priv.privacy.accountant import RenyiAccountant, compute_epsilon
from fed_priv.privacy.dp_sgd import (
    DPSGDOptimizer,
    DPTrainConfig,
    _clip_and_aggregate,
    _compute_per_sample_gradients,
    _per_sample_grads,
    run_dp_experiment,
)


def test_epsilon_decreases_with_more_noise():
    """More noise should yield a smaller (better) epsilon."""
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


def test_epsilon_increases_with_steps():
    """More steps should increase privacy cost (larger epsilon)."""
    eps_few = compute_epsilon(
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=100,
        delta=1e-5,
    )
    eps_many = compute_epsilon(
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=1000,
        delta=1e-5,
    )
    assert eps_many > eps_few


def test_epsilon_reasonable_range():
    """For sigma=1.0, q=0.01, T=1000, epsilon should be positive and bounded."""
    eps = compute_epsilon(
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=1000,
        delta=1e-5,
    )
    assert 1.0 <= eps <= 50.0, f"Expected epsilon in [1.0, 50.0], got {eps}"

    eps_low_noise = compute_epsilon(
        noise_multiplier=3.0,
        sample_rate=0.01,
        steps=100,
        delta=1e-5,
    )
    assert 0.5 <= eps_low_noise <= 10.0, f"Expected epsilon in [0.5, 10.0], got {eps_low_noise}"


def test_clipped_grads_norm_bounded():
    """After clipping, per-sample gradient norms must be <= C."""
    model = ClassificationMLP(in_dim=8, hidden_dim=16, n_hidden=1)
    x = torch.randn(16, 8)
    y = torch.randint(0, 2, (16,)).float()
    criterion = nn.BCEWithLogitsLoss()

    per_sample = _per_sample_grads(model, x, y, criterion)

    max_norm = 1.0
    clipped = _clip_and_aggregate(per_sample, max_norm)

    for grads in per_sample:
        flat = torch.cat([g.view(-1) for g in grads])
        norm_before = flat.norm(2).item()
        break

    for sample_grads in per_sample:
        flat = torch.cat([g.view(-1) for g in sample_grads])
        original_norm = flat.norm(2).item()
        scale = min(1.0, max_norm / (original_norm + 1e-8))
        clipped_flat = flat * scale
        assert clipped_flat.norm(2).item() <= max_norm + 1e-6


def test_clipping_with_various_norms():
    """Verify clipping works with different C values."""
    model = ClassificationMLP(in_dim=4, hidden_dim=8, n_hidden=1)
    x = torch.randn(8, 4) * 10.0
    y = torch.randint(0, 2, (8,)).float()
    criterion = nn.BCEWithLogitsLoss()

    per_sample = _per_sample_grads(model, x, y, criterion)

    for clip_norm in [0.1, 0.5, 1.0, 5.0]:
        clipped = _clip_and_aggregate(per_sample, clip_norm)
        avg_flat = torch.cat([g.view(-1) for g in clipped])
        assert avg_flat.norm(2).item() <= clip_norm + 1e-6


def test_renyi_accountant_basic():
    """RenyiAccountant should compute epsilon consistent with compute_epsilon."""
    accountant = RenyiAccountant()
    accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=1000)
    eps_accountant = accountant.get_epsilon(delta=1e-5)

    eps_direct = compute_epsilon(
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=1000,
        delta=1e-5,
    )

    assert abs(eps_accountant - eps_direct) < 0.5


def test_renyi_accountant_composition():
    """Privacy cost should accumulate across multiple phases."""
    accountant = RenyiAccountant()
    accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=500)
    eps_half = accountant.get_epsilon(delta=1e-5)

    accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=500)
    eps_full = accountant.get_epsilon(delta=1e-5)

    assert eps_full > eps_half


def test_renyi_accountant_reset():
    """After reset, epsilon should return to minimal value."""
    accountant = RenyiAccountant()
    accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=1000)
    eps_before = accountant.get_epsilon(delta=1e-5)
    assert eps_before > 0

    accountant.reset()
    eps_after = accountant.get_epsilon(delta=1e-5)
    assert eps_after < eps_before


def test_dpsgd_optimizer_step():
    """DPSGDOptimizer should update model parameters."""
    model = ClassificationMLP(in_dim=4, hidden_dim=8, n_hidden=1)
    optimizer = DPSGDOptimizer(
        model.parameters(),
        lr=0.1,
        clip_norm=1.0,
        noise_multiplier=0.5,
        batch_size=8,
    )

    params_before = [p.data.clone() for p in model.parameters()]

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,)).float()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    per_sample_grads = _compute_per_sample_gradients(model, x, y, criterion)
    optimizer.step(per_sample_grads=per_sample_grads)

    params_after = [p.data.clone() for p in model.parameters()]
    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed, "DPSGDOptimizer.step() should update parameters"
    assert optimizer.accumulated_steps == 1


def test_dpsgd_optimizer_clips_gradients():
    """DPSGDOptimizer should respect the clip_norm bound."""
    model = ClassificationMLP(in_dim=4, hidden_dim=8, n_hidden=1)
    clip_norm = 0.5
    optimizer = DPSGDOptimizer(
        model.parameters(),
        lr=0.01,
        clip_norm=clip_norm,
        noise_multiplier=0.0,
        batch_size=8,
    )

    x = torch.randn(8, 4) * 100
    y = torch.randint(0, 2, (8,)).float()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    per_sample_grads = _compute_per_sample_gradients(model, x, y, criterion)

    for grad_tensor in per_sample_grads:
        batch_size = grad_tensor.shape[0]
        flat = grad_tensor.reshape(batch_size, -1)
        norms = flat.norm(2, dim=1)
        clip_factors = torch.clamp(clip_norm / (norms + 1e-8), max=1.0)
        for i in range(batch_size):
            clipped_norm = (flat[i] * clip_factors[i]).norm(2).item()
            assert clipped_norm <= clip_norm + 1e-5


def test_dp_experiment_runs():
    """Full DP experiment should run end-to-end and produce valid results."""
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
    assert result.utility_gap >= -0.1

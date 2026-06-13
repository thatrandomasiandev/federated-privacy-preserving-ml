"""Differentially private SGD with per-example gradient clipping.

Implements both a functional DP-SGD training pipeline and a reusable
DPSGDOptimizer that wraps any model's training step with per-sample
gradient clipping and Gaussian noise injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.federated.metrics import evaluate_classifier
from fed_priv.models.mlp import ClassificationMLP
from fed_priv.privacy.accountant import compute_epsilon
from fed_priv.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


@dataclass
class DPTrainConfig:
    """Configuration for differentially private training.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size (also determines the sampling rate q = B/N).
        lr: Learning rate.
        weight_decay: L2 regularization.
        hidden_dim: Hidden layer width for MLP.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
        max_grad_norm: Per-sample gradient clipping bound C.
        noise_multiplier: Noise scale sigma (noise std = sigma * C).
        delta: Target delta for (epsilon, delta)-DP.
        seed: Random seed.
    """

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
    """Result container for DP-SGD training.

    Args:
        private_test_acc: Test accuracy of DP-trained model.
        private_val_acc: Validation accuracy of DP-trained model.
        non_private_test_acc: Test accuracy of non-private baseline.
        utility_gap: non_private_test_acc - private_test_acc.
        epsilon: Computed (epsilon, delta)-DP guarantee.
        noise_multiplier: Noise multiplier used.
    """

    private_test_acc: float
    private_val_acc: float
    non_private_test_acc: float
    utility_gap: float
    epsilon: float
    noise_multiplier: float


def _compute_per_sample_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
) -> list[torch.Tensor]:
    """Compute per-sample gradients for a mini-batch.

    Uses torch.vmap (functorch) if available for vectorized computation,
    otherwise falls back to a sequential loop over individual samples.

    Args:
        model: Neural network module.
        xb: Input batch tensor of shape (B, *).
        yb: Target batch tensor of shape (B,) or (B, *).
        loss_fn: Loss function (must support reduction='sum' or per-sample).

    Returns:
        List of per-sample gradient tensors, each of shape (B, num_params_in_layer).
        Organized as list over parameters, each entry is (B, *param_shape).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    try:
        from torch.func import grad, vmap
        from torch.func import functional_call

        params_dict = {k: v for k, v in model.named_parameters() if v.requires_grad}
        param_names = list(params_dict.keys())
        param_values = tuple(params_dict.values())

        def compute_loss_stateless(params_tuple: tuple, x_single: torch.Tensor, y_single: torch.Tensor) -> torch.Tensor:
            params_map = dict(zip(param_names, params_tuple))
            output = functional_call(model, params_map, (x_single.unsqueeze(0),))
            return loss_fn(output.squeeze(0), y_single)

        grad_fn = grad(compute_loss_stateless)
        per_sample_grads_tuple = vmap(grad_fn, in_dims=(None, 0, 0))(param_values, xb, yb)
        return list(per_sample_grads_tuple)

    except (ImportError, RuntimeError):
        pass

    batch_size = xb.shape[0]
    per_sample: list[list[torch.Tensor]] = []
    for i in range(batch_size):
        model.zero_grad()
        output = model(xb[i:i + 1])
        loss = loss_fn(output.squeeze(0), yb[i])
        loss.backward()
        per_sample.append([p.grad.clone().detach() for p in params])

    result = []
    for param_idx in range(len(params)):
        stacked = torch.stack([per_sample[i][param_idx] for i in range(batch_size)])
        result.append(stacked)
    return result


class DPSGDOptimizer(torch.optim.Optimizer):
    """Differentially Private SGD optimizer with per-sample clipping and noise.

    Implements the DP-SGD mechanism from Abadi et al. (2016):
        1. Compute per-sample gradients g_i for each sample in the batch
        2. Clip: g_i_clipped = g_i * min(1, C / ||g_i||_2)
        3. Aggregate: g_avg = (1/B) * sum(g_i_clipped)
        4. Add noise: g_noisy = g_avg + N(0, (sigma * C / B)^2 * I)
        5. Update: theta = theta - lr * g_noisy

    Args:
        params: Model parameters to optimize.
        lr: Learning rate (default: 0.01).
        clip_norm: Maximum L2 norm for per-sample gradient clipping C.
        noise_multiplier: Noise scale sigma; noise std = sigma * C / batch_size.
        batch_size: Expected batch size B for noise calibration.

    Math:
        The per-step (epsilon, delta)-DP guarantee depends on:
        - Sampling rate q = B / N
        - Noise multiplier sigma
        - Number of steps T
        Composed via RDP: epsilon(delta) = min_alpha [T * rdp(alpha) + log(1/delta)/(alpha-1)]
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        batch_size: int = 64,
    ) -> None:
        defaults = dict(lr=lr, clip_norm=clip_norm, noise_multiplier=noise_multiplier, batch_size=batch_size)
        super().__init__(params, defaults)
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self._step_count = 0

    @property
    def accumulated_steps(self) -> int:
        """Number of optimization steps performed so far."""
        return self._step_count

    def step(self, per_sample_grads: list[torch.Tensor] | None = None, closure: Callable | None = None) -> None:
        """Perform one DP-SGD optimization step.

        Args:
            per_sample_grads: List of per-sample gradient tensors, one per
                parameter group. Each tensor has shape (B, *param_shape).
                If None, uses .grad attributes (assumes already processed).
            closure: Optional closure for loss re-evaluation (unused).
        """
        if per_sample_grads is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        with torch.no_grad():
                            p.add_(-group["lr"] * p.grad)
            self._step_count += 1
            return

        param_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    param_list.append((p, group))

        for idx, (param, group) in enumerate(param_list):
            if idx >= len(per_sample_grads):
                break

            grads = per_sample_grads[idx]
            batch_size = grads.shape[0]

            flat_grads = grads.reshape(batch_size, -1)
            norms = flat_grads.norm(2, dim=1)
            clip_factors = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)

            for _ in range(grads.dim() - 1):
                clip_factors = clip_factors.unsqueeze(-1)
            clipped = grads * clip_factors

            aggregated = clipped.mean(dim=0)

            noise_std = self.noise_multiplier * self.clip_norm / batch_size
            noise = torch.normal(mean=0.0, std=noise_std, size=aggregated.shape, device=aggregated.device)
            noisy_grad = aggregated + noise

            with torch.no_grad():
                param.add_(-group["lr"] * noisy_grad)

        self._step_count += 1


def _per_sample_grads(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
) -> list[list[torch.Tensor]]:
    """Compute per-example gradients for a mini-batch (legacy loop implementation).

    Args:
        model: Neural network.
        xb: Input batch.
        yb: Target batch.
        criterion: Loss function.

    Returns:
        List of per-sample gradient lists (outer: samples, inner: parameters).
    """
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
    """Clip per-sample gradients to max_norm and average.

    Args:
        per_sample_grads: Per-sample gradient lists.
        max_norm: Maximum L2 norm C.

    Returns:
        List of averaged clipped gradients (one per parameter).
    """
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
    """Add calibrated Gaussian noise to clipped averaged gradients.

    Args:
        grads: Clipped and averaged gradient tensors.
        noise_multiplier: Sigma parameter.
        max_norm: Clipping bound C (noise std = sigma * C).

    Returns:
        Noisy gradient tensors.
    """
    noisy = []
    sigma = noise_multiplier * max_norm
    for g in grads:
        noise = torch.normal(mean=0.0, std=sigma, size=g.shape)
        noisy.append(g + noise)
    return noisy


def _apply_grads(model: nn.Module, grads: list[torch.Tensor], lr: float) -> None:
    """Apply gradient update to model parameters in-place."""
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
    """Train a model without privacy and return test accuracy."""
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
    """Train with DP-SGD and compare to non-private baseline.

    Args:
        X_train: Training features of shape (N, D).
        y_train: Training labels of shape (N,).
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        config: DP training configuration.

    Returns:
        DPTrainResult with privacy guarantee and accuracy metrics.
    """
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

    logger.info(
        "DP experiment: eps=%.2f sigma=%.2f private_acc=%.4f non_private_acc=%.4f",
        epsilon,
        cfg.noise_multiplier,
        private_test.accuracy,
        non_private_acc,
    )

    return DPTrainResult(
        private_test_acc=private_test.accuracy,
        private_val_acc=private_val.accuracy,
        non_private_test_acc=non_private_acc,
        utility_gap=non_private_acc - private_test.accuracy,
        epsilon=epsilon,
        noise_multiplier=cfg.noise_multiplier,
    )

"""Privacy attack metrics: membership inference and gradient inversion.

Provides empirical privacy evaluation tools that measure the practical
privacy leakage of trained models, complementing the theoretical guarantees
from differential privacy accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.models.mlp import ClassificationMLP
from fed_priv.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


@dataclass
class MembershipInferenceResult:
    """Result of a membership inference attack.

    Args:
        attack_auc: AUC of the membership classifier (0.5 = random, 1.0 = perfect).
        attack_accuracy: Binary accuracy of the attack at optimal threshold.
        threshold: Loss threshold used for classification.
        mean_member_loss: Mean loss on member (training) samples.
        mean_nonmember_loss: Mean loss on non-member (test) samples.
    """

    attack_auc: float
    attack_accuracy: float
    threshold: float
    mean_member_loss: float
    mean_nonmember_loss: float


@dataclass
class GradientInversionResult:
    """Result of a gradient inversion (DLG) attack evaluation.

    Args:
        reconstruction_mse: MSE between reconstructed and original input.
        reconstruction_cosine: Cosine similarity between gradient of reconstructed
            and original gradient.
        converged: Whether the inversion optimization converged.
        n_iterations: Number of optimization iterations performed.
    """

    reconstruction_mse: float
    reconstruction_cosine: float
    converged: bool
    n_iterations: int


def _compute_sample_losses(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-sample losses for a dataset.

    Args:
        model: Trained model to evaluate.
        X: Features of shape (N, D).
        y: Labels of shape (N,).
        batch_size: Batch size for efficient evaluation.

    Returns:
        Array of per-sample losses of shape (N,).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=False)

    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            batch_losses = criterion(logits, yb)
            losses.append(batch_losses.cpu().numpy())

    return np.concatenate(losses)


def _train_shadow_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    in_dim: int,
    hidden_dim: int = 32,
    n_hidden: int = 1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.01,
    seed: int = 0,
) -> ClassificationMLP:
    """Train a shadow model on a subset of data.

    Shadow models mimic the target model's behavior and are used to
    calibrate the membership inference attack threshold.

    Args:
        X_train: Training features.
        y_train: Training labels.
        in_dim: Input dimensionality.
        hidden_dim: Hidden layer width.
        n_hidden: Number of hidden layers.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        Trained shadow model.
    """
    set_torch_seed(seed)
    model = ClassificationMLP(in_dim=in_dim, hidden_dim=hidden_dim, n_hidden=n_hidden)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model


def membership_inference_attack(
    target_model: nn.Module,
    X_member: np.ndarray,
    y_member: np.ndarray,
    X_nonmember: np.ndarray,
    y_nonmember: np.ndarray,
    n_shadow_models: int = 3,
    shadow_train_size: int = 500,
    seed: int = 42,
) -> MembershipInferenceResult:
    """Perform a loss-threshold membership inference attack.

    Trains shadow models to calibrate the attack threshold, then classifies
    samples as members (training data) or non-members based on whether their
    loss is below the threshold.

    The attack exploits the observation that trained models tend to have lower
    loss on their training data than on unseen data.

    Args:
        target_model: The model being attacked.
        X_member: Features of known training (member) samples.
        y_member: Labels of known training (member) samples.
        X_nonmember: Features of known non-training (non-member) samples.
        y_nonmember: Labels of non-member samples.
        n_shadow_models: Number of shadow models to train for calibration.
        shadow_train_size: Training set size for each shadow model.
        seed: Random seed for reproducibility.

    Returns:
        MembershipInferenceResult with AUC, accuracy, and diagnostic stats.

    Math:
        Attack rule: predict "member" if L(model, x, y) < tau
        Threshold tau is chosen to maximize accuracy on shadow model data.
        AUC measures the overall discriminative power of the loss signal.
    """
    set_torch_seed(seed)
    rng = np.random.default_rng(seed)
    in_dim = X_member.shape[1]

    member_losses = _compute_sample_losses(target_model, X_member, y_member)
    nonmember_losses = _compute_sample_losses(target_model, X_nonmember, y_nonmember)

    shadow_thresholds: list[float] = []
    all_data = np.vstack([X_member, X_nonmember])
    all_labels = np.concatenate([y_member, y_nonmember])

    for s in range(n_shadow_models):
        n_total = len(all_data)
        perm = rng.permutation(n_total)
        train_size = min(shadow_train_size, n_total // 2)
        shadow_train_idx = perm[:train_size]
        shadow_test_idx = perm[train_size:train_size * 2]

        shadow_model = _train_shadow_model(
            all_data[shadow_train_idx],
            all_labels[shadow_train_idx],
            in_dim=in_dim,
            seed=seed + s + 100,
        )

        shadow_member_loss = _compute_sample_losses(
            shadow_model, all_data[shadow_train_idx], all_labels[shadow_train_idx]
        )
        shadow_nonmember_loss = _compute_sample_losses(
            shadow_model, all_data[shadow_test_idx], all_labels[shadow_test_idx]
        )
        threshold = float((shadow_member_loss.mean() + shadow_nonmember_loss.mean()) / 2.0)
        shadow_thresholds.append(threshold)

    threshold = float(np.mean(shadow_thresholds)) if shadow_thresholds else float(
        (member_losses.mean() + nonmember_losses.mean()) / 2.0
    )

    all_losses = np.concatenate([member_losses, nonmember_losses])
    true_labels = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(nonmember_losses)),
    ])

    scores = -all_losses
    auc = float(roc_auc_score(true_labels, scores))

    predictions = (all_losses < threshold).astype(float)
    accuracy = float(np.mean(predictions == true_labels))

    logger.info(
        "MIA attack: AUC=%.4f accuracy=%.4f threshold=%.4f",
        auc,
        accuracy,
        threshold,
    )

    return MembershipInferenceResult(
        attack_auc=auc,
        attack_accuracy=accuracy,
        threshold=threshold,
        mean_member_loss=float(member_losses.mean()),
        mean_nonmember_loss=float(nonmember_losses.mean()),
    )


def gradient_inversion_loss(
    model: nn.Module,
    target_gradient: list[torch.Tensor],
    input_shape: tuple[int, ...],
    n_iterations: int = 300,
    lr: float = 0.1,
    seed: int = 42,
) -> GradientInversionResult:
    """Deep Leakage from Gradients (DLG) attack metric.

    Attempts to reconstruct the original input by optimizing a dummy input
    to minimize the distance between its gradient and the observed gradient.

    This measures how much information about training data leaks through
    shared gradients in federated learning.

    Args:
        model: The model whose gradient was observed.
        target_gradient: List of gradient tensors (one per parameter) that
            were shared/observed by the attacker.
        input_shape: Shape of the input to reconstruct (e.g., (D,) for a
            single sample or (B, D) for a batch).
        n_iterations: Maximum number of optimization iterations.
        lr: Learning rate for the reconstruction optimizer.
        seed: Random seed for initialization.

    Returns:
        GradientInversionResult with reconstruction quality metrics.

    Math:
        Reconstruction objective:
            x*, y* = argmin_{x, y} || grad_theta L(model(x), y) - target_gradient ||^2

        The cosine similarity between reconstructed and target gradients
        indicates convergence quality.
    """
    set_torch_seed(seed)
    model.eval()

    dummy_x = torch.randn(input_shape, requires_grad=True)
    dummy_y = torch.sigmoid(torch.randn(input_shape[0] if len(input_shape) > 1 else 1))
    dummy_y = dummy_y.detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_x, dummy_y], lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    converged = False

    for iteration in range(n_iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(dummy_x)
            target_y = torch.clamp(dummy_y, 0.0, 1.0)
            if pred.dim() == 1 and target_y.dim() == 1:
                loss = criterion(pred, target_y)
            else:
                loss = criterion(pred.squeeze(), target_y.squeeze())
            loss.backward(create_graph=True)

            grad_diff = torch.tensor(0.0)
            for param, target_g in zip(model.parameters(), target_gradient):
                if param.grad is not None:
                    grad_diff = grad_diff + ((param.grad - target_g) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        loss_val = optimizer.step(closure)
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()

        if loss_val < best_loss:
            best_loss = loss_val
        if loss_val < 1e-6:
            converged = True
            break

    model.zero_grad()
    pred = model(dummy_x)
    target_y = torch.clamp(dummy_y, 0.0, 1.0).detach()
    if pred.dim() == 1 and target_y.dim() == 1:
        loss = criterion(pred, target_y)
    else:
        loss = criterion(pred.squeeze(), target_y.squeeze())
    loss.backward()

    recon_grads = [p.grad.clone().detach() for p in model.parameters() if p.grad is not None]

    flat_recon = torch.cat([g.view(-1) for g in recon_grads])
    flat_target = torch.cat([g.view(-1) for g in target_gradient])
    cosine_sim = float(
        torch.nn.functional.cosine_similarity(flat_recon.unsqueeze(0), flat_target.unsqueeze(0)).item()
    )

    reconstruction_mse = best_loss

    logger.info(
        "DLG attack: MSE=%.6f cosine=%.4f converged=%s iters=%d",
        reconstruction_mse,
        cosine_sim,
        converged,
        n_iterations,
    )

    return GradientInversionResult(
        reconstruction_mse=reconstruction_mse,
        reconstruction_cosine=cosine_sim,
        converged=converged,
        n_iterations=n_iterations,
    )

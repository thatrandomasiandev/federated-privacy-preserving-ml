"""Personalized federated learning: pFedMe and MAML-based methods.

Implements personalization strategies where each client maintains a
local model that adapts to their specific data distribution while still
benefiting from collaborative knowledge through federation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fed_priv.data.base import ClientPartition, FederatedDataset
from fed_priv.federated.metrics import evaluate_classifier
from fed_priv.models.mlp import ClassificationMLP, flatten_state_dict, load_flat_state_dict
from fed_priv.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


@dataclass
class PFedMeConfig:
    """Configuration for pFedMe personalized federated learning.

    Args:
        federated_rounds: Number of global communication rounds.
        local_epochs: Number of local Moreau-envelope optimization steps per round.
        inner_steps: Number of inner gradient steps to solve the Moreau envelope.
        batch_size: Mini-batch size for local training.
        lr: Learning rate for inner optimization (solving w_i*).
        beta: Server-side learning rate for updating global theta.
        lambd: Moreau envelope regularization strength lambda.
        hidden_dim: Hidden layer width.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
        seed: Random seed.
    """

    federated_rounds: int = 30
    local_epochs: int = 5
    inner_steps: int = 5
    batch_size: int = 64
    lr: float = 0.01
    beta: float = 1.0
    lambd: float = 15.0
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.0
    seed: int = 42


@dataclass
class PFedMeResult:
    """Result container for pFedMe experiments.

    Args:
        personalized_test_acc: Mean test accuracy using personalized local models.
        global_test_acc: Test accuracy of the global (theta) model.
        per_client_acc: Per-client personalized test accuracies.
        round_history: Per-round global validation accuracy.
    """

    personalized_test_acc: float
    global_test_acc: float
    per_client_acc: list[float] = field(default_factory=list)
    round_history: list[float] = field(default_factory=list)


@dataclass
class MAMLFedConfig:
    """Configuration for MAML-based federated meta-learning.

    Args:
        federated_rounds: Number of global meta-training rounds.
        meta_lr: Outer-loop (meta) learning rate.
        inner_lr: Inner-loop (task-specific adaptation) learning rate.
        inner_steps: Number of gradient steps for task adaptation.
        batch_size: Mini-batch size for support/query split.
        hidden_dim: Hidden layer width.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
        seed: Random seed.
    """

    federated_rounds: int = 30
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 5
    batch_size: int = 64
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.0
    seed: int = 42


@dataclass
class MAMLFedResult:
    """Result container for MAML federated experiments.

    Args:
        adapted_test_acc: Mean test accuracy after local adaptation.
        meta_test_acc: Test accuracy of the meta-model without adaptation.
        per_client_acc: Per-client adapted test accuracies.
        round_history: Per-round meta-validation accuracy.
    """

    adapted_test_acc: float
    meta_test_acc: float
    per_client_acc: list[float] = field(default_factory=list)
    round_history: list[float] = field(default_factory=list)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


class pFedMe:
    """Personalized Federated Learning with Moreau Envelopes (pFedMe).

    Each client solves a personalized optimization via the Moreau envelope:
        w_i* = argmin_w [F_i(w) + (lambda/2) * ||w - theta||^2]

    The global model theta is updated as:
        theta <- theta - beta * (theta - (1/K) * sum_k w_k*)

    This decouples personalization (through w_i*) from the global model
    (theta), allowing clients to have locally adapted models while still
    contributing to a shared representation.

    Args:
        config: pFedMe training configuration.
        feature_dim: Input feature dimensionality.

    Math:
        Inner problem (per client i):
            w_i* = argmin_w [ F_i(w) + (lambda/2)||w - theta||^2 ]
        Solved via K gradient steps:
            w <- w - lr * (grad_F_i(w) + lambda * (w - theta))

        Outer update (server):
            theta <- theta - beta * (theta - mean(w_i*))
            Equivalently: theta <- (1-beta)*theta + beta*mean(w_i*)

    Example:
        >>> pfedme = pFedMe(PFedMeConfig(federated_rounds=20), feature_dim=12)
        >>> result = pfedme.train(data)
    """

    def __init__(self, config: PFedMeConfig, feature_dim: int) -> None:
        self.config = config
        self.feature_dim = feature_dim
        set_torch_seed(config.seed)
        self.global_model = ClassificationMLP(
            in_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        self.client_models: list[ClassificationMLP] = []

    def _solve_moreau_envelope(
        self,
        client: ClientPartition,
        theta: ClassificationMLP,
    ) -> ClassificationMLP:
        """Solve the personalized Moreau envelope subproblem for one client.

        Finds w_i* = argmin_w [F_i(w) + (lambda/2)||w - theta||^2] via
        gradient descent on the augmented objective.

        Args:
            client: Client's local data partition.
            theta: Current global model (reference point for regularization).

        Returns:
            Personalized local model w_i*.
        """
        local_model = ClassificationMLP(
            in_dim=self.feature_dim,
            hidden_dim=self.config.hidden_dim,
            n_hidden=self.config.n_hidden,
            dropout=self.config.dropout,
        )
        local_model.load_state_dict(copy.deepcopy(theta.state_dict()))
        theta_params = [p.data.clone().detach() for p in theta.parameters()]

        loader = _to_loader(client.X, client.y, self.config.batch_size)
        criterion = nn.BCEWithLogitsLoss()

        local_model.train()
        for _ in range(self.config.local_epochs):
            for xb, yb in loader:
                for _ in range(self.config.inner_steps):
                    local_model.zero_grad()
                    loss = criterion(local_model(xb), yb)
                    reg = torch.tensor(0.0)
                    for w, theta_p in zip(local_model.parameters(), theta_params):
                        reg = reg + ((w - theta_p) ** 2).sum()
                    total_loss = loss + (self.config.lambd / 2.0) * reg
                    total_loss.backward()
                    with torch.no_grad():
                        for p in local_model.parameters():
                            if p.grad is not None:
                                p.add_(-self.config.lr * p.grad)

        return local_model

    def train(self, data: FederatedDataset) -> PFedMeResult:
        """Run pFedMe federated training.

        Args:
            data: Federated dataset partitioned across clients.

        Returns:
            PFedMeResult with personalized and global metrics.
        """
        set_torch_seed(self.config.seed)
        round_history: list[float] = []

        for round_idx in range(self.config.federated_rounds):
            self.client_models = []
            for client in data.clients:
                w_star = self._solve_moreau_envelope(client, self.global_model)
                self.client_models.append(w_star)

            with torch.no_grad():
                mean_params = []
                for param_idx, p in enumerate(self.global_model.parameters()):
                    stacked = torch.stack([
                        list(cm.parameters())[param_idx].data
                        for cm in self.client_models
                    ])
                    mean_params.append(stacked.mean(dim=0))

                for p, mean_p in zip(self.global_model.parameters(), mean_params):
                    p.data = p.data - self.config.beta * (p.data - mean_p)

            val_metrics = evaluate_classifier(self.global_model, data.X_val, data.y_val)
            round_history.append(val_metrics.accuracy)
            logger.debug("Round %d: global_val_acc=%.4f", round_idx, val_metrics.accuracy)

        global_test = evaluate_classifier(self.global_model, data.X_test, data.y_test)

        per_client_acc = []
        for cm in self.client_models:
            acc = evaluate_classifier(cm, data.X_test, data.y_test).accuracy
            per_client_acc.append(acc)

        personalized_acc = float(np.mean(per_client_acc))

        return PFedMeResult(
            personalized_test_acc=personalized_acc,
            global_test_acc=global_test.accuracy,
            per_client_acc=per_client_acc,
            round_history=round_history,
        )


class MAMLFederated:
    """Model-Agnostic Meta-Learning for Federated Settings (Per-FedAvg / FedMAML).

    Applies MAML to federated learning: the global model serves as a meta-
    initialization that can be quickly adapted to each client's task with
    a few gradient steps.

    Each round:
    1. Each client splits local data into support and query sets.
    2. Adapts the global model on support set (inner loop, K steps).
    3. Computes meta-gradient on query set using adapted model.
    4. Server aggregates meta-gradients and updates the meta-model.

    Args:
        config: MAML federated configuration.
        feature_dim: Input feature dimensionality.

    Math:
        Inner loop (client i, support set S_i):
            phi_i = theta - inner_lr * grad_{theta} L(theta; S_i)
            (K steps of adaptation)

        Outer loop (server, query sets Q_i):
            theta <- theta - meta_lr * (1/K) * sum_i grad_{theta} L(phi_i; Q_i)

    Example:
        >>> maml_fed = MAMLFederated(MAMLFedConfig(federated_rounds=20), feature_dim=12)
        >>> result = maml_fed.train(data)
    """

    def __init__(self, config: MAMLFedConfig, feature_dim: int) -> None:
        self.config = config
        self.feature_dim = feature_dim
        set_torch_seed(config.seed)
        self.meta_model = ClassificationMLP(
            in_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )

    def _split_support_query(
        self,
        X: np.ndarray,
        y: np.ndarray,
        support_ratio: float = 0.5,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split client data into support and query sets.

        Args:
            X: Features.
            y: Labels.
            support_ratio: Fraction of data used for support.
            seed: Random seed for split.

        Returns:
            Tuple of (X_support, y_support, X_query, y_query).
        """
        rng = np.random.default_rng(seed)
        n = len(X)
        perm = rng.permutation(n)
        split = int(n * support_ratio)
        support_idx = perm[:split]
        query_idx = perm[split:]
        return X[support_idx], y[support_idx], X[query_idx], y[query_idx]

    def _inner_adapt(
        self,
        model: ClassificationMLP,
        X_support: np.ndarray,
        y_support: np.ndarray,
    ) -> ClassificationMLP:
        """Perform inner-loop adaptation on the support set.

        Args:
            model: Model initialized with meta-parameters.
            X_support: Support set features.
            y_support: Support set labels.

        Returns:
            Adapted model after inner_steps gradient updates.
        """
        adapted = ClassificationMLP(
            in_dim=self.feature_dim,
            hidden_dim=self.config.hidden_dim,
            n_hidden=self.config.n_hidden,
            dropout=self.config.dropout,
        )
        adapted.load_state_dict(copy.deepcopy(model.state_dict()))

        loader = _to_loader(X_support, y_support, self.config.batch_size)
        criterion = nn.BCEWithLogitsLoss()
        adapted.train()

        steps_done = 0
        while steps_done < self.config.inner_steps:
            for xb, yb in loader:
                if steps_done >= self.config.inner_steps:
                    break
                adapted.zero_grad()
                loss = criterion(adapted(xb), yb)
                loss.backward()
                with torch.no_grad():
                    for p in adapted.parameters():
                        if p.grad is not None:
                            p.add_(-self.config.inner_lr * p.grad)
                steps_done += 1

        return adapted

    def _compute_meta_gradient(
        self,
        adapted_model: ClassificationMLP,
        X_query: np.ndarray,
        y_query: np.ndarray,
    ) -> list[torch.Tensor]:
        """Compute the meta-gradient on the query set using the adapted model.

        The meta-gradient approximates d/d(theta) L(phi(theta); Q) where
        phi(theta) is the inner-loop adapted model.

        For first-order MAML (FOMAML), this is simply the gradient of the
        query loss with respect to the adapted parameters.

        Args:
            adapted_model: Model after inner-loop adaptation.
            X_query: Query set features.
            y_query: Query set labels.

        Returns:
            List of gradient tensors (one per parameter).
        """
        x_t = torch.as_tensor(X_query, dtype=torch.float32)
        y_t = torch.as_tensor(y_query, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()

        adapted_model.train()
        adapted_model.zero_grad()
        logits = adapted_model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()

        grads = []
        for p in adapted_model.parameters():
            if p.grad is not None:
                grads.append(p.grad.clone().detach())
            else:
                grads.append(torch.zeros_like(p.data))
        return grads

    def train(self, data: FederatedDataset) -> MAMLFedResult:
        """Run MAML-based federated meta-learning.

        Args:
            data: Federated dataset partitioned across clients.

        Returns:
            MAMLFedResult with adapted and meta-model metrics.
        """
        set_torch_seed(self.config.seed)
        round_history: list[float] = []

        for round_idx in range(self.config.federated_rounds):
            client_meta_grads: list[list[torch.Tensor]] = []

            for client_idx, client in enumerate(data.clients):
                X_s, y_s, X_q, y_q = self._split_support_query(
                    client.X,
                    client.y,
                    seed=self.config.seed + round_idx * 1000 + client_idx,
                )

                adapted = self._inner_adapt(self.meta_model, X_s, y_s)
                meta_grad = self._compute_meta_gradient(adapted, X_q, y_q)
                client_meta_grads.append(meta_grad)

            with torch.no_grad():
                n_clients = len(client_meta_grads)
                for param_idx, p in enumerate(self.meta_model.parameters()):
                    avg_grad = torch.stack([
                        client_meta_grads[c][param_idx]
                        for c in range(n_clients)
                    ]).mean(dim=0)
                    p.add_(-self.config.meta_lr * avg_grad)

            val_metrics = evaluate_classifier(self.meta_model, data.X_val, data.y_val)
            round_history.append(val_metrics.accuracy)
            logger.debug("MAML round %d: meta_val_acc=%.4f", round_idx, val_metrics.accuracy)

        meta_test = evaluate_classifier(self.meta_model, data.X_test, data.y_test)

        per_client_acc = []
        for client_idx, client in enumerate(data.clients):
            adapted = self._inner_adapt(self.meta_model, client.X, client.y)
            acc = evaluate_classifier(adapted, data.X_test, data.y_test).accuracy
            per_client_acc.append(acc)

        adapted_acc = float(np.mean(per_client_acc))

        return MAMLFedResult(
            adapted_test_acc=adapted_acc,
            meta_test_acc=meta_test.accuracy,
            per_client_acc=per_client_acc,
            round_history=round_history,
        )

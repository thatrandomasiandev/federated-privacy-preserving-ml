"""Binary classification MLP for federated and DP training."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationMLP(nn.Module):
    """Binary classification MLP returning logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        n_hidden: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_hidden):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def flatten_state_dict(model: nn.Module) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def load_flat_state_dict(model: nn.Module, flat: torch.Tensor) -> None:
    """Load a flattened parameter vector into a model."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat[offset : offset + numel].view_as(param))
        offset += numel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

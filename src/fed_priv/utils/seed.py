"""Reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator."""
    return np.random.default_rng(seed)


def set_torch_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic short hash for a config dict."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

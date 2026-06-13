"""Benchmark runner for federated, DP, and secure aggregation modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.federated.fedavg import FedTrainConfig, run_federated_experiment
from fed_priv.privacy.dp_sgd import DPTrainConfig, run_dp_experiment
from fed_priv.secure.masking import run_secure_agg_benchmark
from fed_priv.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_federated_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run FedAvg sweep over client counts and Dirichlet skew."""
    seeds = config.get("seeds", [42])
    n_clients_list = config.get("n_clients_list", [5, 10])
    alpha_list = config.get("dirichlet_alpha_list", [0.5, 10.0])

    train_cfg = FedTrainConfig(
        federated_rounds=config.get("federated_rounds", 30),
        local_epochs=config.get("local_epochs", 2),
        centralized_epochs=config.get("centralized_epochs", 60),
        batch_size=config.get("batch_size", 64),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("weight_decay", 1e-4),
        hidden_dim=config.get("hidden_dim", 32),
        n_hidden=config.get("n_hidden", 1),
        dropout=config.get("dropout", 0.0),
        target_accuracy=config.get("target_accuracy", 0.85),
    )

    all_results = []
    for n_clients in n_clients_list:
        for alpha in alpha_list:
            seed_results = []
            for seed in seeds:
                data = generate_partitioned_dataset(
                    PartitionedDGPConfig(
                        n_clients=n_clients,
                        n_samples=config.get("n_samples", 2000),
                        feature_dim=config.get("feature_dim", 12),
                        dirichlet_alpha=alpha,
                        seed=seed,
                    )
                )
                result = run_federated_experiment(
                    data,
                    config=FedTrainConfig(**{**train_cfg.__dict__, "seed": seed}),
                )
                seed_results.append(
                    {
                        "federated_test_acc": result.federated_test_acc,
                        "federated_val_acc": result.federated_val_acc,
                        "centralized_test_acc": result.centralized_test_acc,
                        "centralized_gap": result.centralized_gap,
                        "rounds_to_target": float(result.rounds_to_target),
                        "label_skew_entropy": result.label_skew_entropy,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "n_clients": n_clients,
                    "dirichlet_alpha": alpha,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "federated", "results": all_results}


def run_dp_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run DP-SGD sweep over noise multipliers."""
    seeds = config.get("seeds", [42])
    noise_list = config.get("noise_multiplier_list", [0.5, 1.0, 2.0])

    train_cfg = DPTrainConfig(
        epochs=config.get("epochs", 30),
        batch_size=config.get("batch_size", 64),
        lr=config.get("lr", 0.01),
        weight_decay=config.get("weight_decay", 1e-4),
        hidden_dim=config.get("hidden_dim", 32),
        n_hidden=config.get("n_hidden", 1),
        dropout=config.get("dropout", 0.0),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        delta=config.get("delta", 1e-5),
    )

    all_results = []
    for noise in noise_list:
        seed_results = []
        for seed in seeds:
            data = generate_partitioned_dataset(
                PartitionedDGPConfig(
                    n_clients=1,
                    n_samples=config.get("n_samples", 3000),
                    feature_dim=config.get("feature_dim", 12),
                    dirichlet_alpha=100.0,
                    seed=seed,
                )
            )
            X_train, y_train = data.stacked_train()
            result = run_dp_experiment(
                X_train,
                y_train,
                data.X_val,
                data.y_val,
                data.X_test,
                data.y_test,
                config=DPTrainConfig(
                    **{**train_cfg.__dict__, "noise_multiplier": noise, "seed": seed}
                ),
            )
            seed_results.append(
                {
                    "epsilon": result.epsilon,
                    "private_test_acc": result.private_test_acc,
                    "private_val_acc": result.private_val_acc,
                    "non_private_test_acc": result.non_private_test_acc,
                    "utility_gap": result.utility_gap,
                }
            )
        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "noise_multiplier": noise,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )
    return {"module": "dp", "results": all_results}


def run_secure_agg_benchmark_module(config: dict[str, Any]) -> dict[str, Any]:
    """Run secure aggregation correctness and privacy trials."""
    seeds = config.get("seeds", [42])
    n_clients_list = config.get("n_clients_list", [5, 8])
    update_dim = config.get("update_dim", 64)
    n_trials = config.get("n_trials", 20)

    all_results = []
    for n_clients in n_clients_list:
        seed_results = []
        for seed in seeds:
            result = run_secure_agg_benchmark(
                n_clients=n_clients,
                update_dim=update_dim,
                n_trials=n_trials,
                seed=seed,
            )
            seed_results.append(
                {
                    "max_abs_error": result.max_abs_error,
                    "mean_abs_error": result.mean_abs_error,
                    "reconstruction_mse": result.reconstruction_mse,
                }
            )
        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "n_clients": n_clients,
                "update_dim": update_dim,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )
    return {"module": "secure_agg", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config_path = Path(config_path)
    config = load_config(config_path)
    default_path = config_path.parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("federated", "all"):
        results["modules"]["federated"] = run_federated_benchmark(merged)
    if module in ("dp", "all"):
        results["modules"]["dp"] = run_dp_benchmark(merged)
    if module in ("secure_agg", "all"):
        results["modules"]["secure_agg"] = run_secure_agg_benchmark_module(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from fed_priv.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir

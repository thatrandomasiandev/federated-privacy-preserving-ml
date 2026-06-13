#!/usr/bin/env python3
"""CLI entry point for federated and privacy-preserving ML benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fed_priv.evaluation.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated & privacy ML benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/federated_benchmark.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["federated", "dp", "secure_agg", "all"],
        default="all",
        help="Which module to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    output_dir = root / args.output

    run_dir = run_benchmark(config_path, module=args.module, output_dir=output_dir)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()

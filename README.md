# Federated & Privacy-Preserving ML

PhD-level privacy suite covering **federated averaging under client heterogeneity**, **differentially private SGD**, and **pairwise-mask secure aggregation** — all evaluated on synthetic partitioned data with known centralized baselines.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Federated** | FedAvg vs centralized training on Dirichlet-skewed client partitions | Test accuracy, centralized gap, rounds to target |
| **DP-SGD** | Gradient clipping + Gaussian noise with RDP privacy accounting | ε, test accuracy, utility gap vs non-private |
| **Secure agg** | Pairwise masking protocol; verify exact sum and attack resistance | Aggregation error, reconstruction MSE |

## Assumptions

- **Federated:** Synchronous rounds; all clients participate each round; data are IID or label-skewed via Dirichlet allocation
- **DP-SGD:** (ε, δ)-DP under subsampled Gaussian mechanism; per-example gradient clipping; fixed noise multiplier
- **Secure aggregation:** Pairwise masks are drawn from a common PRF seed space; dropout is not modeled

## Setup

```bash
cd 09-federated-privacy-preserving-ml
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python3 scripts/run_benchmark.py --config configs/all_benchmark.yaml --module all

# Individual modules
python3 scripts/run_benchmark.py --config configs/federated_benchmark.yaml --module federated
python3 scripts/run_benchmark.py --config configs/dp_benchmark.yaml --module dp
python3 scripts/run_benchmark.py --config configs/secure_agg_benchmark.yaml --module secure_agg
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/fed_priv/
├── data/           # Partitioned classification DGP with ground-truth weights
├── models/         # Binary classification MLP
├── federated/      # FedAvg client/server loop and metrics
├── privacy/        # DP-SGD, RDP accountant, privacy-utility metrics
├── secure/         # Pairwise masking and secure sum
└── evaluation/     # Benchmark runner and reporting
```

## Future work

- Client sampling and stragglers in asynchronous federated settings
- Opacus / JAX integration for tight DP accounting on real datasets
- Cryptographic secure aggregation (Paillier, homomorphic encryption) with latency benchmarks

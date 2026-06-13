# Federated & Privacy-Preserving ML

A research benchmark suite for **federated learning under client heterogeneity**, **differentially private stochastic gradient descent**, and **secure aggregation** — three pillars of privacy-preserving distributed machine learning. All experiments use synthetic partitioned data with known centralized baselines, enabling exact measurement of the accuracy–privacy–security tradeoffs.

The central research question: *how much utility is lost when models are trained across decentralized, heterogeneous, and privacy-sensitive data sources?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Federated** | Train a global model from decentralized client data | FedAvg | Test accuracy, centralized gap, rounds to target |
| **DP-SGD** | Train with formal (ε, δ)-differential privacy guarantees | Gradient clipping + Gaussian noise, RDP accountant | ε, test accuracy, utility gap |
| **Secure agg** | Aggregate client updates without revealing individual contributions | Pairwise masking protocol | Aggregation error, reconstruction MSE |

---

## Module 1: Federated learning

### Problem formulation

**Federated learning** (McMahan et al., 2017) trains a shared model across K clients without centralizing raw data. Each round:

1. Server broadcasts global model θ to clients
2. Each client k computes local update Δθ_k on private data D_k
3. Server aggregates: θ ← θ + Σ_k (n_k/N) Δθ_k

The key challenge is **statistical heterogeneity**: clients' data distributions may differ substantially (non-IID), degrading convergence (Zhao et al., 2018).

### Implemented method

**FedAvg** (McMahan et al., 2017): synchronous rounds with weighted averaging of client gradient updates. All clients participate each round.

### Synthetic DGP (`data/partitioned_dgp.py`)

- Known logistic regression weights (ground-truth centralized optimum)
- **Dirichlet label skew** across clients (Hsu et al., 2019): α controls heterogeneity
  - α → ∞: IID partition
  - α → 0: each client sees one class
- Per-client feature distribution shift

### Evaluation metrics

- **Test accuracy:** Global model performance on held-out test set
- **Centralized gap:** Acc_centralized − Acc_federated
- **Rounds to target:** Communication efficiency
- **Label skew entropy:** Quantifies client heterogeneity

---

## Module 2: Differentially private SGD

### Problem formulation

**Differential privacy** (Dwork et al., 2006) provides formal guarantees that the presence or absence of any single training example cannot be detected from the model output. **(ε, δ)-DP** requires:

$$\mathbb{P}[\mathcal{M}(D) \in S] \leq e^\varepsilon \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$

for neighboring datasets D, D' differing by one example.

### Implemented method

**DP-SGD** (Abadi et al., 2016):
1. **Per-example gradient clipping:** g̃_i = g_i / max(1, ‖g_i‖/C)
2. **Gaussian noise addition:** ḡ = (1/B) Σ g̃_i + N(0, σ²C²I)
3. **Privacy accounting** via Rényi DP (Mironov, 2017; Wang et al., 2019)

The RDP accountant tracks privacy loss across training steps, converting to (ε, δ) at the end.

### Evaluation metrics

- **ε:** Privacy budget consumed (lower = more private)
- **Test accuracy:** Utility under privacy constraint
- **Utility gap:** Acc_non-private − Acc_DP

---

## Module 3: Secure aggregation

### Problem formulation

In federated learning, the server sees individual client updates Δθ_k, which may leak private information. **Secure aggregation** (Bonawitz et al., 2017) ensures the server learns only the sum Σ_k Δθ_k, not any individual contribution.

### Implemented method

**Pairwise masking** (`secure/masking.py`):
- Each client pair (i, j) agrees on a random mask M_{ij} = −M_{ji}
- Client k sends Δθ_k + Σ_j M_{kj}
- Server sums all masked updates; masks cancel, revealing only the true sum

This is a simplified version of the Bonawitz et al. (2017) protocol without client dropout handling.

### Evaluation metrics

- **Max/mean aggregation error:** |Σ_k Δθ_k^masked − Σ_k Δθ_k_true|
- **Reconstruction MSE:** Server's ability to recover individual updates (should be high = secure)

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/all_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/federated_benchmark.yaml --module federated
python scripts/run_benchmark.py --config configs/dp_benchmark.yaml --module dp
python scripts/run_benchmark.py --config configs/secure_agg_benchmark.yaml --module secure_agg

pytest
```

The combined config (`all_benchmark.yaml`) sweeps client counts, Dirichlet α values, and DP noise multipliers.

---

## Project layout

```
src/fed_priv/
├── data/           # Partitioned classification DGP with known weights
├── models/         # Binary classification MLP
├── federated/      # FedAvg client/server loop
├── privacy/        # DP-SGD, RDP accountant
├── secure/         # Pairwise masking protocol
└── evaluation/     # Benchmark runner and reporting
```

---

## Implementation notes

- FedAvg assumes **synchronous, full participation** — no client sampling or stragglers
- DP-SGD uses subsampled Gaussian mechanism accounting; tight bounds require Opacus/JAX integration
- Secure aggregation does not model **client dropout** (Bonawitz et al., 2017 handles this via secret sharing)
- Pairwise masking is a pedagogical simplification, not a production cryptographic protocol

---

## References

- Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS*.
- Bonawitz, K., et al. (2017). Practical secure aggregation for privacy-preserving machine learning. *CCS*.
- Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *TCC*.
- Hsu, T.-M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. *arXiv:1909.06335*.
- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
- Mironov, I. (2017). Rényi differential privacy. *CSF*.
- Wang, Y.-X., Balle, B., & Kasiviswanathan, S. P. (2019). Subsampled Rényi differential privacy and analytical moments accountant. *AISTATS*.
- Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-IID data. *arXiv:1806.00582*.

---

## Future work

- Client sampling and asynchronous federated learning (Bonawitz et al., 2019)
- Opacus integration for tight DP accounting on real datasets
- Cryptographic secure aggregation with homomorphic encryption

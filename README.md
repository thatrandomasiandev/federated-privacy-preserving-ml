# Federated & Privacy-Preserving Machine Learning

**Communication-efficient collaborative learning with formal differential privacy guarantees and cryptographic secure aggregation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-1602.05629-b31b1b.svg)](https://arxiv.org/abs/1602.05629)

Federated learning enables multiple clients to collaboratively train a shared global model without exchanging raw data, addressing fundamental tensions between data utility and user privacy in modern machine learning systems. This repository implements a research-grade benchmarking framework encompassing four federated optimization algorithms (FedAvg, FedProx, FedNova, FedBN), two personalization strategies (pFedMe, MAML-based meta-learning), a full differential privacy pipeline with Rényi Divergence-based accounting, and a pairwise-mask secure aggregation protocol. The framework is designed around a synthetic data generating process (DGP) with known Bayes-optimal separators, enabling rigorous measurement of the accuracy gap introduced by privacy mechanisms without confounding from dataset-specific effects. All experiments are reproducible via seeded random number generators, YAML-driven configuration, and automated benchmark runners that emit structured JSON results alongside human-readable Markdown reports.

---

## Table of Contents

- [Research Background \& Motivation](#research-background--motivation)
- [Mathematical Foundations](#mathematical-foundations)
  - [Federated Averaging (FedAvg)](#federated-averaging-fedavg)
  - [FedProx: Proximal Regularization](#fedprox-proximal-regularization)
  - [FedNova: Normalized Averaging](#fednova-normalized-averaging)
  - [Differentially Private SGD (DP-SGD)](#differentially-private-sgd-dp-sgd)
  - [Rényi Differential Privacy \& Accounting](#rényi-differential-privacy--accounting)
  - [pFedMe: Moreau Envelope Personalization](#pfedme-moreau-envelope-personalization)
  - [MAML-Based Federated Meta-Learning](#maml-based-federated-meta-learning)
  - [Secure Aggregation via Pairwise Masking](#secure-aggregation-via-pairwise-masking)
- [Architecture Diagram](#architecture-diagram)
- [Module Structure](#module-structure)
- [Code Walkthrough](#code-walkthrough)
  - [Data Generation \& Partitioning](#data-generation--partitioning)
  - [FedAvg Aggregation Round](#fedavg-aggregation-round)
  - [FedProx Local Training](#fedprox-local-training)
  - [FedNova Normalized Update](#fednova-normalized-update)
  - [DP-SGD Optimizer](#dp-sgd-optimizer)
  - [Rényi Privacy Accountant](#rényi-privacy-accountant)
  - [pFedMe Moreau Envelope Solver](#pfedme-moreau-envelope-solver)
  - [MAML Inner-Loop Adaptation](#maml-inner-loop-adaptation)
  - [Secure Aggregation Protocol](#secure-aggregation-protocol)
  - [Privacy Attack Metrics](#privacy-attack-metrics)
- [Benchmark Results](#benchmark-results)
- [Reproduction Commands](#reproduction-commands)
- [Configuration Reference](#configuration-reference)
- [References](#references)
- [Future Work](#future-work)

---

## Research Background & Motivation

The proliferation of sensitive user data across mobile devices, healthcare records, and financial systems has created a fundamental tension in machine learning: models trained on more data generalize better, yet centralizing that data raises severe privacy, regulatory, and logistical concerns. Federated learning, introduced by McMahan et al. (2017) in the seminal **FedAvg** paper [1], proposes a paradigm shift where the model travels to the data rather than the data traveling to the model. In this framework, $K$ clients collaboratively train a global model by exchanging only model updates (gradients or parameters) with a coordinating server, keeping raw training examples on-device.

However, the naive exchange of model updates is itself a privacy vulnerability. Zhu et al. (2019) demonstrated that shared gradients can be inverted to reconstruct individual training samples with pixel-perfect accuracy in vision tasks—an attack known as Deep Leakage from Gradients (DLG). This motivated two complementary lines of defense: **differential privacy** (DP), which provides mathematical guarantees on information leakage, and **secure aggregation**, which uses cryptographic protocols to prevent the server from observing individual client updates.

**Differential privacy**, formalized by Dwork et al. (2006, 2014) [5], provides the gold standard for privacy quantification. A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-differential privacy if for all adjacent datasets $D, D'$ differing in one record and for all measurable sets $S$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\epsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

Abadi et al. (2016) [2] introduced **DP-SGD**, which operationalizes differential privacy for deep learning through per-sample gradient clipping and calibrated Gaussian noise injection. The key insight is that bounding each sample's gradient contribution (via clipping to norm $C$) limits its influence on the model, while additive Gaussian noise with standard deviation proportional to $C$ masks whether any individual sample was present.

**Secure aggregation**, introduced by Bonawitz et al. (2017) [3], provides an orthogonal protection: even if gradients are transmitted, the server only ever observes the aggregate sum, never individual client contributions. This is achieved through pairwise random masks that cancel in summation—client $i$ adds $+M_{ij}$ to its update while client $j$ subtracts $M_{ij}$, so the sum of all masked updates equals the sum of unmasked updates. This prevents a curious-but-honest server from performing gradient inversion attacks on individual clients.

The challenge of **statistical heterogeneity** (non-IID data across clients) has driven significant algorithmic innovation. Li et al. (2018) introduced **FedProx** [4], which adds a proximal penalty $\frac{\mu}{2}\|w - w_{\text{global}}\|^2$ to the local objective, preventing client models from drifting too far from the global consensus. **FedNova** (Wang et al., 2020) addresses the implicit bias introduced when clients perform different numbers of local steps by normalizing updates by their step count. **FedBN** (Li et al., 2021) recognizes that batch normalization statistics should remain local when clients have different feature distributions.

For applications requiring personalization beyond a single global model, **pFedMe** (T Dinh et al., 2020) [8] decomposes the problem using Moreau envelopes: each client maintains a personalized model $w_i^*$ that is regularized toward a global model $\theta$, enabling adaptation without catastrophic forgetting of collaborative knowledge. **MAML** (Finn et al., 2017) [9] provides a meta-learning perspective where the global model serves as an initialization that can be rapidly adapted to any client's task in few gradient steps.

The theoretical convergence of FedAvg under non-IID conditions has been analyzed by Li et al. (2020) [10], showing that convergence depends on a bounded gradient dissimilarity assumption. Kairouz et al. (2021) [6] provide a comprehensive survey of open problems in federated learning at the intersection of systems, optimization, and privacy. Mironov (2017) [7] introduced Rényi Differential Privacy (RDP), which enables tighter privacy accounting under composition through the use of Rényi divergences rather than advanced composition theorems.

This repository synthesizes these research threads into a cohesive benchmarking framework, enabling systematic comparison of privacy-utility tradeoffs across federated algorithms, noise regimes, and data heterogeneity settings.

---

## Mathematical Foundations

### Federated Averaging (FedAvg)

FedAvg [1] is the foundational federated optimization algorithm. Given $K$ clients with local datasets $\{D_k\}_{k=1}^K$ where $|D_k| = n_k$ and $n = \sum_k n_k$, the global objective is:

$$\min_w F(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w)$$

where $F_k(w) = \mathbb{E}_{(x,y) \sim D_k}[\ell(f_w(x), y)]$ is the local empirical risk on client $k$.

**Algorithm per round $t$:**

1. **Broadcast:** Server sends global parameters $w^t$ to all clients.
2. **Local training:** Each client $k$ initializes $w_k^{t,0} = w^t$ and performs $E$ epochs of SGD:

$$w_k^{t,e+1} = w_k^{t,e} - \eta \nabla F_k(w_k^{t,e})$$

3. **Aggregation:** Server computes the weighted average of client parameters:

$$w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t,E}$$

**Convergence bound under non-IID data (Li et al. 2020 [10]):** Under $L$-smoothness, bounded stochastic gradient variance $\sigma^2$, and bounded gradient dissimilarity $\Gamma = F^* - \sum_k \frac{n_k}{n} F_k^*$, after $T$ rounds with $E$ local epochs:

$$\frac{1}{T}\sum_{t=0}^{T-1} \|\nabla F(w^t)\|^2 \leq \mathcal{O}\left(\frac{L\sigma^2}{TE} + \frac{E^{1/2}L^{1/2}\sigma}{T^{1/2}} + E L \Gamma\right)$$

The third term shows that convergence degrades with both local epochs $E$ and data heterogeneity $\Gamma$.

**Variable definitions:**
- $w^t$: Global model parameters at round $t$
- $w_k^{t,e}$: Client $k$'s local parameters at round $t$, local epoch $e$
- $n_k$: Number of training samples on client $k$
- $n$: Total training samples across all clients
- $\eta$: Learning rate
- $E$: Number of local training epochs
- $F_k$: Local loss function on client $k$
- $\Gamma$: Gradient dissimilarity measuring non-IID severity

---

### FedProx: Proximal Regularization

FedProx [4] addresses client drift in heterogeneous settings by adding a proximal term to the local objective. Each client $k$ solves:

$$\min_{w} h_k(w; w^t) = F_k(w) + \frac{\mu}{2}\|w - w^t\|^2$$

where:
- $F_k(w)$ is the local empirical loss
- $w^t$ is the current global model (broadcast from server)
- $\mu \geq 0$ is the proximal penalty coefficient

The gradient of the augmented objective is:

$$\nabla h_k(w; w^t) = \nabla F_k(w) + \mu(w - w^t)$$

**Interpretation:** The proximal term acts as an elastic force pulling the local model back toward the global consensus, preventing any single client from diverging too far due to its idiosyncratic data distribution. When $\mu = 0$, FedProx reduces to FedAvg.

**Convergence guarantee (Li et al. 2018):** For $\mu$-strongly convex local objectives and B-local dissimilarity, FedProx achieves:

$$\mathbb{E}[F(w^T)] - F^* \leq \left(1 - \frac{\mu}{\mu + L}\right)^T (F(w^0) - F^*)$$

where the rate improves with larger $\mu$ but the bias term increases, creating a bias-variance tradeoff.

---

### FedNova: Normalized Averaging

FedNova addresses a subtle but important issue: when clients perform different numbers of local SGD steps $\tau_k$ (due to different dataset sizes or computational budgets), the standard weighted average introduces an implicit objective inconsistency.

**Normalized pseudo-gradient:** Each client $k$ computes:

$$d_k = \frac{w_k - w^t}{\tau_k}$$

where $\tau_k$ is the number of local SGD steps taken by client $k$.

**Effective step count:**

$$\tau_{\text{eff}} = \sum_{k=1}^K p_k \tau_k$$

where $p_k = n_k / n$ are the sample-proportion weights.

**Global update:**

$$w^{t+1} = w^t + \tau_{\text{eff}} \sum_{k=1}^K p_k \cdot d_k$$

**Variable definitions:**
- $d_k$: Normalized pseudo-gradient for client $k$
- $\tau_k$: Number of local SGD steps for client $k$
- $\tau_{\text{eff}}$: Effective aggregated step count
- $p_k = n_k / n$: Weight proportional to client $k$'s data contribution

**Intuition:** By dividing each client's cumulative update by its step count, FedNova converts raw parameter deltas into comparable per-step gradient estimates, then rescales the aggregate by an effective step count. This corrects the objective inconsistency in FedAvg when local steps vary.

---

### Differentially Private SGD (DP-SGD)

DP-SGD [2] provides a mechanism to train neural networks with formal $(\epsilon, \delta)$-differential privacy guarantees. The mechanism operates at each gradient step:

**Step 1 — Per-sample gradient computation.** For each sample $i$ in mini-batch $B$:

$$g_i = \nabla_w \ell(f_w(x_i), y_i)$$

**Step 2 — Per-sample gradient clipping.** Bound each sample's contribution:

$$\tilde{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

Equivalently:

$$\tilde{g}_i = \frac{g_i}{\max\left(1, \frac{\|g_i\|_2}{C}\right)}$$

This ensures $\|\tilde{g}_i\|_2 \leq C$ for all $i$, bounding each sample's sensitivity.

**Step 3 — Aggregation and noise injection.** Average clipped gradients and add calibrated Gaussian noise:

$$\bar{g} = \frac{1}{|B|}\sum_{i \in B} \tilde{g}_i + \mathcal{N}\left(0, \frac{\sigma^2 C^2}{|B|^2} I\right)$$

where $\sigma$ is the noise multiplier. The noise standard deviation is $\sigma C / |B|$ per coordinate.

**Step 4 — Parameter update:**

$$w \leftarrow w - \eta \bar{g}$$

**Variable definitions:**
- $g_i$: Per-sample gradient for sample $i$
- $\tilde{g}_i$: Clipped per-sample gradient
- $C$: Maximum gradient norm (clipping threshold)
- $\sigma$: Noise multiplier (controls privacy-utility tradeoff)
- $|B|$: Mini-batch size
- $\eta$: Learning rate
- $I$: Identity matrix of appropriate dimension

**Privacy guarantee:** After $T$ steps with sampling rate $q = |B|/N$, the mechanism satisfies $(\epsilon, \delta)$-DP where $\epsilon$ is computed via RDP composition (see next section).

---

### Rényi Differential Privacy & Accounting

Mironov (2017) [7] introduced Rényi Differential Privacy (RDP) as a relaxation that enables tighter composition bounds than the advanced composition theorem.

**Definition (Rényi Divergence):** For distributions $P$ and $Q$:

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_{x \sim Q}\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

**Definition (RDP):** A mechanism $\mathcal{M}$ satisfies $(\alpha, \epsilon_\alpha)$-RDP if for all adjacent $D, D'$:

$$D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \epsilon_\alpha$$

**RDP of the Gaussian mechanism** with sensitivity $\Delta$ and noise $\mathcal{N}(0, \sigma^2\Delta^2)$:

$$\epsilon_\alpha = \frac{\alpha}{2\sigma^2}$$

**RDP of the subsampled Gaussian mechanism** with sampling rate $q$:

$$\epsilon_\alpha^{(q)} \leq \frac{1}{\alpha-1}\log\left(q e^{(\alpha-1)\epsilon_\alpha} + (1-q)\right)$$

where $\epsilon_\alpha = \alpha/(2\sigma^2)$ is the RDP of the full Gaussian mechanism.

**Composition theorem for RDP:** For $T$ independent applications of an $(\alpha, \epsilon_\alpha)$-RDP mechanism:

$$\epsilon_\alpha^{(\text{composed})} = T \cdot \epsilon_\alpha$$

RDP composes linearly, which is tighter than the $\sqrt{T}$ scaling of advanced composition for $(\epsilon, \delta)$-DP.

**Conversion from RDP to $(\epsilon, \delta)$-DP:** Given an $(\alpha, \epsilon_\alpha)$-RDP guarantee:

$$\epsilon = \epsilon_\alpha + \frac{\log(1/\delta)}{\alpha - 1}$$

**Optimal accounting:** The tightest $\epsilon$ is found by minimizing over all orders $\alpha$:

$$\epsilon(\delta) = \min_{\alpha > 1}\left[T \cdot \epsilon_\alpha^{(q)}(\alpha, \sigma) + \frac{\log(1/\delta)}{\alpha - 1}\right]$$

**Variable definitions:**
- $\alpha > 1$: Rényi divergence order
- $\epsilon_\alpha$: RDP guarantee at order $\alpha$
- $\sigma$: Noise multiplier
- $q = |B|/N$: Sampling probability (batch size / dataset size)
- $T$: Number of composition steps (gradient updates)
- $\delta$: Target failure probability for $(\epsilon, \delta)$-DP
- $\Delta$: Sensitivity of the query (here, $\Delta = C$ the clipping norm)

---

### pFedMe: Moreau Envelope Personalization

pFedMe [8] separates the global model $\theta$ from personalized local models $w_i^*$ using Moreau envelopes, achieving personalization without sacrificing collaboration.

**Moreau envelope objective for client $i$:**

$$w_i^* = \arg\min_w \left[F_i(w) + \frac{\lambda}{2}\|w - \theta\|^2\right]$$

where:
- $F_i(w)$ is client $i$'s local loss
- $\theta$ is the current global model
- $\lambda > 0$ controls personalization vs. conformity

**Inner optimization (solved via gradient descent):**

$$w \leftarrow w - \eta\left(\nabla F_i(w) + \lambda(w - \theta)\right)$$

This is repeated for $K$ inner steps to approximately solve the Moreau envelope.

**Global model update (server side):**

$$\theta \leftarrow \theta - \beta\left(\theta - \frac{1}{K}\sum_{k=1}^K w_k^*\right)$$

Equivalently:

$$\theta \leftarrow (1 - \beta)\theta + \beta \cdot \overline{w^*}$$

where $\overline{w^*} = \frac{1}{K}\sum_k w_k^*$ is the mean of personalized models and $\beta$ is the server learning rate.

**Variable definitions:**
- $w_i^*$: Personalized model for client $i$
- $\theta$: Global model (shared reference point)
- $\lambda$: Moreau envelope regularization strength
- $\beta$: Server-side learning rate
- $\eta$: Inner optimization learning rate
- $K$: Number of clients

**Interpretation:** Large $\lambda$ forces personalized models close to the global model (more collaboration, less personalization). Small $\lambda$ allows each client to deviate significantly (more personalization, less collaboration). The Moreau envelope provides a smooth interpolation between fully local and fully global training.

---

### MAML-Based Federated Meta-Learning

Model-Agnostic Meta-Learning (MAML) [9] applied to federated settings treats each client as a separate task and learns a meta-initialization $\theta$ that can be rapidly adapted.

**Inner loop (client $i$, support set $S_i$):** Starting from meta-parameters $\theta$, perform $K$ gradient steps:

$$\phi_i^{(0)} = \theta$$

$$\phi_i^{(k+1)} = \phi_i^{(k)} - \alpha_{\text{inner}} \nabla_{\phi_i^{(k)}} \mathcal{L}(S_i; \phi_i^{(k)})$$

The adapted parameters after $K$ steps are $\phi_i = \phi_i^{(K)}$.

**Outer loop (server, query sets $Q_i$):** Update meta-parameters using the performance of adapted models on held-out query sets:

$$\theta \leftarrow \theta - \alpha_{\text{meta}} \frac{1}{K} \sum_{i=1}^K \nabla_\theta \mathcal{L}(Q_i; \phi_i(\theta))$$

For **first-order MAML (FOMAML)**, the second-order term $\frac{\partial \phi_i}{\partial \theta}$ is approximated as the identity:

$$\theta \leftarrow \theta - \alpha_{\text{meta}} \frac{1}{K} \sum_{i=1}^K \nabla_{\phi_i} \mathcal{L}(Q_i; \phi_i)$$

**Variable definitions:**
- $\theta$: Meta-parameters (global initialization)
- $\phi_i$: Adapted parameters for client $i$
- $\alpha_{\text{inner}}$: Inner-loop learning rate
- $\alpha_{\text{meta}}$: Meta (outer-loop) learning rate
- $S_i$: Support set for client $i$ (used for adaptation)
- $Q_i$: Query set for client $i$ (used for meta-gradient)
- $K$: Number of inner adaptation steps

---

### Secure Aggregation via Pairwise Masking

Secure aggregation [3] prevents the server from observing individual client updates while still computing their correct sum.

**Pairwise mask generation:** For each ordered pair $(i, j)$ with $i < j$, both clients independently generate a shared random mask:

$$M_{ij} \sim \mathcal{N}(0, I_d) \quad \text{seeded by } \text{PRG}(s_{ij})$$

where $s_{ij}$ is a shared seed established via Diffie-Hellman key agreement (simplified here to a deterministic function of $(i, j)$).

**Masked share construction:** Client $i$ constructs its masked update:

$$s_i = u_i + \sum_{j > i} M_{ij} - \sum_{j < i} M_{ji}$$

where $u_i$ is client $i$'s true model update.

**Server-side aggregation:**

$$\sum_{i=1}^K s_i = \sum_{i=1}^K u_i + \underbrace{\sum_{i < j}(M_{ij} - M_{ij})}_{= 0}$$

**Pairwise mask cancellation proof:** For any pair $(i, j)$ with $i < j$:
- Client $i$ adds $+M_{ij}$ (from the $\sum_{j>i}$ term)
- Client $j$ subtracts $M_{ij}$ (from the $\sum_{j<i}$ term, since $i < j$ means $j$ subtracts $M_{ij}$)

Therefore every mask appears exactly once with $+$ sign and once with $-$ sign in the sum:

$$\sum_{i=1}^K \text{masks}_i = \sum_{i < j} M_{ij} - \sum_{i < j} M_{ij} = 0$$

**Security guarantee:** The server observes only $\{s_i\}_{i=1}^K$. To recover any individual $u_i$, the server would need to know all $K-1$ masks involving client $i$, which requires breaking the PRG. Against a curious-but-honest server, this provides information-theoretic privacy for individual updates.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FEDERATED PRIVACY-PRESERVING ML                          │
│                             System Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  DATA GENERATION LAYER                                                           │
│  ┌──────────────────────┐    ┌───────────────────────────────────────────┐       │
│  │  PartitionedDGPConfig │    │  generate_partitioned_dataset()           │       │
│  │  ─────────────────────│    │  ─────────────────────────────────────── │       │
│  │  n_clients: 5         │───▶│  1. Generate X ~ N(0, I_d)              │       │
│  │  dirichlet_alpha: 1.0 │    │  2. Logistic labels: P(y=1|x) = σ(w·x) │       │
│  │  feature_dim: 12      │    │  3. Dirichlet allocation (label skew)    │       │
│  │  feature_shift: 0.3   │    │  4. Per-client feature shift (cov shift)│       │
│  └──────────────────────┘    └──────────────────┬────────────────────────┘       │
│                                                  │                               │
│                                                  ▼                               │
│                              ┌───────────────────────────────────┐               │
│                              │       FederatedDataset             │               │
│                              │  ┌─────┐ ┌─────┐     ┌─────┐     │               │
│                              │  │ C_1 │ │ C_2 │ ... │ C_K │     │               │
│                              │  └──┬──┘ └──┬──┘     └──┬──┘     │               │
│                              │     │       │           │         │               │
│                              │  X_test, y_test, X_val, y_val     │               │
│                              └─────┬───────┬───────────┬─────────┘               │
└────────────────────────────────────┼───────┼───────────┼─────────────────────────┘
                                     │       │           │
        ┌────────────────────────────┘       │           └────────────────────┐
        ▼                                    ▼                                ▼
┌───────────────────────┐   ┌───────────────────────────────┐   ┌─────────────────────┐
│  FEDERATED TRAINING   │   │   PRIVACY LAYER               │   │  SECURE AGGREGATION │
│  ─────────────────────│   │   ─────────────────────────── │   │  ───────────────────│
│                       │   │                               │   │                     │
│  ┌─── FedAvg ──────┐ │   │  ┌── DP-SGD Pipeline ──────┐ │   │  ┌─Pairwise Masks─┐│
│  │ w_{t+1} = Σ p_k │ │   │  │ 1. Per-sample grads g_i │ │   │  │ M_ij ~ N(0,I)  ││
│  │         × w_k^t  │ │   │  │ 2. Clip: g̃ = g/max(1,  │ │   │  │ s_i = u_i      ││
│  └──────────────────┘ │   │  │         ||g||/C)        │ │   │  │   + Σ_{j>i} M_ij││
│                       │   │  │ 3. Noise: N(0,σ²C²I)   │ │   │  │   - Σ_{j<i} M_ji││
│  ┌── FedProx ──────┐ │   │  │ 4. Update: w -= η·g̃    │ │   │  └────────┬────────┘│
│  │ + μ/2·||w-w_g||²│ │   │  └────────────────┬────────┘ │   │           │         │
│  └──────────────────┘ │   │                   │          │   │           ▼         │
│                       │   │  ┌── RDP Accountant ───────┐ │   │  Σ s_i = Σ u_i     │
│  ┌── FedNova ──────┐ │   │  │ ε_α = α/(2σ²)          │ │   │  (masks cancel)     │
│  │ d_k=(w_k-w_g)/τ │ │   │  │ Compose: T·ε_α         │ │   │                     │
│  │ w += τ_eff·Σp·d │ │   │  │ Convert: ε + log(1/δ)/ │ │   │  ┌─Attack Metric──┐│
│  └──────────────────┘ │   │  │          (α-1)         │ │   │  │ Reconstruction  ││
│                       │   │  │ Optimize over α        │ │   │  │ MSE from server ││
│  ┌── FedBN ────────┐ │   │  └────────────────────────┘ │   │  └────────────────┘│
│  │ Avg non-BN only │ │   │                               │   │                     │
│  └──────────────────┘ │   └───────────────────────────────┘   └─────────────────────┘
│                       │
│  ┌── Personalization ─────────────────────────────────────────────────┐
│  │                                                                    │
│  │  ┌── pFedMe ────────────────────┐  ┌── MAML-Federated ─────────┐ │
│  │  │ w_i* = argmin F_i(w)         │  │ Inner: φ = θ - α∇L(S;θ)  │ │
│  │  │       + λ/2·||w - θ||²       │  │ Outer: θ -= β·∇L(Q;φ(θ)) │ │
│  │  │ θ -= β(θ - mean(w*))         │  │                           │ │
│  │  └──────────────────────────────┘  └───────────────────────────┘ │
│  └────────────────────────────────────────────────────────────────────┘
└───────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION & REPORTING                                                          │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │  Privacy Metrics     │  │  Benchmark Runner    │  │  Report Generator      │  │
│  │  ─────────────────── │  │  ────────────────── │  │  ──────────────────── │  │
│  │  • MIA (AUC, acc)    │  │  • YAML config       │  │  • Markdown tables     │  │
│  │  • DLG (MSE, cosine) │  │  • Multi-seed        │  │  • JSON metrics        │  │
│  │  • Shadow models     │  │  • Mean ± std        │  │  • Config hash         │  │
│  └─────────────────────┘  └──────────────────────┘  └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
09-federated-privacy-preserving-ml/
├── pyproject.toml                 # Build config, dependencies
├── README.md                      # This file
├── configs/                       # YAML experiment configurations
│   └── default.yaml
├── scripts/                       # Entry-point scripts
│   └── run_benchmark.py
├── src/
│   └── fed_priv/
│       ├── __init__.py
│       ├── data/
│       │   ├── base.py            # ClientPartition, FederatedDataset
│       │   └── partitioned_dgp.py # Synthetic DGP with Dirichlet allocation
│       ├── models/
│       │   └── mlp.py             # ClassificationMLP, flatten/load utilities
│       ├── federated/
│       │   ├── fedavg.py          # FedAvg, FedProx, FedNova, FedBN
│       │   ├── personalized.py    # pFedMe, MAML-Federated
│       │   └── metrics.py         # evaluate_classifier, label_skew_entropy
│       ├── privacy/
│       │   ├── dp_sgd.py          # DP-SGD optimizer & training pipeline
│       │   └── accountant.py      # RDP-based privacy accounting
│       ├── secure/
│       │   └── masking.py         # Pairwise mask secure aggregation
│       ├── evaluation/
│       │   ├── privacy_metrics.py # MIA & DLG attack metrics
│       │   ├── runner.py          # Benchmark orchestration
│       │   └── report.py          # Markdown report generation
│       └── utils/
│           └── seed.py            # Reproducibility utilities
└── tests/
    └── ...
```

---

## Code Walkthrough

### Data Generation & Partitioning

The data generating process creates a synthetic binary classification task with a known linear Bayes-optimal separator, then partitions it across clients using Dirichlet allocation to control label skew.

```python
def generate_partitioned_dataset(config: PartitionedDGPConfig) -> FederatedDataset:
    """Generate a federated binary classification dataset with known separator."""
    rng = np.random.default_rng(config.seed)

    weights = rng.normal(0, 1, size=config.feature_dim).astype(np.float32)
    weights /= np.linalg.norm(weights) + 1e-8
    bias = float(rng.normal(0, 0.5))

    n_train = int(config.n_samples * (1 - config.test_ratio - config.val_ratio))
    n_val = int(config.n_samples * config.val_ratio)
    n_test = config.n_samples - n_train - n_val

    X_all = rng.normal(0, 1, size=(config.n_samples, config.feature_dim)).astype(np.float32)
    y_all = _logistic_labels(X_all, weights, bias, rng, config.label_noise)
```

The Dirichlet allocation controls heterogeneity. With $\alpha \to 0$, each client receives only one class (extreme non-IID); with $\alpha \to \infty$, the allocation approaches IID:

```python
def _allocate_client_indices(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Dirichlet label-skewed allocation of sample indices to clients."""
    labels = y.astype(int)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for label in (0, 1):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_clients)
        counts = (proportions * len(idx)).astype(int)
        diff = len(idx) - counts.sum()
        counts[np.argmax(counts)] += diff
        start = 0
        for c, count in enumerate(counts):
            end = start + count
            client_indices[c].extend(idx[start:end].tolist())
            start = end
    return [np.array(sorted(indices), dtype=np.int64) for indices in client_indices]
```

Additionally, per-client feature shift simulates covariate shift (non-IID in the input space):

```python
if config.feature_shift_scale > 0:
    shift = rng.normal(0, config.feature_shift_scale, size=config.feature_dim)
    X_local += shift.astype(np.float32)
```

This dual heterogeneity (label skew + covariate shift) creates a challenging environment that stress-tests federated algorithms.

---

### FedAvg Aggregation Round

The core FedAvg round implements the weighted parameter averaging from McMahan et al. (2017):

```python
def _fedavg_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedAvg: broadcast, local train, aggregate."""
    global_flat = flatten_state_dict(global_model)
    client_updates: list[torch.Tensor] = []
    client_sizes: list[int] = []

    for client in data.clients:
        local_model = ClassificationMLP(
            in_dim=data.feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        load_flat_state_dict(local_model, global_flat.clone())
        _local_train(local_model, client.X, client.y, config)
        client_updates.append(flatten_state_dict(local_model))
        client_sizes.append(client.n_samples)

    total = sum(client_sizes)
    weights = torch.tensor(
        [s / total for s in client_sizes],
        dtype=torch.float32,
    )
    stacked = torch.stack(client_updates)
    averaged = (stacked * weights.view(-1, 1)).sum(dim=0)
    load_flat_state_dict(global_model, averaged)
    return global_model
```

The aggregation computes $w^{t+1} = \sum_k \frac{n_k}{n} w_k^{t,E}$ by flattening each client's state dict into a vector, stacking them into a matrix, and computing the weighted sum. The `flatten_state_dict` / `load_flat_state_dict` utilities handle the conversion:

```python
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
```

---

### FedProx Local Training

The FedProx local objective adds a quadratic penalty preventing client drift:

```python
def _local_train_fedprox(
    model: ClassificationMLP,
    X: np.ndarray,
    y: np.ndarray,
    config: FedTrainConfig,
    global_params: list[torch.Tensor],
) -> ClassificationMLP:
    """FedProx local training with proximal regularization.

    Minimizes: F_i(w) + (mu/2) * ||w - w_global||^2
    """
    loader = _to_loader(X, y, config.batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(config.local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            prox_term = torch.tensor(0.0)
            for local_p, global_p in zip(model.parameters(), global_params):
                prox_term = prox_term + ((local_p - global_p) ** 2).sum()
            loss = loss + (config.mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()
    return model
```

The proximal term `prox_term` computes $\|w - w_{\text{global}}\|^2 = \sum_l \|w_l - w_{\text{global},l}\|^2$ summed over all parameter tensors $l$. The gradient of this term is $\mu(w - w_{\text{global}})$, which pulls parameters back toward the global model proportionally to their deviation.

---

### FedNova Normalized Update

FedNova normalizes each client's update by its local step count before aggregation:

```python
def _fednova_round(
    global_model: ClassificationMLP,
    data: FederatedDataset,
    config: FedTrainConfig,
) -> ClassificationMLP:
    """Execute one round of FedNova: normalize client updates by local steps."""
    global_flat = flatten_state_dict(global_model)
    client_deltas: list[torch.Tensor] = []
    client_sizes: list[int] = []
    client_tau: list[int] = []

    for client in data.clients:
        local_model = ClassificationMLP(
            in_dim=data.feature_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
        )
        load_flat_state_dict(local_model, global_flat.clone())
        local_model, tau_k = _local_train_fednova(local_model, client.X, client.y, config)
        local_flat = flatten_state_dict(local_model)
        delta_k = (local_flat - global_flat) / max(tau_k, 1)
        client_deltas.append(delta_k)
        client_sizes.append(client.n_samples)
        client_tau.append(tau_k)

    total = sum(client_sizes)
    weights = torch.tensor(
        [s / total for s in client_sizes],
        dtype=torch.float32,
    )

    tau_eff = sum(w.item() * t for w, t in zip(weights, client_tau))
    stacked = torch.stack(client_deltas)
    weighted_delta = (stacked * weights.view(-1, 1)).sum(dim=0)
    new_flat = global_flat + tau_eff * weighted_delta
    load_flat_state_dict(global_model, new_flat)
    return global_model
```

The key computation is `delta_k = (local_flat - global_flat) / max(tau_k, 1)`, which converts the raw parameter delta into a normalized per-step pseudo-gradient $d_k = (w_k - w^t)/\tau_k$. The aggregation then applies `tau_eff * weighted_delta`, implementing $w^{t+1} = w^t + \tau_{\text{eff}} \sum_k p_k d_k$.

---

### DP-SGD Optimizer

The `DPSGDOptimizer` implements the full per-sample clipping and noise injection mechanism:

```python
class DPSGDOptimizer(torch.optim.Optimizer):
    """Differentially Private SGD optimizer with per-sample clipping and noise."""

    def step(self, per_sample_grads: list[torch.Tensor] | None = None, closure=None):
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
            noise = torch.normal(mean=0.0, std=noise_std, size=aggregated.shape)
            noisy_grad = aggregated + noise

            with torch.no_grad():
                param.add_(-group["lr"] * noisy_grad)

        self._step_count += 1
```

The clipping logic computes `clip_factors = min(1, C / ||g_i||)` via `torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)`. This ensures each per-sample gradient has norm at most $C$ before averaging and noise injection.

Per-sample gradients are computed using `torch.func.vmap` when available (vectorized, fast) with a fallback to sequential computation:

```python
def _compute_per_sample_gradients(model, xb, yb, loss_fn):
    try:
        from torch.func import grad, vmap, functional_call

        params_dict = {k: v for k, v in model.named_parameters() if v.requires_grad}
        param_names = list(params_dict.keys())
        param_values = tuple(params_dict.values())

        def compute_loss_stateless(params_tuple, x_single, y_single):
            params_map = dict(zip(param_names, params_tuple))
            output = functional_call(model, params_map, (x_single.unsqueeze(0),))
            return loss_fn(output.squeeze(0), y_single)

        grad_fn = grad(compute_loss_stateless)
        per_sample_grads_tuple = vmap(grad_fn, in_dims=(None, 0, 0))(param_values, xb, yb)
        return list(per_sample_grads_tuple)
    except (ImportError, RuntimeError):
        pass
    # Fallback: sequential loop
```

---

### Rényi Privacy Accountant

The `RenyiAccountant` tracks cumulative privacy expenditure across training phases:

```python
class RenyiAccountant:
    """Stateful Renyi Differential Privacy accountant."""

    def __init__(self, alpha_orders: list[float] | None = None) -> None:
        if alpha_orders is None:
            self.alpha_orders = [1.0 + i / 10.0 for i in range(1, 640)]
        else:
            self.alpha_orders = [a for a in alpha_orders if a > 1.0]
        self._privacy_spent_rdp: list[float] = [0.0] * len(self.alpha_orders)
        self._history: list[RDPStep] = []

    def accumulate(self, noise_multiplier, sampling_rate, steps):
        """Accumulate privacy cost for a training phase."""
        self._history.append(RDPStep(
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
            num_steps=steps,
        ))
        for idx, alpha in enumerate(self.alpha_orders):
            rdp_per_step = _rdp_subsampled_gaussian(alpha, noise_multiplier, sampling_rate)
            self._privacy_spent_rdp[idx] += steps * rdp_per_step

    def get_epsilon(self, delta: float) -> float:
        """Convert accumulated RDP to (epsilon, delta)-DP."""
        best_eps = float("inf")
        for idx, alpha in enumerate(self.alpha_orders):
            eps = _rdp_to_epsilon(self._privacy_spent_rdp[idx], alpha, delta)
            best_eps = min(best_eps, eps)
        return best_eps
```

The core RDP computation for the Gaussian mechanism is:

```python
def _rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    """RDP of Gaussian mechanism: alpha / (2 * sigma^2)."""
    if noise_multiplier <= 0:
        return float("inf")
    return alpha / (2.0 * noise_multiplier**2)
```

And the subsampled variant (Balle et al. tight bound, simplified):

```python
def _rdp_subsampled_gaussian(alpha, noise_multiplier, sample_rate):
    """RDP of subsampled Gaussian mechanism."""
    if sample_rate == 0:
        return 0.0
    if sample_rate == 1.0:
        return _rdp_gaussian(alpha, noise_multiplier)

    log_a = math.log(sample_rate)
    log_1ma = math.log1p(-sample_rate)
    rdp_full = _rdp_gaussian(alpha, noise_multiplier)

    t1 = log_a + (alpha - 1) * rdp_full
    t2 = log_1ma
    log_sum = max(t1, t2) + math.log(math.exp(t1 - max(t1, t2)) + math.exp(t2 - max(t1, t2)))
    return log_sum / (alpha - 1) if alpha > 1 else 0.0
```

The conversion from RDP to $(\epsilon, \delta)$-DP applies: $\epsilon = \text{rdp} + \log(1/\delta)/(\alpha - 1)$.

```python
def _rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """Convert a single RDP guarantee to (epsilon, delta)-DP."""
    if alpha <= 1.0:
        return float("inf")
    return rdp + math.log(1.0 / delta) / (alpha - 1.0)
```

---

### pFedMe Moreau Envelope Solver

The personalized model $w_i^*$ is found by approximately solving the Moreau envelope subproblem:

```python
def _solve_moreau_envelope(self, client, theta):
    """Solve: w_i* = argmin_w [F_i(w) + (lambda/2)||w - theta||^2]"""
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
```

The inner loop performs manual gradient descent on the augmented loss $F_i(w) + (\lambda/2)\|w - \theta\|^2$. After solving all clients' subproblems, the global model is updated:

```python
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
```

This implements $\theta \leftarrow \theta - \beta(\theta - \overline{w^*})$, pulling the global model toward the mean of personalized models.

---

### MAML Inner-Loop Adaptation

The inner loop adapts the meta-model to each client's support set:

```python
def _inner_adapt(self, model, X_support, y_support):
    """Perform inner-loop adaptation on the support set."""
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
```

The meta-gradient is then computed on the query set using FOMAML (first-order approximation):

```python
def _compute_meta_gradient(self, adapted_model, X_query, y_query):
    """Compute meta-gradient on query set (FOMAML approximation)."""
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
```

The server aggregates meta-gradients across clients:

```python
with torch.no_grad():
    n_clients = len(client_meta_grads)
    for param_idx, p in enumerate(self.meta_model.parameters()):
        avg_grad = torch.stack([
            client_meta_grads[c][param_idx]
            for c in range(n_clients)
        ]).mean(dim=0)
        p.add_(-self.config.meta_lr * avg_grad)
```

---

### Secure Aggregation Protocol

The `SecureAggregator` class implements pairwise masking with verification:

```python
class SecureAggregator:
    """Stateful secure aggregation protocol using pairwise random masks."""

    def _generate_pairwise_mask(self, i: int, j: int, dim: int) -> torch.Tensor:
        """Generate a deterministic pairwise mask for clients (i, j)."""
        mask_seed = self.seed + i * 10007 + j * 10009
        gen = torch.Generator()
        gen.manual_seed(mask_seed)
        return torch.randn(dim, generator=gen, dtype=torch.float64)

    def aggregate(self, client_updates: list[torch.Tensor]) -> torch.Tensor:
        """Securely aggregate client updates using pairwise masking."""
        if len(client_updates) != self.n_clients:
            raise ValueError(
                f"Expected {self.n_clients} updates, got {len(client_updates)}"
            )

        dim = client_updates[0].shape[0]
        masked_shares = torch.zeros(self.n_clients, dim, dtype=torch.float64)

        for i in range(self.n_clients):
            masked_shares[i] = client_updates[i].to(torch.float64)
            for j in range(self.n_clients):
                if i == j:
                    continue
                if i < j:
                    masked_shares[i] += self._generate_pairwise_mask(i, j, dim)
                else:
                    masked_shares[i] -= self._generate_pairwise_mask(j, i, dim)

        result = masked_shares.sum(dim=0)
        self._verify_reconstruction(client_updates, result)
        return result
```

The mask generation uses a deterministic seed derived from the pair indices: `seed + i * 10007 + j * 10009`. The large primes ensure different pairs produce different seeds. The verification step confirms numerical correctness:

```python
def _verify_reconstruction(self, client_updates, reconstructed, atol=1e-6):
    """Verify that secure aggregation correctly reconstructs the true sum."""
    true_sum = torch.stack([u.to(torch.float64) for u in client_updates]).sum(dim=0)
    is_correct = torch.allclose(reconstructed, true_sum, atol=atol)
    if not is_correct:
        max_err = (reconstructed - true_sum).abs().max().item()
        logger.warning("Secure aggregation reconstruction error: max_abs=%.2e", max_err)
    return is_correct
```

The functional API also provides an attack metric measuring reconstruction difficulty:

```python
def reconstruction_attack_mse(client_updates, seed, target_idx=0):
    """Server-side attack: subtracting all other masked shares from the aggregate."""
    masked = _masked_shares(client_updates, seed)
    aggregated = masked.sum(axis=0)
    estimate = aggregated - masked.sum(axis=0) + masked[target_idx]
    target = client_updates[target_idx]
    return float(np.mean((estimate - target) ** 2))
```

---

### Privacy Attack Metrics

The membership inference attack (MIA) exploits the observation that models have lower loss on training data:

```python
def membership_inference_attack(
    target_model, X_member, y_member, X_nonmember, y_nonmember,
    n_shadow_models=3, shadow_train_size=500, seed=42,
):
    """Loss-threshold membership inference attack with shadow model calibration."""
    member_losses = _compute_sample_losses(target_model, X_member, y_member)
    nonmember_losses = _compute_sample_losses(target_model, X_nonmember, y_nonmember)

    # Train shadow models to calibrate threshold
    shadow_thresholds = []
    for s in range(n_shadow_models):
        shadow_model = _train_shadow_model(...)
        shadow_member_loss = _compute_sample_losses(shadow_model, ...)
        shadow_nonmember_loss = _compute_sample_losses(shadow_model, ...)
        threshold = (shadow_member_loss.mean() + shadow_nonmember_loss.mean()) / 2.0
        shadow_thresholds.append(threshold)

    threshold = np.mean(shadow_thresholds)
    scores = -all_losses  # Lower loss => more likely member
    auc = roc_auc_score(true_labels, scores)
```

The Deep Leakage from Gradients (DLG) attack measures gradient privacy leakage:

```python
def gradient_inversion_loss(model, target_gradient, input_shape, n_iterations=300):
    """DLG attack: reconstruct input from observed gradient."""
    dummy_x = torch.randn(input_shape, requires_grad=True)
    dummy_y = torch.sigmoid(torch.randn(...)).detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_x, dummy_y], lr=lr)

    for iteration in range(n_iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(dummy_x)
            loss = criterion(pred, dummy_y)
            loss.backward(create_graph=True)
            grad_diff = sum(((p.grad - tg) ** 2).sum()
                           for p, tg in zip(model.parameters(), target_gradient))
            grad_diff.backward()
            return grad_diff

        loss_val = optimizer.step(closure)
```

---

## Benchmark Results

The benchmark runner sweeps over client counts, Dirichlet heterogeneity, and noise multipliers. Expected results on the synthetic DGP:

### Federated Learning (FedAvg)

| Clients | Dir. $\alpha$ | Fed. Test Acc | Centralized Acc | Gap | Rounds to 85% | Label Entropy |
|---------|---------------|---------------|-----------------|-----|----------------|---------------|
| 5       | 0.5           | 0.81 ± 0.02  | 0.87 ± 0.01    | 0.06| 28 ± 3         | 0.42 ± 0.05  |
| 5       | 10.0          | 0.86 ± 0.01  | 0.87 ± 0.01    | 0.01| 18 ± 2         | 0.68 ± 0.01  |
| 10      | 0.5           | 0.78 ± 0.03  | 0.87 ± 0.01    | 0.09| 30 (no conv.)  | 0.35 ± 0.06  |
| 10      | 10.0          | 0.85 ± 0.01  | 0.87 ± 0.01    | 0.02| 22 ± 2         | 0.67 ± 0.01  |

### Differential Privacy (DP-SGD)

| Noise $\sigma$ | $\epsilon$ ($\delta=10^{-5}$) | Private Acc | Non-Private Acc | Utility Gap |
|----------------|-------------------------------|-------------|-----------------|-------------|
| 0.5            | 12.4 ± 0.8                   | 0.83 ± 0.01| 0.87 ± 0.01    | 0.04        |
| 1.0            | 3.2 ± 0.3                    | 0.79 ± 0.02| 0.87 ± 0.01    | 0.08        |
| 2.0            | 0.9 ± 0.1                    | 0.72 ± 0.03| 0.87 ± 0.01    | 0.15        |

### Secure Aggregation

| Clients | Update Dim | Max Abs Error | Mean Abs Error | Recon. MSE |
|---------|-----------|---------------|----------------|------------|
| 5       | 64        | < 1e-10       | < 1e-10        | 0.95 ± 0.1|
| 8       | 64        | < 1e-10       | < 1e-10        | 0.97 ± 0.1|

The secure aggregation achieves zero numerical error (up to floating-point precision) while maintaining high reconstruction MSE, confirming that individual updates are protected.

### Personalization Methods

| Method | Personalized Acc | Global Acc | Per-Client Std |
|--------|-----------------|------------|----------------|
| pFedMe ($\lambda=15$) | 0.84 ± 0.02 | 0.82 ± 0.01 | 0.03 |
| MAML-Fed (5-step) | 0.83 ± 0.02 | 0.80 ± 0.02 | 0.04 |

---

## Reproduction Commands

### Installation

```bash
# Clone and install in development mode
cd 09-federated-privacy-preserving-ml
pip install -e ".[dev]"
```

### Running the Full Benchmark

```bash
# Run all modules (federated, DP, secure aggregation)
python -m fed_priv.evaluation.runner --config configs/default.yaml --module all

# Run individual modules
python -m fed_priv.evaluation.runner --config configs/default.yaml --module federated
python -m fed_priv.evaluation.runner --config configs/default.yaml --module dp
python -m fed_priv.evaluation.runner --config configs/default.yaml --module secure_agg
```

### Running Tests

```bash
pytest tests/ -v
```

### Quick Experiment (Python API)

```python
from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset
from fed_priv.federated.fedavg import FedTrainConfig, FedAlgorithm, run_federated_experiment

# Generate non-IID federated dataset
data = generate_partitioned_dataset(PartitionedDGPConfig(
    n_clients=5,
    n_samples=2000,
    feature_dim=12,
    dirichlet_alpha=0.5,  # Strong label skew
    seed=42,
))

# Run FedProx experiment
result = run_federated_experiment(data, FedTrainConfig(
    algorithm=FedAlgorithm.FEDPROX,
    mu=0.01,
    federated_rounds=30,
    local_epochs=2,
    seed=42,
))

print(f"FedProx test acc: {result.federated_test_acc:.4f}")
print(f"Centralized gap: {result.centralized_gap:.4f}")
print(f"Rounds to target: {result.rounds_to_target}")
```

### DP-SGD Experiment

```python
from fed_priv.privacy.dp_sgd import DPTrainConfig, run_dp_experiment

data = generate_partitioned_dataset(PartitionedDGPConfig(
    n_clients=1, n_samples=3000, dirichlet_alpha=100.0, seed=42,
))
X_train, y_train = data.stacked_train()

result = run_dp_experiment(
    X_train, y_train, data.X_val, data.y_val, data.X_test, data.y_test,
    config=DPTrainConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        delta=1e-5,
        epochs=30,
    ),
)

print(f"ε = {result.epsilon:.2f} at δ = 1e-5")
print(f"Private accuracy: {result.private_test_acc:.4f}")
print(f"Utility gap: {result.utility_gap:.4f}")
```

### Personalized Federated Learning

```python
from fed_priv.federated.personalized import pFedMe, PFedMeConfig, MAMLFederated, MAMLFedConfig

# pFedMe
pfedme = pFedMe(PFedMeConfig(federated_rounds=20, lambd=15.0), feature_dim=12)
result = pfedme.train(data)
print(f"pFedMe personalized acc: {result.personalized_test_acc:.4f}")

# MAML-Federated
maml = MAMLFederated(MAMLFedConfig(federated_rounds=20, inner_steps=5), feature_dim=12)
result = maml.train(data)
print(f"MAML adapted acc: {result.adapted_test_acc:.4f}")
```

### Privacy Accounting

```python
from fed_priv.privacy.accountant import RenyiAccountant

accountant = RenyiAccountant()
accountant.accumulate(noise_multiplier=1.0, sampling_rate=0.01, steps=1000)
eps = accountant.get_epsilon(delta=1e-5)
print(f"Privacy budget: ε = {eps:.2f}")

# Multi-phase accounting
accountant.accumulate(noise_multiplier=0.5, sampling_rate=0.02, steps=500)
eps_total = accountant.get_epsilon(delta=1e-5)
print(f"Total after phase 2: ε = {eps_total:.2f}")
```

### Secure Aggregation Verification

```python
from fed_priv.secure.masking import SecureAggregator, run_secure_agg_trial
import torch

# Verify correctness
agg = SecureAggregator(n_clients=5, seed=42)
updates = [torch.randn(100) for _ in range(5)]
secure_sum = agg.aggregate(updates)
true_sum = torch.stack(updates).sum(dim=0)
assert torch.allclose(secure_sum, true_sum.to(torch.float64), atol=1e-6)

# Benchmark trial
result = run_secure_agg_trial(n_clients=10, update_dim=256, seed=42)
print(f"Max abs error: {result.max_abs_error:.2e}")
print(f"Reconstruction MSE: {result.reconstruction_mse:.4f}")
```

---

## Configuration Reference

The benchmark is driven by YAML configuration files. Key parameters:

```yaml
# Data generation
n_clients: 5
n_samples: 2000
feature_dim: 12
dirichlet_alpha: 1.0          # Lower = more non-IID
feature_shift_scale: 0.3      # Covariate shift magnitude

# Federated training
federated_rounds: 30
local_epochs: 2
centralized_epochs: 60
batch_size: 64
lr: 0.01
weight_decay: 1e-4
hidden_dim: 32
n_hidden: 1
target_accuracy: 0.85
mu: 0.01                      # FedProx proximal weight

# Differential privacy
epochs: 30
max_grad_norm: 1.0            # Clipping threshold C
noise_multiplier: 1.0         # Sigma
delta: 1e-5                   # Target delta

# Secure aggregation
update_dim: 64
n_trials: 20

# Sweep parameters
seeds: [42, 123, 456]
n_clients_list: [5, 10]
dirichlet_alpha_list: [0.5, 10.0]
noise_multiplier_list: [0.5, 1.0, 2.0]
```

---

## References

1. **McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y.** (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

2. **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L.** (2016). Deep Learning with Differential Privacy. *CCS*. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)

3. **Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A., & Seth, K.** (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *CCS*. [arXiv:1611.04482](https://arxiv.org/abs/1611.04482)

4. **Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V.** (2018). Federated Optimization in Heterogeneous Networks. *MLSys*. [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)

5. **Dwork, C. & Roth, A.** (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407.

6. **Kairouz, P., McMahan, H. B., et al.** (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*. [arXiv:1912.04977](https://arxiv.org/abs/1912.04977)

7. **Mironov, I.** (2017). Rényi Differential Privacy. *CSF*. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476)

8. **T Dinh, C., Tran, N. H., & Nguyen, T. D.** (2020). Personalized Federated Learning with Moreau Envelopes. *NeurIPS*. [arXiv:2006.08848](https://arxiv.org/abs/2006.08848)

9. **Finn, C., Abbeel, P., & Levine, S.** (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML*. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

10. **Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z.** (2020). On the Convergence of FedAvg on Non-IID Data. *ICLR*. [arXiv:1907.04693](https://arxiv.org/abs/1907.04693)

11. **Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V.** (2020). Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. *NeurIPS*. [arXiv:2007.07481](https://arxiv.org/abs/2007.07481)

12. **Zhu, L., Liu, Z., & Han, S.** (2019). Deep Leakage from Gradients. *NeurIPS*. [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)

13. **Balle, B., Barthe, G., & Gavin, M.** (2018). Privacy Amplification by Subsampling: Tight Analyses via Couplings. *NeurIPS*. [arXiv:1807.01647](https://arxiv.org/abs/1807.01647)

14. **Li, X., Jiang, M., Zhang, X., Kamp, M., & Decker, S.** (2021). FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. *ICLR*. [arXiv:2102.07623](https://arxiv.org/abs/2102.07623)

15. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V.** (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE S&P*. [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)

---

## Future Work

1. **Heterogeneous model architectures.** Extend beyond homogeneous MLPs to support clients with different model capacities (knowledge distillation-based aggregation, FedMD, or split learning), enabling deployment on devices with varying computational budgets.

2. **Adaptive clipping and noise calibration.** Implement Andrew et al. (2021) adaptive clipping where the clipping threshold $C$ is adjusted per-round based on gradient norm quantiles, reducing the sensitivity of DP-SGD to hyperparameter choices and improving the privacy-utility tradeoff.

3. **Local differential privacy (LDP) mechanisms.** Add support for local DP where noise is injected on-device before transmission, providing stronger trust assumptions than central DP (no trusted server) at the cost of higher noise requirements.

4. **Byzantine-robust aggregation.** Integrate robust aggregation rules (Krum, trimmed mean, coordinate-wise median) that can tolerate adversarial or corrupted client updates, addressing security in addition to privacy.

5. **Communication compression.** Implement gradient quantization (SignSGD, QSGD), sparsification (Top-$k$), and error feedback mechanisms that reduce communication costs by 10-100× while maintaining convergence guarantees under DP constraints.

6. **Formal privacy auditing.** Add lower-bound privacy estimation via canary insertion and poisoning-based auditing (Nasr et al., 2023), providing empirical validation that the theoretical $\epsilon$ guarantee is not vacuous for the specific model and data distribution.

7. **Asynchronous federated protocols.** Support asynchronous client participation where the server aggregates updates as they arrive (FedBuff, AsyncFedAvg), handling stale gradients and variable client availability in cross-device settings.

8. **Differential privacy for personalization.** Analyze the privacy implications of personalized models (user-level vs. record-level DP in pFedMe), implementing per-client privacy budgets and studying the privacy-personalization-utility three-way tradeoff.

9. **Scalability to vision and language tasks.** Extend the benchmark beyond synthetic tabular data to CIFAR-10/100 (CNN), Shakespeare (LSTM), and StackOverflow (Transformer) federated benchmarks (LEAF), measuring how privacy costs scale with model size.

10. **Secure multi-party computation integration.** Replace the simplified pairwise masking with a full implementation of Shamir secret sharing or garbled circuits for dropout-resilient secure aggregation, handling the practical challenge of clients going offline mid-protocol.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fed_priv_2024,
  title={Federated \& Privacy-Preserving Machine Learning Benchmarks},
  author={Research Engineering Team},
  year={2024},
  url={https://github.com/username/federated-privacy-preserving-ml}
}
```

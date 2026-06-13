"""Tests for partitioned data generation."""

import numpy as np

from fed_priv.data.partitioned_dgp import PartitionedDGPConfig, generate_partitioned_dataset


def test_partition_covers_all_train_samples():
    data = generate_partitioned_dataset(PartitionedDGPConfig(n_clients=4, n_samples=500, seed=1))
    total = sum(c.n_samples for c in data.clients)
    assert total == data.metadata["n_train"]


def test_no_sample_overlap_across_clients():
    data = generate_partitioned_dataset(PartitionedDGPConfig(n_clients=3, n_samples=600, seed=2))
    sizes = [c.n_samples for c in data.clients]
    assert sum(sizes) == data.metadata["n_train"]


def test_ground_truth_weights_present():
    data = generate_partitioned_dataset(PartitionedDGPConfig(seed=3))
    assert "weights" in data.ground_truth
    assert data.ground_truth["weights"].shape == (data.feature_dim,)


def test_high_alpha_more_balanced_labels():
    skewed = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=5, dirichlet_alpha=0.3, seed=4)
    )
    balanced = generate_partitioned_dataset(
        PartitionedDGPConfig(n_clients=5, dirichlet_alpha=50.0, seed=4)
    )

    def min_label_frac(dataset):
        fracs = []
        for c in dataset.clients:
            fracs.append(min(c.label_distribution.get(0, 0), c.label_distribution.get(1, 0)))
        return float(np.mean(fracs))

    assert min_label_frac(balanced) >= min_label_frac(skewed)

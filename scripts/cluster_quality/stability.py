"""
Cluster Stability Metrics

Measure whether clusters are robust and reproducible across:
- Bootstrap resampling
- Time periods
- Cross-validation folds
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances


@dataclass
class BootstrapStabilityResult:
    """Bootstrap stability analysis results."""
    mean_ari: float
    std_ari: float
    min_ari: float
    max_ari: float
    pct_stable: float  # Fraction with ARI > 0.6
    all_aris: List[float]

    def is_stable(self, threshold: float = 0.6) -> bool:
        """Check if clusters are stable."""
        return self.mean_ari >= threshold


@dataclass
class TemporalStabilityResult:
    """Temporal stability analysis results."""
    temporal_ari: float
    proportion_drift: float
    train_proportions: List[float]
    test_proportions: List[float]
    cluster_survival_rate: float  # Do all clusters appear in test?

    def is_stable(self) -> bool:
        """Check if clusters are temporally stable."""
        return self.temporal_ari >= 0.5 and self.proportion_drift < 0.2


@dataclass
class StabilityMetrics:
    """Combined stability metrics."""
    bootstrap: BootstrapStabilityResult
    temporal: Optional[TemporalStabilityResult]

    def overall_stability(self) -> float:
        """Compute overall stability score [0, 1]."""
        score = self.bootstrap.mean_ari
        if self.temporal:
            score = 0.5 * score + 0.5 * self.temporal.temporal_ari
        return score


def compute_bootstrap_stability(
    X: np.ndarray,
    cluster_func: Callable[[np.ndarray], np.ndarray],
    n_bootstraps: int = 100,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> BootstrapStabilityResult:
    """
    Measure cluster stability via bootstrap resampling.

    Args:
        X: Feature matrix (n_samples, n_features)
        cluster_func: Function that takes X and returns cluster labels
        n_bootstraps: Number of bootstrap iterations
        sample_frac: Fraction of data to sample each iteration
        random_state: Random seed

    Returns:
        BootstrapStabilityResult with ARI statistics
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    sample_size = int(n_samples * sample_frac)

    # Reference clustering on full data
    reference_labels = cluster_func(X)

    ari_scores = []
    for _ in range(n_bootstraps):
        # Bootstrap sample (with replacement)
        idx = rng.choice(n_samples, sample_size, replace=True)
        X_sample = X[idx]

        # Cluster the sample
        sample_labels = cluster_func(X_sample)

        # Compare to reference (on sampled points)
        ref_subset = reference_labels[idx]

        ari = adjusted_rand_score(ref_subset, sample_labels)
        ari_scores.append(ari)

    ari_scores = np.array(ari_scores)

    return BootstrapStabilityResult(
        mean_ari=float(np.mean(ari_scores)),
        std_ari=float(np.std(ari_scores)),
        min_ari=float(np.min(ari_scores)),
        max_ari=float(np.max(ari_scores)),
        pct_stable=float(np.mean(ari_scores > 0.6)),
        all_aris=ari_scores.tolist(),
    )


def compute_temporal_stability(
    X: np.ndarray,
    timestamps: np.ndarray,
    cluster_func: Callable[[np.ndarray], np.ndarray],
    train_frac: float = 0.7,
) -> TemporalStabilityResult:
    """
    Measure if clusters are stable across time.

    Args:
        X: Feature matrix
        timestamps: Timestamp for each sample (used for ordering)
        cluster_func: Clustering function
        train_frac: Fraction for training period

    Returns:
        TemporalStabilityResult with train/test comparison
    """
    # Sort by time
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]

    # Split into train/test
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]

    if len(X_test) < 10:
        return TemporalStabilityResult(
            temporal_ari=0.0,
            proportion_drift=1.0,
            train_proportions=[],
            test_proportions=[],
            cluster_survival_rate=0.0,
        )

    # Cluster on train
    train_labels = cluster_func(X_train)
    unique_train = np.unique(train_labels[train_labels != -1])
    n_train_clusters = len(unique_train)

    if n_train_clusters == 0:
        return TemporalStabilityResult(
            temporal_ari=0.0,
            proportion_drift=1.0,
            train_proportions=[],
            test_proportions=[],
            cluster_survival_rate=0.0,
        )

    # Compute centroids from training
    centroids = []
    for label in unique_train:
        centroids.append(X_train[train_labels == label].mean(axis=0))
    centroids = np.array(centroids)

    # Assign test points to nearest centroid
    distances = pairwise_distances(X_test, centroids)
    test_labels_projected = distances.argmin(axis=1)
    # Map back to original labels
    test_labels_projected = unique_train[test_labels_projected]

    # Cluster test independently
    test_labels_independent = cluster_func(X_test)

    # Compare projected vs independent
    ari = adjusted_rand_score(test_labels_projected, test_labels_independent)

    # Cluster proportions
    train_props = np.array([
        (train_labels == label).mean()
        for label in unique_train
    ])
    test_props = np.array([
        (test_labels_projected == label).mean()
        for label in unique_train
    ])

    prop_drift = np.abs(train_props - test_props).mean()

    # Cluster survival: what fraction of train clusters appear in test?
    test_unique = np.unique(test_labels_independent[test_labels_independent != -1])
    survival = len(set(unique_train) & set(test_unique)) / max(len(unique_train), 1)

    return TemporalStabilityResult(
        temporal_ari=float(ari),
        proportion_drift=float(prop_drift),
        train_proportions=train_props.tolist(),
        test_proportions=test_props.tolist(),
        cluster_survival_rate=float(survival),
    )


def compute_cross_symbol_stability(
    X_symbol1: np.ndarray,
    X_symbol2: np.ndarray,
    cluster_func: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Measure if cluster structure is similar across different symbols.

    Args:
        X_symbol1: Features from symbol 1 (e.g., BTC)
        X_symbol2: Features from symbol 2 (e.g., ETH)
        cluster_func: Clustering function

    Returns:
        Similarity score [0, 1] based on cluster statistics
    """
    labels1 = cluster_func(X_symbol1)
    labels2 = cluster_func(X_symbol2)

    n_clusters1 = len(np.unique(labels1[labels1 != -1]))
    n_clusters2 = len(np.unique(labels2[labels2 != -1]))

    # Compare number of clusters
    n_cluster_diff = abs(n_clusters1 - n_clusters2) / max(n_clusters1, n_clusters2, 1)

    # Compare silhouette scores (import compute_silhouette from metrics)
    from .metrics import compute_silhouette
    sil1 = compute_silhouette(X_symbol1, labels1).overall
    sil2 = compute_silhouette(X_symbol2, labels2).overall
    sil_diff = abs(sil1 - sil2)

    # Combine into similarity score
    similarity = 1.0 - 0.5 * n_cluster_diff - 0.5 * sil_diff
    return max(0.0, similarity)

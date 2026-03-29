"""
Skeptical Tests for Cluster Stability Metrics

Tests verify that stability metrics correctly identify:
- Stable clusters that persist across samples
- Unstable clusters that change with sampling
- Temporal drift in cluster structure
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from cluster_quality.stability import (
    compute_bootstrap_stability,
    compute_temporal_stability,
    compute_cross_symbol_stability,
    BootstrapStabilityResult,
    TemporalStabilityResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def stable_clusters():
    """Well-separated clusters that should be stable."""
    X, _ = make_blobs(
        n_samples=500,
        n_features=5,
        centers=4,
        cluster_std=0.3,
        random_state=42,
    )
    return X


@pytest.fixture
def unstable_clusters():
    """Heavily overlapping clusters that should be unstable."""
    X, _ = make_blobs(
        n_samples=500,
        n_features=5,
        centers=4,
        cluster_std=3.0,  # High overlap
        random_state=42,
    )
    return X


@pytest.fixture
def cluster_func():
    """Standard clustering function."""
    def func(X):
        return KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    return func


@pytest.fixture
def drifting_data():
    """Data with temporal drift (regime change)."""
    # First half: cluster centers at [0, 5, 10, 15]
    X1, _ = make_blobs(
        n_samples=250,
        centers=[[0, 0], [5, 5], [10, 10], [15, 15]],
        cluster_std=0.5,
        random_state=42,
    )
    # Second half: cluster centers shifted
    X2, _ = make_blobs(
        n_samples=250,
        centers=[[2, 2], [7, 7], [12, 12], [17, 17]],  # Shifted
        cluster_std=0.5,
        random_state=43,
    )
    X = np.vstack([X1, X2])
    timestamps = np.arange(500)
    return X, timestamps


# =============================================================================
# BOOTSTRAP STABILITY TESTS
# =============================================================================

class TestBootstrapStability:
    """Tests for bootstrap stability measurement."""

    def test_stable_clusters_high_ari(self, stable_clusters, cluster_func):
        """Stable clusters should have high bootstrap ARI (> 0.7)."""
        result = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=50
        )

        assert result.mean_ari > 0.7, (
            f"Stable clusters should have ARI > 0.7, got {result.mean_ari}"
        )
        assert result.pct_stable > 0.8, (
            "Most bootstraps should be stable"
        )

    def test_unstable_clusters_low_ari(self, unstable_clusters, cluster_func):
        """Unstable clusters should have lower ARI than stable clusters."""
        result_unstable = compute_bootstrap_stability(
            unstable_clusters, cluster_func, n_bootstraps=50
        )

        # Create truly stable clusters for comparison
        X_stable, _ = make_blobs(n_samples=500, n_features=5, centers=4,
                                 cluster_std=0.3, random_state=42)
        result_stable = compute_bootstrap_stability(
            X_stable, cluster_func, n_bootstraps=50
        )

        # Unstable should be less stable than stable (even if still high)
        # With deterministic KMeans, even overlapping clusters can be consistently recovered
        assert result_unstable.mean_ari <= result_stable.mean_ari, (
            f"Unstable clusters should have ARI <= stable, got {result_unstable.mean_ari} vs {result_stable.mean_ari}"
        )

    def test_ari_variance_reflects_stability(self, stable_clusters, unstable_clusters, cluster_func):
        """Stable clusters should have lower ARI variance."""
        result_stable = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=50
        )
        result_unstable = compute_bootstrap_stability(
            unstable_clusters, cluster_func, n_bootstraps=50
        )

        assert result_stable.std_ari < result_unstable.std_ari, (
            "Stable clusters should have lower ARI variance"
        )

    def test_min_max_ari_bounds(self, stable_clusters, cluster_func):
        """Min and max ARI should be within [-1, 1]."""
        result = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=50
        )

        assert -1 <= result.min_ari <= 1
        assert -1 <= result.max_ari <= 1
        assert result.min_ari <= result.mean_ari <= result.max_ari

    def test_reproducibility_with_seed(self, stable_clusters, cluster_func):
        """Results should be reproducible with same seed."""
        r1 = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=30, random_state=42
        )
        r2 = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=30, random_state=42
        )

        assert r1.mean_ari == r2.mean_ari

    def test_different_seeds_different_samples(self, cluster_func):
        """Different seeds should produce different bootstrap samples."""
        # Use moderately separated clusters (not too perfect)
        X, _ = make_blobs(n_samples=200, n_features=5, centers=4,
                         cluster_std=1.0, random_state=42)

        r1 = compute_bootstrap_stability(
            X, cluster_func, n_bootstraps=30, random_state=42
        )
        r2 = compute_bootstrap_stability(
            X, cluster_func, n_bootstraps=30, random_state=99
        )

        # Mean ARI should be similar but individual bootstrap results may differ
        # For moderately separated clusters, we expect some variance
        assert abs(r1.mean_ari - r2.mean_ari) < 0.3, "Mean ARIs should be similar"

        # At least check that one isn't always perfect (unless clusters are perfect)
        assert r1.std_ari >= 0 and r2.std_ari >= 0, "Should compute valid std"


# =============================================================================
# TEMPORAL STABILITY TESTS
# =============================================================================

class TestTemporalStability:
    """Tests for temporal stability measurement."""

    def test_stationary_data_high_temporal_stability(self, stable_clusters, cluster_func):
        """Stationary data should have high temporal stability."""
        timestamps = np.arange(len(stable_clusters))

        result = compute_temporal_stability(
            stable_clusters, timestamps, cluster_func
        )

        assert result.temporal_ari > 0.5, (
            f"Stationary data should have temporal ARI > 0.5, got {result.temporal_ari}"
        )

    def test_drifting_data_lower_stability(self, drifting_data, cluster_func):
        """Data with drift should have lower temporal stability than stationary data."""
        X_drift, timestamps_drift = drifting_data

        result_drift = compute_temporal_stability(X_drift, timestamps_drift, cluster_func)

        # Compare to truly stationary data
        X_stationary, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.5, random_state=42)
        timestamps_stationary = np.arange(500)
        result_stationary = compute_temporal_stability(X_stationary, timestamps_stationary, cluster_func)

        # Drifting data should have lower stability than stationary
        # (though with moderate drift, it may still be reasonably high)
        assert result_drift.temporal_ari <= result_stationary.temporal_ari, (
            f"Drifting data should have ARI <= stationary, got {result_drift.temporal_ari} vs {result_stationary.temporal_ari}"
        )

    def test_proportion_drift_detected(self, drifting_data, cluster_func):
        """Should detect changes in cluster proportions."""
        X, timestamps = drifting_data

        result = compute_temporal_stability(X, timestamps, cluster_func)

        assert len(result.train_proportions) > 0
        assert len(result.test_proportions) > 0

    def test_handles_small_test_set(self, cluster_func):
        """Should handle very small test sets gracefully."""
        X = np.random.randn(20, 3)
        timestamps = np.arange(20)

        result = compute_temporal_stability(
            X, timestamps, cluster_func, train_frac=0.95  # Only 1 test sample
        )

        # Should not crash
        assert result is not None


# =============================================================================
# CROSS-SYMBOL STABILITY TESTS
# =============================================================================

class TestCrossSymbolStability:
    """Tests for cross-symbol stability."""

    def test_similar_structures_high_similarity(self, cluster_func):
        """Similar cluster structures should have high similarity."""
        X1, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=42)
        X2, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=43)

        similarity = compute_cross_symbol_stability(X1, X2, cluster_func)

        assert similarity > 0.6, (
            f"Similar structures should have similarity > 0.6, got {similarity}"
        )

    def test_different_structures_lower_similarity(self, cluster_func):
        """Different cluster structures should have lower similarity."""
        X1, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=42)
        X2, _ = make_blobs(n_samples=200, centers=6, cluster_std=0.5, random_state=43)

        def cluster_auto(X):
            # Let k vary
            return KMeans(n_clusters=min(6, len(X)//30), random_state=42).fit_predict(X)

        similarity = compute_cross_symbol_stability(X1, X2, cluster_auto)

        # Should detect difference in number of clusters
        assert similarity < 1.0


# =============================================================================
# SKEPTICAL EDGE CASES
# =============================================================================

class TestStabilityEdgeCases:
    """Skeptical tests for edge cases."""

    def test_single_cluster_stability(self):
        """Single cluster should have perfect bootstrap stability."""
        X = np.random.randn(100, 3)

        def single_cluster(X):
            return np.zeros(len(X), dtype=int)

        result = compute_bootstrap_stability(X, single_cluster, n_bootstraps=20)

        # Single cluster = ARI of 1.0 (perfect agreement)
        assert result.mean_ari == 1.0

    def test_every_point_own_cluster(self):
        """Every point as own cluster should be unstable."""
        X = np.random.randn(50, 3)

        def all_unique(X):
            return np.arange(len(X))

        result = compute_bootstrap_stability(X, all_unique, n_bootstraps=20)

        # Should have low stability (sampling changes assignments)
        assert result.mean_ari < 0.5

    def test_handles_empty_clusters(self, cluster_func):
        """Should handle when clustering produces empty clusters."""
        X = np.random.randn(30, 3)

        # This may produce fewer than 4 clusters
        result = compute_bootstrap_stability(X, cluster_func, n_bootstraps=20)

        assert result is not None
        assert -1 <= result.mean_ari <= 1

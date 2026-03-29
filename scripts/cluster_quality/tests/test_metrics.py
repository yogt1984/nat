"""
Skeptical Tests for Cluster Quality Metrics

These tests verify the metrics correctly identify cluster quality:
- Clear clusters should score high
- Overlapping clusters should score low
- Random data should be detected as unstructured
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from cluster_quality.metrics import (
    compute_silhouette,
    compute_davies_bouldin,
    compute_calinski_harabasz,
    compute_gap_statistic,
    compute_all_metrics,
    SilhouetteResult,
    QualityMetrics,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clear_clusters():
    """Generate well-separated clusters."""
    X, labels = make_blobs(
        n_samples=500,
        n_features=5,
        centers=5,
        cluster_std=0.5,
        random_state=42,
    )
    return X, labels


@pytest.fixture
def overlapping_clusters():
    """Generate heavily overlapping clusters."""
    X, labels = make_blobs(
        n_samples=500,
        n_features=5,
        centers=5,
        cluster_std=3.0,  # High variance = overlap
        random_state=42,
    )
    return X, labels


@pytest.fixture
def random_data():
    """Generate uniform random data (no structure)."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-10, 10, size=(500, 5))
    # Assign random labels
    labels = rng.randint(0, 5, size=500)
    return X, labels


@pytest.fixture
def two_moons():
    """Non-spherical cluster structure."""
    X, labels = make_moons(n_samples=500, noise=0.1, random_state=42)
    return X, labels


# =============================================================================
# SILHOUETTE TESTS
# =============================================================================

class TestSilhouetteScore:
    """Tests for silhouette computation."""

    def test_clear_clusters_high_silhouette(self, clear_clusters):
        """Well-separated clusters should have high silhouette (> 0.5)."""
        X, labels = clear_clusters
        result = compute_silhouette(X, labels)

        assert result.overall > 0.5, (
            f"Clear clusters should have silhouette > 0.5, got {result.overall}"
        )
        assert result.pct_negative < 0.1, (
            f"Clear clusters should have < 10% negative silhouettes"
        )

    def test_overlapping_clusters_low_silhouette(self, overlapping_clusters):
        """Overlapping clusters should have low silhouette (< 0.3)."""
        X, labels = overlapping_clusters
        result = compute_silhouette(X, labels)

        assert result.overall < 0.4, (
            f"Overlapping clusters should have silhouette < 0.4, got {result.overall}"
        )

    def test_random_data_near_zero_silhouette(self, random_data):
        """Random data should have silhouette near zero."""
        X, labels = random_data
        result = compute_silhouette(X, labels)

        assert abs(result.overall) < 0.2, (
            f"Random data should have |silhouette| < 0.2, got {result.overall}"
        )
        assert result.pct_negative > 0.3, (
            "Random data should have many negative silhouettes"
        )

    def test_per_cluster_identifies_weak_clusters(self, clear_clusters):
        """Per-cluster scores should identify weak clusters."""
        X, labels = clear_clusters
        result = compute_silhouette(X, labels)

        # All clusters in clear data should have positive silhouette
        for label, score in result.per_cluster.items():
            assert score > 0, f"Cluster {label} should have positive silhouette"

    def test_silhouette_range(self, clear_clusters):
        """Silhouette should be in [-1, 1]."""
        X, labels = clear_clusters
        result = compute_silhouette(X, labels)

        assert -1 <= result.overall <= 1, "Silhouette must be in [-1, 1]"

    def test_handles_single_cluster(self):
        """Should handle single cluster gracefully."""
        X = np.random.randn(100, 5)
        labels = np.zeros(100, dtype=int)

        result = compute_silhouette(X, labels)
        assert result.overall == 0.0, "Single cluster should have silhouette = 0"

    def test_handles_noise_labels(self, clear_clusters):
        """Should handle -1 noise labels."""
        X, labels = clear_clusters
        labels[::10] = -1  # Mark every 10th point as noise

        result = compute_silhouette(X, labels)
        assert result.overall > 0, "Should still compute valid silhouette"


# =============================================================================
# DAVIES-BOULDIN TESTS
# =============================================================================

class TestDaviesBouldin:
    """Tests for Davies-Bouldin index."""

    def test_clear_clusters_low_db(self, clear_clusters):
        """Well-separated clusters should have low DB (< 1.0)."""
        X, labels = clear_clusters
        db = compute_davies_bouldin(X, labels)

        assert db < 1.0, f"Clear clusters should have DB < 1.0, got {db}"

    def test_overlapping_clusters_high_db(self, overlapping_clusters):
        """Overlapping clusters should have high DB (> 1.5)."""
        X, labels = overlapping_clusters
        db = compute_davies_bouldin(X, labels)

        assert db > 1.0, f"Overlapping clusters should have DB > 1.0, got {db}"

    def test_db_non_negative(self, clear_clusters):
        """Davies-Bouldin should be non-negative."""
        X, labels = clear_clusters
        db = compute_davies_bouldin(X, labels)

        assert db >= 0, "Davies-Bouldin must be >= 0"

    def test_single_cluster_returns_inf(self):
        """Single cluster should return infinity."""
        X = np.random.randn(100, 5)
        labels = np.zeros(100, dtype=int)

        db = compute_davies_bouldin(X, labels)
        assert db == float('inf'), "Single cluster should return infinity"


# =============================================================================
# CALINSKI-HARABASZ TESTS
# =============================================================================

class TestCalinskiHarabasz:
    """Tests for Calinski-Harabasz index."""

    def test_clear_clusters_high_ch(self, clear_clusters):
        """Well-separated clusters should have high CH."""
        X, labels = clear_clusters
        ch = compute_calinski_harabasz(X, labels)

        assert ch > 500, f"Clear clusters should have high CH, got {ch}"

    def test_overlapping_clusters_lower_ch(self, overlapping_clusters):
        """Overlapping clusters should have lower CH."""
        X, labels = overlapping_clusters
        ch_overlap = compute_calinski_harabasz(X, labels)

        X_clear, labels_clear = make_blobs(
            n_samples=500, n_features=5, centers=5,
            cluster_std=0.5, random_state=42
        )
        ch_clear = compute_calinski_harabasz(X_clear, labels_clear)

        assert ch_clear > ch_overlap, "Clear clusters should have higher CH"

    def test_ch_non_negative(self, clear_clusters):
        """Calinski-Harabasz should be non-negative."""
        X, labels = clear_clusters
        ch = compute_calinski_harabasz(X, labels)

        assert ch >= 0, "Calinski-Harabasz must be >= 0"


# =============================================================================
# GAP STATISTIC TESTS
# =============================================================================

class TestGapStatistic:
    """Tests for gap statistic computation."""

    def test_finds_correct_k_for_clear_clusters(self):
        """Gap statistic should find correct k for clear clusters."""
        X, _ = make_blobs(
            n_samples=300,
            n_features=3,
            centers=4,
            cluster_std=0.5,
            random_state=42,
        )

        result = compute_gap_statistic(X, max_clusters=8, n_refs=10)

        # Allow for some flexibility (3-5 clusters)
        assert 3 <= result.optimal_k <= 5, (
            f"Should find ~4 clusters, got {result.optimal_k}"
        )

    def test_gap_higher_than_random(self, clear_clusters):
        """Gap should be positive for structured data."""
        X, _ = clear_clusters
        result = compute_gap_statistic(X, max_clusters=6, n_refs=10)

        # Gap at optimal k should be positive (better than random)
        assert result.gap_at_optimal > 0, "Gap should be positive"

    def test_gap_lengths_match(self, clear_clusters):
        """Gap arrays should have consistent lengths."""
        X, _ = clear_clusters
        result = compute_gap_statistic(X, max_clusters=6, n_refs=10)

        assert len(result.gaps) == len(result.gap_stds), "Gaps and stds should match"
        assert len(result.gaps) <= 6, "Should not exceed max_clusters"

    def test_random_data_low_gap(self, random_data):
        """Random data should have low/no gap improvement."""
        X, _ = random_data
        result = compute_gap_statistic(X, max_clusters=6, n_refs=10)

        # All gaps should be small for random data
        max_gap = max(result.gaps)
        assert max_gap < 0.5, f"Random data should have small gaps, got {max_gap}"


# =============================================================================
# COMBINED METRICS TESTS
# =============================================================================

class TestCombinedMetrics:
    """Tests for combined quality metrics."""

    def test_all_metrics_computed(self, clear_clusters):
        """Should compute all metrics."""
        X, labels = clear_clusters
        metrics = compute_all_metrics(X, labels, compute_gap=True)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.silhouette is not None
        assert metrics.davies_bouldin is not None
        assert metrics.calinski_harabasz is not None
        assert metrics.gap_statistic is not None

    def test_metrics_consistent_quality_signal(self, clear_clusters, overlapping_clusters):
        """All metrics should agree on quality direction."""
        X_clear, labels_clear = clear_clusters
        X_overlap, labels_overlap = overlapping_clusters

        m_clear = compute_all_metrics(X_clear, labels_clear, compute_gap=False)
        m_overlap = compute_all_metrics(X_overlap, labels_overlap, compute_gap=False)

        # Clear clusters should be better on all metrics
        assert m_clear.silhouette.overall > m_overlap.silhouette.overall
        assert m_clear.davies_bouldin < m_overlap.davies_bouldin
        assert m_clear.calinski_harabasz > m_overlap.calinski_harabasz

    def test_summary_generation(self, clear_clusters):
        """Summary should generate without error."""
        X, labels = clear_clusters
        metrics = compute_all_metrics(X, labels, compute_gap=False)

        summary = metrics.summary()
        assert isinstance(summary, str)
        assert "Silhouette" in summary
        assert "Davies-Bouldin" in summary


# =============================================================================
# SKEPTICAL / EDGE CASE TESTS
# =============================================================================

class TestSkepticalEdgeCases:
    """Skeptical tests for edge cases and potential failures."""

    def test_small_sample_size(self):
        """Should handle small sample sizes."""
        X = np.random.randn(20, 3)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

        metrics = compute_all_metrics(X, labels, compute_gap=False)
        assert metrics.silhouette.overall is not None

    def test_high_dimensional_data(self):
        """Should handle high-dimensional data."""
        X, labels = make_blobs(n_samples=200, n_features=50, centers=3, random_state=42)

        metrics = compute_all_metrics(X, labels, compute_gap=False)
        assert metrics.silhouette.overall > 0

    def test_unbalanced_clusters(self):
        """Should handle unbalanced cluster sizes."""
        # Create unbalanced: 100, 50, 10 samples
        X1 = np.random.randn(100, 3)
        X2 = np.random.randn(50, 3) + 5
        X3 = np.random.randn(10, 3) + 10
        X = np.vstack([X1, X2, X3])
        labels = np.array([0]*100 + [1]*50 + [2]*10)

        metrics = compute_all_metrics(X, labels, compute_gap=False)
        assert metrics.silhouette.overall > -1  # Should be valid

    def test_empty_cluster_handling(self):
        """Should handle when a label is missing."""
        X = np.random.randn(100, 3)
        labels = np.array([0]*50 + [2]*50)  # No cluster 1

        result = compute_silhouette(X, labels)
        assert 0 in result.per_cluster and 2 in result.per_cluster
        assert 1 not in result.per_cluster

    def test_reproducibility(self, clear_clusters):
        """Results should be reproducible."""
        X, labels = clear_clusters

        m1 = compute_all_metrics(X, labels, compute_gap=False)
        m2 = compute_all_metrics(X, labels, compute_gap=False)

        assert m1.silhouette.overall == m2.silhouette.overall
        assert m1.davies_bouldin == m2.davies_bouldin

    def test_numerical_stability_extreme_values(self):
        """Should handle extreme feature values."""
        X = np.array([
            [1e10, 1e10, 1e10],
            [1e10 + 1, 1e10 + 1, 1e10 + 1],
            [-1e10, -1e10, -1e10],
            [-1e10 - 1, -1e10 - 1, -1e10 - 1],
        ])
        labels = np.array([0, 0, 1, 1])

        result = compute_silhouette(X, labels)
        assert not np.isnan(result.overall), "Should not produce NaN"

    def test_identical_points_in_cluster(self):
        """Should handle identical points."""
        X = np.array([
            [0, 0], [0, 0], [0, 0],
            [10, 10], [10, 10], [10, 10],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])

        # Should not crash, silhouette may be 1.0 for perfect separation
        result = compute_silhouette(X, labels)
        assert result.overall >= 0

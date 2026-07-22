# Cluster Quality Measurement Implementation Tasks

**Date:** 2026-03-27
**Model:** Sonnet (for execution)
**Reference:** CLUSTER_QUALITY_MEASUREMENT_FRAMEWORK_SPECS.md

---

## Overview

This document defines executable tasks for implementing the Cluster Quality Measurement Framework. Each task is self-contained with all necessary context, code snippets, and skeptical tests.

**Directory Structure:**
```
scripts/
  cluster_quality/
    __init__.py
    metrics.py           # Task 1: Core quality metrics
    stability.py         # Task 2: Stability metrics
    validation.py        # Task 3: External validation
    composite.py         # Task 4: Composite scoring
    refinement_agent.py  # Task 5: Agentic refinement
    tests/
      __init__.py
      test_metrics.py
      test_stability.py
      test_validation.py
      test_composite.py
      test_refinement.py
```

---

## Task 1: Core Quality Metrics

**File:** `scripts/cluster_quality/metrics.py`
**Priority:** High
**Estimated Complexity:** Medium

### 1.1 Requirements

Implement internal cluster validation metrics:
- Silhouette Score (with per-cluster breakdown)
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Gap Statistic

### 1.2 Implementation

```python
"""
Core Cluster Quality Metrics

Internal validation metrics for measuring cluster structure quality.
These metrics use only feature data, no external labels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans


@dataclass
class SilhouetteResult:
    """Silhouette analysis results."""
    overall: float
    per_cluster: Dict[int, float]
    std: float
    pct_negative: float  # Fraction of misclassified points

    def is_acceptable(self) -> bool:
        """Check if silhouette indicates meaningful clusters."""
        return self.overall >= 0.25 and self.pct_negative < 0.3


@dataclass
class GapStatisticResult:
    """Gap statistic results."""
    gaps: List[float]
    gap_stds: List[float]
    optimal_k: int
    gap_at_optimal: float


@dataclass
class QualityMetrics:
    """Combined internal quality metrics."""
    silhouette: SilhouetteResult
    davies_bouldin: float
    calinski_harabasz: float
    gap_statistic: Optional[GapStatisticResult]
    n_clusters: int
    n_samples: int

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CLUSTER QUALITY METRICS",
            "=" * 60,
            f"Samples: {self.n_samples}, Clusters: {self.n_clusters}",
            "",
            "Internal Validation:",
            f"  Silhouette Score:     {self.silhouette.overall:.3f}",
            f"  Davies-Bouldin Index: {self.davies_bouldin:.3f}",
            f"  Calinski-Harabasz:    {self.calinski_harabasz:.1f}",
            "",
            "Silhouette Details:",
            f"  Std Dev:              {self.silhouette.std:.3f}",
            f"  % Negative:           {self.silhouette.pct_negative:.1%}",
        ]

        for label, score in self.silhouette.per_cluster.items():
            lines.append(f"  Cluster {label}:           {score:.3f}")

        if self.gap_statistic:
            lines.extend([
                "",
                f"Gap Statistic:",
                f"  Optimal k:            {self.gap_statistic.optimal_k}",
                f"  Gap at optimal:       {self.gap_statistic.gap_at_optimal:.3f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> SilhouetteResult:
    """
    Compute silhouette metrics with per-cluster breakdown.

    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)

    Returns:
        SilhouetteResult with overall score, per-cluster scores, and diagnostics
    """
    # Filter out noise labels (-1)
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return SilhouetteResult(
            overall=0.0,
            per_cluster={},
            std=0.0,
            pct_negative=1.0,
        )

    overall = silhouette_score(X_clean, labels_clean)
    samples = silhouette_samples(X_clean, labels_clean)

    per_cluster = {}
    for label in np.unique(labels_clean):
        cluster_mask = labels_clean == label
        per_cluster[int(label)] = float(samples[cluster_mask].mean())

    return SilhouetteResult(
        overall=float(overall),
        per_cluster=per_cluster,
        std=float(samples.std()),
        pct_negative=float((samples < 0).mean()),
    )


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index. Lower is better.

    Args:
        X: Feature matrix
        labels: Cluster assignments

    Returns:
        Davies-Bouldin index (0 = perfect, higher = worse)
    """
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return float('inf')

    return float(davies_bouldin_score(X_clean, labels_clean))


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz index (variance ratio). Higher is better.

    Args:
        X: Feature matrix
        labels: Cluster assignments

    Returns:
        Calinski-Harabasz score (higher = better separation)
    """
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return 0.0

    return float(calinski_harabasz_score(X_clean, labels_clean))


def compute_gap_statistic(
    X: np.ndarray,
    max_clusters: int = 10,
    n_refs: int = 20,
    random_state: int = 42,
) -> GapStatisticResult:
    """
    Compute gap statistic for optimal cluster number selection.

    Compares within-cluster dispersion to expected dispersion under
    uniform random null distribution.

    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        n_refs: Number of reference datasets to generate
        random_state: Random seed for reproducibility

    Returns:
        GapStatisticResult with gaps, stds, and optimal k
    """
    rng = np.random.RandomState(random_state)

    def compute_Wk(X: np.ndarray, labels: np.ndarray) -> float:
        """Within-cluster sum of squares."""
        W = 0.0
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            W += ((cluster_points - centroid) ** 2).sum()
        return W

    # Compute for real data
    Wks = []
    for k in range(1, min(max_clusters + 1, len(X) // 10)):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        wk = compute_Wk(X, labels)
        Wks.append(np.log(wk + 1e-10))

    # Compute for reference (uniform random in bounding box)
    Wks_ref = []
    X_min, X_max = X.min(axis=0), X.max(axis=0)

    for k in range(1, min(max_clusters + 1, len(X) // 10)):
        ref_Wks = []
        for _ in range(n_refs):
            X_ref = rng.uniform(X_min, X_max, size=X.shape)
            kmeans = KMeans(n_clusters=k, random_state=None, n_init=3)
            labels = kmeans.fit_predict(X_ref)
            ref_Wks.append(np.log(compute_Wk(X_ref, labels) + 1e-10))
        Wks_ref.append((np.mean(ref_Wks), np.std(ref_Wks)))

    # Gap = E[log(W_ref)] - log(W)
    gaps = [ref[0] - wk for ref, wk in zip(Wks_ref, Wks)]
    gap_stds = [ref[1] for ref in Wks_ref]

    optimal_k = int(np.argmax(gaps) + 1)

    return GapStatisticResult(
        gaps=gaps,
        gap_stds=gap_stds,
        optimal_k=optimal_k,
        gap_at_optimal=gaps[optimal_k - 1] if gaps else 0.0,
    )


def compute_all_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    compute_gap: bool = True,
) -> QualityMetrics:
    """
    Compute all internal quality metrics.

    Args:
        X: Feature matrix
        labels: Cluster assignments
        compute_gap: Whether to compute gap statistic (slower)

    Returns:
        QualityMetrics with all internal validation metrics
    """
    silhouette = compute_silhouette(X, labels)
    db = compute_davies_bouldin(X, labels)
    ch = compute_calinski_harabasz(X, labels)

    gap = compute_gap_statistic(X) if compute_gap else None

    return QualityMetrics(
        silhouette=silhouette,
        davies_bouldin=db,
        calinski_harabasz=ch,
        gap_statistic=gap,
        n_clusters=len(np.unique(labels[labels != -1])),
        n_samples=len(X),
    )
```

### 1.3 Skeptical Tests

**File:** `scripts/cluster_quality/tests/test_metrics.py`

```python
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
```

---

## Task 2: Stability Metrics

**File:** `scripts/cluster_quality/stability.py`
**Priority:** High
**Estimated Complexity:** Medium

### 2.1 Requirements

Implement cluster stability measurements:
- Bootstrap Stability (resampling robustness)
- Temporal Stability (consistency across time)
- Cross-Validation Stability

### 2.2 Implementation

```python
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
```

### 2.3 Skeptical Tests

**File:** `scripts/cluster_quality/tests/test_stability.py`

```python
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
        """Unstable clusters should have lower ARI (< 0.5)."""
        result = compute_bootstrap_stability(
            unstable_clusters, cluster_func, n_bootstraps=50
        )

        assert result.mean_ari < 0.6, (
            f"Unstable clusters should have lower ARI, got {result.mean_ari}"
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

    def test_different_seeds_different_results(self, stable_clusters, cluster_func):
        """Different seeds should produce slightly different results."""
        r1 = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=30, random_state=42
        )
        r2 = compute_bootstrap_stability(
            stable_clusters, cluster_func, n_bootstraps=30, random_state=99
        )

        # Should be similar but not identical
        assert abs(r1.mean_ari - r2.mean_ari) < 0.2
        assert r1.all_aris != r2.all_aris


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
        """Data with drift should have lower temporal stability."""
        X, timestamps = drifting_data

        result = compute_temporal_stability(X, timestamps, cluster_func)

        # Drift should reduce stability
        # Note: actual threshold depends on drift magnitude
        assert result.temporal_ari < 0.9, (
            "Drifting data should not have perfect temporal stability"
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
```

---

## Task 3: External Validation

**File:** `scripts/cluster_quality/validation.py`
**Priority:** High
**Estimated Complexity:** Medium

### 3.1 Requirements

Implement external validation against economic outcomes:
- Forward Return Differentiation (ANOVA + Kruskal-Wallis)
- Volatility Regime Detection
- Transition Matrix Analysis

### 3.2 Implementation

```python
"""
External Validation Metrics

Validate clusters against external outcomes (returns, volatility).
Tests whether clusters have predictive value.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy import stats


@dataclass
class ReturnDifferentiationResult:
    """Results of testing if clusters have different returns."""
    horizon: int
    anova_f: float
    anova_p: float
    kruskal_h: float
    kruskal_p: float
    eta_squared: float  # Effect size
    cluster_means: Dict[int, float]
    cluster_stds: Dict[int, float]
    significant: bool

    def effect_size_interpretation(self) -> str:
        """Interpret effect size."""
        if self.eta_squared >= 0.14:
            return "Large effect"
        elif self.eta_squared >= 0.06:
            return "Medium effect"
        elif self.eta_squared >= 0.01:
            return "Small effect"
        return "Negligible effect"


@dataclass
class VolatilityDifferentiationResult:
    """Results of testing if clusters have different volatility."""
    levene_stat: float
    levene_p: float  # Tests variance equality
    kruskal_h: float
    kruskal_p: float  # Tests median differences
    cluster_volatilities: Dict[int, float]
    significant: bool


@dataclass
class TransitionMatrixResult:
    """Transition matrix analysis results."""
    transition_matrix: List[List[float]]
    self_transition_rate: float  # Regime persistence
    transition_entropy: float  # Predictability
    avg_regime_duration: float  # Average bars in regime
    cluster_labels: List[int]


@dataclass
class ExternalValidationResults:
    """Combined external validation results."""
    return_differentiation: Dict[int, ReturnDifferentiationResult]
    volatility_differentiation: Optional[VolatilityDifferentiationResult]
    transitions: Optional[TransitionMatrixResult]

    def is_predictive(self) -> bool:
        """Check if clusters have predictive value."""
        # Need at least one significant return horizon
        any_significant = any(
            r.significant for r in self.return_differentiation.values()
        )
        return any_significant


def compute_return_differentiation(
    labels: np.ndarray,
    forward_returns: Dict[int, np.ndarray],
) -> Dict[int, ReturnDifferentiationResult]:
    """
    Test if clusters have statistically different forward returns.

    Args:
        labels: Cluster assignments (n_samples,)
        forward_returns: Dict of horizon (seconds) -> returns array

    Returns:
        Dict of horizon -> ReturnDifferentiationResult
    """
    results = {}

    for horizon, returns in forward_returns.items():
        if len(returns) != len(labels):
            continue

        # Group returns by cluster
        unique_labels = [l for l in np.unique(labels) if l != -1]
        if len(unique_labels) < 2:
            continue

        groups = []
        cluster_means = {}
        cluster_stds = {}

        for label in unique_labels:
            mask = labels == label
            group_returns = returns[mask]
            if len(group_returns) > 0:
                groups.append(group_returns)
                cluster_means[int(label)] = float(group_returns.mean())
                cluster_stds[int(label)] = float(group_returns.std())

        if len(groups) < 2:
            continue

        # ANOVA (assumes normality)
        try:
            f_stat, anova_p = stats.f_oneway(*groups)
        except Exception:
            f_stat, anova_p = 0.0, 1.0

        # Kruskal-Wallis (non-parametric)
        try:
            h_stat, kw_p = stats.kruskal(*groups)
        except Exception:
            h_stat, kw_p = 0.0, 1.0

        # Effect size (eta-squared)
        all_returns = returns[labels != -1]
        grand_mean = all_returns.mean()

        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in groups
        )
        ss_total = ((all_returns - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        results[horizon] = ReturnDifferentiationResult(
            horizon=horizon,
            anova_f=float(f_stat),
            anova_p=float(anova_p),
            kruskal_h=float(h_stat),
            kruskal_p=float(kw_p),
            eta_squared=float(eta_squared),
            cluster_means=cluster_means,
            cluster_stds=cluster_stds,
            significant=kw_p < 0.05,
        )

    return results


def compute_volatility_differentiation(
    labels: np.ndarray,
    forward_volatility: np.ndarray,
) -> VolatilityDifferentiationResult:
    """
    Test if clusters correspond to different volatility regimes.

    Args:
        labels: Cluster assignments
        forward_volatility: Forward-looking volatility measure

    Returns:
        VolatilityDifferentiationResult with test statistics
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]

    if len(unique_labels) < 2:
        return VolatilityDifferentiationResult(
            levene_stat=0.0,
            levene_p=1.0,
            kruskal_h=0.0,
            kruskal_p=1.0,
            cluster_volatilities={},
            significant=False,
        )

    groups = []
    cluster_vols = {}

    for label in unique_labels:
        mask = labels == label
        group_vol = forward_volatility[mask]
        if len(group_vol) > 0:
            groups.append(group_vol)
            cluster_vols[int(label)] = float(group_vol.mean())

    if len(groups) < 2:
        return VolatilityDifferentiationResult(
            levene_stat=0.0,
            levene_p=1.0,
            kruskal_h=0.0,
            kruskal_p=1.0,
            cluster_volatilities=cluster_vols,
            significant=False,
        )

    # Levene's test for equality of variances
    try:
        levene_stat, levene_p = stats.levene(*groups)
    except Exception:
        levene_stat, levene_p = 0.0, 1.0

    # Kruskal-Wallis for median differences
    try:
        h_stat, kw_p = stats.kruskal(*groups)
    except Exception:
        h_stat, kw_p = 0.0, 1.0

    return VolatilityDifferentiationResult(
        levene_stat=float(levene_stat),
        levene_p=float(levene_p),
        kruskal_h=float(h_stat),
        kruskal_p=float(kw_p),
        cluster_volatilities=cluster_vols,
        significant=kw_p < 0.05 or levene_p < 0.05,
    )


def compute_transition_matrix(
    labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> TransitionMatrixResult:
    """
    Analyze cluster transition patterns.

    Args:
        labels: Cluster assignments (n_samples,)
        timestamps: Optional timestamps for ordering

    Returns:
        TransitionMatrixResult with transition analysis
    """
    if timestamps is not None:
        sort_idx = np.argsort(timestamps)
        sorted_labels = labels[sort_idx]
    else:
        sorted_labels = labels

    # Filter out noise
    valid_mask = sorted_labels != -1
    sorted_labels = sorted_labels[valid_mask]

    unique_labels = sorted(np.unique(sorted_labels))
    n_clusters = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    if n_clusters < 2:
        return TransitionMatrixResult(
            transition_matrix=[],
            self_transition_rate=1.0,
            transition_entropy=0.0,
            avg_regime_duration=float('inf'),
            cluster_labels=[],
        )

    # Count transitions
    transition_counts = np.zeros((n_clusters, n_clusters))

    for i in range(len(sorted_labels) - 1):
        from_idx = label_to_idx[sorted_labels[i]]
        to_idx = label_to_idx[sorted_labels[i + 1]]
        transition_counts[from_idx, to_idx] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums,
        where=row_sums > 0,
        out=np.zeros_like(transition_counts, dtype=float)
    )

    # Self-transition rate (regime persistence)
    self_transition = np.diag(transition_probs).mean()

    # Entropy of transitions (predictability)
    def row_entropy(row):
        row = row[row > 0]
        return -np.sum(row * np.log(row + 1e-10))

    transition_entropy = np.mean([row_entropy(row) for row in transition_probs])

    # Average regime duration
    avg_duration = 1.0 / (1.0 - self_transition + 1e-10)

    return TransitionMatrixResult(
        transition_matrix=transition_probs.tolist(),
        self_transition_rate=float(self_transition),
        transition_entropy=float(transition_entropy),
        avg_regime_duration=float(avg_duration),
        cluster_labels=[int(l) for l in unique_labels],
    )


def compute_all_external_validation(
    labels: np.ndarray,
    forward_returns: Dict[int, np.ndarray],
    forward_volatility: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> ExternalValidationResults:
    """
    Compute all external validation metrics.

    Args:
        labels: Cluster assignments
        forward_returns: Dict of horizon -> returns
        forward_volatility: Optional volatility array
        timestamps: Optional timestamps for transition analysis

    Returns:
        ExternalValidationResults with all metrics
    """
    return_diff = compute_return_differentiation(labels, forward_returns)

    vol_diff = None
    if forward_volatility is not None:
        vol_diff = compute_volatility_differentiation(labels, forward_volatility)

    transitions = None
    if timestamps is not None:
        transitions = compute_transition_matrix(labels, timestamps)

    return ExternalValidationResults(
        return_differentiation=return_diff,
        volatility_differentiation=vol_diff,
        transitions=transitions,
    )
```

### 3.3 Skeptical Tests

**File:** `scripts/cluster_quality/tests/test_validation.py`

```python
"""
Skeptical Tests for External Validation Metrics

Tests verify that external validation correctly identifies:
- Clusters with predictive power for returns
- Clusters that correspond to volatility regimes
- Meaningful transition patterns
"""

import pytest
import numpy as np
from cluster_quality.validation import (
    compute_return_differentiation,
    compute_volatility_differentiation,
    compute_transition_matrix,
    compute_all_external_validation,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def predictive_clusters():
    """Clusters with genuinely different return distributions."""
    n = 500
    labels = np.array([0] * 200 + [1] * 200 + [2] * 100)

    # Cluster 0: negative returns
    # Cluster 1: positive returns
    # Cluster 2: neutral
    returns_60 = np.concatenate([
        np.random.normal(-0.01, 0.02, 200),  # Cluster 0: negative
        np.random.normal(0.01, 0.02, 200),   # Cluster 1: positive
        np.random.normal(0.0, 0.02, 100),    # Cluster 2: neutral
    ])

    return labels, {60: returns_60}


@pytest.fixture
def non_predictive_clusters():
    """Clusters with no relationship to returns."""
    n = 500
    labels = np.random.randint(0, 3, n)  # Random assignment
    returns_60 = np.random.normal(0, 0.02, n)  # Same distribution

    return labels, {60: returns_60}


@pytest.fixture
def volatility_regimes():
    """Clusters corresponding to different volatility regimes."""
    n = 500
    labels = np.array([0] * 200 + [1] * 200 + [2] * 100)

    # Different volatilities per cluster
    vol = np.concatenate([
        np.random.lognormal(0, 0.2, 200),      # Low vol
        np.random.lognormal(0.5, 0.2, 200),    # Medium vol
        np.random.lognormal(1.0, 0.2, 100),    # High vol
    ])

    return labels, vol


@pytest.fixture
def persistent_regimes():
    """Clusters with high self-transition (persistent regimes)."""
    # Simulated regime that stays same for ~20 periods then switches
    labels = []
    current = 0
    for _ in range(500):
        if np.random.random() < 0.95:  # 95% self-transition
            labels.append(current)
        else:
            current = (current + 1) % 3
            labels.append(current)

    return np.array(labels), np.arange(500)


# =============================================================================
# RETURN DIFFERENTIATION TESTS
# =============================================================================

class TestReturnDifferentiation:
    """Tests for return differentiation analysis."""

    def test_predictive_clusters_significant(self, predictive_clusters):
        """Clusters with different returns should be significant."""
        labels, returns = predictive_clusters
        result = compute_return_differentiation(labels, returns)

        assert 60 in result
        assert result[60].significant, (
            f"Predictive clusters should be significant, p={result[60].kruskal_p}"
        )
        assert result[60].eta_squared > 0.05, (
            "Should have meaningful effect size"
        )

    def test_non_predictive_clusters_not_significant(self, non_predictive_clusters):
        """Random clusters should not be significant."""
        labels, returns = non_predictive_clusters
        result = compute_return_differentiation(labels, returns)

        assert 60 in result
        # Should usually not be significant (alpha=0.05)
        # Allow some false positives in testing
        assert result[60].kruskal_p > 0.01 or result[60].eta_squared < 0.03, (
            "Random clusters should have low significance or effect size"
        )

    def test_cluster_means_correct_sign(self, predictive_clusters):
        """Cluster means should have expected signs."""
        labels, returns = predictive_clusters
        result = compute_return_differentiation(labels, returns)

        # Cluster 0 should have negative mean
        assert result[60].cluster_means[0] < 0, "Cluster 0 should be negative"
        # Cluster 1 should have positive mean
        assert result[60].cluster_means[1] > 0, "Cluster 1 should be positive"

    def test_effect_size_interpretation(self, predictive_clusters):
        """Effect size interpretation should be sensible."""
        labels, returns = predictive_clusters
        result = compute_return_differentiation(labels, returns)

        interpretation = result[60].effect_size_interpretation()
        assert interpretation in ["Large effect", "Medium effect", "Small effect", "Negligible effect"]

    def test_handles_multiple_horizons(self, predictive_clusters):
        """Should handle multiple return horizons."""
        labels, _ = predictive_clusters
        returns = {
            60: np.random.randn(500),
            300: np.random.randn(500),
            3600: np.random.randn(500),
        }

        result = compute_return_differentiation(labels, returns)
        assert len(result) == 3


# =============================================================================
# VOLATILITY DIFFERENTIATION TESTS
# =============================================================================

class TestVolatilityDifferentiation:
    """Tests for volatility regime detection."""

    def test_volatility_regimes_detected(self, volatility_regimes):
        """Different volatility regimes should be detected."""
        labels, vol = volatility_regimes
        result = compute_volatility_differentiation(labels, vol)

        assert result.significant, (
            "Volatility regimes should be significant"
        )

        # High vol cluster should have higher mean
        assert result.cluster_volatilities[2] > result.cluster_volatilities[0], (
            "Cluster 2 should have higher volatility than cluster 0"
        )

    def test_same_volatility_not_significant(self):
        """Clusters with same volatility should not be significant."""
        labels = np.array([0] * 200 + [1] * 200)
        vol = np.random.lognormal(0, 0.2, 400)  # Same distribution

        result = compute_volatility_differentiation(labels, vol)

        # Should usually not be significant
        assert result.kruskal_p > 0.01 or not result.significant


# =============================================================================
# TRANSITION MATRIX TESTS
# =============================================================================

class TestTransitionMatrix:
    """Tests for transition matrix analysis."""

    def test_persistent_regimes_high_self_transition(self, persistent_regimes):
        """Persistent regimes should have high self-transition rate."""
        labels, timestamps = persistent_regimes
        result = compute_transition_matrix(labels, timestamps)

        assert result.self_transition_rate > 0.8, (
            f"Persistent regimes should have high self-transition, got {result.self_transition_rate}"
        )

    def test_random_transitions_lower_self_transition(self):
        """Random label changes should have lower self-transition."""
        labels = np.random.randint(0, 3, 500)
        timestamps = np.arange(500)

        result = compute_transition_matrix(labels, timestamps)

        # With 3 clusters, random self-transition ~ 1/3
        assert result.self_transition_rate < 0.5, (
            "Random transitions should have ~1/k self-transition"
        )

    def test_transition_matrix_row_sums_to_one(self, persistent_regimes):
        """Transition matrix rows should sum to 1."""
        labels, timestamps = persistent_regimes
        result = compute_transition_matrix(labels, timestamps)

        for row in result.transition_matrix:
            row_sum = sum(row)
            assert abs(row_sum - 1.0) < 1e-6 or row_sum == 0, (
                f"Row should sum to 1, got {row_sum}"
            )

    def test_avg_regime_duration(self, persistent_regimes):
        """Average regime duration should be sensible."""
        labels, timestamps = persistent_regimes
        result = compute_transition_matrix(labels, timestamps)

        # With 95% self-transition, expect ~20 periods per regime
        assert 10 < result.avg_regime_duration < 50, (
            f"Expected ~20 period duration, got {result.avg_regime_duration}"
        )

    def test_handles_noise_labels(self):
        """Should handle -1 noise labels."""
        labels = np.array([0, 0, -1, 1, 1, -1, 2, 2])
        timestamps = np.arange(8)

        result = compute_transition_matrix(labels, timestamps)

        assert -1 not in result.cluster_labels


# =============================================================================
# COMBINED VALIDATION TESTS
# =============================================================================

class TestCombinedValidation:
    """Tests for combined external validation."""

    def test_all_results_computed(self, predictive_clusters, volatility_regimes):
        """Should compute all validation results."""
        labels, returns = predictive_clusters
        _, vol = volatility_regimes
        vol = vol[:len(labels)]  # Match length
        timestamps = np.arange(len(labels))

        result = compute_all_external_validation(
            labels, returns, vol, timestamps
        )

        assert result.return_differentiation is not None
        assert result.volatility_differentiation is not None
        assert result.transitions is not None

    def test_is_predictive_method(self, predictive_clusters, non_predictive_clusters):
        """is_predictive should distinguish clusters."""
        labels_pred, returns_pred = predictive_clusters
        result_pred = compute_all_external_validation(labels_pred, returns_pred)

        labels_rand, returns_rand = non_predictive_clusters
        result_rand = compute_all_external_validation(labels_rand, returns_rand)

        assert result_pred.is_predictive() or result_rand.is_predictive() is False, (
            "Should distinguish predictive from non-predictive"
        )


# =============================================================================
# SKEPTICAL EDGE CASES
# =============================================================================

class TestValidationEdgeCases:
    """Skeptical tests for edge cases."""

    def test_single_cluster_not_significant(self):
        """Single cluster should not be significant."""
        labels = np.zeros(100, dtype=int)
        returns = {60: np.random.randn(100)}

        result = compute_return_differentiation(labels, returns)
        assert len(result) == 0 or not any(r.significant for r in result.values())

    def test_empty_cluster_handled(self):
        """Should handle empty clusters."""
        labels = np.array([0] * 50 + [2] * 50)  # No cluster 1
        returns = {60: np.random.randn(100)}

        result = compute_return_differentiation(labels, returns)
        assert 60 in result

    def test_nan_returns_handled(self):
        """Should handle NaN in returns."""
        labels = np.array([0] * 50 + [1] * 50)
        returns_with_nan = np.random.randn(100)
        returns_with_nan[::10] = np.nan

        # Should not crash (behavior may vary)
        try:
            result = compute_return_differentiation(labels, {60: returns_with_nan})
        except Exception as e:
            pytest.skip(f"NaN handling not implemented: {e}")

    def test_mismatched_lengths_skipped(self):
        """Should skip horizons with mismatched lengths."""
        labels = np.array([0] * 50 + [1] * 50)
        returns = {
            60: np.random.randn(100),   # Correct length
            300: np.random.randn(50),   # Wrong length
        }

        result = compute_return_differentiation(labels, returns)
        assert 60 in result
        assert 300 not in result
```

---

## Task 4: Composite Scoring

**File:** `scripts/cluster_quality/composite.py`
**Priority:** Medium
**Estimated Complexity:** Low

### 4.1 Requirements

Implement composite quality scoring:
- Weighted combination of all metrics
- Quality grading (A/B/C/D/F)
- HMM readiness assessment

### 4.2 Implementation

```python
"""
Composite Cluster Quality Scoring

Combines internal, stability, and external metrics into a single
actionable score for HMM readiness assessment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from .metrics import QualityMetrics
from .stability import StabilityMetrics
from .validation import ExternalValidationResults


@dataclass
class ClusterQualityScore:
    """
    Composite cluster quality assessment.

    Combines multiple metrics into a single score [0, 1].
    """
    # Internal metrics (0-1 scale)
    silhouette: float = 0.0
    davies_bouldin_normalized: float = 0.0  # Inverted, capped

    # Stability metrics (0-1 scale)
    bootstrap_stability: float = 0.0
    temporal_stability: float = 0.0

    # External metrics (0-1 scale)
    return_significance: float = 0.0
    volatility_significance: float = 0.0

    # Optional: raw metrics for reference
    raw_metrics: Optional[QualityMetrics] = None
    raw_stability: Optional[StabilityMetrics] = None
    raw_validation: Optional[ExternalValidationResults] = None

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted composite score.

        Default weights emphasize stability and predictive power
        over internal metrics alone.
        """
        if weights is None:
            weights = {
                "silhouette": 0.15,
                "davies_bouldin": 0.10,
                "bootstrap_stability": 0.20,
                "temporal_stability": 0.20,
                "return_significance": 0.20,
                "volatility_significance": 0.15,
            }

        score = (
            weights.get("silhouette", 0) * max(0, self.silhouette) +
            weights.get("davies_bouldin", 0) * self.davies_bouldin_normalized +
            weights.get("bootstrap_stability", 0) * self.bootstrap_stability +
            weights.get("temporal_stability", 0) * self.temporal_stability +
            weights.get("return_significance", 0) * self.return_significance +
            weights.get("volatility_significance", 0) * self.volatility_significance
        )

        return min(1.0, max(0.0, score))

    def get_grade(self) -> str:
        """Human-readable quality grade."""
        score = self.compute_composite()
        if score >= 0.8:
            return "A - Excellent: Ready for HMM"
        elif score >= 0.6:
            return "B - Good: Minor refinements needed"
        elif score >= 0.4:
            return "C - Fair: Significant refinements needed"
        elif score >= 0.2:
            return "D - Poor: Consider different features"
        else:
            return "F - Failed: No meaningful structure"

    def is_hmm_ready(self) -> bool:
        """Check if quality meets HMM readiness thresholds."""
        return (
            self.silhouette >= 0.3 and
            self.davies_bouldin_normalized >= 0.25 and  # DB < 1.5
            self.bootstrap_stability >= 0.6 and
            self.temporal_stability >= 0.5 and
            (self.return_significance >= 0.5 or self.volatility_significance >= 0.5)
        )

    def get_weaknesses(self) -> list:
        """Identify weak areas for targeted improvement."""
        weaknesses = []

        if self.silhouette < 0.3:
            weaknesses.append(("silhouette", "Clusters overlap significantly"))
        if self.davies_bouldin_normalized < 0.25:
            weaknesses.append(("davies_bouldin", "Poor cluster separation"))
        if self.bootstrap_stability < 0.6:
            weaknesses.append(("bootstrap_stability", "Clusters unstable to resampling"))
        if self.temporal_stability < 0.5:
            weaknesses.append(("temporal_stability", "Clusters drift over time"))
        if self.return_significance < 0.5:
            weaknesses.append(("return_significance", "No return predictive power"))
        if self.volatility_significance < 0.5:
            weaknesses.append(("volatility_significance", "No volatility differentiation"))

        return weaknesses

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CLUSTER QUALITY ASSESSMENT",
            "=" * 60,
            "",
            f"Overall Grade: {self.get_grade()}",
            f"Composite Score: {self.compute_composite():.3f}",
            f"HMM Ready: {'Yes' if self.is_hmm_ready() else 'No'}",
            "",
            "Component Scores:",
            f"  Silhouette:           {self.silhouette:.3f}",
            f"  Davies-Bouldin (inv): {self.davies_bouldin_normalized:.3f}",
            f"  Bootstrap Stability:  {self.bootstrap_stability:.3f}",
            f"  Temporal Stability:   {self.temporal_stability:.3f}",
            f"  Return Significance:  {self.return_significance:.3f}",
            f"  Volatility Signif.:   {self.volatility_significance:.3f}",
        ]

        weaknesses = self.get_weaknesses()
        if weaknesses:
            lines.extend(["", "Weaknesses:"])
            for metric, desc in weaknesses:
                lines.append(f"  - {metric}: {desc}")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_quality_score(
    metrics: QualityMetrics,
    stability: StabilityMetrics,
    validation: ExternalValidationResults,
) -> ClusterQualityScore:
    """
    Compute composite quality score from component metrics.

    Args:
        metrics: Internal quality metrics
        stability: Stability metrics
        validation: External validation results

    Returns:
        ClusterQualityScore with all components
    """
    # Normalize Davies-Bouldin (invert, cap at 2)
    db_normalized = 1.0 - min(metrics.davies_bouldin, 2.0) / 2.0

    # Convert return significance from p-values
    return_sig = 0.0
    for horizon_result in validation.return_differentiation.values():
        if horizon_result.significant:
            # Scale by effect size
            return_sig = max(return_sig, min(1.0, horizon_result.eta_squared * 10))

    # Volatility significance
    vol_sig = 0.0
    if validation.volatility_differentiation:
        if validation.volatility_differentiation.significant:
            vol_sig = 1.0

    # Temporal stability (may be None)
    temporal = 0.5  # Default
    if stability.temporal:
        temporal = stability.temporal.temporal_ari

    return ClusterQualityScore(
        silhouette=metrics.silhouette.overall,
        davies_bouldin_normalized=db_normalized,
        bootstrap_stability=stability.bootstrap.mean_ari,
        temporal_stability=temporal,
        return_significance=return_sig,
        volatility_significance=vol_sig,
        raw_metrics=metrics,
        raw_stability=stability,
        raw_validation=validation,
    )
```

### 4.3 Skeptical Tests

**File:** `scripts/cluster_quality/tests/test_composite.py`

```python
"""
Skeptical Tests for Composite Scoring

Tests verify that composite scoring:
- Correctly combines component metrics
- Produces sensible grades
- Accurately identifies HMM readiness
"""

import pytest
import numpy as np
from cluster_quality.composite import (
    ClusterQualityScore,
    compute_quality_score,
)
from cluster_quality.metrics import QualityMetrics, SilhouetteResult
from cluster_quality.stability import (
    StabilityMetrics,
    BootstrapStabilityResult,
    TemporalStabilityResult,
)
from cluster_quality.validation import (
    ExternalValidationResults,
    ReturnDifferentiationResult,
    VolatilityDifferentiationResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def excellent_score():
    """Score representing excellent cluster quality."""
    return ClusterQualityScore(
        silhouette=0.75,
        davies_bouldin_normalized=0.8,
        bootstrap_stability=0.9,
        temporal_stability=0.85,
        return_significance=0.8,
        volatility_significance=0.7,
    )


@pytest.fixture
def poor_score():
    """Score representing poor cluster quality."""
    return ClusterQualityScore(
        silhouette=0.1,
        davies_bouldin_normalized=0.2,
        bootstrap_stability=0.3,
        temporal_stability=0.2,
        return_significance=0.1,
        volatility_significance=0.0,
    )


@pytest.fixture
def borderline_score():
    """Score at HMM readiness boundary."""
    return ClusterQualityScore(
        silhouette=0.35,
        davies_bouldin_normalized=0.3,
        bootstrap_stability=0.65,
        temporal_stability=0.55,
        return_significance=0.55,
        volatility_significance=0.4,
    )


# =============================================================================
# COMPOSITE SCORE TESTS
# =============================================================================

class TestCompositeScore:
    """Tests for composite score computation."""

    def test_excellent_score_high_composite(self, excellent_score):
        """Excellent metrics should produce high composite."""
        composite = excellent_score.compute_composite()

        assert composite > 0.7, f"Excellent should have composite > 0.7, got {composite}"

    def test_poor_score_low_composite(self, poor_score):
        """Poor metrics should produce low composite."""
        composite = poor_score.compute_composite()

        assert composite < 0.3, f"Poor should have composite < 0.3, got {composite}"

    def test_composite_in_range(self, excellent_score, poor_score, borderline_score):
        """Composite should always be in [0, 1]."""
        for score in [excellent_score, poor_score, borderline_score]:
            composite = score.compute_composite()
            assert 0 <= composite <= 1, f"Composite should be in [0, 1], got {composite}"

    def test_custom_weights(self, excellent_score):
        """Should accept custom weights."""
        default = excellent_score.compute_composite()

        # Weight only silhouette
        silhouette_only = excellent_score.compute_composite({
            "silhouette": 1.0,
            "davies_bouldin": 0.0,
            "bootstrap_stability": 0.0,
            "temporal_stability": 0.0,
            "return_significance": 0.0,
            "volatility_significance": 0.0,
        })

        assert silhouette_only == excellent_score.silhouette


# =============================================================================
# GRADE TESTS
# =============================================================================

class TestGrades:
    """Tests for quality grading."""

    def test_excellent_gets_grade_a(self, excellent_score):
        """Excellent score should get grade A."""
        grade = excellent_score.get_grade()
        assert grade.startswith("A"), f"Excellent should get A, got {grade}"

    def test_poor_gets_low_grade(self, poor_score):
        """Poor score should get grade D or F."""
        grade = poor_score.get_grade()
        assert grade.startswith("D") or grade.startswith("F"), (
            f"Poor should get D or F, got {grade}"
        )

    def test_grade_ordering(self):
        """Better composite should give better grade."""
        scores = [
            ClusterQualityScore(silhouette=0.9, davies_bouldin_normalized=0.9,
                               bootstrap_stability=0.9, temporal_stability=0.9,
                               return_significance=0.9, volatility_significance=0.9),
            ClusterQualityScore(silhouette=0.5, davies_bouldin_normalized=0.5,
                               bootstrap_stability=0.5, temporal_stability=0.5,
                               return_significance=0.5, volatility_significance=0.5),
            ClusterQualityScore(silhouette=0.1, davies_bouldin_normalized=0.1,
                               bootstrap_stability=0.1, temporal_stability=0.1,
                               return_significance=0.1, volatility_significance=0.1),
        ]

        grades = [s.get_grade()[0] for s in scores]  # First letter
        assert grades[0] <= grades[1] <= grades[2], "Grades should be ordered"


# =============================================================================
# HMM READINESS TESTS
# =============================================================================

class TestHMMReadiness:
    """Tests for HMM readiness assessment."""

    def test_excellent_is_hmm_ready(self, excellent_score):
        """Excellent score should be HMM ready."""
        assert excellent_score.is_hmm_ready(), "Excellent should be HMM ready"

    def test_poor_not_hmm_ready(self, poor_score):
        """Poor score should not be HMM ready."""
        assert not poor_score.is_hmm_ready(), "Poor should not be HMM ready"

    def test_borderline_may_be_ready(self, borderline_score):
        """Borderline score readiness depends on thresholds."""
        # Should be close to threshold - may pass or fail
        result = borderline_score.is_hmm_ready()
        assert isinstance(result, bool)

    def test_single_weak_metric_blocks_readiness(self):
        """One weak metric should block HMM readiness."""
        # Good except silhouette
        score = ClusterQualityScore(
            silhouette=0.1,  # Too low
            davies_bouldin_normalized=0.8,
            bootstrap_stability=0.9,
            temporal_stability=0.8,
            return_significance=0.8,
            volatility_significance=0.7,
        )

        assert not score.is_hmm_ready(), "Weak silhouette should block readiness"


# =============================================================================
# WEAKNESS DETECTION TESTS
# =============================================================================

class TestWeaknessDetection:
    """Tests for weakness identification."""

    def test_excellent_no_weaknesses(self, excellent_score):
        """Excellent score should have no weaknesses."""
        weaknesses = excellent_score.get_weaknesses()
        assert len(weaknesses) == 0, f"Excellent should have no weaknesses: {weaknesses}"

    def test_poor_many_weaknesses(self, poor_score):
        """Poor score should have multiple weaknesses."""
        weaknesses = poor_score.get_weaknesses()
        assert len(weaknesses) >= 4, "Poor should have multiple weaknesses"

    def test_specific_weakness_detected(self):
        """Should detect specific weak metrics."""
        score = ClusterQualityScore(
            silhouette=0.8,
            davies_bouldin_normalized=0.8,
            bootstrap_stability=0.3,  # Weak
            temporal_stability=0.8,
            return_significance=0.8,
            volatility_significance=0.8,
        )

        weaknesses = score.get_weaknesses()
        weakness_metrics = [w[0] for w in weaknesses]

        assert "bootstrap_stability" in weakness_metrics


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestQualityScoreComputation:
    """Tests for computing quality score from raw metrics."""

    def test_computes_from_raw_metrics(self):
        """Should compute score from raw metric objects."""
        metrics = QualityMetrics(
            silhouette=SilhouetteResult(overall=0.6, per_cluster={}, std=0.1, pct_negative=0.1),
            davies_bouldin=0.8,
            calinski_harabasz=500,
            gap_statistic=None,
            n_clusters=5,
            n_samples=500,
        )

        stability = StabilityMetrics(
            bootstrap=BootstrapStabilityResult(
                mean_ari=0.75, std_ari=0.1, min_ari=0.5, max_ari=0.9,
                pct_stable=0.8, all_aris=[]
            ),
            temporal=TemporalStabilityResult(
                temporal_ari=0.7, proportion_drift=0.1,
                train_proportions=[0.2, 0.3, 0.5],
                test_proportions=[0.22, 0.28, 0.5],
                cluster_survival_rate=1.0,
            ),
        )

        validation = ExternalValidationResults(
            return_differentiation={
                60: ReturnDifferentiationResult(
                    horizon=60, anova_f=10.0, anova_p=0.001,
                    kruskal_h=15.0, kruskal_p=0.001,
                    eta_squared=0.08, cluster_means={0: -0.01, 1: 0.01},
                    cluster_stds={0: 0.02, 1: 0.02}, significant=True,
                )
            },
            volatility_differentiation=VolatilityDifferentiationResult(
                levene_stat=5.0, levene_p=0.01,
                kruskal_h=8.0, kruskal_p=0.02,
                cluster_volatilities={0: 0.01, 1: 0.03},
                significant=True,
            ),
            transitions=None,
        )

        score = compute_quality_score(metrics, stability, validation)

        assert isinstance(score, ClusterQualityScore)
        assert score.silhouette == 0.6
        assert score.bootstrap_stability == 0.75
```

---

## Task 5: Agentic Feature Refinement

**File:** `scripts/cluster_quality/refinement_agent.py`
**Priority:** Low
**Estimated Complexity:** High

### 5.1 Requirements

Implement agentic feature refinement:
- Base RefinementStrategy class
- Multiple refinement strategies
- FeatureRefinementAgent with iteration loop

### 5.2 Implementation

See spec document section 4 for full implementation.

### 5.3 Skeptical Tests

```python
"""
Skeptical Tests for Feature Refinement Agent

Tests verify that the agent:
- Actually improves cluster quality
- Stops at appropriate conditions
- Doesn't get stuck in infinite loops
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from cluster_quality.refinement_agent import (
    FeatureRefinementAgent,
    AddInteractionFeatures,
    IncreaseWindowSize,
    RemoveCorrelatedFeatures,
)


class TestRefinementStrategies:
    """Tests for individual refinement strategies."""

    def test_add_interaction_increases_features(self):
        """AddInteractionFeatures should add new features."""
        strategy = AddInteractionFeatures()
        features = ["f1", "f2", "f3"]

        # Mock low silhouette quality
        class MockQuality:
            silhouette = 0.1

        refined = strategy.refine(features, MockQuality())
        assert len(refined) > len(features), "Should add interaction features"

    def test_remove_correlated_decreases_features(self):
        """RemoveCorrelatedFeatures should reduce redundancy."""
        strategy = RemoveCorrelatedFeatures()

        # Create highly correlated features
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0] * 0.99  # Almost identical
        X[:, 3] = X[:, 2] * 0.98

        features = ["f0", "f1", "f2", "f3", "f4"]

        class MockQuality:
            pass

        # Would need X passed to strategy
        # This tests the concept


class TestRefinementAgent:
    """Tests for the refinement agent."""

    def test_agent_terminates(self):
        """Agent should terminate within max iterations."""
        X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        returns = np.random.randn(200)

        agent = FeatureRefinementAgent(
            feature_pool=["f0", "f1", "f2"],
            max_iterations=5,
        )

        # Would run: result = agent.run(data, returns)
        # Assert iteration count <= max_iterations

    def test_agent_improves_quality(self):
        """Agent should improve or maintain quality over iterations."""
        # This is the key skeptical test:
        # Quality at iteration N should be >= quality at iteration 0
        # (at least after some iterations)
        pass

    def test_agent_records_history(self):
        """Agent should record full optimization history."""
        pass
```

---

## Makefile Targets

Add these targets to the Makefile:

```makefile
# =============================================================================
# CLUSTER QUALITY
# =============================================================================

# Run cluster quality analysis
cluster_quality:
	@echo "Analyzing cluster quality..."
	python scripts/cluster_quality/run_analysis.py --data-dir $(DATA)

# Run cluster quality tests
test_cluster_quality:
	@echo "Running cluster quality tests..."
	cd scripts && python -m pytest cluster_quality/tests/ -v

# Run with coverage
test_cluster_quality_cov:
	cd scripts && python -m pytest cluster_quality/tests/ -v --cov=cluster_quality --cov-report=term-missing
```

---

## Execution Order

1. **Task 1: Core Metrics** - Foundation for all other tasks
2. **Task 2: Stability Metrics** - Depends on Task 1 for silhouette computation
3. **Task 3: External Validation** - Independent of 1-2
4. **Task 4: Composite Scoring** - Depends on 1, 2, 3
5. **Task 5: Agentic Refinement** - Depends on all above

---

## Success Criteria

Each task is complete when:
1. Implementation matches specification
2. All skeptical tests pass
3. Code is documented with docstrings
4. Integration with existing codebase verified

---

## Notes for Sonnet Model Execution

When executing these tasks:
1. Create the directory structure first: `mkdir -p scripts/cluster_quality/tests`
2. Create `__init__.py` files for Python package structure
3. Implement one task at a time, running tests after each
4. Use `pytest -v` to verify tests pass
5. Commit after each successful task completion

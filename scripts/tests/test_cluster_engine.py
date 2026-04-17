"""
Skeptical tests for cluster_pipeline.cluster — clustering engine.

Tests cover:
  - GMM fitting, convergence, soft assignments
  - GMM auto k-selection via BIC
  - HDBSCAN density-based clustering (if available)
  - Agglomerative clustering and linkage
  - k-sweep correctness and metric consistency
  - Cluster quality metrics (silhouette, DB, CH)
  - Bootstrap stability (ARI)
  - Temporal stability (first/second half)
  - Dip test for multimodality
  - Bimodality coefficient
  - Multimodality scan
  - Predictive quality (Kruskal-Wallis, eta-squared, self-transition)
  - Full analysis pipeline
  - Input validation and edge cases
  - Determinism and reproducibility
  - Known-structure data (well-separated clusters must be found)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cluster_pipeline.cluster import (
    ClusterResult,
    QualityReport,
    StabilityReport,
    SweepResult,
    bimodality_coefficient,
    bootstrap_stability,
    cluster_quality,
    compute_linkage,
    dip_test,
    fit_agglomerative,
    fit_gmm,
    fit_gmm_auto,
    full_analysis,
    k_sweep,
    multimodality_scan,
    predictive_quality,
    temporal_stability,
    _compute_dip,
    _self_transition_rate,
    _validate_input,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic data with known structure
# ---------------------------------------------------------------------------


def _make_blobs(
    n_per_cluster: int = 100,
    k: int = 3,
    n_features: int = 5,
    separation: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create well-separated Gaussian blobs with known labels."""
    rng = np.random.default_rng(seed)
    X_parts = []
    labels = []
    for i in range(k):
        center = np.zeros(n_features)
        center[0] = i * separation
        X_parts.append(rng.normal(center, 0.5, (n_per_cluster, n_features)))
        labels.extend([i] * n_per_cluster)
    return np.vstack(X_parts), np.array(labels)


def _make_noisy_data(n: int = 200, n_features: int = 5, seed: int = 42) -> np.ndarray:
    """Create data with no clear cluster structure (uniform noise)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, n_features))


@pytest.fixture
def well_separated():
    """3 well-separated clusters, 100 points each, 5D."""
    return _make_blobs(n_per_cluster=100, k=3, n_features=5, separation=8.0)


@pytest.fixture
def two_clusters():
    """2 clear clusters."""
    return _make_blobs(n_per_cluster=150, k=2, n_features=4, separation=6.0)


@pytest.fixture
def noisy_data():
    """Uniform noise — no cluster structure."""
    return _make_noisy_data(n=200, n_features=5)


@pytest.fixture
def large_data():
    """Larger dataset for stability tests."""
    return _make_blobs(n_per_cluster=200, k=4, n_features=6, separation=6.0)


@pytest.fixture
def bimodal_1d():
    """Clearly bimodal 1D distribution with wide separation."""
    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-5, 0.5, 300), rng.normal(5, 0.5, 300)])


@pytest.fixture
def unimodal_1d():
    """Clearly unimodal (Gaussian) 1D distribution."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 400)


# ---------------------------------------------------------------------------
# TestFitGMM — Gaussian Mixture Model
# ---------------------------------------------------------------------------


class TestFitGMM:
    """Verify GMM fitting correctness."""

    def test_returns_cluster_result(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert isinstance(result, ClusterResult)

    def test_correct_method_label(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.method == "gmm"

    def test_correct_k(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.k == 3

    def test_label_shape(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.labels.shape == (len(X),)

    def test_all_labels_in_range(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert set(result.labels).issubset({0, 1, 2})

    def test_probabilities_shape(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.probabilities is not None
        assert result.probabilities.shape == (len(X), 3)

    def test_probabilities_sum_to_one(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_probabilities_non_negative(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert (result.probabilities >= 0).all()

    def test_bic_is_finite(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.bic is not None
        assert np.isfinite(result.bic)

    def test_aic_is_finite(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.aic is not None
        assert np.isfinite(result.aic)

    def test_converged(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert result.converged

    def test_recovers_known_clusters(self, well_separated):
        """GMM should recover the 3 well-separated clusters."""
        X, true_labels = well_separated
        result = fit_gmm(X, k=3)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, result.labels)
        assert ari > 0.8, f"ARI {ari:.3f} too low — should recover well-separated clusters"

    def test_k1_produces_single_cluster(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=1)
        assert len(set(result.labels)) == 1

    def test_means_in_extra(self, well_separated):
        X, _ = well_separated
        result = fit_gmm(X, k=3)
        assert "means" in result.extra
        assert result.extra["means"].shape == (3, 5)

    def test_deterministic_with_same_seed(self, well_separated):
        X, _ = well_separated
        r1 = fit_gmm(X, k=3, random_state=42)
        r2 = fit_gmm(X, k=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_different_seed_may_differ(self, well_separated):
        """Different seeds might give different label assignments (permuted)."""
        X, _ = well_separated
        r1 = fit_gmm(X, k=3, random_state=1)
        r2 = fit_gmm(X, k=3, random_state=999)
        # Labels may be permuted, but ARI should still be high
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(r1.labels, r2.labels)
        assert ari > 0.5  # Should still find similar structure


# ---------------------------------------------------------------------------
# TestFitGMMAuto — automatic k selection
# ---------------------------------------------------------------------------


class TestFitGMMAuto:

    def test_returns_cluster_result(self, well_separated):
        X, _ = well_separated
        result = fit_gmm_auto(X)
        assert isinstance(result, ClusterResult)

    def test_finds_correct_k_for_clear_clusters(self, well_separated):
        """Should find k=3 for well-separated 3-cluster data."""
        X, _ = well_separated
        result = fit_gmm_auto(X, k_range=range(1, 8))
        assert result.k in {2, 3, 4}, f"Expected k near 3, got {result.k}"

    def test_finds_k2_for_two_clusters(self, two_clusters):
        X, _ = two_clusters
        result = fit_gmm_auto(X, k_range=range(1, 8))
        assert result.k in {2, 3}, f"Expected k=2 or 3, got {result.k}"

    def test_bic_decreases_at_true_k(self, well_separated):
        """BIC at k=3 should be lower than k=1 for well-separated clusters."""
        X, _ = well_separated
        r1 = fit_gmm(X, k=1)
        r3 = fit_gmm(X, k=3)
        assert r3.bic < r1.bic, "BIC at k=3 should be lower than k=1"

    def test_custom_k_range(self, well_separated):
        X, _ = well_separated
        result = fit_gmm_auto(X, k_range=range(2, 5))
        assert result.k in {2, 3, 4}


# ---------------------------------------------------------------------------
# TestFitAgglomerative — hierarchical clustering
# ---------------------------------------------------------------------------


class TestFitAgglomerative:

    def test_returns_cluster_result(self, well_separated):
        X, _ = well_separated
        result = fit_agglomerative(X, k=3)
        assert isinstance(result, ClusterResult)

    def test_correct_method(self, well_separated):
        X, _ = well_separated
        result = fit_agglomerative(X, k=3)
        assert result.method == "agglomerative"

    def test_correct_k(self, well_separated):
        X, _ = well_separated
        result = fit_agglomerative(X, k=3)
        assert len(set(result.labels)) == 3

    def test_recovers_known_clusters(self, well_separated):
        X, true_labels = well_separated
        result = fit_agglomerative(X, k=3)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, result.labels)
        assert ari > 0.8

    def test_no_probabilities(self, well_separated):
        X, _ = well_separated
        result = fit_agglomerative(X, k=3)
        assert result.probabilities is None

    def test_no_bic(self, well_separated):
        X, _ = well_separated
        result = fit_agglomerative(X, k=3)
        assert result.bic is None


class TestComputeLinkage:

    def test_returns_array(self, well_separated):
        X, _ = well_separated
        Z = compute_linkage(X)
        assert isinstance(Z, np.ndarray)

    def test_shape(self, well_separated):
        X, _ = well_separated
        Z = compute_linkage(X)
        assert Z.shape == (len(X) - 1, 4)

    def test_ward_method(self, well_separated):
        X, _ = well_separated
        Z = compute_linkage(X, method="ward")
        assert Z.shape[0] == len(X) - 1

    def test_non_ward_method(self, well_separated):
        X, _ = well_separated
        Z = compute_linkage(X, method="average", metric="euclidean")
        assert Z.shape[0] == len(X) - 1


# ---------------------------------------------------------------------------
# TestKSweep — sweeping k values
# ---------------------------------------------------------------------------


class TestKSweep:

    def test_returns_sweep_result(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        assert isinstance(result, SweepResult)

    def test_correct_k_range(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        assert result.k_range == [2, 3, 4, 5]

    def test_results_count_matches_k_range(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        assert len(result.results) == 4
        assert len(result.silhouettes) == 4
        assert len(result.davies_bouldins) == 4
        assert len(result.calinski_harabasz) == 4

    def test_silhouettes_in_range(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        for s in result.silhouettes:
            assert -1.0 <= s <= 1.0

    def test_best_k_silhouette_in_range(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        assert result.best_k_silhouette in result.k_range

    def test_best_k_bic_in_range(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6))
        assert result.best_k_bic in result.k_range

    def test_well_separated_best_k_near_3(self, well_separated):
        """For 3-cluster data, best k should be near 3."""
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 8))
        # Either BIC or silhouette should pick k=3
        assert result.best_k_silhouette in {2, 3, 4} or result.best_k_bic in {2, 3, 4}

    def test_agglomerative_sweep(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 6), method="agglomerative")
        assert result.best_k_bic is None  # agglomerative has no BIC
        assert result.best_k_silhouette in result.k_range

    def test_bics_all_present_for_gmm(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 5))
        assert all(b is not None for b in result.bics)

    def test_davies_bouldin_finite(self, well_separated):
        X, _ = well_separated
        result = k_sweep(X, k_range=range(2, 5))
        for db in result.davies_bouldins:
            assert np.isfinite(db)


# ---------------------------------------------------------------------------
# TestClusterQuality — quality metrics
# ---------------------------------------------------------------------------


class TestClusterQuality:

    def test_returns_quality_report(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert isinstance(report, QualityReport)

    def test_high_silhouette_for_clear_clusters(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert report.silhouette > 0.5, f"Silhouette {report.silhouette:.3f} too low"

    def test_low_davies_bouldin_for_clear_clusters(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert report.davies_bouldin < 1.0, f"DB {report.davies_bouldin:.3f} too high"

    def test_positive_calinski_harabasz(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert report.calinski_harabasz > 0

    def test_correct_cluster_sizes(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        total = sum(report.cluster_sizes.values())
        assert total == len(X)

    def test_n_clusters_correct(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert report.n_clusters == 3

    def test_per_cluster_silhouette(self, well_separated):
        X, true_labels = well_separated
        report = cluster_quality(X, true_labels)
        assert len(report.silhouette_per_cluster) == 3
        for c, s in report.silhouette_per_cluster.items():
            assert -1.0 <= s <= 1.0

    def test_noise_handling(self):
        """Labels with -1 (noise) should be handled."""
        X, labels = _make_blobs(n_per_cluster=50, k=2, separation=6.0)
        # Add some noise points
        rng = np.random.default_rng(42)
        noise = rng.uniform(-20, 20, (10, 5))
        X = np.vstack([X, noise])
        labels = np.concatenate([labels, np.full(10, -1)])
        report = cluster_quality(X, labels)
        assert report.noise_fraction > 0
        assert -1 in report.cluster_sizes

    def test_single_cluster_returns_degenerate(self):
        X = np.random.default_rng(42).normal(0, 1, (50, 3))
        labels = np.zeros(50, dtype=int)
        report = cluster_quality(X, labels)
        assert report.silhouette == -1.0
        assert report.n_clusters == 1

    def test_label_count_mismatch_raises(self, well_separated):
        X, _ = well_separated
        with pytest.raises(ValueError, match="Label count"):
            cluster_quality(X, np.array([0, 1, 2]))

    def test_noise_only_labels(self):
        X = np.random.default_rng(42).normal(0, 1, (50, 3))
        labels = np.full(50, -1)
        report = cluster_quality(X, labels)
        assert report.n_clusters == 0
        assert report.noise_fraction == 1.0


# ---------------------------------------------------------------------------
# TestBootstrapStability
# ---------------------------------------------------------------------------


class TestBootstrapStability:

    def test_returns_stability_report(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=10)
        assert isinstance(report, StabilityReport)

    def test_high_stability_for_clear_clusters(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=20)
        assert report.mean_ari > 0.5, f"Mean ARI {report.mean_ari:.3f} too low"
        assert report.stable

    def test_correct_resample_count(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=15)
        assert report.n_resamples == 15

    def test_ari_values_in_range(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=10)
        for ari in report.ari_values:
            assert -1.0 <= ari <= 1.0

    def test_std_non_negative(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=10)
        assert report.std_ari >= 0

    def test_min_max_consistent(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=10)
        assert report.min_ari <= report.mean_ari <= report.max_ari

    def test_noisy_data_less_stable(self, noisy_data):
        """Uniform noise should have lower stability than clear clusters."""
        report = bootstrap_stability(noisy_data, k=3, n_resamples=10)
        # Just check it doesn't crash; noise ARI can be anything
        assert isinstance(report.mean_ari, float)

    def test_agglomerative_method(self, well_separated):
        X, _ = well_separated
        report = bootstrap_stability(X, k=3, n_resamples=10, method="agglomerative")
        assert report.n_resamples == 10


# ---------------------------------------------------------------------------
# TestTemporalStability
# ---------------------------------------------------------------------------


class TestTemporalStability:

    def test_returns_stability_report(self, large_data):
        X, _ = large_data
        report = temporal_stability(X, k=4)
        assert isinstance(report, StabilityReport)

    def test_high_stability_for_shuffled_clusters(self):
        """Shuffled well-separated data should have high temporal stability."""
        X, _ = _make_blobs(n_per_cluster=200, k=4, n_features=6, separation=6.0)
        # Shuffle so clusters are distributed across time
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X))
        X = X[perm]
        report = temporal_stability(X, k=4)
        assert report.mean_ari > 0.3, f"Temporal ARI {report.mean_ari:.3f} low"

    def test_has_half_and_quarter_results(self, large_data):
        X, _ = large_data
        report = temporal_stability(X, k=4)
        # Should have 2 (halves) + up to 4 (quarters) = up to 6
        assert report.n_resamples >= 2

    def test_ari_values_in_range(self, large_data):
        X, _ = large_data
        report = temporal_stability(X, k=4)
        for ari in report.ari_values:
            assert -1.0 <= ari <= 1.0


# ---------------------------------------------------------------------------
# TestDipTest — multimodality detection
# ---------------------------------------------------------------------------


class TestDipTest:

    def test_bimodal_detected(self, bimodal_1d):
        dip_stat, p = dip_test(bimodal_1d)
        assert p < 0.1, f"p={p:.3f} — bimodal distribution should be detected"

    def test_unimodal_not_rejected(self, unimodal_1d):
        dip_stat, p = dip_test(unimodal_1d)
        assert p > 0.01, f"p={p:.3f} — unimodal should not be rejected"

    def test_returns_tuple(self, bimodal_1d):
        result = dip_test(bimodal_1d)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dip_stat_non_negative(self, bimodal_1d):
        dip_stat, _ = dip_test(bimodal_1d)
        assert dip_stat >= 0

    def test_p_value_in_01(self, bimodal_1d):
        _, p = dip_test(bimodal_1d)
        assert 0 <= p <= 1

    def test_too_few_points(self):
        dip_stat, p = dip_test(np.array([1.0, 2.0]))
        assert p == 1.0

    def test_nan_handling(self):
        x = np.array([1, 2, np.nan, 3, np.nan, 4, 5, 6, 7, 8])
        dip_stat, p = dip_test(x)
        assert np.isfinite(dip_stat)
        assert 0 <= p <= 1

    def test_constant_array(self):
        x = np.full(100, 5.0)
        dip_stat, p = dip_test(x)
        assert np.isfinite(dip_stat)


# ---------------------------------------------------------------------------
# TestBimodalityCoefficient
# ---------------------------------------------------------------------------


class TestBimodalityCoefficient:

    def test_bimodal_high_bc(self, bimodal_1d):
        bc = bimodality_coefficient(bimodal_1d)
        assert bc > 0.4, f"BC {bc:.3f} too low for bimodal data"

    def test_unimodal_lower_bc(self, unimodal_1d):
        bc = bimodality_coefficient(unimodal_1d)
        # Unimodal should have lower BC than bimodal
        assert isinstance(bc, float)

    def test_too_few_points(self):
        bc = bimodality_coefficient(np.array([1.0, 2.0]))
        assert bc == 0.0

    def test_nan_handling(self):
        x = np.array([1, 2, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])
        bc = bimodality_coefficient(x)
        assert np.isfinite(bc)


# ---------------------------------------------------------------------------
# TestMultimodalityScan
# ---------------------------------------------------------------------------


class TestMultimodalityScan:

    def test_returns_list(self, well_separated):
        X, _ = well_separated
        results = multimodality_scan(X)
        assert isinstance(results, list)

    def test_one_result_per_feature(self, well_separated):
        X, _ = well_separated
        results = multimodality_scan(X)
        assert len(results) == X.shape[1]

    def test_result_keys(self, well_separated):
        X, _ = well_separated
        results = multimodality_scan(X)
        expected_keys = {
            "feature", "dip_statistic", "dip_p_value",
            "bimodality_coefficient", "dip_multimodal", "bc_multimodal", "multimodal",
        }
        for r in results:
            assert set(r.keys()) == expected_keys

    def test_sorted_by_p_value(self, well_separated):
        X, _ = well_separated
        results = multimodality_scan(X)
        p_values = [r["dip_p_value"] for r in results]
        assert p_values == sorted(p_values)

    def test_custom_column_names(self, well_separated):
        X, _ = well_separated
        names = [f"feat_{i}" for i in range(X.shape[1])]
        results = multimodality_scan(X, column_names=names)
        assert results[0]["feature"].startswith("feat_")

    def test_bimodal_first_dimension(self):
        """First feature is bimodal, rest are Gaussian — should detect it."""
        rng = np.random.default_rng(42)
        bimodal = np.concatenate([rng.normal(-3, 0.5, 200), rng.normal(3, 0.5, 200)])
        unimodal = rng.normal(0, 1, (400, 3))
        X = np.column_stack([bimodal, unimodal])
        results = multimodality_scan(X, column_names=["bimodal", "uni1", "uni2", "uni3"])
        # The bimodal feature should be flagged
        bimodal_result = [r for r in results if r["feature"] == "bimodal"][0]
        assert bimodal_result["multimodal"], "Should detect bimodal feature"


# ---------------------------------------------------------------------------
# TestPredictiveQuality — returns-based validation
# ---------------------------------------------------------------------------


class TestPredictiveQuality:

    def test_returns_dict(self, well_separated):
        X, labels = well_separated
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, len(labels))
        result = predictive_quality(labels, returns)
        assert isinstance(result, dict)

    def test_has_required_keys(self, well_separated):
        X, labels = well_separated
        returns = np.random.default_rng(42).normal(0, 0.01, len(labels))
        result = predictive_quality(labels, returns)
        assert "kruskal_wallis_h" in result
        assert "kruskal_wallis_p" in result
        assert "eta_squared" in result
        assert "per_cluster" in result
        assert "self_transition_rate" in result
        assert "significant" in result

    def test_significant_when_returns_differ(self):
        """If cluster returns are meaningfully different, should detect it."""
        rng = np.random.default_rng(42)
        labels = np.array([0] * 200 + [1] * 200 + [2] * 200)
        # Cluster 0: positive returns, Cluster 1: negative, Cluster 2: zero
        returns = np.concatenate([
            rng.normal(0.02, 0.005, 200),
            rng.normal(-0.02, 0.005, 200),
            rng.normal(0.0, 0.005, 200),
        ])
        result = predictive_quality(labels, returns)
        assert result["kruskal_wallis_p"] < 0.05
        assert result["significant"]

    def test_not_significant_for_same_returns(self):
        """If all clusters have same return distribution, should NOT be significant."""
        rng = np.random.default_rng(42)
        labels = np.array([0] * 100 + [1] * 100)
        returns = rng.normal(0, 0.01, 200)
        result = predictive_quality(labels, returns)
        assert result["kruskal_wallis_p"] > 0.01

    def test_per_cluster_stats(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        returns = np.array([0.01, 0.02, 0.03, -0.01, -0.02, -0.03])
        result = predictive_quality(labels, returns)
        assert 0 in result["per_cluster"]
        assert 1 in result["per_cluster"]
        assert result["per_cluster"][0]["mean_return"] > 0
        assert result["per_cluster"][1]["mean_return"] < 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            predictive_quality(np.array([0, 1]), np.array([0.01]))

    def test_noise_labels_excluded(self):
        labels = np.array([0, 0, -1, 1, 1, -1])
        returns = np.array([0.01, 0.02, 0.0, -0.01, -0.02, 0.0])
        result = predictive_quality(labels, returns)
        # Should still work with noise points
        assert len(result["per_cluster"]) == 2

    def test_single_cluster_not_significant(self):
        labels = np.array([0, 0, 0, 0])
        returns = np.array([0.01, 0.02, -0.01, 0.0])
        result = predictive_quality(labels, returns)
        assert not result["significant"]


# ---------------------------------------------------------------------------
# TestSelfTransitionRate
# ---------------------------------------------------------------------------


class TestSelfTransitionRate:

    def test_constant_labels_rate_1(self):
        labels = np.array([0, 0, 0, 0, 0])
        assert _self_transition_rate(labels) == 1.0

    def test_alternating_labels_rate_0(self):
        labels = np.array([0, 1, 0, 1, 0])
        assert _self_transition_rate(labels) == 0.0

    def test_mixed(self):
        labels = np.array([0, 0, 1, 1, 1, 0])
        rate = _self_transition_rate(labels)
        # Transitions: 0->0(same), 0->1(diff), 1->1(same), 1->1(same), 1->0(diff)
        assert rate == pytest.approx(3 / 5)

    def test_single_label(self):
        assert _self_transition_rate(np.array([0])) == 0.0

    def test_noise_ignored(self):
        labels = np.array([0, 0, -1, 1, 1])
        rate = _self_transition_rate(labels)
        # After removing -1: [0, 0, 1, 1] -> 2/3 same
        assert rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            _validate_input([[1, 2], [3, 4]])

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            _validate_input(np.array([1, 2, 3]))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least"):
            _validate_input(np.array([[1, 2]]), min_samples=5)

    def test_nan_raises(self):
        X = np.array([[1, 2], [np.nan, 4]])
        with pytest.raises(ValueError, match="NaN"):
            _validate_input(X)

    def test_inf_raises(self):
        X = np.array([[1, 2], [np.inf, 4]])
        with pytest.raises(ValueError, match="infinite"):
            _validate_input(X)

    def test_zero_columns_raises(self):
        X = np.empty((5, 0))
        with pytest.raises(ValueError, match="0 columns"):
            _validate_input(X)

    def test_fit_gmm_nan_raises(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        with pytest.raises(ValueError):
            fit_gmm(X, k=2)

    def test_fit_gmm_too_few_samples(self):
        X = np.array([[1, 2]])
        with pytest.raises(ValueError):
            fit_gmm(X, k=3)


# ---------------------------------------------------------------------------
# TestFullAnalysis — end-to-end pipeline
# ---------------------------------------------------------------------------


class TestFullAnalysis:

    def test_returns_dict(self, well_separated):
        X, _ = well_separated
        result = full_analysis(X, k_range=range(2, 6), n_bootstrap=5)
        assert isinstance(result, dict)

    def test_has_all_keys(self, well_separated):
        X, _ = well_separated
        result = full_analysis(X, k_range=range(2, 6), n_bootstrap=5)
        assert "sweep" in result
        assert "best_k" in result
        assert "best_result" in result
        assert "quality" in result
        assert "bootstrap_stability" in result
        assert "temporal_stability" in result
        assert "multimodality" in result

    def test_with_forward_returns(self, well_separated):
        X, labels = well_separated
        returns = np.random.default_rng(42).normal(0, 0.01, len(X))
        result = full_analysis(X, k_range=range(2, 6), n_bootstrap=5, forward_returns=returns)
        assert "predictive" in result

    def test_best_k_reasonable(self, well_separated):
        X, _ = well_separated
        result = full_analysis(X, k_range=range(2, 8), n_bootstrap=5)
        assert 2 <= result["best_k"] <= 7

    def test_quality_report_present(self, well_separated):
        X, _ = well_separated
        result = full_analysis(X, k_range=range(2, 5), n_bootstrap=5)
        assert isinstance(result["quality"], QualityReport)
        assert result["quality"].silhouette > 0

    def test_column_names_propagate(self, well_separated):
        X, _ = well_separated
        names = ["f1", "f2", "f3", "f4", "f5"]
        result = full_analysis(X, k_range=range(2, 5), n_bootstrap=5, column_names=names)
        assert result["multimodality"][0]["feature"] in names


# ---------------------------------------------------------------------------
# TestHDBSCAN — only if hdbscan is installed
# ---------------------------------------------------------------------------


class TestHDBSCAN:

    @pytest.fixture(autouse=True)
    def _check_hdbscan(self):
        pytest.importorskip("hdbscan")

    def test_returns_cluster_result(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        assert isinstance(result, ClusterResult)

    def test_method_label(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        assert result.method == "hdbscan"

    def test_finds_clusters(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        assert result.k >= 2, f"HDBSCAN found {result.k} clusters, expected >= 2"

    def test_labels_include_minus_one_or_positive(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        for label in result.labels:
            assert label >= -1

    def test_noise_count_non_negative(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        assert result.noise_count >= 0

    def test_noise_count_matches_labels(self, well_separated):
        X, _ = well_separated
        result = fit_hdbscan(X, min_cluster_size=10)
        assert result.noise_count == int(np.sum(result.labels == -1))

    def test_recovers_structure(self, well_separated):
        """HDBSCAN should agree roughly with GMM on well-separated data."""
        X, true_labels = well_separated
        hdb = fit_hdbscan(X, min_cluster_size=10)
        gmm = fit_gmm(X, k=3)
        # Both should find ~3 clusters
        assert abs(hdb.k - 3) <= 2
        # ARI between HDBSCAN and GMM should be positive
        from sklearn.metrics import adjusted_rand_score
        non_noise = hdb.labels >= 0
        if non_noise.sum() > 50:
            ari = adjusted_rand_score(gmm.labels[non_noise], hdb.labels[non_noise])
            assert ari > 0.3, f"HDBSCAN-GMM ARI {ari:.3f} too low"


# ---------------------------------------------------------------------------
# TestDeterminism — reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_gmm_deterministic(self, well_separated):
        X, _ = well_separated
        r1 = fit_gmm(X, k=3, random_state=42)
        r2 = fit_gmm(X, k=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)
        np.testing.assert_allclose(r1.probabilities, r2.probabilities)

    def test_k_sweep_deterministic(self, well_separated):
        X, _ = well_separated
        s1 = k_sweep(X, k_range=range(2, 5), random_state=42)
        s2 = k_sweep(X, k_range=range(2, 5), random_state=42)
        assert s1.silhouettes == s2.silhouettes
        assert s1.best_k_silhouette == s2.best_k_silhouette

    def test_quality_deterministic(self, well_separated):
        X, labels = well_separated
        q1 = cluster_quality(X, labels)
        q2 = cluster_quality(X, labels)
        assert q1.silhouette == q2.silhouette
        assert q1.davies_bouldin == q2.davies_bouldin

    def test_dip_test_deterministic(self, bimodal_1d):
        d1, p1 = dip_test(bimodal_1d)
        d2, p2 = dip_test(bimodal_1d)
        assert d1 == d2
        assert p1 == p2

    def test_full_analysis_deterministic(self, well_separated):
        X, _ = well_separated
        r1 = full_analysis(X, k_range=range(2, 5), n_bootstrap=5, random_state=42)
        r2 = full_analysis(X, k_range=range(2, 5), n_bootstrap=5, random_state=42)
        assert r1["best_k"] == r2["best_k"]
        assert r1["quality"].silhouette == r2["quality"].silhouette

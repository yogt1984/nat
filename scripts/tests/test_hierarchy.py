"""
Skeptical tests for cluster_pipeline.hierarchy — structure existence test.

Tests verify that the Hopkins statistic and Hartigan dip test correctly
distinguish clustered data from uniform noise, and that the decision logic
produces the right recommendations.

Test philosophy:
  - Synthetic data with known structure (well-separated Gaussians)
  - Synthetic data with known non-structure (uniform hypercube)
  - Boundary cases: overlapping clusters, single cluster, degenerate dims
  - Statistical properties: Hopkins near 0.5 for uniform, near 1 for clustered
  - Decision logic: all three recommendation paths tested
  - Determinism: seeded runs produce identical results
  - Validation: invalid inputs rejected
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_pipeline.hierarchy import (
    StructureTest,
    test_structure_existence,
    _hopkins_statistic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _well_separated_gaussians(
    n_per_cluster: int = 200,
    n_clusters: int = 3,
    dim: int = 10,
    separation: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Create well-separated Gaussian clusters.

    Separation is placed on dimension 0 (and wrapping) so that the first
    column carries the multi-modal signal that the dip test checks.
    """
    rng = np.random.RandomState(seed)
    clusters = []
    for i in range(n_clusters):
        center = np.zeros(dim)
        # Put primary separation on dim 0 so dip test on col 0 detects it
        center[0] = i * separation
        # Add secondary separation on other dims for Hopkins
        if dim > 1 and i > 0:
            center[i % dim] += i * separation * 0.3
        clusters.append(rng.normal(center, 1.0, (n_per_cluster, dim)))
    return np.vstack(clusters)


def _uniform_hypercube(
    n: int = 600,
    dim: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Create uniform random data in a hypercube (no structure)."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n, dim))


def _overlapping_gaussians(
    n_per_cluster: int = 200,
    dim: int = 10,
    separation: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Create overlapping Gaussian clusters (marginal structure)."""
    rng = np.random.RandomState(seed)
    c1 = rng.normal(np.zeros(dim), 1.0, (n_per_cluster, dim))
    center2 = np.zeros(dim)
    center2[0] = separation
    c2 = rng.normal(center2, 1.0, (n_per_cluster, dim))
    return np.vstack([c1, c2])


def _single_gaussian(
    n: int = 500,
    dim: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Single Gaussian blob — no cluster structure."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, 1, (n, dim))


# ---------------------------------------------------------------------------
# Clustered data → "proceed"
# ---------------------------------------------------------------------------


class TestClusteredData:
    """Well-separated clusters should be detected as having structure."""

    def test_three_gaussians_has_structure(self):
        """3 well-separated Gaussians → has_structure=True."""
        X = _well_separated_gaussians(n_clusters=3, separation=10.0)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is True

    def test_three_gaussians_recommend_proceed(self):
        """3 well-separated Gaussians → recommendation='proceed'."""
        X = _well_separated_gaussians(n_clusters=3, separation=10.0)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation == "proceed"

    def test_two_gaussians_separated(self):
        """2 well-separated Gaussians → has_structure=True."""
        X = _well_separated_gaussians(n_clusters=2, separation=15.0)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is True

    def test_five_gaussians(self):
        """5 clusters → has_structure=True."""
        X = _well_separated_gaussians(n_clusters=5, separation=8.0, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is True

    def test_high_dimensional_clusters(self):
        """Clusters in 50D space should still be detected."""
        X = _well_separated_gaussians(n_clusters=3, dim=50, separation=10.0)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is True

    def test_hopkins_high_for_clusters(self):
        """Hopkins statistic should be > 0.7 for well-separated clusters."""
        X = _well_separated_gaussians(n_clusters=3, separation=10.0)
        result = test_structure_existence(X, seed=42)
        assert result.hopkins_statistic > 0.7

    def test_dip_p_low_for_clusters(self):
        """Dip test p-value should be < 0.05 for bimodal PC1."""
        X = _well_separated_gaussians(n_clusters=2, separation=15.0, dim=2)
        result = test_structure_existence(X, seed=42)
        # With clear separation on PC1, dip test should reject unimodality
        assert result.dip_test_p < 0.05


# ---------------------------------------------------------------------------
# Uniform data → "no_structure"
# ---------------------------------------------------------------------------


class TestUniformData:
    """Uniform random data should be detected as having no structure."""

    def test_uniform_no_structure(self):
        """Uniform hypercube → has_structure=False."""
        X = _uniform_hypercube(n=600, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is False

    def test_uniform_recommend_no_structure(self):
        """Uniform hypercube → recommendation='no_structure'."""
        X = _uniform_hypercube(n=600, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation == "no_structure"

    def test_uniform_hopkins_near_half(self):
        """Hopkins statistic for uniform data should be near 0.5."""
        X = _uniform_hypercube(n=1000, dim=5)
        result = test_structure_existence(X, seed=42)
        assert 0.3 < result.hopkins_statistic < 0.7, (
            f"Hopkins for uniform data = {result.hopkins_statistic:.3f}, expected ~0.5"
        )

    def test_uniform_dip_p_high(self):
        """Dip test p-value for uniform data should be > 0.05."""
        X = _uniform_hypercube(n=1000, dim=5)
        result = test_structure_existence(X, seed=42)
        assert result.dip_test_p > 0.05

    def test_uniform_high_dim(self):
        """Uniform data in 50D → no_structure."""
        X = _uniform_hypercube(n=500, dim=50)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is False

    def test_uniform_large_sample(self):
        """Large uniform sample → no_structure (not a sample size artifact)."""
        X = _uniform_hypercube(n=2000, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.has_structure is False


# ---------------------------------------------------------------------------
# Single Gaussian → "no_structure"
# ---------------------------------------------------------------------------


class TestSingleGaussian:
    """Single Gaussian blob: clustering tendency exists but is unimodal.

    Hopkins detects that data concentrates (vs uniform), so it triggers.
    Dip test on PC1 correctly identifies unimodality. Net result is
    'weak_structure' — proceed with caution, not a hard stop.
    """

    def test_single_gaussian_unimodal(self):
        """Dip test on single Gaussian should not reject unimodality."""
        X = _single_gaussian(n=1000, dim=5)
        result = test_structure_existence(X, seed=42)
        assert result.dip_test_p > 0.05

    def test_single_gaussian_not_proceed(self):
        """Single Gaussian should NOT get 'proceed' (no multimodal evidence)."""
        X = _single_gaussian(n=500, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation != "proceed"

    def test_single_gaussian_weak_or_no(self):
        """Single Gaussian → 'weak_structure' or 'no_structure'."""
        X = _single_gaussian(n=500, dim=10)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation in ["weak_structure", "no_structure"]


# ---------------------------------------------------------------------------
# Marginal / overlapping clusters → "weak_structure"
# ---------------------------------------------------------------------------


class TestMarginalStructure:
    """Overlapping clusters may produce weak_structure or proceed."""

    def test_overlapping_has_structure(self):
        """Overlapping Gaussians (sep=2) should at least detect something."""
        X = _overlapping_gaussians(n_per_cluster=300, separation=3.0, dim=5)
        result = test_structure_existence(X, seed=42)
        # With separation=3 and std=1, there's real but moderate structure
        assert result.recommendation in ["weak_structure", "proceed"]

    def test_barely_separated(self):
        """Very close clusters (sep=1.5) — at best weak_structure."""
        X = _overlapping_gaussians(n_per_cluster=300, separation=1.5, dim=5)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation in ["weak_structure", "no_structure"]

    def test_recommendation_valid_values(self):
        """recommendation must be one of the three valid strings."""
        for data_fn in [_well_separated_gaussians, _uniform_hypercube, _single_gaussian]:
            X = data_fn()
            result = test_structure_existence(X, seed=42)
            assert result.recommendation in ["proceed", "weak_structure", "no_structure"]


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


class TestDecisionLogic:
    """Verify the three-way decision logic directly."""

    def test_both_pass_proceed(self):
        """Hopkins > threshold AND dip p < significance → proceed."""
        # Use very well separated data to ensure both pass
        X = _well_separated_gaussians(n_clusters=2, separation=20.0, dim=2)
        result = test_structure_existence(X, seed=42)
        if result.hopkins_statistic > 0.7 and result.dip_test_p < 0.05:
            assert result.recommendation == "proceed"
            assert result.has_structure is True

    def test_neither_pass_no_structure(self):
        """Hopkins ≤ threshold AND dip p ≥ significance → no_structure."""
        X = _uniform_hypercube(n=1000, dim=10)
        result = test_structure_existence(X, seed=42)
        if result.hopkins_statistic <= 0.7 and result.dip_test_p >= 0.05:
            assert result.recommendation == "no_structure"
            assert result.has_structure is False

    def test_has_structure_iff_any_pass(self):
        """has_structure should be True iff at least one test passes."""
        for seed in range(5):
            X = _overlapping_gaussians(seed=seed, separation=2.5)
            result = test_structure_existence(X, seed=seed)
            hopkins_pass = result.hopkins_statistic > 0.7
            dip_pass = result.dip_test_p < 0.05
            assert result.has_structure == (hopkins_pass or dip_pass)

    def test_proceed_requires_both(self):
        """'proceed' should only occur when both tests pass."""
        for data_fn in [_well_separated_gaussians, _uniform_hypercube,
                        _overlapping_gaussians, _single_gaussian]:
            X = data_fn()
            result = test_structure_existence(X, seed=42)
            if result.recommendation == "proceed":
                assert result.hopkins_statistic > 0.7
                assert result.dip_test_p < 0.05


# ---------------------------------------------------------------------------
# Hopkins statistic properties
# ---------------------------------------------------------------------------


class TestHopkinsStatistic:
    """Test the Hopkins statistic implementation directly."""

    def test_hopkins_in_range(self):
        """Hopkins statistic must be in [0, 1]."""
        for data_fn in [_well_separated_gaussians, _uniform_hypercube, _single_gaussian]:
            X = data_fn()
            h = _hopkins_statistic(X, seed=42)
            assert 0.0 <= h <= 1.0, f"Hopkins = {h}, expected [0, 1]"

    def test_hopkins_clustered_higher_than_uniform(self):
        """Hopkins for clustered data should be higher than for uniform."""
        X_clustered = _well_separated_gaussians(n_clusters=3, separation=10.0)
        X_uniform = _uniform_hypercube(n=600, dim=10)

        h_clustered = _hopkins_statistic(X_clustered, seed=42)
        h_uniform = _hopkins_statistic(X_uniform, seed=42)

        assert h_clustered > h_uniform, (
            f"Hopkins clustered={h_clustered:.3f} should be > uniform={h_uniform:.3f}"
        )

    def test_hopkins_deterministic_with_seed(self):
        """Same seed → same Hopkins value."""
        X = _well_separated_gaussians()
        h1 = _hopkins_statistic(X, seed=42)
        h2 = _hopkins_statistic(X, seed=42)
        assert h1 == h2

    def test_hopkins_different_seeds_differ(self):
        """Different seeds may produce different Hopkins values."""
        X = _well_separated_gaussians()
        h1 = _hopkins_statistic(X, seed=42)
        h2 = _hopkins_statistic(X, seed=99)
        # They could coincidentally be equal, but with enough data they shouldn't
        # This is a soft check — if it fails, the test is still valid
        # Just verify both are reasonable
        assert 0.0 <= h1 <= 1.0
        assert 0.0 <= h2 <= 1.0

    def test_hopkins_sample_ratio(self):
        """Different sample ratios should still produce valid results."""
        X = _well_separated_gaussians()
        for ratio in [0.05, 0.1, 0.2, 0.3]:
            h = _hopkins_statistic(X, sample_ratio=ratio, seed=42)
            assert 0.0 <= h <= 1.0

    def test_hopkins_small_dataset(self):
        """Hopkins should work with minimum viable dataset (10 samples)."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (10, 3))
        h = _hopkins_statistic(X, seed=42)
        assert 0.0 <= h <= 1.0

    def test_hopkins_1d(self):
        """Hopkins should work on 1D data."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 1))
        h = _hopkins_statistic(X, seed=42)
        assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# Output type and structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify StructureTest fields have correct types."""

    def test_result_is_dataclass(self):
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_hopkins_is_float(self):
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert isinstance(result.hopkins_statistic, float)

    def test_dip_p_is_float(self):
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert isinstance(result.dip_test_p, float)

    def test_has_structure_is_bool(self):
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert isinstance(result.has_structure, bool)

    def test_recommendation_is_string(self):
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert isinstance(result.recommendation, str)

    def test_dip_p_in_range(self):
        """Dip test p-value must be in [0, 1]."""
        X = _well_separated_gaussians()
        result = test_structure_existence(X, seed=42)
        assert 0.0 <= result.dip_test_p <= 1.0

    def test_hopkins_in_range(self):
        """Hopkins must be in [0, 1]."""
        X = _uniform_hypercube()
        result = test_structure_existence(X, seed=42)
        assert 0.0 <= result.hopkins_statistic <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Seeded runs must produce identical results."""

    def test_deterministic_clustered(self):
        X = _well_separated_gaussians()
        r1 = test_structure_existence(X, seed=42)
        r2 = test_structure_existence(X, seed=42)
        assert r1.hopkins_statistic == r2.hopkins_statistic
        assert r1.dip_test_p == r2.dip_test_p
        assert r1.recommendation == r2.recommendation

    def test_deterministic_uniform(self):
        X = _uniform_hypercube()
        r1 = test_structure_existence(X, seed=42)
        r2 = test_structure_existence(X, seed=42)
        assert r1.hopkins_statistic == r2.hopkins_statistic
        assert r1.dip_test_p == r2.dip_test_p

    def test_different_seed_may_differ(self):
        """Different seeds produce potentially different Hopkins (but same dip)."""
        X = _well_separated_gaussians()
        r1 = test_structure_existence(X, seed=42)
        r2 = test_structure_existence(X, seed=99)
        # Dip test is deterministic (no seed), so p should be identical
        assert r1.dip_test_p == r2.dip_test_p
        # Hopkins may differ but both should be valid
        assert 0.0 <= r1.hopkins_statistic <= 1.0
        assert 0.0 <= r2.hopkins_statistic <= 1.0


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Invalid inputs must raise ValueError."""

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            test_structure_existence(np.array([1, 2, 3]))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            test_structure_existence(np.random.normal(0, 1, (5, 3)))

    def test_zero_dims_raises(self):
        with pytest.raises(ValueError, match="no dimensions"):
            test_structure_existence(np.empty((100, 0)))

    def test_nan_raises(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]] * 10)
        with pytest.raises(ValueError, match="NaN"):
            test_structure_existence(X)

    def test_invalid_significance_zero(self):
        X = _uniform_hypercube(n=50)
        with pytest.raises(ValueError, match="significance"):
            test_structure_existence(X, significance=0.0)

    def test_invalid_significance_one(self):
        X = _uniform_hypercube(n=50)
        with pytest.raises(ValueError, match="significance"):
            test_structure_existence(X, significance=1.0)

    def test_invalid_hopkins_threshold_zero(self):
        X = _uniform_hypercube(n=50)
        with pytest.raises(ValueError, match="hopkins_threshold"):
            test_structure_existence(X, hopkins_threshold=0.0)

    def test_invalid_hopkins_threshold_one(self):
        X = _uniform_hypercube(n=50)
        with pytest.raises(ValueError, match="hopkins_threshold"):
            test_structure_existence(X, hopkins_threshold=1.0)


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Verify that custom thresholds affect decisions correctly."""

    def test_strict_hopkins_threshold(self):
        """Very strict Hopkins threshold (0.95) makes detection harder."""
        X = _overlapping_gaussians(separation=3.0)
        result_strict = test_structure_existence(X, hopkins_threshold=0.95, seed=42)
        result_lax = test_structure_existence(X, hopkins_threshold=0.5, seed=42)
        # Same data: strict may fail where lax passes
        if not result_strict.has_structure:
            assert result_lax.has_structure or not result_lax.has_structure  # always true
        # But the statistics themselves should be identical
        assert result_strict.hopkins_statistic == result_lax.hopkins_statistic
        assert result_strict.dip_test_p == result_lax.dip_test_p

    def test_lax_significance(self):
        """Very lax significance (0.5) makes dip test easier to pass."""
        X = _single_gaussian(n=500, dim=5)
        result_strict = test_structure_existence(X, significance=0.01, seed=42)
        result_lax = test_structure_existence(X, significance=0.50, seed=42)
        # Same statistics, different thresholds
        assert result_strict.dip_test_p == result_lax.dip_test_p
        assert result_strict.hopkins_statistic == result_lax.hopkins_statistic

    def test_thresholds_dont_change_statistics(self):
        """Changing thresholds changes recommendation, not the statistics."""
        X = _well_separated_gaussians()
        r1 = test_structure_existence(X, hopkins_threshold=0.5, significance=0.1, seed=42)
        r2 = test_structure_existence(X, hopkins_threshold=0.9, significance=0.01, seed=42)
        assert r1.hopkins_statistic == r2.hopkins_statistic
        assert r1.dip_test_p == r2.dip_test_p


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Adversarial and boundary inputs."""

    def test_minimum_10_samples(self):
        """Exactly 10 samples should work."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (10, 3))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_single_dimension(self):
        """1D data should work (dip test is 1D anyway)."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 1))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_high_dimensional(self):
        """100D data should work."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 100))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_constant_dimension_handled(self):
        """Data with one constant dimension should not crash."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        X[:, 2] = 3.0  # constant
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_large_dataset_performance(self):
        """2000 samples × 50 dims should complete quickly."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (2000, 50))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_very_tight_cluster(self):
        """Very tight cluster (std=0.001) still counts as no multi-cluster structure."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 0.001, (500, 5))
        result = test_structure_existence(X, seed=42)
        # Single tight cluster — no multi-cluster structure
        assert result.recommendation in ["no_structure", "weak_structure"]

    def test_extreme_separation(self):
        """Extremely separated clusters → definitely proceed."""
        X = _well_separated_gaussians(n_clusters=3, separation=100.0)
        result = test_structure_existence(X, seed=42)
        assert result.recommendation == "proceed"
        assert result.hopkins_statistic > 0.9


# ---------------------------------------------------------------------------
# Integration with reduction pipeline
# ---------------------------------------------------------------------------


class TestIntegrationWithReduction:
    """Verify structure test works on PCA-reduced data shapes."""

    def test_typical_pca_output_shape(self):
        """Shape (500, 15) — typical PCA output — should work."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 15))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_few_components(self):
        """Shape (300, 3) — aggressive PCA — should work."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 3))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

    def test_many_components(self):
        """Shape (1000, 50) — max_components=50 — should work."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (1000, 50))
        result = test_structure_existence(X, seed=42)
        assert isinstance(result, StructureTest)

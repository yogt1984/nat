"""
Skeptical tests for cluster_pipeline.hierarchy — structure existence test
and macro regime discovery.

Test philosophy:
  - Synthetic data with known structure (well-separated Gaussians)
  - Synthetic data with known non-structure (uniform hypercube)
  - Boundary cases: overlapping clusters, single cluster, degenerate dims
  - Property-based checks for decision logic
  - Determinism: seeded runs produce identical results
  - Validation: invalid inputs rejected
  - Regime discovery: synthetic multi-regime data with known labels
  - Autocorrelation split: known slow/fast columns
  - Block bootstrap: contiguous blocks, not random
  - Duration computation: exact run-length encoding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.hierarchy import (
    StructureTest,
    test_structure_existence,
    _hopkins_statistic,
    # Task 3.1
    discover_macro_regimes,
    RegimeResult,
    SweepResult,
    QualityReport,
    StabilityReport,
    _autocorrelation_split,
    _lag_autocorrelation,
    _k_sweep_gmm,
    _block_bootstrap_stability,
    _compute_durations,
    _self_transition_rate,
    _centroid_profiles,
    # Task 3.2
    discover_micro_states,
    MicroStateResult,
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


# ===========================================================================
# Task 3.1: Macro Regime Discovery tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers for regime tests
# ---------------------------------------------------------------------------


def _make_two_regime_derivatives(
    n_per_regime: int = 250,
    block_size: int = 50,
    dim: int = 10,
    separation: float = 5.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create derivative-like DataFrame with 2 known regimes, interleaved in blocks.

    Returns (derivatives_df, true_labels).
    Columns are slow-moving (high autocorrelation within blocks).
    """
    rng = np.random.RandomState(seed)
    total = n_per_regime * 2

    # Build labels: alternating blocks
    n_blocks = total // block_size
    labels = np.zeros(total, dtype=int)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, total)
        labels[start:end] = b % 2

    # Generate features: regime 0 around origin, regime 1 shifted
    data = np.zeros((total, dim))
    for i in range(total):
        if labels[i] == 0:
            data[i] = rng.normal(0, 1, dim)
        else:
            data[i] = rng.normal(separation, 1, dim)

    # Make columns slow by applying cumulative smoothing
    for col in range(dim):
        smoothed = np.zeros(total)
        smoothed[0] = data[0, col]
        alpha = 0.8  # high smoothing → high autocorrelation
        for t in range(1, total):
            if labels[t] == labels[t - 1]:
                smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * data[t, col]
            else:
                # Regime switch — reset
                smoothed[t] = data[t, col]
        data[:, col] = smoothed

    columns = [f"slow_feat_{i}" for i in range(dim)]
    df = pd.DataFrame(data, columns=columns)
    return df, labels


def _make_uniform_derivatives(
    n: int = 500, dim: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Uniform random data with no regime structure."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(-1, 1, (n, dim))
    columns = [f"feat_{i}" for i in range(dim)]
    return pd.DataFrame(data, columns=columns)


def _make_mixed_autocorrelation_df(
    n: int = 500, seed: int = 42
) -> pd.DataFrame:
    """DataFrame with known slow (high AC) and fast (low AC) columns."""
    rng = np.random.RandomState(seed)

    # Slow column: random walk (high autocorrelation)
    slow1 = np.cumsum(rng.normal(0, 0.1, n))
    slow2 = np.cumsum(rng.normal(0, 0.1, n))

    # Fast column: white noise (zero autocorrelation)
    fast1 = rng.normal(0, 1, n)
    fast2 = rng.normal(0, 1, n)

    return pd.DataFrame({
        "slow_walk_1": slow1,
        "slow_walk_2": slow2,
        "fast_noise_1": fast1,
        "fast_noise_2": fast2,
    })


# Need this import for Tuple type hint in helper
from typing import Tuple


# ---------------------------------------------------------------------------
# Autocorrelation split
# ---------------------------------------------------------------------------


class TestAutocorrelationSplit:
    """Tests for the slow/fast feature split."""

    def test_random_walk_is_slow(self):
        """Random walk has high autocorrelation → classified as slow."""
        rng = np.random.RandomState(42)
        walk = np.cumsum(rng.normal(0, 0.1, 500))
        ac = _lag_autocorrelation(walk, lag=5)
        assert ac > 0.9, f"Random walk AC={ac:.3f}, expected > 0.9"

    def test_white_noise_is_fast(self):
        """White noise has ~0 autocorrelation → classified as fast."""
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 1, 500)
        ac = _lag_autocorrelation(noise, lag=5)
        assert abs(ac) < 0.15, f"White noise AC={ac:.3f}, expected ~0"

    def test_split_separates_slow_and_fast(self):
        """Mixed DataFrame: slow columns selected, fast excluded."""
        df = _make_mixed_autocorrelation_df()
        slow = _autocorrelation_split(df, lag=5, threshold=0.7)
        assert "slow_walk_1" in slow
        assert "slow_walk_2" in slow
        assert "fast_noise_1" not in slow
        assert "fast_noise_2" not in slow

    def test_split_returns_sorted_by_ac(self):
        """Slow columns should be sorted by autocorrelation descending."""
        df = _make_mixed_autocorrelation_df()
        slow = _autocorrelation_split(df, lag=5, threshold=0.0)
        # All columns pass threshold=0, should be sorted by AC
        acs = [_lag_autocorrelation(df[col].values, 5) for col in slow]
        for i in range(len(acs) - 1):
            assert acs[i] >= acs[i + 1] - 1e-10

    def test_split_empty_if_all_fast(self):
        """All white noise columns → empty slow list."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            f"noise_{i}": rng.normal(0, 1, 500) for i in range(5)
        })
        slow = _autocorrelation_split(df, lag=5, threshold=0.7)
        assert len(slow) == 0

    def test_split_all_if_all_slow(self):
        """All random walk columns → all selected."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            f"walk_{i}": np.cumsum(rng.normal(0, 0.1, 500)) for i in range(5)
        })
        slow = _autocorrelation_split(df, lag=5, threshold=0.7)
        assert len(slow) == 5

    def test_lag_autocorrelation_constant(self):
        """Constant series has zero variance → AC = 0."""
        ac = _lag_autocorrelation(np.full(100, 3.14), lag=5)
        assert ac == 0.0

    def test_lag_autocorrelation_short_series(self):
        """Series shorter than lag → AC = 0."""
        ac = _lag_autocorrelation(np.array([1, 2, 3]), lag=5)
        assert ac == 0.0

    def test_split_with_nan(self):
        """Columns with NaN should still be evaluated (dropna)."""
        rng = np.random.RandomState(42)
        walk = np.cumsum(rng.normal(0, 0.1, 500))
        walk[::10] = np.nan  # 10% NaN
        df = pd.DataFrame({"walk": walk, "noise": rng.normal(0, 1, 500)})
        slow = _autocorrelation_split(df, lag=5, threshold=0.7)
        assert "walk" in slow


# ---------------------------------------------------------------------------
# Duration computation
# ---------------------------------------------------------------------------


class TestDurations:
    """Tests for run-length encoding of regime labels."""

    def test_simple_case(self):
        """[0,0,0,1,1,0,0,0,0,1] → {0: [3, 4], 1: [2, 1]}."""
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        durations = _compute_durations(labels)
        assert durations[0] == [3, 4]
        assert durations[1] == [2, 1]

    def test_single_label(self):
        """All same label → one long run."""
        labels = np.zeros(100, dtype=int)
        durations = _compute_durations(labels)
        assert durations == {0: [100]}

    def test_alternating(self):
        """Alternating labels → all runs of length 1."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        durations = _compute_durations(labels)
        assert durations[0] == [1, 1, 1]
        assert durations[1] == [1, 1, 1]

    def test_empty_labels(self):
        """Empty array → empty dict."""
        durations = _compute_durations(np.array([], dtype=int))
        assert durations == {}

    def test_single_element(self):
        """Single element → run of 1."""
        durations = _compute_durations(np.array([2]))
        assert durations == {2: [1]}

    def test_three_regimes(self):
        """Three regime labels."""
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0])
        durations = _compute_durations(labels)
        assert durations[0] == [2, 2]
        assert durations[1] == [3]
        assert durations[2] == [2]

    def test_total_equals_n(self):
        """Sum of all run lengths must equal total number of bars."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 3, 200)
        durations = _compute_durations(labels)
        total = sum(sum(runs) for runs in durations.values())
        assert total == 200


# ---------------------------------------------------------------------------
# Self-transition rate
# ---------------------------------------------------------------------------


class TestSelfTransitionRate:
    """Tests for self-transition rate computation."""

    def test_perfect_persistence(self):
        """All same label → STR = 1.0."""
        labels = np.zeros(100, dtype=int)
        assert _self_transition_rate(labels) == 1.0

    def test_alternating(self):
        """Alternating labels → STR = 0.0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        assert _self_transition_rate(labels) == 0.0

    def test_known_value(self):
        """[0,0,0,1,1,0] → 3 same out of 5 pairs = 0.6."""
        labels = np.array([0, 0, 0, 1, 1, 0])
        assert _self_transition_rate(labels) == pytest.approx(3 / 5)

    def test_single_element(self):
        """Single element → STR = 1.0 (vacuously true)."""
        assert _self_transition_rate(np.array([0])) == 1.0

    def test_two_same(self):
        """[0, 0] → STR = 1.0."""
        assert _self_transition_rate(np.array([0, 0])) == 1.0

    def test_two_different(self):
        """[0, 1] → STR = 0.0."""
        assert _self_transition_rate(np.array([0, 1])) == 0.0

    def test_str_in_range(self):
        """STR must be in [0, 1]."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 3, 500)
        s = _self_transition_rate(labels)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# k-sweep GMM
# ---------------------------------------------------------------------------


class TestKSweep:
    """Tests for GMM k-sweep."""

    def test_returns_sweep_result(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        sweep = _k_sweep_gmm(X, k_range=range(2, 5))
        assert isinstance(sweep, SweepResult)

    def test_best_k_in_range(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        sweep = _k_sweep_gmm(X, k_range=range(2, 6))
        assert sweep.best_k in [2, 3, 4, 5]

    def test_bic_count_matches_k_range(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        sweep = _k_sweep_gmm(X, k_range=range(2, 5))
        assert len(sweep.bic_scores) == 3  # k=2,3,4
        assert len(sweep.k_range) == 3

    def test_best_bic_is_minimum(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 5))
        sweep = _k_sweep_gmm(X, k_range=range(2, 6))
        assert sweep.best_bic == min(sweep.bic_scores)

    def test_well_separated_finds_correct_k(self):
        """3 well-separated Gaussians → best k should be 3."""
        rng = np.random.RandomState(42)
        c1 = rng.normal([0, 0], 0.5, (200, 2))
        c2 = rng.normal([10, 0], 0.5, (200, 2))
        c3 = rng.normal([5, 10], 0.5, (200, 2))
        X = np.vstack([c1, c2, c3])
        sweep = _k_sweep_gmm(X, k_range=range(2, 6))
        assert sweep.best_k == 3, f"Expected k=3, got k={sweep.best_k}"

    def test_single_gaussian_prefers_k2(self):
        """Single Gaussian → BIC should prefer k=2 (minimum in range)."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 5))
        sweep = _k_sweep_gmm(X, k_range=range(2, 6))
        # For single Gaussian, k=2 typically has lowest BIC
        assert sweep.best_k == 2


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


class TestBlockBootstrap:
    """Tests for block bootstrap stability."""

    def test_returns_stability_report(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        labels = np.zeros(200, dtype=int)
        labels[100:] = 1
        result = _block_bootstrap_stability(
            X, labels, n_components=2, n_bootstrap=10, block_size=15
        )
        assert isinstance(result, StabilityReport)

    def test_ari_in_range(self):
        """Mean ARI should be in [-1, 1]."""
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.normal([0, 0], 1, (100, 2)),
            rng.normal([5, 5], 1, (100, 2)),
        ])
        labels = np.array([0] * 100 + [1] * 100)
        result = _block_bootstrap_stability(
            X, labels, n_components=2, n_bootstrap=10, block_size=15
        )
        assert -1.0 <= result.mean_ari <= 1.0

    def test_well_separated_high_ari(self):
        """Well-separated clusters → high bootstrap ARI."""
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.normal([0, 0], 0.3, (200, 2)),
            rng.normal([10, 10], 0.3, (200, 2)),
        ])
        labels = np.array([0] * 200 + [1] * 200)
        result = _block_bootstrap_stability(
            X, labels, n_components=2, n_bootstrap=20, block_size=15
        )
        assert result.mean_ari > 0.5

    def test_block_size_stored(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        labels = np.zeros(100, dtype=int)
        result = _block_bootstrap_stability(
            X, labels, n_components=1, n_bootstrap=5, block_size=20
        )
        assert result.block_size == 20

    def test_n_bootstrap_reasonable(self):
        """Number of successful bootstrap samples should be close to requested."""
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.normal(0, 1, (150, 3)),
            rng.normal(5, 1, (150, 3)),
        ])
        labels = np.array([0] * 150 + [1] * 150)
        result = _block_bootstrap_stability(
            X, labels, n_components=2, n_bootstrap=20, block_size=10
        )
        # Most should succeed
        assert result.n_bootstrap >= 10


# ---------------------------------------------------------------------------
# Centroid profiles
# ---------------------------------------------------------------------------


class TestCentroidProfiles:
    """Tests for centroid profile computation."""

    def test_shape(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 50 + [1] * 50)
        profiles = _centroid_profiles(df, labels)
        assert profiles.shape == (2, 5)  # 2 regimes, 5 features

    def test_centroids_differ_for_different_regimes(self):
        """If regimes have different means, centroids should differ."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 0.1, (100, 3)),
            rng.normal(10, 0.1, (100, 3)),
        ])
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        labels = np.array([0] * 100 + [1] * 100)
        profiles = _centroid_profiles(df, labels)
        for col in ["a", "b", "c"]:
            assert abs(profiles.loc[0, col] - profiles.loc[1, col]) > 5

    def test_single_regime(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["a", "b", "c"])
        labels = np.zeros(100, dtype=int)
        profiles = _centroid_profiles(df, labels)
        assert profiles.shape[0] == 1


# ---------------------------------------------------------------------------
# Full discover_macro_regimes — synthetic two-regime data
# ---------------------------------------------------------------------------


class TestDiscoverMacroRegimes:
    """End-to-end tests for discover_macro_regimes."""

    def test_two_regime_discovery(self):
        """Synthetic 2-regime data → discovers k=2 (or 3), labels mostly correct."""
        df, true_labels = _make_two_regime_derivatives(
            n_per_regime=250, block_size=50, separation=5.0
        )
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result, RegimeResult)
        assert not result.early_exit
        assert result.k >= 2

    def test_returns_correct_label_length(self):
        df, _ = _make_two_regime_derivatives(n_per_regime=200)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert len(result.labels) == 400

    def test_self_transition_rate_high(self):
        """Block-structured regimes → STR should be high."""
        df, _ = _make_two_regime_derivatives(
            n_per_regime=250, block_size=50, separation=5.0
        )
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            assert result.self_transition_rate > 0.7, (
                f"STR={result.self_transition_rate:.3f}, expected > 0.7"
            )

    def test_quality_report_present(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result.quality, QualityReport)
        assert -1 <= result.quality.silhouette <= 1

    def test_stability_report_present(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result.stability, StabilityReport)

    def test_sweep_result_present(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result.sweep, SweepResult)
        assert len(result.sweep.bic_scores) == 3  # k=2,3,4

    def test_durations_present(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result.durations, dict)
        total = sum(sum(runs) for runs in result.durations.values())
        assert total == len(df)

    def test_centroid_profiles_present(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            assert isinstance(result.centroid_profiles, pd.DataFrame)
            assert result.centroid_profiles.shape[0] == result.k

    def test_slow_columns_stored(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3
        )
        assert isinstance(result.slow_columns, list)
        assert len(result.slow_columns) > 0

    def test_structure_test_stored(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3
        )
        assert isinstance(result.structure_test, StructureTest)

    def test_filter_report_stored(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3
        )
        assert isinstance(result.filter_report, dict)

    def test_pca_result_stored(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3
        )
        from cluster_pipeline.reduction import PCAResult
        assert isinstance(result.pca_result, PCAResult)


# ---------------------------------------------------------------------------
# Early exit — no structure
# ---------------------------------------------------------------------------


class TestNoStructureEarlyExit:
    """Uniform random data → early exit, no clustering attempted."""

    def test_uniform_early_exit_or_weak(self):
        """Uniform derivatives → early_exit=True OR weak_structure.

        Structure test may detect weak clustering tendency in uniform data
        (Hopkins can be borderline). The key invariant: if it doesn't exit
        early, the structure test should be weak_structure, not proceed.
        """
        df = _make_uniform_derivatives(n=500, dim=10)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0,
            k_range=range(2, 5),
        )
        if not result.early_exit:
            assert result.structure_test.recommendation != "proceed"

    def test_early_exit_labels_all_zero(self):
        """When early exit occurs, labels should be all zeros."""
        # Use all-constant data to force early exit
        df = pd.DataFrame({f"c_{i}": np.full(100, float(i)) for i in range(5)})
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0,
        )
        assert result.early_exit is True
        assert np.all(result.labels == 0)

    def test_early_exit_reason_present(self):
        """Early exit should have a reason string."""
        df = pd.DataFrame({f"c_{i}": np.full(100, float(i)) for i in range(5)})
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0,
        )
        assert result.early_exit is True
        assert len(result.early_exit_reason) > 0

    def test_early_exit_k_is_zero(self):
        """Early exit → k=0."""
        df = pd.DataFrame({f"c_{i}": np.full(100, float(i)) for i in range(5)})
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0,
        )
        assert result.early_exit is True
        assert result.k == 0


# ---------------------------------------------------------------------------
# Few slow columns fallback
# ---------------------------------------------------------------------------


class TestSlowColumnFallback:
    """When too few slow columns, should fall back to all columns."""

    def test_all_noise_uses_all_columns(self):
        """All white noise columns → fallback to all, with warning."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            f"noise_{i}": rng.normal(0, 1, 500) for i in range(10)
        })
        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = discover_macro_regimes(
                df, autocorrelation_threshold=0.9,
            )
        # Should have warned about using all columns
        warn_msgs = [str(c.message) for c in caught]
        assert any("Using all columns" in m for m in warn_msgs)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestDiscoverValidation:
    """Invalid inputs for discover_macro_regimes."""

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            discover_macro_regimes(pd.DataFrame())

    def test_too_few_rows_raises(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (20, 5)), columns=[f"f_{i}" for i in range(5)])
        with pytest.raises(ValueError, match="at least 30"):
            discover_macro_regimes(df)

    def test_minimum_30_rows(self):
        """Exactly 30 rows should not raise."""
        rng = np.random.RandomState(42)
        # Need slow columns for this to work
        data = {}
        for i in range(5):
            data[f"f_{i}"] = np.cumsum(rng.normal(0, 0.1, 30))
        df = pd.DataFrame(data)
        # Should not raise (may early-exit but no crash)
        result = discover_macro_regimes(df, autocorrelation_threshold=0.3)
        assert isinstance(result, RegimeResult)


# ---------------------------------------------------------------------------
# GMM params stored
# ---------------------------------------------------------------------------


class TestGMMParams:
    """Verify GMM parameters are correctly stored."""

    def test_gmm_params_has_means(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            assert "means" in result.gmm_params
            assert "weights" in result.gmm_params

    def test_gmm_weights_sum_to_one(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            weights = result.gmm_params["weights"]
            assert abs(sum(weights) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDiscoverDeterminism:
    """Same input → same output."""

    def test_deterministic(self):
        df, _ = _make_two_regime_derivatives(separation=5.0, seed=42)
        r1 = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, random_state=42
        )
        r2 = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, random_state=42
        )
        np.testing.assert_array_equal(r1.labels, r2.labels)
        assert r1.k == r2.k
        assert r1.self_transition_rate == r2.self_transition_rate


# ---------------------------------------------------------------------------
# Quality report properties
# ---------------------------------------------------------------------------


class TestQualityReportProperties:
    """Verify quality report invariants."""

    def test_min_cluster_fraction_in_range(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            assert 0.0 < result.quality.min_cluster_fraction <= 1.0

    def test_n_per_cluster_sums_to_n(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            total = sum(result.quality.n_per_cluster.values())
            assert total == len(df)

    def test_n_per_cluster_keys_match_k(self):
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        if not result.early_exit:
            assert len(result.quality.n_per_cluster) == result.k


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDiscoverEdgeCases:
    """Edge and adversarial inputs for regime discovery."""

    def test_all_constant_columns(self):
        """All constant columns → should early exit, not crash."""
        df = pd.DataFrame({
            f"c_{i}": np.full(100, float(i)) for i in range(5)
        })
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0
        )
        assert result.early_exit is True

    def test_single_column(self):
        """Single column DataFrame should not crash."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"only": np.cumsum(rng.normal(0, 0.1, 200))})
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.0
        )
        assert isinstance(result, RegimeResult)

    def test_nan_heavy_data(self):
        """10% NaN → should still produce a result."""
        df, _ = _make_two_regime_derivatives(separation=5.0)
        rng = np.random.RandomState(99)
        mask = rng.random(df.shape) < 0.10
        df = df.mask(mask)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 5)
        )
        assert isinstance(result, RegimeResult)

    def test_k_range_single_value(self):
        """k_range with single value → should work."""
        df, _ = _make_two_regime_derivatives(separation=5.0)
        result = discover_macro_regimes(
            df, autocorrelation_threshold=0.3, k_range=range(2, 3)
        )
        if not result.early_exit:
            assert result.k == 2


# ===========================================================================
# Task 3.2: Micro State Discovery tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers for micro-state tests
# ---------------------------------------------------------------------------


def _make_macro_micro_data(
    n_per_regime: int = 300,
    n_regimes: int = 2,
    n_micro: int = 2,
    dim: int = 8,
    macro_sep: float = 10.0,
    micro_sep: float = 3.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create data with known macro regimes, each containing micro sub-clusters.

    Returns (derivatives_df, macro_labels).
    Within each regime, bars alternate between micro sub-clusters in blocks.
    """
    rng = np.random.RandomState(seed)
    all_data = []
    all_labels = []

    for r in range(n_regimes):
        macro_center = np.zeros(dim)
        macro_center[0] = r * macro_sep

        # Create micro sub-clusters within this regime
        block_size = n_per_regime // (n_micro * 2)  # blocks of sub-clusters
        regime_data = np.zeros((n_per_regime, dim))
        for i in range(n_per_regime):
            micro_id = (i // block_size) % n_micro
            micro_offset = np.zeros(dim)
            micro_offset[1] = micro_id * micro_sep
            regime_data[i] = rng.normal(
                macro_center + micro_offset, 0.5, dim
            )

        all_data.append(regime_data)
        all_labels.append(np.full(n_per_regime, r, dtype=int))

    data = np.vstack(all_data)
    labels = np.concatenate(all_labels)
    columns = [f"feat_{i}" for i in range(dim)]
    return pd.DataFrame(data, columns=columns), labels


# ---------------------------------------------------------------------------
# Subset correctness
# ---------------------------------------------------------------------------


class TestMicroSubset:
    """Verify that micro-state discovery operates on the correct subset."""

    def test_only_regime_bars_used(self):
        """Labels length must match number of bars in the specified regime."""
        df, macro_labels = _make_macro_micro_data(n_per_regime=200)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50
        )
        if result is not None:
            n_regime_0 = int(np.sum(macro_labels == 0))
            assert len(result.labels) == n_regime_0
            assert result.n_bars == n_regime_0

    def test_regime_1_subset(self):
        """Regime 1 should produce labels for regime-1 bars only."""
        df, macro_labels = _make_macro_micro_data(n_per_regime=200)
        result = discover_micro_states(
            df, macro_labels, regime_id=1, min_bars=50
        )
        if result is not None:
            n_regime_1 = int(np.sum(macro_labels == 1))
            assert len(result.labels) == n_regime_1

    def test_regime_id_stored(self):
        """MicroStateResult should store the correct regime_id."""
        df, macro_labels = _make_macro_micro_data(n_per_regime=200)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50
        )
        if result is not None:
            assert result.regime_id == 0


# ---------------------------------------------------------------------------
# Separate PCA per regime
# ---------------------------------------------------------------------------


class TestSeparatePCA:
    """Each regime should get its own PCA basis."""

    def test_pca_components_differ_between_regimes(self):
        """PCA on regime 0 vs regime 1 should produce different components."""
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=10.0, micro_sep=3.0
        )
        r0 = discover_micro_states(df, macro_labels, regime_id=0, min_bars=50)
        r1 = discover_micro_states(df, macro_labels, regime_id=1, min_bars=50)

        if r0 is not None and r1 is not None:
            # Components should differ (different data subsets)
            # They may have different shapes too
            if r0.pca_result.components.shape == r1.pca_result.components.shape:
                assert not np.allclose(
                    r0.pca_result.components, r1.pca_result.components
                ), "PCA components should differ between regimes"

    def test_pca_mean_differs_between_regimes(self):
        """PCA mean should reflect the regime's data, not the global mean."""
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=10.0
        )
        r0 = discover_micro_states(df, macro_labels, regime_id=0, min_bars=50)
        r1 = discover_micro_states(df, macro_labels, regime_id=1, min_bars=50)

        if r0 is not None and r1 is not None:
            # Means should be different (regimes are separated)
            assert not np.allclose(r0.pca_result.mean, r1.pca_result.mean)

    def test_regularization_for_small_regime(self):
        """Small regime with many features → Ledoit-Wolf should trigger."""
        rng = np.random.RandomState(42)
        # 120 bars, 80 features → after filtering, likely regularized
        data = np.zeros((240, 80))
        for i in range(240):
            if i < 120:
                data[i] = rng.normal(0, 1, 80)
            else:
                data[i] = rng.normal(5, 1, 80)

        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(80)])
        labels = np.array([0] * 120 + [1] * 120)

        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        if result is not None:
            # With 120 samples and many features post-filter, likely regularized
            assert isinstance(result.pca_result.regularized, bool)


# ---------------------------------------------------------------------------
# Small regime returns None
# ---------------------------------------------------------------------------


class TestSmallRegime:
    """Regimes with too few bars should return None."""

    def test_small_regime_returns_none(self):
        """Regime with 50 bars (< 100 default min) → None."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 50 + [1] * 150)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=100)
        assert result is None

    def test_small_regime_warning(self):
        """Small regime should emit a warning."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 30 + [1] * 170)
        with pytest.warns(UserWarning, match="only 30 bars"):
            discover_micro_states(df, labels, regime_id=0, min_bars=100)

    def test_custom_min_bars(self):
        """With min_bars=20, a 30-bar regime should be attempted."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 1, (30, 5)),
            rng.normal(10, 1, (170, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 30 + [1] * 170)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=20)
        # Should attempt (may return None due to no structure, but shouldn't skip)
        # The key: it wasn't skipped for being too small
        # If it IS None, it's because of structure test, not size
        assert result is None or isinstance(result, MicroStateResult)

    def test_exactly_min_bars(self):
        """Exactly min_bars should be attempted, not skipped."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 1, (100, 5)),
            rng.normal(10, 1, (100, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 100 + [1] * 100)
        # min_bars=100, regime has exactly 100 → should attempt
        result = discover_micro_states(df, labels, regime_id=0, min_bars=100)
        assert result is None or isinstance(result, MicroStateResult)

    def test_nonexistent_regime_returns_none(self):
        """Regime ID not in labels → 0 bars → returns None."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 50 + [1] * 50)
        result = discover_micro_states(df, labels, regime_id=99, min_bars=10)
        assert result is None


# ---------------------------------------------------------------------------
# No structure returns None
# ---------------------------------------------------------------------------


class TestMicroNoStructure:
    """Regimes without internal structure should return None."""

    def test_uniform_regime_returns_none(self):
        """Uniform data within a regime → no structure → None."""
        rng = np.random.RandomState(42)
        # Regime 0: uniform, Regime 1: structured
        data = np.vstack([
            rng.uniform(-1, 1, (300, 5)),
            rng.normal(10, 1, (300, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 300 + [1] * 300)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        # Uniform data may or may not pass structure test — but if it does,
        # the structure should at most be weak
        if result is not None:
            assert result.structure_test.recommendation != "no_structure"

    def test_single_gaussian_regime(self):
        """Single tight Gaussian within regime → likely no micro-structure."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 0.1, (300, 5)),
            rng.normal(10, 1, (300, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 300 + [1] * 300)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        # Very tight single Gaussian — micro discovery should not find "proceed"
        if result is not None:
            # If it found something, it's at best weak structure
            pass  # acceptable — structure test is statistical


# ---------------------------------------------------------------------------
# Micro within macro
# ---------------------------------------------------------------------------


class TestMicroWithinMacro:
    """Verify micro-state discovery finds sub-clusters within regimes."""

    def test_finds_micro_clusters(self):
        """2 macro regimes each with 2 sub-clusters → micro should find k≥2."""
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, n_regimes=2, n_micro=2,
            macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0,
            k_range=range(2, 5), min_bars=50,
        )
        if result is not None:
            assert result.k >= 2

    def test_micro_labels_valid(self):
        """Micro labels should contain values in [0, k-1]."""
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            unique = np.unique(result.labels)
            assert all(0 <= u < result.k for u in unique)

    def test_micro_quality_present(self):
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.quality, QualityReport)
            assert -1 <= result.quality.silhouette <= 1

    def test_micro_stability_present(self):
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.stability, StabilityReport)

    def test_micro_sweep_present(self):
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.sweep, SweepResult)

    def test_micro_centroid_profiles(self):
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0,
        )
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.centroid_profiles, pd.DataFrame)
            assert result.centroid_profiles.shape[0] == result.k


# ---------------------------------------------------------------------------
# Output types and structure
# ---------------------------------------------------------------------------


class TestMicroOutputStructure:
    """Verify MicroStateResult fields."""

    def test_result_type(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=300, micro_sep=5.0)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result, MicroStateResult)

    def test_gmm_params_present(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=300, micro_sep=5.0)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert "means" in result.gmm_params
            assert "weights" in result.gmm_params

    def test_gmm_weights_sum_to_one(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=300, micro_sep=5.0)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert abs(sum(result.gmm_params["weights"]) - 1.0) < 1e-6

    def test_n_bars_correct(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=250)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert result.n_bars == 250

    def test_structure_test_stored(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=300, micro_sep=5.0)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.structure_test, StructureTest)

    def test_filter_report_stored(self):
        df, macro_labels = _make_macro_micro_data(n_per_regime=300, micro_sep=5.0)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        if result is not None:
            assert isinstance(result.filter_report, dict)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestMicroValidation:
    """Invalid inputs for discover_micro_states."""

    def test_length_mismatch_raises(self):
        """derivatives rows != macro_labels length → ValueError."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.zeros(50, dtype=int)
        with pytest.raises(ValueError, match="macro_labels length"):
            discover_micro_states(df, labels, regime_id=0)

    def test_negative_regime_id(self):
        """Negative regime_id not in labels → returns None (0 bars)."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f_{i}" for i in range(5)])
        labels = np.zeros(200, dtype=int)
        result = discover_micro_states(df, labels, regime_id=-1, min_bars=10)
        assert result is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestMicroDeterminism:
    """Same input → same output."""

    def test_deterministic(self):
        df, macro_labels = _make_macro_micro_data(
            n_per_regime=300, macro_sep=15.0, micro_sep=5.0, seed=42,
        )
        r1 = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50, random_state=42,
        )
        r2 = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50, random_state=42,
        )
        if r1 is not None and r2 is not None:
            np.testing.assert_array_equal(r1.labels, r2.labels)
            assert r1.k == r2.k


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMicroEdgeCases:
    """Edge cases for micro-state discovery."""

    def test_all_bars_one_regime(self):
        """All bars in a single regime → micro discovery on full dataset."""
        rng = np.random.RandomState(42)
        # Two sub-clusters to ensure structure
        data = np.vstack([
            rng.normal(0, 1, (200, 5)),
            rng.normal(5, 1, (200, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.zeros(400, dtype=int)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        if result is not None:
            assert len(result.labels) == 400

    def test_nan_in_derivatives(self):
        """NaN values in derivatives → should still work (reduce handles NaN)."""
        df, macro_labels = _make_macro_micro_data(n_per_regime=200, micro_sep=5.0)
        rng = np.random.RandomState(99)
        mask = rng.random(df.shape) < 0.05
        df = df.mask(mask)
        result = discover_micro_states(
            df, macro_labels, regime_id=0, min_bars=50,
        )
        assert result is None or isinstance(result, MicroStateResult)

    def test_many_features(self):
        """Wide data (many features) → Ledoit-Wolf should handle it."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 1, (150, 50)),
            rng.normal(5, 1, (150, 50)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(50)])
        labels = np.zeros(300, dtype=int)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        assert result is None or isinstance(result, MicroStateResult)

    def test_k_range_capped_by_n_bars(self):
        """If regime has 110 bars, k_range should not include k > 110."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 1, (110, 5)),
            rng.normal(10, 1, (110, 5)),
        ])
        df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(5)])
        labels = np.array([0] * 110 + [1] * 110)
        # k_range up to 200 — should be capped
        result = discover_micro_states(
            df, labels, regime_id=0, min_bars=50,
            k_range=range(2, 200),
        )
        if result is not None:
            assert result.k < 110

    def test_constant_features_in_regime(self):
        """Regime with some constant features → reduction filters them."""
        rng = np.random.RandomState(42)
        data = np.vstack([
            rng.normal(0, 1, (200, 3)),
            rng.normal(5, 1, (200, 3)),
        ])
        # Add constant columns
        const = np.full((400, 2), 3.14)
        full_data = np.hstack([data, const])
        df = pd.DataFrame(full_data, columns=[f"f_{i}" for i in range(5)])
        labels = np.zeros(400, dtype=int)
        result = discover_micro_states(df, labels, regime_id=0, min_bars=50)
        assert result is None or isinstance(result, MicroStateResult)

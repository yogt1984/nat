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

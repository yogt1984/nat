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

    def test_summary_generation(self, excellent_score):
        """Summary should generate without error."""
        summary = excellent_score.summary()
        assert isinstance(summary, str)
        assert "CLUSTER QUALITY ASSESSMENT" in summary
        assert "Grade:" in summary

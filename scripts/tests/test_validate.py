"""
Skeptical tests for cluster_pipeline.validate — quality gate runner.

Test philosophy:
  - Decision logic: Q1 fail→DROP, Q2 fail→COLLECT, Q3 fail→PIVOT, all pass→GO
  - Q2 any_horizon: passes if significant at ANY horizon
  - Thresholds are configurable
  - Edge cases: early exit, single state, all states same returns
  - Mock profiling results with known quality metrics
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.validate import (
    ValidationVerdict,
    validate,
    _evaluate_q1,
    _evaluate_q2,
    _evaluate_q3,
    Q1_DEFAULTS,
    Q2_DEFAULTS,
    Q3_DEFAULTS,
)
from cluster_pipeline.hierarchy import (
    HierarchicalLabels,
    ProfilingResult,
    RegimeResult,
    StructureTest,
    MicroStateResult,
)
from cluster_pipeline.reduction import PCAResult
from cluster_pipeline.transitions import empirical_transitions


# ===========================================================================
# Helpers
# ===========================================================================


def _make_pca_result(n: int) -> PCAResult:
    return PCAResult(
        X_reduced=np.zeros((n, 2)),
        n_components=2,
        explained_variance_ratio=np.array([0.6, 0.3]),
        cumulative_variance=np.array([0.6, 0.9]),
        components=np.eye(2),
        mean=np.zeros(2),
        std=np.ones(2),
        column_names=["f0", "f1"],
        loadings={"f0": [1.0, 0.0], "f1": [0.0, 1.0]},
        regularized=False,
    )


def _make_quality_report(silhouette=0.5, min_frac=0.3):
    from cluster_pipeline.hierarchy import QualityReport
    return QualityReport(silhouette=silhouette, min_cluster_fraction=min_frac,
                         n_per_cluster={0: 50, 1: 50})


def _make_stability_report(mean_ari=0.8):
    from cluster_pipeline.hierarchy import StabilityReport
    return StabilityReport(mean_ari=mean_ari, std_ari=0.1, n_bootstrap=30, block_size=15)


def _make_sweep():
    from cluster_pipeline.hierarchy import SweepResult
    return SweepResult(k_range=[2, 3], bic_scores=[100, 90], best_k=2, best_bic=90)


def _make_profiling_result(
    n=200,
    k=2,
    silhouette=0.5,
    mean_ari=0.8,
    self_transition_rate=0.9,
    early_exit=False,
    durations=None,
) -> ProfilingResult:
    """Build a ProfilingResult with controllable quality metrics."""
    if durations is None:
        durations = {0: [20, 30, 25], 1: [25, 30, 20]}

    macro_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    micro_labels = macro_labels.copy()
    composite = np.array(["R0_S0"] * (n // 2) + ["R1_S0"] * (n - n // 2))
    label_map = {0: (0, 0), 1: (1, 0)}

    hierarchy = HierarchicalLabels(
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        composite_labels=composite,
        n_macro=k,
        n_micro_per_regime={0: 1, 1: 1},
        n_micro_total=2,
        label_map=label_map,
    )

    structure = StructureTest(
        hopkins_statistic=0.8, dip_test_p=0.01,
        has_structure=True, recommendation="proceed",
    )

    macro = RegimeResult(
        labels=macro_labels,
        k=k,
        pca_result=_make_pca_result(n),
        gmm_params={"means": [], "weights": []},
        quality=_make_quality_report(silhouette=silhouette),
        stability=_make_stability_report(mean_ari=mean_ari),
        sweep=_make_sweep(),
        centroid_profiles=pd.DataFrame(),
        self_transition_rate=self_transition_rate,
        durations=durations,
        structure_test=structure,
        slow_columns=["f0"],
        filter_report={},
        early_exit=early_exit,
        early_exit_reason="no structure" if early_exit else "",
    )

    bars = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])

    return ProfilingResult(
        hierarchy=hierarchy,
        macro=macro,
        micros={0: None, 1: None},
        derivatives_meta={"n_total": 10, "base_features": ["f0"]},
        reduction_report={},
        bars=bars,
        derivative_columns=["a", "b", "c"],
        breaks_detected=[],
        structure_test=structure,
    )


def _make_prices_with_regime_signal(n=200, seed=42):
    """
    Create prices where state 0 has positive drift and state 1 has negative drift.
    This ensures Kruskal-Wallis will be significant.
    """
    rng = np.random.RandomState(seed)
    half = n // 2
    # State 0 (first half): positive drift
    ret_0 = rng.normal(0.01, 0.005, half)
    # State 1 (second half): negative drift
    ret_1 = rng.normal(-0.01, 0.005, n - half)
    all_returns = np.concatenate([ret_0, ret_1])
    log_prices = np.cumsum(all_returns) + 5
    return np.exp(log_prices)


def _make_prices_no_signal(n=200, seed=42):
    """Prices with no regime-dependent signal (random walk)."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0, 0.01, n)
    log_prices = np.cumsum(returns) + 5
    return np.exp(log_prices)


# ===========================================================================
# Decision Logic
# ===========================================================================


class TestDecisionLogic:
    """Overall verdict follows Q1→Q2→Q3 cascade."""

    def test_all_pass_is_go(self):
        """High quality + predictive + operational → GO."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8,
                                    self_transition_rate=0.9)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "GO"

    def test_q1_fail_is_drop(self):
        """Low silhouette → DROP."""
        pr = _make_profiling_result(silhouette=0.1, mean_ari=0.3)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "DROP"

    def test_q1_fail_silhouette_only(self):
        """Silhouette below threshold, ARI ok → still DROP."""
        pr = _make_profiling_result(silhouette=0.1, mean_ari=0.9)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "DROP"

    def test_q1_fail_ari_only(self):
        """ARI below threshold, silhouette ok → still DROP."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.3)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "DROP"

    def test_q2_fail_is_collect(self):
        """Q1 passes but no predictive signal → COLLECT."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8)
        prices = _make_prices_no_signal(200)
        verdict = validate(pr, prices)
        # With random prices, Kruskal-Wallis likely not significant
        assert verdict.overall in ("COLLECT", "GO", "PIVOT")
        # Force: use very strict threshold so Q2 definitely fails
        verdict2 = validate(pr, prices, q2_thresholds={"kruskal_p": 0.001,
                                                        "eta_squared": 0.5,
                                                        "min_sharpe": 5.0,
                                                        "any_horizon": True})
        assert verdict2.overall == "COLLECT"

    def test_q3_fail_is_pivot(self):
        """Q1+Q2 pass but low persistence → PIVOT."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8,
                                    self_transition_rate=0.3,
                                    durations={0: [1, 1, 1], 1: [1, 1, 1]})
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "PIVOT"

    def test_early_exit_is_drop(self):
        """Early exit (no structure) → DROP."""
        pr = _make_profiling_result(early_exit=True)
        prices = _make_prices_no_signal(200)
        verdict = validate(pr, prices)
        assert verdict.overall == "DROP"


# ===========================================================================
# Q1 Structural
# ===========================================================================


class TestQ1Structural:
    """Q1 gate checks silhouette and bootstrap ARI."""

    def test_passes_when_above_thresholds(self):
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8)
        q1 = _evaluate_q1(pr.macro, Q1_DEFAULTS)
        assert q1["pass"] is True

    def test_fails_low_silhouette(self):
        pr = _make_profiling_result(silhouette=0.1)
        q1 = _evaluate_q1(pr.macro, Q1_DEFAULTS)
        assert q1["pass"] is False
        assert "silhouette" in q1.get("reason", "")

    def test_fails_low_ari(self):
        pr = _make_profiling_result(mean_ari=0.2)
        q1 = _evaluate_q1(pr.macro, Q1_DEFAULTS)
        assert q1["pass"] is False

    def test_custom_thresholds(self):
        """Relaxed thresholds → passes."""
        pr = _make_profiling_result(silhouette=0.1, mean_ari=0.3)
        q1 = _evaluate_q1(pr.macro, {"silhouette": 0.05, "block_bootstrap_ari": 0.1})
        assert q1["pass"] is True

    def test_early_exit_fails(self):
        pr = _make_profiling_result(early_exit=True)
        q1 = _evaluate_q1(pr.macro, Q1_DEFAULTS)
        assert q1["pass"] is False


# ===========================================================================
# Q2 Predictive (Any Horizon)
# ===========================================================================


class TestQ2Predictive:
    """Q2 passes if Kruskal-Wallis significant at any horizon."""

    def test_significant_signal_passes(self):
        """Clear regime-dependent returns → Q2 passes."""
        n = 200
        micro_labels = np.array([0] * 100 + [1] * 100)
        composite = np.array(["R0_S0"] * 100 + ["R1_S0"] * 100)
        hierarchy = HierarchicalLabels(
            macro_labels=micro_labels.copy(),
            micro_labels=micro_labels,
            composite_labels=composite,
            n_macro=2, n_micro_per_regime={0: 1, 1: 1},
            n_micro_total=2, label_map={0: (0, 0), 1: (1, 0)},
        )
        prices = _make_prices_with_regime_signal(n)
        q2 = _evaluate_q2(hierarchy, prices, Q2_DEFAULTS)
        assert q2["pass"] is True
        assert len(q2["pass_horizons"]) > 0

    def test_no_signal_fails(self):
        """Random walk → Q2 fails (likely)."""
        n = 200
        micro_labels = np.array([0] * 100 + [1] * 100)
        composite = np.array(["R0_S0"] * 100 + ["R1_S0"] * 100)
        hierarchy = HierarchicalLabels(
            macro_labels=micro_labels.copy(),
            micro_labels=micro_labels,
            composite_labels=composite,
            n_macro=2, n_micro_per_regime={0: 1, 1: 1},
            n_micro_total=2, label_map={0: (0, 0), 1: (1, 0)},
        )
        prices = _make_prices_no_signal(n)
        # Very strict thresholds to ensure failure
        q2 = _evaluate_q2(hierarchy, prices, {"kruskal_p": 0.001,
                                               "eta_squared": 0.5,
                                               "min_sharpe": 5.0,
                                               "any_horizon": True})
        assert q2["pass"] is False

    def test_any_horizon_flag(self):
        """any_horizon=True: passes if ANY horizon significant."""
        n = 200
        micro_labels = np.array([0] * 100 + [1] * 100)
        composite = np.array(["R0_S0"] * 100 + ["R1_S0"] * 100)
        hierarchy = HierarchicalLabels(
            macro_labels=micro_labels.copy(),
            micro_labels=micro_labels,
            composite_labels=composite,
            n_macro=2, n_micro_per_regime={0: 1, 1: 1},
            n_micro_total=2, label_map={0: (0, 0), 1: (1, 0)},
        )
        prices = _make_prices_with_regime_signal(n)
        q2 = _evaluate_q2(hierarchy, prices, Q2_DEFAULTS)
        # Should pass at some horizons
        if q2["pass"]:
            assert len(q2["pass_horizons"]) >= 1

    def test_single_state_fails(self):
        """Only one state → can't do Kruskal-Wallis → Q2 fails."""
        n = 100
        micro_labels = np.zeros(n, dtype=int)
        composite = np.array(["R0_S0"] * n)
        hierarchy = HierarchicalLabels(
            macro_labels=micro_labels.copy(),
            micro_labels=micro_labels,
            composite_labels=composite,
            n_macro=1, n_micro_per_regime={0: 1},
            n_micro_total=1, label_map={0: (0, 0)},
        )
        prices = _make_prices_with_regime_signal(n)
        q2 = _evaluate_q2(hierarchy, prices, Q2_DEFAULTS)
        assert q2["pass"] is False


# ===========================================================================
# Q3 Operational
# ===========================================================================


class TestQ3Operational:
    """Q3 checks regime persistence and duration."""

    def test_high_persistence_passes(self):
        pr = _make_profiling_result(self_transition_rate=0.95,
                                    durations={0: [20, 30], 1: [25, 35]})
        q3 = _evaluate_q3(pr.macro, pr.hierarchy, Q3_DEFAULTS)
        assert q3["pass"] is True

    def test_low_str_fails(self):
        pr = _make_profiling_result(self_transition_rate=0.3,
                                    durations={0: [20, 30], 1: [25, 35]})
        q3 = _evaluate_q3(pr.macro, pr.hierarchy, Q3_DEFAULTS)
        assert q3["pass"] is False
        assert "macro_str" in q3.get("reason", "")

    def test_short_durations_fails(self):
        pr = _make_profiling_result(self_transition_rate=0.9,
                                    durations={0: [1, 1, 2], 1: [1, 2, 1]})
        q3 = _evaluate_q3(pr.macro, pr.hierarchy, Q3_DEFAULTS)
        assert q3["pass"] is False

    def test_custom_thresholds(self):
        """Relaxed thresholds → passes."""
        pr = _make_profiling_result(self_transition_rate=0.5,
                                    durations={0: [2, 2], 1: [2, 2]})
        q3 = _evaluate_q3(pr.macro, pr.hierarchy,
                           {"macro_str": 0.4, "micro_str": 0.3,
                            "min_duration": 1, "entry_lead": 0})
        assert q3["pass"] is True

    def test_early_exit_fails(self):
        pr = _make_profiling_result(early_exit=True)
        q3 = _evaluate_q3(pr.macro, pr.hierarchy, Q3_DEFAULTS)
        assert q3["pass"] is False


# ===========================================================================
# Return Type
# ===========================================================================


class TestReturnType:
    """validate() returns ValidationVerdict with correct types."""

    def test_returns_validation_verdict(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert isinstance(verdict, ValidationVerdict)

    def test_overall_is_string(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert isinstance(verdict.overall, str)
        assert verdict.overall in ("GO", "PIVOT", "COLLECT", "DROP")

    def test_per_state_verdicts_is_dict(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert isinstance(verdict.per_state_verdicts, dict)

    def test_summary_is_string(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert isinstance(verdict.summary, str)
        assert len(verdict.summary) > 0

    def test_q1_q2_q3_are_dicts(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert isinstance(verdict.q1_structural, dict)
        assert isinstance(verdict.q2_predictive, dict)
        assert isinstance(verdict.q3_operational, dict)


# ===========================================================================
# Validation Errors
# ===========================================================================


class TestValidationErrors:
    """Invalid inputs raise errors."""

    def test_price_length_mismatch(self):
        pr = _make_profiling_result(n=200)
        prices = np.ones(100)  # wrong length
        with pytest.raises(ValueError, match="prices length"):
            validate(pr, prices)


# ===========================================================================
# Threshold Override
# ===========================================================================


class TestThresholdOverride:
    """Custom thresholds override defaults."""

    def test_strict_q1_causes_drop(self):
        """Very strict Q1 threshold → DROP even with good metrics."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices, q1_thresholds={"silhouette": 0.99,
                                                       "block_bootstrap_ari": 0.99})
        assert verdict.overall == "DROP"

    def test_relaxed_q1_causes_pass(self):
        """Relaxed Q1 → passes."""
        pr = _make_profiling_result(silhouette=0.1, mean_ari=0.3)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices, q1_thresholds={"silhouette": 0.05,
                                                       "block_bootstrap_ari": 0.1})
        assert verdict.overall != "DROP"

    def test_strict_q3_causes_pivot(self):
        """Strict Q3 + good Q1/Q2 → PIVOT."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8,
                                    self_transition_rate=0.85)
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices, q3_thresholds={"macro_str": 0.99,
                                                       "micro_str": 0.99,
                                                       "min_duration": 100,
                                                       "entry_lead": 1})
        assert verdict.overall == "PIVOT"


# ===========================================================================
# Per-State Verdicts
# ===========================================================================


class TestPerStateVerdicts:
    """Per-state verdicts are computed for all states."""

    def test_all_states_have_verdict(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        assert set(verdict.per_state_verdicts.keys()) == set(range(pr.hierarchy.n_micro_total))

    def test_verdict_values_valid(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        verdict = validate(pr, prices)
        for v in verdict.per_state_verdicts.values():
            assert v in ("GO", "COLLECT")


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same inputs → same verdict."""

    def test_deterministic(self):
        pr = _make_profiling_result()
        prices = _make_prices_with_regime_signal(200)
        v1 = validate(pr, prices)
        v2 = validate(pr, prices)
        assert v1.overall == v2.overall
        assert v1.per_state_verdicts == v2.per_state_verdicts


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases."""

    def test_constant_prices(self):
        """Constant prices → Q2 fails (no signal) → COLLECT."""
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8,
                                    self_transition_rate=0.9)
        prices = np.ones(200) * 100
        verdict = validate(pr, prices)
        # Constant prices → all returns = 0 → no variance → Kruskal fails
        assert verdict.overall == "COLLECT"

    def test_very_short_data(self):
        """Short data (30 bars) still works."""
        pr = _make_profiling_result(n=30, silhouette=0.5, mean_ari=0.8,
                                    self_transition_rate=0.9,
                                    durations={0: [15], 1: [15]})
        prices = _make_prices_with_regime_signal(30)
        verdict = validate(pr, prices)
        assert verdict.overall in ("GO", "PIVOT", "COLLECT", "DROP")

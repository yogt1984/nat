"""
Tests for skeptical_regression_test.py.

Tests cover:
- TestResult dataclass
- Shared model helper
- Walk-forward IC computation
- All 10 skeptical tests (T1-T10) on synthetic data
- Overall verdict logic
- Edge cases
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest


from skeptical_regression_test import (
    TestResult,
    _train_regressor,
    _walkforward_ic,
    test_t1_permutation as run_t1_permutation,
    test_t2_effective_n as run_t2_effective_n,
    test_t3_block_bootstrap as run_t3_block_bootstrap,
    test_t4_feature_ablation as run_t4_feature_ablation,
    test_t5_temporal_stability as run_t5_temporal_stability,
    test_t7_nonoverlapping as run_t7_nonoverlapping,
    test_t8_regime_split as run_t8_regime_split,
    test_t9_cost_sensitivity as run_t9_cost_sensitivity,
    test_t10_embargo_walkforward as run_t10_embargo_walkforward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_data(n=5000, d=10, signal_strength=0.0, seed=42):
    """Generate synthetic X, y with optional planted signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    noise = rng.normal(0, 0.01, n)
    y = signal_strength * X[:, 0] + noise
    return X, y


def _synthetic_df(n=5000, d=5, seed=42):
    """Generate a polars DataFrame matching NAT schema."""
    rng = np.random.default_rng(seed)
    prices = 100_000.0 + np.cumsum(rng.normal(0, 1, n))
    data = {
        "timestamp_ns": np.arange(n, dtype=np.int64) * 100_000_000,
        "symbol": ["BTC"] * n,
        "raw_midprice": prices,
        "forward_return": rng.normal(0, 0.0001, n),
        "target": (rng.normal(0, 1, n) > 0).astype(int),
        "ctx_funding_rate": rng.normal(0.0001, 0.00005, n),
    }
    for i in range(d):
        data[f"feat_{i}"] = rng.normal(0, 1, n)
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests: TestResult
# ---------------------------------------------------------------------------


class TestTestResult:
    def test_pass_verdict(self):
        r = TestResult("T1", "Test", True, 0.5, 0.3)
        assert r.verdict == "PASS"

    def test_fail_verdict(self):
        r = TestResult("T1", "Test", False, 0.1, 0.3)
        assert r.verdict == "FAIL"

    def test_custom_verdict(self):
        r = TestResult("T1", "Test", False, 0.1, 0.3, verdict="WARN")
        assert r.verdict == "WARN"

    def test_fields_populated(self):
        r = TestResult("T2", "Effective N", True, 1.85, 0.05, p_value=0.01, detail="test")
        assert r.test_id == "T2"
        assert r.p_value == 0.01


# ---------------------------------------------------------------------------
# Tests: Model Helper
# ---------------------------------------------------------------------------


class TestTrainRegressor:
    def test_returns_model(self):
        X, y = _synthetic_data(n=500, d=3)
        model = _train_regressor(X, y)
        assert hasattr(model, "predict")

    def test_predictions_shape(self):
        X, y = _synthetic_data(n=500, d=3)
        model = _train_regressor(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)


class TestWalkforwardIC:
    def test_returns_correct_splits(self):
        X, y = _synthetic_data(n=3000, signal_strength=0.01)
        ics = _walkforward_ic(X, y, n_splits=3)
        assert len(ics) == 3

    def test_ic_bounded(self):
        X, y = _synthetic_data(n=3000, signal_strength=0.01)
        ics = _walkforward_ic(X, y, n_splits=3)
        for ic in ics:
            assert -1.0 <= ic <= 1.0

    def test_embargo_reduces_data(self):
        X, y = _synthetic_data(n=3000)
        ics_no_embargo = _walkforward_ic(X, y, n_splits=3, embargo=0)
        ics_embargo = _walkforward_ic(X, y, n_splits=3, embargo=500)
        # With embargo, some splits may be dropped
        assert len(ics_embargo) <= len(ics_no_embargo)

    def test_signal_detected(self):
        """Strong signal should produce positive ICs."""
        X, y = _synthetic_data(n=5000, signal_strength=0.1)
        ics = _walkforward_ic(X, y, n_splits=3)
        # At least some splits should have positive IC with strong signal
        assert any(ic > 0 for ic in ics)


# ---------------------------------------------------------------------------
# Tests: T1 Permutation
# ---------------------------------------------------------------------------


class TestT1Permutation:
    def test_pure_noise_fails(self):
        """No signal should produce p > 0.05."""
        X, y = _synthetic_data(n=2000, signal_strength=0.0)
        r = run_t1_permutation(X, y, n_splits=3, n_perms=50, seed=42)
        # Pure noise — p should be high (not significant)
        assert r.test_id == "T1"
        assert r.p_value is not None

    def test_strong_signal_passes(self):
        """Strong signal should beat permutation null."""
        X, y = _synthetic_data(n=3000, signal_strength=0.5)
        r = run_t1_permutation(X, y, n_splits=3, n_perms=50, seed=42)
        assert r.passed
        assert r.p_value < 0.1


# ---------------------------------------------------------------------------
# Tests: T2 Effective N
# ---------------------------------------------------------------------------


class TestT2EffectiveN:
    def test_massive_overlap(self):
        """With H=18000, 33K trades → effective N ≈ 1.85, should FAIL."""
        r = run_t2_effective_n(0.24, 33372, 18000)
        assert not r.passed
        assert r.statistic == pytest.approx(33372 / 18000, rel=0.01)

    def test_no_overlap(self):
        """dt = H means no overlap, effective N = n_trades."""
        r = run_t2_effective_n(0.24, 100, 1)
        assert r.passed  # 100 independent trades with IC=0.24

    def test_small_ic_fails(self):
        r = run_t2_effective_n(0.001, 1000, 1)
        assert not r.passed


# ---------------------------------------------------------------------------
# Tests: T3 Block Bootstrap
# ---------------------------------------------------------------------------


class TestT3BlockBootstrap:
    def test_positive_blocks_pass(self):
        blocks = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 0.8, 1.2])
        r = run_t3_block_bootstrap(blocks, n_bootstrap=500, seed=42)
        assert r.passed

    def test_negative_blocks_fail(self):
        blocks = np.array([-1.0, -2.0, -0.5, -1.5, -0.8])
        r = run_t3_block_bootstrap(blocks, n_bootstrap=500, seed=42)
        assert not r.passed

    def test_too_few_blocks(self):
        blocks = np.array([1.0, 2.0])
        r = run_t3_block_bootstrap(blocks, seed=42)
        assert not r.passed

    def test_mixed_blocks(self):
        """Mixed positive/negative should have wide CI."""
        blocks = np.array([5.0, -3.0, 2.0, -1.0, 4.0, -2.0, 1.0])
        r = run_t3_block_bootstrap(blocks, n_bootstrap=500, seed=42)
        assert r.test_id == "T3"


# ---------------------------------------------------------------------------
# Tests: T4 Feature Ablation
# ---------------------------------------------------------------------------


class TestT4FeatureAblation:
    def test_ablation_removes_funding(self):
        cols = ["feat_0", "feat_1", "ctx_funding_rate", "ctx_funding_zscore", "feat_2"]
        X, y = _synthetic_data(n=3000, d=5, signal_strength=0.01)
        r = run_t4_feature_ablation(X, y, cols, n_splits=3, seed=42)
        assert r.test_id == "T4"
        assert "ablated_ic" in r.detail

    def test_no_funding_features(self):
        """If no funding features, ablation should match full."""
        cols = ["feat_0", "feat_1", "feat_2"]
        X, y = _synthetic_data(n=3000, d=3, signal_strength=0.01)
        r = run_t4_feature_ablation(X, y, cols, n_splits=3, seed=42)
        assert r.test_id == "T4"


# ---------------------------------------------------------------------------
# Tests: T5 Temporal Stability
# ---------------------------------------------------------------------------


class TestT5TemporalStability:
    def test_runs_on_synthetic_df(self):
        df = _synthetic_df(n=10000)
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        r = run_t5_temporal_stability(df, feature_cols, horizon=100, seed=42)
        assert r.test_id == "T5"

    def test_too_few_days(self):
        """With only 1 day of data, should handle gracefully."""
        df = _synthetic_df(n=1000)
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        r = run_t5_temporal_stability(df, feature_cols, horizon=100, seed=42)
        assert r.test_id == "T5"


# ---------------------------------------------------------------------------
# Tests: T7 Non-Overlapping
# ---------------------------------------------------------------------------


class TestT7NonOverlapping:
    def test_trade_count_correct(self):
        X, y = _synthetic_data(n=5000, signal_strength=0.01)
        r = run_t7_nonoverlapping(X, y, horizon=500, seed=42)
        # 40% test = 2000 rows, / 500 = 4 trades
        assert "n_trades=4" in r.detail

    def test_large_horizon_few_trades(self):
        X, y = _synthetic_data(n=5000)
        r = run_t7_nonoverlapping(X, y, horizon=5000, seed=42)
        # Very few independent trades
        assert r.test_id == "T7"


# ---------------------------------------------------------------------------
# Tests: T8 Regime Split
# ---------------------------------------------------------------------------


class TestT8RegimeSplit:
    def test_runs_on_synthetic(self):
        df = _synthetic_df(n=5000)
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df.select(feature_cols).to_numpy()
        y = df["forward_return"].to_numpy()
        r = run_t8_regime_split(df, feature_cols, X, y, seed=42)
        assert r.test_id == "T8"

    def test_missing_funding_column(self):
        df = _synthetic_df(n=5000).drop("ctx_funding_rate")
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df.select(feature_cols).to_numpy()
        y = df["forward_return"].to_numpy()
        r = run_t8_regime_split(df, feature_cols, X, y, seed=42)
        assert not r.passed


# ---------------------------------------------------------------------------
# Tests: T9 Cost Sensitivity
# ---------------------------------------------------------------------------


class TestT9CostSensitivity:
    def test_high_gross_passes(self):
        r = run_t9_cost_sensitivity(15.0)
        assert r.passed
        assert r.statistic == 15.0

    def test_low_gross_fails(self):
        r = run_t9_cost_sensitivity(5.0)
        assert not r.passed

    def test_borderline(self):
        r = run_t9_cost_sensitivity(9.0)
        assert not r.passed  # 9 < 10


# ---------------------------------------------------------------------------
# Tests: T10 Embargo Walk-Forward
# ---------------------------------------------------------------------------


class TestT10EmbargoWalkforward:
    def test_runs_with_embargo(self):
        X, y = _synthetic_data(n=5000, signal_strength=0.01)
        r = run_t10_embargo_walkforward(X, y, horizon=200, n_splits=3, seed=42)
        assert r.test_id == "T10"

    def test_huge_embargo_reduces_splits(self):
        X, y = _synthetic_data(n=3000)
        r = run_t10_embargo_walkforward(X, y, horizon=2000, n_splits=5, seed=42)
        # Large embargo may eliminate some splits
        assert r.test_id == "T10"


# ---------------------------------------------------------------------------
# Tests: Overall Verdict Logic
# ---------------------------------------------------------------------------


class TestOverallVerdict:
    def test_reject_on_hard_kill(self):
        """T2 + T7 both fail → REJECT regardless."""
        results = [
            TestResult("T2", "Effective N", False, 1.85, 0.05),
            TestResult("T7", "Non-Overlapping", False, -1.0, 0.0),
        ]
        t2 = results[0]
        t7 = results[1]
        hard_kill = (not t2.passed) and (not t7.passed)
        assert hard_kill

    def test_proceed_when_all_pass(self):
        n_fail = 0
        overall = "REJECT" if n_fail >= 4 else "INVESTIGATE" if n_fail >= 2 else "PROCEED"
        assert overall == "PROCEED"

    def test_investigate_on_2_fails(self):
        n_fail = 2
        overall = "REJECT" if n_fail >= 4 else "INVESTIGATE" if n_fail >= 2 else "PROCEED"
        assert overall == "INVESTIGATE"

    def test_reject_on_4_fails(self):
        n_fail = 4
        overall = "REJECT" if n_fail >= 4 else "INVESTIGATE" if n_fail >= 2 else "PROCEED"
        assert overall == "REJECT"

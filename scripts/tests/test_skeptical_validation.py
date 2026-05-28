"""Tests for skeptical_validation.py — TestResult, ValidationReport, and core test functions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from skeptical_validation import (
    TestResult,
    ValidationReport,
    ALPHA,
    ENTROPY_COLS,
    FEATURE_COLS_ALL,
    FORWARD_HORIZONS,
)


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

class TestTestResult:
    def test_creation(self):
        r = TestResult(
            name="test:example",
            passed=True,
            statistic=1.5,
            p_value=0.01,
            detail="some detail",
            verdict="SURVIVES",
        )
        assert r.name == "test:example"
        assert r.passed is True
        assert r.verdict == "SURVIVES"

    def test_verdict_values(self):
        for v in ("SURVIVES", "REJECTED", "INCONCLUSIVE"):
            r = TestResult("t", True, 0.0, 0.0, "", v)
            assert r.verdict == v


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_add_and_count(self):
        report = ValidationReport()
        report.add(TestResult("a", True, 1.0, 0.01, "", "SURVIVES"))
        report.add(TestResult("b", False, 0.5, 0.1, "", "REJECTED"))
        report.add(TestResult("c", False, 0.0, 0.5, "", "INCONCLUSIVE"))
        assert len(report.tests) == 3

    def test_print_report_populates_summary(self, capsys):
        report = ValidationReport()
        report.add(TestResult("Cat:t1", True, 1.0, 0.01, "d1", "SURVIVES"))
        report.add(TestResult("Cat:t2", False, 0.5, 0.1, "d2", "REJECTED"))
        report.print_report()

        assert report.summary["total"] == 2
        assert report.summary["survived"] == 1
        assert report.summary["rejected"] == 1

    def test_to_json(self):
        report = ValidationReport()
        report.add(TestResult("t", True, 1.0, 0.01, "detail", "SURVIVES"))
        report.summary = {"total": 1}
        j = report.to_json()
        assert '"SURVIVES"' in j
        assert '"total"' in j


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_alpha_is_005(self):
        assert ALPHA == 0.05

    def test_entropy_cols_non_empty(self):
        assert len(ENTROPY_COLS) >= 3
        assert all(isinstance(c, str) for c in ENTROPY_COLS)

    def test_forward_horizons(self):
        assert isinstance(FORWARD_HORIZONS, list)
        assert all(isinstance(h, int) and h > 0 for h in FORWARD_HORIZONS)
        # Should be sorted ascending
        assert FORWARD_HORIZONS == sorted(FORWARD_HORIZONS)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic DataFrame mimicking real feature data."""
    rng = np.random.RandomState(seed)
    price = 100.0 + np.cumsum(rng.randn(n) * 0.001)
    df = pd.DataFrame({
        "raw_mid_price": price,
        "ent_permutation_8": rng.rand(n),
        "ent_permutation_16": rng.rand(n),
        "ent_permutation_32": rng.rand(n),
        "ent_book_shape": rng.rand(n),
        "ent_trade_size": rng.rand(n),
        "ent_rate_of_change": rng.randn(n) * 0.1,
        "ent_zscore": rng.randn(n),
        "flow_aggressor_ratio": rng.rand(n),
        "flow_aggressor_momentum": rng.randn(n) * 0.1,
        "flow_volume_5s": rng.exponential(100, n),
        "flow_trade_count_5s": rng.poisson(10, n).astype(float),
        "imbalance_l5": rng.randn(n) * 0.5,
        "imbalance_l10": rng.randn(n) * 0.5,
        "imbalance_persistence": rng.rand(n),
        "vol_realized_100": rng.exponential(0.01, n),
        "vol_realized_20": rng.exponential(0.01, n),
        "vol_ratio": rng.exponential(1.0, n),
        "raw_spread_bps": rng.exponential(2.0, n),
        "composite_regime_signal": rng.randn(n),
    })
    # Derived columns (as load_data would produce)
    df["returns"] = df["raw_mid_price"].pct_change()
    df["log_returns"] = np.log(df["raw_mid_price"] / df["raw_mid_price"].shift(1))
    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}"] = df["raw_mid_price"].shift(-h) / df["raw_mid_price"] - 1
    return df


# ---------------------------------------------------------------------------
# Integration: test functions don't crash on synthetic data
# ---------------------------------------------------------------------------

class TestFunctionsRunOnSyntheticData:
    """Smoke tests — each test function should run without error on synthetic data."""

    @pytest.fixture
    def df(self):
        return _make_df()

    @pytest.fixture
    def report(self):
        return ValidationReport()

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path

    def test_entropy_distribution(self, df, report, output_dir):
        from skeptical_validation import test_entropy_distribution
        test_entropy_distribution(df, report, output_dir)
        assert len(report.tests) > 0

    def test_entropy_persistence(self, df, report, output_dir):
        from skeptical_validation import test_entropy_persistence
        test_entropy_persistence(df, report, output_dir)
        assert len(report.tests) >= 2

    def test_feature_return_correlations(self, df, report, output_dir):
        from skeptical_validation import test_feature_return_correlations
        test_feature_return_correlations(df, report, output_dir)
        assert len(report.tests) > 0

    def test_feature_redundancy(self, df, report, output_dir):
        from skeptical_validation import test_feature_redundancy
        test_feature_redundancy(df, report, output_dir)
        assert len(report.tests) >= 1

    def test_return_properties(self, df, report, output_dir):
        from skeptical_validation import test_return_properties
        test_return_properties(df, report, output_dir)
        assert len(report.tests) >= 2

    def test_data_sufficiency(self, df, report, output_dir):
        from skeptical_validation import test_data_sufficiency
        test_data_sufficiency(df, report, output_dir)
        assert len(report.tests) >= 2

    def test_entropy_agreement(self, df, report, output_dir):
        from skeptical_validation import test_entropy_agreement
        test_entropy_agreement(df, report, output_dir)
        assert len(report.tests) >= 1

    def test_effect_sizes(self, df, report, output_dir):
        from skeptical_validation import test_effect_sizes
        test_effect_sizes(df, report, output_dir)
        assert len(report.tests) >= 1

    def test_walk_forward_stability(self, df, report, output_dir):
        from skeptical_validation import test_walk_forward_stability
        test_walk_forward_stability(df, report, output_dir)
        assert len(report.tests) >= 1

    def test_nonlinear_predictability(self, df, report, output_dir):
        from skeptical_validation import test_nonlinear_predictability
        test_nonlinear_predictability(df, report, output_dir)
        assert len(report.tests) >= 1

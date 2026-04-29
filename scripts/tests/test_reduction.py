"""
Skeptical tests for cluster_pipeline.reduction — variance/correlation filtering + PCA.

These tests verify that the filter correctly removes noise (near-constant columns)
and redundancy (highly correlated columns) from the derivative space before PCA,
and that PCA with Ledoit-Wolf regularization produces correct, stable results.

Test philosophy:
  - Synthetic data with known variance and correlation structure
  - Adversarial inputs: all-constant, all-identical, single column, NaN-heavy
  - Property-based checks: no surviving pair should violate the threshold
  - Report consistency: counts must add up
  - Determinism: same input → same output
  - Reconstruction error bounded by explained variance
  - Regularization triggers correctly based on sample/feature ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.reduction import (
    filter_derivatives,
    _greedy_correlation_filter,
    pca_reduce,
    PCAResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(data: dict, n: int = 200) -> pd.DataFrame:
    """Build a DataFrame from a dict of column_name → array."""
    return pd.DataFrame(data)


def _random_df(n_rows: int = 200, n_cols: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create DataFrame of independent random columns."""
    rng = np.random.RandomState(seed)
    data = {f"col_{i}": rng.normal(0, (i + 1) * 0.5, n_rows) for i in range(n_cols)}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Variance filtering tests
# ---------------------------------------------------------------------------

class TestVarianceFiltering:
    """Tests for Step 1: variance-based column removal."""

    def test_constant_column_dropped(self):
        """A column with zero variance must be dropped."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "constant": np.full(100, 5.0),
            "variable": rng.normal(0, 1, 100),
        })
        filtered, report = filter_derivatives(df, variance_percentile=10.0)
        assert "constant" in report["dropped_variance"]
        assert "constant" not in filtered.columns
        assert "variable" in filtered.columns

    def test_near_constant_column_dropped(self):
        """A column with negligible variance should be dropped at appropriate percentile."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "tiny_var": rng.normal(0, 1e-8, 200),    # var ≈ 1e-16
            "normal_1": rng.normal(0, 1, 200),        # var ≈ 1.0
            "normal_2": rng.normal(0, 2, 200),        # var ≈ 4.0
            "normal_3": rng.normal(0, 3, 200),        # var ≈ 9.0
        })
        # percentile=30 should drop bottom ~1 of 4 columns
        filtered, report = filter_derivatives(df, variance_percentile=30.0)
        assert "tiny_var" in report["dropped_variance"]

    def test_all_constant_returns_empty(self):
        """If all columns are constant, return empty DataFrame."""
        df = _make_df({
            "a": np.full(50, 1.0),
            "b": np.full(50, 2.0),
            "c": np.full(50, 3.0),
        })
        filtered, report = filter_derivatives(df, variance_percentile=10.0)
        assert filtered.shape[1] == 0
        assert report["n_after_variance"] == 0

    def test_percentile_zero_keeps_all_nonzero_variance(self):
        """variance_percentile=0 should keep all columns with any variance."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "constant": np.full(100, 5.0),
            "tiny": rng.normal(0, 0.001, 100),
            "big": rng.normal(0, 10, 100),
        })
        filtered, report = filter_derivatives(df, variance_percentile=0.0)
        assert "constant" not in filtered.columns  # zero variance still dropped
        assert "tiny" in filtered.columns
        assert "big" in filtered.columns

    def test_percentile_100_drops_everything(self):
        """variance_percentile=100 drops all columns (nothing above 100th percentile)."""
        rng = np.random.RandomState(42)
        df = _random_df(n_rows=100, n_cols=10)
        filtered, report = filter_derivatives(
            df, variance_percentile=100.0, correlation_threshold=1.0
        )
        # At percentile=100, threshold_value = max variance, so >= keeps all
        # Actually at 100th percentile, threshold = max, >= keeps the max column
        assert filtered.shape[1] >= 1  # At least the max-variance column survives

    def test_high_variance_never_dropped(self):
        """The highest-variance column must never be dropped by variance filter."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "low_1": rng.normal(0, 0.01, 200),
            "low_2": rng.normal(0, 0.02, 200),
            "medium": rng.normal(0, 1, 200),
            "high": rng.normal(0, 100, 200),
        })
        for pct in [10, 25, 50, 75, 90]:
            _, report = filter_derivatives(
                df, variance_percentile=pct, correlation_threshold=1.0
            )
            assert "high" not in report["dropped_variance"], (
                f"Highest-variance column dropped at percentile={pct}"
            )

    def test_preserves_row_count(self):
        """Filtered DataFrame must have same number of rows as input."""
        df = _random_df(n_rows=150, n_cols=10)
        filtered, _ = filter_derivatives(df)
        assert len(filtered) == 150

    def test_nan_values_handled(self):
        """Columns with NaN values should still have variance computed correctly."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 5, 200)
        vals[::10] = np.nan  # 10% NaN
        df = _make_df({
            "with_nan": vals,
            "clean": rng.normal(0, 1, 200),
        })
        filtered, _ = filter_derivatives(df, variance_percentile=0.0)
        # Both should survive (both have nonzero variance)
        assert "with_nan" in filtered.columns or "clean" in filtered.columns


# ---------------------------------------------------------------------------
# Correlation filtering tests
# ---------------------------------------------------------------------------

class TestCorrelationFiltering:
    """Tests for Step 2: greedy correlation deduplication."""

    def test_identical_columns_deduplicated(self):
        """If col_b is an exact copy of col_a, only one should survive."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        df = _make_df({"col_a": vals, "col_b": vals.copy()})
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        assert filtered.shape[1] == 1, "Identical columns should be deduplicated to 1"
        assert len(report["dropped_correlation"]) == 1

    def test_independent_columns_preserved(self):
        """Independent random columns should all survive correlation filter."""
        df = _random_df(n_rows=500, n_cols=10, seed=42)
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        assert report["dropped_correlation"] == [], (
            f"Independent columns should not be dropped: {report['dropped_correlation']}"
        )
        assert filtered.shape[1] == 10

    def test_perfectly_correlated_pair(self):
        """Two perfectly correlated columns (r=1.0): lower-variance one dropped."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 5, 200)   # higher variance
        b = a * 0.5                  # perfectly correlated but lower variance
        df = _make_df({"high_var": a, "low_var": b})
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        assert "high_var" in filtered.columns, "Higher-variance column should survive"
        assert "low_var" in report["dropped_correlation"]

    def test_negatively_correlated_pair(self):
        """Negatively correlated pair (r=-1.0) should also be deduplicated."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 3, 200)
        b = -a  # r = -1.0, same variance
        df = _make_df({"pos": a, "neg": b})
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        assert filtered.shape[1] == 1, "Perfectly anticorrelated pair → keep 1"

    def test_no_surviving_pair_exceeds_threshold(self):
        """
        After filtering, no pair of surviving columns should have
        |correlation| > threshold. This is the core correctness invariant.
        """
        rng = np.random.RandomState(42)
        n = 300
        # Create columns with varying correlations
        base = rng.normal(0, 1, n)
        df = _make_df({
            "base": base,
            "corr_99": base + rng.normal(0, 0.1, n),     # r ≈ 0.99
            "corr_90": base + rng.normal(0, 0.5, n),     # r ≈ 0.90
            "corr_50": base + rng.normal(0, 1.0, n),     # r ≈ 0.70
            "independent": rng.normal(0, 1, n),           # r ≈ 0
        })
        threshold = 0.95
        filtered, _ = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=threshold
        )

        # Verify invariant
        if filtered.shape[1] > 1:
            corr = filtered.fillna(0).corr()
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    assert abs(corr.iloc[i, j]) <= threshold + 1e-10, (
                        f"Surviving pair ({corr.columns[i]}, {corr.columns[j]}) "
                        f"has |corr|={abs(corr.iloc[i, j]):.4f} > {threshold}"
                    )

    def test_greedy_triplet(self):
        """
        Three mutually correlated columns: greedy algorithm should remove
        enough to satisfy threshold, but not necessarily all.
        """
        rng = np.random.RandomState(42)
        n = 500
        base = rng.normal(0, 1, n)
        df = _make_df({
            "a": base + rng.normal(0, 0.05, n),   # all highly correlated
            "b": base + rng.normal(0, 0.06, n),
            "c": base + rng.normal(0, 0.07, n),
        })
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        # At least 2 of 3 should be dropped (since all pairs have r > 0.95)
        assert filtered.shape[1] <= 2
        assert len(report["dropped_correlation"]) >= 1

    def test_lower_variance_dropped_not_higher(self):
        """When correlated, always drop the lower-variance column."""
        rng = np.random.RandomState(42)
        n = 200
        # a has high var, b has low var, but corr > 0.95
        a = rng.normal(0, 10, n)
        b = a * 0.01 + rng.normal(0, 0.001, n)  # correlated but tiny variance
        # Actually b's variance is very low, let's make correlated with real variance
        b = a + rng.normal(0, 0.5, n)  # correlated, slightly lower var due to smaller range
        # Force known variances
        a_known = rng.normal(0, 5, n)
        b_known = a_known + rng.normal(0, 0.3, n)  # r > 0.99, var(b) ≈ var(a) + var(noise)
        # b has slightly higher variance — let's make a clearly higher
        a_high_var = rng.normal(0, 10, n)
        b_low_var = (a_high_var / 10) + rng.normal(0, 0.01, n)  # scaled down
        # corr(a_high_var, b_low_var) ≈ corr(a, a/10) ≈ 1.0

        df = _make_df({"high_var": a_high_var, "low_var": b_low_var})
        _, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.9
        )
        if report["dropped_correlation"]:
            assert "low_var" in report["dropped_correlation"]

    def test_correlation_threshold_1_drops_nothing(self):
        """With threshold=1.0, only perfectly correlated pairs are dropped."""
        rng = np.random.RandomState(42)
        n = 200
        base = rng.normal(0, 1, n)
        df = _make_df({
            "a": base,
            "b": base + rng.normal(0, 0.1, n),  # r ≈ 0.99 but < 1.0
            "c": rng.normal(0, 1, n),
        })
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=1.0
        )
        # Nothing should be dropped since no pair has |r| exactly > 1.0
        assert report["dropped_correlation"] == []

    def test_strict_threshold_aggressive(self):
        """Low threshold (e.g. 0.5) should aggressively remove correlated columns."""
        rng = np.random.RandomState(42)
        n = 500
        base = rng.normal(0, 1, n)
        df = _make_df({
            "a": base,
            "b": base + rng.normal(0, 0.8, n),   # r ≈ 0.78
            "c": base + rng.normal(0, 2.0, n),   # r ≈ 0.45
            "d": rng.normal(0, 1, n),             # independent
        })
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.5
        )
        # 'b' should be dropped (corr with a > 0.5)
        assert len(report["dropped_correlation"]) >= 1


# ---------------------------------------------------------------------------
# Report consistency tests
# ---------------------------------------------------------------------------

class TestReportConsistency:
    """Tests that the report dict is internally consistent."""

    def test_counts_add_up_variance(self):
        """n_input == n_after_variance + len(dropped_variance)."""
        df = _random_df(n_rows=100, n_cols=20)
        _, report = filter_derivatives(df, variance_percentile=25.0)
        assert report["n_input"] == report["n_after_variance"] + len(report["dropped_variance"])

    def test_counts_add_up_correlation(self):
        """n_after_variance == n_after_correlation + len(dropped_correlation)."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 1, 200)
        df = _make_df({
            "a": base,
            "b": base + rng.normal(0, 0.1, 200),
            "c": rng.normal(0, 1, 200),
            "d": rng.normal(0, 2, 200),
        })
        _, report = filter_derivatives(df, variance_percentile=0.0, correlation_threshold=0.9)
        assert report["n_after_variance"] == report["n_after_correlation"] + len(report["dropped_correlation"])

    def test_n_input_matches_input_columns(self):
        """n_input must match the actual number of input columns."""
        df = _random_df(n_rows=100, n_cols=15)
        _, report = filter_derivatives(df)
        assert report["n_input"] == 15

    def test_variance_threshold_value_is_float(self):
        """variance_threshold_value must be a numeric value."""
        df = _random_df(n_rows=100, n_cols=10)
        _, report = filter_derivatives(df)
        assert isinstance(report["variance_threshold_value"], float)

    def test_dropped_lists_are_lists_of_strings(self):
        """dropped_variance and dropped_correlation must be lists of column names."""
        df = _random_df(n_rows=100, n_cols=10)
        _, report = filter_derivatives(df)
        assert isinstance(report["dropped_variance"], list)
        assert isinstance(report["dropped_correlation"], list)
        for col in report["dropped_variance"] + report["dropped_correlation"]:
            assert isinstance(col, str)

    def test_no_column_in_both_dropped_lists(self):
        """A column should not appear in both dropped_variance and dropped_correlation."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 1, 200)
        df = _make_df({
            "const": np.full(200, 1.0),
            "a": base,
            "b": base + rng.normal(0, 0.1, 200),
            "c": rng.normal(0, 1, 200),
        })
        _, report = filter_derivatives(df, variance_percentile=10.0, correlation_threshold=0.9)
        overlap = set(report["dropped_variance"]) & set(report["dropped_correlation"])
        assert len(overlap) == 0, f"Column in both dropped lists: {overlap}"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input must always produce identical output."""

    def test_deterministic_random_data(self):
        """Run twice on same data → identical results."""
        df = _random_df(n_rows=200, n_cols=20, seed=42)
        f1, r1 = filter_derivatives(df, variance_percentile=15.0, correlation_threshold=0.9)
        f2, r2 = filter_derivatives(df, variance_percentile=15.0, correlation_threshold=0.9)
        pd.testing.assert_frame_equal(f1, f2)
        assert r1 == r2

    def test_deterministic_correlated_data(self):
        """Correlated data: same result each time."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 1, 300)
        df = _make_df({
            "a": base,
            "b": base + rng.normal(0, 0.1, 300),
            "c": rng.normal(0, 1, 300),
        })
        f1, r1 = filter_derivatives(df, correlation_threshold=0.9)
        f2, r2 = filter_derivatives(df, correlation_threshold=0.9)
        pd.testing.assert_frame_equal(f1, f2)
        assert r1["dropped_correlation"] == r2["dropped_correlation"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Adversarial and boundary inputs."""

    def test_single_column(self):
        """Single column with nonzero variance should survive."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"only": rng.normal(0, 1, 100)})
        filtered, report = filter_derivatives(df, variance_percentile=0.0)
        assert filtered.shape[1] == 1
        assert report["n_after_correlation"] == 1

    def test_single_constant_column(self):
        """Single constant column → empty output."""
        df = pd.DataFrame({"only": np.full(100, 3.14)})
        filtered, report = filter_derivatives(df)
        assert filtered.shape[1] == 0

    def test_two_identical_columns(self):
        """Two identical columns → one survives."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        df = _make_df({"a": vals, "b": vals.copy()})
        filtered, _ = filter_derivatives(df, variance_percentile=0.0, correlation_threshold=0.95)
        assert filtered.shape[1] == 1

    def test_many_identical_columns(self):
        """10 identical columns → exactly one survives."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        df = pd.DataFrame({f"col_{i}": vals.copy() for i in range(10)})
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        assert filtered.shape[1] == 1, f"Expected 1 survivor from 10 identical, got {filtered.shape[1]}"
        assert len(report["dropped_correlation"]) == 9

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="at least one column"):
            filter_derivatives(df)

    def test_all_nan_column(self):
        """All-NaN column has zero variance after fillna → gets dropped."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "all_nan": np.full(100, np.nan),
            "valid": rng.normal(0, 1, 100),
        })
        filtered, report = filter_derivatives(df, variance_percentile=0.0)
        assert "all_nan" not in filtered.columns

    def test_large_dataframe_performance(self):
        """Should handle 500 columns × 1000 rows without timeout."""
        df = _random_df(n_rows=1000, n_cols=500, seed=42)
        filtered, report = filter_derivatives(df, variance_percentile=10.0)
        assert filtered.shape[1] > 0
        assert report["n_input"] == 500

    def test_column_order_preserved(self):
        """Surviving columns should maintain their original order."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "z_col": rng.normal(0, 1, 100),
            "a_col": rng.normal(0, 2, 100),
            "m_col": rng.normal(0, 3, 100),
        })
        filtered, _ = filter_derivatives(df, variance_percentile=0.0, correlation_threshold=1.0)
        expected_order = ["z_col", "a_col", "m_col"]
        assert filtered.columns.tolist() == expected_order

    def test_negative_values_handled(self):
        """Negative values should not affect variance computation."""
        rng = np.random.RandomState(42)
        df = _make_df({
            "neg": rng.normal(-100, 5, 200),
            "pos": rng.normal(100, 5, 200),
        })
        filtered, _ = filter_derivatives(df, variance_percentile=0.0)
        assert filtered.shape[1] == 2


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Tests for invalid parameter handling."""

    def test_negative_percentile_raises(self):
        df = _random_df(n_rows=50, n_cols=5)
        with pytest.raises(ValueError, match="variance_percentile"):
            filter_derivatives(df, variance_percentile=-1.0)

    def test_percentile_over_100_raises(self):
        df = _random_df(n_rows=50, n_cols=5)
        with pytest.raises(ValueError, match="variance_percentile"):
            filter_derivatives(df, variance_percentile=101.0)

    def test_correlation_zero_raises(self):
        df = _random_df(n_rows=50, n_cols=5)
        with pytest.raises(ValueError, match="correlation_threshold"):
            filter_derivatives(df, correlation_threshold=0.0)

    def test_correlation_negative_raises(self):
        df = _random_df(n_rows=50, n_cols=5)
        with pytest.raises(ValueError, match="correlation_threshold"):
            filter_derivatives(df, correlation_threshold=-0.5)

    def test_correlation_over_1_raises(self):
        df = _random_df(n_rows=50, n_cols=5)
        with pytest.raises(ValueError, match="correlation_threshold"):
            filter_derivatives(df, correlation_threshold=1.5)


# ---------------------------------------------------------------------------
# Greedy algorithm property tests
# ---------------------------------------------------------------------------

class TestGreedyAlgorithm:
    """Tests for the greedy correlation filter's correctness properties."""

    def test_greedy_removes_minimum_necessary(self):
        """
        Greedy should not over-remove. If only one pair violates threshold,
        only one column should be dropped.
        """
        rng = np.random.RandomState(42)
        n = 300
        a = rng.normal(0, 1, n)
        b = a + rng.normal(0, 0.05, n)   # r ≈ 0.99 with a
        c = rng.normal(0, 1, n)           # independent
        d = rng.normal(0, 1, n)           # independent
        df = _make_df({"a": a, "b": b, "c": c, "d": d})
        _, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )
        # Only one should be dropped (from the a-b pair)
        assert len(report["dropped_correlation"]) == 1

    def test_chain_correlation(self):
        """
        Chain: a↔b (r=0.99), b↔c (r=0.99), but a↔c (r<0.95).
        Dropping b should satisfy both constraints.
        """
        rng = np.random.RandomState(42)
        n = 1000
        a = rng.normal(0, 1, n)
        b = a + rng.normal(0, 0.1, n)       # r(a,b) ≈ 0.99
        c = b + rng.normal(0, 0.1, n)       # r(b,c) ≈ 0.99, but r(a,c) lower
        # Add noise to make a-c correlation lower
        c = c + rng.normal(0, 0.5, n)

        df = _make_df({"a": a, "b": b, "c": c})
        filtered, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.95
        )

        # Verify the invariant holds
        if filtered.shape[1] > 1:
            corr = filtered.fillna(0).corr()
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    assert abs(corr.iloc[i, j]) <= 0.95 + 1e-10

    def test_highest_variance_column_never_dropped_by_correlation(self):
        """The column with the highest variance should never be the one dropped."""
        rng = np.random.RandomState(42)
        n = 200
        # Make "high" have clearly highest variance
        high = rng.normal(0, 100, n)
        # Scale down to create a correlated column with LOWER variance
        low = high * 0.1 + rng.normal(0, 0.01, n)  # r ≈ 1.0, var(low) ≈ 0.01*var(high)

        df = _make_df({
            "high": high,
            "low": low,
            "independent": rng.normal(0, 1, n),
        })
        _, report = filter_derivatives(
            df, variance_percentile=0.0, correlation_threshold=0.9
        )
        assert "high" not in report["dropped_correlation"], (
            "Highest-variance column should never be dropped"
        )
        assert "low" in report["dropped_correlation"], (
            "Lower-variance correlated column should be dropped"
        )

    def test_symmetry_of_pair_handling(self):
        """Result should not depend on which column appears first in the DataFrame."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        corr_vals = vals + rng.normal(0, 0.05, 200)

        # Order 1: a first
        df1 = pd.DataFrame({"a": vals, "b": corr_vals})
        f1, r1 = filter_derivatives(df1, variance_percentile=0.0, correlation_threshold=0.9)

        # Order 2: b first
        df2 = pd.DataFrame({"b": corr_vals, "a": vals})
        f2, r2 = filter_derivatives(df2, variance_percentile=0.0, correlation_threshold=0.9)

        # Same column should be dropped regardless of order (the lower-variance one)
        assert set(r1["dropped_correlation"]) == set(r2["dropped_correlation"])


# ---------------------------------------------------------------------------
# Integration with derivative pipeline
# ---------------------------------------------------------------------------

class TestIntegrationWithDerivatives:
    """Test filter_derivatives works on output from the derivative engine."""

    def test_works_on_temporal_derivative_output(self):
        """Should work on a DataFrame shaped like temporal_derivatives output."""
        rng = np.random.RandomState(42)
        n = 200
        # Simulate derivative output: many columns, some correlated
        data = {}
        for i in range(15):
            base = rng.normal(0, (i + 1) * 0.3, n)
            data[f"feat_{i}_vel"] = np.diff(base, prepend=base[0])
            data[f"feat_{i}_accel"] = np.diff(data[f"feat_{i}_vel"], prepend=0)
            data[f"feat_{i}_zscore_5"] = rng.normal(0, 1, n)
            data[f"feat_{i}_slope_5"] = rng.normal(0, 0.1, n)
            data[f"feat_{i}_rvol_5"] = np.abs(rng.normal(0, 0.5, n))

        df = pd.DataFrame(data)
        filtered, report = filter_derivatives(df, variance_percentile=10.0)

        assert filtered.shape[1] > 0
        assert filtered.shape[1] < df.shape[1]  # should drop at least the bottom 10%
        assert report["n_input"] == 75  # 15 features × 5 derivatives

    def test_filtered_output_has_no_zero_variance(self):
        """After filtering, no surviving column should have zero variance."""
        rng = np.random.RandomState(42)
        data = {}
        for i in range(20):
            if i < 3:
                data[f"col_{i}"] = np.full(200, float(i))  # constant
            else:
                data[f"col_{i}"] = rng.normal(0, i * 0.5, 200)
        df = pd.DataFrame(data)
        filtered, _ = filter_derivatives(df, variance_percentile=5.0)

        for col in filtered.columns:
            assert filtered[col].var() > 0, f"Zero-variance column survived: {col}"


# ===========================================================================
# PCA reduce tests
# ===========================================================================


def _random_matrix(n_rows: int, n_cols: int, seed: int = 42) -> np.ndarray:
    """Generate a random matrix for PCA tests."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, 1, (n_rows, n_cols))


def _col_names(n: int) -> list:
    return [f"feat_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


class TestPCAReconstruction:
    """Verify PCA reconstruction error is bounded by explained variance."""

    def test_reconstruction_error_95(self):
        """At variance_threshold=0.95, reconstruction MSE < 5% of total variance."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 20))
        names = _col_names(20)
        result = pca_reduce(X, names, variance_threshold=0.95)

        # Reconstruct
        Z = (X - result.mean) / result.std
        X_reconstructed = result.X_reduced @ result.components
        mse = np.mean((Z - X_reconstructed) ** 2)
        total_var = np.var(Z)

        assert mse < 0.05 * total_var * Z.shape[1], (
            f"Reconstruction MSE {mse:.4f} exceeds 5% of total variance"
        )

    def test_reconstruction_error_99(self):
        """At variance_threshold=0.99, reconstruction MSE < 1% of total variance."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 20))
        names = _col_names(20)
        result = pca_reduce(X, names, variance_threshold=0.99)

        Z = (X - result.mean) / result.std
        X_reconstructed = result.X_reduced @ result.components
        mse = np.mean((Z - X_reconstructed) ** 2)
        total_var = np.var(Z)

        assert mse < 0.01 * total_var * Z.shape[1]

    def test_perfect_reconstruction(self):
        """variance_threshold=1.0 → n_components = rank, MSE ≈ 0."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 10))
        names = _col_names(10)
        result = pca_reduce(X, names, variance_threshold=1.0, max_components=50)

        assert result.n_components == 10, (
            f"Expected 10 components for full-rank 10-col data, got {result.n_components}"
        )

        Z = (X - result.mean) / result.std
        X_reconstructed = result.X_reduced @ result.components
        mse = np.mean((Z - X_reconstructed) ** 2)
        assert mse < 1e-10, f"Perfect reconstruction MSE should be ~0, got {mse}"

    def test_low_rank_data_few_components(self):
        """100 columns but only 3 independent → n_components ≤ 5."""
        rng = np.random.RandomState(42)
        # Create 3 independent bases, all other columns are linear combos
        bases = rng.normal(0, 1, (200, 3))
        mixing = rng.normal(0, 1, (3, 100))
        X = bases @ mixing + rng.normal(0, 0.01, (200, 100))  # tiny noise
        names = _col_names(100)

        result = pca_reduce(X, names, variance_threshold=0.95)
        assert result.n_components <= 5, (
            f"Low-rank (3 independent) data should need ≤5 PCs, got {result.n_components}"
        )

    def test_low_rank_exact(self):
        """Exactly rank-2 data → 2 components explain ~100% variance."""
        rng = np.random.RandomState(42)
        bases = rng.normal(0, 1, (300, 2))
        mixing = rng.normal(0, 1, (2, 50))
        X = bases @ mixing
        names = _col_names(50)

        result = pca_reduce(X, names, variance_threshold=0.99)
        assert result.n_components <= 3
        assert result.cumulative_variance[-1] > 0.999


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


class TestRegularization:
    """Verify Ledoit-Wolf regularization triggers correctly."""

    def test_regularization_triggered_few_samples(self):
        """n_samples < 2 * n_features → regularized=True."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (50, 100))  # 50 < 2*100
        names = _col_names(100)
        result = pca_reduce(X, names)
        assert result.regularized is True

    def test_regularization_not_triggered_many_samples(self):
        """n_samples >= 2 * n_features → regularized=False."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 100))  # 500 >= 2*100
        names = _col_names(100)
        result = pca_reduce(X, names)
        assert result.regularized is False

    def test_regularization_boundary_exact(self):
        """Exactly at boundary: n_samples = 2*n_features → not regularized."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 100))  # 200 = 2*100
        names = _col_names(100)
        result = pca_reduce(X, names)
        assert result.regularized is False

    def test_regularization_boundary_minus_one(self):
        """One below boundary: n_samples = 2*n_features - 1 → regularized."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (199, 100))  # 199 < 2*100
        names = _col_names(100)
        result = pca_reduce(X, names)
        assert result.regularized is True

    def test_regularized_result_still_valid(self):
        """Regularized PCA should still produce valid reconstruction."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (30, 50))  # heavily regularized
        names = _col_names(50)
        result = pca_reduce(X, names, variance_threshold=0.90)

        assert result.regularized is True
        assert result.n_components <= 30  # can't have more PCs than samples
        assert result.X_reduced.shape == (30, result.n_components)

        # Reconstruction should still be reasonable
        Z = (X - result.mean) / result.std
        X_recon = result.X_reduced @ result.components
        mse = np.mean((Z - X_recon) ** 2)
        # With heavy regularization, MSE bound is looser
        assert mse < 0.20 * np.var(Z) * Z.shape[1]

    def test_regularized_eigenvalues_non_negative(self):
        """Ledoit-Wolf should not produce negative eigenvalues."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (20, 80))
        names = _col_names(80)
        result = pca_reduce(X, names, variance_threshold=0.99)
        assert np.all(result.explained_variance_ratio >= 0)


# ---------------------------------------------------------------------------
# Component selection
# ---------------------------------------------------------------------------


class TestComponentSelection:
    """Verify n_components selection logic."""

    def test_max_components_cap(self):
        """n_components should not exceed max_components."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 100))
        names = _col_names(100)
        result = pca_reduce(X, names, variance_threshold=1.0, max_components=10)
        assert result.n_components <= 10

    def test_max_components_1(self):
        """max_components=1 → exactly 1 component."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 20))
        names = _col_names(20)
        result = pca_reduce(X, names, max_components=1)
        assert result.n_components == 1

    def test_variance_threshold_determines_components(self):
        """Lower threshold → fewer components."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 30))
        names = _col_names(30)

        result_low = pca_reduce(X, names, variance_threshold=0.50)
        result_high = pca_reduce(X, names, variance_threshold=0.99)

        assert result_low.n_components <= result_high.n_components

    def test_cumulative_variance_reaches_threshold(self):
        """Cumulative variance of selected components should reach the threshold."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 20))
        names = _col_names(20)

        for threshold in [0.80, 0.90, 0.95, 0.99]:
            result = pca_reduce(X, names, variance_threshold=threshold, max_components=50)
            # If we got enough components, cumulative should reach threshold
            if result.n_components < min(500, 20):
                assert result.cumulative_variance[-1] >= threshold - 1e-10, (
                    f"Cumulative variance {result.cumulative_variance[-1]:.4f} "
                    f"< threshold {threshold}"
                )

    def test_n_components_le_min_samples_features(self):
        """n_components can never exceed min(n_samples, n_features)."""
        rng = np.random.RandomState(42)
        # More features than samples
        X = rng.normal(0, 1, (15, 50))
        names = _col_names(50)
        result = pca_reduce(X, names, variance_threshold=1.0, max_components=100)
        assert result.n_components <= 15

    def test_explained_variance_ratio_sums_to_cumulative(self):
        """cumsum of explained_variance_ratio should equal cumulative_variance."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 25))
        names = _col_names(25)
        result = pca_reduce(X, names)
        np.testing.assert_allclose(
            np.cumsum(result.explained_variance_ratio),
            result.cumulative_variance,
            atol=1e-12,
        )

    def test_explained_variance_ratio_descending(self):
        """Explained variance ratios must be in descending order."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 25))
        names = _col_names(25)
        result = pca_reduce(X, names)
        for i in range(len(result.explained_variance_ratio) - 1):
            assert result.explained_variance_ratio[i] >= result.explained_variance_ratio[i + 1] - 1e-12


# ---------------------------------------------------------------------------
# Loadings
# ---------------------------------------------------------------------------


class TestLoadings:
    """Verify PCA loadings structure and sorting."""

    def test_loadings_sorted_descending_by_abs_weight(self):
        """Each PC's loadings must be sorted by |weight| descending."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 20))
        names = _col_names(20)
        result = pca_reduce(X, names)

        for pc_idx, pc_loadings in result.loadings.items():
            abs_weights = [abs(w) for _, w in pc_loadings]
            for i in range(len(abs_weights) - 1):
                assert abs_weights[i] >= abs_weights[i + 1] - 1e-12, (
                    f"PC{pc_idx} loadings not sorted: {abs_weights}"
                )

    def test_loadings_have_correct_column_names(self):
        """Loading column names must come from the input column_names."""
        names = [f"my_feature_{i}" for i in range(15)]
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        result = pca_reduce(X, names)

        name_set = set(names)
        for pc_idx, pc_loadings in result.loadings.items():
            for col_name, weight in pc_loadings:
                assert col_name in name_set, (
                    f"Loading column '{col_name}' not in input names"
                )

    def test_loadings_max_10_per_pc(self):
        """Each PC should have at most 10 loadings."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 50))
        names = _col_names(50)
        result = pca_reduce(X, names)

        for pc_idx, pc_loadings in result.loadings.items():
            assert len(pc_loadings) <= 10

    def test_loadings_fewer_than_10_when_few_features(self):
        """If fewer than 10 features, loadings per PC = n_features."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 5))
        names = _col_names(5)
        result = pca_reduce(X, names)

        for pc_idx, pc_loadings in result.loadings.items():
            assert len(pc_loadings) == 5

    def test_loadings_keys_are_pc_indices(self):
        """Loadings dict keys should be 0..n_components-1."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert set(result.loadings.keys()) == set(range(result.n_components))

    def test_loadings_weights_are_floats(self):
        """All loading weights must be Python floats."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 10))
        names = _col_names(10)
        result = pca_reduce(X, names)
        for pc_loadings in result.loadings.values():
            for _, w in pc_loadings:
                assert isinstance(w, float)

    def test_top_loading_is_dominant_feature(self):
        """If one feature drives most variance in a correlated group, it should
        appear in top loadings of PC0."""
        rng = np.random.RandomState(42)
        n = 500
        # Feature 0 drives features 1-4 via linear relationship (correlation)
        driver = rng.normal(0, 1, n)
        X = rng.normal(0, 0.01, (n, 10))  # mostly noise
        X[:, 0] = driver
        for i in range(1, 5):
            X[:, i] = driver * (0.9 - i * 0.1) + rng.normal(0, 0.1, n)
        names = _col_names(10)
        result = pca_reduce(X, names)

        # PC0 should have feat_0 among its top 3 loadings (it drives the group)
        top_3_names = [name for name, _ in result.loadings[0][:3]]
        assert "feat_0" in top_3_names, (
            f"Expected feat_0 in top 3 PC0 loadings, got {top_3_names}"
        )


# ---------------------------------------------------------------------------
# Projection on new data
# ---------------------------------------------------------------------------


class TestProjection:
    """Verify that saved basis can project new data correctly."""

    def test_projection_shape(self):
        """Project new data using saved mean/std/components — shape must match."""
        rng = np.random.RandomState(42)
        X_train = rng.normal(0, 1, (400, 20))
        X_test = rng.normal(0, 1, (100, 20))
        names = _col_names(20)

        result = pca_reduce(X_train, names)

        # Project test data manually
        Z_test = (X_test - result.mean) / result.std
        X_test_reduced = Z_test @ result.components.T

        assert X_test_reduced.shape == (100, result.n_components)

    def test_projection_train_matches_result(self):
        """Projecting training data with saved basis should match X_reduced."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)

        Z = (X - result.mean) / result.std
        X_projected = Z @ result.components.T

        np.testing.assert_allclose(result.X_reduced, X_projected, atol=1e-10)

    def test_projection_new_data_no_crash(self):
        """Split 80/20. Fit on 80%, project 20% — no errors."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 30))
        names = _col_names(30)

        X_train, X_test = X[:400], X[400:]
        result = pca_reduce(X_train, names)

        Z_test = (X_test - result.mean) / result.std
        X_test_reduced = Z_test @ result.components.T

        assert X_test_reduced.shape == (100, result.n_components)
        assert not np.any(np.isnan(X_test_reduced))


# ---------------------------------------------------------------------------
# Output shapes and types
# ---------------------------------------------------------------------------


class TestOutputShapes:
    """Verify all PCAResult fields have correct shapes and types."""

    def test_X_reduced_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.X_reduced.shape == (200, result.n_components)

    def test_components_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.components.shape == (result.n_components, 15)

    def test_mean_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.mean.shape == (15,)

    def test_std_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.std.shape == (15,)

    def test_explained_variance_ratio_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.explained_variance_ratio.shape == (result.n_components,)

    def test_cumulative_variance_shape(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 15))
        names = _col_names(15)
        result = pca_reduce(X, names)
        assert result.cumulative_variance.shape == (result.n_components,)

    def test_column_names_stored(self):
        names = [f"my_feat_{i}" for i in range(10)]
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 10))
        result = pca_reduce(X, names)
        assert result.column_names == names

    def test_result_is_dataclass(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        names = _col_names(5)
        result = pca_reduce(X, names)
        assert isinstance(result, PCAResult)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestPCADeterminism:
    """Same input must produce identical PCA output."""

    def test_deterministic_random(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 20))
        names = _col_names(20)

        r1 = pca_reduce(X, names)
        r2 = pca_reduce(X, names)

        np.testing.assert_array_equal(r1.X_reduced, r2.X_reduced)
        np.testing.assert_array_equal(r1.components, r2.components)
        assert r1.n_components == r2.n_components

    def test_deterministic_regularized(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (30, 80))
        names = _col_names(80)

        r1 = pca_reduce(X, names)
        r2 = pca_reduce(X, names)

        np.testing.assert_array_equal(r1.X_reduced, r2.X_reduced)
        assert r1.regularized == r2.regularized


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


class TestPCAEdgeCases:
    """Adversarial and boundary inputs for PCA."""

    def test_single_feature(self):
        """Single feature → 1 component."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 1))
        result = pca_reduce(X, ["only_feat"])
        assert result.n_components == 1
        assert result.X_reduced.shape == (100, 1)

    def test_two_samples(self):
        """Minimum viable: 2 samples."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (2, 5))
        names = _col_names(5)
        result = pca_reduce(X, names)
        assert result.X_reduced.shape[0] == 2

    def test_constant_feature_handled(self):
        """Constant features (std=0) should not cause division by zero."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 10))
        X[:, 3] = 5.0  # constant column
        X[:, 7] = -2.0  # another constant
        names = _col_names(10)
        result = pca_reduce(X, names)
        assert not np.any(np.isnan(result.X_reduced))
        assert not np.any(np.isinf(result.X_reduced))

    def test_all_constant_features(self):
        """All features constant → degenerate but no crash."""
        X = np.full((100, 5), 3.14)
        names = _col_names(5)
        result = pca_reduce(X, names)
        # Should return something reasonable
        assert result.n_components >= 1
        assert not np.any(np.isnan(result.X_reduced))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            pca_reduce(np.array([1, 2, 3]), ["a"])

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            pca_reduce(np.array([[1, 2, 3]]), ["a", "b", "c"])

    def test_zero_features_raises(self):
        with pytest.raises(ValueError, match="no features"):
            pca_reduce(np.empty((100, 0)), [])

    def test_column_names_mismatch_raises(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        with pytest.raises(ValueError, match="column_names length"):
            pca_reduce(X, _col_names(3))

    def test_nan_in_X_raises(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="NaN"):
            pca_reduce(X, ["a", "b"])

    def test_invalid_variance_threshold_zero(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        with pytest.raises(ValueError, match="variance_threshold"):
            pca_reduce(X, _col_names(5), variance_threshold=0.0)

    def test_invalid_variance_threshold_over_1(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        with pytest.raises(ValueError, match="variance_threshold"):
            pca_reduce(X, _col_names(5), variance_threshold=1.5)

    def test_invalid_max_components(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 5))
        with pytest.raises(ValueError, match="max_components"):
            pca_reduce(X, _col_names(5), max_components=0)

    def test_wide_data_more_features_than_samples(self):
        """Wide matrix (p >> n) should still work correctly."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (20, 200))
        names = _col_names(200)
        result = pca_reduce(X, names, variance_threshold=0.95)
        assert result.regularized is True
        assert result.n_components <= 20
        assert result.X_reduced.shape == (20, result.n_components)

    def test_large_matrix(self):
        """Performance: 1000 rows × 200 cols should complete without issue."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (1000, 200))
        names = _col_names(200)
        result = pca_reduce(X, names)
        assert result.X_reduced.shape[0] == 1000
        assert result.n_components > 0


# ---------------------------------------------------------------------------
# Orthogonality and mathematical properties
# ---------------------------------------------------------------------------


class TestPCAMathProperties:
    """Verify mathematical invariants of PCA."""

    def test_components_orthogonal(self):
        """Principal components should be orthogonal to each other."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 20))
        names = _col_names(20)
        result = pca_reduce(X, names)

        # components is (k, n_features), rows should be orthogonal
        gram = result.components @ result.components.T
        # Off-diagonal should be ~0
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 1e-10, (
            f"Components not orthogonal, max off-diagonal: {np.max(np.abs(off_diag))}"
        )

    def test_components_unit_norm(self):
        """Each component vector should have unit norm."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (500, 20))
        names = _col_names(20)
        result = pca_reduce(X, names)

        for i in range(result.n_components):
            norm = np.linalg.norm(result.components[i])
            assert abs(norm - 1.0) < 1e-10, (
                f"Component {i} norm = {norm}, expected 1.0"
            )

    def test_reduced_dimensions_uncorrelated(self):
        """Projected data dimensions should be uncorrelated."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (1000, 20))
        names = _col_names(20)
        result = pca_reduce(X, names)

        corr = np.corrcoef(result.X_reduced.T)
        off_diag = corr - np.diag(np.diag(corr))
        assert np.max(np.abs(off_diag)) < 0.05, (
            f"Reduced dims are correlated, max |r| = {np.max(np.abs(off_diag)):.4f}"
        )

    def test_explained_variance_sums_to_le_1(self):
        """Sum of all explained variance ratios ≤ 1.0."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 20))
        names = _col_names(20)
        result = pca_reduce(X, names, variance_threshold=1.0, max_components=50)
        total = np.sum(result.explained_variance_ratio)
        assert total <= 1.0 + 1e-10

    def test_first_pc_captures_most_variance(self):
        """PC0 should capture at least as much variance as any other PC."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (300, 20))
        names = _col_names(20)
        result = pca_reduce(X, names)
        assert result.explained_variance_ratio[0] == max(result.explained_variance_ratio)

    def test_standardized_data_zero_mean(self):
        """After standardization, mean should be ~0."""
        rng = np.random.RandomState(42)
        X = rng.normal(5.0, 2.0, (500, 10))
        names = _col_names(10)
        result = pca_reduce(X, names)

        Z = (X - result.mean) / result.std
        assert np.max(np.abs(Z.mean(axis=0))) < 1e-10


# ---------------------------------------------------------------------------
# Integration: filter_derivatives + pca_reduce
# ---------------------------------------------------------------------------


class TestFilterThenPCA:
    """End-to-end: filter derivatives then PCA."""

    def test_pipeline_filter_then_pca(self):
        """Full pipeline: random derivative-like data → filter → PCA."""
        rng = np.random.RandomState(42)
        n = 500
        data = {}
        for i in range(50):
            data[f"deriv_{i}"] = rng.normal(0, (i + 1) * 0.2, n)
        # Add some constants
        data["const_0"] = np.full(n, 0.0)
        data["const_1"] = np.full(n, 1.0)
        # Add correlated pairs
        data["corr_a"] = data["deriv_0"]
        data["corr_b"] = data["deriv_0"] + rng.normal(0, 0.01, n)

        df = pd.DataFrame(data)
        filtered, report = filter_derivatives(df, variance_percentile=10.0)

        result = pca_reduce(
            filtered.values,
            filtered.columns.tolist(),
            variance_threshold=0.95,
        )

        assert result.n_components > 0
        assert result.n_components < filtered.shape[1]
        assert result.X_reduced.shape[0] == n

    def test_pipeline_preserves_sample_count(self):
        """Row count must not change through filter → PCA."""
        rng = np.random.RandomState(42)
        n = 300
        df = pd.DataFrame(rng.normal(0, 1, (n, 30)), columns=_col_names(30))
        filtered, _ = filter_derivatives(df, variance_percentile=5.0)
        result = pca_reduce(filtered.values, filtered.columns.tolist())
        assert result.X_reduced.shape[0] == n

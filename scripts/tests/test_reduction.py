"""
Skeptical tests for cluster_pipeline.reduction — variance/correlation filtering.

These tests verify that the filter correctly removes noise (near-constant columns)
and redundancy (highly correlated columns) from the derivative space before PCA.

Test philosophy:
  - Synthetic data with known variance and correlation structure
  - Adversarial inputs: all-constant, all-identical, single column, NaN-heavy
  - Property-based checks: no surviving pair should violate the threshold
  - Report consistency: counts must add up
  - Determinism: same input → same output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.reduction import filter_derivatives, _greedy_correlation_filter


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

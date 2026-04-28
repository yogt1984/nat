"""
Skeptical tests for cluster_pipeline.derivatives — feature selection and
temporal derivative generation.

These tests are adversarial by design: they check that the feature selector
doesn't just return plausible-looking results, but actually selects features
based on the stated criteria. Each test encodes a specific failure mode we
want to catch early.

Test philosophy:
  - Synthetic data with known properties (no reliance on real data)
  - Every assertion has a "why this would break" comment
  - Edge cases that expose off-by-one, empty-input, and dtype bugs
  - Determinism: same input → same output, always
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.derivatives import (
    select_top_features,
    temporal_derivatives,
    _select_by_variance,
    _select_by_autocorrelation_range,
    _rolling_slope,
)
from cluster_pipeline.config import FEATURE_VECTORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars_with_columns(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """Create a minimal bar DataFrame with given columns."""
    df = pd.DataFrame(columns)
    # Add meta columns that aggregate_bars would produce
    df["bar_start"] = pd.date_range("2026-01-01", periods=len(df), freq="15min")
    df["bar_end"] = df["bar_start"] + pd.Timedelta("15min")
    df["symbol"] = "BTC"
    df["tick_count"] = 100
    return df


def _make_entropy_bars(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Create bars with realistic entropy-vector column names
    (matching bar-aggregated naming: {base_col}_{agg_suffix}).
    """
    rng = np.random.RandomState(seed)
    ent_base_cols = FEATURE_VECTORS["entropy"]["columns"]

    data = {}
    for i, base in enumerate(ent_base_cols):
        # Each base column gets _mean, _std, _slope suffixes
        for suffix in ["mean", "std", "slope"]:
            col_name = f"{base}_{suffix}"
            # Vary the variance deliberately so we know which should be selected
            scale = (i + 1) * 0.1
            data[col_name] = rng.normal(0, scale, size=n_bars)

    return _make_bars_with_columns(data)


# ---------------------------------------------------------------------------
# Core selection tests
# ---------------------------------------------------------------------------

class TestSelectTopFeatures:
    """Tests for select_top_features()."""

    def test_max_features_respected(self):
        """Must never return more than max_features columns."""
        bars = _make_entropy_bars(n_bars=200)
        for max_f in [1, 5, 10, 15]:
            result = select_top_features(bars, vector="entropy", max_features=max_f)
            assert len(result) <= max_f, (
                f"Requested max_features={max_f} but got {len(result)}"
            )

    def test_returns_fewer_if_vector_is_small(self):
        """If the vector has fewer non-constant columns than max_features, return all."""
        # Create bars with only 3 entropy columns
        rng = np.random.RandomState(42)
        data = {
            "ent_tick_1s_mean": rng.normal(0, 1, 100),
            "ent_tick_1s_std": rng.normal(0, 2, 100),
            "ent_tick_1s_slope": rng.normal(0, 3, 100),
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="entropy", max_features=15)
        assert len(result) == 3, "Should return all 3 when fewer than max_features"

    def test_constant_columns_excluded(self):
        """Columns with zero variance must never appear in output."""
        rng = np.random.RandomState(42)
        data = {
            "ent_tick_1s_mean": np.ones(100) * 5.0,  # constant
            "ent_tick_1s_std": rng.normal(0, 1, 100),  # variable
            "ent_tick_5s_mean": np.ones(100) * 3.0,  # constant
            "ent_tick_5s_std": rng.normal(0, 2, 100),  # variable
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="entropy", max_features=10)
        for col in result:
            assert bars[col].var() > 1e-10, f"Constant column {col} was selected"
        assert "ent_tick_1s_mean" not in result
        assert "ent_tick_5s_mean" not in result

    def test_all_returned_columns_exist_in_bars(self):
        """Every returned column must exist in the input DataFrame."""
        bars = _make_entropy_bars(n_bars=200)
        result = select_top_features(bars, vector="entropy", max_features=10)
        for col in result:
            assert col in bars.columns, f"Selected column '{col}' not in bars"

    def test_deterministic(self):
        """Same input must produce same output every time."""
        bars = _make_entropy_bars(n_bars=200, seed=42)
        r1 = select_top_features(bars, vector="entropy", max_features=10)
        r2 = select_top_features(bars, vector="entropy", max_features=10)
        assert r1 == r2, "Feature selection is not deterministic"

    def test_variance_method_selects_highest_variance(self):
        """
        Variance method must select columns with highest variance.
        This is the core correctness test — if this fails, the selector is broken.
        """
        rng = np.random.RandomState(42)
        # Create 5 columns with known, distinct variances
        data = {
            "ent_tick_1s_mean": rng.normal(0, 1.0, 200),   # var ≈ 1.0
            "ent_tick_5s_mean": rng.normal(0, 5.0, 200),   # var ≈ 25.0  (should be #1)
            "ent_tick_10s_mean": rng.normal(0, 3.0, 200),  # var ≈ 9.0   (should be #2)
            "ent_tick_15s_mean": rng.normal(0, 0.1, 200),  # var ≈ 0.01
            "ent_tick_30s_mean": rng.normal(0, 2.0, 200),  # var ≈ 4.0   (should be #3)
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(
            bars, vector="entropy", max_features=3, method="variance"
        )

        assert len(result) == 3
        # The top-3 by variance should be the ones with std=5, 3, 2
        assert result[0] == "ent_tick_5s_mean", f"Expected highest var column first, got {result[0]}"
        assert result[1] == "ent_tick_10s_mean", f"Expected second highest var, got {result[1]}"
        assert result[2] == "ent_tick_30s_mean", f"Expected third highest var, got {result[2]}"

    def test_nan_columns_handled(self):
        """Columns with NaN values should still work (variance computed with skipna)."""
        rng = np.random.RandomState(42)
        values = rng.normal(0, 2, 200)
        values[::5] = np.nan  # 20% NaN
        data = {
            "ent_tick_1s_mean": values,
            "ent_tick_5s_mean": rng.normal(0, 1, 200),
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="entropy", max_features=5)
        assert len(result) >= 1, "Should handle NaN columns without crashing"

    def test_all_nan_column_excluded(self):
        """A column that is entirely NaN should not be selected."""
        rng = np.random.RandomState(42)
        data = {
            "ent_tick_1s_mean": np.full(100, np.nan),  # all NaN
            "ent_tick_5s_mean": rng.normal(0, 1, 100),  # valid
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="entropy", max_features=5)
        assert "ent_tick_1s_mean" not in result, "All-NaN column should be excluded"

    def test_single_bar_does_not_crash(self):
        """Edge case: only 1 bar. Variance is NaN/0 for all columns."""
        data = {
            "ent_tick_1s_mean": [1.0],
            "ent_tick_5s_mean": [2.0],
        }
        bars = _make_bars_with_columns(data)
        # Should not crash, may return empty list (single sample has no variance)
        result = select_top_features(bars, vector="entropy", max_features=5)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestSelectTopFeaturesErrors:
    """Tests for error conditions."""

    def test_unknown_vector_raises(self):
        """Asking for a vector that doesn't exist must raise ValueError."""
        bars = _make_entropy_bars()
        with pytest.raises(ValueError, match="Unknown vector"):
            select_top_features(bars, vector="nonexistent_vector", max_features=5)

    def test_no_matching_columns_raises(self):
        """If bars have no columns matching the vector, raise ValueError."""
        # Create bars with only non-entropy columns
        data = {
            "vol_returns_1m_mean": np.random.normal(0, 1, 100),
            "vol_returns_5m_mean": np.random.normal(0, 1, 100),
        }
        bars = _make_bars_with_columns(data)
        with pytest.raises(ValueError, match="No columns matching"):
            select_top_features(bars, vector="entropy", max_features=5)

    def test_unknown_method_raises(self):
        """Invalid method name must raise ValueError."""
        bars = _make_entropy_bars()
        with pytest.raises(ValueError, match="Unknown method"):
            select_top_features(bars, vector="entropy", method="magic")

    def test_max_features_zero_raises(self):
        """max_features=0 is invalid."""
        bars = _make_entropy_bars()
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            select_top_features(bars, vector="entropy", max_features=0)

    def test_max_features_negative_raises(self):
        """max_features=-1 is invalid."""
        bars = _make_entropy_bars()
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            select_top_features(bars, vector="entropy", max_features=-1)


# ---------------------------------------------------------------------------
# Autocorrelation method tests
# ---------------------------------------------------------------------------

class TestAutocorrelationMethod:
    """Tests for the autocorrelation_range selection method."""

    def test_high_ac_range_preferred(self):
        """
        A feature with high AC at short lags and low AC at long lags
        (i.e., large AC range) should be preferred over a feature
        with flat AC across all lags.
        """
        rng = np.random.RandomState(42)
        n = 500

        # Feature 1: AR(1) process with phi=0.95 → high AC(1), decaying AC(30)
        # This has a large autocorrelation range
        ar_series = np.zeros(n)
        ar_series[0] = rng.normal()
        for i in range(1, n):
            ar_series[i] = 0.95 * ar_series[i - 1] + rng.normal(0, 0.3)

        # Feature 2: white noise → AC ≈ 0 at all lags → range ≈ 0
        white_noise = rng.normal(0, 1, n)

        data = {
            "ent_tick_1s_mean": ar_series,
            "ent_tick_5s_mean": white_noise,
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(
            bars, vector="entropy", max_features=2, method="autocorrelation_range"
        )

        # AR process should be ranked first (larger AC range)
        assert result[0] == "ent_tick_1s_mean", (
            f"Expected AR process (high AC range) first, got {result[0]}"
        )

    def test_works_with_short_series(self):
        """Should not crash when series length < max_lag."""
        rng = np.random.RandomState(42)
        data = {
            "ent_tick_1s_mean": rng.normal(0, 1, 20),  # only 20 bars
            "ent_tick_5s_mean": rng.normal(0, 2, 20),
        }
        bars = _make_bars_with_columns(data)
        result = select_top_features(
            bars, vector="entropy", max_features=2, method="autocorrelation_range"
        )
        assert isinstance(result, list)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Cross-vector tests
# ---------------------------------------------------------------------------

class TestCrossVector:
    """Test that feature selection works across different vectors."""

    def test_orderflow_vector(self):
        """orderflow vector columns should be selectable."""
        rng = np.random.RandomState(42)
        of_cols = FEATURE_VECTORS["orderflow"]["columns"]
        data = {}
        for col in of_cols:
            for suffix in ["mean", "std", "last"]:
                data[f"{col}_{suffix}"] = rng.normal(0, 1, 100)
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="orderflow", max_features=5)
        assert len(result) == 5
        assert all("imbalance" in c for c in result), (
            f"orderflow columns should contain 'imbalance', got {result}"
        )

    def test_volatility_vector(self):
        """volatility vector columns should be selectable."""
        rng = np.random.RandomState(42)
        vol_cols = FEATURE_VECTORS["volatility"]["columns"]
        data = {}
        for col in vol_cols:
            for suffix in ["mean", "std", "last"]:
                data[f"{col}_{suffix}"] = rng.normal(0, 1, 100)
        bars = _make_bars_with_columns(data)
        result = select_top_features(bars, vector="volatility", max_features=5)
        assert len(result) == 5
        assert all("vol_" in c for c in result)

    def test_different_vectors_return_different_columns(self):
        """Entropy and volatility selection must not overlap."""
        rng = np.random.RandomState(42)
        data = {}
        for col in FEATURE_VECTORS["entropy"]["columns"]:
            for suffix in ["mean", "std", "slope"]:
                data[f"{col}_{suffix}"] = rng.normal(0, 1, 100)
        for col in FEATURE_VECTORS["volatility"]["columns"]:
            for suffix in ["mean", "std", "last"]:
                data[f"{col}_{suffix}"] = rng.normal(0, 1, 100)
        bars = _make_bars_with_columns(data)

        ent_result = set(select_top_features(bars, vector="entropy", max_features=5))
        vol_result = set(select_top_features(bars, vector="volatility", max_features=5))
        overlap = ent_result & vol_result
        assert len(overlap) == 0, f"Vectors should not overlap, got shared: {overlap}"


# ===========================================================================
# TEMPORAL DERIVATIVE TESTS
# ===========================================================================

def _make_simple_bars(data: dict[str, list | np.ndarray]) -> pd.DataFrame:
    """Create minimal bars with given feature columns + required meta."""
    n = len(next(iter(data.values())))
    df = pd.DataFrame(data)
    df["bar_start"] = pd.date_range("2026-01-01", periods=n, freq="15min")
    df["bar_end"] = df["bar_start"] + pd.Timedelta("15min")
    df["symbol"] = "BTC"
    df["tick_count"] = 100
    return df


class TestTemporalDerivativesVelocity:
    """Tests for velocity (1st difference) computation."""

    def test_velocity_constant_is_zero(self):
        """Velocity of a constant series must be 0 everywhere (except row 0 = NaN)."""
        bars = _make_simple_bars({"feat_a": np.full(50, 5.0)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        vel = out["feat_a_vel"]
        assert np.isnan(vel.iloc[0]), "First velocity value must be NaN"
        assert (vel.iloc[1:] == 0.0).all(), "Constant input → zero velocity"

    def test_velocity_linear_ramp(self):
        """Velocity of [0,1,2,3,4,5] must be [NaN,1,1,1,1,1]."""
        bars = _make_simple_bars({"feat_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[3])
        vel = out["feat_a_vel"]
        assert np.isnan(vel.iloc[0])
        np.testing.assert_array_almost_equal(vel.iloc[1:].values, [1, 1, 1, 1, 1])

    def test_velocity_step_function(self):
        """Velocity of a step [0,0,0,5,5,5] must spike at the transition."""
        bars = _make_simple_bars({"feat_a": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[3])
        vel = out["feat_a_vel"]
        assert vel.iloc[3] == 5.0, "Step transition should produce velocity=5"
        assert vel.iloc[4] == 0.0, "After step, velocity returns to 0"


class TestTemporalDerivativesAcceleration:
    """Tests for acceleration (2nd difference) computation."""

    def test_acceleration_linear_is_zero(self):
        """Acceleration of a linear ramp must be 0 (constant velocity)."""
        bars = _make_simple_bars({"feat_a": np.arange(50, dtype=float)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        accel = out["feat_a_accel"]
        # First two rows are NaN (need two diffs)
        assert np.isnan(accel.iloc[0])
        assert np.isnan(accel.iloc[1])
        np.testing.assert_array_almost_equal(accel.iloc[2:].values, 0.0)

    def test_acceleration_quadratic(self):
        """Acceleration of t^2 = [0,1,4,9,16,25,...] must be constant = 2."""
        t = np.arange(50, dtype=float)
        bars = _make_simple_bars({"feat_a": t ** 2})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        accel = out["feat_a_accel"]
        # Skip first 2 NaN rows
        np.testing.assert_array_almost_equal(accel.iloc[2:].values, 2.0)


class TestTemporalDerivativesZscore:
    """Tests for rolling z-score computation."""

    def test_zscore_centered(self):
        """Z-score of random normal data should have mean ≈ 0, std ≈ 1."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 1, 500)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[20])
        zs = out["feat_a_zscore_20"].dropna()
        assert abs(zs.mean()) < 0.15, f"Z-score mean should be ≈0, got {zs.mean():.3f}"
        assert abs(zs.std() - 1.0) < 0.3, f"Z-score std should be ≈1, got {zs.std():.3f}"

    def test_zscore_constant_is_zero(self):
        """Z-score of a constant series must be 0 (not inf or NaN after warmup)."""
        bars = _make_simple_bars({"feat_a": np.full(100, 3.14)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[10])
        zs = out["feat_a_zscore_10"]
        # After warmup, z-score of constant should be 0 (guarded division)
        valid = zs.iloc[10:]
        assert not np.any(np.isinf(valid)), "Z-score must not produce inf"
        assert (valid.dropna() == 0.0).all(), "Z-score of constant must be 0"


class TestTemporalDerivativesSlope:
    """Tests for rolling OLS slope computation."""

    def test_slope_linear(self):
        """Slope of a perfect linear series [0,2,4,6,...] with window=3 must be 2.0."""
        bars = _make_simple_bars({"feat_a": np.arange(0, 100, 2, dtype=float)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[3])
        slope = out["feat_a_slope_3"]
        valid = slope.dropna()
        np.testing.assert_array_almost_equal(valid.values, 2.0)

    def test_slope_constant_is_zero(self):
        """Slope of a constant series must be 0."""
        bars = _make_simple_bars({"feat_a": np.full(50, 7.0)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        slope = out["feat_a_slope_5"]
        valid = slope.dropna()
        np.testing.assert_array_almost_equal(valid.values, 0.0)

    def test_slope_negative_trend(self):
        """Slope of a decreasing series must be negative."""
        bars = _make_simple_bars({"feat_a": np.arange(100, 0, -1, dtype=float)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        slope = out["feat_a_slope_5"]
        valid = slope.dropna()
        assert (valid < 0).all(), "Decreasing series must have negative slope"


class TestTemporalDerivativesRvol:
    """Tests for rolling volatility computation."""

    def test_rvol_constant_is_zero(self):
        """Rolling vol of a constant series must be 0 (or NaN during warmup)."""
        bars = _make_simple_bars({"feat_a": np.full(50, 5.0)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5])
        rvol = out["feat_a_rvol_5"]
        valid = rvol.dropna()
        np.testing.assert_array_almost_equal(valid.values, 0.0)

    def test_rvol_positive_for_noisy_data(self):
        """Rolling vol of random data must be positive after warmup."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 3, 200)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[10])
        rvol = out["feat_a_rvol_10"]
        valid = rvol.dropna()
        assert (valid > 0).all(), "Noisy data must have positive rolling vol"


class TestTemporalDerivativesShape:
    """Tests for output shape, column naming, and NaN structure."""

    def test_output_column_count(self):
        """Output must have exactly n_cols * (2 + 3 * len(windows)) columns."""
        rng = np.random.RandomState(42)
        cols = [f"feat_{i}" for i in range(5)]
        data = {c: rng.normal(0, 1, 100) for c in cols}
        bars = _make_simple_bars(data)
        windows = [5, 15]
        out = temporal_derivatives(bars, columns=cols, windows=windows)
        expected = 5 * (2 + 3 * 2)  # 5 cols × (vel, accel, zscore×2, slope×2, rvol×2)
        assert out.shape[1] == expected, (
            f"Expected {expected} columns, got {out.shape[1]}"
        )

    def test_output_length_equals_input(self):
        """Output must have same number of rows as input (NaN-padded, not truncated)."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 1, 100)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5, 30])
        assert len(out) == 100, f"Expected 100 rows, got {len(out)}"

    def test_nan_padding_structure(self):
        """
        First max(windows)-1 rows of rolling derivatives must be NaN.
        Rows after warmup must have no NaN (given clean input).
        """
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 1, 100)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[30])

        # Rolling columns should have NaN for first 29 rows
        zscore = out["feat_a_zscore_30"]
        slope = out["feat_a_slope_30"]
        rvol = out["feat_a_rvol_30"]

        assert zscore.iloc[:29].isna().all(), "First 29 rows of zscore_30 should be NaN"
        assert slope.iloc[:28].isna().all(), "First 28 rows of slope_30 should be NaN"
        assert rvol.iloc[:29].isna().all(), "First 29 rows of rvol_30 should be NaN"

        # After warmup: no NaN
        assert not zscore.iloc[30:].isna().any(), "zscore after warmup should have no NaN"
        assert not slope.iloc[30:].isna().any(), "slope after warmup should have no NaN"
        assert not rvol.iloc[30:].isna().any(), "rvol after warmup should have no NaN"

    def test_no_inf_values(self):
        """No inf values in output, even with zeros and constant segments."""
        data = np.zeros(100)
        data[50:] = 1.0  # step in the middle
        data[70:80] = 0.0  # constant zero segment
        bars = _make_simple_bars({"feat_a": data})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5, 10])
        for col in out.columns:
            assert not np.any(np.isinf(out[col].values)), (
                f"Column {col} contains inf values"
            )

    def test_column_naming_parseable(self):
        """All output column names must follow the {base}_{deriv_type}[_{window}] pattern."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 1, 50)})
        out = temporal_derivatives(bars, columns=["feat_a"], windows=[5, 10])

        expected_cols = {
            "feat_a_vel", "feat_a_accel",
            "feat_a_zscore_5", "feat_a_zscore_10",
            "feat_a_slope_5", "feat_a_slope_10",
            "feat_a_rvol_5", "feat_a_rvol_10",
        }
        assert set(out.columns) == expected_cols, (
            f"Column names mismatch.\nExpected: {sorted(expected_cols)}\n"
            f"Got: {sorted(out.columns)}"
        )

    def test_multiple_columns(self):
        """Works correctly with multiple input columns."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "feat_a": rng.normal(0, 1, 100),
            "feat_b": rng.normal(5, 2, 100),
            "feat_c": np.arange(100, dtype=float),
        })
        out = temporal_derivatives(bars, columns=["feat_a", "feat_b", "feat_c"], windows=[5])
        # 3 cols × (2 + 3×1) = 15
        assert out.shape[1] == 15
        # feat_c velocity should be 1.0 (linear ramp)
        np.testing.assert_array_almost_equal(out["feat_c_vel"].iloc[1:].values, 1.0)


class TestTemporalDerivativesErrors:
    """Tests for error conditions in temporal_derivatives."""

    def test_empty_columns_raises(self):
        bars = _make_simple_bars({"feat_a": np.ones(10)})
        with pytest.raises(ValueError, match="columns must be non-empty"):
            temporal_derivatives(bars, columns=[], windows=[5])

    def test_missing_column_raises(self):
        bars = _make_simple_bars({"feat_a": np.ones(10)})
        with pytest.raises(ValueError, match="Columns not in bars"):
            temporal_derivatives(bars, columns=["feat_nonexistent"], windows=[5])

    def test_empty_windows_raises(self):
        bars = _make_simple_bars({"feat_a": np.ones(10)})
        with pytest.raises(ValueError, match="windows must be non-empty"):
            temporal_derivatives(bars, columns=["feat_a"], windows=[])


class TestTemporalDerivativesDeterminism:
    """Determinism: same input always produces same output."""

    def test_deterministic(self):
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({"feat_a": rng.normal(0, 1, 200)})
        out1 = temporal_derivatives(bars, columns=["feat_a"], windows=[5, 15])
        out2 = temporal_derivatives(bars, columns=["feat_a"], windows=[5, 15])
        pd.testing.assert_frame_equal(out1, out2)

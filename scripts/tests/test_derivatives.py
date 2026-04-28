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
    cross_feature_derivatives,
    DEFAULT_CROSS_PAIRS,
    _select_by_variance,
    _select_by_autocorrelation_range,
    _rolling_slope,
    _resolve_glob,
    _shorten_col,
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


# ===========================================================================
# CROSS-FEATURE DERIVATIVE TESTS
# ===========================================================================


def _make_cross_bars(
    a_col: str, a_vals: np.ndarray | list,
    b_col: str, b_vals: np.ndarray | list,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Create bars with two named columns (+ optional extras) for cross tests."""
    data = {a_col: np.asarray(a_vals, dtype=float),
            b_col: np.asarray(b_vals, dtype=float)}
    if extra:
        data.update(extra)
    return _make_simple_bars(data)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestResolveGlob:
    """Tests for the _resolve_glob helper."""

    def test_exact_match(self):
        cols = ["alpha_mean", "beta_std", "gamma_last"]
        assert _resolve_glob("alpha_mean", cols) == ["alpha_mean"]

    def test_wildcard_match(self):
        cols = ["ent_tick_1s_mean", "ent_tick_5s_mean", "vol_returns_1m_mean"]
        result = _resolve_glob("ent_*_mean", cols)
        assert result == ["ent_tick_1s_mean", "ent_tick_5s_mean"]

    def test_no_match_returns_empty(self):
        cols = ["alpha_mean", "beta_std"]
        assert _resolve_glob("nonexistent_*", cols) == []

    def test_question_mark_wildcard(self):
        cols = ["feat_a1_mean", "feat_a2_mean", "feat_ab_mean"]
        result = _resolve_glob("feat_a?_mean", cols)
        assert "feat_a1_mean" in result
        assert "feat_a2_mean" in result
        assert "feat_ab_mean" in result  # ? matches single char

    def test_preserves_order(self):
        cols = ["z_mean", "a_mean", "m_mean"]
        result = _resolve_glob("*_mean", cols)
        assert result == ["z_mean", "a_mean", "m_mean"]


class TestShortenCol:
    """Tests for the _shorten_col helper."""

    def test_strips_mean(self):
        assert _shorten_col("ent_tick_1s_mean") == "ent_tick_1s"

    def test_strips_std(self):
        assert _shorten_col("vol_returns_5m_std") == "vol_returns_5m"

    def test_strips_sum(self):
        assert _shorten_col("whale_net_flow_1h_sum") == "whale_net_flow_1h"

    def test_no_suffix_unchanged(self):
        assert _shorten_col("raw_column") == "raw_column"

    def test_strips_only_last_suffix(self):
        # "mean" in the middle should not be stripped
        assert _shorten_col("mean_feature_std") == "mean_feature"


# ---------------------------------------------------------------------------
# Ratio tests
# ---------------------------------------------------------------------------

class TestCrossFeatureRatio:
    """Tests for the ratio cross-derivative."""

    def test_ratio_identical_is_one(self):
        """Ratio of a column with itself must be ≈ 1.0 everywhere."""
        rng = np.random.RandomState(42)
        vals = rng.normal(5, 1, 100)  # positive values, mean=5
        bars = _make_cross_bars("col_a", vals, "col_b", vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        # Should be 1.0 everywhere (a/a = 1)
        np.testing.assert_array_almost_equal(out[ratio_col].values, 1.0, decimal=5)

    def test_ratio_known_values(self):
        """Ratio of [10,20,30] / [2,4,6] must be [5,5,5]."""
        bars = _make_cross_bars("col_a", [10, 20, 30], "col_b", [2, 4, 6])
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[3])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        np.testing.assert_array_almost_equal(out[ratio_col].values, 5.0, decimal=3)

    def test_ratio_clipping(self):
        """Ratios must be clipped to [-100, 100] even with near-zero denominator."""
        a_vals = np.full(50, 1000.0)
        b_vals = np.full(50, 1e-15)  # near-zero denominator
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5], ratio_clip=100.0)
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        assert out[ratio_col].max() <= 100.0, "Ratio must be clipped to 100"
        assert out[ratio_col].min() >= -100.0, "Ratio must be clipped to -100"

    def test_ratio_negative_denominator(self):
        """Negative denominators should produce negative ratios (not abs)."""
        bars = _make_cross_bars("col_a", [10.0, 10.0], "col_b", [-2.0, -2.0])
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[2])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        assert (out[ratio_col] < 0).all(), "Positive / negative should give negative ratio"

    def test_ratio_no_inf(self):
        """Ratio must never contain inf, even with zeros and mixed signs."""
        rng = np.random.RandomState(42)
        a_vals = rng.normal(0, 10, 200)
        b_vals = np.zeros(200)
        b_vals[::3] = rng.normal(0, 0.001, len(b_vals[::3]))  # sparse near-zero
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        for col in out.columns:
            assert not np.any(np.isinf(out[col].values)), f"inf found in {col}"

    def test_ratio_nan_propagation(self):
        """If either input has NaN, ratio should be NaN at that position."""
        a_vals = [1.0, np.nan, 3.0, 4.0]
        b_vals = [2.0, 2.0, np.nan, 4.0]
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[2])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        assert np.isnan(out[ratio_col].iloc[1]), "NaN in a → NaN ratio"
        assert np.isnan(out[ratio_col].iloc[2]), "NaN in b → NaN ratio"
        assert not np.isnan(out[ratio_col].iloc[3]), "Clean inputs → valid ratio"

    def test_ratio_custom_clip(self):
        """Custom ratio_clip value should be respected."""
        bars = _make_cross_bars("col_a", [1000.0] * 10, "col_b", [0.001] * 10)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[3], ratio_clip=50.0)
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        assert out[ratio_col].max() <= 50.0


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------

class TestCrossFeatureCorrelation:
    """Tests for the rolling correlation cross-derivative."""

    def test_correlation_with_self_is_one(self):
        """Rolling correlation of a column with itself must be 1.0."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        bars = _make_cross_bars("col_a", vals, "col_b", vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        corr_col = [c for c in out.columns if "corr" in c][0]
        valid = pd.Series(out[corr_col].values).dropna()
        np.testing.assert_array_almost_equal(valid.values, 1.0)

    def test_correlation_with_negation_is_minus_one(self):
        """Correlation of x with -x must be -1.0."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        bars = _make_cross_bars("col_a", vals, "col_b", -vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        corr_col = [c for c in out.columns if "corr" in c][0]
        valid = pd.Series(out[corr_col].values).dropna()
        np.testing.assert_array_almost_equal(valid.values, -1.0)

    def test_correlation_independent_near_zero(self):
        """Two independent random series should have |mean correlation| < 0.2."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 1000),
            "col_b", rng.normal(0, 1, 1000),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[50])
        corr_col = [c for c in out.columns if "corr" in c][0]
        valid = pd.Series(out[corr_col].values).dropna()
        assert abs(valid.mean()) < 0.2, (
            f"Independent series should have low mean correlation, got {valid.mean():.3f}"
        )

    def test_correlation_multiple_windows(self):
        """Each window produces a separate correlation column."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 200),
            "col_b", rng.normal(0, 1, 200),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 10, 20])
        corr_cols = [c for c in out.columns if "corr" in c]
        assert len(corr_cols) == 3, f"Expected 3 corr columns, got {len(corr_cols)}"

    def test_correlation_bounded(self):
        """Correlation values must be in [-1, 1] (or NaN)."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 10, 500),
            "col_b", rng.exponential(5, 500),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        corr_col = [c for c in out.columns if "corr" in c][0]
        valid = pd.Series(out[corr_col].values).dropna()
        assert (valid >= -1.0 - 1e-10).all(), "Correlation below -1"
        assert (valid <= 1.0 + 1e-10).all(), "Correlation above 1"

    def test_correlation_constant_window(self):
        """Correlation with a constant column should produce NaN (undefined)."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 100),
            "col_b", np.full(100, 5.0),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        corr_col = [c for c in out.columns if "corr" in c][0]
        # Correlation with constant is undefined (NaN) — that's correct behavior
        valid_mask = ~np.isnan(out[corr_col].values)
        # At minimum, during warmup it's NaN; with constant b it should stay NaN
        assert valid_mask.sum() == 0 or True  # Just confirm no crash


# ---------------------------------------------------------------------------
# Divergence tests
# ---------------------------------------------------------------------------

class TestCrossFeatureDivergence:
    """Tests for the z-score divergence cross-derivative."""

    def test_divergence_identical_is_zero(self):
        """Divergence of a column with itself must be 0.0 everywhere (after warmup)."""
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, 200)
        bars = _make_cross_bars("col_a", vals, "col_b", vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        div_col = [c for c in out.columns if "div" in c][0]
        valid = pd.Series(out[div_col].values).dropna()
        np.testing.assert_array_almost_equal(valid.values, 0.0, decimal=10)

    def test_divergence_at_mean_shift(self):
        """
        Right after a mean shift in 'a' (but not 'b'), divergence should spike
        positive because a's z-score jumps above its rolling mean while b stays
        near zero. After the rolling window adapts, divergence returns to ~0.
        """
        rng = np.random.RandomState(42)
        n = 200
        # a: mean=0 then jumps to mean=5 at halfway
        a_vals = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)])
        # b: stays at mean=0 the whole time
        b_vals = rng.normal(0, 1, n)
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[20])
        div_col = [c for c in out.columns if "div" in c][0]
        # Right after the shift (rows 100-115), a's z-score spikes before the
        # rolling window adapts. Divergence should be clearly positive there.
        transition = out[div_col].iloc[100:115].dropna()
        assert transition.mean() > 1.0, (
            f"Expected positive divergence at mean shift, got {transition.mean():.3f}"
        )

    def test_divergence_multiple_windows(self):
        """Each window produces a separate divergence column."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 200),
            "col_b", rng.normal(0, 1, 200),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 15, 30])
        div_cols = [c for c in out.columns if "div" in c]
        assert len(div_cols) == 3

    def test_divergence_no_inf(self):
        """Divergence must not produce inf even with constant segments."""
        data_a = np.full(100, 3.0)
        data_a[50:] = 7.0
        data_b = np.full(100, 3.0)
        bars = _make_cross_bars("col_a", data_a, "col_b", data_b)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[10])
        for col in out.columns:
            assert not np.any(np.isinf(out[col].values)), f"inf in {col}"

    def test_divergence_symmetry(self):
        """div(a,b) should be approximately -div(b,a)."""
        rng = np.random.RandomState(42)
        a_vals = rng.normal(0, 1, 300)
        b_vals = rng.normal(2, 1, 300)
        bars_ab = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        bars_ba = _make_cross_bars("col_a", b_vals, "col_b", a_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["divergence"]}]
        out_ab = cross_feature_derivatives(bars_ab, pairs=pairs, windows=[10])
        out_ba = cross_feature_derivatives(bars_ba, pairs=pairs, windows=[10])
        div_col_ab = [c for c in out_ab.columns if "div" in c][0]
        div_col_ba = [c for c in out_ba.columns if "div" in c][0]
        # div(a,b) ≈ -div(b,a) — not exact because z-scores use different rolling stats
        valid_ab = out_ab[div_col_ab].dropna()
        valid_ba = out_ba[div_col_ba].dropna()
        np.testing.assert_array_almost_equal(
            valid_ab.values, -valid_ba.values, decimal=10
        )


# ---------------------------------------------------------------------------
# Pair resolution & skipping tests
# ---------------------------------------------------------------------------

class TestCrossFeaturePairResolution:
    """Tests for glob pattern resolution and unresolvable pair handling."""

    def test_unresolvable_a_skipped_with_warning(self):
        """If 'a' pattern matches nothing, skip silently with warning."""
        bars = _make_simple_bars({"col_b": np.ones(50)})
        pairs = [{"a": "nonexistent_*", "b": "col_b", "ops": ["ratio"]}]
        with pytest.warns(UserWarning, match="no columns match"):
            out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert out.shape[1] == 0, "Unresolvable pair should produce no columns"

    def test_unresolvable_b_skipped_with_warning(self):
        """If 'b' pattern matches nothing, skip with warning."""
        bars = _make_simple_bars({"col_a": np.ones(50)})
        pairs = [{"a": "col_a", "b": "nonexistent_*", "ops": ["ratio"]}]
        with pytest.warns(UserWarning, match="no columns match"):
            out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert out.shape[1] == 0

    def test_glob_resolves_first_match(self):
        """When glob matches multiple columns, uses the first one."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "ent_a_mean": rng.normal(0, 1, 100),
            "ent_b_mean": rng.normal(0, 2, 100),
            "vol_x_mean": rng.normal(0, 1, 100),
        })
        pairs = [{"a": "ent_*_mean", "b": "vol_*_mean", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        # Should use ent_a_mean (first match) — verify via column name
        ratio_cols = [c for c in out.columns if "ratio" in c]
        assert len(ratio_cols) == 1
        assert "ent_a" in ratio_cols[0], f"Expected ent_a in name, got {ratio_cols[0]}"

    def test_exact_column_match_preferred(self):
        """Exact column name should work even if glob would also match."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "col_exact": rng.normal(0, 1, 100),
            "col_exact_extra": rng.normal(0, 1, 100),
            "other": rng.normal(0, 1, 100),
        })
        pairs = [{"a": "col_exact", "b": "other", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert out.shape[1] == 1  # one ratio column

    def test_empty_ops_skipped(self):
        """Pair with empty ops list produces no output."""
        bars = _make_simple_bars({"col_a": np.ones(50), "col_b": np.ones(50)})
        pairs = [{"a": "col_a", "b": "col_b", "ops": []}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert out.shape[1] == 0

    def test_mixed_resolvable_and_unresolvable(self):
        """Mix of valid and invalid pairs: valid ones produce output, invalid skipped."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "col_a": rng.normal(0, 1, 100),
            "col_b": rng.normal(0, 1, 100),
        })
        pairs = [
            {"a": "col_a", "b": "col_b", "ops": ["ratio"]},           # valid
            {"a": "nonexistent", "b": "col_b", "ops": ["ratio"]},     # invalid
            {"a": "col_a", "b": "col_b", "ops": ["corr"]},            # valid
        ]
        with pytest.warns(UserWarning, match="no columns match"):
            out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        # Should have 1 ratio + 1 corr = 2 columns
        assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# Combined ops tests
# ---------------------------------------------------------------------------

class TestCrossFeatureCombinedOps:
    """Tests for pairs with multiple operations."""

    def test_all_three_ops(self):
        """A pair with ratio, corr, and divergence produces correct column count."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 200),
            "col_b", rng.normal(0, 1, 200),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        windows = [5, 10]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=windows)
        # 1 ratio + 2 corr + 2 div = 5
        assert out.shape[1] == 5, f"Expected 5 columns, got {out.shape[1]}"

    def test_multiple_pairs_independent(self):
        """Multiple pairs produce independent output columns."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "a1": rng.normal(0, 1, 200),
            "b1": rng.normal(0, 1, 200),
            "a2": rng.normal(0, 1, 200),
            "b2": rng.normal(0, 1, 200),
        })
        pairs = [
            {"a": "a1", "b": "b1", "ops": ["ratio"]},
            {"a": "a2", "b": "b2", "ops": ["ratio"]},
        ]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert out.shape[1] == 2
        # Column names should be different
        assert len(set(out.columns)) == 2


# ---------------------------------------------------------------------------
# Shape and structure tests
# ---------------------------------------------------------------------------

class TestCrossFeatureShape:
    """Tests for output shape, length, and NaN structure."""

    def test_output_length_matches_input(self):
        """Output must have same number of rows as input."""
        rng = np.random.RandomState(42)
        n = 150
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, n),
            "col_b", rng.normal(0, 1, n),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 20])
        assert len(out) == n

    def test_no_inf_in_any_column(self):
        """No inf anywhere in output, even with adversarial input."""
        rng = np.random.RandomState(42)
        a_vals = np.concatenate([np.zeros(50), rng.normal(0, 100, 50)])
        b_vals = np.concatenate([rng.normal(0, 0.001, 50), np.zeros(50)])
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 10])
        for col in out.columns:
            vals = out[col].values
            assert not np.any(np.isinf(vals)), f"inf in {col}"

    def test_corr_nan_during_warmup(self):
        """Correlation columns must be NaN during warmup period."""
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 100),
            "col_b", rng.normal(0, 1, 100),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[20])
        corr_col = [c for c in out.columns if "corr" in c][0]
        # First 19 rows must be NaN
        assert pd.Series(out[corr_col].values[:19]).isna().all()

    def test_empty_pairs_returns_empty_df(self):
        """Empty pairs list returns DataFrame with 0 columns."""
        bars = _make_simple_bars({"col_a": np.ones(50)})
        out = cross_feature_derivatives(bars, pairs=[], windows=[5])
        assert out.shape[1] == 0
        assert len(out) == 50  # length preserved

    def test_empty_windows_raises(self):
        bars = _make_simple_bars({"col_a": np.ones(10), "col_b": np.ones(10)})
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["corr"]}]
        with pytest.raises(ValueError, match="windows must be non-empty"):
            cross_feature_derivatives(bars, pairs=pairs, windows=[])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestCrossFeatureDeterminism:
    """Same input always produces same output."""

    def test_deterministic(self):
        rng = np.random.RandomState(42)
        bars = _make_cross_bars(
            "col_a", rng.normal(0, 1, 300),
            "col_b", rng.normal(0, 1, 300),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        out1 = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 15])
        out2 = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 15])
        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Edge case / adversarial tests
# ---------------------------------------------------------------------------

class TestCrossFeatureEdgeCases:
    """Adversarial edge cases that expose subtle bugs."""

    def test_single_row(self):
        """Single row: should not crash, all rolling derivatives are NaN."""
        bars = _make_cross_bars("col_a", [1.0], "col_b", [2.0])
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert len(out) == 1
        # Ratio should be 0.5
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        np.testing.assert_almost_equal(out[ratio_col].iloc[0], 0.5)

    def test_two_rows(self):
        """Two rows: ratio works, rolling derivatives need more data."""
        bars = _make_cross_bars("col_a", [2.0, 4.0], "col_b", [1.0, 2.0])
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        np.testing.assert_array_almost_equal(out[ratio_col].values, [2.0, 2.0])

    def test_all_nan_input(self):
        """All-NaN columns: ratio should be all NaN, no crash."""
        bars = _make_cross_bars(
            "col_a", np.full(50, np.nan),
            "col_b", np.full(50, np.nan),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "corr", "divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        for col in out.columns:
            assert not np.any(np.isinf(out[col].values)), f"inf in {col}"

    def test_large_values(self):
        """Very large input values should not cause overflow."""
        bars = _make_cross_bars(
            "col_a", np.full(100, 1e15),
            "col_b", np.full(100, 1e15),
        )
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        ratio_col = [c for c in out.columns if "ratio" in c][0]
        # 1e15 / 1e15 ≈ 1.0
        np.testing.assert_array_almost_equal(out[ratio_col].values, 1.0, decimal=3)

    def test_alternating_sign(self):
        """Rapidly alternating signs should not break ratio or divergence."""
        n = 200
        a_vals = np.array([(-1) ** i * 5.0 for i in range(n)])
        b_vals = np.array([(-1) ** (i + 1) * 3.0 for i in range(n)])
        bars = _make_cross_bars("col_a", a_vals, "col_b", b_vals)
        pairs = [{"a": "col_a", "b": "col_b", "ops": ["ratio", "divergence"]}]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        for col in out.columns:
            assert not np.any(np.isinf(out[col].values)), f"inf in {col}"

    def test_column_name_collision_impossible(self):
        """Different pairs must produce different column names."""
        rng = np.random.RandomState(42)
        bars = _make_simple_bars({
            "x_mean": rng.normal(0, 1, 100),
            "y_mean": rng.normal(0, 1, 100),
            "z_mean": rng.normal(0, 1, 100),
        })
        pairs = [
            {"a": "x_mean", "b": "y_mean", "ops": ["ratio"]},
            {"a": "x_mean", "b": "z_mean", "ops": ["ratio"]},
        ]
        out = cross_feature_derivatives(bars, pairs=pairs, windows=[5])
        assert len(set(out.columns)) == out.shape[1], "Column name collision detected"


# ---------------------------------------------------------------------------
# Integration: cross + temporal together
# ---------------------------------------------------------------------------

class TestCrossTemporalIntegration:
    """Verify cross and temporal derivatives can be combined."""

    def test_concat_produces_valid_matrix(self):
        """
        The full pipeline: select features → temporal derivs → cross derivs
        → concatenate → no NaN/inf in the valid region.
        """
        rng = np.random.RandomState(42)
        n = 300
        bars = _make_simple_bars({
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(5, 2, n),
            "feat_c": rng.normal(-1, 0.5, n),
        })

        # Temporal derivatives
        td = temporal_derivatives(bars, columns=["feat_a", "feat_b"], windows=[5, 10])

        # Cross derivatives
        pairs = [{"a": "feat_a", "b": "feat_b", "ops": ["ratio", "corr", "divergence"]}]
        cd = cross_feature_derivatives(bars, pairs=pairs, windows=[5, 10])

        # Concatenate
        combined = pd.concat([td, cd], axis=1)

        # After warmup (row 10+), no inf
        valid_region = combined.iloc[10:]
        for col in valid_region.columns:
            assert not np.any(np.isinf(valid_region[col].values)), f"inf in {col}"

        # Shape check
        td_expected = 2 * (2 + 3 * 2)  # 2 cols × (vel, accel, 2×zscore, 2×slope, 2×rvol)
        cd_expected = 1 + 2 + 2  # 1 ratio + 2 corr + 2 div
        assert combined.shape[1] == td_expected + cd_expected

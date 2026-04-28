"""
Skeptical tests for cluster_pipeline.derivatives — feature selection.

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
    _select_by_variance,
    _select_by_autocorrelation_range,
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

"""
Tests for spectral derivatives (Task 3 from TASKS_5_5_2026.md).

Covers: output shape, frequency detection, NaN handling, edge cases,
and integration with generate_derivatives.
"""

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.derivatives import spectral_derivatives, generate_derivatives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_bars(n=100, period=10, cols=None, amplitude=1.0):
    """Generate bars with a known sinusoidal signal."""
    if cols is None:
        cols = ["feat_a"]
    t = np.arange(n, dtype=float)
    data = {col: amplitude * np.sin(2 * np.pi * t / period) for col in cols}
    return pd.DataFrame(data)


def _random_bars(n=100, cols=None, seed=42):
    """Random bars with no periodic structure."""
    if cols is None:
        cols = ["feat_a"]
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.standard_normal(n) for col in cols})


# ---------------------------------------------------------------------------
# TestOutputShape
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Tests for output DataFrame shape and columns."""

    def test_single_column_single_window(self):
        df = _random_bars(50, ["x"])
        result = spectral_derivatives(df, columns=["x"], window=20)
        # 4 outputs per column: spec_low, spec_high, spec_ratio, spec_period
        assert result.shape == (50, 4)

    def test_multiple_columns(self):
        df = _random_bars(50, ["a", "b", "c"])
        result = spectral_derivatives(df, columns=["a", "b", "c"], window=15)
        assert result.shape == (50, 12)  # 3 cols * 4 outputs

    def test_column_names(self):
        df = _random_bars(50, ["feat"])
        result = spectral_derivatives(df, columns=["feat"], window=20)
        expected = ["feat_spec_low_20", "feat_spec_high_20",
                    "feat_spec_ratio_20", "feat_spec_period_20"]
        assert list(result.columns) == expected

    def test_warmup_rows_are_nan(self):
        df = _random_bars(50, ["x"])
        result = spectral_derivatives(df, columns=["x"], window=20)
        # First 19 rows should be NaN
        assert result.iloc[:19].isna().all().all()
        # Row 19 should have values
        assert not result.iloc[19].isna().all()

    def test_index_preserved(self):
        df = _random_bars(50, ["x"])
        df.index = range(100, 150)
        result = spectral_derivatives(df, columns=["x"], window=15)
        assert list(result.index) == list(range(100, 150))


# ---------------------------------------------------------------------------
# TestFrequencyDetection
# ---------------------------------------------------------------------------


class TestFrequencyDetection:
    """Tests that spectral features correctly detect known frequencies."""

    def test_dominant_period_sine(self):
        """Pure sine wave should have dominant period ≈ actual period."""
        period = 10
        df = _sine_bars(n=100, period=period, cols=["x"])
        result = spectral_derivatives(df, columns=["x"], window=30)
        # After warmup, check dominant period
        valid = result["x_spec_period_30"].dropna()
        # Should be close to 10
        mean_period = valid.mean()
        assert 8 <= mean_period <= 12, f"Expected ~10, got {mean_period:.1f}"

    def test_low_freq_dominates_slow_signal(self):
        """Slowly varying signal has more low-freq power."""
        n = 100
        # Very slow signal (period=50 relative to window=30)
        t = np.arange(n, dtype=float)
        df = pd.DataFrame({"x": np.sin(2 * np.pi * t / 50)})
        result = spectral_derivatives(df, columns=["x"], window=30)
        valid_ratio = result["x_spec_ratio_30"].dropna()
        # Low/high ratio should be > 1 for slow signal
        assert valid_ratio.mean() > 1.0

    def test_high_freq_dominates_fast_signal(self):
        """Rapidly oscillating signal has more high-freq power."""
        n = 100
        t = np.arange(n, dtype=float)
        # Very fast signal (period=3 relative to window=30)
        df = pd.DataFrame({"x": np.sin(2 * np.pi * t / 3)})
        result = spectral_derivatives(df, columns=["x"], window=30)
        valid_ratio = result["x_spec_ratio_30"].dropna()
        # Low/high ratio should be < 1 for fast signal
        assert valid_ratio.mean() < 1.0

    def test_sine_has_dominant_period_random_less_consistent(self):
        """Sine has consistent dominant period; random periods are scattered."""
        df_sine = _sine_bars(n=100, period=10, cols=["x"])
        df_rand = _random_bars(100, ["x"], seed=77)
        r_sine = spectral_derivatives(df_sine, columns=["x"], window=30)
        r_rand = spectral_derivatives(df_rand, columns=["x"], window=30)
        # Sine should have very consistent dominant period
        sine_periods = r_sine["x_spec_period_30"].dropna()
        rand_periods = r_rand["x_spec_period_30"].dropna()
        # Sine periods are all ~10, so std should be very low
        assert sine_periods.std() < 1.0
        # Random periods (when detected) should be more scattered
        if len(rand_periods) > 5:
            assert rand_periods.std() > sine_periods.std()


# ---------------------------------------------------------------------------
# TestNaNHandling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """Tests for NaN input handling."""

    def test_nan_in_window_produces_nan(self):
        df = pd.DataFrame({"x": [1.0] * 10 + [np.nan] + [1.0] * 39})
        result = spectral_derivatives(df, columns=["x"], window=20)
        # Bar 10 is NaN, so windows containing it should be NaN
        # Window ending at t=10..29 contains the NaN
        assert np.isnan(result["x_spec_low_20"].iloc[10])

    def test_all_nan_column(self):
        df = pd.DataFrame({"x": [np.nan] * 50})
        result = spectral_derivatives(df, columns=["x"], window=20)
        assert result.isna().all().all()


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation tests."""

    def test_empty_columns_raises(self):
        df = _random_bars(50, ["x"])
        with pytest.raises(ValueError, match="columns must be non-empty"):
            spectral_derivatives(df, columns=[], window=20)

    def test_small_window_raises(self):
        df = _random_bars(50, ["x"])
        with pytest.raises(ValueError, match="window must be >= 10"):
            spectral_derivatives(df, columns=["x"], window=5)

    def test_missing_column_skipped(self):
        """Missing columns are silently skipped."""
        df = _random_bars(50, ["x"])
        result = spectral_derivatives(df, columns=["x", "nonexistent"], window=20)
        assert result.shape[1] == 4  # only x's 4 outputs


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Tests for spectral integration into generate_derivatives."""

    def test_spectral_included_by_default(self):
        """generate_derivatives includes spectral when include_spectral=True."""
        from cluster_pipeline.config import get_vector_columns
        ent_cols = get_vector_columns("entropy")
        # Make bars with entropy columns
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            f"{col}_mean": rng.standard_normal(100) for col in ent_cols[:6]
        })
        result = generate_derivatives(
            df, vector="entropy", max_base_features=4,
            temporal_windows=[5, 10], include_spectral=True, spectral_window=15,
        )
        # Should have spectral columns
        spec_cols = [c for c in result.derivatives.columns if "_spec_" in c]
        assert len(spec_cols) > 0
        assert result.metadata["include_spectral"] is True
        assert result.metadata["n_spectral"] > 0

    def test_spectral_excluded_when_disabled(self):
        """No spectral columns when include_spectral=False."""
        from cluster_pipeline.config import get_vector_columns
        ent_cols = get_vector_columns("entropy")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            f"{col}_mean": rng.standard_normal(100) for col in ent_cols[:6]
        })
        result = generate_derivatives(
            df, vector="entropy", max_base_features=4,
            temporal_windows=[5, 10], include_spectral=False,
        )
        spec_cols = [c for c in result.derivatives.columns if "_spec_" in c]
        assert len(spec_cols) == 0

    def test_warmup_accounts_for_spectral(self):
        """Warmup rows should be max(temporal_windows, spectral_window)."""
        from cluster_pipeline.config import get_vector_columns
        ent_cols = get_vector_columns("entropy")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            f"{col}_mean": rng.standard_normal(100) for col in ent_cols[:4]
        })
        result = generate_derivatives(
            df, vector="entropy", max_base_features=4,
            temporal_windows=[5, 10], include_spectral=True, spectral_window=25,
        )
        assert result.warmup_rows == 25  # spectral_window > max(temporal)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases."""

    def test_constant_signal(self):
        """Constant signal: all power is 0 (after detrending)."""
        df = pd.DataFrame({"x": [3.14] * 50})
        result = spectral_derivatives(df, columns=["x"], window=20)
        valid_low = result["x_spec_low_20"].dropna()
        # After detrending constant signal, all FFT bins are 0
        assert (valid_low == 0.0).all()

    def test_step_function(self):
        """Step function has broadband spectrum."""
        data = [0.0] * 50 + [1.0] * 50
        df = pd.DataFrame({"x": data})
        result = spectral_derivatives(df, columns=["x"], window=20)
        # Should compute without error
        assert result.shape == (100, 4)

    def test_minimum_window(self):
        """Window=10 works."""
        df = _random_bars(20, ["x"])
        result = spectral_derivatives(df, columns=["x"], window=10)
        assert result.shape == (20, 4)
        # First 9 NaN, from row 9 onward should have values
        assert not result.iloc[9].isna().all()

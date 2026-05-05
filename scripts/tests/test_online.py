"""
Tests for Task 7.1: Rolling Derivative Buffer (DerivativeBuffer).

Covers: warmup behavior, batch equivalence, memory bounds, reset,
validation errors, edge cases, and derivative name consistency.
"""

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.online import DerivativeBuffer
from cluster_pipeline.derivatives import temporal_derivatives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n, columns, seed=42):
    """Generate n random bars as a DataFrame."""
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n) for col in columns}
    return pd.DataFrame(data)


def _bar_series(df, i):
    """Extract row i from DataFrame as a Series."""
    return df.iloc[i]


COLUMNS = ["feat_a", "feat_b", "feat_c"]
WINDOWS = [3, 5]


# ---------------------------------------------------------------------------
# TestWarmup
# ---------------------------------------------------------------------------


class TestWarmup:
    """Tests for warmup behavior (returns None until buffer full)."""

    def test_warmup_returns_none(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window - 1, COLUMNS)
        for i in range(len(bars)):
            result = buf.update(_bar_series(bars, i))
            assert result is None

    def test_warmup_exact_boundary(self):
        """max_window-1 bars → None, max_window-th bar → vector."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window, COLUMNS)
        for i in range(buf.max_window - 1):
            assert buf.update(_bar_series(bars, i)) is None
        result = buf.update(_bar_series(bars, buf.max_window - 1))
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_warmup_count(self):
        """Exactly max_window bars needed for first output."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=[5, 10])
        assert buf.max_window == 11  # max(windows) + 1
        bars = _make_bars(11, COLUMNS)
        for i in range(10):
            assert buf.update(_bar_series(bars, i)) is None
        assert buf.update(_bar_series(bars, 10)) is not None

    def test_is_warm_property(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window, COLUMNS)
        assert not buf.is_warm
        for i in range(buf.max_window - 1):
            buf.update(_bar_series(bars, i))
            assert not buf.is_warm
        buf.update(_bar_series(bars, buf.max_window - 1))
        assert buf.is_warm


# ---------------------------------------------------------------------------
# TestBatchEquivalence
# ---------------------------------------------------------------------------


class TestBatchEquivalence:
    """Tests that online output matches batch temporal_derivatives."""

    def test_matches_batch_last_row(self):
        """Push max_window bars, compare to batch output's last row."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        n = buf.max_window
        bars = _make_bars(n, COLUMNS, seed=77)

        # Online: push all bars
        result = None
        for i in range(n):
            result = buf.update(_bar_series(bars, i))

        # Batch
        batch = temporal_derivatives(bars, columns=COLUMNS, windows=WINDOWS)
        expected = batch.iloc[-1].values

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_batch_after_100_bars(self):
        """Push 100 bars, compare last output to batch on last max_window bars."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        n_total = 100
        bars = _make_bars(n_total, COLUMNS, seed=99)

        result = None
        for i in range(n_total):
            out = buf.update(_bar_series(bars, i))
            if out is not None:
                result = out

        # Batch on last max_window bars
        last_chunk = bars.iloc[n_total - buf.max_window:].reset_index(drop=True)
        batch = temporal_derivatives(last_chunk, columns=COLUMNS, windows=WINDOWS)
        expected = batch.iloc[-1].values

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_multiple_outputs_match_batch(self):
        """Every output after warmup matches batch on the corresponding window."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        n_total = 30
        bars = _make_bars(n_total, COLUMNS, seed=55)

        outputs = []
        for i in range(n_total):
            out = buf.update(_bar_series(bars, i))
            if out is not None:
                outputs.append((i, out))

        # Verify each output
        for bar_idx, online_vec in outputs:
            start = bar_idx - buf.max_window + 1
            chunk = bars.iloc[start:bar_idx + 1].reset_index(drop=True)
            batch = temporal_derivatives(chunk, columns=COLUMNS, windows=WINDOWS)
            expected = batch.iloc[-1].values
            np.testing.assert_allclose(online_vec, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# TestMemory
# ---------------------------------------------------------------------------


class TestMemory:
    """Tests for constant memory usage."""

    def test_buffer_size_constant(self):
        """Internal buffer never exceeds max_window."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(200, COLUMNS)
        for i in range(200):
            buf.update(_bar_series(bars, i))
        assert len(buf._buffer) == buf.max_window

    def test_n_pushed_tracks_total(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(50, COLUMNS)
        for i in range(50):
            buf.update(_bar_series(bars, i))
        assert buf.n_pushed == 50

    def test_custom_max_window(self):
        """Can set max_window larger than minimum."""
        buf = DerivativeBuffer(
            columns=COLUMNS, temporal_windows=[5], max_window=50
        )
        assert buf.max_window == 50
        bars = _make_bars(50, COLUMNS)
        for i in range(49):
            assert buf.update(_bar_series(bars, i)) is None
        assert buf.update(_bar_series(bars, 49)) is not None


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for reset behavior."""

    def test_reset_clears_buffer(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window + 5, COLUMNS)
        for i in range(buf.max_window + 5):
            buf.update(_bar_series(bars, i))
        assert buf.is_warm
        buf.reset()
        assert not buf.is_warm
        assert buf.n_pushed == 0
        assert len(buf._buffer) == 0

    def test_reset_requires_rewarm(self):
        """After reset, must push max_window bars again before output."""
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window * 2, COLUMNS)

        # Warm up
        for i in range(buf.max_window):
            buf.update(_bar_series(bars, i))
        assert buf.is_warm

        # Reset
        buf.reset()

        # Must warm up again
        for i in range(buf.max_window, buf.max_window * 2 - 1):
            assert buf.update(_bar_series(bars, i)) is None
        result = buf.update(_bar_series(bars, buf.max_window * 2 - 1))
        assert result is not None


# ---------------------------------------------------------------------------
# TestValidationErrors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Tests for input validation."""

    def test_empty_columns_raises(self):
        with pytest.raises(ValueError, match="columns must be non-empty"):
            DerivativeBuffer(columns=[])

    def test_empty_windows_raises(self):
        with pytest.raises(ValueError, match="temporal_windows must be non-empty"):
            DerivativeBuffer(columns=["a"], temporal_windows=[])

    def test_max_window_too_small_raises(self):
        with pytest.raises(ValueError, match="max_window"):
            DerivativeBuffer(columns=["a"], temporal_windows=[5, 10], max_window=5)

    def test_bar_missing_columns_raises(self):
        buf = DerivativeBuffer(columns=["a", "b", "c"], temporal_windows=[3])
        bar = pd.Series({"a": 1.0, "b": 2.0})  # missing "c"
        with pytest.raises(ValueError, match="missing columns"):
            buf.update(bar)


# ---------------------------------------------------------------------------
# TestDerivativeNames
# ---------------------------------------------------------------------------


class TestDerivativeNames:
    """Tests for derivative_names() method."""

    def test_names_length_matches_output(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window, COLUMNS)
        for i in range(buf.max_window - 1):
            buf.update(_bar_series(bars, i))
        result = buf.update(_bar_series(bars, buf.max_window - 1))
        names = buf.derivative_names()
        assert len(names) == len(result)

    def test_names_contain_expected_patterns(self):
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[5])
        names = buf.derivative_names()
        assert "x_vel" in names
        assert "x_accel" in names
        assert "x_zscore_5" in names
        assert "x_slope_5" in names
        assert "x_rvol_5" in names

    def test_names_count_formula(self):
        """Total = n_cols * (2 + 3 * n_windows)."""
        cols = ["a", "b"]
        wins = [3, 7]
        buf = DerivativeBuffer(columns=cols, temporal_windows=wins)
        names = buf.derivative_names()
        expected_count = len(cols) * (2 + 3 * len(wins))
        assert len(names) == expected_count


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for public properties."""

    def test_columns_property(self):
        buf = DerivativeBuffer(columns=["x", "y"], temporal_windows=[5])
        assert buf.columns == ["x", "y"]

    def test_temporal_windows_property(self):
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[3, 7, 11])
        assert buf.temporal_windows == [3, 7, 11]

    def test_max_window_default(self):
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[5, 10, 20])
        assert buf.max_window == 21  # max(20) + 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_column(self):
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[3])
        bars = _make_bars(4, ["x"])
        for i in range(3):
            assert buf.update(_bar_series(bars, i)) is None
        result = buf.update(_bar_series(bars, 3))
        assert result is not None
        assert len(result) == 5  # vel, accel, zscore_3, slope_3, rvol_3

    def test_single_window(self):
        buf = DerivativeBuffer(columns=["a", "b"], temporal_windows=[2])
        assert buf.max_window == 3
        bars = _make_bars(3, ["a", "b"])
        for i in range(2):
            buf.update(_bar_series(bars, i))
        result = buf.update(_bar_series(bars, 2))
        assert result is not None

    def test_constant_input(self):
        """Constant bars produce zero velocity/acceleration, zero zscore."""
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[3])
        bars = pd.DataFrame({"x": [5.0] * 4})
        for i in range(3):
            buf.update(_bar_series(bars, i))
        result = buf.update(_bar_series(bars, 3))
        # vel=0, accel=0, zscore=0 (std=0 → zscore forced to 0), slope=0, rvol=0
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_extra_columns_in_bar_ok(self):
        """Bar can have extra columns beyond what buffer needs."""
        buf = DerivativeBuffer(columns=["a"], temporal_windows=[2])
        bars = _make_bars(3, ["a", "b", "c", "extra"])
        for i in range(3):
            out = buf.update(_bar_series(bars, i))
        assert out is not None

    def test_output_is_float64(self):
        buf = DerivativeBuffer(columns=COLUMNS, temporal_windows=WINDOWS)
        bars = _make_bars(buf.max_window, COLUMNS)
        for i in range(buf.max_window - 1):
            buf.update(_bar_series(bars, i))
        result = buf.update(_bar_series(bars, buf.max_window - 1))
        assert result.dtype == np.float64

    def test_nan_in_bar_propagates(self):
        """NaN in input should appear in derivative output (not crash)."""
        buf = DerivativeBuffer(columns=["x"], temporal_windows=[3])
        bars = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        for i in range(3):
            buf.update(_bar_series(bars, i))
        result = buf.update(_bar_series(bars, 3))
        # Should not crash; some values will be NaN
        assert result is not None

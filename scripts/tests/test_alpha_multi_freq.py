"""Tests for alpha.multi_freq — macro filter, signal gating, PnL helpers."""

from pathlib import Path


import numpy as np
import pandas as pd
import pytest
from alpha.multi_freq import (
    compute_macro_filter,
    align_macro_to_micro,
    apply_macro_gate,
    profit_sensitive_exit,
    _compute_signal_pnl,
    _sharpe,
    _max_dd,
    MacroFilter,
    MultiFreqResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(n=300, trend="up"):
    """Create synthetic daily OHLCV DataFrame."""
    np.random.seed(42)
    if trend == "up":
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    else:
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# compute_macro_filter
# ---------------------------------------------------------------------------


class TestComputeMacroFilter:
    def test_uptrend_allows_long(self):
        df = _make_daily_df(300, trend="up")
        result = compute_macro_filter(df, fast_period=50, slow_period=200)
        assert len(result) == 300
        assert "long_allowed" in result.columns
        assert "short_allowed" in result.columns
        assert "trend_strength" in result.columns
        # In a clear uptrend after warmup, longs should be allowed
        late = result.iloc[-1]
        assert late["long_allowed"] == True

    def test_before_warmup_allows_both(self):
        df = _make_daily_df(300)
        result = compute_macro_filter(df, fast_period=50, slow_period=200)
        # Before SMA(200) warmup (first 199 rows), both directions allowed
        assert result["long_allowed"].iloc[0] == True
        assert result["short_allowed"].iloc[0] == True

    def test_trend_strength_range(self):
        df = _make_daily_df(300)
        result = compute_macro_filter(df)
        assert result["trend_strength"].min() >= 0.0
        assert result["trend_strength"].max() <= 1.0

    def test_transition_zone(self):
        """When SMA50 ≈ SMA200 (within 1%), both directions allowed."""
        # Create flat price data
        df = pd.DataFrame({"close": np.full(300, 100.0)})
        result = compute_macro_filter(df, fast_period=50, slow_period=200)
        # After warmup, flat prices → SMAs equal → transition zone
        late = result.iloc[-1]
        assert late["long_allowed"] == True
        assert late["short_allowed"] == True


# ---------------------------------------------------------------------------
# align_macro_to_micro
# ---------------------------------------------------------------------------


class TestAlignMacroToMicro:
    def test_basic_alignment(self):
        macro = pd.DataFrame({
            "long_allowed": [True, False, True],
            "short_allowed": [False, True, False],
            "trend_strength": [0.5, 0.8, 0.3],
        })
        micro = pd.DataFrame({"x": range(96 * 3)})  # 3 days at 15min
        result = align_macro_to_micro(macro, micro, micro_timeframe="15min")

        assert result.shape == (96 * 3, 3)
        # First 96 bars should have day 0's values
        assert result[0, 0] == 1.0  # long_allowed=True
        assert result[0, 1] == 0.0  # short_allowed=False
        # Second day's bars
        assert result[96, 0] == 0.0  # long_allowed=False

    def test_different_timeframes(self):
        macro = pd.DataFrame({
            "long_allowed": [True, False],
            "short_allowed": [False, True],
            "trend_strength": [0.5, 0.8],
        })
        micro = pd.DataFrame({"x": range(48)})  # 2 days at 1h
        result = align_macro_to_micro(macro, micro, micro_timeframe="1h")
        assert result.shape == (48, 3)


# ---------------------------------------------------------------------------
# apply_macro_gate
# ---------------------------------------------------------------------------


class TestApplyMacroGate:
    def test_gates_long_when_not_allowed(self):
        micro_signal = np.array([0.5, -0.5, 0.8, -0.8])
        macro_state = np.array([
            [0, 1, 0.5],  # long not allowed, short allowed
            [0, 1, 0.5],
            [0, 1, 0.5],
            [0, 1, 0.5],
        ], dtype=float)
        gated = apply_macro_gate(micro_signal, macro_state)
        # Positive signals should be zeroed (long not allowed)
        assert gated[0] == 0.0
        assert gated[2] == 0.0
        # Negative signals should remain (short allowed)
        assert gated[1] != 0.0
        assert gated[3] != 0.0

    def test_gates_short_when_not_allowed(self):
        micro_signal = np.array([0.5, -0.5])
        macro_state = np.array([
            [1, 0, 0.5],  # long allowed, short not
            [1, 0, 0.5],
        ], dtype=float)
        gated = apply_macro_gate(micro_signal, macro_state)
        assert gated[0] != 0.0  # long passes
        assert gated[1] == 0.0  # short gated

    def test_trend_strength_scaling(self):
        signal = np.array([1.0])
        macro_state = np.array([[1, 1, 1.0]])  # full strength
        gated = apply_macro_gate(signal, macro_state)
        assert abs(gated[0] - 1.0) < 1e-10  # 1.0 * (0.5 + 0.5*1.0) = 1.0

        macro_state_weak = np.array([[1, 1, 0.0]])  # no strength
        gated_weak = apply_macro_gate(signal, macro_state_weak)
        assert abs(gated_weak[0] - 0.5) < 1e-10  # 1.0 * (0.5 + 0.5*0.0) = 0.5


# ---------------------------------------------------------------------------
# profit_sensitive_exit
# ---------------------------------------------------------------------------


class TestProfitSensitiveExit:
    def test_no_modification_without_position(self):
        signal = np.array([0.0, 0.0, 0.0])
        prices = np.array([100, 101, 102], dtype=float)
        result = profit_sensitive_exit(signal, prices, entry_threshold=0.3)
        np.testing.assert_array_equal(result, signal)

    def test_output_length_matches(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = profit_sensitive_exit(signal, prices)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# PnL helpers
# ---------------------------------------------------------------------------


class TestComputeSignalPnl:
    def test_basic(self):
        signal = np.array([1.0, 1.0, 1.0, 0.0])
        prices = np.array([100, 102, 104, 106], dtype=float)
        pnl = _compute_signal_pnl(signal, prices)
        # pnl[0] = 1.0 * (102-100)/100 = 0.02
        # pnl[1] = 1.0 * (104-102)/102 ≈ 0.0196
        # pnl[2] = 1.0 * (106-104)/104 ≈ 0.0192
        assert len(pnl) == 3
        assert abs(pnl[0] - 0.02) < 1e-10

    def test_short_position(self):
        signal = np.array([-1.0, -1.0])
        prices = np.array([100, 102], dtype=float)
        pnl = _compute_signal_pnl(signal, prices)
        assert pnl[0] < 0  # short loses when price goes up


class TestSharpe:
    def test_zero_volatility(self):
        assert _sharpe(np.zeros(100)) == 0.0

    def test_positive_sharpe(self):
        np.random.seed(42)
        pnl = np.ones(96 * 5) * 0.001 + np.random.randn(96 * 5) * 0.0001
        s = _sharpe(pnl)
        assert s > 0

    def test_empty_array(self):
        assert _sharpe(np.array([])) == 0.0


class TestMaxDD:
    def test_no_drawdown(self):
        pnl = np.ones(100) * 0.01
        assert _max_dd(pnl) == 0.0

    def test_with_drawdown(self):
        pnl = np.array([0.01, 0.01, -0.05, 0.01, 0.01])
        dd = _max_dd(pnl)
        assert dd > 0

    def test_empty(self):
        assert _max_dd(np.array([])) == 0.0

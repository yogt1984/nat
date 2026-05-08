"""Tests for data.macro — indicators, RSI, ATR, macro signal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from data.macro import add_indicators, get_macro_signal, _rsi, _atr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n=300, base=100.0, trend=0.0):
    """Synthetic OHLCV DataFrame."""
    np.random.seed(42)
    close = base + np.cumsum(np.random.randn(n) * 0.5 + trend)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.abs(np.random.randn(n) * 1000) + 100
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# _rsi
# ---------------------------------------------------------------------------


class TestRSI:
    def test_uptrend_high_rsi(self):
        # Steadily rising prices → RSI should be high
        prices = np.linspace(100, 200, 100)
        rsi = _rsi(prices, period=14)
        assert rsi[-1] > 70

    def test_downtrend_low_rsi(self):
        prices = np.linspace(200, 100, 100)
        rsi = _rsi(prices, period=14)
        assert rsi[-1] < 30

    def test_rsi_range(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(500))
        rsi = _rsi(prices, period=14)
        # RSI should always be in [0, 100]
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

    def test_flat_prices(self):
        prices = np.full(100, 100.0)
        rsi = _rsi(prices, period=14)
        # No change → gains=0, losses=0 → RSI near 50 or 0
        assert np.all(np.isfinite(rsi))


# ---------------------------------------------------------------------------
# _atr
# ---------------------------------------------------------------------------


class TestATR:
    def test_basic(self):
        high = np.array([102, 104, 106, 108, 110], dtype=float)
        low = np.array([98, 96, 94, 92, 90], dtype=float)
        close = np.array([100, 100, 100, 100, 100], dtype=float)
        atr = _atr(high, low, close, period=3)
        # TR = max(H-L, |H-prev_C|, |L-prev_C|) = max(4, 2+, 2+) = range
        assert len(atr) == 5
        # ATR should be positive after warmup
        assert atr[-1] > 0

    def test_zero_volatility(self):
        high = np.full(20, 100.0)
        low = np.full(20, 100.0)
        close = np.full(20, 100.0)
        atr = _atr(high, low, close, period=5)
        # No volatility → ATR = 0
        assert atr[-1] == 0.0

    def test_output_length(self):
        n = 50
        atr = _atr(np.ones(n) * 105, np.ones(n) * 95, np.ones(n) * 100, period=14)
        assert len(atr) == n


# ---------------------------------------------------------------------------
# add_indicators
# ---------------------------------------------------------------------------


class TestAddIndicators:
    def test_adds_sma_columns(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        for w in [7, 21, 50, 100, 200]:
            assert f"sma_{w}" in result.columns
        for w in [12, 26]:
            assert f"ema_{w}" in result.columns

    def test_adds_crossover_signals(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "cross_50_200" in result.columns
        assert "cross_7_21" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_cross" in result.columns

    def test_adds_rsi(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "rsi_14" in result.columns
        rsi_valid = result["rsi_14"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_adds_support_resistance(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "resist_20" in result.columns
        assert "support_20" in result.columns
        assert "dist_resist_pct" in result.columns
        assert "dist_support_pct" in result.columns

    def test_adds_volume_profile(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "rel_volume" in result.columns

    def test_adds_atr(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "atr_14" in result.columns
        assert "atr_pct" in result.columns

    def test_adds_psychological_levels(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        assert "psych_level" in result.columns
        assert "dist_psych_pct" in result.columns

    def test_preserves_original_columns(self):
        df = _make_ohlcv(100)
        original_cols = set(df.columns)
        result = add_indicators(df)
        for col in original_cols:
            assert col in result.columns

    def test_crossover_values(self):
        df = _make_ohlcv(300)
        result = add_indicators(df)
        # cross_50_200 should be +1 or -1
        valid = result["cross_50_200"].dropna()
        assert set(valid.unique()).issubset({1, -1})


# ---------------------------------------------------------------------------
# get_macro_signal
# ---------------------------------------------------------------------------


class TestGetMacroSignal:
    def test_output_columns(self):
        df = _make_ohlcv(300)
        df = add_indicators(df)
        signal = get_macro_signal(df)
        assert "direction" in signal.columns
        assert "strength" in signal.columns
        assert "regime" in signal.columns
        assert "close" in signal.columns

    def test_direction_range(self):
        df = _make_ohlcv(300)
        df = add_indicators(df)
        signal = get_macro_signal(df)
        assert set(signal["direction"].unique()).issubset({-1, 0, 1})

    def test_strength_range(self):
        df = _make_ohlcv(300)
        df = add_indicators(df)
        signal = get_macro_signal(df)
        assert (signal["strength"] >= 0).all()
        assert (signal["strength"] <= 1).all()

    def test_regime_values(self):
        df = _make_ohlcv(300)
        df = add_indicators(df)
        signal = get_macro_signal(df)
        valid_regimes = {"trending_up", "trending_down", "ranging", "breakout", "unknown"}
        assert set(signal["regime"].unique()).issubset(valid_regimes)

    def test_length_preserved(self):
        df = _make_ohlcv(300)
        df = add_indicators(df)
        signal = get_macro_signal(df)
        assert len(signal) == 300

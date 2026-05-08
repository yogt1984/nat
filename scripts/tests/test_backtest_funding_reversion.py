"""Tests for backtest_funding_reversion.py — backtest_signal function."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_funding_reversion import backtest_signal, MAKER_FEE, TAKER_FEE, COST_PER_TRADE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_fee_structure(self):
        assert MAKER_FEE == 0.0002
        assert TAKER_FEE == 0.00035
        assert COST_PER_TRADE == TAKER_FEE


# ---------------------------------------------------------------------------
# backtest_signal
# ---------------------------------------------------------------------------

class TestBacktestSignal:
    def test_basic_output_keys(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        expected_keys = {
            "n_bars", "total_return_pct", "gross_return_pct",
            "total_costs_pct", "sharpe_ratio", "max_drawdown_pct",
            "n_trades", "win_rate", "time_in_market",
            "mean_bar_return_bps", "std_bar_return_bps",
        }
        assert expected_keys.issubset(result.keys())

    def test_long_only_in_uptrend(self):
        """Constant long signal in a rising market → positive return."""
        n = 100
        prices = 100.0 + np.arange(n) * 0.1  # steadily rising
        signal = np.ones(n)
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        assert result["total_return_pct"] > 0
        assert result["gross_return_pct"] > 0

    def test_short_in_uptrend_loses(self):
        """Constant short signal in a rising market → negative return."""
        n = 100
        prices = 100.0 + np.arange(n) * 0.1
        signal = -np.ones(n)
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        assert result["gross_return_pct"] < 0

    def test_zero_signal_zero_return(self):
        """No position → zero gross return."""
        n = 50
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        signal = np.zeros(n)
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        assert result["gross_return_pct"] == pytest.approx(0.0, abs=0.001)
        assert result["time_in_market"] == pytest.approx(0.0)

    def test_costs_reduce_returns(self):
        """With transaction costs, net return should be lower than gross."""
        n = 100
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        signal = np.sign(rng.randn(n))  # frequent flips

        result_no_cost = backtest_signal(prices, signal, cost_per_trade=0.0)
        result_with_cost = backtest_signal(prices, signal, cost_per_trade=0.001)

        assert result_with_cost["total_return_pct"] < result_no_cost["total_return_pct"]
        assert result_with_cost["total_costs_pct"] > 0

    def test_n_bars_matches_input(self):
        prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
        signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        assert result["n_bars"] == 4  # diff produces n-1 returns

    def test_max_drawdown_non_negative(self):
        n = 200
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        signal = np.sign(rng.randn(n))
        result = backtest_signal(prices, signal)
        assert result["max_drawdown_pct"] >= 0

    def test_win_rate_in_range(self):
        n = 200
        rng = np.random.RandomState(42)
        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        signal = np.sign(rng.randn(n))
        result = backtest_signal(prices, signal)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_time_in_market_fraction(self):
        n = 100
        # Half the time in market
        signal = np.zeros(n)
        signal[:50] = 1.0
        prices = 100.0 + np.arange(n, dtype=float) * 0.01
        result = backtest_signal(prices, signal, cost_per_trade=0.0)
        assert 0.0 < result["time_in_market"] < 1.0

    def test_empty_returns_error(self):
        """All-NaN prices should produce error result."""
        prices = np.array([np.nan, np.nan, np.nan])
        signal = np.array([1.0, 1.0, 1.0])
        result = backtest_signal(prices, signal)
        assert "error" in result

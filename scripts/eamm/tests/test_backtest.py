"""Unit tests for EAMM Backtester."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.backtest import run_backtest, BacktestResult


def _make_data(n=2000, trend=0.0, vol=0.5):
    np.random.seed(42)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend / 10000 + np.random.randn() * vol / 10000))
    midprices = np.array(prices)
    timestamps = np.arange(n) * 100_000_000  # 100ms apart
    volatility = np.full(n, vol / 10000)
    return midprices, timestamps, volatility


class TestBasicBacktest:
    def test_returns_result(self):
        midprices, timestamps, volatility = _make_data()
        spreads = np.full(len(midprices), 5.0)  # 5 bps constant
        result = run_backtest(midprices, timestamps, spreads, volatility,
                              horizon=100)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(midprices)
        assert len(result.inventory_curve) == len(midprices)

    def test_has_trades(self):
        midprices, timestamps, volatility = _make_data(vol=2.0)
        spreads = np.full(len(midprices), 3.0)
        result = run_backtest(midprices, timestamps, spreads, volatility,
                              horizon=50)
        assert result.n_trades > 0

    def test_inventory_bounded(self):
        midprices, timestamps, volatility = _make_data(vol=3.0)
        spreads = np.full(len(midprices), 2.0)
        result = run_backtest(midprices, timestamps, spreads, volatility,
                              gamma=0.1, q_max=2.0, horizon=50)
        assert np.all(np.abs(result.inventory_curve) <= 2.01)


class TestWideSpreadsFewerFills:
    def test_wider_spread_fewer_fills(self):
        midprices, timestamps, volatility = _make_data(vol=2.0)
        narrow = np.full(len(midprices), 1.0)
        wide = np.full(len(midprices), 20.0)
        r_narrow = run_backtest(midprices, timestamps, narrow, volatility, horizon=50)
        r_wide = run_backtest(midprices, timestamps, wide, volatility, horizon=50)
        assert r_narrow.n_trades >= r_wide.n_trades


class TestLiquidation:
    def test_final_inventory_zero(self):
        midprices, timestamps, volatility = _make_data(vol=3.0)
        spreads = np.full(len(midprices), 2.0)
        result = run_backtest(midprices, timestamps, spreads, volatility,
                              horizon=50, q_max=1.0)
        # After liquidation, final inventory should be 0
        assert abs(result.inventory_curve[-1]) < 0.01


class TestMetrics:
    def test_sharpe_defined(self):
        midprices, timestamps, volatility = _make_data(vol=2.0)
        spreads = np.full(len(midprices), 5.0)
        result = run_backtest(midprices, timestamps, spreads, volatility, horizon=50)
        # Sharpe should be a finite number
        assert np.isfinite(result.sharpe)

    def test_max_drawdown_non_negative(self):
        midprices, timestamps, volatility = _make_data()
        spreads = np.full(len(midprices), 5.0)
        result = run_backtest(midprices, timestamps, spreads, volatility, horizon=100)
        assert result.max_drawdown >= 0.0

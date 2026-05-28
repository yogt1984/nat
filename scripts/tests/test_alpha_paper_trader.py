"""Tests for alpha.paper_trader — trade logging, IC decay, reconciliation."""

from pathlib import Path


import numpy as np
import pytest
from alpha.paper_trader import (
    log_signal,
    close_trade,
    compute_rolling_ic,
    detect_ic_decay,
    reconcile_day,
    PaperTrade,
    DailyReconciliation,
    PaperTradingResult,
)


# ---------------------------------------------------------------------------
# log_signal
# ---------------------------------------------------------------------------


class TestLogSignal:
    def test_long_signal(self):
        trade = log_signal("BTC", signal=0.7, price=50000.0, timestamp="2026-01-01T00:00:00")
        assert trade.symbol == "BTC"
        assert trade.direction == "long"
        assert trade.signal_value == 0.7
        assert trade.entry_price == 50000.0
        assert trade.exit_price is None

    def test_short_signal(self):
        trade = log_signal("ETH", signal=-0.5, price=3000.0)
        assert trade.direction == "short"
        assert trade.signal_value == -0.5

    def test_auto_timestamp(self):
        trade = log_signal("SOL", signal=0.3, price=100.0)
        assert trade.timestamp is not None
        assert len(trade.timestamp) > 0


# ---------------------------------------------------------------------------
# close_trade
# ---------------------------------------------------------------------------


class TestCloseTrade:
    def test_long_profit(self):
        trade = PaperTrade(
            timestamp="t0", symbol="BTC", direction="long",
            signal_value=0.5, entry_price=100.0,
        )
        closed = close_trade(trade, exit_price=110.0, exit_reason="signal", holding_bars=10)
        assert closed.exit_price == 110.0
        assert closed.exit_reason == "signal"
        assert closed.pnl_pct == pytest.approx(10.0)
        assert closed.holding_bars == 10

    def test_long_loss(self):
        trade = PaperTrade(
            timestamp="t0", symbol="BTC", direction="long",
            signal_value=0.5, entry_price=100.0,
        )
        closed = close_trade(trade, exit_price=95.0)
        assert closed.pnl_pct == pytest.approx(-5.0)

    def test_short_profit(self):
        trade = PaperTrade(
            timestamp="t0", symbol="ETH", direction="short",
            signal_value=-0.5, entry_price=100.0,
        )
        closed = close_trade(trade, exit_price=90.0)
        assert closed.pnl_pct == pytest.approx(10.0)

    def test_short_loss(self):
        trade = PaperTrade(
            timestamp="t0", symbol="ETH", direction="short",
            signal_value=-0.5, entry_price=100.0,
        )
        closed = close_trade(trade, exit_price=105.0)
        assert closed.pnl_pct == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# compute_rolling_ic
# ---------------------------------------------------------------------------


class TestComputeRollingIC:
    def test_correlated_signal(self):
        np.random.seed(42)
        n = 1000
        signals = np.arange(n, dtype=float)
        returns = signals * 0.01 + np.random.randn(n) * 0.001
        ic = compute_rolling_ic(signals, returns, window=100)
        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0
        assert np.mean(valid) > 0.8

    def test_uncorrelated(self):
        np.random.seed(42)
        n = 1000
        signals = np.random.randn(n)
        returns = np.random.randn(n)
        ic = compute_rolling_ic(signals, returns, window=100)
        valid = ic[~np.isnan(ic)]
        if len(valid) > 0:
            assert abs(np.mean(valid)) < 0.3

    def test_short_data(self):
        signals = np.arange(50, dtype=float)
        returns = np.arange(50, dtype=float)
        ic = compute_rolling_ic(signals, returns, window=100)
        # All NaN since window > data length
        assert np.all(np.isnan(ic))

    def test_nan_handling(self):
        np.random.seed(42)
        n = 500
        signals = np.arange(n, dtype=float)
        returns = np.arange(n, dtype=float)
        signals[10:20] = np.nan
        ic = compute_rolling_ic(signals, returns, window=100)
        # Should still produce some valid values
        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# detect_ic_decay
# ---------------------------------------------------------------------------


class TestDetectICDecay:
    def test_no_decay(self):
        rolling_ic = np.full(960, 0.04)  # 10 days at 96 bars/day
        is_decayed, ratio, days = detect_ic_decay(rolling_ic, backtest_ic=0.05)
        assert is_decayed is False
        assert ratio > 0.5

    def test_decay_detected(self):
        # IC drops to near zero
        rolling_ic = np.concatenate([
            np.full(480, 0.05),  # 5 days healthy
            np.full(480, 0.001),  # 5 days decayed
        ])
        is_decayed, ratio, days = detect_ic_decay(
            rolling_ic, backtest_ic=0.05, decay_threshold=0.5, consecutive_days=3,
        )
        assert is_decayed is True
        assert days >= 3

    def test_all_nan(self):
        rolling_ic = np.full(100, np.nan)
        is_decayed, ratio, days = detect_ic_decay(rolling_ic, backtest_ic=0.05)
        assert is_decayed is False
        assert ratio == 0.0
        assert days == 0

    def test_zero_backtest_ic(self):
        """When backtest IC is 0, ratio=0 for all days → triggers decay."""
        rolling_ic = np.full(960, 0.03)
        is_decayed, ratio, days = detect_ic_decay(rolling_ic, backtest_ic=0.0)
        # ratio is 0 when backtest_ic=0, so all days count as "low"
        assert is_decayed is True


# ---------------------------------------------------------------------------
# PaperTradingResult
# ---------------------------------------------------------------------------


class TestPaperTradingResult:
    def test_gate_all_pass(self):
        result = PaperTradingResult(
            start_date="d0", end_date="d14", n_days=14, n_trades=50,
            total_pnl_pct=5.0, paper_sharpe=1.5, backtest_sharpe=1.0,
            sharpe_ratio=1.5, max_daily_loss_pct=-0.5,
            ic_decay_pct=10.0, error_free_days=14, daily_reports=[],
            gate_sharpe_within_2x=True, gate_no_big_daily_loss=True,
            gate_ic_stable=True, gate_infra_stable=True, gate_pass=True,
        )
        assert result.gate_pass is True

    def test_gate_fail(self):
        result = PaperTradingResult(
            start_date="d0", end_date="d7", n_days=7, n_trades=5,
            total_pnl_pct=-3.0, paper_sharpe=0.2, backtest_sharpe=1.0,
            sharpe_ratio=0.2, max_daily_loss_pct=-3.0,
            ic_decay_pct=60.0, error_free_days=5, daily_reports=[],
            gate_sharpe_within_2x=False, gate_no_big_daily_loss=False,
            gate_ic_stable=False, gate_infra_stable=True, gate_pass=False,
        )
        assert result.gate_pass is False

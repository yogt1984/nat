"""
Skeptical Tests for Backtest Engine

These tests are designed to catch common backtesting pitfalls:
- Look-ahead bias
- Incorrect P&L calculation
- Off-by-one errors
- Edge cases that crash production
"""

import pytest
import polars as pl
import numpy as np
from backtest.engine import run_backtest, Trade, BacktestResult
from backtest.strategy import Strategy
from backtest.costs import CostModel, zero_cost


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_uptrend_df():
    """Create simple uptrending price data."""
    n = 1000
    return pl.DataFrame({
        "timestamp_ms": list(range(n)),
        "raw_midprice": [100.0 + i * 0.1 for i in range(n)],  # 100 -> 200
        "signal": [True if i % 100 == 0 else False for i in range(n)],
        "exit_signal": [True if i % 100 == 50 else False for i in range(n)],
    })


@pytest.fixture
def simple_downtrend_df():
    """Create simple downtrending price data."""
    n = 1000
    return pl.DataFrame({
        "timestamp_ms": list(range(n)),
        "raw_midprice": [200.0 - i * 0.1 for i in range(n)],  # 200 -> 100
        "signal": [True if i % 100 == 0 else False for i in range(n)],
        "exit_signal": [True if i % 100 == 50 else False for i in range(n)],
    })


@pytest.fixture
def flat_price_df():
    """Create flat price data."""
    n = 1000
    return pl.DataFrame({
        "timestamp_ms": list(range(n)),
        "raw_midprice": [100.0] * n,
        "signal": [True if i % 100 == 0 else False for i in range(n)],
        "exit_signal": [True if i % 100 == 50 else False for i in range(n)],
    })


@pytest.fixture
def always_enter_strategy():
    """Strategy that enters at specific intervals."""
    return Strategy(
        name="test_strategy",
        entry_condition=lambda df: df["signal"],
        exit_condition=lambda df: df["exit_signal"],
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        max_holding_bars=100,
        direction="long",
    )


@pytest.fixture
def always_true_entry():
    """Strategy that always wants to enter."""
    return Strategy(
        name="always_enter",
        entry_condition=lambda df: pl.Series([True] * len(df)),
        exit_condition=lambda df: pl.Series([False] * len(df)),
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        max_holding_bars=50,
        direction="long",
    )


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


class TestBasicBacktest:
    """Basic backtest functionality tests."""

    def test_backtest_returns_result(self, simple_uptrend_df, always_enter_strategy):
        """Backtest should return BacktestResult."""
        result = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        assert isinstance(result, BacktestResult)

    def test_equity_curve_starts_at_capital(self, simple_uptrend_df, always_enter_strategy):
        """Equity curve should start at initial capital."""
        result = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
            initial_capital=10000.0,
        )
        assert result.equity_curve[0] == 10000.0

    def test_no_trades_when_no_entry_signals(self, simple_uptrend_df):
        """No trades when entry condition is never true."""
        never_enter = Strategy(
            name="never_enter",
            entry_condition=lambda df: pl.Series([False] * len(df)),
            exit_condition=lambda df: pl.Series([False] * len(df)),
            direction="long",
        )
        result = run_backtest(simple_uptrend_df, never_enter, zero_cost())
        assert result.total_trades == 0

    def test_trades_recorded(self, simple_uptrend_df, always_enter_strategy):
        """Trades should be recorded."""
        result = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        assert len(result.trades) > 0

    def test_all_trades_closed(self, simple_uptrend_df, always_enter_strategy):
        """All trades should be closed at end of backtest."""
        result = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        for trade in result.trades:
            assert trade.is_closed, "All trades should be closed"


# =============================================================================
# P&L CALCULATION TESTS
# =============================================================================


class TestPnLCalculation:
    """Test P&L is calculated correctly."""

    def test_long_profit_on_uptrend(self, simple_uptrend_df, always_enter_strategy):
        """Long position should profit in uptrend."""
        result = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        # With zero costs and uptrend, should be profitable
        assert result.total_return_pct > 0, "Should profit on uptrend"

    def test_long_loss_on_downtrend(self, simple_downtrend_df, always_enter_strategy):
        """Long position should lose in downtrend."""
        result = run_backtest(
            simple_downtrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        # Long in downtrend should lose
        assert result.total_return_pct < 0, "Should lose on downtrend"

    def test_short_profit_on_downtrend(self, simple_downtrend_df):
        """Short position should profit in downtrend."""
        short_strategy = Strategy(
            name="short_test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
            max_holding_bars=100,
            direction="short",
        )
        result = run_backtest(
            simple_downtrend_df,
            short_strategy,
            zero_cost(),
        )
        # Short in downtrend should profit
        assert result.total_return_pct > 0, "Short should profit on downtrend"

    def test_costs_reduce_pnl(self, simple_uptrend_df, always_enter_strategy):
        """Adding costs should reduce P&L."""
        result_no_cost = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            zero_cost(),
        )
        result_with_cost = run_backtest(
            simple_uptrend_df,
            always_enter_strategy,
            CostModel(fee_bps=10.0, slippage_bps=5.0),
        )

        assert result_with_cost.total_return_pct < result_no_cost.total_return_pct, (
            "Costs should reduce returns"
        )

    def test_flat_price_loses_costs(self, flat_price_df, always_enter_strategy):
        """Flat price with costs should result in losses."""
        result = run_backtest(
            flat_price_df,
            always_enter_strategy,
            CostModel(fee_bps=10.0, slippage_bps=5.0),
        )

        if result.total_trades > 0:
            assert result.total_return_pct < 0, "Should lose costs on flat price"


# =============================================================================
# STOP LOSS AND TAKE PROFIT TESTS
# =============================================================================


class TestStopLossAndTakeProfit:
    """Test stop loss and take profit work correctly."""

    def test_stop_loss_triggered(self):
        """Stop loss should trigger on large drop."""
        # Create data with big drop
        n = 100
        prices = [100.0] * 10 + [90.0] * 90  # 10% drop after entry

        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": prices,
            "signal": [True] + [False] * (n - 1),
            "exit_signal": [False] * n,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=5.0,  # 5% stop
            take_profit_pct=20.0,
            max_holding_bars=1000,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        assert result.total_trades == 1
        assert result.trades[0].exit_reason == "stop"

    def test_take_profit_triggered(self):
        """Take profit should trigger on large gain."""
        # Create data with big rise
        n = 100
        prices = [100.0] * 10 + [115.0] * 90  # 15% rise after entry

        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": prices,
            "signal": [True] + [False] * (n - 1),
            "exit_signal": [False] * n,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=20.0,
            take_profit_pct=10.0,  # 10% take profit
            max_holding_bars=1000,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        assert result.total_trades == 1
        assert result.trades[0].exit_reason == "target"

    def test_timeout_triggered(self):
        """Position should close after max holding bars."""
        n = 200
        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0] * n,
            "signal": [True] + [False] * (n - 1),
            "exit_signal": [False] * n,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=50.0,  # Very wide stops
            take_profit_pct=50.0,
            max_holding_bars=50,  # Short timeout
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        assert result.total_trades == 1
        assert result.trades[0].exit_reason == "timeout"
        assert result.trades[0].holding_bars == 50


# =============================================================================
# LOOK-AHEAD BIAS TESTS
# =============================================================================


class TestNoLookAheadBias:
    """Tests to ensure no look-ahead bias in the backtest."""

    def test_entry_uses_current_bar_price(self):
        """Entry price should be the bar where entry signal occurred."""
        n = 100
        prices = list(range(100, 200))  # Increasing prices

        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [float(p) for p in prices],
            "signal": [False] * 50 + [True] + [False] * 49,  # Enter at bar 50
            "exit_signal": [False] * 70 + [True] + [False] * 29,  # Exit at bar 70
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=50.0,
            take_profit_pct=50.0,
            max_holding_bars=1000,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        assert result.total_trades == 1
        trade = result.trades[0]

        # Entry should be at bar 50, price 150
        assert trade.entry_idx == 50
        assert trade.raw_entry_price == 150.0

        # Exit should be at bar 70, price 170
        assert trade.exit_idx == 70
        assert trade.raw_exit_price == 170.0

    def test_cannot_trade_on_future_information(self):
        """
        If we flip the price series after a trade,
        P&L should change accordingly.

        This tests that we're not accidentally using future prices.
        """
        n = 100

        # Uptrend after entry
        df_up = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0] * 50 + [float(100 + i) for i in range(50)],
            "signal": [False] * 49 + [True] + [False] * 50,
            "exit_signal": [False] * 99 + [True],
        })

        # Downtrend after entry
        df_down = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0] * 50 + [float(100 - i) for i in range(50)],
            "signal": [False] * 49 + [True] + [False] * 50,
            "exit_signal": [False] * 99 + [True],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=100.0,  # Very wide to avoid stops
            take_profit_pct=100.0,
            max_holding_bars=1000,
            direction="long",
        )

        result_up = run_backtest(df_up, strategy, zero_cost())
        result_down = run_backtest(df_down, strategy, zero_cost())

        # Results should be different
        assert result_up.total_return_pct != result_down.total_return_pct
        assert result_up.total_return_pct > 0  # Should profit on uptrend
        assert result_down.total_return_pct < 0  # Should lose on downtrend


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases that could crash or produce wrong results."""

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        df = pl.DataFrame({
            "timestamp_ms": [],
            "raw_midprice": [],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: pl.Series([]),
            exit_condition=lambda df: pl.Series([]),
            direction="long",
        )

        with pytest.raises(ValueError):
            run_backtest(df, strategy, zero_cost())

    def test_single_row_dataframe(self):
        """Should handle single row dataframe."""
        df = pl.DataFrame({
            "timestamp_ms": [1],
            "raw_midprice": [100.0],
            "signal": [True],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: pl.Series([False]),
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())
        # Should handle gracefully
        assert isinstance(result, BacktestResult)

    def test_missing_price_column(self):
        """Should raise error if price column missing."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3],
            "signal": [True, False, False],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: pl.Series([False] * len(df)),
            direction="long",
        )

        with pytest.raises(ValueError):
            run_backtest(df, strategy, zero_cost())

    def test_nan_prices_handled(self):
        """Should handle NaN prices gracefully."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3, 4, 5],
            "raw_midprice": [100.0, float("nan"), 102.0, float("nan"), 104.0],
            "signal": [True, False, False, False, False],
            "exit_signal": [False, False, False, False, True],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            direction="long",
        )

        # Should not crash
        result = run_backtest(df, strategy, zero_cost())
        assert isinstance(result, BacktestResult)

    def test_zero_prices_handled(self):
        """Should handle zero prices gracefully."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3, 4, 5],
            "raw_midprice": [100.0, 0.0, 102.0, 0.0, 104.0],
            "signal": [True, False, False, False, False],
            "exit_signal": [False, False, False, False, True],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            direction="long",
        )

        # Should not crash
        result = run_backtest(df, strategy, zero_cost())
        assert isinstance(result, BacktestResult)

    def test_very_long_backtest(self):
        """Should handle long backtest without memory issues."""
        n = 100_000
        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0 + np.sin(i / 100) for i in range(n)],
            "signal": [i % 1000 == 0 for i in range(n)],
            "exit_signal": [i % 1000 == 500 for i in range(n)],
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())
        assert result.total_trades == 100  # 100k / 1000


# =============================================================================
# METRICS CALCULATION TESTS
# =============================================================================


class TestMetricsCalculation:
    """Test that metrics are calculated correctly."""

    def test_win_rate_calculation(self):
        """Win rate should be correct."""
        # Create trades with known outcomes
        n = 100
        # 60 winning trades, 40 losing
        prices = []
        signals = []
        exits = []

        for i in range(100):
            # Each "trade" is 10 bars
            start_price = 100.0
            if i < 60:
                # Winner: price goes up
                end_price = 105.0
            else:
                # Loser: price goes down
                end_price = 95.0

            prices.extend([start_price] * 5 + [end_price] * 5)
            signals.extend([True] + [False] * 9)
            exits.extend([False] * 9 + [True])

        df = pl.DataFrame({
            "timestamp_ms": list(range(len(prices))),
            "raw_midprice": prices,
            "signal": signals,
            "exit_signal": exits,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=50.0,
            take_profit_pct=50.0,
            max_holding_bars=100,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        assert result.total_trades == 100
        assert abs(result.win_rate - 0.60) < 0.01, f"Win rate should be 60%, got {result.win_rate}"

    def test_max_drawdown_calculation(self):
        """Max drawdown should capture largest peak-to-trough decline."""
        # Create data with multiple trades to build equity curve with drawdown
        n = 200

        # Build arrays of correct length
        prices = [100.0] * 25 + [120.0] * 25 + [120.0] * 25 + [84.0] * 125
        signal = [False] * n
        exit_signal = [False] * n

        signal[0] = True      # Entry 1 at bar 0
        exit_signal[25] = True  # Exit 1 at bar 25
        signal[50] = True     # Entry 2 at bar 50
        exit_signal[75] = True  # Exit 2 at bar 75

        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": prices,
            "signal": signal,
            "exit_signal": exit_signal,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=100.0,
            take_profit_pct=100.0,
            max_holding_bars=1000,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        # After first trade: capital = 12000 (peak)
        # After second trade: capital = 12000 * 0.7 = 8400
        # Drawdown from 12000 to 8400 = 30%
        assert result.max_drawdown_pct > 25, f"Expected DD > 25%, got {result.max_drawdown_pct}"

    def test_profit_factor_calculation(self):
        """Profit factor should be gross profit / gross loss."""
        n = 200

        # Build arrays of correct length
        prices = [100.0] * 25 + [110.0] * 25 + [110.0] * 25 + [104.5] * 125
        signal = [False] * n
        exit_signal = [False] * n

        signal[0] = True       # Entry 1 at bar 0
        exit_signal[25] = True  # Exit 1 at bar 25 (+10%)
        signal[50] = True      # Entry 2 at bar 50
        exit_signal[75] = True  # Exit 2 at bar 75 (-5%)

        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": prices,
            "signal": signal,
            "exit_signal": exit_signal,
        })

        strategy = Strategy(
            name="test",
            entry_condition=lambda df: df["signal"],
            exit_condition=lambda df: df["exit_signal"],
            stop_loss_pct=100.0,
            take_profit_pct=100.0,
            max_holding_bars=1000,
            direction="long",
        )

        result = run_backtest(df, strategy, zero_cost())

        # First trade: +10%, Second trade: -5%
        # Profit factor should be 10 / 5 = 2.0
        assert result.total_trades == 2, f"Expected 2 trades, got {result.total_trades}"
        assert result.profit_factor > 1.5, f"Expected PF > 1.5, got {result.profit_factor}"

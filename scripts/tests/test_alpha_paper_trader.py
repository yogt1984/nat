"""Tests for alpha.paper_trader — zscore params, signal generation, trade logic, summarization."""

import numpy as np
import pandas as pd
import pytest
from alpha.paper_trader import (
    PaperTrade,
    DailySummary,
    compute_zscore_params_3f,
    apply_signal_3f,
    generate_trades,
    summarize_day,
    aggregate_to_bars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic bar DataFrame matching paper_trader expectations."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "timestamp_ns": np.arange(n) * 300_000_000_000,
        "midprice_last": 50000.0 + rng.randn(n).cumsum() * 10,
        "spread_bps_last": rng.uniform(1.0, 5.0, n),
        "depth_5_std": rng.uniform(0.0, 100.0, n),
        "vwap_deviation_std": rng.uniform(0.0, 0.01, n),
        "n_ticks": rng.randint(20, 100, n),
    })


# ---------------------------------------------------------------------------
# compute_zscore_params_3f
# ---------------------------------------------------------------------------


class TestComputeZscoreParams3f:
    def test_basic(self):
        bars = _make_bars(200)
        params = compute_zscore_params_3f([bars])
        assert params is not None
        for key in ["spread_mean", "spread_std", "depth_mean", "depth_std",
                     "vwap_mean", "vwap_std", "p_long", "p_short"]:
            assert key in params
        assert params["p_long"] > params["p_short"]

    def test_multiple_train_windows(self):
        bars1 = _make_bars(100, seed=1)
        bars2 = _make_bars(100, seed=2)
        params = compute_zscore_params_3f([bars1, bars2])
        assert params is not None
        assert params["spread_std"] > 0

    def test_insufficient_data(self):
        bars = _make_bars(5)
        params = compute_zscore_params_3f([bars])
        assert params is None

    def test_missing_vwap_column(self):
        bars = _make_bars(100)
        bars = bars.drop(columns=["vwap_deviation_std"])
        params = compute_zscore_params_3f([bars])
        assert params is None


# ---------------------------------------------------------------------------
# apply_signal_3f
# ---------------------------------------------------------------------------


class TestApplySignal3f:
    def test_adds_direction_column(self):
        bars = _make_bars(200)
        params = compute_zscore_params_3f([bars])
        result = apply_signal_3f(bars, params)
        assert "composite" in result.columns
        assert "direction" in result.columns
        assert set(result["direction"].unique()).issubset({-1, 0, 1})

    def test_long_short_thresholds(self):
        bars = _make_bars(500, seed=99)
        params = compute_zscore_params_3f([bars])
        result = apply_signal_3f(bars, params)
        # Should have some longs (composite >= p_long) and shorts (composite <= p_short)
        assert (result["direction"] == 1).sum() > 0
        assert (result["direction"] == -1).sum() > 0
        # Neutral should be the majority
        assert (result["direction"] == 0).sum() > (result["direction"] != 0).sum()

    def test_does_not_mutate_input(self):
        bars = _make_bars(100)
        params = compute_zscore_params_3f([bars])
        original_cols = set(bars.columns)
        apply_signal_3f(bars, params)
        assert set(bars.columns) == original_cols


# ---------------------------------------------------------------------------
# generate_trades
# ---------------------------------------------------------------------------


class TestGenerateTrades:
    def test_basic(self):
        bars = _make_bars(200)
        params = compute_zscore_params_3f([bars])
        scored = apply_signal_3f(bars, params)
        trades = generate_trades(scored, "2026-05-01", "BTC")
        assert isinstance(trades, list)
        # Should produce some trades (P80/P20 thresholds → ~40% of bars)
        assert len(trades) > 0
        for t in trades:
            assert isinstance(t, PaperTrade)
            assert t.symbol == "BTC"
            assert t.direction in (1, -1)
            assert t.exit_price is not None
            assert t.gross_bps is not None
            assert t.net_bps is not None

    def test_no_signal_no_trades(self):
        bars = _make_bars(200)
        params = compute_zscore_params_3f([bars])
        scored = apply_signal_3f(bars, params)
        # Set all directions to 0
        scored["direction"] = 0
        trades = generate_trades(scored, "2026-05-01", "BTC")
        assert trades == []

    def test_net_less_than_gross(self):
        """Net PnL should be less than gross due to fees."""
        bars = _make_bars(300)
        params = compute_zscore_params_3f([bars])
        scored = apply_signal_3f(bars, params)
        trades = generate_trades(scored, "2026-05-01", "BTC")
        for t in trades:
            assert t.net_bps <= t.gross_bps


# ---------------------------------------------------------------------------
# summarize_day
# ---------------------------------------------------------------------------


class TestSummarizeDay:
    def test_basic(self):
        trades = [
            PaperTrade("2026-05-01", 0, "BTC", 1, 0.5, 50000.0, 50100.0, 20, 20.0, 13.0),
            PaperTrade("2026-05-01", 5, "BTC", -1, -0.3, 50100.0, 50000.0, 25, 19.96, 12.96),
            PaperTrade("2026-05-01", 10, "BTC", 1, 0.6, 50000.0, 49900.0, 30, -20.0, -27.0),
        ]
        summary = summarize_day(trades, "2026-05-01", "BTC")
        assert isinstance(summary, DailySummary)
        assert summary.n_trades == 3
        assert summary.n_long == 2
        assert summary.n_short == 1
        assert summary.max_loss_bps < 0

    def test_empty_trades(self):
        summary = summarize_day([], "2026-05-01", "BTC")
        assert summary.n_trades == 0
        assert summary.gross_bps == 0.0
        assert summary.net_bps == 0.0
        assert summary.win_rate == 0.0

    def test_all_winners(self):
        trades = [
            PaperTrade("d", 0, "BTC", 1, 0.5, 100.0, 110.0, 20, 100.0, 93.0),
            PaperTrade("d", 5, "BTC", 1, 0.6, 100.0, 105.0, 25, 50.0, 43.0),
        ]
        summary = summarize_day(trades, "d", "BTC")
        assert summary.win_rate == 1.0
        assert summary.total_net_bps > 0


# ---------------------------------------------------------------------------
# aggregate_to_bars
# ---------------------------------------------------------------------------


class TestAggregateToBars:
    def test_basic(self):
        n = 1000
        rng = np.random.RandomState(42)
        ticks = pd.DataFrame({
            "timestamp_ns": np.arange(n) * 100_000_000,  # 100ms apart
            "raw_midprice": 50000.0 + rng.randn(n).cumsum(),
            "raw_spread_bps": rng.uniform(1.0, 5.0, n),
            "raw_ask_depth_5": rng.uniform(10.0, 100.0, n),
            "flow_vwap_deviation": rng.uniform(-0.01, 0.01, n),
        })
        bars = aggregate_to_bars(ticks)
        assert "midprice_last" in bars.columns
        assert "spread_bps_last" in bars.columns
        assert "n_ticks" in bars.columns
        assert len(bars) > 0
        # All bars should have >= 10 ticks
        assert (bars["n_ticks"] >= 10).all()

    def test_short_data_filtered(self):
        """Very short data should produce bars that get filtered by min_ticks."""
        ticks = pd.DataFrame({
            "timestamp_ns": [0, 100_000_000, 200_000_000],
            "raw_midprice": [100.0, 101.0, 102.0],
            "raw_spread_bps": [1.0, 1.0, 1.0],
            "raw_ask_depth_5": [50.0, 50.0, 50.0],
        })
        bars = aggregate_to_bars(ticks)
        # 3 ticks in one bar < min_ticks=10 → empty
        assert len(bars) == 0

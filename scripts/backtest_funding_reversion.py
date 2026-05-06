#!/usr/bin/env python3
"""
Quick backtest of FundingReversion strategy on collected data.

Loads parquet data, aggregates to 15-min bars, runs the strategy,
and reports Sharpe, edge, and net P&L after Hyperliquid maker fees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from cluster_pipeline.loader import load_parquet
from cluster_pipeline.preprocess import aggregate_bars
from strategies.funding_reversion import FundingReversion

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "features"

# Hyperliquid fee structure
MAKER_FEE = 0.0002   # 2 bps maker
TAKER_FEE = 0.00035  # 3.5 bps taker
COST_PER_TRADE = TAKER_FEE  # assume taker entry/exit


def backtest_signal(
    prices: np.ndarray,
    signal: np.ndarray,
    cost_per_trade: float = COST_PER_TRADE,
) -> dict:
    """
    Simple vectorized backtest for continuous signal in [-1, +1].

    Returns are: signal[t] * (price[t+1] - price[t]) / price[t]
    minus costs on position changes.
    """
    n = len(prices)
    returns = np.diff(prices) / prices[:-1]  # bar returns

    # Align signal with forward returns
    sig = signal[:-1]  # signal at bar t predicts return t→t+1

    # Strategy return per bar
    strat_returns = sig * returns

    # Transaction costs: proportional to position change
    position_changes = np.abs(np.diff(np.concatenate([[0], sig])))
    costs = position_changes * cost_per_trade

    net_returns = strat_returns - costs

    # Mask NaN bars
    valid = ~np.isnan(net_returns)
    net_returns_clean = net_returns[valid]

    if len(net_returns_clean) == 0:
        return {"error": "no valid returns"}

    # Metrics
    total_return = np.prod(1 + net_returns_clean) - 1
    mean_ret = np.mean(net_returns_clean)
    std_ret = np.std(net_returns_clean)
    sharpe = (mean_ret / std_ret * np.sqrt(96 * 365)) if std_ret > 0 else 0  # annualized, 96 bars/day

    # Drawdown
    cumulative = np.cumprod(1 + net_returns_clean)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_dd = np.max(drawdown)

    # Trade count (position changes)
    n_trades = int(np.sum(position_changes[valid] > 0.01))

    # Total costs
    total_costs = np.sum(costs[valid])

    # Gross return (before costs)
    gross_returns = strat_returns[valid]
    gross_total = np.prod(1 + gross_returns) - 1

    # Win rate (bars with positive return)
    active_bars = net_returns_clean[np.abs(sig[valid]) > 0.01]
    win_rate = np.mean(active_bars > 0) if len(active_bars) > 0 else 0

    # Time in market
    time_in_market = np.mean(np.abs(sig[valid]) > 0.01)

    return {
        "n_bars": int(valid.sum()),
        "total_return_pct": round(total_return * 100, 4),
        "gross_return_pct": round(gross_total * 100, 4),
        "total_costs_pct": round(total_costs * 100, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 3),
        "time_in_market": round(time_in_market, 3),
        "mean_bar_return_bps": round(mean_ret * 10000, 4),
        "std_bar_return_bps": round(std_ret * 10000, 4),
    }


def main():
    print("Loading data...")
    df = load_parquet(str(DATA_DIR))
    print(f"  {len(df):,} rows loaded")

    print("Aggregating to 15-min bars...")
    bars = aggregate_bars(df, timeframe="15min")
    print(f"  {len(bars)} bars")

    # Check available columns
    funding_cols = [c for c in bars.columns if "funding" in c.lower()]
    print(f"  Funding columns: {funding_cols}")

    if not funding_cols:
        print("ERROR: No funding rate columns found.")
        return

    # Get midprice
    price_col = None
    for col in ["raw_midprice_mean", "raw_midprice"]:
        if col in bars.columns:
            price_col = col
            break
    if price_col is None:
        print("ERROR: No price column found.")
        return

    prices = bars[price_col].values.astype(np.float64)
    print(f"  Price range: {np.nanmin(prices):.2f} - {np.nanmax(prices):.2f}")

    # Show funding rate stats
    for col in funding_cols:
        vals = bars[col].values
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            print(f"  {col}: mean={np.mean(valid):.6f}, std={np.std(valid):.6f}, "
                  f"range=[{np.min(valid):.4f}, {np.max(valid):.4f}]")

    print()
    print("=" * 65)
    print("  FUNDING REVERSION BACKTEST")
    print("=" * 65)

    # Test multiple parameter combinations
    configs = [
        {"zscore_entry": 1.5, "zscore_exit": 0.3, "max_position": 1.0, "label": "Aggressive (z>1.5)"},
        {"zscore_entry": 2.0, "zscore_exit": 0.5, "max_position": 1.0, "label": "Default (z>2.0)"},
        {"zscore_entry": 3.0, "zscore_exit": 1.0, "max_position": 1.0, "label": "Conservative (z>3.0)"},
        {"zscore_entry": 2.0, "zscore_exit": 0.5, "max_position": 0.5, "label": "Half-size (z>2.0, 0.5x)"},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        strat = FundingReversion(**cfg)
        features = strat.compute_features(bars)
        signal = strat.generate_signal(features)

        result = backtest_signal(prices, signal.values)

        print(f"\n  --- {label} ---")
        if "error" in result:
            print(f"  {result['error']}")
            continue

        print(f"  Bars:          {result['n_bars']}")
        print(f"  Trades:        {result['n_trades']}")
        print(f"  Time in mkt:   {result['time_in_market']:.1%}")
        print(f"  Gross return:  {result['gross_return_pct']:+.4f}%")
        print(f"  Net return:    {result['total_return_pct']:+.4f}%")
        print(f"  Costs:         {result['total_costs_pct']:.4f}%")
        print(f"  Sharpe:        {result['sharpe_ratio']:.3f}")
        print(f"  Max DD:        {result['max_drawdown_pct']:.4f}%")
        print(f"  Win rate:      {result['win_rate']:.1%}")
        print(f"  Mean ret/bar:  {result['mean_bar_return_bps']:+.4f} bps")

    print()
    print("=" * 65)
    print("  NOTE: This is ~1 day of data. Need 7+ days for significance.")
    print("=" * 65)


if __name__ == "__main__":
    main()

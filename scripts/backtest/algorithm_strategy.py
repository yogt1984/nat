"""
Algorithm-to-Backtest Strategy Adapter

Bridges the MicrostructureAlgorithm framework to the backtest engine.
Runs an algorithm on data, then backtests the resulting signal with
configurable entry/exit thresholds and cost model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.registry import get_algorithm, list_algorithms
from algorithms.runner import AlgorithmRunner


def run_algorithm_backtest(
    algorithm_name: str,
    data_dir: str = "data/features",
    symbol: str = "BTC",
    feature_col: str | None = None,
    entry_threshold: float = 0.3,
    exit_threshold: float = -0.1,
    max_memory_mb: float = 2000.0,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 3.5,
) -> dict:
    """Run a backtest using an algorithm's features as the trading signal.

    Args:
        algorithm_name: Name of registered algorithm.
        data_dir: Path to parquet data.
        symbol: Trading symbol.
        feature_col: Which algorithm feature to use as signal. If None, uses first.
        entry_threshold: Enter position when |signal| > threshold.
        exit_threshold: Exit when |signal| < threshold (or sign flip).
        max_memory_mb: Memory budget for data loading.
        maker_fee_bps: Maker fee in basis points.
        taker_fee_bps: Taker fee in basis points.

    Returns:
        Dict with backtest results (sharpe, max_dd, trades, returns, etc.)
    """
    alg = get_algorithm(algorithm_name)
    runner = AlgorithmRunner(alg)

    result = runner.run_on_parquet(
        data_dir, symbol,
        max_memory_mb=max_memory_mb,
        columns=["raw_spread", "ent_book_shape"],
    )

    features_df = result.features_df
    base_df = result.base_df
    midprices = base_df["raw_midprice"].values.astype(np.float64)

    # Select signal feature
    if feature_col is None:
        feature_col = features_df.columns[0]
    if feature_col not in features_df.columns:
        raise ValueError(f"Feature '{feature_col}' not in {list(features_df.columns)}")

    signal = features_df[feature_col].values.astype(np.float64)

    # Forward returns
    n = len(midprices)
    fwd_ret = np.zeros(n)
    fwd_ret[:-1] = (midprices[1:] - midprices[:-1]) / midprices[:-1]

    # Generate positions from signal with threshold
    positions = np.zeros(n)
    in_position = False
    position_side = 0.0

    for i in range(n):
        if not np.isfinite(signal[i]):
            positions[i] = 0.0
            continue

        abs_sig = abs(signal[i])

        if not in_position:
            if abs_sig > entry_threshold:
                position_side = np.sign(signal[i])
                in_position = True
                positions[i] = position_side
            else:
                positions[i] = 0.0
        else:
            # Exit on sign flip or signal below exit threshold
            if (np.sign(signal[i]) != position_side and abs_sig > entry_threshold * 0.5):
                in_position = False
                positions[i] = 0.0
            elif abs_sig < abs(exit_threshold):
                in_position = False
                positions[i] = 0.0
            else:
                positions[i] = position_side

    # PnL calculation
    strat_returns = positions[:-1] * fwd_ret[:-1]

    # Transaction costs
    position_changes = np.abs(np.diff(positions))
    cost_per_trade = (maker_fee_bps + taker_fee_bps) / 2 / 10000
    costs = position_changes * cost_per_trade
    net_returns = strat_returns - costs

    # Metrics
    valid = np.isfinite(net_returns)
    net_returns_clean = net_returns[valid]

    if len(net_returns_clean) < 100:
        return {
            "algorithm": algorithm_name,
            "feature": feature_col,
            "symbol": symbol,
            "error": "Insufficient data for backtest",
        }

    n_trades = int(np.sum(position_changes > 0))
    total_return = float(np.sum(net_returns_clean))
    mean_return = float(np.mean(net_returns_clean))
    std_return = float(np.std(net_returns_clean)) + 1e-20

    # Sharpe (annualized at 10Hz)
    sharpe = mean_return / std_return * np.sqrt(86400 * 10)

    # Max drawdown
    equity = np.cumsum(net_returns_clean)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Win rate
    winning = net_returns_clean[net_returns_clean > 0]
    win_rate = len(winning) / max(n_trades, 1)

    # Profit factor
    gross_profit = float(np.sum(net_returns_clean[net_returns_clean > 0]))
    gross_loss = float(abs(np.sum(net_returns_clean[net_returns_clean < 0]))) + 1e-20
    profit_factor = gross_profit / gross_loss

    return {
        "algorithm": algorithm_name,
        "feature": feature_col,
        "symbol": symbol,
        "n_ticks": n,
        "n_trades": n_trades,
        "total_return_bps": round(total_return * 10000, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_bps": round(max_dd * 10000, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "avg_return_per_trade_bps": round(total_return / max(n_trades, 1) * 10000, 4),
        "cost_bps": round(float(np.sum(costs)) * 10000, 2),
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "maker_fee_bps": maker_fee_bps,
        "taker_fee_bps": taker_fee_bps,
    }

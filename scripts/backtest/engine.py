"""
Backtest Engine for NAT

Core simulation engine that iterates through data and tracks positions.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from .strategy import Strategy
from .costs import CostModel


@dataclass
class Trade:
    """
    Record of a single trade.

    Attributes
    ----------
    entry_idx : int
        Index in dataframe where position was opened
    entry_price : float
        Effective entry price (after costs)
    entry_time : int
        Timestamp of entry
    direction : str
        "long" or "short"
    exit_idx : int, optional
        Index where position was closed
    exit_price : float, optional
        Effective exit price (after costs)
    exit_time : int, optional
        Timestamp of exit
    exit_reason : str, optional
        Why the trade was closed: "signal", "stop", "target", "timeout", "end"
    pnl_pct : float, optional
        Profit/loss as percentage
    raw_entry_price : float, optional
        Entry price before costs (for analysis)
    raw_exit_price : float, optional
        Exit price before costs (for analysis)
    """

    entry_idx: int
    entry_price: float
    entry_time: int
    direction: Literal["long", "short"] = "long"
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    raw_entry_price: Optional[float] = None
    raw_exit_price: Optional[float] = None

    @property
    def holding_bars(self) -> int:
        """Number of bars the position was held."""
        if self.exit_idx is None:
            return 0
        return self.exit_idx - self.entry_idx

    @property
    def is_winner(self) -> bool:
        """Whether trade was profitable."""
        return self.pnl_pct is not None and self.pnl_pct > 0

    @property
    def is_closed(self) -> bool:
        """Whether trade has been closed."""
        return self.exit_idx is not None


@dataclass
class BacktestResult:
    """
    Complete backtest results.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy tested
    trades : List[Trade]
        All trades executed
    equity_curve : List[float]
        Equity value at each bar
    total_return_pct : float
        Total return as percentage
    sharpe_ratio : float
        Risk-adjusted return (annualized approximation)
    max_drawdown_pct : float
        Maximum peak-to-trough decline
    win_rate : float
        Fraction of winning trades
    profit_factor : float
        Gross profit / gross loss
    avg_trade_pnl_pct : float
        Average P&L per trade
    total_trades : int
        Number of trades executed
    avg_holding_bars : float
        Average bars per trade
    cost_model : CostModel
        Cost model used
    total_costs_pct : float
        Total transaction costs as percentage of initial capital
    """

    strategy_name: str
    trades: List[Trade]
    equity_curve: List[float]
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl_pct: float
    total_trades: int
    avg_holding_bars: float
    cost_model: CostModel = field(default_factory=CostModel)
    total_costs_pct: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'='*60}",
            f"BACKTEST RESULTS: {self.strategy_name}",
            f"{'='*60}",
            f"Total Return:     {self.total_return_pct:+.2f}%",
            f"Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"Max Drawdown:     {self.max_drawdown_pct:.2f}%",
            f"Win Rate:         {self.win_rate:.1%}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            f"Total Trades:     {self.total_trades}",
            f"Avg Trade P&L:    {self.avg_trade_pnl_pct:+.2f}%",
            f"Avg Holding:      {self.avg_holding_bars:.0f} bars",
            f"Total Costs:      {self.total_costs_pct:.2f}%",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def exit_reason_breakdown(self) -> dict:
        """Count trades by exit reason."""
        reasons = {}
        for t in self.trades:
            reason = t.exit_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        return dict(sorted(reasons.items(), key=lambda x: -x[1]))


def run_backtest(
    df: pl.DataFrame,
    strategy: Strategy,
    cost_model: CostModel,
    initial_capital: float = 10000.0,
    allow_multiple_positions: bool = False,
) -> BacktestResult:
    """
    Run backtest on feature dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Feature data with timestamp_ms and raw_midprice columns
    strategy : Strategy
        Strategy to test
    cost_model : CostModel
        Transaction cost model
    initial_capital : float
        Starting capital
    allow_multiple_positions : bool
        If False (default), only one position at a time

    Returns
    -------
    BacktestResult
        Complete backtest results

    Notes
    -----
    - Assumes df is sorted by timestamp
    - Uses vectorized signal evaluation for speed
    - Iterates bar-by-bar for position management
    """
    # Validate inputs
    if len(df) == 0:
        raise ValueError("Empty dataframe")

    if "raw_midprice" not in df.columns:
        raise ValueError("Missing raw_midprice column")

    if "timestamp_ms" not in df.columns:
        raise ValueError("Missing timestamp_ms column")

    # Extract numpy arrays for fast iteration (make writable copy)
    prices = df["raw_midprice"].to_numpy().copy()
    timestamps = df["timestamp_ms"].to_numpy().copy()
    n = len(prices)

    # Validate prices
    if np.any(np.isnan(prices)) or np.any(prices <= 0):
        # Clean up invalid prices
        valid_mask = ~np.isnan(prices) & (prices > 0)
        if not np.any(valid_mask):
            raise ValueError("No valid prices in data")
        # For invalid prices, use last valid price
        last_valid = prices[valid_mask][0]
        for i in range(n):
            if np.isnan(prices[i]) or prices[i] <= 0:
                prices[i] = last_valid
            else:
                last_valid = prices[i]

    # Evaluate conditions once (vectorized)
    try:
        entry_signals = strategy.entry_condition(df).to_numpy()
    except Exception as e:
        raise ValueError(f"Error evaluating entry condition: {e}")

    try:
        exit_signals = strategy.exit_condition(df).to_numpy()
    except Exception as e:
        raise ValueError(f"Error evaluating exit condition: {e}")

    # Convert to boolean, handle nulls
    entry_signals = np.array(entry_signals, dtype=bool)
    exit_signals = np.array(exit_signals, dtype=bool)

    # State tracking
    trades: List[Trade] = []
    equity_curve = [initial_capital]
    capital = initial_capital
    position: Optional[Trade] = None
    total_costs = 0.0

    for i in range(n):
        current_price = prices[i]

        if position is None:
            # Not in position - check for entry
            if entry_signals[i]:
                entry_price_eff = cost_model.apply_entry_cost(
                    current_price, strategy.direction
                )
                position = Trade(
                    entry_idx=i,
                    entry_price=entry_price_eff,
                    entry_time=timestamps[i],
                    direction=strategy.direction,
                    raw_entry_price=current_price,
                )
                # Track entry cost
                entry_cost = abs(entry_price_eff - current_price)
                total_costs += entry_cost / current_price * 100  # As % of price

        else:
            # In position - check for exit
            holding_bars = i - position.entry_idx

            # Calculate current P&L (without exit costs yet)
            if strategy.direction == "long":
                current_pnl_pct = (current_price / position.raw_entry_price - 1) * 100
            else:
                current_pnl_pct = (1 - current_price / position.raw_entry_price) * 100

            # Subtract entry cost from P&L calculation for stop/target checks
            entry_cost_pct = cost_model.one_way_cost_fraction * 100
            adjusted_pnl = current_pnl_pct - entry_cost_pct

            exit_reason = None

            # Check exit conditions in priority order
            if adjusted_pnl <= -strategy.stop_loss_pct:
                exit_reason = "stop"
            elif adjusted_pnl >= strategy.take_profit_pct:
                exit_reason = "target"
            elif holding_bars >= strategy.max_holding_bars:
                exit_reason = "timeout"
            elif exit_signals[i]:
                exit_reason = "signal"

            if exit_reason:
                exit_price_eff = cost_model.apply_exit_cost(
                    current_price, strategy.direction
                )

                # Final P&L including all costs
                pnl_pct = cost_model.compute_pnl(
                    position.raw_entry_price,
                    current_price,
                    strategy.direction,
                    include_costs=True,
                )

                position.exit_idx = i
                position.exit_price = exit_price_eff
                position.exit_time = timestamps[i]
                position.exit_reason = exit_reason
                position.pnl_pct = pnl_pct
                position.raw_exit_price = current_price

                # Track exit cost
                exit_cost = abs(exit_price_eff - current_price)
                total_costs += exit_cost / current_price * 100

                # Update capital
                capital *= (1 + pnl_pct / 100)
                trades.append(position)
                position = None

        equity_curve.append(capital)

    # Close any open position at end of data
    if position is not None:
        exit_price_eff = cost_model.apply_exit_cost(prices[-1], strategy.direction)
        pnl_pct = cost_model.compute_pnl(
            position.raw_entry_price,
            prices[-1],
            strategy.direction,
            include_costs=True,
        )

        position.exit_idx = n - 1
        position.exit_price = exit_price_eff
        position.exit_time = timestamps[-1]
        position.exit_reason = "end"
        position.pnl_pct = pnl_pct
        position.raw_exit_price = prices[-1]

        capital *= (1 + pnl_pct / 100)
        trades.append(position)
        equity_curve.append(capital)

    # Compute metrics
    return _compute_metrics(
        strategy.name, trades, equity_curve, initial_capital, cost_model, total_costs
    )


def _compute_metrics(
    name: str,
    trades: List[Trade],
    equity_curve: List[float],
    initial_capital: float,
    cost_model: CostModel,
    total_costs: float,
) -> BacktestResult:
    """Compute backtest performance metrics."""

    # Handle empty trades
    if not trades:
        return BacktestResult(
            strategy_name=name,
            trades=[],
            equity_curve=equity_curve,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl_pct=0.0,
            total_trades=0,
            avg_holding_bars=0.0,
            cost_model=cost_model,
            total_costs_pct=0.0,
        )

    pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]

    if not pnls:
        return BacktestResult(
            strategy_name=name,
            trades=trades,
            equity_curve=equity_curve,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl_pct=0.0,
            total_trades=len(trades),
            avg_holding_bars=0.0,
            cost_model=cost_model,
            total_costs_pct=total_costs,
        )

    # Total return
    total_return_pct = (equity_curve[-1] / initial_capital - 1) * 100

    # Win rate
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001  # Avoid division by zero
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_drawdown_pct = abs(drawdowns.min())

    # Sharpe ratio (based on trade returns)
    # This is a simplified calculation - proper Sharpe needs time normalization
    pnl_array = np.array(pnls)
    if len(pnl_array) > 1 and pnl_array.std() > 0:
        # Approximate annualization factor based on trade frequency
        # Assume average of 1 trade per day, 252 trading days
        trades_per_year = 252
        sharpe_ratio = (pnl_array.mean() / pnl_array.std()) * np.sqrt(
            min(len(pnls), trades_per_year)
        )
    else:
        sharpe_ratio = 0.0

    # Average holding time
    holding_bars = [t.holding_bars for t in trades if t.exit_idx is not None]
    avg_holding_bars = np.mean(holding_bars) if holding_bars else 0

    return BacktestResult(
        strategy_name=name,
        trades=trades,
        equity_curve=equity_curve,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl_pct=float(np.mean(pnls)),
        total_trades=len(trades),
        avg_holding_bars=avg_holding_bars,
        cost_model=cost_model,
        total_costs_pct=total_costs,
    )

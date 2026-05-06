"""
EAMM Module 6: Stateful Backtester with Inventory Management

Simulates EAMM as a real market maker with inventory state, Avellaneda-Stoikov
spread adjustment, position limits, and end-of-period liquidation.

Reference: EAMM_SPEC.md §1.9
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    """A single fill event."""
    timestamp_ns: int
    side: str  # "bid" or "ask"
    price: float
    quantity: float
    inventory_after: float


@dataclass
class BacktestResult:
    """Full backtest output."""
    equity_curve: np.ndarray  # cumulative PnL at each timestamp (N,)
    inventory_curve: np.ndarray  # inventory at each timestamp (N,)
    trades: List[Trade]
    total_pnl: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    n_round_trips: int
    avg_holding_bars: float
    fill_rate_bid: float
    fill_rate_ask: float
    avg_spread_quoted: float
    adverse_selection_rate: float  # fraction of single-side fills that lost money


def run_backtest(
    midprices: np.ndarray,
    timestamps: np.ndarray,
    predicted_spreads_bps: np.ndarray,
    volatility: np.ndarray,
    gamma: float = 0.1,
    q_max: float = 1.0,
    horizon: int = 3000,
    maker_fee_bps: float = 0.0,
) -> BacktestResult:
    """Run stateful MM backtest.

    Parameters
    ----------
    midprices : np.ndarray, shape (N,)
        Midprice series.
    timestamps : np.ndarray, shape (N,)
        Timestamps.
    predicted_spreads_bps : np.ndarray, shape (N,)
        Model-predicted optimal half-spread at each bar.
    volatility : np.ndarray, shape (N,)
        vol_returns_1m at each bar (for inventory penalty).
    gamma : float
        Risk aversion parameter.
    q_max : float
        Maximum absolute inventory (in base units, e.g. BTC).
    horizon : int
        Quote lifetime in bars. After this, quote expires.
    maker_fee_bps : float
        Per-side maker fee.

    Returns
    -------
    BacktestResult
    """
    N = len(midprices)
    equity = np.zeros(N)
    inventory = np.zeros(N)
    trades: List[Trade] = []

    q = 0.0  # current inventory
    cumulative_pnl = 0.0
    n_bid_fills = 0
    n_ask_fills = 0
    n_bid_quotes = 0
    n_ask_quotes = 0
    total_spread_quoted = 0.0
    n_quotes = 0
    adverse_losses = 0
    single_side_fills = 0

    # Step size for quote cycles (don't quote every 100ms — quote every horizon bars)
    step = max(1, horizon // 10)  # re-quote 10x per horizon

    for t in range(0, N - horizon, step):
        mid = midprices[t]
        sigma = volatility[t] if volatility[t] > 0 else 1e-8

        # Avellaneda-Stoikov adjustment
        base_spread_bps = predicted_spreads_bps[t]
        inventory_penalty_bps = gamma * q * q * sigma * sigma * 10000 * 100
        skew_bps = -gamma * q * sigma * sigma * 10000 * 100

        adjusted_spread_bps = base_spread_bps + abs(inventory_penalty_bps)
        delta_frac = adjusted_spread_bps / 10000.0
        skew_frac = skew_bps / 10000.0

        # Quote prices
        bid_price = mid * (1.0 - delta_frac + skew_frac)
        ask_price = mid * (1.0 + delta_frac + skew_frac)

        # Position limits: only quote reducing side if at max
        can_bid = abs(q + 1.0) <= q_max + 0.01  # buying increases inventory
        can_ask = abs(q - 1.0) <= q_max + 0.01  # selling decreases inventory

        # Check fills over horizon window
        window = midprices[t + 1: t + horizon + 1]
        if len(window) == 0:
            break

        future_min = np.min(window)
        future_max = np.max(window)
        future_mid = midprices[min(t + horizon, N - 1)]

        bid_filled = can_bid and (future_min <= bid_price)
        ask_filled = can_ask and (future_max >= ask_price)

        n_quotes += 1
        total_spread_quoted += adjusted_spread_bps
        if can_bid:
            n_bid_quotes += 1
        if can_ask:
            n_ask_quotes += 1

        # Compute PnL
        fee = maker_fee_bps / 10000.0 * mid

        if bid_filled and ask_filled:
            # Round trip — captured spread
            pnl = (ask_price - bid_price) - 2 * fee
            q_change = 0.0  # net flat
            n_bid_fills += 1
            n_ask_fills += 1
        elif bid_filled:
            # Long position opened
            pnl = (future_mid - bid_price) - fee
            q_change = 1.0
            n_bid_fills += 1
            single_side_fills += 1
            if pnl < 0:
                adverse_losses += 1
        elif ask_filled:
            # Short position opened
            pnl = (ask_price - future_mid) - fee
            q_change = -1.0
            n_ask_fills += 1
            single_side_fills += 1
            if pnl < 0:
                adverse_losses += 1
        else:
            pnl = 0.0
            q_change = 0.0

        cumulative_pnl += pnl
        q += q_change
        # Clamp inventory
        q = np.clip(q, -q_max, q_max)

        # Record trade
        if bid_filled or ask_filled:
            trades.append(Trade(
                timestamp_ns=int(timestamps[t]),
                side="both" if (bid_filled and ask_filled) else ("bid" if bid_filled else "ask"),
                price=mid,
                quantity=1.0,
                inventory_after=q,
            ))

        # Update curves at this step
        for s in range(t, min(t + step, N)):
            equity[s] = cumulative_pnl
            inventory[s] = q

    # End-of-period liquidation
    if abs(q) > 0.01:
        # Force close at last midprice (market order — pay taker fee)
        last_mid = midprices[N - 1]
        taker_fee = 3.5 / 10000.0 * last_mid
        liquidation_pnl = -abs(q) * taker_fee  # just the fee cost
        cumulative_pnl += liquidation_pnl
        q = 0.0

    # Fill remaining equity curve
    for s in range(N):
        if equity[s] == 0.0 and s > 0:
            equity[s] = equity[s - 1]
    equity[-1] = cumulative_pnl
    inventory[-1] = q

    # Metrics
    pnl_per_trade = np.diff(equity[equity != 0]) if np.any(equity != 0) else np.array([0.0])
    sharpe = _sharpe(pnl_per_trade)
    max_dd = _max_drawdown_curve(equity)
    fill_rate_bid = n_bid_fills / n_bid_quotes if n_bid_quotes > 0 else 0.0
    fill_rate_ask = n_ask_fills / n_ask_quotes if n_ask_quotes > 0 else 0.0
    adverse_rate = adverse_losses / single_side_fills if single_side_fills > 0 else 0.0
    n_round_trips = min(n_bid_fills, n_ask_fills)

    return BacktestResult(
        equity_curve=equity,
        inventory_curve=inventory,
        trades=trades,
        total_pnl=cumulative_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=len(trades),
        n_round_trips=n_round_trips,
        avg_holding_bars=float(horizon),
        fill_rate_bid=fill_rate_bid,
        fill_rate_ask=fill_rate_ask,
        avg_spread_quoted=total_spread_quoted / n_quotes if n_quotes > 0 else 0.0,
        adverse_selection_rate=adverse_rate,
    )


def print_backtest_report(result: BacktestResult):
    """Print backtest summary."""
    print("\n" + "=" * 70)
    print("EAMM BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Total PnL:           ${result.total_pnl:,.2f}")
    print(f"  Sharpe ratio:        {result.sharpe:.2f}")
    print(f"  Max drawdown:        ${result.max_drawdown:,.2f}")
    print(f"  Trades:              {result.n_trades}")
    print(f"  Round trips:         {result.n_round_trips}")
    print(f"  Fill rate (bid):     {result.fill_rate_bid:.1%}")
    print(f"  Fill rate (ask):     {result.fill_rate_ask:.1%}")
    print(f"  Avg spread quoted:   {result.avg_spread_quoted:.2f} bps")
    print(f"  Adverse selection:   {result.adverse_selection_rate:.1%}")
    print(f"  Final inventory:     {result.inventory_curve[-1]:.4f}")


def _sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    std = np.std(pnl)
    if std < 1e-12:
        return 0.0
    return float(np.mean(pnl) / std * np.sqrt(len(pnl)))


def _max_drawdown_curve(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if len(dd) > 0 else 0.0

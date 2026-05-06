"""
EAMM Module 1: Market Maker Fill Simulator

Simulates market maker quoting at various half-spread levels against
historical trade data. For each timestamp and candidate spread, determines
whether bid/ask quotes would have been filled and computes realized PnL.

Reference: EAMM_SPEC.md §1.2–§1.3
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import List, Optional


# Default candidate half-spreads in basis points
DEFAULT_SPREAD_LEVELS_BPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]


@dataclass
class SimulationResult:
    """Result of MM simulation across multiple spread levels.

    Attributes
    ----------
    timestamps : np.ndarray
        Timestamp for each row (N,)
    spread_levels_bps : list[float]
        The K spread levels tested
    pnl : np.ndarray
        PnL matrix (N, K) — realized PnL per row per spread level
    fill_bid : np.ndarray
        Bid fill indicator (N, K) — 1 if bid was filled
    fill_ask : np.ndarray
        Ask fill indicator (N, K) — 1 if ask was filled
    fill_round_trip : np.ndarray
        Round-trip fill indicator (N, K) — 1 if both sides filled
    midprice : np.ndarray
        Midprice at each timestamp (N,)
    midprice_at_horizon : np.ndarray
        Midprice h rows ahead (N,)
    horizon : int
        Horizon used (in rows)
    """
    timestamps: np.ndarray
    spread_levels_bps: List[float]
    pnl: np.ndarray
    fill_bid: np.ndarray
    fill_ask: np.ndarray
    fill_round_trip: np.ndarray
    midprice: np.ndarray
    midprice_at_horizon: np.ndarray
    horizon: int


def simulate_mm(
    df: pl.DataFrame,
    spread_levels_bps: Optional[List[float]] = None,
    horizon: int = 3000,
    price_col: str = "raw_midprice",
    maker_fee_bps: float = 0.0,
) -> SimulationResult:
    """Simulate a market maker quoting at various spread levels.

    For each row t and spread delta_k (in bps):
      - P_bid(t) = P_mid(t) * (1 - delta_k / 10000)
      - P_ask(t) = P_mid(t) * (1 + delta_k / 10000)
      - Look ahead h rows for fills using min/max of midprice as proxy
      - Compute PnL based on fill outcomes

    Parameters
    ----------
    df : pl.DataFrame
        Must contain `price_col` and `timestamp_ns` columns, sorted by time.
    spread_levels_bps : list of float
        Candidate half-spreads in basis points.
    horizon : int
        Number of rows to look ahead for fills (1 row = 100ms).
    price_col : str
        Column name for midprice.
    maker_fee_bps : float
        Per-side maker fee in bps (0 on Hyperliquid).

    Returns
    -------
    SimulationResult
    """
    if spread_levels_bps is None:
        spread_levels_bps = DEFAULT_SPREAD_LEVELS_BPS

    prices = df[price_col].to_numpy().astype(np.float64)
    timestamps = df["timestamp_ns"].to_numpy()
    N = len(prices)
    K = len(spread_levels_bps)

    # Precompute rolling min and max over the horizon window
    # For each t, we need min(prices[t+1:t+h+1]) and max(prices[t+1:t+h+1])
    # Use a forward-looking rolling computation
    future_min = np.full(N, np.nan)
    future_max = np.full(N, np.nan)
    future_price = np.full(N, np.nan)

    # Compute using cumulative min/max from the end
    # More efficient: slide a window
    valid_end = N - horizon
    if valid_end <= 0:
        raise ValueError(
            f"Data has {N} rows but horizon is {horizon}. "
            f"Need at least {horizon + 1} rows."
        )

    for t in range(valid_end):
        window = prices[t + 1 : t + horizon + 1]
        future_min[t] = np.min(window)
        future_max[t] = np.max(window)
        future_price[t] = prices[t + horizon]

    # Vectorized version for speed: use stride tricks or just numpy
    # The loop above is O(N*h) which is slow for large data.
    # Replace with efficient rolling min/max:
    future_min, future_max, future_price = _rolling_min_max(prices, horizon)

    # Allocate output matrices
    pnl = np.zeros((N, K), dtype=np.float64)
    fill_bid = np.zeros((N, K), dtype=np.float64)
    fill_ask = np.zeros((N, K), dtype=np.float64)
    fill_rt = np.zeros((N, K), dtype=np.float64)

    maker_fee_frac = maker_fee_bps / 10000.0

    for k, delta_bps in enumerate(spread_levels_bps):
        delta_frac = delta_bps / 10000.0

        bid_prices = prices * (1.0 - delta_frac)
        ask_prices = prices * (1.0 + delta_frac)

        # Fill conditions: future price touched or crossed the quote
        fb = (future_min <= bid_prices).astype(np.float64)
        fa = (future_max >= ask_prices).astype(np.float64)
        frt = fb * fa

        # PnL calculation
        # Round trip: captured 2 * delta (spread)
        pnl_rt = frt * 2.0 * delta_frac * prices

        # Bid only: bought at bid, mark-to-market at future midprice
        pnl_bid_only = fb * (1.0 - fa) * (future_price - bid_prices)

        # Ask only: sold at ask, mark-to-market at future midprice
        pnl_ask_only = (1.0 - fb) * fa * (ask_prices - future_price)

        # Maker fees (per side)
        fee_cost = (fb + fa) * maker_fee_frac * prices

        row_pnl = pnl_rt + pnl_bid_only + pnl_ask_only - fee_cost

        # NaN out rows beyond valid range
        row_pnl[valid_end:] = np.nan
        fb[valid_end:] = np.nan
        fa[valid_end:] = np.nan
        frt[valid_end:] = np.nan

        pnl[:, k] = row_pnl
        fill_bid[:, k] = fb
        fill_ask[:, k] = fa
        fill_rt[:, k] = frt

    return SimulationResult(
        timestamps=timestamps,
        spread_levels_bps=spread_levels_bps,
        pnl=pnl,
        fill_bid=fill_bid,
        fill_ask=fill_ask,
        fill_round_trip=fill_rt,
        midprice=prices,
        midprice_at_horizon=future_price,
        horizon=horizon,
    )


def _rolling_min_max(
    prices: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward-looking rolling min and max efficiently.

    For each index t, computes:
      future_min[t] = min(prices[t+1 : t+horizon+1])
      future_max[t] = max(prices[t+1 : t+horizon+1])
      future_price[t] = prices[t+horizon]

    Uses backward scan for O(N) computation.

    Returns (future_min, future_max, future_price), each of shape (N,).
    Entries where the window extends beyond the array are set to NaN.
    """
    N = len(prices)
    future_min = np.full(N, np.nan)
    future_max = np.full(N, np.nan)
    future_price = np.full(N, np.nan)

    valid_end = N - horizon
    if valid_end <= 0:
        return future_min, future_max, future_price

    # Build using a deque-based approach for O(N), but for simplicity
    # and correctness, use numpy stride-based approach:
    # Create a 2D view of rolling windows, then take min/max along axis 1.
    # This is O(N*h) in memory but vectorized and fast for h < 50000.
    if horizon <= 50000:
        # Strided approach
        from numpy.lib.stride_tricks import sliding_window_view
        # We want windows starting at t+1, of length horizon
        windows = sliding_window_view(prices[1:], horizon)
        # windows[t] = prices[t+1 : t+1+horizon] for t in 0..N-1-horizon
        n_windows = windows.shape[0]  # = N - horizon
        future_min[:n_windows] = np.min(windows, axis=1)
        future_max[:n_windows] = np.max(windows, axis=1)
    else:
        # Fallback for very large horizons
        for t in range(valid_end):
            w = prices[t + 1 : t + horizon + 1]
            future_min[t] = np.min(w)
            future_max[t] = np.max(w)

    future_price[:valid_end] = prices[horizon:N]

    return future_min, future_max, future_price


def pnl_to_bps(result: SimulationResult) -> np.ndarray:
    """Convert PnL from price units to basis points relative to midprice.

    Returns (N, K) array in bps.
    """
    # Avoid division by zero
    mid = result.midprice.copy()
    mid[mid == 0] = np.nan
    return (result.pnl / mid[:, np.newaxis]) * 10000.0

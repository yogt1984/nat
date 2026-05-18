"""
Phase 2: Maker Fill Simulator

Simulates passive limit order fills from historical tick data.
Quantifies adverse selection by measuring post-fill price drift.

Fill model (conservative):
- Post limit order at best bid/ask (mid ∓ spread/2)
- Fill when midprice crosses through order price (adverse move causes fill)
- Latency: configurable ticks between signal and order placement
- No partial fills, no queue priority modeling
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FillEvent:
    """A single simulated limit order fill."""
    signal_tick: int      # when signal triggered
    fill_tick: int        # when order was filled
    fill_price: float     # execution price
    side: str             # "buy" or "sell"
    signal_strength: float  # imbalance value at signal time
    regime_value: float   # ent_book_shape at signal time
    midprice_at_fill: float  # midprice when filled


class MakerFillSimulator:
    """Simulate passive limit order fills from historical tick data.

    Strategy logic:
    - Buy when imbalance > threshold (bullish)
    - Sell when imbalance < -threshold (bearish)
    - Post at best bid (buy) or best ask (sell)
    - Fill when mid crosses order level
    """

    def __init__(
        self,
        entry_threshold: float = 0.3,
        regime_percentile: float = 30.0,
        latency_ticks: int = 2,
        order_timeout_ticks: int = 50,
        min_ticks_between_signals: int = 10,
    ):
        self.entry_threshold = entry_threshold
        self.regime_percentile = regime_percentile
        self.latency_ticks = latency_ticks
        self.order_timeout_ticks = order_timeout_ticks
        self.min_ticks_between_signals = min_ticks_between_signals

    def simulate(
        self,
        midprices: np.ndarray,
        spreads: np.ndarray,
        signal: np.ndarray,
        regime_vals: np.ndarray,
    ) -> list[FillEvent]:
        """Run fill simulation over historical data.

        Args:
            midprices: raw midprice series
            spreads: raw spread series (in price units)
            signal: imbalance signal (positive = bullish)
            regime_vals: regime feature values (lower = more structured)

        Returns:
            List of FillEvent objects
        """
        n = len(midprices)
        regime_thresh = np.nanpercentile(regime_vals, self.regime_percentile)

        fills = []
        last_signal_tick = -self.min_ticks_between_signals

        i = 0
        while i < n - self.order_timeout_ticks - self.latency_ticks:
            # Check signal + regime gate
            if not np.isfinite(signal[i]) or not np.isfinite(regime_vals[i]):
                i += 1
                continue

            if regime_vals[i] >= regime_thresh:
                i += 1
                continue

            if i - last_signal_tick < self.min_ticks_between_signals:
                i += 1
                continue

            # Determine side
            if signal[i] > self.entry_threshold:
                side = "buy"
            elif signal[i] < -self.entry_threshold:
                side = "sell"
            else:
                i += 1
                continue

            last_signal_tick = i

            # Order placed after latency
            order_tick = i + self.latency_ticks
            if order_tick >= n:
                break

            spread = spreads[order_tick]
            if not np.isfinite(spread) or spread <= 0:
                i += 1
                continue

            # Order price at best bid/ask
            if side == "buy":
                order_price = midprices[order_tick] - spread / 2
            else:
                order_price = midprices[order_tick] + spread / 2

            # Check for fill: mid crosses order level
            filled = False
            for j in range(order_tick + 1,
                           min(order_tick + self.order_timeout_ticks, n)):
                if side == "buy" and midprices[j] <= order_price:
                    fills.append(FillEvent(
                        signal_tick=i,
                        fill_tick=j,
                        fill_price=order_price,
                        side=side,
                        signal_strength=float(signal[i]),
                        regime_value=float(regime_vals[i]),
                        midprice_at_fill=float(midprices[j]),
                    ))
                    filled = True
                    break
                elif side == "sell" and midprices[j] >= order_price:
                    fills.append(FillEvent(
                        signal_tick=i,
                        fill_tick=j,
                        fill_price=order_price,
                        side=side,
                        signal_strength=float(signal[i]),
                        regime_value=float(regime_vals[i]),
                        midprice_at_fill=float(midprices[j]),
                    ))
                    filled = True
                    break

            # Skip past the fill/timeout window
            if filled:
                i = j + 1
            else:
                i = order_tick + self.order_timeout_ticks
            continue

        return fills

    def compute_post_fill_drift(
        self,
        fills: list[FillEvent],
        midprices: np.ndarray,
        horizons: list[int] = None,
    ) -> dict:
        """Compute post-fill price drift at multiple horizons.

        Drift = (mid[fill+h] - fill_price) / fill_price * direction
        Positive drift = price moved in our favor after fill.

        Returns dict with drift arrays per horizon.
        """
        if horizons is None:
            horizons = [1, 5, 10, 50, 100]

        n = len(midprices)
        result = {h: [] for h in horizons}

        for fill in fills:
            direction = 1.0 if fill.side == "buy" else -1.0
            for h in horizons:
                t = fill.fill_tick + h
                if t < n and np.isfinite(midprices[t]):
                    drift_bps = (midprices[t] - fill.fill_price) / fill.fill_price * direction * 10000
                    result[h].append(drift_bps)
                else:
                    result[h].append(np.nan)

        return {h: np.array(v) for h, v in result.items()}

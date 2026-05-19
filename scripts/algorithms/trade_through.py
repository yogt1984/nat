"""
Trade-Through Probability Algorithm
====================================

Mathematical Framework
----------------------
A "trade-through" event occurs when an incoming market order consumes the
entire quantity posted at the best price level, forcing the mid-price to move
by one tick. This algorithm produces a real-valued proxy for the probability of
that event occurring.

Queue Depletion Model
---------------------
Let D^b(t) = total bid depth at the best 5 levels (``raw_bid_depth_5``) and
D^a(t) = total ask depth.  Over a rolling window W_vol, the mean traded volume
per tick is:

  V̄(t) = (1/W_vol) · Σ_{s=t-W_vol+1}^{t} V(s)

where V(s) = ``flow_volume_1s``.  The directional split uses the rolling mean
aggressor ratio ā(t) = mean(``flow_aggressor_ratio_5s``) over window W_agg:

  V̄^sell(t) = V̄(t) · (1 - ā(t))    (estimated sell-side flow)
  V̄^buy(t)  = V̄(t) · ā(t)           (estimated buy-side flow)

The queue-depletion probability is approximated by the ratio of expected
directional flow to the standing depth, clamped to [0, 1]:

  P^b(t) = min( V̄^sell(t) / D^b(t),  1 )
  P^a(t) = min( V̄^buy(t)  / D^a(t),  1 )

This is a reduced-form of the Cont & de Larrard (2013) result that in a
Markovian LOB, the probability of an upward tick equals q_a / (q_b + q_a),
adapted here to use flow rates rather than instantaneous queue sizes.

Numerical Stability
-------------------
A floor of 1e-12 is added to depth in the denominator to avoid division by
zero when the order book is momentarily empty.

Directional Imbalance
---------------------
  IMB(t) = P^a(t) - P^b(t)  ∈ [-1, 1]

Positive: buy-side trade-through more likely → price more likely to tick up.
Negative: sell-side trade-through more likely → price more likely to tick down.

Parameters
----------
  volume_window    (W_vol) : rolling window for mean volume estimate (ticks, default 10)
  aggressor_window (W_agg) : rolling window for mean aggressor ratio (ticks, default 50)

Output Ranges
-------------
  alg_trade_through_bid       : [0, 1]
  alg_trade_through_ask       : [0, 1]
  alg_trade_through_imbalance : [-1, 1]

Complexity: O(W_vol + W_agg) per tick for the rolling-window buffers.

Reference
---------
  Cont & de Larrard (2013) — "Price dynamics in a Markovian limit order market",
    SIAM Journal on Financial Mathematics 4(1), 1–25.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class TradeThrough(MicrostructureAlgorithm):
    """Queue depletion probability from volume vs depth."""

    def __init__(self, volume_window: int = 10, aggressor_window: int = 50):
        self._vol_window = volume_window
        self._agg_window = aggressor_window
        self._vol_buffer: list[float] = []
        self._agg_buffer: list[float] = []

    def name(self) -> str:
        return "trade_through"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_trade_through_bid", warmup=50,
                             description="P(market sell depletes best bid)"),
            AlgorithmFeature("alg_trade_through_ask", warmup=50,
                             description="P(market buy depletes best ask)"),
            AlgorithmFeature("alg_trade_through_imbalance", warmup=50,
                             description="ask_through - bid_through (directional)"),
        ]

    def required_columns(self) -> list[str]:
        return ["raw_bid_depth_5", "raw_ask_depth_5",
                "flow_volume_1s", "flow_aggressor_ratio_5s"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        bid_d = tick.get("raw_bid_depth_5", np.nan)
        ask_d = tick.get("raw_ask_depth_5", np.nan)
        vol = tick.get("flow_volume_1s", np.nan)
        agg = tick.get("flow_aggressor_ratio_5s", np.nan)

        if not all(np.isfinite(x) for x in [bid_d, ask_d, vol, agg]):
            return {f.name: np.nan for f in self.alg_features()}

        self._vol_buffer.append(vol)
        if len(self._vol_buffer) > self._vol_window:
            self._vol_buffer.pop(0)

        self._agg_buffer.append(agg)
        if len(self._agg_buffer) > self._agg_window:
            self._agg_buffer.pop(0)

        if len(self._vol_buffer) < 2:
            return {f.name: np.nan for f in self.alg_features()}

        avg_vol = np.mean(self._vol_buffer)
        avg_agg = np.mean(self._agg_buffer)

        # P(trade-through bid) ~ sell_volume / bid_depth
        sell_vol = avg_vol * (1 - avg_agg)  # sell fraction
        buy_vol = avg_vol * avg_agg          # buy fraction

        # Clamp to [0, 1] — probability estimate
        p_bid = min(sell_vol / (bid_d + 1e-12), 1.0)
        p_ask = min(buy_vol / (ask_d + 1e-12), 1.0)

        return {
            "alg_trade_through_bid": p_bid,
            "alg_trade_through_ask": p_ask,
            "alg_trade_through_imbalance": p_ask - p_bid,
        }

    def reset(self) -> None:
        self._vol_buffer.clear()
        self._agg_buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override using rolling means."""
        import pandas as pd

        bid_d = df["raw_bid_depth_5"].values.astype(np.float64)
        ask_d = df["raw_ask_depth_5"].values.astype(np.float64)
        vol = pd.Series(df["flow_volume_1s"].values.astype(np.float64))
        agg = pd.Series(df["flow_aggressor_ratio_5s"].values.astype(np.float64))

        avg_vol = vol.rolling(self._vol_window, min_periods=2).mean().values
        avg_agg = agg.rolling(self._agg_window, min_periods=2).mean().values

        sell_vol = avg_vol * (1 - avg_agg)
        buy_vol = avg_vol * avg_agg

        p_bid = np.clip(sell_vol / (bid_d + 1e-12), 0, 1)
        p_ask = np.clip(buy_vol / (ask_d + 1e-12), 0, 1)

        result = pd.DataFrame({
            "alg_trade_through_bid": p_bid,
            "alg_trade_through_ask": p_ask,
            "alg_trade_through_imbalance": p_ask - p_bid,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

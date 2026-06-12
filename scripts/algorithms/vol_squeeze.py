"""
Volatility Squeeze / Breakout Algorithm (A3 in feature_algorithm_gaps.md)

Volatility compression precedes expansion: sustained low short-term vol
relative to long-term vol (the "squeeze") marks energy building, and the
direction of the price move at the moment vol expands tends to persist.
Classic formulation: Bollinger band squeeze (Bollinger 2001, "Bollinger on
Bollinger Bands"); TTM squeeze (Carter 2006); statistically grounded in
volatility clustering (Engle 1982) — calm regimes and turbulent regimes
alternate, and the transition is the tradeable event.

Complements jump_detector: the jump test fires *after* a discontinuity;
the squeeze arms *before* the expansion. The input `vol_ratio_short_long`
is computed by the ingestor but consumed directionally by nothing else
(algorithm_classification.md §4.2 gap: momentum family at MF–macro).

State machine:

    IDLE --[ratio < q_low for >= min_squeeze_ticks]--> ARMED
    ARMED --[ratio > q_high]--> BREAKOUT (direction = sign of return over
                                breakout_lookback; signal decays linearly
                                over hold_ticks) --> IDLE

Thresholds q_low/q_high are rolling percentiles of the ratio over `window`
ticks (adaptive per symbol — the ratio's distribution differs by symbol),
recomputed every 100 ticks like regime_gated.

Features:
    alg_vsq_squeeze_on       1.0 while in a qualified squeeze (ARMED)
    alg_vsq_squeeze_ticks    ticks spent in the current/last squeeze
    alg_vsq_breakout_signal  primary: ±(0,1] from breakout, linear decay
    alg_vsq_ratio_pctile     current ratio's rolling percentile [0,1]

NaN inputs produce NaN outputs for that tick; internal state freezes.

First validation (2026-06-11, default params, rolling thresholds — causal,
nothing fitted to the day): ~40 breakouts/symbol/day, conditional 5m forward
IC +0.10 BTC / +0.23 ETH / -0.10 SOL; mean directional 5m forward return
+2.9 / +4.7 / -0.3 bps vs 1.61 bps RT. Preliminary (one day) but ETH/BTC
clear costs on paper; SOL does not — consistent with the house per-symbol
deployment pattern. Next: `nat algorithm evaluate` + the OOS sweep harness.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class VolSqueeze(MicrostructureAlgorithm):
    """Squeeze-then-breakout momentum on the short/long vol ratio."""

    def __init__(
        self,
        window: int = 6000,
        squeeze_pct: float = 25.0,
        expansion_pct: float = 75.0,
        min_squeeze_ticks: int = 600,
        breakout_lookback: int = 300,
        hold_ticks: int = 3000,
    ):
        self._window = int(window)
        self._squeeze_pct = float(squeeze_pct)
        self._expansion_pct = float(expansion_pct)
        self._min_squeeze_ticks = int(min_squeeze_ticks)
        self._breakout_lookback = int(breakout_lookback)
        self._hold_ticks = int(hold_ticks)
        self.reset()

    def name(self) -> str:
        return "vol_squeeze"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature(
                "alg_vsq_squeeze_on", warmup=600,
                description="1.0 while in a qualified vol squeeze",
            ),
            AlgorithmFeature(
                "alg_vsq_squeeze_ticks", warmup=600,
                description="Ticks spent in the current/last squeeze",
            ),
            AlgorithmFeature(
                "alg_vsq_breakout_signal", warmup=600,
                description="±(0,1] breakout momentum, linear decay (primary)",
            ),
            AlgorithmFeature(
                "alg_vsq_ratio_pctile", warmup=600,
                description="Rolling percentile of vol ratio [0,1]",
            ),
        ]

    def required_columns(self) -> list[str]:
        return ["vol_ratio_short_long", "raw_midprice"]

    # ── core scalar step (shared by step() and run_batch()) ──────────

    def _step_scalar(self, ratio: float, mid: float) -> tuple[float, float, float, float]:
        if not (np.isfinite(ratio) and np.isfinite(mid)):
            return (np.nan, np.nan, np.nan, np.nan)

        self._ratio_buf.append(ratio)
        self._price_buf.append(mid)
        self._tick_count += 1

        if len(self._ratio_buf) < 100:
            return (np.nan, np.nan, np.nan, np.nan)

        if self._tick_count % 100 == 0 or np.isnan(self._q_low):
            arr = np.fromiter(self._ratio_buf, dtype=np.float64)
            self._q_low = float(np.percentile(arr, self._squeeze_pct))
            self._q_high = float(np.percentile(arr, self._expansion_pct))
            self._arr_sorted = np.sort(arr)

        pctile = float(
            np.searchsorted(self._arr_sorted, ratio) / len(self._arr_sorted)
        )

        # squeeze tracking
        if ratio < self._q_low:
            self._squeeze_run += 1
            self._midband_run = 0
        elif ratio > self._q_high:
            self._squeeze_run = 0
            self._midband_run = 0
        else:
            # mid-band wobble is tolerated briefly, but a squeeze that
            # drifts in the mid-band for as long as it took to arm is over
            self._midband_run += 1
            if self._midband_run > self._min_squeeze_ticks:
                self._squeeze_run = 0

        armed = self._squeeze_run >= self._min_squeeze_ticks
        if armed:
            self._last_squeeze_ticks = self._squeeze_run
            self._since_armed = 0
        else:
            self._since_armed += 1
            # an armed squeeze that faded without expanding goes stale —
            # don't fire a "breakout" on ancient memory
            if self._since_armed > 2 * self._min_squeeze_ticks:
                self._last_squeeze_ticks = 0

        # breakout trigger: armed squeeze ends with an expansion
        if (
            self._hold_remaining == 0
            and self._last_squeeze_ticks >= self._min_squeeze_ticks
            and ratio > self._q_high
            and len(self._price_buf) > self._breakout_lookback
        ):
            past = self._price_buf[0]
            move = mid - past
            if move != 0.0:
                self._direction = float(np.sign(move))
                self._hold_remaining = self._hold_ticks
                self._last_squeeze_ticks = 0  # consume the squeeze
                self._squeeze_run = 0

        if self._hold_remaining > 0:
            signal = self._direction * self._hold_remaining / self._hold_ticks
            self._hold_remaining -= 1
        else:
            signal = 0.0

        return (
            1.0 if armed else 0.0,
            float(self._squeeze_run),
            signal,
            pctile,
        )

    # ── contract methods ──────────────────────────────────────────────

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        ratio = tick.get("vol_ratio_short_long", np.nan)
        mid = tick.get("raw_midprice", np.nan)
        on, ticks, signal, pctile = self._step_scalar(ratio, mid)
        return {
            "alg_vsq_squeeze_on": on,
            "alg_vsq_squeeze_ticks": ticks,
            "alg_vsq_breakout_signal": signal,
            "alg_vsq_ratio_pctile": pctile,
        }

    def reset(self) -> None:
        self._ratio_buf: deque[float] = deque(maxlen=self._window)
        self._price_buf: deque[float] = deque(maxlen=self._breakout_lookback + 1)
        self._tick_count = 0
        self._q_low = np.nan
        self._q_high = np.nan
        self._arr_sorted = np.empty(0)
        self._squeeze_run = 0
        self._midband_run = 0
        self._last_squeeze_ticks = 0
        self._since_armed = 0
        self._hold_remaining = 0
        self._direction = 0.0

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Array loop through the same scalar step — exact step() parity
        without per-row dict/iloc overhead."""
        self.reset()
        ratio = df["vol_ratio_short_long"].to_numpy(dtype=np.float64)
        mid = df["raw_midprice"].to_numpy(dtype=np.float64)
        n = len(df)
        out = np.empty((n, 4))
        for i in range(n):
            out[i] = self._step_scalar(ratio[i], mid[i])

        result = pd.DataFrame(out, columns=self.feature_names, index=df.index)
        warmup = self.warmup
        if 0 < warmup < n:
            result.iloc[:warmup] = np.nan
        return result

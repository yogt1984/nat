"""
VPIN-gated micro-VWAP mean-reversion (GAP-03 / ALG-3).

A deviation from micro-VWAP has two generators:
  - liquidity noise (inventory shocks, sloppy flow) → reverts to fair value → FADE it,
  - informed flow (someone knows something)         → price CONTINUES → STAND ASIDE.

Fading indiscriminately earns the noise-reversion and pays the informed-continuation —
the adverse-selection tax behind the fill-conditional IC collapse. VPIN is the cheapest
classifier between the two (Easley, López de Prado & O'Hara 2012): its
institutionally-correct use is to *veto* trades, never to pick direction (using VPIN as a
signal was falsified by `vpin_regime`). Direction comes from the VWAP-deviation z-score;
permission comes from low toxicity.

Refs: docs/GAP__26_7_26.md/03_toxic_vwap_reversion.md; FINDINGS §1 axis-5 (flow_vwap_deviation
is a real mean-reverting axis, negative IC ≈ 0.25 @1s); Spannung Phase D–E.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class ToxicVwapReversion(MicrostructureAlgorithm):
    """Fade the VWAP-deviation z-score, but only in low-toxicity (low-VPIN) states."""

    def __init__(self, w_z: int = 96, w_p: int = 288, k_entry: float = 2.0,
                 k_exit: float = 0.5, theta_pct: float = 0.70,
                 spread_pct_max: float = 0.90, size_by_toxicity: bool = True):
        self._w_z = w_z                      # z-score window (deviation)
        self._w_p = w_p                      # percentile window (vpin, spread)
        self._k_entry = k_entry              # |z| entry threshold
        self._k_exit = k_exit                # |z| exit threshold (position-side; runner uses it)
        self._theta = theta_pct              # gate open when vpin percentile < theta
        self._spread_pct_max = spread_pct_max  # skip when spread percentile >= this
        self._size_by_toxicity = size_by_toxicity
        self._dev: deque[float] = deque(maxlen=w_z)
        self._vpin: deque[float] = deque(maxlen=w_p)
        self._spread: deque[float] = deque(maxlen=w_p)

    def name(self) -> str:
        return "toxic_vwap_reversion"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_txvr_z", warmup=self._w_z,
                             description="Rolling z-score of flow_vwap_deviation"),
            AlgorithmFeature("alg_txvr_gate", warmup=self._w_p,
                             description="VPIN/spread permission gate (1 open / 0 stand aside)"),
            AlgorithmFeature("alg_txvr_signal", warmup=max(self._w_z, self._w_p),
                             description="-sign(z)·1[|z|>k_entry]·gate, optionally sized by (1-vpin_pct)"),
        ]

    def required_columns(self) -> list[str]:
        return ["flow_vwap_deviation", "toxic_vpin_50", "raw_spread_bps"]

    @staticmethod
    def _pct_le(buf: deque[float], x: float) -> float:
        """Fraction of the window <= x (max-rank percentile; matches rolling.rank(pct))."""
        n = len(buf)
        if n == 0:
            return np.nan
        return sum(1 for v in buf if v <= x) / n

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        d = tick.get("flow_vwap_deviation", np.nan)
        vpin = tick.get("toxic_vpin_50", np.nan)
        spread = tick.get("raw_spread_bps", np.nan)

        if not all(np.isfinite(x) for x in (d, vpin, spread)):
            return {f.name: np.nan for f in self.alg_features()}

        self._dev.append(d)
        self._vpin.append(vpin)
        self._spread.append(spread)

        # Direction: z-score of the deviation over its trailing window.
        arr = np.fromiter(self._dev, dtype=np.float64)
        mu, sd = arr.mean(), arr.std()  # ddof=0
        z = (d - mu) / sd if sd > 1e-9 else 0.0

        # Permission: low VPIN and non-blown spread (percentile ranks in their windows).
        vpin_pct = self._pct_le(self._vpin, vpin)
        spread_pct = self._pct_le(self._spread, spread)
        gate = 1.0 if (vpin_pct < self._theta and spread_pct < self._spread_pct_max) else 0.0

        base = -np.sign(z) if abs(z) > self._k_entry else 0.0
        size = (1.0 - vpin_pct) if self._size_by_toxicity else 1.0
        signal = base * gate * size

        return {"alg_txvr_z": z, "alg_txvr_gate": gate, "alg_txvr_signal": signal}

    def reset(self) -> None:
        self._dev.clear()
        self._vpin.clear()
        self._spread.clear()

    def run_batch(self, df):
        """Vectorized override (rolling z + rolling percentile ranks)."""
        import pandas as pd

        d = df["flow_vwap_deviation"].astype(np.float64)
        vpin = df["toxic_vpin_50"].astype(np.float64)
        spread = df["raw_spread_bps"].astype(np.float64)

        # z-score of the deviation (ddof=0 to match step()).
        mu = d.rolling(self._w_z, min_periods=self._w_z).mean()
        sd = d.rolling(self._w_z, min_periods=self._w_z).std(ddof=0).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):  # warmup NaNs masked by where
            z = np.where(sd > 1e-9, (d - mu).to_numpy() / sd, 0.0)

        # Percentile ranks (max-method = fraction of window <= current, matches _pct_le).
        vpin_pct = vpin.rolling(self._w_p, min_periods=self._w_p).rank(method="max", pct=True).to_numpy()
        spread_pct = spread.rolling(self._w_p, min_periods=self._w_p).rank(method="max", pct=True).to_numpy()

        gate = ((vpin_pct < self._theta) & (spread_pct < self._spread_pct_max)).astype(np.float64)

        active = np.abs(z) > self._k_entry
        base = np.where(active, -np.sign(z), 0.0)
        size = (1.0 - vpin_pct) if self._size_by_toxicity else 1.0
        signal = base * gate * size

        # NaN-in → NaN-out: any missing required input blanks that row's outputs.
        bad = ~(np.isfinite(d.to_numpy()) & np.isfinite(vpin.to_numpy()) & np.isfinite(spread.to_numpy()))
        for a in (z, gate, signal):
            a[bad] = np.nan

        result = pd.DataFrame(
            {"alg_txvr_z": z, "alg_txvr_gate": gate, "alg_txvr_signal": signal},
            index=df.index,
        )

        warmup = self.warmup
        if 0 < warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

"""Standalone VPIN permission gate (HF4 extraction).

§4.5's 58-day walk-forward validated the VPIN gate *directionally* — as a veto inside
`toxic_vwap_reversion` it improved Sharpe on all three symbols (BTC −8.1 vs −15.1,
ETH −8.1 vs −13.8, SOL −6.1 vs −8.3) by removing the adverse-selection loss tail. But the
gate lived welded to a taker fade that is itself dead at honest costs, so nothing else
could consume it. This unit is the permission half alone, registered so the maker path
(HF1 microprice quoting, A4-gated mean reversion, Track B/C entries) can compose with it.

Institutional use of VPIN (Easley, López de Prado & O'Hara 2012) is to **veto trades,
never to pick direction** — VPIN-as-signal was falsified by `vpin_regime` (−7,331 bps)
and tick-VPIN carries no directional IC at any horizon (§1). Outputs are therefore a
percentile, a binary permission and a size weight; no direction is emitted.

    vpin_pct(t)   = frac of trailing w_p window <= vpin(t)      (max-rank percentile)
    gate(t)       = 1[vpin_pct < theta_pct  AND  spread_pct < spread_pct_max]
    size(t)       = (1 - vpin_pct) * gate                        (calmer -> larger)

Parameters are §4.5's validated ones (`[toxic_vwap_reversion]` in algorithms.toml), not
new choices; `tests/test_vpin_gate.py` pins bit-for-bit parity with the donor's gate.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class VpinGate(MicrostructureAlgorithm):
    """Permission gate: open in low-toxicity, sane-spread states; no direction."""

    def __init__(self, w_p: int = 288, theta_pct: float = 0.70,
                 spread_pct_max: float = 0.90):
        self._w_p = w_p                        # percentile window (vpin, spread)
        self._theta = theta_pct                # open when vpin percentile < theta
        self._spread_pct_max = spread_pct_max  # closed when spread percentile >= this
        self._vpin: deque[float] = deque(maxlen=w_p)
        self._spread: deque[float] = deque(maxlen=w_p)

    def name(self) -> str:
        return "vpin_gate"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_vping_pct", warmup=self._w_p,
                             description="Rolling max-rank percentile of toxic_vpin_50"),
            AlgorithmFeature("alg_vping_gate", warmup=self._w_p,
                             description="Permission (1 open / 0 stand aside): low VPIN and sane spread"),
            AlgorithmFeature("alg_vping_size", warmup=self._w_p,
                             description="(1 - vpin_pct) * gate — toxicity-scaled size weight"),
        ]

    def required_columns(self) -> list[str]:
        return ["toxic_vpin_50", "raw_spread_bps"]

    @staticmethod
    def _pct_le(buf: deque[float], x: float) -> float:
        """Fraction of the window <= x (max-rank; matches rolling.rank(pct) and the
        donor's convention — the §4.5 record depends on this exact tie handling)."""
        n = len(buf)
        if n == 0:
            return np.nan
        return sum(1 for v in buf if v <= x) / n

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        vpin = tick.get("toxic_vpin_50", np.nan)
        spread = tick.get("raw_spread_bps", np.nan)

        if not all(np.isfinite(x) for x in (vpin, spread)):
            return {f.name: np.nan for f in self.alg_features()}

        self._vpin.append(vpin)
        self._spread.append(spread)

        vpin_pct = self._pct_le(self._vpin, vpin)
        spread_pct = self._pct_le(self._spread, spread)
        gate = 1.0 if (vpin_pct < self._theta
                       and spread_pct < self._spread_pct_max) else 0.0
        return {"alg_vping_pct": vpin_pct,
                "alg_vping_gate": gate,
                "alg_vping_size": (1.0 - vpin_pct) * gate}

    def reset(self) -> None:
        self._vpin.clear()
        self._spread.clear()

    def run_batch(self, df):
        """Vectorized override (rolling percentile ranks; donor-identical)."""
        import pandas as pd

        vpin = df["toxic_vpin_50"].astype(np.float64)
        spread = df["raw_spread_bps"].astype(np.float64)

        # .copy(): pandas-3 to_numpy() may hand back a read-only view, and the NaN
        # masking below writes in place.
        vpin_pct = vpin.rolling(self._w_p, min_periods=self._w_p) \
                       .rank(method="max", pct=True).to_numpy().copy()
        spread_pct = spread.rolling(self._w_p, min_periods=self._w_p) \
                           .rank(method="max", pct=True).to_numpy().copy()

        gate = ((vpin_pct < self._theta)
                & (spread_pct < self._spread_pct_max)).astype(np.float64)
        size = (1.0 - vpin_pct) * gate

        bad = ~(np.isfinite(vpin.to_numpy()) & np.isfinite(spread.to_numpy()))
        for a in (vpin_pct, gate, size):
            a[bad] = np.nan

        result = pd.DataFrame(
            {"alg_vping_pct": vpin_pct, "alg_vping_gate": gate,
             "alg_vping_size": size},
            index=df.index,
        )
        warmup = self.warmup
        if 0 < warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

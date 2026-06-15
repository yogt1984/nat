"""Funding-Settlement Window Effects (LF1 — algorithm_candidates_literature.md).

Funding / mark-oracle premium dislocations cluster around the 8h settlement marks
(00/08/16 UTC) that Binance/OKX/Bybit settle on, and mean-revert — cross-venue
arbitrage flow is forced through event time. This *conditions* premium reversion
on the settlement clock (the F1 idea), complementing the already-deployed
`funding_reversion` (100min horizon) with 1–8h event-time structure rather than
competing with it.

Refs: perp-microstructure settlement seasonality (MDPI 2026); intraday
funding/volatility seasonality peaking 16:00–17:00 UTC (Springer 2024;
arXiv 2109.12142). **NOT deployable** until walk-forward OOS on ≥7 clean days —
`momentum_continuation` overfit is the cautionary tale.

Signal: fade the premium z-score, scaled by proximity to the nearest 8h mark.
Near a mark the reversion edge is strongest; far from one the signal decays to 0.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

_SETTLE_PERIOD_S = 8 * 3600  # Binance/OKX/Bybit settle at 00/08/16 UTC


def _settlement_proximity(ts_ns: float, prox_s: float) -> float:
    """1.0 at an 8h settlement mark, decaying exp(-dist/prox_s) → 0 away from one."""
    sec = (int(ts_ns) // 1_000_000_000) % _SETTLE_PERIOD_S
    nearest = min(sec, _SETTLE_PERIOD_S - sec)
    return float(np.exp(-nearest / prox_s))


@register
class FundingSettlement(MicrostructureAlgorithm):
    """LF1: mark/oracle premium mean-reversion conditioned on the 8h settlement clock."""

    def __init__(self, premium_col: str = "ctx_premium_bps",
                 z_window: int = 600, prox_hours: float = 1.0):
        self._premium_col = premium_col
        self._z_window = z_window
        self._prox_s = prox_hours * 3600.0
        self._buf: list[float] = []

    def name(self) -> str:
        return "funding_settlement"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_settlement_signal", warmup=self._z_window,
                             description="Premium reversion fade, scaled by 8h-settlement proximity"),
            AlgorithmFeature("alg_settlement_proximity", warmup=0,
                             description="exp(-dist to nearest 00/08/16 UTC mark), in [0,1]"),
            AlgorithmFeature("alg_settlement_premium_z", warmup=self._z_window,
                             description="Rolling z-score of the mark/oracle premium"),
        ]

    def required_columns(self) -> list[str]:
        return [self._premium_col, "timestamp_ns"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        prem = tick.get(self._premium_col, np.nan)
        ts = tick.get("timestamp_ns", np.nan)
        if not (np.isfinite(prem) and np.isfinite(ts)):
            return {f.name: np.nan for f in self.alg_features()}
        prox = _settlement_proximity(ts, self._prox_s)
        self._buf.append(prem)
        if len(self._buf) > self._z_window:
            self._buf.pop(0)
        if len(self._buf) < self._z_window:
            return {"alg_settlement_signal": np.nan,
                    "alg_settlement_proximity": prox,
                    "alg_settlement_premium_z": np.nan}
        arr = np.asarray(self._buf)
        sd = arr.std()
        z = (prem - arr.mean()) / sd if sd > 1e-12 else 0.0
        return {"alg_settlement_signal": float(-np.tanh(z) * prox),
                "alg_settlement_proximity": prox,
                "alg_settlement_premium_z": float(z)}

    def reset(self) -> None:
        self._buf.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
        """Vectorized override."""
        import pandas as pd

        prem = df[self._premium_col].to_numpy(dtype=np.float64)
        ts = df["timestamp_ns"].to_numpy(dtype=np.float64)
        sec = (ts.astype(np.int64) // 1_000_000_000) % _SETTLE_PERIOD_S
        prox = np.exp(-np.minimum(sec, _SETTLE_PERIOD_S - sec) / self._prox_s)

        s = pd.Series(prem)
        mu = s.rolling(self._z_window, min_periods=self._z_window).mean()
        sd = s.rolling(self._z_window, min_periods=self._z_window).std()
        z = ((s - mu) / sd.replace(0.0, np.nan)).to_numpy()

        out = pd.DataFrame({
            "alg_settlement_signal": -np.tanh(z) * prox,
            "alg_settlement_proximity": prox,
            "alg_settlement_premium_z": z,
        }, index=df.index)
        warmup = self.warmup
        if 0 < warmup < len(df):
            out.iloc[:warmup] = np.nan
        return out

"""
Micro-VWAP mean-reversion (GAP-01 / ALG-1) — the ungated baseline.

Price oscillates around rolling volume-weighted fair value; when it stretches, fade it.
This is the standalone version of the signal that already earns its keep as 1/3 of
`3f_liquidity`, isolated so its contribution is measurable on its own — and so it can
serve as the control that GAP-03 (`toxic_vwap_reversion`) is measured against: same
z-score fade, no VPIN gate.

Polarity: flow_vwap_deviation has NEGATIVE IC (high deviation -> down forward return),
so fading (signal = -sign(z)) is correct. Refs: FINDINGS §1 axis-5; Spannung Phase D-E
(OU half-life 5-7s). k_exit / max_hold are position-management params consumed by the
backtester; the algorithm itself emits the instantaneous fade signal.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class VwapReversion(MicrostructureAlgorithm):
    """Fade the z-scored deviation from rolling micro-VWAP."""

    def __init__(self, z_window: int = 96, k_entry: float = 2.0,
                 k_exit: float = 0.5, max_hold: int = 12):
        self._z_window = z_window
        self._k_entry = k_entry
        self._k_exit = k_exit          # position-side exit threshold (backtester)
        self._max_hold = max_hold      # position-side hold cap (backtester)
        self._dev: deque[float] = deque(maxlen=z_window)

    def name(self) -> str:
        return "vwap_reversion"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_vwaprev_z", warmup=self._z_window,
                             description="Rolling z-score of flow_vwap_deviation"),
            AlgorithmFeature("alg_vwaprev_signal", warmup=self._z_window,
                             description="-sign(z)·1[|z|>k_entry] (fade the deviation)"),
        ]

    def required_columns(self) -> list[str]:
        return ["flow_vwap_deviation"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        d = tick.get("flow_vwap_deviation", np.nan)
        if not np.isfinite(d):
            return {f.name: np.nan for f in self.alg_features()}

        self._dev.append(d)
        arr = np.fromiter(self._dev, dtype=np.float64)
        mu, sd = arr.mean(), arr.std()  # ddof=0
        z = (d - mu) / sd if sd > 1e-9 else 0.0
        signal = -np.sign(z) if abs(z) > self._k_entry else 0.0
        return {"alg_vwaprev_z": z, "alg_vwaprev_signal": signal}

    def reset(self) -> None:
        self._dev.clear()

    def run_batch(self, df):
        """Vectorized override (rolling z + fade signal)."""
        import pandas as pd

        d = df["flow_vwap_deviation"].astype(np.float64)
        mu = d.rolling(self._z_window, min_periods=self._z_window).mean()
        sd = d.rolling(self._z_window, min_periods=self._z_window).std(ddof=0).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):  # warmup NaNs masked by where
            z = np.where(sd > 1e-9, (d - mu).to_numpy() / sd, 0.0)

        signal = np.where(np.abs(z) > self._k_entry, -np.sign(z), 0.0)

        bad = ~np.isfinite(d.to_numpy())
        z[bad] = np.nan
        signal[bad] = np.nan

        result = pd.DataFrame(
            {"alg_vwaprev_z": z, "alg_vwaprev_signal": signal},
            index=df.index,
        )
        warmup = self.warmup
        if 0 < warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

"""
Open Interest Divergence Algorithm

Detects divergence between open interest changes and price movement.
Price moves without OI confirmation are weak; OI confirming price is strong.

Divergence is computed as:
  z_price - z_oi
where z_price and z_oi are z-score-normalized rolling trends. This makes
the signal dimensionless and robust to differing volatility regimes.

Reference:
  Crypto perpetual futures microstructure — OI as a proxy for
  new position creation vs existing position closure.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class OIDivergence(MicrostructureAlgorithm):
    """OI vs price divergence signal."""

    def __init__(self, price_window: int = 300, oi_window: int = 300):
        self._price_window = price_window
        self._oi_window = oi_window
        self._price_buffer: list[float] = []
        self._oi_buffer: list[float] = []
        self._price_trend_buffer: list[float] = []
        self._oi_trend_buffer: list[float] = []

    def name(self) -> str:
        return "oi_divergence"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_oi_price_divergence", warmup=100,
                             description="OI trend vs price trend divergence"),
            AlgorithmFeature("alg_oi_confirmation", warmup=100,
                             description="1.0 if OI confirms price direction"),
            AlgorithmFeature("alg_oi_momentum", warmup=50,
                             description="OI change rate (normalized)"),
        ]

    def required_columns(self) -> list[str]:
        return ["ctx_open_interest", "ctx_oi_change_5m", "raw_midprice",
                "trend_momentum_300"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        oi = tick.get("ctx_open_interest", np.nan)
        oi_change = tick.get("ctx_oi_change_5m", np.nan)
        mid = tick.get("raw_midprice", np.nan)
        momentum = tick.get("trend_momentum_300", np.nan)

        if not all(np.isfinite(x) for x in [oi, oi_change, mid, momentum]):
            return {f.name: np.nan for f in self.alg_features()}

        self._price_buffer.append(mid)
        if len(self._price_buffer) > self._price_window:
            self._price_buffer.pop(0)

        self._oi_buffer.append(oi)
        if len(self._oi_buffer) > self._oi_window:
            self._oi_buffer.pop(0)

        if len(self._price_buffer) < 50 or len(self._oi_buffer) < 50:
            return {f.name: np.nan for f in self.alg_features()}

        # Price trend: log return over window
        price_trend = np.log(self._price_buffer[-1] / self._price_buffer[0])

        # OI trend: relative change over window
        oi_trend = (self._oi_buffer[-1] - self._oi_buffer[0]) / (self._oi_buffer[0] + 1e-12)

        # Z-score normalize both trends for dimensionless comparison
        self._price_trend_buffer.append(price_trend)
        self._oi_trend_buffer.append(oi_trend)
        if len(self._price_trend_buffer) > self._price_window:
            self._price_trend_buffer.pop(0)
        if len(self._oi_trend_buffer) > self._oi_window:
            self._oi_trend_buffer.pop(0)

        pt_std = np.std(self._price_trend_buffer) or 1e-12
        ot_std = np.std(self._oi_trend_buffer) or 1e-12
        z_price = price_trend / pt_std
        z_oi = oi_trend / ot_std

        # Divergence: z_price - z_oi (dimensionless, ~N(0,1) scale)
        price_sign = np.sign(price_trend)
        oi_sign = np.sign(oi_trend)

        divergence = z_price - z_oi

        # Confirmation: same direction
        confirmation = 1.0 if price_sign == oi_sign and price_sign != 0 else 0.0

        # OI momentum (normalized change)
        oi_momentum = oi_change / (oi + 1e-12)

        return {
            "alg_oi_price_divergence": divergence,
            "alg_oi_confirmation": confirmation,
            "alg_oi_momentum": oi_momentum,
        }

    def reset(self) -> None:
        self._price_buffer.clear()
        self._oi_buffer.clear()
        self._price_trend_buffer.clear()
        self._oi_trend_buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        mid = df["raw_midprice"].values.astype(np.float64)
        oi = df["ctx_open_interest"].values.astype(np.float64)
        oi_change = df["ctx_oi_change_5m"].values.astype(np.float64)

        # Rolling price trend (log return over window)
        mid_s = pd.Series(mid)
        price_trend = np.log(mid_s / mid_s.shift(self._price_window))

        # Rolling OI trend (relative change)
        oi_s = pd.Series(oi)
        oi_start = oi_s.shift(self._oi_window)
        oi_trend = (oi_s - oi_start) / (oi_start + 1e-12)

        # Z-score normalize both for dimensionless divergence
        pt_std = price_trend.rolling(self._price_window, min_periods=50).std().replace(0, np.nan).fillna(1e-12)
        ot_std = oi_trend.rolling(self._oi_window, min_periods=50).std().replace(0, np.nan).fillna(1e-12)
        z_price = price_trend / pt_std
        z_oi = oi_trend / ot_std

        divergence = (z_price - z_oi).values

        price_sign = np.sign(price_trend.values)
        oi_sign = np.sign(oi_trend.values)
        confirmation = ((price_sign == oi_sign) & (price_sign != 0)).astype(np.float64)
        confirmation[np.isnan(price_trend.values) | np.isnan(oi_trend.values)] = np.nan

        oi_momentum = oi_change / (oi + 1e-12)

        result = pd.DataFrame({
            "alg_oi_price_divergence": divergence,
            "alg_oi_confirmation": confirmation,
            "alg_oi_momentum": oi_momentum,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

"""
Spread Decomposition Algorithm

Decomposes the bid-ask spread into adverse selection and inventory/order-processing
components. Tracks the adverse selection trend as a regime indicator.

Reference:
  Hendershott, Jones & Menkveld (2011) — "Does algorithmic trading improve liquidity?"
  Huang & Stoll (1997) — "The components of the bid-ask spread"
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class SpreadDecomp(MicrostructureAlgorithm):
    """Effective/realized spread decomposition into adverse selection component."""

    def __init__(self, ema_span: int = 100, regime_window: int = 300,
                 regime_percentile: float = 70.0):
        self._ema_span = ema_span
        self._ema_alpha = 2.0 / (ema_span + 1)
        self._regime_window = regime_window
        self._regime_percentile = regime_percentile
        self._ema_adverse = np.nan
        self._adverse_buffer: list[float] = []
        self._prev_realized: float = np.nan

    def name(self) -> str:
        return "spread_decomp"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_adverse_component", warmup=20,
                             description="Adverse selection = effective - realized spread"),
            AlgorithmFeature("alg_adverse_trend", warmup=100,
                             description="EMA of adverse selection component"),
            AlgorithmFeature("alg_spread_regime", warmup=100,
                             description="1.0 if adverse > P70 (high informed trading)"),
        ]

    def required_columns(self) -> list[str]:
        return ["toxic_effective_spread", "toxic_realized_spread"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        eff = tick.get("toxic_effective_spread", np.nan)
        real = tick.get("toxic_realized_spread", np.nan)

        if not (np.isfinite(eff) and np.isfinite(real)):
            return {f.name: np.nan for f in self.alg_features()}

        # Adverse selection: use *lagged* realized spread to avoid causality
        # violation (realized spread at time t is only known after the trade)
        if np.isnan(self._prev_realized):
            adverse = 0.0  # first tick: no prior realized available
        else:
            adverse = eff - self._prev_realized
        self._prev_realized = real

        # EMA trend
        if np.isnan(self._ema_adverse):
            self._ema_adverse = adverse
        else:
            self._ema_adverse = self._ema_alpha * adverse + (1 - self._ema_alpha) * self._ema_adverse

        # Rolling percentile for regime detection
        self._adverse_buffer.append(adverse)
        if len(self._adverse_buffer) > self._regime_window:
            self._adverse_buffer.pop(0)

        if len(self._adverse_buffer) < 20:
            return {
                "alg_adverse_component": adverse,
                "alg_adverse_trend": self._ema_adverse,
                "alg_spread_regime": np.nan,
            }

        threshold = np.percentile(self._adverse_buffer, self._regime_percentile)
        regime = 1.0 if adverse > threshold else 0.0

        return {
            "alg_adverse_component": adverse,
            "alg_adverse_trend": self._ema_adverse,
            "alg_spread_regime": regime,
        }

    def reset(self) -> None:
        self._ema_adverse = np.nan
        self._adverse_buffer.clear()
        self._prev_realized = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        eff = df["toxic_effective_spread"].values.astype(np.float64)
        real = df["toxic_realized_spread"].values.astype(np.float64)

        # Lag realized spread by 1 tick to avoid causality violation
        real_lagged = np.empty_like(real)
        real_lagged[0] = np.nan
        real_lagged[1:] = real[:-1]

        adverse = eff - real_lagged
        trend = pd.Series(adverse).ewm(span=self._ema_span, min_periods=1).mean().values

        # Rolling percentile for regime
        adverse_s = pd.Series(adverse)
        rolling_thresh = adverse_s.rolling(
            self._regime_window, min_periods=20
        ).quantile(self._regime_percentile / 100.0).values

        regime = (adverse > rolling_thresh).astype(np.float64)
        regime[np.isnan(rolling_thresh)] = np.nan

        result = pd.DataFrame({
            "alg_adverse_component": adverse,
            "alg_adverse_trend": trend,
            "alg_spread_regime": regime,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

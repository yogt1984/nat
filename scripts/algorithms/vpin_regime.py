"""
VPIN Regime Switch Algorithm

Gates trading signals by VPIN (Volume-Synchronized Probability of Informed Trading).
High VPIN = toxic flow regime → go flat. Low VPIN = safe to quote/trade.

Reference:
  Easley, López de Prado & O'Hara (2012) — "Flow toxicity and liquidity in a
  high-frequency world"
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class VPINRegime(MicrostructureAlgorithm):
    """VPIN-triggered regime switch for signal gating."""

    def __init__(self, vpin_threshold_pct: float = 80.0, momentum_span: int = 50,
                 gate_window: int = 300):
        self._threshold_pct = vpin_threshold_pct
        self._momentum_span = momentum_span
        self._gate_window = gate_window
        self._ema_alpha = 2.0 / (momentum_span + 1)
        self._ema_vpin = np.nan
        self._vpin_buffer: list[float] = []

    def name(self) -> str:
        return "vpin_regime"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_vpin_regime", warmup=100,
                             description="1.0 = toxic (VPIN > threshold), 0.0 = safe"),
            AlgorithmFeature("alg_vpin_gated_imbalance", warmup=100,
                             description="Imbalance when VPIN safe, else 0"),
            AlgorithmFeature("alg_vpin_momentum", warmup=50,
                             description="EMA of VPIN (trend in toxicity)"),
        ]

    def required_columns(self) -> list[str]:
        return ["toxic_vpin_50", "imbalance_qty_l1"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        vpin = tick.get("toxic_vpin_50", np.nan)
        imb = tick.get("imbalance_qty_l1", np.nan)

        if not np.isfinite(vpin):
            return {f.name: np.nan for f in self.alg_features()}

        # Update VPIN buffer
        self._vpin_buffer.append(vpin)
        if len(self._vpin_buffer) > self._gate_window:
            self._vpin_buffer.pop(0)

        # EMA momentum
        if np.isnan(self._ema_vpin):
            self._ema_vpin = vpin
        else:
            self._ema_vpin = self._ema_alpha * vpin + (1 - self._ema_alpha) * self._ema_vpin

        if len(self._vpin_buffer) < 50:
            return {
                "alg_vpin_regime": np.nan,
                "alg_vpin_gated_imbalance": np.nan,
                "alg_vpin_momentum": self._ema_vpin,
            }

        # Percentile threshold
        threshold = np.percentile(self._vpin_buffer, self._threshold_pct)
        toxic = 1.0 if vpin > threshold else 0.0

        # Gate imbalance: only active when NOT toxic
        gated_imb = imb * (1.0 - toxic) if np.isfinite(imb) else 0.0

        return {
            "alg_vpin_regime": toxic,
            "alg_vpin_gated_imbalance": gated_imb,
            "alg_vpin_momentum": self._ema_vpin,
        }

    def reset(self) -> None:
        self._ema_vpin = np.nan
        self._vpin_buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        vpin = df["toxic_vpin_50"].values.astype(np.float64)
        imb = df["imbalance_qty_l1"].values.astype(np.float64)

        vpin_s = pd.Series(vpin)
        rolling_thresh = vpin_s.rolling(
            self._gate_window, min_periods=50
        ).quantile(self._threshold_pct / 100.0).values

        toxic = (vpin > rolling_thresh).astype(np.float64)
        toxic[np.isnan(rolling_thresh)] = np.nan

        gated_imb = imb * (1.0 - toxic)
        gated_imb[~np.isfinite(imb)] = 0.0
        gated_imb[np.isnan(toxic)] = np.nan

        momentum = vpin_s.ewm(span=self._momentum_span, min_periods=1).mean().values

        result = pd.DataFrame({
            "alg_vpin_regime": toxic,
            "alg_vpin_gated_imbalance": gated_imb,
            "alg_vpin_momentum": momentum,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

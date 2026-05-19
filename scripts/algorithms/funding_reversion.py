"""
Funding Rate Mean-Reversion Algorithm

Extreme funding rates on perpetual futures tend to revert. This algorithm
generates a signal from the funding z-score and premium divergence.

Reference:
  Crypto-specific — funding rate arbitrage is well-documented in
  perpetual futures market microstructure literature.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class FundingReversion(MicrostructureAlgorithm):
    """Funding rate mean-reversion signal with premium divergence."""

    def __init__(self, zscore_entry: float = 2.0, momentum_span: int = 100,
                 premium_weight: float = 0.3):
        self._zscore_entry = zscore_entry
        self._momentum_span = momentum_span
        self._premium_weight = premium_weight
        self._ema_alpha = 2.0 / (momentum_span + 1)
        self._ema_funding = np.nan

    def name(self) -> str:
        return "funding_reversion"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_funding_signal", warmup=10,
                             description="Funding reversion signal: -sign(zscore) when |z|>threshold"),
            AlgorithmFeature("alg_funding_momentum", warmup=100,
                             description="EMA of funding rate (trend detection)"),
            AlgorithmFeature("alg_premium_divergence", warmup=10,
                             description="Weighted combo: funding_zscore vs premium_bps"),
        ]

    def required_columns(self) -> list[str]:
        return ["ctx_funding_rate", "ctx_funding_zscore", "ctx_premium_bps"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        rate = tick.get("ctx_funding_rate", np.nan)
        zscore = tick.get("ctx_funding_zscore", np.nan)
        premium = tick.get("ctx_premium_bps", np.nan)

        if not all(np.isfinite(x) for x in [rate, zscore, premium]):
            return {f.name: np.nan for f in self.alg_features()}

        # Reversion signal: short when funding extremely positive, long when negative
        if abs(zscore) >= self._zscore_entry:
            signal = -np.sign(zscore) * min(abs(zscore) / self._zscore_entry, 3.0) / 3.0
        else:
            signal = 0.0

        # Momentum
        if np.isnan(self._ema_funding):
            self._ema_funding = rate
        else:
            self._ema_funding = self._ema_alpha * rate + (1 - self._ema_alpha) * self._ema_funding

        # Premium divergence: combine funding zscore with premium
        # Normalize premium to comparable scale (typical premium is ±50 bps)
        premium_z = premium / 10.0  # rough scaling
        divergence = (1 - self._premium_weight) * zscore + self._premium_weight * premium_z

        return {
            "alg_funding_signal": signal,
            "alg_funding_momentum": self._ema_funding,
            "alg_premium_divergence": divergence,
        }

    def reset(self) -> None:
        self._ema_funding = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        zscore = df["ctx_funding_zscore"].values.astype(np.float64)
        rate = df["ctx_funding_rate"].values.astype(np.float64)
        premium = df["ctx_premium_bps"].values.astype(np.float64)

        # Signal: -sign(z) * clamp(|z|/entry, 0, 1) when |z| >= entry, else 0
        abs_z = np.abs(zscore)
        active = abs_z >= self._zscore_entry
        signal = np.where(active,
                          -np.sign(zscore) * np.minimum(abs_z / self._zscore_entry, 3.0) / 3.0,
                          0.0)

        momentum = pd.Series(rate).ewm(span=self._momentum_span, min_periods=1).mean().values

        premium_z = premium / 10.0
        divergence = (1 - self._premium_weight) * zscore + self._premium_weight * premium_z

        result = pd.DataFrame({
            "alg_funding_signal": signal,
            "alg_funding_momentum": momentum,
            "alg_premium_divergence": divergence,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

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
                 premium_weight: float = 0.3, halflife_window: int = 200):
        self._zscore_entry = zscore_entry
        self._momentum_span = momentum_span
        self._premium_weight = premium_weight
        self._halflife_window = halflife_window
        self._ema_alpha = 2.0 / (momentum_span + 1)
        self._ema_funding = np.nan
        self._zscore_buffer: list[float] = []

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
            AlgorithmFeature("alg_funding_halflife_ticks", warmup=100,
                             description="OU half-life from lag-1 AR on funding z-score"),
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

        # OU half-life via lag-1 AR(1) on z-score buffer
        self._zscore_buffer.append(zscore)
        if len(self._zscore_buffer) > self._halflife_window:
            self._zscore_buffer.pop(0)

        halflife = np.nan
        if len(self._zscore_buffer) >= 50:
            z = np.array(self._zscore_buffer)
            z_lag, z_now = z[:-1], z[1:]
            denom = np.var(z_lag)
            if denom > 1e-12:
                rho = np.mean((z_lag - z_lag.mean()) * (z_now - z_now.mean())) / denom
                if 0 < rho < 1:
                    halflife = -np.log(2) / np.log(rho)
                elif rho <= 0:
                    halflife = 0.0  # anti-persistent: instant reversion
                else:
                    halflife = 1e6  # trending / random walk: no reversion

        return {
            "alg_funding_signal": signal,
            "alg_funding_momentum": self._ema_funding,
            "alg_premium_divergence": divergence,
            "alg_funding_halflife_ticks": halflife,
        }

    def reset(self) -> None:
        self._ema_funding = np.nan
        self._zscore_buffer.clear()

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

        # Rolling OU half-life: lag-1 autocorrelation on z-score
        z_s = pd.Series(zscore)
        z_lag = z_s.shift(1)
        roll_cov = z_s.rolling(self._halflife_window, min_periods=50).cov(z_lag).values
        roll_var = z_lag.rolling(self._halflife_window, min_periods=50).var().values
        rho = roll_cov / (roll_var + 1e-12)
        # half-life = -ln(2)/ln(ρ): mean-reverting when 0 < ρ < 1
        halflife = np.full_like(rho, np.nan)
        mr = (rho > 0) & (rho < 1)
        halflife[mr] = -np.log(2) / np.log(rho[mr])
        halflife[rho <= 0] = 0.0      # anti-persistent
        halflife[rho >= 1] = 1e6      # trending / random walk

        result = pd.DataFrame({
            "alg_funding_signal": signal,
            "alg_funding_momentum": momentum,
            "alg_premium_divergence": divergence,
            "alg_funding_halflife_ticks": halflife,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

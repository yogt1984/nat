"""
Propagator Model — Transient and Permanent Price Impact
========================================================

Mathematical Framework
----------------------
The propagator model (Bouchaud et al. 2004) decomposes realized price change
into a transient component (mean-reverts) and a permanent component (persists):

  Δp_t = Σ_{τ=1}^{T} G(τ) · ε_{t-τ}  +  λ · ε_t

where:
  ε_t    = signed order flow at time t  (buy = +1, sell = -1, weighted by volume)
  G(τ)   = impact decay kernel evaluated at age τ
  λ      = permanent impact coefficient (absorbed into the EMA term here)

Power-Law Decay Kernel
-----------------------
The kernel takes the power-law form empirically observed in equity markets:

  G(τ) = τ^(-α),   τ ≥ 1,   α ∈ (0, 1)   (decay_exponent)

For α = 0.5 (default), impact decays as the square root of age, consistent
with the square-root law of price impact. The kernel is non-integrable on
[0, ∞) for α ≤ 1, reflecting that autocorrelations of order flow are long-range.

Transient Impact — Online Estimation
-------------------------------------
The convolution is approximated over a finite causal window of length W:

  TI(t) = (1/W) · Σ_{i=0}^{W-1} ε_{t-i} · G(W - i)
         = (1/W) · Σ_{τ=1}^{W} ε_{t-τ+1} · τ^(-α)

where ε_t = sign(t) · V(t) with:
  sign(t) = 2 · agg(t) - 1  ∈ [-1, 1]
  agg(t)  = ``flow_aggressor_ratio_5s``  (fraction of buy-initiated volume)
  V(t)    = ``flow_volume_1s``

The (1/W) normalisation makes TI(t) comparable across window lengths.
The oldest observation in the buffer has age τ = W (smallest kernel weight);
the most recent has age τ = 1 (largest kernel weight).

In run_batch(), the convolution is computed explicitly via np.dot(chunk, kernel)
where kernel[i] = (W - i)^(-α) / W, reversed to be causal (index 0 = oldest).

Permanent Impact — EMA of Window Log-Return
--------------------------------------------
The permanent component captures the information content of order flow:

  p_perm(t) = ln(mid_t / mid_{t-W})    (log-return over impact window)

This is then smoothed with a long EMA (span = permanent_ema_span = 500):

  PI(t) = α_p · p_perm(t) + (1 - α_p) · PI(t-1),   α_p = 2/(span+1)

The EMA acts as a low-pass filter, preserving only the persistent drift.

Impact Decay Ratio
-------------------
  IDR(t) = TI(t) / (|PI(t)| + ε),   ε = 1e-12

Large IDR: current transient impact is large relative to permanent drift —
mean reversion expected. IDR near 0: price moves are dominantly permanent.

Numerical Stability
-------------------
- ε = 1e-12 guards against |PI| = 0 (quiescent periods with no net drift).
- The kernel is evaluated as arange(1, W+1)^(-α) then reversed, computed
  once for run_batch() and re-evaluated each tick in step().

Parameters
----------
  decay_exponent      (α)      : power-law exponent ∈ (0, 1), default 0.5
  impact_window       (W)      : causal window length in ticks, default 100
  permanent_ema_span          : EMA span for permanent impact, default 500

Output Ranges
-------------
  alg_transient_impact    : ℝ (signed, units of volume × tick)
  alg_permanent_impact    : ℝ (log-return, dimensionless)
  alg_impact_decay_ratio  : ℝ (unitless ratio, unbounded above)

Complexity: O(W) per tick for the transient convolution.

References
----------
  Bouchaud, J.-P., Gefen, Y., Potters, M. & Wyart, M. (2004) — "Fluctuations
    and response in financial markets: the subtle nature of 'random' price
    changes", Quantitative Finance 4(2), 176–190.
  Bouchaud, J.-P. (2009) — "Price impact" in Encyclopedia of Quantitative Finance,
    Wiley, pp. 1408–1411.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class Propagator(MicrostructureAlgorithm):
    """Transient impact model with power-law decay kernel."""

    def __init__(self, decay_exponent: float = 0.5, impact_window: int = 100,
                 permanent_ema_span: int = 500):
        self._decay_exp = decay_exponent
        self._window = impact_window
        self._perm_span = permanent_ema_span
        self._perm_alpha = 2.0 / (permanent_ema_span + 1)
        # Buffers
        self._signed_vol_buffer: list[float] = []
        self._price_buffer: list[float] = []
        self._ema_permanent = np.nan

    def name(self) -> str:
        return "propagator"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_transient_impact", warmup=100,
                             description="Cumulative transient impact (decayed signed volume)"),
            AlgorithmFeature("alg_permanent_impact", warmup=100,
                             description="EMA of realized permanent impact"),
            AlgorithmFeature("alg_impact_decay_ratio", warmup=100,
                             description="Transient / (|permanent| + eps) — impact reversion ratio"),
        ]

    def required_columns(self) -> list[str]:
        return ["flow_volume_1s", "flow_aggressor_ratio_5s", "raw_midprice"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        vol = tick.get("flow_volume_1s", np.nan)
        agg = tick.get("flow_aggressor_ratio_5s", np.nan)
        mid = tick.get("raw_midprice", np.nan)

        if not all(np.isfinite(x) for x in [vol, agg, mid]):
            return {f.name: np.nan for f in self.alg_features()}

        # Signed volume: buy_fraction - sell_fraction
        sign = 2 * agg - 1  # maps [0,1] → [-1,1]
        signed_vol = sign * vol

        self._signed_vol_buffer.append(signed_vol)
        self._price_buffer.append(mid)
        if len(self._signed_vol_buffer) > self._window:
            self._signed_vol_buffer.pop(0)
            self._price_buffer.pop(0)

        if len(self._signed_vol_buffer) < 10:
            return {f.name: np.nan for f in self.alg_features()}

        n = len(self._signed_vol_buffer)

        # Transient impact: sum of decayed signed volumes
        # G(τ) = τ^(-decay_exp) for τ >= 1
        transient = 0.0
        for i in range(n):
            tau = n - i  # age of this observation
            kernel = tau ** (-self._decay_exp)
            transient += self._signed_vol_buffer[i] * kernel

        # Normalize by window
        transient /= n

        # Permanent impact: price change relative to window start
        if self._price_buffer[0] > 0:
            permanent = np.log(self._price_buffer[-1] / self._price_buffer[0])
        else:
            permanent = 0.0

        # EMA of permanent impact
        if np.isnan(self._ema_permanent):
            self._ema_permanent = permanent
        else:
            self._ema_permanent = (self._perm_alpha * permanent +
                                   (1 - self._perm_alpha) * self._ema_permanent)

        # Decay ratio: how much impact is transient vs permanent
        decay_ratio = transient / (abs(self._ema_permanent) + 1e-12)

        return {
            "alg_transient_impact": transient,
            "alg_permanent_impact": self._ema_permanent,
            "alg_impact_decay_ratio": decay_ratio,
        }

    def reset(self) -> None:
        self._signed_vol_buffer.clear()
        self._price_buffer.clear()
        self._ema_permanent = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override using convolution with power-law kernel."""
        import pandas as pd

        vol = df["flow_volume_1s"].values.astype(np.float64)
        agg = df["flow_aggressor_ratio_5s"].values.astype(np.float64)
        mid = df["raw_midprice"].values.astype(np.float64)

        signed_vol = (2 * agg - 1) * vol

        # Build decay kernel
        kernel = np.arange(1, self._window + 1, dtype=np.float64) ** (-self._decay_exp)
        kernel = kernel[::-1]  # most recent has smallest τ (largest weight)
        kernel /= self._window

        # Convolve (causal — output[i] depends on input[i-window+1:i+1])
        n = len(df)
        transient = np.full(n, np.nan)
        for i in range(self._window - 1, n):
            chunk = signed_vol[i - self._window + 1:i + 1]
            transient[i] = np.dot(chunk, kernel)

        # Permanent: log return over window
        mid_s = pd.Series(mid)
        permanent = np.log(mid_s / mid_s.shift(self._window)).values
        perm_ema = pd.Series(permanent).ewm(
            span=self._perm_span, min_periods=1
        ).mean().values

        decay_ratio = transient / (np.abs(perm_ema) + 1e-12)

        result = pd.DataFrame({
            "alg_transient_impact": transient,
            "alg_permanent_impact": perm_ema,
            "alg_impact_decay_ratio": decay_ratio,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

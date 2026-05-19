"""
Bipower Variation Jump Decomposition (BNS Estimator)
=====================================================

Mathematical Framework
----------------------
Consider log-price p_t following a jump-diffusion:

  dp_t = μ_t dt + σ_t dW_t + dJ_t

where W_t is a standard Brownian motion, σ_t is a stochastic volatility
process, and J_t is a pure-jump process (finite activity assumed here).

Given a rolling window of n log-returns r_i = ln(p_i / p_{i-1}), the
decomposition into continuous and jump variance components proceeds as follows.

Realized Variance (RV)
----------------------
  RV_n = Σ_{i=1}^{n} r_i²

RV is a consistent estimator of the quadratic variation
[p]_t = IV_t + Σ (ΔJ)²  as the sampling frequency increases.
In the presence of jumps, RV overestimates integrated variance.

Bipower Variation (BV)
-----------------------
The key insight of Barndorff-Nielsen & Shephard (2004) is that the product of
adjacent absolute returns is robust to jumps because two consecutive ticks
are very unlikely to both contain a jump.

  BV_n = (π/2) · (n/(n-1)) · Σ_{i=2}^{n} |r_i| · |r_{i-1}|

The factor π/2 = 1/μ₁² where μ₁ = E[|Z|] = √(2/π) for Z ~ N(0,1) is the
correction term that makes BV an unbiased estimator of integrated variance
IV = ∫ σ_s² ds under the null of no jumps:

  E[BV_n] → IV    as sampling interval → 0

The factor n/(n-1) is a finite-sample bias correction.

RV = BV + J Decomposition
--------------------------
  JV = max(RV - BV, 0)   (jump variation, non-negative by construction)
  JR = JV / (RV + ε)      (jump ratio ∈ [0, 1], ε = 1e-20 for stability)

JR → 0 when volatility is entirely diffusive (p_t is continuous).
JR → 1 when nearly all variance comes from jumps.

Continuous Volatility Proxy
----------------------------
  σ̂_c = √(max(BV, 0))   (per-window, in units of log-return)

This is NOT annualised in the online step(); annualisation would require
multiplying by √(ticks_per_year). At 10 Hz, ticks_per_year = 3.15 × 10⁸.

Online Update Rule
------------------
step() maintains a circular buffer of the last ``window`` log-returns.
At each new tick:
  1. Append r_t = ln(mid_t / mid_{t-1})  to the buffer (pop oldest if full).
  2. Compute RV, BV, JV, JR, σ̂_c over the buffer in O(window) time.
  3. Requires ≥ 20 observations before emitting (partial warmup guard).

Note: BV requires adjacent-pair products, so a window of n returns
yields (n-1) pairs, hence the BV denominator is (n-1).

Numerical Stability
-------------------
- JV is clamped to ≥ 0 to prevent spurious negative values from floating-
  point noise when RV ≈ BV.
- JR uses ε = 1e-20 in the denominator to avoid 0/0 when returns are zero.
- BV is clamped to ≥ 0 before taking the square root.

Output Ranges
-------------
  alg_bipower_variation  : [0, ∞), same units as r² (dimensionless squared log-return)
  alg_jump_variation     : [0, ∞)
  alg_jump_ratio         : [0, 1]
  alg_continuous_vol     : [0, ∞), units of log-return per window

Complexity: O(window) per tick (no closed-form recursion for BV cross-product sum).

Reference
---------
  Barndorff-Nielsen, O.E. & Shephard, N. (2004) — "Power and bipower variation
    with stochastic volatility and jumps", Journal of Financial Econometrics 2(1), 1–37.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

# μ₁ = E[|Z|] = sqrt(2/π) for standard normal
_MU1 = np.sqrt(2.0 / np.pi)


@register
class BipowerJump(MicrostructureAlgorithm):
    """Bipower variation: separate continuous vol from jumps."""

    def __init__(self, window: int = 100):
        self._window = window
        self._return_buffer: list[float] = []
        self._prev_mid: float = np.nan

    def name(self) -> str:
        return "bipower_jump"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_bipower_variation", warmup=100,
                             description="Bipower variation (continuous component)"),
            AlgorithmFeature("alg_jump_variation", warmup=100,
                             description="Jump variation = RV - BV"),
            AlgorithmFeature("alg_jump_ratio", warmup=100,
                             description="Jump fraction = max(0, 1 - BV/RV)"),
            AlgorithmFeature("alg_continuous_vol", warmup=100,
                             description="sqrt(BV) annualized proxy"),
        ]

    def required_columns(self) -> list[str]:
        return ["raw_midprice"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        mid = tick.get("raw_midprice", np.nan)
        if not np.isfinite(mid):
            return {f.name: np.nan for f in self.alg_features()}

        if np.isfinite(self._prev_mid) and self._prev_mid > 0:
            ret = np.log(mid / self._prev_mid)
            self._return_buffer.append(ret)
            if len(self._return_buffer) > self._window:
                self._return_buffer.pop(0)
        self._prev_mid = mid

        if len(self._return_buffer) < 20:
            return {f.name: np.nan for f in self.alg_features()}

        rets = np.array(self._return_buffer)
        n = len(rets)

        # Realized variance: sum of squared returns
        rv = np.sum(rets ** 2)

        # Bipower variation: (π/2) * sum(|r_i| * |r_{i-1}|)
        abs_rets = np.abs(rets)
        bv = (np.pi / 2.0) * np.sum(abs_rets[1:] * abs_rets[:-1]) * n / (n - 1)

        # Jump variation
        jv = max(rv - bv, 0.0)

        # Jump ratio
        jr = jv / (rv + 1e-20)

        # Continuous vol (annualized at 10Hz → 86400*10 ticks/day)
        continuous_vol = np.sqrt(max(bv, 0.0))

        return {
            "alg_bipower_variation": bv,
            "alg_jump_variation": jv,
            "alg_jump_ratio": jr,
            "alg_continuous_vol": continuous_vol,
        }

    def reset(self) -> None:
        self._return_buffer.clear()
        self._prev_mid = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        mid = df["raw_midprice"].values.astype(np.float64)
        log_ret = np.diff(np.log(mid), prepend=np.nan)
        log_ret[0] = np.nan

        n = len(df)
        abs_ret = np.abs(log_ret)

        # Rolling RV
        ret_sq = log_ret ** 2
        rv = pd.Series(ret_sq).rolling(self._window, min_periods=20).sum().values

        # Rolling BV: (π/2) * rolling_sum(|r_i| * |r_{i-1}|) * n/(n-1)
        cross = abs_ret[1:] * abs_ret[:-1]
        cross = np.concatenate([[np.nan], cross])
        bv_raw = pd.Series(cross).rolling(self._window - 1, min_periods=19).sum().values
        scale = self._window / (self._window - 1)
        bv = (np.pi / 2.0) * bv_raw * scale

        jv = np.maximum(rv - bv, 0.0)
        jr = jv / (rv + 1e-20)
        continuous_vol = np.sqrt(np.maximum(bv, 0.0))

        result = pd.DataFrame({
            "alg_bipower_variation": bv,
            "alg_jump_variation": jv,
            "alg_jump_ratio": jr,
            "alg_continuous_vol": continuous_vol,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

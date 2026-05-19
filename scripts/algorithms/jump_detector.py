"""
Lee-Mykland Nonparametric Jump Detection
==========================================

Mathematical Framework
----------------------
The Lee-Mykland (2008) test identifies jump arrivals in a continuous-time
price process by comparing each log-return to a locally-estimated diffusion
volatility.  Let:

  r_t = ln(p_t / p_{t-1})   (tick-level log-return)

Under the null hypothesis of no jump at time t, and conditioned on the
local stochastic volatility σ_{t-}:

  r_t | (no jump) ~ N(0, σ_{t-}² · Δt)

so the standardised return |r_t| / σ̂_{t-} ≈ half-normal under H₀.

Local Volatility via Bipower Variation
---------------------------------------
The critical feature of the estimator is that σ̂ must be robust to other
jumps in the window, otherwise a prior jump inflates σ̂ and masks the test
target.  The bipower-variation-based estimator achieves this:

  σ̂_BV(t) = √( (π/2) · mean_{i=2}^{K} |r_{t-i}| · |r_{t-i-1}| )

where the product of adjacent absolute returns is computed over the K-1
most recent past returns (excluding r_t itself to maintain causal computation).

The constant π/2 = 1/μ₁² corrects for the expected product of two adjacent
half-normal random variables:
  μ₁ = E[|Z|] = √(2/π),   Z ~ N(0,1)
  E[|r_i| · |r_{i-1}|] = μ₁² · σ²  →  σ² = (π/2) · E[|r_i| · |r_{i-1}|]

Lee-Mykland Test Statistic
---------------------------
  L(t) = |r_t| / σ̂_BV(t)

Under H₀ (no jump), as K → ∞:
  L(t) → |Z| / 1 = half-normal

A jump is declared when L(t) > c for a threshold c (``significance``, default 3.0).
This corresponds approximately to a 3-σ event under the normal approximation.
The exact critical value follows an extreme-value (Gumbel) distribution
under the continuous record asymptotic theory; c = 3.0 is a practical choice
for tick-frequency data.

Post-Jump Reversion
--------------------
After a detected jump at tick t_J with log-return r_J and price p_J, the
reversion signal at subsequent tick t (0 < t - t_J ≤ H) is:

  REV(t) = - ln(p_t / p_J) / r_J

Interpretation:
  REV > 0: price has moved against the jump direction (reversion occurring).
  REV < 0: price has continued in the jump direction (momentum).
  REV = 0: outside the reversion horizon H (``reversion_horizon``, default 50 ticks).

The negation is by convention: +1 ≡ "fully reverted", so that a mean-
reversion trading signal is directly readable from the sign.

Numerical Stability
-------------------
- σ̂_BV uses a floor of 1e-20 to prevent L → ∞ during flat periods.
- The reversion denominator r_J + 1e-20 guards against zero-return jumps
  (e.g. if a tick spuriously triggers detection at r_J ≈ 0).
- run_batch() uses a floor of 1e-40 inside the rolling mean before sqrt.

Parameters
----------
  window              (K) : local volatility estimation window (ticks, default 100)
  significance        (c) : detection threshold for L(t) (default 3.0)
  reversion_horizon   (H) : ticks post-jump over which reversion is tracked (default 50)

Output Ranges
-------------
  alg_jump_statistic      : [0, ∞), dimensionless
  alg_jump_detected       : {0.0, 1.0}
  alg_jump_magnitude      : ℝ (signed log-return at jump; 0 if no jump this tick)
  alg_post_jump_reversion : ℝ (dimensionless; 0 outside horizon or before first jump)

Complexity: O(K) per tick for the rolling BV mean.

Reference
---------
  Lee, S.S. & Mykland, P.A. (2008) — "Jumps in financial markets: a new
    nonparametric test and jump dynamics", Review of Financial Studies 21(6),
    2535–2563.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class JumpDetector(MicrostructureAlgorithm):
    """Lee-Mykland jump test with post-jump reversion tracking."""

    def __init__(self, window: int = 100, significance: float = 3.0,
                 reversion_horizon: int = 50):
        self._window = window
        self._significance = significance
        self._reversion_horizon = reversion_horizon
        self._return_buffer: list[float] = []
        self._prev_mid: float = np.nan
        # Track last jump for reversion measurement
        self._last_jump_tick: int = -9999
        self._last_jump_mag: float = 0.0
        self._last_jump_price: float = np.nan
        self._tick_count: int = 0

    def name(self) -> str:
        return "jump_detector"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_jump_statistic", warmup=100,
                             description="Lee-Mykland L(t) = |return| / local_vol"),
            AlgorithmFeature("alg_jump_detected", warmup=100,
                             description="1.0 if L(t) > significance threshold"),
            AlgorithmFeature("alg_jump_magnitude", warmup=100,
                             description="Signed return at jump (0 if no jump)"),
            AlgorithmFeature("alg_post_jump_reversion", warmup=100,
                             description="Price change since last jump (reversion signal)"),
        ]

    def required_columns(self) -> list[str]:
        return ["raw_midprice"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        mid = tick.get("raw_midprice", np.nan)
        if not np.isfinite(mid):
            return {f.name: np.nan for f in self.alg_features()}

        self._tick_count += 1

        # Compute log return
        if np.isfinite(self._prev_mid) and self._prev_mid > 0:
            ret = np.log(mid / self._prev_mid)
        else:
            self._prev_mid = mid
            return {f.name: np.nan for f in self.alg_features()}

        self._prev_mid = mid

        self._return_buffer.append(ret)
        if len(self._return_buffer) > self._window:
            self._return_buffer.pop(0)

        if len(self._return_buffer) < 20:
            return {f.name: np.nan for f in self.alg_features()}

        # Local bipower volatility (more robust to jumps than std)
        rets = np.array(self._return_buffer[:-1])  # exclude current
        abs_rets = np.abs(rets)
        if len(abs_rets) > 1:
            bv = np.sqrt(np.pi / 2.0 * np.mean(abs_rets[1:] * abs_rets[:-1]))
        else:
            bv = np.std(rets) + 1e-20

        bv = max(bv, 1e-20)

        # Lee-Mykland statistic
        L = abs(ret) / bv
        detected = 1.0 if L > self._significance else 0.0
        magnitude = ret if detected else 0.0

        # Track jump for reversion
        if detected:
            self._last_jump_tick = self._tick_count
            self._last_jump_mag = ret
            self._last_jump_price = mid

        # Post-jump reversion: price change since last jump
        ticks_since = self._tick_count - self._last_jump_tick
        if (ticks_since > 0 and ticks_since <= self._reversion_horizon
                and np.isfinite(self._last_jump_price)):
            reversion = np.log(mid / self._last_jump_price) / (self._last_jump_mag + 1e-20)
            # Negative reversion means price reverted from jump direction
            reversion = -reversion  # flip so positive = reversion happening
        else:
            reversion = 0.0

        return {
            "alg_jump_statistic": L,
            "alg_jump_detected": detected,
            "alg_jump_magnitude": magnitude,
            "alg_post_jump_reversion": reversion,
        }

    def reset(self) -> None:
        self._return_buffer.clear()
        self._prev_mid = np.nan
        self._last_jump_tick = -9999
        self._last_jump_mag = 0.0
        self._last_jump_price = np.nan
        self._tick_count = 0

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        mid = df["raw_midprice"].values.astype(np.float64)
        n = len(df)
        log_ret = np.diff(np.log(mid), prepend=np.nan)
        log_ret[0] = np.nan
        abs_ret = np.abs(log_ret)

        # Rolling bipower vol (excluding current tick)
        cross = abs_ret[1:] * abs_ret[:-1]
        cross = np.concatenate([[np.nan], cross])
        bv_sq = pd.Series(cross).rolling(self._window - 1, min_periods=19).mean().values
        bv = np.sqrt(np.pi / 2.0 * np.maximum(bv_sq, 1e-40))

        L = np.abs(log_ret) / (bv + 1e-20)
        detected = (L > self._significance).astype(np.float64)
        magnitude = np.where(detected > 0, log_ret, 0.0)

        # Post-jump reversion (vectorized forward-fill of jump events)
        reversion = np.zeros(n)
        last_jump_idx = -9999
        last_jump_mag = 0.0
        last_jump_price = np.nan
        for i in range(n):
            if detected[i]:
                last_jump_idx = i
                last_jump_mag = log_ret[i]
                last_jump_price = mid[i]
            ticks_since = i - last_jump_idx
            if (0 < ticks_since <= self._reversion_horizon
                    and np.isfinite(last_jump_price) and abs(last_jump_mag) > 1e-20):
                reversion[i] = -np.log(mid[i] / last_jump_price) / last_jump_mag

        L[0] = np.nan
        detected[0] = np.nan

        result = pd.DataFrame({
            "alg_jump_statistic": L,
            "alg_jump_detected": detected,
            "alg_jump_magnitude": magnitude,
            "alg_post_jump_reversion": reversion,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

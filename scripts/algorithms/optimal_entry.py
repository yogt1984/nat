"""
Optimal Entry via Sequential Probability Ratio Test (SPRT)
============================================================

Mathematical Framework
----------------------
The SPRT (Wald 1947) is a sequential hypothesis test that makes a minimum
expected-sample-size decision between two simple hypotheses.  Here it is
applied to the innovation sequence produced by a Kalman OU filter.

Hypothesis Setup
----------------
Let ν_t = z_t - ẑ_{t|t-1} denote the one-step-ahead Kalman innovation, and
let σ̂²(t) be an EMA estimate of the innovation variance.

  H₀ : ν_t ~ N(0, σ̂²)          (no drift — no entry signal)
  H₁ : ν_t ~ N(μ, σ̂²)          (mean drift μ > 0 — signal present)

where μ = sprt_drift (default 0.001) is the minimum detectable drift.

Log-Likelihood Ratio Increment
--------------------------------
The per-step LLR of ν_t under H₁ vs H₀ is:

  Λ_t = log( f₁(ν_t) / f₀(ν_t) )
       = log( exp(-(ν_t - μ)² / (2σ̂²)) / exp(-ν_t² / (2σ̂²)) )
       = (μ / σ̂²) · ν_t  -  μ² / (2σ̂²)

This closed form follows directly from the ratio of two Gaussian densities
with equal variance.  No approximation is required.

Cumulative Test Statistic
--------------------------
The SPRT accumulates evidence as a running sum:

  S_n = Σ_{t=1}^{n} Λ_t

S_n is updated recursively at each tick:  S_n = S_{n-1} + Λ_t  (O(1)).

Decision Boundaries
-------------------
Given Type I error rate α (false entry) and Type II error rate β (missed entry),
Wald's optimal boundaries are:

  A = log((1 - β) / α)   — upper boundary (accept H₁, fire entry signal)
  B = log(β / (1 - α))   — lower boundary (accept H₀, no entry)

with A > 0 > B.  After a decision is reached (|S_n| ≥ A or S_n ≤ B), S_n is
reset to 0 to allow the test to restart (continuous monitoring).

Default parameters α = 0.05, β = 0.20 give:
  A ≈ log(0.80/0.05) = log(16) ≈ 2.77
  B ≈ log(0.20/0.95) ≈ -1.55

Entry Direction
---------------
When S_n ≥ A, the direction is determined by the sign of the most recent innovation:

  entry = +1  if ν_t > 0  (drift upward)
  entry = -1  if ν_t < 0  (drift downward)

Innovation Variance Estimation
--------------------------------
σ̂² is tracked with an EMA with smoothing factor α_ema = 0.02:

  σ̂²(t) = α_ema · ν_t² + (1 - α_ema) · σ̂²(t-1)

Initial condition: σ̂²(t_0) = ν_{t_0}² (seeded at first valid tick).
The floor σ̂² ≥ 1e-20 prevents numerical overflow in Λ_t.

The Kalman Filter Layer
------------------------
The innovation ν_t is produced by an OUKalmanFilter that tracks a latent OU
process.  This pre-whitening step is essential: the SPRT is valid only for
i.i.d. observations.  The Kalman filter removes the OU autocorrelation so that
the residual innovations ν_t are approximately i.i.d. N(0, σ̂²) under H₀.

Note: σ² in the LLR formula above uses the EMA-estimated innovation variance,
not the filter's posterior variance P, because P converges to a small steady-
state value that underestimates the effective noise in the raw signal.

Numerical Stability
-------------------
- A + ε in the evidence normalisation prevents 0/0 when A = 0 exactly.
- After each decision, S is zeroed; this is mathematically equivalent to
  treating each inter-decision interval as an independent SPRT.

Parameters
----------
  theta          : OU mean-reversion speed for the Kalman layer (default 0.1)
  sigma_process  : process noise (default 0.01)
  sigma_obs      : observation noise (default 0.1)
  dt             : tick size in seconds (default 0.1)
  sprt_drift (μ) : minimum detectable drift under H₁ (default 0.001)
  alpha_error (α): Type I error probability (default 0.05)
  beta_error  (β): Type II error probability (default 0.20)

Output Ranges
-------------
  alg_sprt_statistic      : ℝ (S_n, resets to 0 at each decision)
  alg_entry_signal        : {-1.0, 0.0, +1.0}
  alg_cumulative_evidence : [0, ∞), normalised as |S_n| / A

Complexity: O(1) per tick.

References
----------
  Wald, A. (1947) — "Sequential Analysis", Wiley, New York.
  Shiryaev, A.N. (1978) — "Optimal Stopping Rules", Springer.
  du Toit, J. & Peskir, G. (2009) — "Selling a stock at the ultimate maximum",
    Annals of Applied Probability 19(3), 983–1014.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalman.ou_filter import OUKalmanFilter


@register
class OptimalEntry(MicrostructureAlgorithm):
    """SPRT on Kalman innovation for optimal entry timing.

    H0: innovation ~ N(0, σ²)     (no signal)
    H1: innovation ~ N(μ, σ²)     (signal present, drift = sprt_drift)

    Decision boundaries:
      A = log((1-β)/α)   — accept H1 (enter)
      B = log(β/(1-α))   — accept H0 (no entry)

    Cumulative evidence: S_n = Σ log(L1/L0) for each innovation
    """

    def __init__(self, theta: float = 0.1, sigma_process: float = 0.01,
                 sigma_obs: float = 0.1, dt: float = 0.1,
                 sprt_drift: float = 0.001,
                 alpha_error: float = 0.05, beta_error: float = 0.20):
        self._theta = theta
        self._sigma_obs = sigma_obs
        self._dt = dt
        self._sprt_drift = sprt_drift

        # Decision boundaries
        self._A = np.log((1 - beta_error) / (alpha_error + 1e-20))
        self._B = np.log((beta_error + 1e-20) / (1 - alpha_error))

        # Internal Kalman filter
        self._kf = OUKalmanFilter(
            theta=theta, sigma_process=sigma_process,
            sigma_obs=sigma_obs, dt=dt)

        # SPRT state
        self._S = 0.0  # cumulative log-likelihood ratio
        self._innov_var_ema = np.nan  # running estimate of innovation variance
        self._ema_alpha = 0.02

    def name(self) -> str:
        return "optimal_entry"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_sprt_statistic", warmup=50,
                             description="SPRT cumulative log-likelihood ratio"),
            AlgorithmFeature("alg_entry_signal", warmup=50,
                             description="+1 enter long, -1 enter short, 0 no signal"),
            AlgorithmFeature("alg_cumulative_evidence", warmup=50,
                             description="Normalized |S| / A — 1.0 = decision threshold"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        z = tick.get("imbalance_qty_l1", np.nan)
        if not np.isfinite(z):
            return {f.name: np.nan for f in self.alg_features()}

        # Run Kalman filter
        x, P, innov = self._kf.step(z)

        # Estimate innovation variance with EMA
        if np.isnan(self._innov_var_ema):
            self._innov_var_ema = innov ** 2
        else:
            self._innov_var_ema = (self._ema_alpha * innov ** 2 +
                                   (1 - self._ema_alpha) * self._innov_var_ema)

        sigma2 = max(self._innov_var_ema, 1e-20)
        mu = self._sprt_drift

        # Log-likelihood ratio increment:
        # log(L1/L0) = (μ/σ²)*innov - μ²/(2σ²)
        # where innov is the observation under H0
        llr = (mu / sigma2) * innov - (mu ** 2) / (2 * sigma2)

        self._S += llr

        # Decision
        signal = 0.0
        if self._S >= self._A:
            signal = np.sign(innov) if innov != 0 else 1.0
            self._S = 0.0  # reset after decision
        elif self._S <= self._B:
            self._S = 0.0  # reset, no signal

        # Normalized evidence
        evidence = abs(self._S) / (abs(self._A) + 1e-12)

        return {
            "alg_sprt_statistic": self._S,
            "alg_entry_signal": signal,
            "alg_cumulative_evidence": evidence,
        }

    def reset(self) -> None:
        self._kf.reset()
        self._S = 0.0
        self._innov_var_ema = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Loop-based batch (SPRT is inherently sequential)."""
        import pandas as pd

        obs = df["imbalance_qty_l1"].values.astype(np.float64)
        n = len(df)

        stats = np.full(n, np.nan)
        signals = np.full(n, np.nan)
        evidence = np.full(n, np.nan)

        kf = OUKalmanFilter(
            theta=self._theta, sigma_process=0.01,
            sigma_obs=self._sigma_obs, dt=self._dt)

        S = 0.0
        var_ema = np.nan
        mu = self._sprt_drift
        ema_a = self._ema_alpha

        for i in range(n):
            z = obs[i]
            if not np.isfinite(z):
                continue

            x, P, innov = kf.step(z)

            if np.isnan(var_ema):
                var_ema = innov ** 2
            else:
                var_ema = ema_a * innov ** 2 + (1 - ema_a) * var_ema

            sigma2 = max(var_ema, 1e-20)
            llr = (mu / sigma2) * innov - (mu ** 2) / (2 * sigma2)
            S += llr

            sig = 0.0
            if S >= self._A:
                sig = np.sign(innov) if innov != 0 else 1.0
                S = 0.0
            elif S <= self._B:
                S = 0.0

            stats[i] = S
            signals[i] = sig
            evidence[i] = abs(S) / (abs(self._A) + 1e-12)

        result = pd.DataFrame({
            "alg_sprt_statistic": stats,
            "alg_entry_signal": signals,
            "alg_cumulative_evidence": evidence,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

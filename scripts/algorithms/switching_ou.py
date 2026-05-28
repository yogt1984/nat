"""
Switching Ornstein-Uhlenbeck Process with Bayesian Regime Filtering
====================================================================

Mathematical Framework
----------------------
The observable z_t follows a hidden Markov model where each hidden state S_t
∈ {0 (fast), 1 (slow)} selects an OU dynamics:

  dz_t = -θ_{S_t} · z_t · dt + σ_proc · dW_t   (continuous-time OU per regime)

In the discretised form (Euler-Maruyama with step Δt = 0.1 s):

  z_{t+1} = e^{-θ · Δt} · z_t + η_t,   η_t ~ N(0, σ_proc² · (1 - e^{-2θΔt}) / (2θ))

Two parallel Kalman filters (OUKalmanFilter) track:
  Regime 0 (fast): θ₀ = 0.5  →  strong mean-reversion, range-bound market
  Regime 1 (slow): θ₁ = 0.05 →  weak mean-reversion,  trending market

Each filter maintains its own posterior mean x̂_k and variance P_k.

Transition Matrix
-----------------
The discrete-time regime Markov chain has transition matrix:

  Π = [[1 - ρ,  ρ ],
       [ρ,   1 - ρ]]

where ρ = transition_rate (default 0.01) is the per-tick switching probability.
Diagonal elements give the probability of staying in the current regime.

Bayesian Filtering Update (Hamilton Filter)
-------------------------------------------
At each tick, the algorithm performs:

  Step 1 — Predict (prior):
    π̃_0(t) = Π[0,0] · π_0(t-1) + Π[1,0] · π_1(t-1)
    π̃_1(t) = 1 - π̃_0(t)

  Step 2 — Kalman update in each regime:
    (x̂_k, P_k, ν_k) = KalmanStep_k(z_t)
    where ν_k = z_t - x̂_k^{-} is the innovation

  Step 3 — Likelihood (Gaussian observation model):
    f_k(z_t) = N(ν_k ; 0, P_k + ε)
             = (2π(P_k+ε))^{-1/2} · exp(-ν_k² / (2(P_k+ε)))

  Step 4 — Posterior (Bayes rule):
    π_0(t) = f_0 · π̃_0 / (f_0 · π̃_0 + f_1 · π̃_1)

The denominator is the predictive likelihood of the observation.
ε = 1e-12 ensures numerical stability when P_k → 0.

Regime-Weighted Outputs
------------------------
The filtered state estimate is the probability-weighted mixture of the two
Kalman estimates:

  State(t)  = π_0(t) · x̂_0(t) + (1 - π_0(t)) · x̂_1(t)
  Speed(t)  = π_0(t) · θ_0    + (1 - π_0(t)) · θ_1
  TRate(t)  = |π_0(t) - π_0(t-1)| / Δt   (instantaneous regime-switching rate)

Interpretation:
  π_0 ≈ 1.0  →  fast regime (mean-reverting): fade extremes
  π_0 ≈ 0.0  →  slow regime (trending): follow momentum

Numerical Stability
-------------------
- Likelihoods can underflow to zero for highly improbable observations;
  the denominator total += 1e-20 prevents 0/0 in the posterior.
- Both Kalman filters are initialised fresh in run_batch() to avoid
  contamination from any previous online step() calls.

Parameters
----------
  theta_fast        (θ₀) : mean-reversion speed, fast regime (default 0.5)
  theta_slow        (θ₁) : mean-reversion speed, slow regime (default 0.05)
  transition_rate   (ρ)  : per-tick regime-switch probability (default 0.01)
  dt                     : tick size in seconds (default 0.1)

Output Ranges
-------------
  alg_switching_ou_state       : ℝ, same units as input (normalised imbalance)
  alg_switching_ou_regime      : [0, 1] — P(fast regime)
  alg_switching_ou_speed       : [θ₁, θ₀] — effective mean-reversion speed
  alg_regime_transition_rate   : [0, ∞), per-second

Complexity: O(1) per tick (two parallel Kalman steps + 4 scalar multiplies).

References
----------
  Elliott, R.J., Aggoun, L. & Moore, J.B. (2005) — "Hidden Markov Models:
    Estimation and Control", Springer, 2nd ed. (Chapter 3: filtering).
  Hamilton, J.D. (1989) — "A new approach to the economic analysis of
    nonstationary time series and the business cycle", Econometrica 57(2),
    357–384.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

# Import existing Kalman filter
from kalman.ou_filter import OUKalmanFilter


@register
class SwitchingOU(MicrostructureAlgorithm):
    """Two-regime OU with Bayesian regime switching.

    Regime 0: fast mean-reversion (high θ) — range-bound market
    Regime 1: slow mean-reversion (low θ) — trending market
    """

    def __init__(self, theta_fast: float = 0.5, theta_slow: float = 0.05,
                 transition_rate: float = 0.01, dt: float = 0.1):
        self._theta_fast = theta_fast
        self._theta_slow = theta_slow
        self._transition_rate = transition_rate
        self._dt = dt

        # Two parallel Kalman filters
        self._kf_fast = OUKalmanFilter(
            theta=theta_fast, sigma_process=0.02, sigma_obs=0.1, dt=dt)
        self._kf_slow = OUKalmanFilter(
            theta=theta_slow, sigma_process=0.02, sigma_obs=0.1, dt=dt)

        # Regime probability: P(fast regime)
        self._p_fast = 0.5

        # Transition matrix: [[stay_fast, switch_to_slow],
        #                      [switch_to_fast, stay_slow]]
        self._trans = np.array([
            [1 - transition_rate, transition_rate],
            [transition_rate, 1 - transition_rate],
        ])

    def name(self) -> str:
        return "switching_ou"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_switching_ou_state", warmup=50,
                             description="Regime-weighted filtered state"),
            AlgorithmFeature("alg_switching_ou_regime", warmup=50,
                             description="P(fast regime) — 1.0=mean-reverting, 0.0=trending"),
            AlgorithmFeature("alg_switching_ou_speed", warmup=50,
                             description="Effective θ (regime-weighted mean-reversion speed)"),
            AlgorithmFeature("alg_regime_transition_rate", warmup=100,
                             description="Estimated transition rate (dP/dt)"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        z = tick.get("imbalance_qty_l1", np.nan)
        if not np.isfinite(z):
            return {f.name: np.nan for f in self.alg_features()}

        p_fast_prev = self._p_fast

        # Predict step for each regime
        x_fast, P_fast, innov_fast = self._kf_fast.step(z)
        x_slow, P_slow, innov_slow = self._kf_slow.step(z)

        # Likelihoods: Gaussian with variance = P + R
        var_fast = P_fast + 1e-12
        var_slow = P_slow + 1e-12
        ll_fast = np.exp(-0.5 * innov_fast**2 / var_fast) / np.sqrt(var_fast)
        ll_slow = np.exp(-0.5 * innov_slow**2 / var_slow) / np.sqrt(var_slow)

        # Prior via transition matrix
        prior_fast = (self._trans[0, 0] * self._p_fast +
                      self._trans[1, 0] * (1 - self._p_fast))
        prior_slow = 1 - prior_fast

        # Posterior via Bayes
        numerator_fast = ll_fast * prior_fast
        numerator_slow = ll_slow * prior_slow
        total = numerator_fast + numerator_slow + 1e-20

        self._p_fast = numerator_fast / total

        # Regime-weighted outputs
        state = self._p_fast * x_fast + (1 - self._p_fast) * x_slow
        speed = self._p_fast * self._theta_fast + (1 - self._p_fast) * self._theta_slow
        transition_rate = abs(self._p_fast - p_fast_prev) / self._dt

        return {
            "alg_switching_ou_state": state,
            "alg_switching_ou_regime": self._p_fast,
            "alg_switching_ou_speed": speed,
            "alg_regime_transition_rate": transition_rate,
        }

    def reset(self) -> None:
        self._kf_fast.reset()
        self._kf_slow.reset()
        self._p_fast = 0.5

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Loop-based batch (recursive state prevents vectorization)."""
        import pandas as pd

        obs = df["imbalance_qty_l1"].values.astype(np.float64)
        n = len(df)

        states = np.full(n, np.nan)
        regimes = np.full(n, np.nan)
        speeds = np.full(n, np.nan)
        transitions = np.full(n, np.nan)

        # Fresh filters for batch
        kf_fast = OUKalmanFilter(
            theta=self._theta_fast, sigma_process=0.02,
            sigma_obs=0.1, dt=self._dt)
        kf_slow = OUKalmanFilter(
            theta=self._theta_slow, sigma_process=0.02,
            sigma_obs=0.1, dt=self._dt)
        p_fast = 0.5

        for i in range(n):
            z = obs[i]
            if not np.isfinite(z):
                continue

            p_prev = p_fast

            x_f, P_f, inn_f = kf_fast.step(z)
            x_s, P_s, inn_s = kf_slow.step(z)

            var_f = P_f + 1e-12
            var_s = P_s + 1e-12
            ll_f = np.exp(-0.5 * inn_f**2 / var_f) / np.sqrt(var_f)
            ll_s = np.exp(-0.5 * inn_s**2 / var_s) / np.sqrt(var_s)

            prior_f = (self._trans[0, 0] * p_fast +
                       self._trans[1, 0] * (1 - p_fast))

            num_f = ll_f * prior_f
            num_s = ll_s * (1 - prior_f)
            total = num_f + num_s + 1e-20
            p_fast = num_f / total

            states[i] = p_fast * x_f + (1 - p_fast) * x_s
            regimes[i] = p_fast
            speeds[i] = p_fast * self._theta_fast + (1 - p_fast) * self._theta_slow
            transitions[i] = abs(p_fast - p_prev) / self._dt

        result = pd.DataFrame({
            "alg_switching_ou_state": states,
            "alg_switching_ou_regime": regimes,
            "alg_switching_ou_speed": speeds,
            "alg_regime_transition_rate": transitions,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

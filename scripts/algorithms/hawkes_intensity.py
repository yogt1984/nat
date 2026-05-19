"""
Hawkes Self-Exciting Point Process Intensity
=============================================

Mathematical Framework
----------------------
A univariate Hawkes process on the real line has conditional intensity:

  λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

where the sum runs over all past event times t_i.  The three parameters are:
  μ  : baseline (background) intensity  [events / unit time]
  α  : self-excitation coefficient (jump size in intensity per event)
  β  : exponential decay rate  [1 / unit time]

Stationarity requires α < β (branching ratio α/β < 1).

Recursive (Online) Intensity Update
------------------------------------
Define the auxiliary process A(t) as the running sum of past excitation:

  A(t) = Σ_{t_i < t} exp(-β · (t - t_i))

Between two consecutive observation times t_{n-1} and t_n separated by Δt:

  Step 1 — Decay existing excitation:
    A_decayed = exp(-β · Δt) · A(t_{n-1})

  Step 2 — Add new events N(t_n) arriving at time t_n:
    A(t_n) = A_decayed + N(t_n)

  Step 3 — Compute intensity:
    λ(t_n) = μ + α · A(t_n)

Here Δt = 0.1 s (100 ms tick grid, ``_dt``), so the discrete recursion is:

  A(t_n) = exp(-β · 0.1) · A(t_{n-1}) + N_n

This is implemented in step() for A_total, A_bid, A_ask simultaneously.
A hard cap (A_total ≤ 1000, A_bid/A_ask ≤ 500) prevents overflow during
burst periods.

Bid/Ask Decomposition
----------------------
Directional excitation uses the pressure proxy:

  bid_frac(t) = |P^b(t)| / (|P^b(t)| + |P^a(t)| + ε)
  ask_frac(t) = 1 - bid_frac(t)

where P^b, P^a = ``imbalance_pressure_bid/ask``.  Event counts are then split:

  N^bid_n = N_n · bid_frac_n
  N^ask_n = N_n · ask_frac_n

giving separate intensities:

  λ^bid(t) = μ/2 + α · A^bid(t)
  λ^ask(t) = μ/2 + α · A^ask(t)

The μ/2 term splits the baseline equally between sides.

Baseline Intensity
------------------
μ(t) is estimated as a rolling mean over a long window (baseline_window = 300 ticks):

  μ(t) = (1/W_μ) · Σ_{s=t-W_μ+1}^{t} λ_raw(s)

where λ_raw = ``flow_intensity`` from the upstream pipeline.

Excitement Fraction
-------------------
  φ(t) = α · A(t) / λ(t)  ∈ [0, 1)

This is the fraction of current intensity attributable to self-excitation
(as opposed to background). φ → 1 indicates a burst / cascade.

Bid/Ask Hawkes Imbalance
-------------------------
  HI(t) = (λ^ask - λ^bid) / (λ^ask + λ^bid + ε)  ∈ (-1, 1)

Positive: ask-side arrival rate exceeds bid-side → upward pressure.
Negative: bid-side dominates → downward pressure.

Numerical Stability
-------------------
- ε = 1e-12 in denominator guards against both sides going to zero.
- Hard caps on A prevent intensity from diverging if α/β ≥ 1.
- Baseline falls back to the raw tick intensity when buffer < 10 points.

Parameters
----------
  baseline_window  : window for rolling μ estimate (ticks, default 300)
  decay_beta (β)   : exponential decay rate (1/s, default 0.1 → half-life ≈ 6.9 s)
  alpha_fraction(α): self-excitation amplitude (default 0.5)

Output Ranges
-------------
  alg_hawkes_intensity          : [0, ∞), events/s
  alg_hawkes_baseline           : [0, ∞), events/s
  alg_hawkes_excitement         : [0, 1)
  alg_bid_ask_hawkes_imbalance  : (-1, 1)

Complexity: O(1) per tick (recursive update) after O(W_μ) baseline initialisation.

References
----------
  Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015) — "Hawkes processes in
    finance", Market Microstructure and Liquidity 1(1), 1550005.
  Lu, X. & Abergel, F. (2018) — "High-dimensional Hawkes processes for limit
    order books", Quantitative Finance 18(2), 177–188.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class HawkesIntensity(MicrostructureAlgorithm):
    """Self-exciting intensity model with bid/ask decomposition.

    Recursive intensity: λ(t) = μ + α * A(t)
    where A(t) = exp(-β*Δt) * (A(t-1) + N(t))
    and N(t) is the number of new events at time t.
    """

    def __init__(self, baseline_window: int = 300, decay_beta: float = 0.1,
                 alpha_fraction: float = 0.5):
        self._baseline_window = baseline_window
        self._beta = decay_beta
        self._alpha = alpha_fraction
        # Recursive state
        self._A_total = 0.0
        self._A_bid = 0.0
        self._A_ask = 0.0
        self._baseline_buffer: list[float] = []
        self._tick_count = 0
        self._dt = 0.1  # 100ms ticks

    def name(self) -> str:
        return "hawkes_intensity"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_hawkes_intensity", warmup=50,
                             description="Self-exciting trade intensity λ(t)"),
            AlgorithmFeature("alg_hawkes_baseline", warmup=300,
                             description="Baseline intensity μ (rolling average)"),
            AlgorithmFeature("alg_hawkes_excitement", warmup=50,
                             description="Excitement component α*A(t) / λ(t)"),
            AlgorithmFeature("alg_bid_ask_hawkes_imbalance", warmup=50,
                             description="(λ_ask - λ_bid) / (λ_ask + λ_bid)"),
        ]

    def required_columns(self) -> list[str]:
        return ["flow_count_1s", "flow_intensity",
                "imbalance_pressure_bid", "imbalance_pressure_ask"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        count = tick.get("flow_count_1s", np.nan)
        intensity = tick.get("flow_intensity", np.nan)
        p_bid = tick.get("imbalance_pressure_bid", np.nan)
        p_ask = tick.get("imbalance_pressure_ask", np.nan)

        if not all(np.isfinite(x) for x in [count, intensity, p_bid, p_ask]):
            return {f.name: np.nan for f in self.alg_features()}

        self._tick_count += 1

        # Approximate event counts on bid/ask side from pressure ratio
        total_pressure = abs(p_bid) + abs(p_ask) + 1e-12
        bid_frac = abs(p_bid) / total_pressure
        ask_frac = abs(p_ask) / total_pressure

        # Event proxy: flow_count_1s scaled by intensity
        n_events = max(count, 0)
        n_bid = n_events * bid_frac
        n_ask = n_events * ask_frac

        # Recursive update: A(t) = exp(-β*dt) * (A(t-1) + n_events)
        decay = np.exp(-self._beta * self._dt)
        self._A_total = min(decay * (self._A_total + n_events), 1000.0)
        self._A_bid = min(decay * (self._A_bid + n_bid), 500.0)
        self._A_ask = min(decay * (self._A_ask + n_ask), 500.0)

        # Baseline: rolling mean of raw intensity
        self._baseline_buffer.append(intensity)
        if len(self._baseline_buffer) > self._baseline_window:
            self._baseline_buffer.pop(0)

        if len(self._baseline_buffer) < 10:
            mu = intensity
        else:
            mu = np.mean(self._baseline_buffer)

        # Total intensity
        lambda_total = mu + self._alpha * self._A_total
        lambda_bid = mu / 2 + self._alpha * self._A_bid
        lambda_ask = mu / 2 + self._alpha * self._A_ask

        # Excitement fraction
        excitement = (self._alpha * self._A_total) / (lambda_total + 1e-12)

        # Bid/ask imbalance in Hawkes intensity
        hawkes_imb = ((lambda_ask - lambda_bid) /
                      (lambda_ask + lambda_bid + 1e-12))

        return {
            "alg_hawkes_intensity": lambda_total,
            "alg_hawkes_baseline": mu,
            "alg_hawkes_excitement": excitement,
            "alg_bid_ask_hawkes_imbalance": hawkes_imb,
        }

    def reset(self) -> None:
        self._A_total = 0.0
        self._A_bid = 0.0
        self._A_ask = 0.0
        self._baseline_buffer.clear()
        self._tick_count = 0

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized via loop (recursive state prevents full vectorization)."""
        import pandas as pd

        count = df["flow_count_1s"].values.astype(np.float64)
        intensity = df["flow_intensity"].values.astype(np.float64)
        p_bid = df["imbalance_pressure_bid"].values.astype(np.float64)
        p_ask = df["imbalance_pressure_ask"].values.astype(np.float64)

        n = len(df)
        decay = np.exp(-self._beta * self._dt)

        # Pre-compute bid/ask fractions
        total_p = np.abs(p_bid) + np.abs(p_ask) + 1e-12
        bid_frac = np.abs(p_bid) / total_p
        ask_frac = np.abs(p_ask) / total_p
        n_events = np.maximum(count, 0)

        # Recursive loop
        A_total = np.zeros(n)
        A_bid = np.zeros(n)
        A_ask = np.zeros(n)
        a_t, a_b, a_a = 0.0, 0.0, 0.0

        for i in range(n):
            if np.isfinite(n_events[i]):
                a_t = min(decay * (a_t + n_events[i]), 1000.0)
                a_b = min(decay * (a_b + n_events[i] * bid_frac[i]), 500.0)
                a_a = min(decay * (a_a + n_events[i] * ask_frac[i]), 500.0)
            A_total[i] = a_t
            A_bid[i] = a_b
            A_ask[i] = a_a

        # Baseline via rolling mean
        mu = pd.Series(intensity).rolling(
            self._baseline_window, min_periods=10
        ).mean().values

        lambda_total = mu + self._alpha * A_total
        lambda_bid = mu / 2 + self._alpha * A_bid
        lambda_ask = mu / 2 + self._alpha * A_ask

        excitement = (self._alpha * A_total) / (lambda_total + 1e-12)
        hawkes_imb = (lambda_ask - lambda_bid) / (lambda_ask + lambda_bid + 1e-12)

        result = pd.DataFrame({
            "alg_hawkes_intensity": lambda_total,
            "alg_hawkes_baseline": mu,
            "alg_hawkes_excitement": excitement,
            "alg_bid_ask_hawkes_imbalance": hawkes_imb,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

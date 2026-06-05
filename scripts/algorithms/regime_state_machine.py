"""
Regime State Machine — 6-State Manual Threshold Classifier (#4)
================================================================

Classifies market microstructure into 6 discrete regimes using
feature-based scoring. Acts as a GATING mechanism for downstream
algorithms: momentum (#1) in trending, mean-reversion (#2) in ranging,
no trade in volatile noise.

State Space:
  0: ACCUMULATION    — whale buying at support, low entropy
  1: DISTRIBUTION    — whale selling into strength, low entropy
  2: TRENDING_UP     — persistent upward momentum, whale alignment
  3: TRENDING_DOWN   — persistent downward momentum, whale alignment
  4: RANGING         — two-sided flow, high entropy, anti-persistent
  5: VOLATILE_NOISE  — high vol, high toxicity, no trade

Scoring: count conditions met per state, state = argmax(scores).
Tie-break favors VOLATILE_NOISE (conservative).
Hysteresis: state held for min_duration bars before transition allowed.

Output Features (4):
  alg_rsm_regime            {0..5}   Discrete regime label
  alg_rsm_confidence        [0, 1]   max_score / total_score
  alg_rsm_transition_risk   [0, 1]   Decaying function of regime age
  alg_rsm_trade_allowed     {0, 1}   0 if VOLATILE_NOISE, 1 otherwise

References:
  Hamilton (1989) — A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle, Econometrica.
  Wyckoff (1931) — The Richard D. Wyckoff Method of Trading and
      Investing in Stocks.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


# State indices
ACCUMULATION = 0
DISTRIBUTION = 1
TRENDING_UP = 2
TRENDING_DOWN = 3
RANGING = 4
VOLATILE_NOISE = 5

STATE_NAMES = [
    "ACCUMULATION", "DISTRIBUTION", "TRENDING_UP",
    "TRENDING_DOWN", "RANGING", "VOLATILE_NOISE",
]


@register
class RegimeStateMachine(MicrostructureAlgorithm):
    """6-state regime classifier using manual feature thresholds."""

    bar_level = True

    def __init__(
        self,
        accum_thresh: float = 0.70,
        distrib_thresh: float = 0.70,
        entropy_low: float = 0.30,
        entropy_high: float = 0.60,
        hurst_trend: float = 0.55,
        vol_noise_mult: float = 2.0,
        min_duration: int = 5,
    ):
        self._accum_thresh = accum_thresh
        self._distrib_thresh = distrib_thresh
        self._entropy_low = entropy_low
        self._entropy_high = entropy_high
        self._hurst_trend = hurst_trend
        self._vol_noise_mult = vol_noise_mult
        self._min_duration = min_duration

        self._reset_state()

    def _reset_state(self) -> None:
        self._current_regime = VOLATILE_NOISE
        self._regime_age = 0
        self._vol_buf: list[float] = []
        self._vol_median = 0.01

    def name(self) -> str:
        return "regime_state_machine"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_rsm_regime", warmup=20,
                             description="Discrete regime label {0..5}"),
            AlgorithmFeature("alg_rsm_confidence", warmup=20,
                             description="Confidence: max_score / total"),
            AlgorithmFeature("alg_rsm_transition_risk", warmup=20,
                             description="Risk of regime change (decays with age)"),
            AlgorithmFeature("alg_rsm_trade_allowed", warmup=0,
                             description="1 if not VOLATILE_NOISE, 0 otherwise"),
        ]

    def required_columns(self) -> list[str]:
        return [
            "vol_returns_5m_last",
            "trend_hurst_300_mean",
            "ent_tick_1m_mean",
            "whale_net_flow_4h_sum",
            "toxic_vpin_50_mean",
            "regime_accumulation_score_mean",
        ]

    def _score_states(self, tick: dict[str, float]) -> np.ndarray:
        """Compute score for each of 6 states. Each condition met adds 1."""
        vol = tick.get("vol_returns_5m_last", 0.0)
        hurst = tick.get("trend_hurst_300_mean", 0.5)
        ent = tick.get("ent_tick_1m_mean", 0.5)
        whale = tick.get("whale_net_flow_4h_sum", 0.0)
        vpin = tick.get("toxic_vpin_50_mean", 0.5)
        accum = tick.get("regime_accumulation_score_mean", 0.0)

        # Use accumulation score as proxy for distribution (1 - accum)
        distrib = 1.0 - accum

        # Momentum proxy from hurst + whale direction
        momentum_up = whale > 0 and hurst > self._hurst_trend
        momentum_down = whale < 0 and hurst > self._hurst_trend

        scores = np.zeros(6)

        # State 0: ACCUMULATION
        scores[ACCUMULATION] += float(accum > self._accum_thresh)
        scores[ACCUMULATION] += float(whale > 0)
        scores[ACCUMULATION] += float(ent < self._entropy_high)

        # State 1: DISTRIBUTION
        scores[DISTRIBUTION] += float(distrib > self._distrib_thresh)
        scores[DISTRIBUTION] += float(whale < 0)
        scores[DISTRIBUTION] += float(ent < self._entropy_high)

        # State 2: TRENDING_UP
        scores[TRENDING_UP] += float(momentum_up)
        scores[TRENDING_UP] += float(hurst > self._hurst_trend)
        scores[TRENDING_UP] += float(ent < self._entropy_low)
        scores[TRENDING_UP] += float(whale > 0)

        # State 3: TRENDING_DOWN
        scores[TRENDING_DOWN] += float(momentum_down)
        scores[TRENDING_DOWN] += float(hurst > self._hurst_trend)
        scores[TRENDING_DOWN] += float(ent < self._entropy_low)
        scores[TRENDING_DOWN] += float(whale < 0)

        # State 4: RANGING
        scores[RANGING] += float(ent > self._entropy_high)
        scores[RANGING] += float(hurst < 0.45)
        scores[RANGING] += float(abs(whale) < 500)

        # State 5: VOLATILE_NOISE
        scores[VOLATILE_NOISE] += float(abs(vol) > self._vol_noise_mult * self._vol_median)
        scores[VOLATILE_NOISE] += float(vpin > 0.80)
        scores[VOLATILE_NOISE] += float(ent > 0.70)

        return scores

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Check for NaN in required columns
        for col in self.required_columns():
            v = tick.get(col, np.nan)
            if not np.isfinite(v):
                return nan_out

        # Update rolling volatility median
        vol = tick.get("vol_returns_5m_last", 0.0)
        self._vol_buf.append(abs(vol))
        if len(self._vol_buf) > 288:  # ~24h of 5-min bars
            self._vol_buf.pop(0)
        if len(self._vol_buf) >= 10:
            self._vol_median = float(np.median(self._vol_buf))

        # Score each state
        scores = self._score_states(tick)

        # Find best state (tie-break: highest index wins -> VOLATILE_NOISE)
        max_score = scores.max()
        candidates = np.where(scores == max_score)[0]
        best = int(candidates[-1])  # Highest index = VOLATILE_NOISE wins ties

        # Hysteresis: hold current state for min_duration
        self._regime_age += 1
        if best != self._current_regime and self._regime_age >= self._min_duration:
            self._current_regime = best
            self._regime_age = 0

        # Confidence
        total = scores.sum()
        confidence = float(max_score / (total + 1e-10))

        # Transition risk: decays exponentially with regime age
        transition_risk = float(np.exp(-self._regime_age / 10.0))

        # Trade allowed: 0 in VOLATILE_NOISE
        trade_allowed = 0.0 if self._current_regime == VOLATILE_NOISE else 1.0

        return {
            "alg_rsm_regime": float(self._current_regime),
            "alg_rsm_confidence": confidence,
            "alg_rsm_transition_risk": transition_risk,
            "alg_rsm_trade_allowed": trade_allowed,
        }

    def reset(self) -> None:
        self._reset_state()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized batch computation."""
        import pandas as pd

        self.reset()
        n = len(df)
        regimes = np.full(n, np.nan)
        confidences = np.full(n, np.nan)
        transition_risks = np.full(n, np.nan)
        trade_allowed = np.full(n, np.nan)

        cols = self.required_columns()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "alg_rsm_regime": regimes,
                "alg_rsm_confidence": confidences,
                "alg_rsm_transition_risk": transition_risks,
                "alg_rsm_trade_allowed": trade_allowed,
            }, index=df.index)

        for i in range(n):
            row = df.iloc[i]
            tick = {c: float(row[c]) for c in cols}
            result = self.step(tick)
            regimes[i] = result["alg_rsm_regime"]
            confidences[i] = result["alg_rsm_confidence"]
            transition_risks[i] = result["alg_rsm_transition_risk"]
            trade_allowed[i] = result["alg_rsm_trade_allowed"]

        result_df = pd.DataFrame({
            "alg_rsm_regime": regimes,
            "alg_rsm_confidence": confidences,
            "alg_rsm_transition_risk": transition_risks,
            "alg_rsm_trade_allowed": trade_allowed,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan

        return result_df

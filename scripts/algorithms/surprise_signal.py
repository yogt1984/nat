"""
Surprise Signal Algorithm

Detects sudden entropy drops or spikes as regime transition indicators.
A large negative entropy rate-of-change signals the market transitioning
from disordered to ordered state — potentially tradeable.

Reference:
  Bentes & Menezes (2012) — entropy as a measure of market uncertainty
  Schreiber (2000) — transfer entropy and information flow
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class SurpriseSignal(MicrostructureAlgorithm):
    """Entropy regime transition detection via rate-of-change and z-score."""

    def __init__(self, roc_window: int = 50, transition_threshold: float = 2.0):
        self._roc_window = roc_window
        self._transition_threshold = transition_threshold
        self._entropy_buffer: list[float] = []
        self._roc_buffer: list[float] = []

    def name(self) -> str:
        return "surprise_signal"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_entropy_surprise", warmup=100,
                             description="Z-score of entropy rate-of-change"),
            AlgorithmFeature("alg_entropy_roc", warmup=50,
                             description="Entropy rate of change over window"),
            AlgorithmFeature("alg_regime_transition_prob", warmup=100,
                             description="P(regime transition) from entropy dynamics"),
        ]

    def required_columns(self) -> list[str]:
        return ["ent_book_shape", "ent_tick_5s"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        ent_shape = tick.get("ent_book_shape", np.nan)
        ent_tick = tick.get("ent_tick_5s", np.nan)

        if not (np.isfinite(ent_shape) and np.isfinite(ent_tick)):
            return {f.name: np.nan for f in self.alg_features()}

        # Composite entropy (blend shape and tick entropy)
        ent = 0.5 * ent_shape + 0.5 * ent_tick

        self._entropy_buffer.append(ent)
        if len(self._entropy_buffer) > self._roc_window * 4:
            self._entropy_buffer.pop(0)

        if len(self._entropy_buffer) < self._roc_window + 10:
            return {f.name: np.nan for f in self.alg_features()}

        # Rate of change
        current = np.mean(self._entropy_buffer[-5:])
        past = np.mean(self._entropy_buffer[-self._roc_window - 5:-self._roc_window])
        roc = current - past

        self._roc_buffer.append(roc)
        if len(self._roc_buffer) > self._roc_window * 2:
            self._roc_buffer.pop(0)

        if len(self._roc_buffer) < 20:
            return {
                "alg_entropy_surprise": np.nan,
                "alg_entropy_roc": roc,
                "alg_regime_transition_prob": np.nan,
            }

        # Z-score of ROC
        roc_arr = np.array(self._roc_buffer)
        roc_mean = np.mean(roc_arr)
        roc_std = np.std(roc_arr) + 1e-12
        surprise = (roc - roc_mean) / roc_std

        # Transition probability: sigmoid of |surprise|
        # Large |surprise| → high transition probability
        transition_prob = 1.0 / (1.0 + np.exp(-abs(surprise) + self._transition_threshold))

        return {
            "alg_entropy_surprise": surprise,
            "alg_entropy_roc": roc,
            "alg_regime_transition_prob": transition_prob,
        }

    def reset(self) -> None:
        self._entropy_buffer.clear()
        self._roc_buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        ent_shape = df["ent_book_shape"].values.astype(np.float64)
        ent_tick = df["ent_tick_5s"].values.astype(np.float64)

        ent = 0.5 * ent_shape + 0.5 * ent_tick
        ent_s = pd.Series(ent)

        # Smoothed entropy (5-tick moving average)
        ent_smooth = ent_s.rolling(5, min_periods=1).mean()

        # ROC: current smoothed - lagged smoothed
        roc = (ent_smooth - ent_smooth.shift(self._roc_window)).values

        # Z-score of ROC
        roc_s = pd.Series(roc)
        roc_mean = roc_s.rolling(self._roc_window * 2, min_periods=20).mean().values
        roc_std = roc_s.rolling(self._roc_window * 2, min_periods=20).std().values
        surprise = (roc - roc_mean) / (roc_std + 1e-12)

        # Transition probability
        transition = 1.0 / (1.0 + np.exp(-np.abs(surprise) + self._transition_threshold))

        result = pd.DataFrame({
            "alg_entropy_surprise": surprise,
            "alg_entropy_roc": roc,
            "alg_regime_transition_prob": transition,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

"""
Entropy-Gated Momentum Algorithm

Gates momentum signals by entropy percentile. Low entropy = high predictability;
momentum signals are more reliable when the market is in a low-entropy regime.

Reference:
  Bentes & Menezes (2012) — entropy-based approaches to financial time series
  Novel combination of permutation entropy with trend momentum features.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class EntropyMomentum(MicrostructureAlgorithm):
    """Entropy-gated momentum: stronger signal when entropy is low."""

    def __init__(self, low_entropy_pct: float = 30.0, momentum_window: int = 300,
                 enter_pct: float = 25.0, exit_pct: float = 35.0,
                 ema_alpha: float = 0.1):
        self._low_entropy_pct = low_entropy_pct
        self._momentum_window = momentum_window
        self._enter_pct = enter_pct
        self._exit_pct = exit_pct
        self._ema_alpha = ema_alpha
        self._entropy_buffer: list[float] = []
        self._gate_buffer: list[float] = []
        self._ema_entropy: float = np.nan
        self._in_low_entropy: bool = False

    def name(self) -> str:
        return "entropy_momentum"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_entropy_gated_momentum", warmup=100,
                             description="Momentum when entropy < P30, else 0"),
            AlgorithmFeature("alg_entropy_trend_interaction", warmup=100,
                             description="(1 - entropy) * momentum — continuous interaction"),
            AlgorithmFeature("alg_predictability_score", warmup=100,
                             description="Rolling fraction of time in low-entropy regime"),
        ]

    def required_columns(self) -> list[str]:
        return ["ent_book_shape", "trend_momentum_60", "trend_momentum_300"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        ent = tick.get("ent_book_shape", np.nan)
        mom60 = tick.get("trend_momentum_60", np.nan)
        mom300 = tick.get("trend_momentum_300", np.nan)

        if not all(np.isfinite(x) for x in [ent, mom60, mom300]):
            return {f.name: np.nan for f in self.alg_features()}

        self._entropy_buffer.append(ent)
        if len(self._entropy_buffer) > self._momentum_window:
            self._entropy_buffer.pop(0)

        if len(self._entropy_buffer) < 50:
            return {f.name: np.nan for f in self.alg_features()}

        # Composite momentum (blend short + long)
        momentum = 0.6 * mom60 + 0.4 * mom300

        # EMA-smoothed entropy for hysteresis
        if np.isnan(self._ema_entropy):
            self._ema_entropy = ent
        else:
            self._ema_entropy = self._ema_alpha * ent + (1 - self._ema_alpha) * self._ema_entropy

        # Hysteresis gate: enter low-entropy at P25, exit at P35
        enter_thresh = np.percentile(self._entropy_buffer, self._enter_pct)
        exit_thresh = np.percentile(self._entropy_buffer, self._exit_pct)
        if self._in_low_entropy:
            low_entropy = self._ema_entropy < exit_thresh
        else:
            low_entropy = self._ema_entropy < enter_thresh
        self._in_low_entropy = low_entropy

        gated_momentum = momentum if low_entropy else 0.0

        # Continuous interaction: weight momentum by (1 - normalized_entropy)
        ent_min = min(self._entropy_buffer)
        ent_max = max(self._entropy_buffer)
        ent_range = ent_max - ent_min
        if ent_range > 1e-12:
            ent_norm = (ent - ent_min) / ent_range
        else:
            ent_norm = 0.5
        interaction = (1 - ent_norm) * momentum

        # Predictability score: rolling fraction of time in low-entropy regime
        self._gate_buffer.append(1.0 if low_entropy else 0.0)
        if len(self._gate_buffer) > self._momentum_window:
            self._gate_buffer.pop(0)
        predictability = float(np.mean(self._gate_buffer))

        return {
            "alg_entropy_gated_momentum": gated_momentum,
            "alg_entropy_trend_interaction": interaction,
            "alg_predictability_score": predictability,
        }

    def reset(self) -> None:
        self._entropy_buffer.clear()
        self._gate_buffer.clear()
        self._ema_entropy = np.nan
        self._in_low_entropy = False

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        ent = df["ent_book_shape"].values.astype(np.float64)
        mom60 = df["trend_momentum_60"].values.astype(np.float64)
        mom300 = df["trend_momentum_300"].values.astype(np.float64)

        momentum = 0.6 * mom60 + 0.4 * mom300

        ent_s = pd.Series(ent)

        # EMA-smoothed entropy
        ent_ema = ent_s.ewm(alpha=self._ema_alpha, adjust=False).mean().values

        # Hysteresis thresholds
        enter_thresh = ent_s.rolling(
            self._momentum_window, min_periods=50
        ).quantile(self._enter_pct / 100.0).values
        exit_thresh = ent_s.rolling(
            self._momentum_window, min_periods=50
        ).quantile(self._exit_pct / 100.0).values

        # Apply hysteresis state machine
        in_low = np.zeros(len(ent), dtype=bool)
        for i in range(len(ent)):
            if np.isnan(enter_thresh[i]):
                continue
            if i > 0 and in_low[i - 1]:
                in_low[i] = ent_ema[i] < exit_thresh[i]
            else:
                in_low[i] = ent_ema[i] < enter_thresh[i]

        low_entropy = in_low
        gated = np.where(low_entropy, momentum, 0.0)
        gated[np.isnan(enter_thresh)] = np.nan

        # Continuous interaction via rolling min-max normalization
        rolling_min = ent_s.rolling(self._momentum_window, min_periods=50).min().values
        rolling_max = ent_s.rolling(self._momentum_window, min_periods=50).max().values
        ent_range = rolling_max - rolling_min
        ent_norm = np.where(ent_range > 1e-12,
                            (ent - rolling_min) / ent_range,
                            0.5)
        interaction = (1 - ent_norm) * momentum
        interaction[np.isnan(rolling_min)] = np.nan

        # Predictability: rolling mean of low-entropy indicator
        predictability = pd.Series(low_entropy.astype(np.float64)).rolling(
            self._momentum_window, min_periods=50
        ).mean().values.copy()
        predictability[np.isnan(enter_thresh)] = np.nan

        result = pd.DataFrame({
            "alg_entropy_gated_momentum": gated,
            "alg_entropy_trend_interaction": interaction,
            "alg_predictability_score": predictability,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

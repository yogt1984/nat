"""
Regime-Gated Imbalance Algorithm

Gates raw L1 imbalance by ent_book_shape percentile.
Uses a rolling window to estimate the percentile threshold adaptively.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class RegimeGatedImbalance(MicrostructureAlgorithm):
    """Raw imbalance gated by ent_book_shape percentile."""

    def __init__(self, percentile: float = 30.0, window: int = 6000):
        self._percentile = percentile
        self._window = window
        self._regime_buffer: list[float] = []
        self._threshold = np.nan

    def name(self) -> str:
        return "regime_gated"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_regime_gated_imbalance", warmup=100,
                             description="Imbalance when regime gate active, else 0"),
            AlgorithmFeature("alg_regime_gate_active", warmup=100,
                             description="1.0 if regime gate active, 0.0 otherwise"),
            AlgorithmFeature("alg_regime_zscore", warmup=100,
                             description="(ent_book_shape - mean) / std over window"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1", "ent_book_shape"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        imb = tick.get("imbalance_qty_l1", np.nan)
        ent = tick.get("ent_book_shape", np.nan)

        if not np.isfinite(ent):
            return {f.name: np.nan for f in self.alg_features()}

        # Update rolling buffer
        self._regime_buffer.append(ent)
        if len(self._regime_buffer) > self._window:
            self._regime_buffer.pop(0)

        if len(self._regime_buffer) < 100:
            return {f.name: np.nan for f in self.alg_features()}

        # Update threshold periodically (every 100 ticks)
        if len(self._regime_buffer) % 100 == 0 or np.isnan(self._threshold):
            arr = np.array(self._regime_buffer)
            self._threshold = float(np.nanpercentile(arr, self._percentile))

        # Compute features
        gate_active = 1.0 if ent < self._threshold else 0.0
        gated_imb = imb * gate_active if np.isfinite(imb) else 0.0

        arr = np.array(self._regime_buffer)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        zscore = (ent - mean) / (std + 1e-12)

        return {
            "alg_regime_gated_imbalance": gated_imb,
            "alg_regime_gate_active": gate_active,
            "alg_regime_zscore": zscore,
        }

    def reset(self) -> None:
        self._regime_buffer.clear()
        self._threshold = np.nan

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override using pandas rolling."""
        import pandas as pd

        imb = df["imbalance_qty_l1"].values.astype(np.float64)
        ent = df["ent_book_shape"].values.astype(np.float64)
        n = len(df)

        # Rolling percentile threshold via pandas
        ent_series = pd.Series(ent)
        rolling_thresh = ent_series.rolling(
            window=self._window, min_periods=100
        ).quantile(self._percentile / 100.0).values

        # Rolling mean/std for z-score
        rolling_mean = ent_series.rolling(
            window=self._window, min_periods=100
        ).mean().values
        rolling_std = ent_series.rolling(
            window=self._window, min_periods=100
        ).std().values

        # Gate: ent < rolling threshold
        gate_active = (ent < rolling_thresh).astype(np.float64)
        # NaN where threshold not yet available
        gate_active[np.isnan(rolling_thresh)] = np.nan

        # Gated imbalance
        gated_imb = imb * gate_active
        gated_imb[~np.isfinite(imb)] = 0.0
        gated_imb[np.isnan(gate_active)] = np.nan

        # Z-score
        zscore = (ent - rolling_mean) / (rolling_std + 1e-12)
        zscore[np.isnan(rolling_mean)] = np.nan

        result = pd.DataFrame({
            "alg_regime_gated_imbalance": gated_imb,
            "alg_regime_gate_active": gate_active,
            "alg_regime_zscore": zscore,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

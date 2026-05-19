"""
Multi-Level Imbalance Algorithm

Combines L1, L5, L10 order book imbalance into a composite signal.
Measures divergence between depth levels as an information metric.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class MultiLevelImbalance(MicrostructureAlgorithm):
    """Weighted combination of L1/L5/L10 imbalance."""

    def __init__(self, weights: tuple[float, ...] = (0.5, 0.3, 0.2)):
        self._weights = weights

    def name(self) -> str:
        return "multi_level_imb"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_composite_imbalance", warmup=10,
                             description="Weighted L1/L5/L10 imbalance"),
            AlgorithmFeature("alg_l1_l5_divergence", warmup=10,
                             description="L1 - L5 imbalance (near vs mid depth)"),
            AlgorithmFeature("alg_depth_agreement", warmup=10,
                             description="sign(L1) == sign(L5) == sign(L10) → 1.0"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        l1 = tick.get("imbalance_qty_l1", np.nan)
        l5 = tick.get("imbalance_qty_l5", np.nan)
        l10 = tick.get("imbalance_qty_l10", np.nan)

        if not (np.isfinite(l1) and np.isfinite(l5) and np.isfinite(l10)):
            return {f.name: np.nan for f in self.alg_features()}

        w1, w5, w10 = self._weights
        composite = w1 * l1 + w5 * l5 + w10 * l10
        divergence = l1 - l5

        # Agreement: all three levels have the same sign
        signs = [np.sign(l1), np.sign(l5), np.sign(l10)]
        agreement = 1.0 if signs[0] == signs[1] == signs[2] and signs[0] != 0 else 0.0

        return {
            "alg_composite_imbalance": composite,
            "alg_l1_l5_divergence": divergence,
            "alg_depth_agreement": agreement,
        }

    def reset(self) -> None:
        pass  # Stateless algorithm

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override — stateless, no iteration needed."""
        import pandas as pd

        l1 = df["imbalance_qty_l1"].values
        l5 = df["imbalance_qty_l5"].values
        l10 = df["imbalance_qty_l10"].values

        w1, w5, w10 = self._weights
        composite = w1 * l1 + w5 * l5 + w10 * l10
        divergence = l1 - l5
        agreement = ((np.sign(l1) == np.sign(l5)) &
                     (np.sign(l5) == np.sign(l10)) &
                     (np.sign(l1) != 0)).astype(np.float64)

        result = pd.DataFrame({
            "alg_composite_imbalance": composite,
            "alg_l1_l5_divergence": divergence,
            "alg_depth_agreement": agreement,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

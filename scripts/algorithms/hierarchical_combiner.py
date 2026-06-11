"""
Hierarchical Signal Combiner
=============================

Three-layer signal architecture that separates directional bias (slow),
entry timing (fast), and position sizing (volatility).

Layer 1 — Slow directional bias (5m-15m features):
  IC-weighted z-score of regime/trend features -> sign -> {-1, 0, +1}
  Only active when |z| exceeds threshold (filters noise).

Layer 2 — Fast entry timing (1-5s features, aggregated to 5min bars):
  IC-weighted z-score of order book imbalance features -> [-1, +1]
  Gated by Layer 1: only fires when aligned with slow direction.

Layer 3 — Vol sizing (volatility features):
  Inverse vol scaling -> [0, 1]. High vol = smaller size due to
  increased adverse selection (9_6 conditional IC analysis).

Composite = L1_direction * L2_entry * L3_size

Output Features (4):
  alg_hier_directional_bias  {-1, 0, +1}  Slow directional state
  alg_hier_entry_timing      [-1, +1]     Fast entry strength
  alg_hier_vol_scale         [0, 1]       Vol-adjusted size scalar
  alg_hier_composite         [-1, +1]     Final combined signal

References:
  9_6 IC scan: docs/research/new/9_6/full_ic_scan_report.md
  9_6 validation: docs/research/new/9_6/ic_validation_report.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_WEIGHTS_PATH = str(_PROJECT_ROOT / "models" / "hierarchical_combiner" / "weights.json")

# --- Feature groups per layer ---

L1_FEATURES = [
    "regime_divergence_1h_last",
    "raw_spread_bps_mean",
    "trend_ema_short_last",
]

L2_FEATURES = [
    "imbalance_qty_l1_mean",
    "cross_obi_mean_mean",
    "micro_queue_position_bid_mean",
    "flow_vwap_deviation_mean",
    "micro_obi_velocity_mean",
]

L3_FEATURES = [
    "hawkes_intensity_mean",
    "flow_count_30s_sum",
    "vol_returns_5m_last",
]

ALL_FEATURES = L1_FEATURES + L2_FEATURES + L3_FEATURES

# Default IC-based weights (from 9_6 scan, 5min horizon).
# Overridden by trained weights when available.
DEFAULT_L1_WEIGHTS = {
    "regime_divergence_1h_last": 0.07,
    "raw_spread_bps_mean": 0.14,
    "trend_ema_short_last": -0.14,  # negative: mean-reverting
}

DEFAULT_L2_WEIGHTS = {
    "imbalance_qty_l1_mean": 0.09,
    "cross_obi_mean_mean": 0.07,
    "micro_queue_position_bid_mean": 0.06,
    "flow_vwap_deviation_mean": -0.05,  # negative: mean-reverting
    "micro_obi_velocity_mean": 0.04,
}

DEFAULT_L3_WEIGHTS = {
    "hawkes_intensity_mean": 0.35,
    "flow_count_30s_sum": 0.32,
    "vol_returns_5m_last": 0.33,
}


def _rolling_zscore(series: pd.Series, window: int = 200) -> pd.Series:
    """Rolling z-score with expanding fallback for early rows."""
    mu = series.rolling(window, min_periods=20).mean()
    sigma = series.rolling(window, min_periods=20).std()
    sigma = sigma.replace(0, np.nan)
    return (series - mu) / sigma


def _ic_weighted_composite(
    df: pd.DataFrame,
    features: list[str],
    weights: dict[str, float],
    window: int = 200,
) -> pd.Series:
    """IC-weighted z-score composite of available features."""
    available = [f for f in features if f in df.columns and df[f].notna().mean() > 0.1]
    if not available:
        return pd.Series(0.0, index=df.index)

    total_weight = sum(abs(weights.get(f, 0.01)) for f in available)
    if total_weight == 0:
        total_weight = 1.0

    composite = pd.Series(0.0, index=df.index)
    for f in available:
        z = _rolling_zscore(df[f], window=window)
        w = weights.get(f, 0.01) / total_weight
        composite += w * z.fillna(0.0)

    return composite


@register
class HierarchicalCombiner(MicrostructureAlgorithm):
    """Three-layer hierarchical signal combiner.

    Combines slow directional bias, fast entry timing, and vol-based
    sizing into a single composite signal.
    """

    bar_level = True
    bar_timeframe = "5min"

    def __init__(
        self,
        weights_path: str = _DEFAULT_WEIGHTS_PATH,
        symbol: str | None = None,
        l1_threshold: float = 0.5,
        l1_target_percentile: float = 70.0,
        zscore_window: int = 200,
    ):
        self._l1_threshold = l1_threshold  # fixed fallback
        self._l1_target_percentile = l1_target_percentile
        self._zscore_window = zscore_window

        # Load trained weights or use defaults
        self._l1_weights = dict(DEFAULT_L1_WEIGHTS)
        self._l2_weights = dict(DEFAULT_L2_WEIGHTS)
        self._l3_weights = dict(DEFAULT_L3_WEIGHTS)

        # Per-symbol file takes priority over generic
        wp = Path(weights_path)
        if symbol:
            per_symbol = wp.parent / f"weights_{symbol}.json"
            if per_symbol.exists():
                wp = per_symbol

        if wp.exists():
            with open(wp) as f:
                trained = json.load(f)
            self._l1_weights.update(trained.get("l1_weights", {}))
            self._l2_weights.update(trained.get("l2_weights", {}))
            self._l3_weights.update(trained.get("l3_weights", {}))
            self._l1_threshold = trained.get("l1_threshold", self._l1_threshold)
            self._l1_target_percentile = trained.get(
                "l1_target_percentile", self._l1_target_percentile
            )
            logger.info("Loaded trained weights from %s", wp)

    def name(self) -> str:
        return "hierarchical_combiner"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_hier_directional_bias", warmup=50,
                             description="Slow directional state {-1, 0, +1}"),
            AlgorithmFeature("alg_hier_entry_timing", warmup=50,
                             description="Fast entry strength [-1, +1]"),
            AlgorithmFeature("alg_hier_vol_scale", warmup=50,
                             description="Vol-adjusted size scalar [0, 1]"),
            AlgorithmFeature("alg_hier_composite", warmup=50,
                             description="Final combined signal [-1, +1]"),
        ]

    def required_columns(self) -> list[str]:
        return list(ALL_FEATURES)

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        # Bar-level only — step() not used
        return {f.name: np.nan for f in self.alg_features()}

    def reset(self) -> None:
        pass

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized hierarchical combination over bar DataFrame."""
        n = len(df)
        w = self._zscore_window

        # --- Layer 1: Slow directional bias ---
        l1_z = _ic_weighted_composite(df, L1_FEATURES, self._l1_weights, window=w)

        # Adaptive threshold: rolling percentile of |z| with fixed fallback
        l1_abs = l1_z.abs()
        rolling_thresh = l1_abs.rolling(w, min_periods=50).quantile(
            self._l1_target_percentile / 100.0
        )
        rolling_thresh = rolling_thresh.fillna(self._l1_threshold)

        # Discretize: strong positive → +1, strong negative → -1, else 0
        l1_direction = pd.Series(0.0, index=df.index)
        l1_direction[l1_z > rolling_thresh] = 1.0
        l1_direction[l1_z < -rolling_thresh] = -1.0

        # --- Layer 2: Fast entry timing ---
        l2_raw = _ic_weighted_composite(df, L2_FEATURES, self._l2_weights, window=w)

        # Clip to [-1, +1]
        l2_entry = l2_raw.clip(-3, 3) / 3.0  # normalize 3-sigma to [-1, +1]

        # Gate by L1: zero out entries that disagree with directional bias
        l1_active = l1_direction != 0
        aligned = np.sign(l2_entry) == l1_direction
        l2_gated = l2_entry.copy()
        l2_gated[l1_active & ~aligned] = 0.0

        # --- Layer 3: Vol sizing (inverse) ---
        l3_raw = _ic_weighted_composite(df, L3_FEATURES, self._l3_weights, window=w)

        # High vol z-score → smaller size. Map z to [0, 1] via sigmoid-like.
        # z=0 → size=0.5, z=-2 → size~0.9, z=+2 → size~0.1
        l3_scale = 1.0 / (1.0 + np.exp(l3_raw.clip(-5, 5)))

        # --- Composite ---
        composite = l1_direction * l2_gated.abs() * l3_scale
        # Re-sign by L2 when L1 is neutral (passthrough)
        neutral = l1_direction == 0
        composite[neutral] = l2_gated[neutral] * l3_scale[neutral] * 0.3  # dampened

        composite = composite.clip(-1, 1)

        result = pd.DataFrame({
            "alg_hier_directional_bias": l1_direction,
            "alg_hier_entry_timing": l2_gated,
            "alg_hier_vol_scale": l3_scale,
            "alg_hier_composite": composite,
        }, index=df.index)

        # NaN warmup
        if self.warmup > 0 and self.warmup < n:
            result.iloc[:self.warmup] = np.nan

        return result

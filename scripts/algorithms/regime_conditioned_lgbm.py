"""
Regime-Conditioned LightGBM Ensemble
=====================================

Routes predictions to per-regime LightGBM models based on the current
market regime from RegimeStateMachine (#4). Each regime uses a tailored
feature subset. Falls back to a global model when regime confidence is
low or no per-regime model exists.

Routing:
  regime in {2, 3} (TRENDING_UP/DOWN)  -> trending model
  regime in {0, 1} (ACCUM/DISTRIB)     -> ranging model
  regime in {4, 5} (RANGING/NOISE)     -> volatile model
  low confidence or missing            -> global model

Output Features (4):
  alg_rlgbm_signal             [-1, 1]   Directional signal
  alg_rlgbm_predicted_return   (-inf,inf) Raw predicted return
  alg_rlgbm_regime_used        {0..5}    Which regime model was used
  alg_rlgbm_regime_confidence  [0, 1]    Regime classification confidence

References:
  Gu, Kelly & Xiu (2020) — Empirical Asset Pricing via Machine Learning
  Nystrup et al. (2017) — Dynamic Allocation or Diversification
  See docs/research/new/ml_algorithms.txt Section 7 for full specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


# Per-regime feature subsets
TRENDING_FEATURES = [
    "trend_hurst_300_mean",
    "whale_net_flow_4h_sum",
    "regime_accumulation_score_mean",
    "trend_momentum_300_mean",
    "alg_conv_best_score_max",
]

RANGING_FEATURES = [
    "ent_tick_1m_mean",
    "vol_returns_5m_last",
    "imbalance_qty_l1_mean",
    "mf_bb_pctb_5m_last",
    "alg_conv_best_score_max",
]

# Optional features not required to be present — handled by feat_cols_available filter
_OPTIONAL_FEATURES = {"alg_conv_best_score_max"}

VOLATILE_FEATURES = [
    "vol_returns_5m_last",
    "toxic_vpin_50_mean",
    "ent_tick_1m_mean",
]

GLOBAL_FEATURES = sorted(set(TRENDING_FEATURES + RANGING_FEATURES + VOLATILE_FEATURES))

# Regime group mapping: regime_id -> model_key
REGIME_TO_GROUP = {
    0: "ranging",     # ACCUMULATION
    1: "ranging",     # DISTRIBUTION
    2: "trending",    # TRENDING_UP
    3: "trending",    # TRENDING_DOWN
    4: "volatile",    # RANGING
    5: "volatile",    # VOLATILE_NOISE
}

GROUP_FEATURES = {
    "trending": TRENDING_FEATURES,
    "ranging": RANGING_FEATURES,
    "volatile": VOLATILE_FEATURES,
    "global": GLOBAL_FEATURES,
}


@register
class RegimeConditionedLGBM(MicrostructureAlgorithm):
    """Per-regime LightGBM ensemble with global fallback.

    Routes to regime-specific models when confidence is high enough.
    Falls back to global model otherwise. Returns NaN if no models loaded.
    """

    bar_level = True

    def __init__(
        self,
        model_dir: str = "models/regime_conditioned_lgbm",
        confidence_threshold: float = 0.60,
        min_samples_regime: int = 500,
    ):
        self._models = {}  # key -> LightGBM Booster
        self._confidence_threshold = confidence_threshold
        self._min_samples_regime = min_samples_regime

        # Try to load models
        try:
            from utils.model_io import load_lightgbm_model
            model_path = Path(model_dir)
            if model_path.exists():
                for key in ["trending", "ranging", "volatile", "global"]:
                    fpath = model_path / f"model_{key}.txt"
                    if fpath.exists():
                        model, _ = load_lightgbm_model(fpath)
                        self._models[key] = model
        except (ImportError, Exception):
            pass

    def name(self) -> str:
        return "regime_conditioned_lgbm"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_rlgbm_signal", warmup=50,
                             description="Regime-conditioned directional signal [-1, +1]"),
            AlgorithmFeature("alg_rlgbm_predicted_return", warmup=50,
                             description="Raw predicted return"),
            AlgorithmFeature("alg_rlgbm_regime_used", warmup=0,
                             description="Regime model used {0..5}"),
            AlgorithmFeature("alg_rlgbm_regime_confidence", warmup=0,
                             description="Regime classification confidence [0, 1]"),
        ]

    def required_columns(self) -> list[str]:
        # Union of all per-regime features + regime label + confidence
        # Exclude optional features (convolver outputs may not be present)
        cols = set(GLOBAL_FEATURES) - _OPTIONAL_FEATURES
        cols.add("alg_rsm_regime_last")
        cols.add("alg_rsm_confidence_last")
        return sorted(cols)

    def _get_model_and_features(
        self, regime: float, confidence: float
    ) -> tuple[object | None, list[str], float]:
        """Route to appropriate model based on regime and confidence.

        Returns (model, feature_cols, regime_used).
        """
        # Low confidence or missing regime -> global
        if not np.isfinite(regime) or confidence < self._confidence_threshold:
            return self._models.get("global"), GLOBAL_FEATURES, 5.0

        regime_int = int(regime)
        group = REGIME_TO_GROUP.get(regime_int, "global")
        model = self._models.get(group)

        # No per-regime model -> global fallback
        if model is None:
            model = self._models.get("global")
            return model, GLOBAL_FEATURES, 5.0

        return model, GROUP_FEATURES[group], float(regime_int)

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Check required global features are finite (optional default to 0.0)
        x_all = np.array([
            tick.get(c, 0.0 if c in _OPTIONAL_FEATURES else np.nan)
            for c in GLOBAL_FEATURES
        ])
        if not np.all(np.isfinite(x_all)):
            return nan_out

        regime = tick.get("alg_rsm_regime_last", np.nan)
        confidence = tick.get("alg_rsm_confidence_last", 0.0)

        # No models loaded -> neutral output (no edge)
        if not self._models:
            return {
                "alg_rlgbm_signal": 0.0,
                "alg_rlgbm_predicted_return": 0.0,
                "alg_rlgbm_regime_used": float(regime) if np.isfinite(regime) else np.nan,
                "alg_rlgbm_regime_confidence": confidence if np.isfinite(confidence) else 0.0,
            }

        model, feature_cols, regime_used = self._get_model_and_features(regime, confidence)
        if model is None:
            return nan_out

        # Build feature vector for selected model
        x = np.array([tick.get(c, 0.0) for c in feature_cols])
        x_2d = x.reshape(1, -1)

        pred = float(model.predict(x_2d)[0])
        signal = np.clip(np.tanh(pred * 10), -1.0, 1.0)

        return {
            "alg_rlgbm_signal": signal,
            "alg_rlgbm_predicted_return": pred,
            "alg_rlgbm_regime_used": regime_used,
            "alg_rlgbm_regime_confidence": confidence if np.isfinite(confidence) else 0.0,
        }

    def reset(self) -> None:
        pass

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized batch inference with regime routing."""
        import pandas as pd

        n = len(df)
        signals = np.full(n, np.nan)
        pred_returns = np.full(n, np.nan)
        regimes_used = np.full(n, np.nan)
        confidences = np.full(n, np.nan)

        # Check required columns
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "alg_rlgbm_signal": signals,
                "alg_rlgbm_predicted_return": pred_returns,
                "alg_rlgbm_regime_used": regimes_used,
                "alg_rlgbm_regime_confidence": confidences,
            }, index=df.index)

        # Global feature validity (only check required features)
        required_global = [c for c in GLOBAL_FEATURES if c not in _OPTIONAL_FEATURES]
        X_global = df[required_global].values
        valid = np.all(np.isfinite(X_global), axis=1)

        regime_col = df["alg_rsm_regime_last"].values
        conf_col = df["alg_rsm_confidence_last"].values
        confidences = np.where(valid, np.where(np.isfinite(conf_col), conf_col, 0.0), np.nan)

        if not self._models or not np.any(valid):
            # No models loaded -> neutral output (0.0 signal, 0.0 return)
            regimes_used = np.where(valid & np.isfinite(regime_col), regime_col, np.nan)
            result_df = pd.DataFrame({
                "alg_rlgbm_signal": np.where(valid, 0.0, np.nan),
                "alg_rlgbm_predicted_return": np.where(valid, 0.0, np.nan),
                "alg_rlgbm_regime_used": regimes_used,
                "alg_rlgbm_regime_confidence": confidences,
            }, index=df.index)
            warmup = self.warmup
            if warmup > 0 and warmup < n:
                result_df.iloc[:warmup] = np.nan
            return result_df

        # Process each regime group in batch
        for group_name, group_features in GROUP_FEATURES.items():
            model = self._models.get(group_name)
            if model is None:
                continue

            if group_name == "global":
                # Global: low confidence or missing regime
                mask = valid & (
                    ~np.isfinite(regime_col) |
                    (conf_col < self._confidence_threshold)
                )
                # Also rows where per-regime model is missing
                for regime_id, grp in REGIME_TO_GROUP.items():
                    if grp not in self._models:
                        mask |= valid & (regime_col == regime_id)
            else:
                # Per-regime group
                regime_ids = [r for r, g in REGIME_TO_GROUP.items() if g == group_name]
                mask = valid & np.isfinite(regime_col) & (conf_col >= self._confidence_threshold)
                mask &= np.isin(regime_col, regime_ids)

            if not np.any(mask):
                continue

            # Build feature matrix
            feat_cols_available = [c for c in group_features if c in df.columns]
            X = df[feat_cols_available].values[mask]
            preds = model.predict(X)

            pred_returns[mask] = preds
            signals[mask] = np.clip(np.tanh(preds * 10), -1.0, 1.0)
            if group_name == "global":
                regimes_used[mask] = 5.0
            else:
                regimes_used[mask] = regime_col[mask]

        result_df = pd.DataFrame({
            "alg_rlgbm_signal": signals,
            "alg_rlgbm_predicted_return": pred_returns,
            "alg_rlgbm_regime_used": regimes_used,
            "alg_rlgbm_regime_confidence": confidences,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan
        return result_df

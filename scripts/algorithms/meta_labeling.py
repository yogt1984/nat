"""
Meta-Labeling Precision Filter
===============================

Logistic regression classifier that predicts P(trade success) given
non-directional market state features. Acts as a precision filter on
top of base algorithm signals — does NOT predict direction.

The model is trained on triple-barrier labels (De Prado Ch. 3) from
the 5 winner base algorithms. At inference time, it outputs:
  - P(success): probability that a trade taken NOW will be profitable
  - Side: sign of the strongest base signal (direction)
  - Size: P(success) * scaling factor (position sizing)

Output Features (3):
  alg_meta_probability  [0, 1]      P(trade success)
  alg_meta_side         {-1, 0, 1}  Direction from strongest base signal
  alg_meta_size         [0, 1]      Position size = P * gate

References:
  De Prado, M.L. (2018). Advances in Financial Machine Learning, Ch. 3.
  See docs/research/new/ml_algorithms.txt Section 3 for full specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class MetaLabeling(MicrostructureAlgorithm):
    """Logistic regression meta-labeling precision filter.

    Predicts P(trade success) from non-directional market state.
    Returns neutral output (prob=0.5, side=0, size=0) if no model loaded.
    """

    bar_level = True

    # Non-directional market state features
    STATE_COLS = [
        "ent_tick_1m_mean",
        "ent_rate_of_change_5s_mean",
        "toxic_vpin_10_mean",
        "toxic_index_mean",
        "conc_hhi_last",
        "whale_directional_agreement_last",
        "vol_returns_5m_mean",
        "vol_ratio_short_long_last",
        "regime_clarity_last",
        "raw_spread_bps_mean",
    ]

    def __init__(
        self,
        model_path: str = "models/meta_labeling",
        meta_threshold: float = 0.55,
    ):
        self._model = None
        self._scaler = None
        self._meta = None
        self._threshold = meta_threshold

        # Try to load a trained model
        try:
            from utils.model_io import get_latest_model, load_sklearn_model

            path = get_latest_model(Path(model_path))
            if path is not None:
                self._model, self._scaler, self._meta = load_sklearn_model(path)
        except (FileNotFoundError, TypeError, ImportError):
            pass

    def name(self) -> str:
        return "meta_labeling"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_meta_probability", warmup=50,
                             description="P(trade success) [0, 1]"),
            AlgorithmFeature("alg_meta_side", warmup=50,
                             description="Direction from strongest base signal {-1, 0, 1}"),
            AlgorithmFeature("alg_meta_size", warmup=50,
                             description="Position size = P * gate [0, 1]"),
        ]

    def required_columns(self) -> list[str]:
        return list(self.STATE_COLS)

    def _model_feature_names(self) -> list[str]:
        """The exact feature order the loaded model/scaler was trained on (its SoT).
        STATE_COLS is a superset (some cols, e.g. whale_directional_agreement_last,
        feed `side`, not the model), so feeding STATE_COLS would shape-mismatch."""
        names = getattr(self._meta, "feature_names", None)
        if names is None and isinstance(self._meta, dict):
            names = self._meta.get("feature_names")
        return list(names) if names else list(self.STATE_COLS)

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Extract state features
        x = np.array([tick.get(c, np.nan) for c in self.STATE_COLS])
        if not np.all(np.isfinite(x)):
            return nan_out

        # No model -> neutral
        if self._model is None:
            return {
                "alg_meta_probability": 0.5,
                "alg_meta_side": 0.0,
                "alg_meta_size": 0.0,
            }

        # Model inference — feed the model's own feature order, not STATE_COLS.
        x_2d = np.array([tick.get(c, np.nan) for c in self._model_feature_names()],
                        dtype=float).reshape(1, -1)
        if self._scaler is not None:
            x_2d = self._scaler.transform(x_2d)

        prob = float(self._model.predict_proba(x_2d)[0, 1])

        # Side: from directional agreement feature (proxy for strongest base signal)
        agreement = tick.get("whale_directional_agreement_last", 0.0)
        side = float(np.sign(agreement))

        # Size: P(success) if above threshold, else 0
        size = prob if prob > self._threshold else 0.0

        return {
            "alg_meta_probability": prob,
            "alg_meta_side": side,
            "alg_meta_size": size,
        }

    def reset(self) -> None:
        pass

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized batch inference."""
        import pandas as pd

        n = len(df)
        probs = np.full(n, np.nan)
        sides = np.full(n, np.nan)
        sizes = np.full(n, np.nan)

        # Check all required columns exist
        missing = [c for c in self.STATE_COLS if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "alg_meta_probability": probs,
                "alg_meta_side": sides,
                "alg_meta_size": sizes,
            }, index=df.index)

        # Validity from all required state cols; model matrix from the model's
        # own feature order (a subset of STATE_COLS).
        X = df[list(self.STATE_COLS)].values
        valid = np.all(np.isfinite(X), axis=1)

        # No model -> neutral for valid rows
        if self._model is None or not np.any(valid):
            probs = np.where(valid, 0.5, np.nan)
            sides = np.where(valid, 0.0, np.nan)
            sizes = np.where(valid, 0.0, np.nan)
        else:
            X_model = df[self._model_feature_names()].values
            X_valid = X_model[valid]
            if self._scaler is not None:
                X_valid = self._scaler.transform(X_valid)
            probs_valid = self._model.predict_proba(X_valid)[:, 1]

            p = np.full(n, np.nan)
            p[valid] = probs_valid
            probs = p

            # Side from directional agreement
            agreement = df["whale_directional_agreement_last"].values
            sides = np.where(valid, np.sign(agreement), np.nan)

            # Size gated by threshold
            sizes = np.where(valid & (probs > self._threshold), probs, 0.0)
            sizes = np.where(valid, sizes, np.nan)

        result_df = pd.DataFrame({
            "alg_meta_probability": probs,
            "alg_meta_side": sides,
            "alg_meta_size": sizes,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan
        return result_df

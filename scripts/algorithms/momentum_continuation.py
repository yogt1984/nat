"""
Momentum Continuation Classifier
=================================

Logistic regression predicting whether current momentum will continue
over the next 20 bars (100 min). Active only in low-entropy regimes
where trend persistence is highest.

Signal: P(continuation) from StandardScaler + LogisticRegression(L2).
        Mapped to [-1, +1] with dead zone at [p_short, p_long].
        Gated by entropy ceiling.

Output Features (3):
  alg_mc_signal         [-1, 1]   Signed continuation signal (0 if gated)
  alg_mc_confidence     [0, 1]    Raw model P(continuation)
  alg_mc_entropy_gate   {0, 1}    1 if entropy below ceiling

References:
  Bouchaud et al. (2004) — Fluctuations and Response in Financial Markets
  See docs/research/new/ml_algorithms.txt Section 1 for full specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class MomentumContinuation(MicrostructureAlgorithm):
    """Logistic regression momentum continuation classifier.

    Active only in low-entropy regimes. Predicts whether current
    momentum will continue or reverse over the next 20 bars.

    Returns NaN for all outputs if no trained model is loaded.
    """

    bar_level = True

    FEATURE_COLS = [
        "ent_tick_1m_mean",
        "ent_permutation_returns_16_mean",
        "trend_hurst_300_mean",
        "toxic_vpin_50_mean",
        "whale_net_flow_4h_sum",
        "regime_accumulation_score_mean",
        "vol_returns_5m_last",
    ]

    def __init__(
        self,
        model_path: str = "models/momentum_continuation",
        entropy_ceiling: float = 0.85,
        p_long: float = 0.6,
        p_short: float = 0.4,
    ):
        self._model = None
        self._scaler = None
        self._meta = None
        self._entropy_ceiling = entropy_ceiling
        self._p_long = p_long
        self._p_short = p_short

        # Try to load a trained model
        try:
            from utils.model_io import get_latest_model, load_sklearn_model

            path = get_latest_model(Path(model_path))
            if path is not None:
                self._model, self._scaler, self._meta = load_sklearn_model(path)
        except (FileNotFoundError, TypeError, ImportError):
            pass

    def name(self) -> str:
        return "momentum_continuation"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_mc_signal", warmup=50,
                             description="Momentum continuation signal [-1, +1]"),
            AlgorithmFeature("alg_mc_confidence", warmup=50,
                             description="Raw P(continuation) [0, 1]"),
            AlgorithmFeature("alg_mc_entropy_gate", warmup=0,
                             description="1 if entropy below ceiling, 0 if gated off"),
        ]

    def required_columns(self) -> list[str]:
        return list(self.FEATURE_COLS)

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Extract features
        x = np.array([tick.get(c, np.nan) for c in self.FEATURE_COLS])
        if not np.all(np.isfinite(x)):
            return nan_out

        # Entropy gate
        entropy = tick.get("ent_tick_1m_mean", 1.0)
        gate = 1.0 if entropy < self._entropy_ceiling else 0.0

        # No model loaded -> signal=0, confidence=0.5 (no edge)
        if self._model is None:
            return {
                "alg_mc_signal": 0.0,
                "alg_mc_confidence": 0.5,
                "alg_mc_entropy_gate": gate,
            }

        # Model inference
        x_2d = x.reshape(1, -1)
        if self._scaler is not None:
            x_2d = self._scaler.transform(x_2d)

        prob = float(self._model.predict_proba(x_2d)[0, 1])

        # Signal: [-1, +1] with dead zone, gated by entropy
        if prob > self._p_long:
            signal = (prob - 0.5) * 2.0 * gate
        elif prob < self._p_short:
            signal = -(0.5 - prob) * 2.0 * gate
        else:
            signal = 0.0

        return {
            "alg_mc_signal": signal,
            "alg_mc_confidence": prob,
            "alg_mc_entropy_gate": gate,
        }

    def reset(self) -> None:
        # Stateless (model is read-only) — nothing to reset
        pass

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized batch inference."""
        import pandas as pd

        n = len(df)
        signals = np.full(n, np.nan)
        confidences = np.full(n, np.nan)
        gates = np.full(n, np.nan)

        # Check all required columns exist
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "alg_mc_signal": signals,
                "alg_mc_confidence": confidences,
                "alg_mc_entropy_gate": gates,
            }, index=df.index)

        # Build feature matrix
        X = df[list(self.FEATURE_COLS)].values
        valid = np.all(np.isfinite(X), axis=1)

        # Entropy gate (vectorized)
        ent = df["ent_tick_1m_mean"].values if "ent_tick_1m_mean" in df.columns else np.ones(n)
        gates = np.where(ent < self._entropy_ceiling, 1.0, 0.0)

        # No model -> signal=0, confidence=0.5 for valid rows
        if self._model is None or not np.any(valid):
            signals = np.where(valid, 0.0, np.nan)
            confidences = np.where(valid, 0.5, np.nan)
        else:
            # Model inference on valid rows
            X_valid = X[valid]
            if self._scaler is not None:
                X_valid = self._scaler.transform(X_valid)
            probs_valid = self._model.predict_proba(X_valid)[:, 1]

            probs = np.full(n, np.nan)
            probs[valid] = probs_valid
            confidences = probs.copy()

            # Signal with dead zone + entropy gate
            signals = np.where(probs > self._p_long, (probs - 0.5) * 2.0 * gates,
                       np.where(probs < self._p_short, -(0.5 - probs) * 2.0 * gates, 0.0))
            signals[~valid] = np.nan

        result_df = pd.DataFrame({
            "alg_mc_signal": signals,
            "alg_mc_confidence": confidences,
            "alg_mc_entropy_gate": gates,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan
        return result_df

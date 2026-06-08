"""
Mean-Reversion / False-Breakout Detector
=========================================

LightGBM classifier predicting whether price will revert to mean
over the next 20 bars (100 min). Active only in HIGH-entropy regimes
where mean-reversion dominates — complementary to MomentumContinuation.

Signal: -sign(zscore) * P(reversion) — contrarian.
        z-score = (midprice - ema) / (vol * midprice).
        Gated by entropy floor (active when entropy > threshold).

Output Features (4):
  alg_mr_signal         [-1, 1]   Contrarian reversion signal
  alg_mr_probability    [0, 1]    Model P(reversion)
  alg_mr_zscore         (-inf,inf) Displacement z-score
  alg_mr_entropy_gate   {0, 1}    1 if entropy above floor (ranging)

References:
  Avellaneda & Lee (2010) — Statistical Arbitrage in the US Equities Market
  Cont et al. (2010) — A Stochastic Model for Order Book Dynamics
  See docs/research/new/ml_algorithms.txt Section 2 for full specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class MeanReversionDetector(MicrostructureAlgorithm):
    """LightGBM mean-reversion detector with inverted entropy gate.

    Active only in high-entropy (ranging) regimes. Predicts whether
    price will revert to EMA over the next 20 bars.

    Returns NaN for signal/probability if no trained model is loaded,
    but always computes zscore and entropy gate from raw data.
    """

    bar_level = True

    FEATURE_COLS = [
        "vol_returns_5m_last",
        "ent_tick_1m_mean",
        "trend_hurst_300_mean",
        "imbalance_qty_l1_mean",
        "toxic_vpin_50_mean",
        "raw_midprice_mean",
        "mf_ema_15m_last",
    ]

    def __init__(
        self,
        model_path: str = "models/mean_reversion_detector",
        entropy_threshold: float = 0.70,
        zscore_threshold: float = 2.0,
        reversion_prob_thresh: float = 0.65,
    ):
        self._model = None
        self._meta = None
        self._entropy_threshold = entropy_threshold
        self._zscore_threshold = zscore_threshold
        self._reversion_prob_thresh = reversion_prob_thresh

        # Try to load a trained model
        try:
            from utils.model_io import get_latest_model, load_lightgbm_model

            path = get_latest_model(Path(model_path), model_type="lightgbm")
            if path is not None:
                self._model, self._meta = load_lightgbm_model(path)
        except (FileNotFoundError, TypeError, ImportError):
            pass

    def name(self) -> str:
        return "mean_reversion_detector"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_mr_signal", warmup=50,
                             description="Contrarian reversion signal [-1, +1]"),
            AlgorithmFeature("alg_mr_probability", warmup=50,
                             description="Model P(reversion) [0, 1]"),
            AlgorithmFeature("alg_mr_zscore", warmup=0,
                             description="Price displacement z-score"),
            AlgorithmFeature("alg_mr_entropy_gate", warmup=0,
                             description="1 if entropy above floor (ranging regime)"),
        ]

    def required_columns(self) -> list[str]:
        return list(self.FEATURE_COLS)

    def _compute_zscore(self, midprice: float, ema: float, vol: float) -> float:
        """z-score = (midprice - ema) / (vol * midprice)."""
        denom = vol * midprice
        if denom <= 0 or not np.isfinite(denom):
            return 0.0
        return (midprice - ema) / denom

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Extract features
        x = np.array([tick.get(c, np.nan) for c in self.FEATURE_COLS])
        if not np.all(np.isfinite(x)):
            return nan_out

        # Raw computations (always available)
        midprice = tick.get("raw_midprice_mean", np.nan)
        ema = tick.get("mf_ema_15m_last", np.nan)
        vol = tick.get("vol_returns_5m_last", np.nan)
        entropy = tick.get("ent_tick_1m_mean", 0.0)

        zscore = self._compute_zscore(midprice, ema, vol)

        # Entropy gate INVERTED: active when entropy HIGH (ranging)
        gate = 1.0 if entropy > self._entropy_threshold else 0.0

        # No model -> neutral defaults for signal/prob, zscore and gate computed
        if self._model is None:
            return {
                "alg_mr_signal": 0.0,
                "alg_mr_probability": 0.5,
                "alg_mr_zscore": zscore,
                "alg_mr_entropy_gate": gate,
            }

        # Model inference — LightGBM Booster uses predict()
        x_2d = x.reshape(1, -1)
        prob = float(self._model.predict(x_2d)[0])
        prob = np.clip(prob, 0.0, 1.0)

        # Contrarian signal: -sign(zscore) * P(reversion), gated
        if gate > 0 and prob > self._reversion_prob_thresh:
            signal = -np.sign(zscore) * prob
        else:
            signal = 0.0

        signal = np.clip(signal, -1.0, 1.0)

        return {
            "alg_mr_signal": signal,
            "alg_mr_probability": prob,
            "alg_mr_zscore": zscore,
            "alg_mr_entropy_gate": gate,
        }

    def reset(self) -> None:
        # Stateless (model is read-only) — nothing to reset
        pass

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized batch inference."""
        import pandas as pd

        n = len(df)
        signals = np.full(n, np.nan)
        probs = np.full(n, np.nan)
        zscores = np.full(n, np.nan)
        gates = np.full(n, np.nan)

        # Check all required columns exist
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            return pd.DataFrame({
                "alg_mr_signal": signals,
                "alg_mr_probability": probs,
                "alg_mr_zscore": zscores,
                "alg_mr_entropy_gate": gates,
            }, index=df.index)

        # Build feature matrix
        X = df[list(self.FEATURE_COLS)].values
        valid = np.all(np.isfinite(X), axis=1)

        # Z-score (vectorized)
        midprice = df["raw_midprice_mean"].values
        ema = df["mf_ema_15m_last"].values
        vol = df["vol_returns_5m_last"].values
        denom = vol * midprice
        safe_denom = np.where((denom > 0) & np.isfinite(denom), denom, np.nan)
        zs = (midprice - ema) / safe_denom
        zs = np.where(np.isfinite(zs), zs, 0.0)
        zscores = np.where(valid, zs, np.nan)

        # Entropy gate INVERTED (vectorized)
        ent = df["ent_tick_1m_mean"].values
        gates = np.where(valid, np.where(ent > self._entropy_threshold, 1.0, 0.0), np.nan)

        # No model -> neutral defaults for signal/prob
        if self._model is None or not np.any(valid):
            signals = np.where(valid, 0.0, np.nan)
            probs = np.where(valid, 0.5, np.nan)
        else:
            # Model inference on valid rows
            X_valid = X[valid]
            probs_valid = self._model.predict(X_valid)
            probs_valid = np.clip(probs_valid, 0.0, 1.0)

            p = np.full(n, np.nan)
            p[valid] = probs_valid
            probs = p

            # Contrarian signal: -sign(zscore) * prob, gated
            sig = np.where(
                (gates > 0) & (probs > self._reversion_prob_thresh),
                -np.sign(zscores) * probs,
                0.0,
            )
            sig = np.clip(sig, -1.0, 1.0)
            signals = np.where(valid, sig, np.nan)

        result_df = pd.DataFrame({
            "alg_mr_signal": signals,
            "alg_mr_probability": probs,
            "alg_mr_zscore": zscores,
            "alg_mr_entropy_gate": gates,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan
        return result_df

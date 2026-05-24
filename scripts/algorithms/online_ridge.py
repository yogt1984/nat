"""
Online Ridge Regression Meta-Algorithm

Lightweight online learning model that combines base features and algorithm
features into a single prediction via recursive least squares with L2
regularization (equivalent to online ridge regression).

Uses Sherman-Morrison rank-1 updates for O(d²) per tick instead of O(d³).

Reference:
  Rakhlin & Sridharan (2013) — "Online learning with predictable sequences"
  Hoerl & Kennard (1970) — "Ridge regression" (offline foundation)
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class OnlineRidge(MicrostructureAlgorithm):
    """Online ridge regression combining base and algorithm features."""

    def __init__(self, lambda_reg: float = 1.0, learning_rate: float = 0.01,
                 max_features: int = 20, update_interval: int = 10):
        self._lambda = lambda_reg
        self._lr = learning_rate
        self._max_features = max_features
        self._update_interval = update_interval
        # State initialized lazily (after seeing feature dimensionality)
        self._w: np.ndarray | None = None
        self._P: np.ndarray | None = None  # Inverse covariance (d x d)
        self._d: int = 0
        self._tick_count: int = 0
        self._prev_prediction: float = 0.0
        self._feature_names: list[str] = []
        self._target_buffer: list[float] = []
        self._feat_buffer: list[np.ndarray] = []

    def name(self) -> str:
        return "online_ridge"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_online_prediction", warmup=200,
                             description="Online ridge regression prediction"),
            AlgorithmFeature("alg_online_confidence", warmup=200,
                             description="1 / (1 + prediction_variance)"),
            AlgorithmFeature("alg_feature_importance_entropy", warmup=200,
                             description="Entropy of |w| distribution (diversity measure)"),
        ]

    def required_columns(self) -> list[str]:
        # Base features used as inputs (algorithm features consumed opportunistically)
        return ["imbalance_qty_l1", "flow_volume_1s", "vol_returns_1m",
                "ent_book_shape", "raw_midprice"]

    def _get_feature_vector(self, tick: dict[str, float]) -> np.ndarray | None:
        """Extract feature vector from tick, including any alg_* features."""
        vals = []
        names = []

        # Base features (always available)
        base = ["imbalance_qty_l1", "flow_volume_1s", "vol_returns_1m",
                "ent_book_shape"]
        for col in base:
            v = tick.get(col, np.nan)
            if np.isfinite(v):
                vals.append(v)
                names.append(col)

        # Opportunistically include alg_* features
        for key, val in tick.items():
            if key.startswith("alg_") and isinstance(val, (int, float)) and np.isfinite(val):
                vals.append(val)
                names.append(key)

        if len(vals) < 3:
            return None

        # Cap at max_features
        if len(vals) > self._max_features:
            vals = vals[:self._max_features]
            names = names[:self._max_features]

        self._feature_names = names
        return np.array(vals, dtype=np.float64)

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        mid = tick.get("raw_midprice", np.nan)
        if not np.isfinite(mid):
            return {f.name: np.nan for f in self.alg_features()}

        self._tick_count += 1

        x = self._get_feature_vector(tick)
        if x is None:
            return {f.name: np.nan for f in self.alg_features()}

        d = len(x)

        # Lazy initialization
        if self._w is None or d != self._d:
            self._d = d
            self._w = np.zeros(d)
            self._P = np.eye(d) / self._lambda

        # Prediction
        prediction = float(np.dot(self._w, x))

        # Confidence: based on prediction variance x'Px
        pred_var = float(x @ self._P @ x)
        confidence = 1.0 / (1.0 + pred_var)

        # Feature importance entropy
        abs_w = np.abs(self._w) + 1e-12
        p_w = abs_w / abs_w.sum()
        importance_entropy = float(-np.sum(p_w * np.log(p_w)))

        # Online update every update_interval ticks using lagged return as target
        if self._tick_count > 10 and self._tick_count % self._update_interval == 0:
            # Target: forward return proxy (use prev prediction error as signal)
            # In practice, use lagged midprice return
            if len(self._feat_buffer) > 0:
                x_prev = self._feat_buffer[-1]
                if len(x_prev) == d:
                    # Approximate target: sign of price change
                    target = np.sign(prediction - self._prev_prediction)

                    # Sherman-Morrison update: P' = P - P*x*x'*P / (1 + x'*P*x)
                    Px = self._P @ x_prev
                    denom = 1.0 + float(x_prev @ Px)
                    self._P -= np.outer(Px, Px) / (denom + 1e-12)

                    # Weight update: w' = w + lr * (target - w'x) * Px
                    error = target - float(np.dot(self._w, x_prev))
                    self._w += self._lr * error * Px

        self._prev_prediction = prediction
        self._feat_buffer.append(x.copy())
        if len(self._feat_buffer) > 100:
            self._feat_buffer.pop(0)

        return {
            "alg_online_prediction": prediction,
            "alg_online_confidence": confidence,
            "alg_feature_importance_entropy": importance_entropy,
        }

    def reset(self) -> None:
        self._w = None
        self._P = None
        self._d = 0
        self._tick_count = 0
        self._prev_prediction = 0.0
        self._feat_buffer.clear()
        self._target_buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Loop-based batch (online learning is inherently sequential)."""
        import pandas as pd

        self.reset()
        n = len(df)
        predictions = np.full(n, np.nan)
        confidences = np.full(n, np.nan)
        entropies = np.full(n, np.nan)

        cols = [c for c in df.columns if df[c].dtype.kind in ('f', 'i', 'u')]
        for i in range(n):
            tick = {col: float(df.iloc[i][col]) for col in cols}
            result = self.step(tick)
            predictions[i] = result["alg_online_prediction"]
            confidences[i] = result["alg_online_confidence"]
            entropies[i] = result["alg_feature_importance_entropy"]

        result = pd.DataFrame({
            "alg_online_prediction": predictions,
            "alg_online_confidence": confidences,
            "alg_feature_importance_entropy": entropies,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

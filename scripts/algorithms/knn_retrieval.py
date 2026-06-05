"""
Nearest-Neighbor State Retrieval
=================================

Non-parametric, buffer-based algorithm that finds historical states
most similar to the current market state (Mahalanobis distance) and
predicts forward returns from their outcomes. Adapts continuously
as the buffer grows — no offline training required.

Pipeline per bar:
  1. Whiten query vector via Ledoit-Wolf covariance Cholesky factor
  2. Find K nearest neighbors in KD-tree (Euclidean on whitened space)
  3. Apply time-decay weights: w_i = exp(-ln2 * age_i / halflife)
  4. Compute weighted expected_return, win_rate, confidence
  5. Gate by cost_threshold and win_rate_threshold

Output Features (4):
  alg_knn_signal          [-1, 1]   Signed directional signal
  alg_knn_expected_return (-inf,inf) Weighted mean forward return of neighbors
  alg_knn_win_rate        [0, 1]    Fraction of profitable neighbors
  alg_knn_confidence      [0, 1]    Inverse distance confidence

References:
  Cover, T. & Hart, P. (1967) — Nearest Neighbor Pattern Classification
  Mahalanobis, P.C. (1936) — On the Generalized Distance in Statistics
  Ledoit, O. & Wolf, M. (2004) — A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class KNNRetrieval(MicrostructureAlgorithm):
    """Non-parametric nearest-neighbor state retrieval.

    Maintains a ring buffer of (features, forward_return, bar_index)
    and rebuilds a KD-tree every refit_interval bars. Predicts from
    time-decay weighted neighbor outcomes.
    """

    bar_level = True

    FEATURE_COLS = [
        "ent_tick_1m_mean",
        "trend_hurst_300_mean",
        "vol_returns_5m_last",
        "toxic_vpin_50_mean",
        "imbalance_qty_l1_mean",
        "whale_net_flow_4h_sum",
        "regime_accumulation_score_mean",
    ]

    HORIZON = 20  # forward return horizon (bars)

    def __init__(
        self,
        k: int = 20,
        buffer_size: int = 5000,
        time_decay_halflife: int = 500,
        refit_interval: int = 100,
        min_buffer: int = 100,
        cost_threshold_bps: float = 2.0,
        win_rate_threshold: float = 0.60,
    ):
        self._k = k
        self._buffer_size = buffer_size
        self._halflife = time_decay_halflife
        self._refit_interval = refit_interval
        self._min_buffer = min_buffer
        self._cost_threshold = cost_threshold_bps / 10000.0
        self._win_rate_threshold = win_rate_threshold

        # Internal state
        self._features_buf: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._returns_buf: deque[float] = deque(maxlen=buffer_size)
        self._bar_index = 0
        self._bars_since_refit = 0

        # Covariance and KD-tree
        self._cholesky_inv = None  # whitening matrix
        self._kdtree = None

        # Pending forward returns (bar_idx -> midprice at entry)
        self._pending: deque[tuple[int, float]] = deque(maxlen=buffer_size)

    def name(self) -> str:
        return "knn_retrieval"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_knn_signal", warmup=0,
                             description="KNN directional signal [-1, +1]"),
            AlgorithmFeature("alg_knn_expected_return", warmup=0,
                             description="Weighted mean forward return of neighbors"),
            AlgorithmFeature("alg_knn_win_rate", warmup=0,
                             description="Fraction of profitable neighbors [0, 1]"),
            AlgorithmFeature("alg_knn_confidence", warmup=0,
                             description="Inverse distance confidence [0, 1]"),
        ]

    def required_columns(self) -> list[str]:
        return list(self.FEATURE_COLS) + ["raw_midprice_mean"]

    def _resolve_pending(self, current_mid: float):
        """Resolve pending forward returns for bars that are now old enough."""
        while self._pending and (self._bar_index - self._pending[0][0]) >= self.HORIZON:
            idx, entry_price = self._pending.popleft()
            if entry_price > 0 and np.isfinite(current_mid):
                fwd_ret = current_mid / entry_price - 1.0
                # Find the buffer position for this bar
                buf_idx = idx - (self._bar_index - len(self._returns_buf))
                if 0 <= buf_idx < len(self._returns_buf):
                    self._returns_buf[buf_idx] = fwd_ret

    def _refit(self):
        """Rebuild Ledoit-Wolf covariance and KD-tree from buffer."""
        from scipy.spatial import cKDTree

        n = len(self._features_buf)
        if n < self._min_buffer:
            return

        X = np.array(self._features_buf)

        # Ledoit-Wolf shrinkage covariance
        mean = X.mean(axis=0)
        Xc = X - mean
        S = (Xc.T @ Xc) / (n - 1)

        # Shrinkage toward diagonal
        trace_S = np.trace(S)
        p = X.shape[1]
        mu = trace_S / p
        delta = np.sum((S - mu * np.eye(p)) ** 2) / (n * p)
        shrinkage = min(1.0, delta / (trace_S ** 2 / p + 1e-10))
        S_shrunk = (1 - shrinkage) * S + shrinkage * mu * np.eye(p)

        # Cholesky for whitening
        try:
            L = np.linalg.cholesky(S_shrunk)
            self._cholesky_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            self._cholesky_inv = np.eye(p)

        # Build KD-tree on whitened features
        X_white = X @ self._cholesky_inv.T
        self._kdtree = cKDTree(X_white)
        self._bars_since_refit = 0

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        # Extract features
        x = np.array([tick.get(c, np.nan) for c in self.FEATURE_COLS])
        midprice = tick.get("raw_midprice_mean", np.nan)

        if not np.all(np.isfinite(x)) or not np.isfinite(midprice):
            self._bar_index += 1
            return nan_out

        # Resolve pending forward returns
        self._resolve_pending(midprice)

        # Add to buffer (return is NaN until resolved)
        self._features_buf.append(x.copy())
        self._returns_buf.append(np.nan)
        self._pending.append((self._bar_index, midprice))

        self._bar_index += 1
        self._bars_since_refit += 1

        # Refit if needed
        if self._bars_since_refit >= self._refit_interval or self._kdtree is None:
            self._refit()

        # Not enough data yet
        n = len(self._features_buf)
        if n < self._min_buffer or self._kdtree is None or self._cholesky_inv is None:
            return nan_out

        # Whiten query
        q = x @ self._cholesky_inv.T
        k = min(self._k, len(self._kdtree.data))

        # Query KD-tree
        dists, indices = self._kdtree.query(q, k=k)
        if k == 1:
            dists = np.array([dists])
            indices = np.array([indices])

        # Gather neighbor returns (skip NaN returns)
        returns = np.array([self._returns_buf[i] for i in indices])
        ages = np.array([n - 1 - i for i in indices], dtype=float)

        valid = np.isfinite(returns)
        if valid.sum() == 0:
            return nan_out

        returns_v = returns[valid]
        ages_v = ages[valid]
        dists_v = dists[valid]

        # Time-decay weights
        weights = np.exp(-np.log(2) * ages_v / self._halflife)
        weights /= weights.sum() + 1e-10

        # Weighted expected return
        expected_return = float(np.dot(weights, returns_v))

        # Win rate
        win_rate = float(np.dot(weights, (returns_v > 0).astype(float)))

        # Confidence: inverse mean distance (0 = far, 1 = identical)
        mean_dist = np.mean(dists_v) if len(dists_v) > 0 else 1.0
        confidence = float(1.0 / (1.0 + mean_dist))

        # Signal: gated by cost and win rate
        if abs(expected_return) > self._cost_threshold and win_rate > self._win_rate_threshold:
            signal = np.clip(np.sign(expected_return) * confidence, -1.0, 1.0)
        else:
            signal = 0.0

        return {
            "alg_knn_signal": signal,
            "alg_knn_expected_return": expected_return,
            "alg_knn_win_rate": win_rate,
            "alg_knn_confidence": confidence,
        }

    def reset(self) -> None:
        self._features_buf.clear()
        self._returns_buf.clear()
        self._pending.clear()
        self._bar_index = 0
        self._bars_since_refit = 0
        self._cholesky_inv = None
        self._kdtree = None

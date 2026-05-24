"""
Cascade Probability Estimator

Online logistic regression that predicts the probability of a liquidation
cascade event from the 8 heatmap spatial features plus 3 interaction terms.

Target: binary cascade event y_t = 1 if |price_move| > X% AND liq_volume > Y
within T ticks. Uses delayed SGD (target observed at t + T).

References:
  Cont & Wagalath (2016) - Fire sales forensics
  Brunnermeier & Pedersen (2009) - Market liquidity and funding liquidity
  See docs/liquidity_heatmap_model.md for full specification.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


def _sigmoid(z: float) -> float:
    """Logistic sigmoid, clipped for numerical stability."""
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + np.exp(-z))


class WelfordNormalizer:
    """Online mean/variance tracker (Welford's algorithm)."""

    def __init__(self, d: int):
        self.n = 0
        self.mean = np.zeros(d)
        self.M2 = np.zeros(d)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x - self.mean
        std = np.sqrt(self.M2 / (self.n - 1))
        std = np.where(std < 1e-10, 1.0, std)
        return (x - self.mean) / std


@register
class CascadeProbability(MicrostructureAlgorithm):
    """Online logistic regression for liquidation cascade detection.

    Consumes 8 heatmap features (hm_*) + 3 existing features to build
    an 11-dimensional feature vector with interaction terms. Outputs
    cascade probability, predicted direction, and model diagnostic.
    """

    # Heatmap spatial features (from Rust heatmap.rs)
    HEATMAP_COLS = [
        "hm_nearest_cluster_dist",
        "hm_cluster_mass_ratio",
        "hm_cascade_chain_length",
        "hm_asymmetric_cascade_pot",
        "hm_absorption_capacity",
        "hm_cluster_velocity",
        "hm_mass_weighted_distance",
        "hm_heatmap_entropy",
    ]

    # Existing features used for interaction terms
    CONTEXT_COLS = [
        "illiq_composite",
        "ctx_funding_rate",
    ]

    # Cascade event definition
    CASCADE_HORIZON = 3000       # 5 min at 100ms ticks
    CASCADE_PRICE_THRESH = 0.03  # 3% log-return
    CASCADE_VOL_THRESH = 500_000 # $500K liquidation volume

    def __init__(
        self,
        lr: float = 0.01,
        lr_decay: float = 0.9999,
        l2_reg: float = 1e-3,
    ):
        self._lr0 = lr
        self._lr = lr
        self._lr_decay = lr_decay
        self._l2 = l2_reg
        self._d = 11  # 8 heatmap + 3 interactions
        self._beta: np.ndarray | None = None
        self._beta0: float = 0.0
        self._normalizer: WelfordNormalizer | None = None
        self._tick_count: int = 0

        # Delayed target buffer: (features, midprice, timestamp)
        self._buffer: deque[tuple[np.ndarray, float, int]] = deque()

    def name(self) -> str:
        return "cascade_probability"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_cascade_prob", warmup=3000,
                             description="P(cascade) from online logistic model"),
            AlgorithmFeature("alg_cascade_direction", warmup=3000,
                             description="Predicted cascade direction (sign of ACP)"),
            AlgorithmFeature("alg_cascade_beta_norm", warmup=6000,
                             description="||beta|| model confidence diagnostic"),
        ]

    def required_columns(self) -> list[str]:
        return self.HEATMAP_COLS + self.CONTEXT_COLS + ["raw_midprice"]

    def _build_features(self, tick: dict[str, float]) -> np.ndarray | None:
        """Build 11-dim feature vector: 8 heatmap + 3 interaction terms."""
        hm = np.array([tick.get(c, np.nan) for c in self.HEATMAP_COLS])
        if not np.all(np.isfinite(hm)):
            return None

        illiq = tick.get("illiq_composite", np.nan)
        funding = tick.get("ctx_funding_rate", np.nan)
        if not (np.isfinite(illiq) and np.isfinite(funding)):
            return None

        # Interaction terms (see docs/liquidity_heatmap_model.md §3.2)
        chain_length = hm[2]      # F3
        absorption = hm[4]        # F5
        velocity = hm[5]          # F6
        acp = hm[3]               # F4

        # L * A^{-1}: long chain with insufficient damping
        inv_absorption = 1.0 / max(absorption, 1e-6)
        interact_chain_absorb = chain_length * inv_absorption

        # v * I: cluster approaching in thin market
        interact_vel_illiq = velocity * illiq

        # ACP * sgn(f): directional alignment
        interact_acp_funding = acp * np.sign(funding) if funding != 0.0 else 0.0

        x = np.concatenate([hm, [interact_chain_absorb, interact_vel_illiq,
                                  interact_acp_funding]])
        return x

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        nan_out = {f.name: np.nan for f in self.alg_features()}

        mid = tick.get("raw_midprice", np.nan)
        if not np.isfinite(mid):
            return nan_out

        self._tick_count += 1

        x = self._build_features(tick)
        if x is None:
            return nan_out

        d = len(x)

        # Lazy init
        if self._beta is None:
            self._d = d
            self._beta = np.zeros(d)
            self._normalizer = WelfordNormalizer(d)

        # Handle dimension change
        if d != self._d:
            return nan_out

        # Normalize
        self._normalizer.update(x)
        x_norm = self._normalizer.normalize(x)

        # Predict
        z = self._beta0 + float(np.dot(self._beta, x_norm))
        prob = _sigmoid(z)

        # Direction from ACP (F4)
        acp = tick.get("hm_asymmetric_cascade_pot", 0.0)
        direction = float(np.sign(acp)) if abs(acp) > 0.01 else 0.0

        # Model diagnostic
        beta_norm = float(np.linalg.norm(self._beta))

        # Buffer for delayed SGD update
        self._buffer.append((x_norm.copy(), mid, self._tick_count))

        # Apply delayed SGD updates for observations where target is now available
        while self._buffer and (self._tick_count - self._buffer[0][2]) >= self.CASCADE_HORIZON:
            x_old, mid_old, t_old = self._buffer.popleft()
            # Compute target: did a cascade happen in the window [t_old, t_old + T]?
            log_return = np.log(mid / mid_old) if mid_old > 0 else 0.0
            # Approximate cascade event from price move alone
            # (liquidation volume not available in tick data — use price threshold as proxy)
            y = 1.0 if abs(log_return) > self.CASCADE_PRICE_THRESH else 0.0

            # SGD step on cross-entropy loss with L2
            p_old = _sigmoid(self._beta0 + float(np.dot(self._beta, x_old)))
            grad = p_old - y  # dL/dz for cross-entropy

            self._beta -= self._lr * (grad * x_old + self._l2 * self._beta)
            self._beta0 -= self._lr * grad

            # Decay learning rate
            self._lr *= self._lr_decay

        # Cap buffer size (prevent unbounded growth if price data stalls)
        while len(self._buffer) > self.CASCADE_HORIZON * 2:
            self._buffer.popleft()

        return {
            "alg_cascade_prob": prob,
            "alg_cascade_direction": direction,
            "alg_cascade_beta_norm": beta_norm,
        }

    def reset(self) -> None:
        self._beta = None
        self._beta0 = 0.0
        self._normalizer = None
        self._lr = self._lr0
        self._tick_count = 0
        self._buffer.clear()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Sequential batch (online learning is inherently sequential)."""
        import pandas as pd

        self.reset()
        n = len(df)
        probs = np.full(n, np.nan)
        directions = np.full(n, np.nan)
        beta_norms = np.full(n, np.nan)

        cols = [c for c in df.columns if df[c].dtype.kind in ("f", "i", "u")]
        for i in range(n):
            tick = {col: float(df.iloc[i][col]) for col in cols}
            result = self.step(tick)
            probs[i] = result["alg_cascade_prob"]
            directions[i] = result["alg_cascade_direction"]
            beta_norms[i] = result["alg_cascade_beta_norm"]

        result_df = pd.DataFrame({
            "alg_cascade_prob": probs,
            "alg_cascade_direction": directions,
            "alg_cascade_beta_norm": beta_norms,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan
        return result_df

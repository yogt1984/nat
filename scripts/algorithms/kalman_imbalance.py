"""
Kalman-Filtered Imbalance Algorithm

Wraps the existing OUKalmanFilter from scripts/kalman/ou_filter.py.
Produces: filtered signal, uncertainty, innovation, signal strength.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

# Import existing Kalman filter
from kalman.ou_filter import OUKalmanFilter, estimate_ou_params, auto_tune_filter


@register
class KalmanImbalance(MicrostructureAlgorithm):
    """OU Kalman filter on L1 imbalance."""

    def __init__(self, theta: float = 0.1, r_mult: float = 1.0, dt: float = 0.1,
                 auto_tune: bool = True):
        self._theta = theta
        self._r_mult = r_mult
        self._dt = dt
        self._auto_tune = auto_tune
        self._kf = OUKalmanFilter(
            theta=theta,
            sigma_process=0.01,
            sigma_obs=0.1 * r_mult,
            dt=dt,
        )

    def name(self) -> str:
        return "kalman_imbalance"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_kalman_filtered_imb", warmup=50,
                             description="OU Kalman filtered L1 imbalance"),
            AlgorithmFeature("alg_kalman_uncertainty", warmup=50,
                             description="Kalman filter uncertainty (P)"),
            AlgorithmFeature("alg_kalman_innovation", warmup=50,
                             description="Innovation (obs - prediction)"),
            AlgorithmFeature("alg_kalman_signal_strength", warmup=50,
                             description="Filtered / sqrt(uncertainty)"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        z = tick.get("imbalance_qty_l1", np.nan)
        if not np.isfinite(z):
            return {f.name: np.nan for f in self.alg_features()}

        x, P, innov = self._kf.step(z)
        strength = x / (P**0.5 + 1e-12)

        return {
            "alg_kalman_filtered_imb": x,
            "alg_kalman_uncertainty": P,
            "alg_kalman_innovation": innov,
            "alg_kalman_signal_strength": strength,
        }

    def reset(self) -> None:
        self._kf.reset()

    def _make_kf(self, obs_window: np.ndarray) -> OUKalmanFilter:
        """Create a Kalman filter, auto-tuned if enabled."""
        if self._auto_tune:
            valid = obs_window[np.isfinite(obs_window)]
            if len(valid) > 200:
                return auto_tune_filter(valid, dt=self._dt)
        return OUKalmanFilter(
            theta=self._theta,
            sigma_process=0.01,
            sigma_obs=0.1 * self._r_mult,
            dt=self._dt,
        )

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override with regime-conditional re-estimation."""
        import pandas as pd
        from .regime_retune import has_regime_column, segment_by_regime, REGIME_COL

        obs = df["imbalance_qty_l1"].values.astype(np.float64)
        n = len(obs)
        states = np.full(n, np.nan)
        uncerts = np.full(n, np.nan)
        innovs = np.full(n, np.nan)

        # Regime-conditional: re-tune at each regime transition
        if self._auto_tune and has_regime_column(df):
            segments = segment_by_regime(df[REGIME_COL].values)
            for start, end, _regime in segments:
                kf = self._make_kf(obs[start:end])
                s, u, iv = kf.filter_series_full(obs[start:end])
                states[start:end] = s
                uncerts[start:end] = u
                innovs[start:end] = iv
        else:
            kf = self._make_kf(obs)
            states, uncerts, innovs = kf.filter_series_full(obs)

        strengths = states / (np.sqrt(uncerts) + 1e-12)

        # NaN-out where input was NaN
        nan_mask = ~np.isfinite(obs)
        states[nan_mask] = np.nan
        uncerts[nan_mask] = np.nan
        innovs[nan_mask] = np.nan
        strengths[nan_mask] = np.nan

        result = pd.DataFrame({
            "alg_kalman_filtered_imb": states,
            "alg_kalman_uncertainty": uncerts,
            "alg_kalman_innovation": innovs,
            "alg_kalman_signal_strength": strengths,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result

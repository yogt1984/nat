"""
Kalman-Filtered Imbalance Algorithm

Wraps the existing OUKalmanFilter from scripts/kalman/ou_filter.py.
Produces: filtered signal, uncertainty, innovation, signal strength.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

# Import existing Kalman filter
sys.path.insert(0, str(Path(__file__).parent.parent))
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

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override using filter_series_full."""
        import pandas as pd

        obs = df["imbalance_qty_l1"].values.astype(np.float64)

        # Auto-tune filter parameters from data
        if self._auto_tune:
            valid_obs = obs[np.isfinite(obs)]
            if len(valid_obs) > 200:
                kf = auto_tune_filter(valid_obs, dt=self._dt)
            else:
                kf = OUKalmanFilter(
                    theta=self._theta,
                    sigma_process=0.01,
                    sigma_obs=0.1 * self._r_mult,
                    dt=self._dt,
                )
        else:
            kf = OUKalmanFilter(
                theta=self._theta,
                sigma_process=0.01,
                sigma_obs=0.1 * self._r_mult,
                dt=self._dt,
            )

        # filter_series_full handles the loop in Python but it's a tight
        # numeric loop (no dict/DataFrame overhead per tick)
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

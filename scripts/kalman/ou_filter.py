"""
Ornstein-Uhlenbeck Kalman Filter for L1 book imbalance.

Extracts the slow (ultra-low frequency) component of imbalance_qty_l1
using a scalar Kalman filter with OU process prior.

State-space model:
    State:       x(t) = slow imbalance component
    Transition:  x(t+1) = (1 - theta*dt)*x(t) + mu*theta*dt + w,  w ~ N(0, Q)
    Observation: z(t) = x(t) + v,  v ~ N(0, R)

OU parameters from spectral analysis (2026-05-14):
    theta ≈ 0.1 (half-life 5-7s), brown noise slope -1.86
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OUParams:
    """Estimated OU process parameters."""
    theta: float   # mean-reversion speed (1/s)
    mu: float      # long-run mean
    sigma: float   # volatility (diffusion coefficient)
    half_life: float  # ln(2) / theta


class OUKalmanFilter:
    """Scalar Kalman filter with OU prior for imbalance slow component."""

    def __init__(self, theta: float, sigma_process: float, sigma_obs: float,
                 dt: float = 0.1, mu: float = 0.0):
        self.theta = theta
        self.sigma_process = sigma_process
        self.sigma_obs = sigma_obs
        self.dt = dt
        self.mu = mu

        # Discrete-time OU transition: F = exp(-theta * dt) ≈ 1 - theta*dt
        self.F = np.exp(-theta * dt)
        # Process noise variance: Q = sigma^2 * (1 - F^2) / (2*theta)
        self.Q = sigma_process**2 * (1 - self.F**2) / (2 * theta + 1e-12)
        # Measurement noise variance
        self.R = sigma_obs**2

        # State estimate and covariance
        self.x = mu       # state estimate
        self.P = self.Q   # state covariance (start at steady-state)

    def predict(self) -> tuple[float, float]:
        """Predict step: propagate state through OU transition."""
        self.x = self.F * self.x + (1 - self.F) * self.mu
        self.P = self.F**2 * self.P + self.Q
        return self.x, self.P

    def update(self, z: float) -> tuple[float, float]:
        """Update step: incorporate observation z.

        Returns (filtered_state, innovation).
        """
        # Innovation
        y = z - self.x
        # Innovation covariance
        S = self.P + self.R
        # Kalman gain
        K = self.P / S
        # Update state
        self.x = self.x + K * y
        self.P = (1 - K) * self.P
        return self.x, y

    def step(self, z: float) -> tuple[float, float, float]:
        """Single predict+update cycle.

        Returns (filtered_state, uncertainty, innovation).
        """
        self.predict()
        x_filt, innov = self.update(z)
        return x_filt, self.P, innov

    def filter_series(self, observations: np.ndarray) -> np.ndarray:
        """Run filter over an array of observations.

        Returns array of filtered states (same length as input).
        """
        n = len(observations)
        filtered = np.empty(n)
        for i in range(n):
            self.predict()
            self.update(observations[i])
            filtered[i] = self.x
        return filtered

    def filter_series_full(self, observations: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run filter, return filtered states, uncertainties, and innovations."""
        n = len(observations)
        states = np.empty(n)
        uncerts = np.empty(n)
        innovs = np.empty(n)
        for i in range(n):
            x, P, innov = self.step(observations[i])
            states[i] = x
            uncerts[i] = P
            innovs[i] = innov
        return states, uncerts, innovs

    def reset(self, x0: float = None, P0: float = None):
        """Reset filter state."""
        self.x = x0 if x0 is not None else self.mu
        self.P = P0 if P0 is not None else self.Q


def estimate_ou_params(series: np.ndarray, dt: float = 0.1) -> OUParams:
    """Estimate OU parameters from time series via regression method.

    Uses the discrete AR(1) representation:
        x(t+1) = a + b*x(t) + eps
    where b = exp(-theta*dt), a = mu*(1-b).
    """
    x = series[:-1]
    y = series[1:]

    # Remove NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 100:
        return OUParams(theta=0.1, mu=0.0, sigma=0.01, half_life=6.93)

    # OLS regression: y = a + b*x
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-12)
    a = y_mean - b * x_mean

    # Recover OU parameters
    b = np.clip(b, 1e-6, 1 - 1e-6)  # ensure 0 < b < 1
    theta = -np.log(b) / dt
    mu = a / (1 - b)

    # Residual variance → diffusion coefficient
    residuals = y - (a + b * x)
    var_eps = np.var(residuals)
    # sigma^2 = 2*theta*var_eps / (1 - b^2)
    sigma = np.sqrt(2 * theta * var_eps / (1 - b**2 + 1e-12))

    half_life = np.log(2) / (theta + 1e-12)

    return OUParams(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
    )


def auto_tune_filter(series: np.ndarray, dt: float = 0.1) -> OUKalmanFilter:
    """Estimate OU params from data and return a tuned Kalman filter."""
    params = estimate_ou_params(series, dt)

    # Process noise = OU diffusion
    sigma_process = params.sigma

    # Observation noise: estimated from high-frequency residual
    # Use the difference between consecutive observations minus OU prediction
    F = np.exp(-params.theta * dt)
    predicted = F * series[:-1] + (1 - F) * params.mu
    obs_residuals = series[1:] - predicted
    sigma_obs = float(np.std(obs_residuals[np.isfinite(obs_residuals)]))

    return OUKalmanFilter(
        theta=params.theta,
        sigma_process=sigma_process,
        sigma_obs=max(sigma_obs, 1e-6),
        dt=dt,
        mu=params.mu,
    )

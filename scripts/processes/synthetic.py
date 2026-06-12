"""
Synthetic data with planted signal — the conformance gate for processes.

Every evaluation process must flag `feat_signal` (constructed to predict
forward returns at a known horizon with a known IC) as informative, and must
NOT flag `feat_noise` or a shuffled copy. Library code, not test code: the
Stage-3 literature-to-process pipeline reuses these fixtures to auto-gate
generated processes before they touch real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ProcessContext

PRICE_COL = "raw_midprice_close"


def make_planted_frame(
    n: int = 3000,
    ic: float = 0.15,
    horizon: int = 4,
    seed: int = 7,
    sigma: float = 0.001,
) -> pd.DataFrame:
    """Bar-level frame with a feature that predicts the `horizon`-bar return.

    Columns:
      bar_start            15-min bar timestamps
      symbol               "SYN"
      raw_midprice_close   geometric random walk
      feat_signal          standardized forward return + calibrated noise
                           (Pearson corr with fwd return ~= `ic`)
      feat_noise           iid N(0,1), independent of returns
      feat_dead            all-NaN (the K2 dead-column case)
      feat_const           constant
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    prices = 100.0 * np.exp(np.cumsum(eps))

    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = prices[horizon:] / prices[: n - horizon] - 1.0

    z = np.zeros(n)
    valid = ~np.isnan(fwd)
    z[valid] = (fwd[valid] - np.nanmean(fwd)) / (np.nanstd(fwd) + 1e-15)

    # corr(z + k*eta, z) = 1/sqrt(1+k^2)  =>  k = sqrt(1/ic^2 - 1)
    k = np.sqrt(max(1.0 / (ic * ic) - 1.0, 0.0))
    feat_signal = z + k * rng.normal(0.0, 1.0, size=n)
    # Tail bars have no forward return — pure noise there, column stays full
    feat_signal[~valid] = k * rng.normal(0.0, 1.0, size=int((~valid).sum()))

    return pd.DataFrame({
        "bar_start": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "symbol": "SYN",
        PRICE_COL: prices,
        "feat_signal": feat_signal,
        "feat_noise": rng.normal(0.0, 1.0, size=n),
        "feat_dead": np.full(n, np.nan),
        "feat_const": np.ones(n),
    })


def make_ou_series(
    n: int = 20000,
    theta: float = 0.05,
    sigma: float = 1.0,
    seed: int = 7,
) -> np.ndarray:
    """Discrete OU process x[t+1] = (1-theta)x[t] + sigma*eta.

    Analytic ACF half-life = log(2)/(-log(1-theta)) ~= log(2)/theta steps.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = (1.0 - theta) * x[t - 1] + sigma * rng.normal()
    return x


def make_ar1_coupled(
    n: int = 4000,
    coupling: float = 0.6,
    phi: float = 0.3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """(source, target) where target[t] = phi*target[t-1] + coupling*source[t-1] + eta.

    Information flows source -> target only; transfer entropy must be
    positive in that direction and ~zero in reverse.
    """
    rng = np.random.default_rng(seed)
    source = rng.normal(size=n)
    target = np.zeros(n)
    for t in range(1, n):
        target[t] = phi * target[t - 1] + coupling * source[t - 1] + rng.normal()
    return source, target


def make_test_context(
    horizons: dict[str, int] | None = None,
    price_col: str = PRICE_COL,
    timeframe: str = "15min",
    target_col: str | None = None,
) -> ProcessContext:
    """ProcessContext for unit tests — near-zero costs so cost gates pass."""
    return ProcessContext(
        symbol="SYN",
        timeframe=timeframe,
        price_col=price_col,
        horizons=horizons or {"1bar": 1, "4bar": 4, "16bar": 16},
        costs={"hyperliquid": {"taker_bps": 0.01, "maker_bps": 0.0,
                               "round_trip_taker_bps": 0.02}},
        data_dir="synthetic",
        target_col=target_col,
    )

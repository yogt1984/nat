"""Shared fixtures for algorithm tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_synthetic_ticks(n: int = 1000, columns: list[str] | None = None,
                         seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame with realistic synthetic data for the requested columns."""
    rng = np.random.default_rng(seed)
    data = {}

    if columns is None:
        columns = []

    for col in columns:
        if "midprice" in col:
            # GBM-like price series starting at 50000
            returns = rng.normal(0, 0.0001, n)
            data[col] = 50000 * np.exp(np.cumsum(returns))
        elif "spread" in col and "decomp" not in col:
            data[col] = np.abs(rng.normal(0.5, 0.2, n))
        elif "imbalance" in col or "pressure" in col:
            data[col] = rng.uniform(-1, 1, n)
        elif "depth" in col or "orders" in col:
            data[col] = np.abs(rng.exponential(100, n))
        elif "flow_count" in col:
            data[col] = rng.poisson(5, n).astype(float)
        elif "flow_volume" in col:
            data[col] = np.abs(rng.exponential(1000, n))
        elif "flow_aggressor" in col:
            data[col] = rng.uniform(0, 1, n)
        elif "flow_intensity" in col:
            data[col] = np.abs(rng.exponential(2, n))
        elif "vol_returns" in col:
            data[col] = rng.normal(0, 0.001, n)
        elif "vol_" in col:
            data[col] = np.abs(rng.normal(0.01, 0.005, n))
        elif "ent_" in col:
            data[col] = rng.uniform(0, 1, n)
        elif "toxic_vpin" in col:
            data[col] = rng.uniform(0, 1, n)
        elif "toxic_" in col:
            data[col] = np.abs(rng.normal(0.001, 0.0005, n))
        elif "ctx_funding" in col:
            data[col] = rng.normal(0.0001, 0.0005, n)
        elif "ctx_premium" in col:
            data[col] = rng.normal(0, 5, n)
        elif "ctx_open_interest" in col:
            data[col] = np.abs(rng.normal(1e8, 1e7, n))
        elif "ctx_oi_change" in col:
            data[col] = rng.normal(0, 1e5, n)
        elif "trend_momentum" in col or "trend_hurst" in col:
            data[col] = rng.normal(0, 0.1, n)
            data[col] = np.cumsum(data[col]) * 0.01  # AR-like
        elif "trend_" in col:
            data[col] = rng.normal(0, 0.01, n)
        elif "flow_vwap" in col:
            data[col] = rng.normal(50000, 10, n)
        else:
            data[col] = rng.normal(0, 1, n)

    df = pd.DataFrame(data)
    df.index = pd.RangeIndex(n)
    return df

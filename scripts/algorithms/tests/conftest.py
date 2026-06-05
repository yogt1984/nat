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


def make_bar_df(n_bars: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bar-aggregated DataFrame.

    Mimics output of aggregate_bars() with _mean/_std/_last/_sum suffixes.
    """
    rng = np.random.default_rng(seed)

    midprice = 50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars)))

    data = {
        "bar_start": np.arange(n_bars) * 300_000_000_000,  # 5min in ns
        "bar_end": (np.arange(n_bars) + 1) * 300_000_000_000,
        "tick_count": rng.poisson(3000, n_bars),
        # Price OHLC
        "raw_midprice_mean": midprice,
        "raw_midprice_open": midprice * (1 + rng.normal(0, 0.0002, n_bars)),
        "raw_midprice_high": midprice * (1 + np.abs(rng.normal(0, 0.001, n_bars))),
        "raw_midprice_low": midprice * (1 - np.abs(rng.normal(0, 0.001, n_bars))),
        "raw_midprice_close": midprice * (1 + rng.normal(0, 0.0003, n_bars)),
        # Entropy
        "ent_tick_1m_mean": rng.uniform(0, 1, n_bars),
        "ent_tick_1m_std": np.abs(rng.normal(0.1, 0.05, n_bars)),
        "ent_permutation_returns_16_mean": rng.uniform(0, 1, n_bars),
        # Imbalance
        "imbalance_qty_l1_mean": rng.normal(0, 0.3, n_bars),
        "imbalance_qty_l1_std": np.abs(rng.normal(0.1, 0.05, n_bars)),
        # Trend
        "trend_hurst_300_mean": rng.uniform(0.3, 0.7, n_bars),
        # Toxicity
        "toxic_vpin_50_mean": rng.uniform(0, 1, n_bars),
        # Whale flow
        "whale_net_flow_4h_sum": rng.normal(0, 1000, n_bars),
        # Volatility
        "vol_returns_5m_last": rng.normal(0, 0.001, n_bars),
        "vol_returns_5m_mean": rng.normal(0, 0.001, n_bars),
        # Regime
        "regime_accumulation_score_mean": rng.uniform(0, 1, n_bars),
        # Medium-frequency EMA
        "mf_ema_15m_last": midprice * (1 + rng.normal(0, 0.0005, n_bars)),
        # Trend momentum
        "trend_momentum_300_mean": rng.normal(0, 0.01, n_bars),
        # Bollinger %B
        "mf_bb_pctb_5m_last": rng.uniform(0, 1, n_bars),
        # Regime label (from RSM or GMM)
        "alg_rsm_regime_last": rng.choice([0, 1, 2, 3, 4, 5], n_bars).astype(float),
        "alg_rsm_confidence_last": rng.uniform(0.3, 0.9, n_bars),
    }

    return pd.DataFrame(data)


def make_forward_returns(bars_df: pd.DataFrame, horizon: int = 20) -> np.ndarray:
    """Compute forward returns from midprice_mean column.

    fwd_return[t] = midprice_mean[t + horizon] / midprice_mean[t] - 1
    Last `horizon` rows are NaN.
    """
    mid = bars_df["raw_midprice_mean"].values
    fwd = np.full(len(mid), np.nan)
    fwd[:-horizon] = mid[horizon:] / mid[:-horizon] - 1
    return fwd


def make_labeled_df(bars_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Merge bars with binary labels for training pipeline tests.

    Label = 1 if forward return > 0, else 0.
    """
    fwd = make_forward_returns(bars_df, horizon)
    df = bars_df.copy()
    df["fwd_return"] = fwd
    df["label"] = (fwd > 0).astype(float)
    df.loc[np.isnan(fwd), "label"] = np.nan
    return df

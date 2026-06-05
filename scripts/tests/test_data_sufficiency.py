"""Unit tests for data sufficiency checks."""

import numpy as np
import pandas as pd
import pytest

from check_data_sufficiency import (
    check_bar_count,
    check_fold_sizes,
    check_label_balance,
    check_nan_rates,
    run_all_checks,
    MIN_BARS,
    MIN_FOLD_SIZE,
)


def _make_bars(n_bars: int, nan_rate: float = 0.0) -> pd.DataFrame:
    """Create a synthetic bar DataFrame for testing."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "raw_midprice_mean": 100 + rng.standard_normal(n_bars).cumsum() * 0.01,
        "ent_tick_1m_mean": rng.uniform(0, 1, n_bars),
        "trend_hurst_300_mean": rng.uniform(0.3, 0.7, n_bars),
        "toxic_vpin_50_mean": rng.beta(2, 5, n_bars),
        "whale_net_flow_4h_mean": rng.standard_normal(n_bars) * 1000,
        "vol_returns_5m_mean": np.abs(rng.standard_normal(n_bars)) * 0.01,
        "regime_accumulation_score_mean": rng.uniform(0, 1, n_bars),
        "imbalance_qty_l1_mean": rng.standard_normal(n_bars),
    })
    if nan_rate > 0:
        mask = rng.random(n_bars) < nan_rate
        df.loc[mask, "ent_tick_1m_mean"] = np.nan
    return df


def test_bar_count_pass():
    """DataFrame with 5000 rows passes bar count check."""
    result = check_bar_count(5000)
    assert result["passed"] is True
    assert result["value"] == 5000


def test_bar_count_fail():
    """DataFrame with 3000 rows fails bar count check."""
    result = check_bar_count(3000)
    assert result["passed"] is False


def test_label_balance_pass():
    """Forward returns with ~48% positive rate passes."""
    rng = np.random.default_rng(42)
    # Slightly positive-biased so ~48-52% are positive
    fwd = pd.Series(rng.standard_normal(5000) * 0.01)
    result = check_label_balance(fwd)
    assert result["passed"] is True
    assert result["warning"] is False


def test_label_balance_warn():
    """Forward returns with 35% positive rate warns."""
    rng = np.random.default_rng(42)
    # Negative-biased
    fwd = pd.Series(rng.standard_normal(5000) * 0.01 - 0.005)
    pos_rate = (fwd > 0).mean()
    # If the random draw doesn't give us <40%, force it
    if pos_rate >= 0.40:
        # Create explicitly imbalanced
        fwd = pd.Series([-0.01] * 3250 + [0.01] * 1750)
    result = check_label_balance(fwd)
    assert result["warning"] is True
    assert result["passed"] is True  # Warning only, doesn't block


def test_nan_rate_pass():
    """Feature columns with 2% NaN passes."""
    df = _make_bars(5000, nan_rate=0.02)
    result = check_nan_rates(df, ["ent_tick_1m"])
    assert result["passed"] is True


def test_nan_rate_fail():
    """Feature column with 10% NaN fails."""
    df = _make_bars(5000, nan_rate=0.10)
    result = check_nan_rates(df, ["ent_tick_1m"])
    assert result["passed"] is False


def test_fold_size_pass():
    """8000 bars / 4 folds = 2000 per fold passes."""
    result = check_fold_sizes(8000, n_folds=4)
    assert result["passed"] is True
    assert result["value"] == 2000


def test_fold_size_fail():
    """1500 bars / 4 folds = 375 per fold fails."""
    result = check_fold_sizes(1500, n_folds=4)
    assert result["passed"] is False
    assert result["value"] == 375


def test_json_output_structure():
    """run_all_checks output has expected keys."""
    df = _make_bars(5000)
    result = run_all_checks(df)
    assert "bar_count" in result
    assert "checks" in result
    assert "sufficient" in result
    assert isinstance(result["checks"], list)
    check_names = {c["check"] for c in result["checks"]}
    assert check_names == {"bar_count", "label_balance", "nan_rates", "fold_sizes"}


def test_sufficient_overall():
    """Clean DataFrame with 5000 bars passes all checks."""
    df = _make_bars(5000)
    result = run_all_checks(df)
    assert result["sufficient"] is True


def test_insufficient_overall():
    """DataFrame with only 2000 bars fails overall."""
    df = _make_bars(2000)
    result = run_all_checks(df)
    assert result["sufficient"] is False

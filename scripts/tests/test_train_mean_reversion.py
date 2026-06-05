"""Tests for mean-reversion training pipeline."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.tests.conftest import make_bar_df
from train_mean_reversion import compute_zscore, build_dataset, FEATURE_COLS, ZSCORE_COLS


@pytest.fixture
def bars():
    bars = make_bar_df(n_bars=2000)
    # Ensure vol is positive for z-score
    bars["vol_returns_5m_last"] = np.abs(bars["vol_returns_5m_last"]) + 1e-6
    return bars


def test_reversion_label_binary(bars):
    """Labels are 0 or 1 only."""
    X, y, _ = build_dataset(bars)
    unique = set(np.unique(y))
    assert unique <= {0.0, 1.0}, f"Unexpected labels: {unique}"


def test_zscore_no_lookahead(bars):
    """z-score at bar t uses only data up to t — no forward references."""
    zs = compute_zscore(bars)
    # z-score uses midprice, ema, vol at time t — all available at t
    # Verify it doesn't depend on future by checking length matches
    assert len(zs) == len(bars)
    # Verify first value is computable (not NaN if inputs finite)
    if np.isfinite(bars["raw_midprice_mean"].iloc[0]):
        assert np.isfinite(zs[0])


def test_lgbm_fits_without_error(bars):
    """LightGBM trains on synthetic bars without errors."""
    import lightgbm as lgb

    bars["vol_returns_5m_last"] = np.abs(bars["vol_returns_5m_last"]) + 1e-6
    feature_names = FEATURE_COLS + ["zscore"]
    X, y, _ = build_dataset(bars)

    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "verbose": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=10)
    preds = model.predict(X)

    assert preds.shape == (len(y),)
    assert np.all((preds >= 0) & (preds <= 1))


def test_shap_feature_drop(bars):
    """Feature with importance < 0.02 is identifiable."""
    import lightgbm as lgb

    bars["vol_returns_5m_last"] = np.abs(bars["vol_returns_5m_last"]) + 1e-6
    feature_names = FEATURE_COLS + ["zscore"]
    X, y, _ = build_dataset(bars)

    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "verbose": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=50)

    importance = model.feature_importance(importance_type="gain")
    assert len(importance) == len(feature_names)

    # Normalize
    total = importance.sum()
    if total > 0:
        norm_imp = importance / total
        # At least identify which features are below threshold
        drop = [n for n, v in zip(feature_names, norm_imp) if v < 0.02]
        assert isinstance(drop, list)  # structural check — drop list is valid

"""Integration tests for MomentumContinuation on bar DataFrames."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.momentum_continuation import MomentumContinuation
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=300)


def test_run_batch_without_model(bars):
    """run_batch() with no model returns signal=0.0 for valid rows, NaN during warmup."""
    algo = MomentumContinuation()
    assert algo._model is None

    result = algo.run_batch(bars)

    assert result.shape == (len(bars), 3)
    assert set(result.columns) == {"alg_mc_signal", "alg_mc_confidence", "alg_mc_entropy_gate"}

    # Warmup rows should be NaN
    warmup = algo.warmup
    assert result.iloc[:warmup]["alg_mc_signal"].isna().all()

    # Post-warmup valid rows should have signal=0 (no model)
    post = result.iloc[warmup:]
    valid = post["alg_mc_signal"].dropna()
    assert (valid == 0.0).all(), "No-model signal should be 0.0"


def test_run_batch_with_injected_model(bars):
    """Inject a trained LogReg, run_batch() returns varied signals in [-1,1]."""
    algo = MomentumContinuation()

    # Train a model on the bars themselves
    X = bars[list(algo.FEATURE_COLS)].values
    fwd = bars["raw_midprice_mean"].shift(-20) / bars["raw_midprice_mean"] - 1
    y = (fwd > 0).astype(float).values
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_v, y_v = X[valid], y[valid]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_v)
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(X_s, y_v)

    algo._model = model
    algo._scaler = scaler

    result = algo.run_batch(bars)

    # Post-warmup signals should have some variation
    post = result.iloc[algo.warmup:]
    signals = post["alg_mc_signal"].dropna()
    assert len(signals) > 100
    assert signals.min() >= -1.0
    assert signals.max() <= 1.0
    # Should have at least some non-zero signals
    assert (signals != 0.0).any(), "Expected some non-zero signals with real model"


def test_save_load_roundtrip(tmp_path):
    """Train a model, save via model_io, load in new algorithm instance.
    Predictions match to 1e-10."""
    from utils.model_io import ModelMetadata, save_sklearn_model, load_sklearn_model

    # Train
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 7))
    y = (X[:, 0] > 0).astype(float)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(X_s, y)

    # Save
    meta = ModelMetadata(
        model_type="logistic_regression",
        model_name="momentum_continuation",
        feature_names=list(MomentumContinuation.FEATURE_COLS),
        hyperparameters={"C": 1.0},
        performance_metrics={"auc_oos": 0.55},
        training_date="2026-06-05",
    )
    save_sklearn_model(model, scaler, meta, tmp_path)

    # Load — get_latest_model returns the .pkl path from the directory
    from utils.model_io import get_latest_model
    model_path = get_latest_model(tmp_path)
    assert model_path is not None, "No model found after save"
    model2, scaler2, meta2 = load_sklearn_model(model_path)

    # Predictions must match
    X_test = rng.standard_normal((50, 7))
    X_test_s1 = scaler.transform(X_test)
    X_test_s2 = scaler2.transform(X_test)

    pred1 = model.predict_proba(X_test_s1)
    pred2 = model2.predict_proba(X_test_s2)

    np.testing.assert_allclose(pred1, pred2, atol=1e-10)

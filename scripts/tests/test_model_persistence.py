"""Model persistence roundtrip tests for sklearn and LightGBM."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model_io import (
    ModelMetadata,
    get_latest_model,
    load_sklearn_model,
    save_sklearn_model,
)


def _make_metadata(**overrides):
    defaults = dict(
        model_type="sklearn",
        model_name="test_model",
        feature_names=["f1", "f2", "f3"],
        hyperparameters={"C": 1.0, "penalty": "l2"},
        performance_metrics={"auc": 0.75, "accuracy": 0.68},
        training_date="2026-06-01T12:00:00",
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def test_sklearn_save_load_roundtrip(tmp_path):
    """Train LogReg, save, load. Predictions match to 1e-10."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    model = LogisticRegression(C=1.0).fit(X_s, y)

    meta = _make_metadata(model_type="sklearn", model_name="logreg_test")
    path = save_sklearn_model(model, scaler, meta, tmp_path, "test.pkl")

    model2, scaler2, meta2 = load_sklearn_model(path)

    pred_orig = model.predict_proba(X_s)
    pred_loaded = model2.predict_proba(scaler2.transform(X))

    np.testing.assert_allclose(pred_orig, pred_loaded, atol=1e-10)
    assert meta2.model_name == "logreg_test"


def test_lightgbm_save_load_roundtrip(tmp_path):
    """Train LightGBM, save, load. Predictions match exactly."""
    lgb = pytest.importorskip("lightgbm")

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "num_leaves": 8, "verbose": -1}
    model = lgb.train(params, ds, num_boost_round=10)

    from utils.model_io import load_lightgbm_model, save_lightgbm_model
    meta = _make_metadata(model_type="lightgbm", model_name="lgbm_test")
    path = save_lightgbm_model(model, meta, tmp_path, "test.txt")

    model2, meta2 = load_lightgbm_model(path)

    pred_orig = model.predict(X)
    pred_loaded = model2.predict(X)

    np.testing.assert_array_equal(pred_orig, pred_loaded)
    assert meta2.model_name == "lgbm_test"


def test_metadata_json_complete(tmp_path):
    """Saved metadata JSON has all required fields."""
    from sklearn.linear_model import LogisticRegression

    X = np.random.default_rng(0).normal(0, 1, (50, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    meta = _make_metadata()
    save_sklearn_model(model, None, meta, tmp_path, "meta_test.pkl")

    meta_path = tmp_path / "meta_test_metadata.json"
    assert meta_path.exists()

    with open(meta_path) as f:
        data = json.load(f)

    required_fields = {
        "model_type", "model_name", "feature_names",
        "hyperparameters", "performance_metrics", "training_date",
    }
    assert required_fields.issubset(set(data.keys()))
    assert data["n_features"] == 3


def test_get_latest_model_selects_newest(tmp_path):
    """Save model_v1, then model_v2. get_latest_model() returns v2."""
    from sklearn.linear_model import LogisticRegression

    X = np.random.default_rng(0).normal(0, 1, (50, 3))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    meta_v1 = _make_metadata(training_date="2026-01-01T00:00:00")
    save_sklearn_model(model, None, meta_v1, tmp_path, "v1.pkl")

    meta_v2 = _make_metadata(training_date="2026-06-01T00:00:00")
    save_sklearn_model(model, None, meta_v2, tmp_path, "v2.pkl")

    latest = get_latest_model(tmp_path)
    assert latest is not None
    assert latest.name == "v2.pkl"


def test_get_latest_model_empty_dir(tmp_path):
    """Empty directory -> returns None, no crash."""
    result = get_latest_model(tmp_path)
    assert result is None

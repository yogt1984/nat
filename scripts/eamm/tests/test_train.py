"""Unit tests for EAMM Model Training Pipeline."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.train import train_eamm, predict_spread, TrainResult


def _make_features(n=500, d=19):
    np.random.seed(42)
    return np.random.randn(n, d)


def _make_regression_target(X):
    # Target = linear combination + noise
    w = np.random.randn(X.shape[1])
    y = X @ w + np.random.randn(X.shape[0]) * 0.5
    return np.abs(y)  # spreads are positive


def _make_classification_target(X, n_classes=4):
    w = np.random.randn(X.shape[1])
    scores = X @ w
    quantiles = np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1])
    return np.digitize(scores, quantiles).astype(int)


FEATURE_NAMES = [f"feat_{i}" for i in range(19)]


class TestRegressionMode:
    def test_trains_and_returns_result(self):
        X = _make_features()
        y = _make_regression_target(X)
        result = train_eamm(X, y, FEATURE_NAMES, mode="regression", save_dir=None)
        assert isinstance(result, TrainResult)
        assert result.mode == "regression"
        assert result.n_train == 500
        assert result.train_score > 0.0  # R^2 should be positive for learnable target

    def test_predictions_positive(self):
        X = _make_features()
        y = _make_regression_target(X)
        result = train_eamm(X, y, FEATURE_NAMES, mode="regression", save_dir=None)
        preds = predict_spread(result, X[:50])
        assert np.all(preds >= 0.0)

    def test_feature_importances(self):
        X = _make_features()
        y = _make_regression_target(X)
        result = train_eamm(X, y, FEATURE_NAMES, mode="regression", save_dir=None)
        assert len(result.feature_importances) == 19
        assert sum(result.feature_importances.values()) > 0


class TestClassificationMode:
    def test_trains_and_returns_result(self):
        X = _make_features()
        y = _make_classification_target(X, n_classes=4)
        result = train_eamm(X, y, FEATURE_NAMES, mode="classification", save_dir=None)
        assert result.mode == "classification"
        assert result.n_classes == 4
        assert result.train_score > 0.25  # better than random

    def test_predict_returns_proba(self):
        X = _make_features()
        y = _make_classification_target(X, n_classes=4)
        result = train_eamm(X, y, FEATURE_NAMES, mode="classification", save_dir=None)
        proba = predict_spread(result, X[:50])
        assert proba.shape == (50, 4)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestNanHandling:
    def test_nan_in_features(self):
        X = _make_features()
        y = _make_regression_target(X)  # compute y before corrupting X
        X[0, 0] = np.nan
        X[10, 5] = np.inf
        # Should not crash
        result = train_eamm(X, y, FEATURE_NAMES, mode="regression", save_dir=None)
        preds = predict_spread(result, X[:10])
        assert not np.any(np.isnan(preds))


class TestModelSave:
    def test_save_creates_files(self, tmp_path):
        X = _make_features(n=100)
        y = _make_regression_target(X)
        result = train_eamm(X, y, FEATURE_NAMES, mode="regression",
                            save_dir=str(tmp_path))
        assert result.model_path is not None
        assert Path(result.model_path).exists()

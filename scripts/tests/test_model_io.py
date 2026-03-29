"""
Tests for model I/O utilities.

Verifies model saving and loading works correctly.
"""

import pytest
import sys
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_io import (
    ModelMetadata,
    save_sklearn_model,
    load_sklearn_model,
    list_models,
    get_latest_model,
)


def test_model_metadata_creation():
    """Should be able to create model metadata."""
    metadata = ModelMetadata(
        model_type="test",
        model_name="test_model",
        feature_names=["f1", "f2", "f3"],
        hyperparameters={"alpha": 0.1},
        performance_metrics={"test_r2": 0.5},
        training_date=datetime.now().isoformat(),
    )

    assert metadata.model_type == "test"
    assert metadata.model_name == "test_model"
    assert len(metadata.feature_names) == 3


def test_model_metadata_to_dict():
    """Should convert metadata to dict."""
    metadata = ModelMetadata(
        model_type="test",
        model_name="test_model",
        feature_names=["f1", "f2"],
        hyperparameters={},
        performance_metrics={},
        training_date=datetime.now().isoformat(),
    )

    d = metadata.to_dict()
    assert isinstance(d, dict)
    assert d["model_type"] == "test"
    assert d["n_features"] == 2


def test_save_and_load_sklearn_model():
    """Should save and load sklearn model with scaler."""
    # Create simple model
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    metadata = ModelMetadata(
        model_type="linear",
        model_name="test_linear",
        feature_names=["f1", "f2", "f3"],
        hyperparameters={},
        performance_metrics={"test_r2": 0.8},
        training_date=datetime.now().isoformat(),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        model_path = save_sklearn_model(model, scaler, metadata, Path(tmpdir))

        assert model_path.exists()
        assert (Path(tmpdir) / model_path.name.replace(".pkl", "_metadata.json")).exists()

        # Load
        loaded_model, loaded_scaler, loaded_metadata = load_sklearn_model(model_path)

        assert loaded_model is not None
        assert loaded_scaler is not None
        assert loaded_metadata.model_name == "test_linear"
        assert len(loaded_metadata.feature_names) == 3

        # Test predictions match
        X_test = np.random.randn(10, 3)
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


def test_list_models():
    """Should list all models in directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy model
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        model = LinearRegression()
        model.fit(X, y)

        metadata = ModelMetadata(
            model_type="test",
            model_name="test_model",
            feature_names=["f1", "f2"],
            hyperparameters={},
            performance_metrics={"test_r2": 0.6},
            training_date=datetime.now().isoformat(),
        )

        save_sklearn_model(model, None, metadata, Path(tmpdir))

        # List models
        models = list_models(Path(tmpdir))

        assert len(models) == 1
        assert models[0]["model_name"] == "test_model"
        assert models[0]["model_type"] == "test"
        assert models[0]["test_r2"] == 0.6


def test_get_latest_model():
    """Should get most recently trained model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two models with different timestamps
        for i in range(2):
            X = np.random.randn(50, 2)
            y = np.random.randn(50)

            model = LinearRegression()
            model.fit(X, y)

            metadata = ModelMetadata(
                model_type="test",
                model_name=f"model_{i}",
                feature_names=["f1", "f2"],
                hyperparameters={},
                performance_metrics={},
                training_date=datetime.now().isoformat(),
            )

            save_sklearn_model(model, None, metadata, Path(tmpdir))

        latest = get_latest_model(Path(tmpdir))
        assert latest is not None
        assert latest.exists()


def test_list_models_empty_directory():
    """Should handle empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models = list_models(Path(tmpdir))
        assert len(models) == 0


def test_get_latest_model_empty():
    """Should return None for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = get_latest_model(Path(tmpdir))
        assert latest is None

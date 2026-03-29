"""
Tests for model scoring script.

Verifies scoring functionality works correctly.
"""

import pytest
import sys
import tempfile
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_io import ModelMetadata, save_sklearn_model


def test_script_exists():
    """Script file should exist."""
    script_path = Path("scripts/score_data.py")
    assert script_path.exists(), "score_data.py should exist"


def test_can_import_scoring_functions():
    """Should be able to import scoring functions."""
    # Import score_data module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_data",
        Path("scripts/score_data.py")
    )
    score_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_data)

    # Check key functions exist
    assert hasattr(score_data, 'load_parquet_data')
    assert hasattr(score_data, 'extract_features')
    assert hasattr(score_data, 'evaluate_predictions')


def test_extract_features_with_mock_data():
    """Should extract features from DataFrame."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_data",
        Path("scripts/score_data.py")
    )
    score_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_data)

    # Create mock DataFrame
    df = pl.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    })

    # Extract features
    X, valid_mask = score_data.extract_features(df, ["feature1", "feature2", "feature3"])

    assert X.shape == (100, 3)
    assert valid_mask.sum() == 100  # All valid


def test_extract_features_with_nan():
    """Should handle NaN values in features."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_data",
        Path("scripts/score_data.py")
    )
    score_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_data)

    # Create DataFrame with some NaN values
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
    }
    data["feature1"][10] = np.nan  # Add NaN

    df = pl.DataFrame(data)

    # Extract features
    X, valid_mask = score_data.extract_features(df, ["feature1", "feature2"])

    assert X.shape[0] == 99  # One row filtered
    assert valid_mask.sum() == 99


def test_evaluate_predictions():
    """Should compute evaluation metrics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_data",
        Path("scripts/score_data.py")
    )
    score_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_data)

    # Create mock predictions and true values
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1  # Add small noise

    metrics = score_data.evaluate_predictions(y_true, y_pred)

    assert "r2_score" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "correlation" in metrics
    assert "n_samples" in metrics

    # R² should be high since predictions are close to true values
    assert metrics["r2_score"] > 0.8


def test_end_to_end_scoring():
    """Test end-to-end scoring with saved model and mock data."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_data",
        Path("scripts/score_data.py")
    )
    score_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save a simple model
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)

        model = LinearRegression()
        model.fit(X_train, y_train)

        scaler = StandardScaler()
        scaler.fit(X_train)

        metadata = ModelMetadata(
            model_type="linear",
            model_name="test_model",
            feature_names=["feature1", "feature2", "feature3"],
            hyperparameters={},
            performance_metrics={"test_r2": 0.8},
            training_date=datetime.now().isoformat(),
        )

        model_path = save_sklearn_model(model, scaler, metadata, Path(tmpdir))

        # Create mock data to score
        df = pl.DataFrame({
            "timestamp": [datetime.now()] * 50,
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50),
            "midprice": np.cumsum(np.random.randn(50)) + 50000,
        })

        # Score the data
        predictions, metadata_dict, eval_metrics = score_data.score_sklearn_model(
            model_path, df, evaluate=False
        )

        assert len(predictions) == 50
        assert isinstance(predictions, np.ndarray)
        assert metadata_dict["model_name"] == "test_model"

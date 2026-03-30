"""
Tests for ML strategy integration.

Verifies ML prediction loading, strategy creation, and backtest integration.
"""

import pytest
import sys
import tempfile
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.ml_strategy import (
    load_predictions,
    join_predictions_with_features,
    create_ml_strategy,
    create_ml_quantile_strategy,
)


def test_ml_strategy_module_exists():
    """ML strategy module should exist."""
    module_path = Path("scripts/backtest/ml_strategy.py")
    assert module_path.exists(), "ml_strategy.py should exist"


def test_can_import_ml_functions():
    """Should be able to import ML strategy functions."""
    from backtest.ml_strategy import (
        load_predictions,
        create_ml_strategy,
        MLPredictions,
    )
    assert callable(load_predictions)
    assert callable(create_ml_strategy)


def test_load_predictions_from_parquet():
    """Should load predictions from Parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.random.randn(100) * 0.001  # Small returns

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Load predictions
        preds = load_predictions(pred_path)

        assert preds.model_name == "test_model"
        assert preds.n_predictions == 100
        assert "mean" in preds.prediction_stats
        assert "std" in preds.prediction_stats


def test_load_predictions_filters_nan():
    """Should filter out NaN predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create predictions with some NaN
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.random.randn(100) * 0.001
        predictions[10:15] = np.nan  # Add 5 NaN values

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Load predictions
        preds = load_predictions(pred_path)

        # Should filter NaN
        assert preds.n_predictions == 95


def test_join_predictions_with_features():
    """Should join predictions with feature DataFrame."""
    # Create features DataFrame
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
    features_df = pl.DataFrame({
        "timestamp": timestamps,
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "midprice": np.cumsum(np.random.randn(100)) + 50000,
    })

    # Create predictions (matching timestamps)
    pred_df = pl.DataFrame({
        "timestamp": timestamps,
        "prediction": np.random.randn(100) * 0.001,
        "model_name": ["test_model"] * 100,
    })

    # Create MLPredictions object
    from backtest.ml_strategy import MLPredictions
    preds = MLPredictions(
        df=pred_df,
        model_name="test_model",
        n_predictions=100,
        prediction_stats={
            "mean": 0.0,
            "std": 0.001,
            "min": -0.003,
            "max": 0.003,
            "q25": -0.001,
            "q50": 0.0,
            "q75": 0.001,
        }
    )

    # Join
    joined = join_predictions_with_features(features_df, preds)

    assert "prediction" in joined.columns
    assert len(joined) == 100
    # All should match
    n_matched = joined.filter(pl.col("prediction").is_not_nan()).height
    assert n_matched == 100


def test_create_ml_long_strategy():
    """Should create ML long strategy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.random.randn(100) * 0.002

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Create strategy
        strategy, preds = create_ml_strategy(
            predictions_path=pred_path,
            entry_threshold=0.001,
            exit_threshold=0.0,
            direction="long",
        )

        assert strategy.name.startswith("ml_long")
        assert strategy.direction == "long"
        assert "prediction" in strategy.required_features
        assert callable(strategy.entry_condition)
        assert callable(strategy.exit_condition)


def test_create_ml_short_strategy():
    """Should create ML short strategy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.random.randn(100) * 0.002

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Create strategy
        strategy, preds = create_ml_strategy(
            predictions_path=pred_path,
            entry_threshold=0.001,
            exit_threshold=0.0,
            direction="short",
        )

        assert strategy.name.startswith("ml_short")
        assert strategy.direction == "short"


def test_ml_strategy_entry_condition():
    """ML strategy entry condition should work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(10)]
        # Create predictions: some above threshold, some below
        predictions = np.array([0.003, 0.002, 0.001, 0.0005, 0.0, -0.001, -0.002, np.nan, 0.0015, 0.0025])

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 10,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Create long strategy with threshold 0.001
        strategy, preds = create_ml_strategy(
            predictions_path=pred_path,
            entry_threshold=0.001,
            exit_threshold=0.0,
            direction="long",
        )

        # Test entry condition on mock data
        test_df = pl.DataFrame({
            "prediction": predictions,
        })

        entry_signals = strategy.entry_condition(test_df)

        # Should be True for predictions > 0.001 (strictly greater than): indices 0, 1, 8, 9
        # Index 2 (0.001) is NOT > 0.001, so it's False
        # Index 7 (NaN) should be False after fill_null
        expected = np.array([True, True, False, False, False, False, False, False, True, True])
        assert np.array_equal(entry_signals.to_numpy(), expected)


def test_ml_quantile_strategy():
    """Should create ML strategy using quantile thresholds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions with known distribution
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.linspace(-0.005, 0.005, 100)  # Uniform distribution

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Create quantile strategy (75th percentile entry)
        strategy, preds = create_ml_quantile_strategy(
            predictions_path=pred_path,
            entry_quantile=0.75,
            exit_quantile=0.50,
            direction="long",
        )

        assert strategy.name.startswith("ml_long")
        assert "q75" in strategy.name
        assert "quantile" in strategy.description.lower()


def test_ml_strategy_with_confidence_threshold():
    """Should create ML strategy with confidence filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock predictions
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        predictions = np.random.randn(100) * 0.002

        df = pl.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": ["test_model"] * 100,
        })

        pred_path = Path(tmpdir) / "predictions.parquet"
        df.write_parquet(pred_path)

        # Create strategy with confidence threshold
        strategy, preds = create_ml_strategy(
            predictions_path=pred_path,
            entry_threshold=0.001,
            exit_threshold=0.0,
            confidence_threshold=0.002,  # Require |prediction| > 0.002
            direction="long",
        )

        assert "Confidence" in strategy.description

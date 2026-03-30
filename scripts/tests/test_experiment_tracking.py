"""
Tests for experiment tracking system.

Verifies experiment registration and linking of artifacts.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_tracking import ExperimentTracker


def test_tracking_module_exists():
    """Tracking module should exist."""
    module_path = Path("scripts/experiment_tracking.py")
    assert module_path.exists(), "experiment_tracking.py should exist"


def test_can_import_tracker():
    """Should be able to import ExperimentTracker."""
    from experiment_tracking import ExperimentTracker
    assert ExperimentTracker is not None


def test_create_tracker():
    """Should create tracker with temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))
        assert tracker.tracking_dir.exists()
        # experiments_file is created on first save, not on init
        assert tracker.experiments == []


def test_register_training():
    """Should register training run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Create mock model metadata
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir()

        model_path = model_dir / "test_model.pkl"
        model_path.touch()

        metadata = {
            "model_type": "elasticnet",
            "model_name": "test_model",
            "feature_names": ["feat1", "feat2"],
            "hyperparameters": {"alpha": 1.0},
            "performance_metrics": {"test_r2": 0.75},
            "training_date": datetime.now().isoformat(),
        }

        metadata_path = model_dir / "test_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Register training
        exp_id = tracker.register_training(
            snapshot_name="test_snapshot",
            model_path=model_path,
            model_metadata=metadata,
        )

        assert exp_id.startswith("exp_")
        assert len(tracker.experiments) == 1

        exp = tracker.get_experiment(exp_id)
        assert exp["stage"] == "training"
        assert exp["snapshot"]["name"] == "test_snapshot"
        assert exp["training"]["model_name"] == "test_model"


def test_register_predictions():
    """Should register predictions linked to model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Create mock model
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir()
        model_path = model_dir / "test_model.pkl"
        model_path.touch()

        metadata = {
            "model_type": "elasticnet",
            "model_name": "test_model",
            "feature_names": ["feat1"],
            "hyperparameters": {},
            "performance_metrics": {"test_r2": 0.75},
            "training_date": datetime.now().isoformat(),
        }

        metadata_path = model_dir / "test_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Register training
        exp_id = tracker.register_training(
            snapshot_name="test_snapshot",
            model_path=model_path,
            model_metadata=metadata,
        )

        # Create mock predictions
        pred_path = Path(tmpdir) / "predictions.parquet"
        pred_path.touch()

        # Register predictions
        exp_id_2 = tracker.register_predictions(
            model_path=model_path,
            predictions_path=pred_path,
            n_predictions=1000,
            prediction_stats={"mean": 0.001, "std": 0.002},
        )

        # Should update same experiment
        assert exp_id == exp_id_2

        exp = tracker.get_experiment(exp_id)
        assert exp["stage"] == "predictions"
        assert exp["predictions"]["n_predictions"] == 1000


def test_register_backtest():
    """Should register backtest results linked to predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Create full experiment chain
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir()
        model_path = model_dir / "test_model.pkl"
        model_path.touch()

        metadata = {
            "model_type": "elasticnet",
            "model_name": "test_model",
            "feature_names": ["feat1"],
            "hyperparameters": {},
            "performance_metrics": {"test_r2": 0.75},
            "training_date": datetime.now().isoformat(),
        }

        metadata_path = model_dir / "test_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Register training
        exp_id = tracker.register_training(
            snapshot_name="test_snapshot",
            model_path=model_path,
            model_metadata=metadata,
        )

        # Register predictions
        pred_path = Path(tmpdir) / "predictions.parquet"
        pred_path.touch()

        tracker.register_predictions(
            model_path=model_path,
            predictions_path=pred_path,
            n_predictions=1000,
        )

        # Register backtest
        backtest_results = {
            "sharpe_ratio": 1.5,
            "total_return_pct": 15.3,
            "total_trades": 42,
        }

        tracker.register_backtest(
            predictions_path=pred_path,
            backtest_results=backtest_results,
        )

        exp = tracker.get_experiment(exp_id)
        assert exp["stage"] == "backtest"
        assert exp["backtest"]["results"]["sharpe_ratio"] == 1.5


def test_list_experiments():
    """Should list all experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Register multiple experiments
        for i in range(3):
            model_dir = Path(tmpdir) / f"models_{i}"
            model_dir.mkdir()
            model_path = model_dir / f"model_{i}.pkl"
            model_path.touch()

            metadata = {
                "model_type": "elasticnet",
                "model_name": f"model_{i}",
                "feature_names": ["feat1"],
                "hyperparameters": {},
                "performance_metrics": {"test_r2": 0.7 + i * 0.1},
                "training_date": datetime.now().isoformat(),
            }

            metadata_path = model_dir / f"model_{i}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            tracker.register_training(
                snapshot_name=f"snapshot_{i}",
                model_path=model_path,
                model_metadata=metadata,
            )

        experiments = tracker.list_experiments()
        assert len(experiments) == 3


def test_filter_experiments_by_stage():
    """Should filter experiments by stage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Create experiments at different stages
        for i in range(2):
            model_dir = Path(tmpdir) / f"models_{i}"
            model_dir.mkdir()
            model_path = model_dir / f"model_{i}.pkl"
            model_path.touch()

            metadata = {
                "model_type": "elasticnet",
                "model_name": f"model_{i}",
                "feature_names": ["feat1"],
                "hyperparameters": {},
                "performance_metrics": {"test_r2": 0.75},
                "training_date": datetime.now().isoformat(),
            }

            metadata_path = model_dir / f"model_{i}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            tracker.register_training(
                snapshot_name=f"snapshot_{i}",
                model_path=model_path,
                model_metadata=metadata,
            )

            # Only add predictions to first experiment
            if i == 0:
                pred_path = Path(tmpdir) / "predictions.parquet"
                pred_path.touch()
                tracker.register_predictions(
                    model_path=model_path,
                    predictions_path=pred_path,
                    n_predictions=1000,
                )

        # Filter by stage
        training_exps = tracker.list_experiments(stage="training")
        predictions_exps = tracker.list_experiments(stage="predictions")

        assert len(training_exps) == 1
        assert len(predictions_exps) == 1


def test_get_best_experiment():
    """Should find best experiment by metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tracking_dir=Path(tmpdir))

        # Create experiments with backtests
        sharpe_ratios = [1.2, 1.8, 1.5]

        for i, sharpe in enumerate(sharpe_ratios):
            model_dir = Path(tmpdir) / f"models_{i}"
            model_dir.mkdir()
            model_path = model_dir / f"model_{i}.pkl"
            model_path.touch()

            metadata = {
                "model_type": "elasticnet",
                "model_name": f"model_{i}",
                "feature_names": ["feat1"],
                "hyperparameters": {},
                "performance_metrics": {"test_r2": 0.75},
                "training_date": datetime.now().isoformat(),
            }

            metadata_path = model_dir / f"model_{i}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            tracker.register_training(
                snapshot_name=f"snapshot_{i}",
                model_path=model_path,
                model_metadata=metadata,
            )

            pred_path = Path(tmpdir) / f"predictions_{i}.parquet"
            pred_path.touch()
            tracker.register_predictions(
                model_path=model_path,
                predictions_path=pred_path,
                n_predictions=1000,
            )

            tracker.register_backtest(
                predictions_path=pred_path,
                backtest_results={
                    "sharpe_ratio": sharpe,
                    "total_trades": 50,
                },
            )

        best = tracker.get_best_experiment(metric="sharpe_ratio", min_trades=30)
        assert best is not None
        assert best["backtest"]["results"]["sharpe_ratio"] == 1.8

#!/usr/bin/env python3
"""
Experiment Tracking System

Automatically tracks and links:
- Training runs → Models
- Models → Predictions
- Predictions → Backtest results

Provides full audit trail and reproducibility.

Usage:
    # Register training run
    python scripts/experiment_tracking.py register-training \\
        --snapshot baseline_30d \\
        --model-path models/lightgbm_baseline_baseline_30d_20260330_120000.txt

    # Register predictions
    python scripts/experiment_tracking.py register-predictions \\
        --model-path models/lightgbm_baseline_baseline_30d_20260330_120000.txt \\
        --predictions-path predictions.parquet

    # Register backtest
    python scripts/experiment_tracking.py register-backtest \\
        --predictions-path predictions.parquet \\
        --backtest-results backtest_results.json

    # List all experiments
    python scripts/experiment_tracking.py list

    # Compare experiments
    python scripts/experiment_tracking.py compare exp1_id exp2_id

    # Get experiment details
    python scripts/experiment_tracking.py get exp_id
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import ModelMetadata


class ExperimentTracker:
    """Tracks experiments and links artifacts."""

    def __init__(self, tracking_dir: Path = Path("./experiments/tracking")):
        self.tracking_dir = tracking_dir
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_file = tracking_dir / "experiments.json"
        self._load_experiments()

    def _load_experiments(self):
        """Load experiments from disk."""
        if self.experiments_file.exists():
            with open(self.experiments_file) as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []

    def _save_experiments(self):
        """Save experiments to disk."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def register_training(
        self,
        snapshot_name: str,
        model_path: Path,
        model_metadata: Optional[Dict] = None,
        training_params: Optional[Dict] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Register a training run.

        Args:
            snapshot_name: Dataset snapshot name
            model_path: Path to saved model
            model_metadata: Model metadata dict
            training_params: Training parameters
            notes: Optional notes

        Returns:
            Experiment ID
        """
        # Load model metadata if not provided
        if model_metadata is None:
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    model_metadata = json.load(f)
            else:
                raise ValueError(f"Model metadata not found: {metadata_path}")

        # Create experiment ID
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_metadata.get('model_type', 'unknown')}"

        # Create experiment record
        experiment = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "stage": "training",
            "snapshot": {
                "name": snapshot_name,
            },
            "training": {
                "model_path": str(model_path),
                "model_type": model_metadata.get("model_type"),
                "model_name": model_metadata.get("model_name"),
                "feature_names": model_metadata.get("feature_names", []),
                "hyperparameters": model_metadata.get("hyperparameters", {}),
                "performance_metrics": model_metadata.get("performance_metrics", {}),
                "training_date": model_metadata.get("training_date"),
                "training_params": training_params or {},
            },
            "predictions": None,
            "backtest": None,
            "notes": notes or "",
            "tags": [],
        }

        # Add to experiments
        self.experiments.append(experiment)
        self._save_experiments()

        print(f"✅ Registered training run: {experiment_id}")
        print(f"   Model: {model_metadata.get('model_name')}")
        print(f"   Snapshot: {snapshot_name}")
        print(f"   Test R²: {model_metadata.get('performance_metrics', {}).get('test_r2', 'N/A')}")

        return experiment_id

    def register_predictions(
        self,
        model_path: Path,
        predictions_path: Path,
        n_predictions: Optional[int] = None,
        prediction_stats: Optional[Dict] = None,
    ) -> str:
        """
        Register predictions generated from a model.

        Args:
            model_path: Path to model that generated predictions
            predictions_path: Path to predictions Parquet file
            n_predictions: Number of predictions
            prediction_stats: Prediction statistics

        Returns:
            Experiment ID
        """
        # Find experiment by model path
        experiment = self._find_experiment_by_model(model_path)
        if experiment is None:
            raise ValueError(
                f"No experiment found for model: {model_path}\n"
                "Register training first with: register-training"
            )

        # Compute predictions hash
        pred_hash = self._compute_file_hash(predictions_path)

        # Update experiment
        experiment["stage"] = "predictions"
        experiment["predictions"] = {
            "predictions_path": str(predictions_path),
            "predictions_hash": pred_hash,
            "n_predictions": n_predictions,
            "prediction_stats": prediction_stats or {},
            "generated_at": datetime.now().isoformat(),
        }

        self._save_experiments()

        print(f"✅ Registered predictions: {experiment['experiment_id']}")
        print(f"   Predictions: {predictions_path}")
        if n_predictions:
            print(f"   Count: {n_predictions}")

        return experiment["experiment_id"]

    def register_backtest(
        self,
        predictions_path: Path,
        backtest_results: Dict,
        strategy_params: Optional[Dict] = None,
    ) -> str:
        """
        Register backtest results.

        Args:
            predictions_path: Path to predictions used
            backtest_results: Backtest results dict
            strategy_params: Strategy parameters

        Returns:
            Experiment ID
        """
        # Find experiment by predictions path
        experiment = self._find_experiment_by_predictions(predictions_path)
        if experiment is None:
            raise ValueError(
                f"No experiment found for predictions: {predictions_path}\n"
                "Register predictions first with: register-predictions"
            )

        # Update experiment
        experiment["stage"] = "backtest"
        experiment["backtest"] = {
            "strategy_params": strategy_params or {},
            "results": backtest_results,
            "backtested_at": datetime.now().isoformat(),
        }

        self._save_experiments()

        print(f"✅ Registered backtest: {experiment['experiment_id']}")
        print(f"   Sharpe: {backtest_results.get('sharpe_ratio', 'N/A')}")
        print(f"   Return: {backtest_results.get('total_return_pct', 'N/A')}%")

        return experiment["experiment_id"]

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        for exp in self.experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        return None

    def list_experiments(
        self,
        stage: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_r2: Optional[float] = None,
    ) -> List[Dict]:
        """
        List experiments with optional filtering.

        Args:
            stage: Filter by stage (training, predictions, backtest)
            min_sharpe: Minimum Sharpe ratio
            min_r2: Minimum R² score

        Returns:
            List of experiments
        """
        filtered = self.experiments

        if stage:
            filtered = [e for e in filtered if e["stage"] == stage]

        if min_sharpe is not None:
            filtered = [
                e for e in filtered
                if e.get("backtest") and e["backtest"]["results"].get("sharpe_ratio", 0) >= min_sharpe
            ]

        if min_r2 is not None:
            filtered = [
                e for e in filtered
                if e.get("training") and e["training"]["performance_metrics"].get("test_r2", 0) >= min_r2
            ]

        return sorted(filtered, key=lambda x: x["created_at"], reverse=True)

    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs

        Returns:
            Comparison dict
        """
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [e for e in experiments if e is not None]

        if len(experiments) < 2:
            raise ValueError("Need at least 2 experiments to compare")

        comparison = {
            "experiments": [],
            "comparison_date": datetime.now().isoformat(),
        }

        for exp in experiments:
            summary = {
                "experiment_id": exp["experiment_id"],
                "created_at": exp["created_at"],
                "stage": exp["stage"],
                "model_type": exp["training"]["model_type"],
                "snapshot": exp["snapshot"]["name"],
            }

            # Training metrics
            if exp.get("training"):
                perf = exp["training"]["performance_metrics"]
                summary["training"] = {
                    "test_r2": perf.get("test_r2"),
                    "test_rmse": perf.get("test_rmse"),
                    "n_features": len(exp["training"]["feature_names"]),
                }

            # Backtest metrics
            if exp.get("backtest"):
                results = exp["backtest"]["results"]
                summary["backtest"] = {
                    "sharpe_ratio": results.get("sharpe_ratio"),
                    "total_return_pct": results.get("total_return_pct"),
                    "max_drawdown_pct": results.get("max_drawdown_pct"),
                    "win_rate": results.get("win_rate"),
                    "total_trades": results.get("total_trades"),
                }

            comparison["experiments"].append(summary)

        return comparison

    def get_best_experiment(
        self,
        metric: str = "sharpe_ratio",
        min_trades: int = 30,
    ) -> Optional[Dict]:
        """
        Get best experiment by metric.

        Args:
            metric: Metric to optimize (sharpe_ratio, total_return_pct, win_rate)
            min_trades: Minimum number of trades for statistical significance

        Returns:
            Best experiment or None
        """
        # Filter to experiments with backtest results
        candidates = [
            e for e in self.experiments
            if e.get("backtest")
            and e["backtest"]["results"].get("total_trades", 0) >= min_trades
        ]

        if not candidates:
            return None

        # Sort by metric
        if metric == "sharpe_ratio":
            best = max(candidates, key=lambda x: x["backtest"]["results"].get("sharpe_ratio", -999))
        elif metric == "total_return_pct":
            best = max(candidates, key=lambda x: x["backtest"]["results"].get("total_return_pct", -999))
        elif metric == "win_rate":
            best = max(candidates, key=lambda x: x["backtest"]["results"].get("win_rate", 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best

    def add_tags(self, experiment_id: str, tags: List[str]):
        """Add tags to an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment["tags"].extend(tags)
            self._save_experiments()

    def _find_experiment_by_model(self, model_path: Path) -> Optional[Dict]:
        """Find experiment by model path."""
        model_path_str = str(model_path)
        for exp in reversed(self.experiments):
            if exp.get("training") and exp["training"]["model_path"] == model_path_str:
                return exp
        return None

    def _find_experiment_by_predictions(self, predictions_path: Path) -> Optional[Dict]:
        """Find experiment by predictions path."""
        pred_path_str = str(predictions_path)
        for exp in reversed(self.experiments):
            if exp.get("predictions") and exp["predictions"]["predictions_path"] == pred_path_str:
                return exp
        return None

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment tracking system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register training
    train_parser = subparsers.add_parser("register-training", help="Register training run")
    train_parser.add_argument("--snapshot", type=str, required=True, help="Snapshot name")
    train_parser.add_argument("--model-path", type=Path, required=True, help="Path to saved model")
    train_parser.add_argument("--notes", type=str, help="Optional notes")

    # Register predictions
    pred_parser = subparsers.add_parser("register-predictions", help="Register predictions")
    pred_parser.add_argument("--model-path", type=Path, required=True, help="Model path")
    pred_parser.add_argument("--predictions-path", type=Path, required=True, help="Predictions file")

    # Register backtest
    backtest_parser = subparsers.add_parser("register-backtest", help="Register backtest results")
    backtest_parser.add_argument("--predictions-path", type=Path, required=True, help="Predictions file")
    backtest_parser.add_argument("--backtest-json", type=Path, required=True, help="Backtest results JSON")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--stage", type=str, choices=["training", "predictions", "backtest"])
    list_parser.add_argument("--min-sharpe", type=float, help="Minimum Sharpe ratio")
    list_parser.add_argument("--min-r2", type=float, help="Minimum R² score")

    # Get experiment
    get_parser = subparsers.add_parser("get", help="Get experiment details")
    get_parser.add_argument("experiment_id", type=str, help="Experiment ID")

    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiment_ids", type=str, nargs="+", help="Experiment IDs")

    # Best experiment
    best_parser = subparsers.add_parser("best", help="Get best experiment")
    best_parser.add_argument("--metric", type=str, default="sharpe_ratio",
                             choices=["sharpe_ratio", "total_return_pct", "win_rate"])
    best_parser.add_argument("--min-trades", type=int, default=30)

    args = parser.parse_args()

    tracker = ExperimentTracker()

    if args.command == "register-training":
        tracker.register_training(
            snapshot_name=args.snapshot,
            model_path=args.model_path,
            notes=args.notes,
        )

    elif args.command == "register-predictions":
        # Load prediction stats if available
        try:
            import polars as pl
            df = pl.read_parquet(args.predictions_path)
            n_predictions = len(df)
            valid = df.filter(pl.col("prediction").is_not_nan())
            pred_stats = {
                "mean": float(valid["prediction"].mean()),
                "std": float(valid["prediction"].std()),
                "min": float(valid["prediction"].min()),
                "max": float(valid["prediction"].max()),
            }
        except Exception:
            n_predictions = None
            pred_stats = None

        tracker.register_predictions(
            model_path=args.model_path,
            predictions_path=args.predictions_path,
            n_predictions=n_predictions,
            prediction_stats=pred_stats,
        )

    elif args.command == "register-backtest":
        # Load backtest results
        with open(args.backtest_json) as f:
            backtest_results = json.load(f)

        tracker.register_backtest(
            predictions_path=args.predictions_path,
            backtest_results=backtest_results,
        )

    elif args.command == "list":
        experiments = tracker.list_experiments(
            stage=args.stage,
            min_sharpe=args.min_sharpe,
            min_r2=args.min_r2,
        )

        print(f"\n{'='*80}")
        print(f"EXPERIMENTS ({len(experiments)} total)")
        print(f"{'='*80}\n")

        for exp in experiments:
            print(f"ID: {exp['experiment_id']}")
            print(f"  Stage: {exp['stage']}")
            print(f"  Created: {exp['created_at']}")
            print(f"  Snapshot: {exp['snapshot']['name']}")

            if exp.get("training"):
                perf = exp["training"]["performance_metrics"]
                print(f"  Training: R²={perf.get('test_r2', 'N/A'):.4f}, "
                      f"RMSE={perf.get('test_rmse', 'N/A')}")

            if exp.get("backtest"):
                results = exp["backtest"]["results"]
                print(f"  Backtest: Sharpe={results.get('sharpe_ratio', 'N/A'):.2f}, "
                      f"Return={results.get('total_return_pct', 'N/A'):.2f}%, "
                      f"Trades={results.get('total_trades', 'N/A')}")

            print()

    elif args.command == "get":
        exp = tracker.get_experiment(args.experiment_id)
        if exp:
            print(json.dumps(exp, indent=2))
        else:
            print(f"Experiment not found: {args.experiment_id}")

    elif args.command == "compare":
        comparison = tracker.compare_experiments(args.experiment_ids)
        print(json.dumps(comparison, indent=2))

    elif args.command == "best":
        best = tracker.get_best_experiment(
            metric=args.metric,
            min_trades=args.min_trades,
        )
        if best:
            print(f"\nBest experiment by {args.metric}:")
            print(f"  ID: {best['experiment_id']}")
            print(f"  {args.metric}: {best['backtest']['results'].get(args.metric)}")
            print(f"  Model: {best['training']['model_name']}")
        else:
            print("No experiments found matching criteria")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Baseline Model Training

Trains tabular baselines (Elastic Net, LightGBM) with walk-forward validation.

Usage:
    python scripts/train_baseline.py --snapshot baseline_30d --model elasticnet
    python scripts/train_baseline.py --snapshot baseline_30d --model lightgbm
    python scripts/train_baseline.py --snapshot baseline_30d --model elasticnet --output-dir ./models
"""

import argparse
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from typing import Tuple

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import (
    ModelMetadata,
    save_sklearn_model,
    save_lightgbm_model,
)


def load_snapshot_data(snapshot_dir: Path) -> pl.DataFrame:
    """Load data from snapshot directory (flat glob, no recursion)."""
    print(f"Loading snapshot from {snapshot_dir}...")

    files = list(snapshot_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files in {snapshot_dir}")

    dfs = [pl.read_parquet(f) for f in sorted(files)]
    return pl.concat(dfs)


def load_data_dir(data_dir: Path) -> pl.DataFrame:
    """Load data from a feature data directory (recursive glob across date subdirs)."""
    print(f"Loading data from {data_dir}...")

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found under {data_dir}")

    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs)
    print(f"  Loaded {len(df)} rows from {len(files)} files")
    return df


# Columns that are metadata / raw values, not computed features
_NON_FEATURE_PREFIXES = ("timestamp", "symbol", "raw_", "target_")
_NON_FEATURE_EXACT = {"sequence_id"}


def auto_detect_features(df: pl.DataFrame) -> list:
    """Auto-detect feature columns by excluding known metadata/raw columns."""
    feature_cols = [
        c for c in df.columns
        if c not in _NON_FEATURE_EXACT
        and not any(c.startswith(p) for p in _NON_FEATURE_PREFIXES)
    ]
    return feature_cols


def prepare_features_labels(
    df: pl.DataFrame,
    feature_cols: list,
    horizon: int = 600,
    max_nan_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare features and labels.

    Drops columns with >max_nan_frac NaN, then drops rows with any remaining NaN.

    Returns:
        (X, y, used_feature_cols) — the feature cols that survived NaN filtering.
    """
    # Drop high-NaN columns first
    n = len(df)
    usable_cols = []
    for c in feature_cols:
        col = df[c]
        null_count = col.null_count()
        nan_count = col.is_nan().sum() if col.dtype in (pl.Float32, pl.Float64) else 0
        if (null_count + nan_count) / max(n, 1) <= max_nan_frac:
            usable_cols.append(c)

    dropped = len(feature_cols) - len(usable_cols)
    if dropped > 0:
        print(f"  Dropped {dropped} columns with >{max_nan_frac:.0%} NaN")

    if not usable_cols:
        raise ValueError("No usable feature columns after NaN filtering")

    X = df.select(usable_cols).to_numpy().astype(np.float64)

    # Compute forward returns as labels
    price_col = "raw_midprice" if "raw_midprice" in df.columns else "midprice"
    prices = df[price_col].to_numpy().astype(np.float64)
    y = np.zeros(len(prices))

    for i in range(len(prices) - horizon):
        y[i] = (prices[i + horizon] - prices[i]) / prices[i]

    # Remove last samples (no valid label)
    X = X[:-horizon]
    y = y[:-horizon]

    # Remove rows with remaining NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]

    print(f"  Prepared {len(X)} samples with {X.shape[1]} features")
    return X, y, usable_cols


def train_elasticnet(X_train, y_train, X_test, y_test):
    """Train Elastic Net with CV.

    Returns:
        (model, scaler, hyperparameters, performance_metrics)
    """
    print("Training Elastic Net...")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with CV
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=5,
        random_state=42,
        max_iter=10000,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Alpha: {model.alpha_:.6f}")
    print(f"  L1 ratio: {model.l1_ratio_:.4f}")

    # Collect hyperparameters
    hyperparameters = {
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "max_iter": 10000,
        "cv_folds": 5,
        "random_state": 42,
    }

    # Collect performance metrics
    performance_metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, scaler, hyperparameters, performance_metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM.

    Returns:
        (model, hyperparameters, performance_metrics)
    """
    print("Training LightGBM...")

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Best iteration: {model.best_iteration}")

    # Collect hyperparameters
    hyperparameters = {
        **params,
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
        "best_iteration": model.best_iteration,
    }

    # Collect performance metrics
    performance_metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, hyperparameters, performance_metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=str, help="Snapshot name (under experiments/snapshots/)")
    group.add_argument("--data-dir", type=Path, help="Direct path to feature data directory")
    parser.add_argument("--model", type=str, choices=["elasticnet", "lightgbm"], required=True)
    parser.add_argument("--horizon", type=int, default=600, help="Forward horizon in ticks")
    parser.add_argument("--symbol", type=str, default=None,
                       help="Filter to a single symbol (e.g. BTC)")
    parser.add_argument("--features", type=str, nargs="*", default=None,
                       help="Explicit feature columns (default: auto-detect)")
    parser.add_argument("--output-dir", type=Path, default=Path("./models"),
                       help="Output directory for models (default: ./models)")
    parser.add_argument("--no-tracking", action="store_true",
                       help="Disable automatic experiment tracking")

    args = parser.parse_args()

    source_label = args.snapshot or str(args.data_dir)
    # Safe name for filenames (no slashes)
    source_safe = args.snapshot or Path(args.data_dir).name

    print("=" * 70)
    print("BASELINE MODEL TRAINING")
    print("=" * 70)
    print(f"Source: {source_label}")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon} ticks")
    print(f"Output: {args.output_dir}")
    if args.symbol:
        print(f"Symbol: {args.symbol}")
    print()

    # Load data
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            return
        df = load_data_dir(data_dir)
    else:
        snapshot_dir = Path("experiments/snapshots") / args.snapshot
        if not snapshot_dir.exists():
            print(f"Error: Snapshot directory not found: {snapshot_dir}")
            print()
            print("Available snapshots:")
            snapshots_base = Path("experiments/snapshots")
            if snapshots_base.exists():
                for d in snapshots_base.iterdir():
                    if d.is_dir():
                        print(f"  - {d.name}")
            else:
                print("  No snapshots found. Create one with:")
                print("  python scripts/experiment_governance.py snapshot --data-dir ./data/features --name baseline_30d")
            return
        df = load_snapshot_data(snapshot_dir)

    # Filter by symbol if specified
    if args.symbol:
        if "symbol" in df.columns:
            df = df.filter(pl.col("symbol") == args.symbol)
            print(f"Filtered to {args.symbol}: {len(df)} rows")
        else:
            print("Warning: No 'symbol' column found, skipping symbol filter")

    if len(df) == 0:
        print("Error: No data after filtering")
        return

    # Detect or use explicit feature columns
    if args.features:
        feature_cols = args.features
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"Error: Feature columns not found in data: {missing}")
            return
    else:
        feature_cols = auto_detect_features(df)
        print(f"Auto-detected {len(feature_cols)} feature columns")

    X, y, feature_cols = prepare_features_labels(df, feature_cols, args.horizon)

    if len(X) < 100:
        print(f"Error: Only {len(X)} valid samples after NaN removal (need >= 100)")
        sys.exit(1)

    # Train/test split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print()

    # Train model and save
    training_date = datetime.now().isoformat()

    if args.model == "elasticnet":
        model, scaler, hyperparameters, performance_metrics = train_elasticnet(
            X_train, y_train, X_test, y_test
        )

        # Create metadata
        metadata = ModelMetadata(
            model_type="elasticnet",
            model_name=f"elasticnet_baseline_{source_safe}",
            feature_names=feature_cols,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            training_date=training_date,
            snapshot_name=source_label,
            notes=f"Baseline Elastic Net model trained on {source_label}, "
                  f"{args.horizon}-tick forward returns"
        )

        # Save model
        print()
        print("Saving model...")
        model_path = save_sklearn_model(model, scaler, metadata, args.output_dir)

    elif args.model == "lightgbm":
        model, hyperparameters, performance_metrics = train_lightgbm(
            X_train, y_train, X_test, y_test
        )

        # Create metadata
        metadata = ModelMetadata(
            model_type="lightgbm",
            model_name=f"lightgbm_baseline_{source_safe}",
            feature_names=feature_cols,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            training_date=training_date,
            snapshot_name=source_label,
            notes=f"Baseline LightGBM model trained on {source_label}, "
                  f"{args.horizon}-tick forward returns"
        )

        # Save model
        print()
        print("Saving model...")
        model_path = save_lightgbm_model(model, metadata, args.output_dir)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Test R²: {performance_metrics['test_r2']:.4f}")
    print(f"Test RMSE: {performance_metrics['test_rmse']:.6f}")
    print()

    # Register experiment (unless disabled)
    if not args.no_tracking:
        try:
            from experiment_tracking import ExperimentTracker
            tracker = ExperimentTracker()
            experiment_id = tracker.register_training(
                snapshot_name=source_label,
                model_path=model_path,
            )
            print(f"📊 Experiment tracked: {experiment_id}")
            print()
        except Exception as e:
            print(f"Warning: Failed to track experiment: {e}")
            print()

    print("To use this model:")
    print(f"  python scripts/score_data.py --model {model_path} --data ./data/features")
    print()


if __name__ == "__main__":
    main()

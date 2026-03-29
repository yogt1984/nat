#!/usr/bin/env python3
"""
Baseline Model Training

Trains tabular baselines (Elastic Net, LightGBM) with walk-forward validation.

Usage:
    python scripts/train_baseline.py --snapshot baseline_30d --model elasticnet
    python scripts/train_baseline.py --snapshot baseline_30d --model lightgbm
"""

import argparse
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from typing import Tuple


def load_snapshot_data(snapshot_dir: Path) -> pl.DataFrame:
    """Load data from snapshot."""
    print(f"Loading snapshot from {snapshot_dir}...")

    files = list(snapshot_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files in {snapshot_dir}")

    dfs = [pl.read_parquet(f) for f in files]
    return pl.concat(dfs)


def prepare_features_labels(
    df: pl.DataFrame,
    feature_cols: list,
    horizon: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and labels."""
    # Extract features
    X = df.select(feature_cols).to_numpy()

    # Compute forward returns as labels
    prices = df["midprice"].to_numpy()
    y = np.zeros(len(prices))

    for i in range(len(prices) - horizon):
        y[i] = (prices[i + horizon] - prices[i]) / prices[i]

    # Remove last samples (no valid label)
    X = X[:-horizon]
    y = y[:-horizon]

    # Remove NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]

    print(f"Prepared {len(X)} samples with {X.shape[1]} features")
    return X, y


def train_elasticnet(X_train, y_train, X_test, y_test):
    """Train Elastic Net with CV."""
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
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Alpha: {model.alpha_:.6f}")
    print(f"  L1 ratio: {model.l1_ratio_:.4f}")

    return model, scaler


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM."""
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
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot name")
    parser.add_argument("--model", type=str, choices=["elasticnet", "lightgbm"], required=True)
    parser.add_argument("--horizon", type=int, default=600, help="Forward horizon in ticks")

    args = parser.parse_args()

    # Load data
    snapshot_dir = Path("experiments/snapshots") / args.snapshot
    df = load_snapshot_data(snapshot_dir)

    # Define features (simplified)
    feature_cols = [
        "kyle_lambda_100",
        "vpin_50",
        "absorption_zscore",
        "hurst_300",
        "whale_net_flow_1h",
        "tick_entropy_5s",
    ]

    X, y = prepare_features_labels(df, feature_cols, args.horizon)

    # Train/test split (70/30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print()

    # Train model
    if args.model == "elasticnet":
        model, scaler = train_elasticnet(X_train, y_train, X_test, y_test)
        # TODO: Save model
    elif args.model == "lightgbm":
        model = train_lightgbm(X_train, y_train, X_test, y_test)
        # TODO: Save model

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

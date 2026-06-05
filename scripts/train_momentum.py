#!/usr/bin/env python3
"""
Train Momentum Continuation classifier.

Usage:
    python scripts/train_momentum.py --symbol BTC --data-dir data/features
    python scripts/train_momentum.py --symbol BTC --data-dir data/features --use-lgbm

Loads 5-min bars, builds 7-feature matrix, trains logistic regression
(or LightGBM), evaluates via expanding-window walk-forward validation,
saves the model via model_io.

See docs/research/new/ml_algorithms.txt Section 1 for full specification.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model_io import ModelMetadata, save_sklearn_model
from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.loader import load_parquet

FEATURE_COLS = [
    "ent_tick_1m_mean",
    "ent_permutation_returns_16_mean",
    "trend_hurst_300_mean",
    "toxic_vpin_50_mean",
    "whale_net_flow_4h_sum",
    "regime_accumulation_score_mean",
    "vol_returns_5m_last",
]

HORIZON_BARS = 20  # 20 * 5min = 100min forward return


def load_bars(data_dir: str, symbol: str) -> pd.DataFrame:
    """Load parquet data and aggregate to 5-min bars."""
    base_cols = ["timestamp_ns", "symbol", "raw_midprice", "raw_spread"]
    # Feature columns needed (raw names before aggregation)
    raw_cols = [
        "ent_tick_1m", "ent_permutation_returns_16",
        "trend_hurst_300", "toxic_vpin_50",
        "whale_net_flow_4h", "regime_accumulation_score",
        "vol_returns_5m",
    ]
    load_cols = list(set(base_cols + raw_cols))

    df = load_parquet(data_dir, symbols=[symbol], columns=load_cols, max_memory_mb=4000)
    print(f"Loaded {len(df):,} ticks for {symbol}")

    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} ticks, need at least 1000")
        sys.exit(1)

    bars = aggregate_bars(df, timeframe="5min")
    print(f"Aggregated to {len(bars):,} bars")
    return bars


def build_dataset(bars: pd.DataFrame, horizon: int = HORIZON_BARS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix X and binary label y from bars.

    Returns (X, y, bars_valid) where rows with NaN features or labels are dropped.
    """
    # Forward returns
    mid = bars["raw_midprice_mean"].values
    fwd = np.full(len(mid), np.nan)
    fwd[:-horizon] = mid[horizon:] / mid[:-horizon] - 1.0

    bars = bars.copy()
    bars["fwd_return"] = fwd
    bars["label"] = (fwd > 0).astype(float)
    bars.loc[np.isnan(fwd), "label"] = np.nan

    # Check required columns
    missing = [c for c in FEATURE_COLS if c not in bars.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}")
        print(f"Available: {sorted(bars.columns.tolist())}")
        sys.exit(1)

    # Drop rows with NaN in features or label
    valid_mask = bars["label"].notna()
    for col in FEATURE_COLS:
        valid_mask &= bars[col].notna()

    bars_valid = bars[valid_mask].copy()
    X = bars_valid[FEATURE_COLS].values
    y = bars_valid["label"].values

    print(f"Valid samples: {len(y):,} / {len(bars):,} ({100*len(y)/len(bars):.1f}%)")
    print(f"Label balance: {y.mean():.3f} (positive rate)")

    if not (0.35 <= y.mean() <= 0.65):
        print(f"WARNING: Label balance {y.mean():.3f} outside [0.35, 0.65]")

    # Feature NaN report
    for col in FEATURE_COLS:
        nan_rate = bars[col].isna().mean()
        if nan_rate > 0.05:
            print(f"WARNING: {col} NaN rate = {nan_rate:.1%}")

    return X, y, bars_valid


def walk_forward_train(X: np.ndarray, y: np.ndarray, n_splits: int = 4,
                        embargo: int = 100, C: float = 1.0) -> dict:
    """Expanding-window walk-forward validation.

    Returns dict with OOS metrics and the final trained model.
    """
    n = len(y)
    min_train = n // (n_splits + 1)
    fold_size = (n - min_train) // n_splits

    print(f"\nWalk-forward: {n_splits} folds, {n} samples, min_train={min_train}")

    fold_results = []
    last_model = None
    last_scaler = None

    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end + embargo
        test_end = min(train_end + fold_size + embargo, n)

        if test_start >= n or test_end <= test_start:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if len(X_test) < 20:
            continue

        # Fit scaler + model
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
        model.fit(X_train_s, y_train)

        # Evaluate
        prob_train = model.predict_proba(X_train_s)[:, 1]
        prob_test = model.predict_proba(X_test_s)[:, 1]

        auc_train = roc_auc_score(y_train, prob_train)
        auc_test = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.5
        acc_train = accuracy_score(y_train, (prob_train > 0.5).astype(int))
        acc_test = accuracy_score(y_test, (prob_test > 0.5).astype(int))

        # Simple Sharpe proxy: sign(pred - 0.5) * fwd_return, annualized
        pred_signal = np.sign(prob_test - 0.5)
        # Use directional accuracy as Sharpe proxy
        correct_dir = np.mean(pred_signal == np.sign(y_test - 0.5))

        fold_results.append({
            "fold": fold,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "auc_train": auc_train,
            "auc_test": auc_test,
            "acc_train": acc_train,
            "acc_test": acc_test,
            "dir_accuracy": correct_dir,
        })

        last_model = model
        last_scaler = scaler

        print(f"  Fold {fold}: train={len(y_train):,} test={len(y_test):,} "
              f"AUC={auc_test:.4f} Acc={acc_test:.4f} DirAcc={correct_dir:.4f}")

    if not fold_results:
        print("ERROR: No valid folds")
        sys.exit(1)

    # Aggregate metrics
    avg_auc_oos = np.mean([f["auc_test"] for f in fold_results])
    avg_auc_is = np.mean([f["auc_train"] for f in fold_results])
    avg_acc_oos = np.mean([f["acc_test"] for f in fold_results])

    print(f"\nOOS Average: AUC={avg_auc_oos:.4f} Acc={avg_acc_oos:.4f}")
    print(f"IS Average:  AUC={avg_auc_is:.4f}")
    print(f"OOS/IS AUC ratio: {avg_auc_oos/avg_auc_is:.3f}" if avg_auc_is > 0 else "")

    # Final model: train on all data
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)
    final_model = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
    final_model.fit(X_all_s, y)

    # Feature importance (coefficients)
    print("\nFeature coefficients (final model):")
    for name, coef in zip(FEATURE_COLS, final_model.coef_[0]):
        print(f"  {name:40s} {coef:+.6f}")
    print(f"  {'intercept':40s} {final_model.intercept_[0]:+.6f}")

    return {
        "model": final_model,
        "scaler": final_scaler,
        "fold_results": fold_results,
        "avg_auc_oos": avg_auc_oos,
        "avg_auc_is": avg_auc_is,
        "avg_acc_oos": avg_acc_oos,
        "n_samples": len(y),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Momentum Continuation classifier")
    parser.add_argument("--symbol", default="BTC", help="Symbol to train on")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization (1/lambda)")
    parser.add_argument("--n-splits", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--embargo", type=int, default=100, help="Embargo bars between train/test")
    parser.add_argument("--output-dir", default="models/momentum_continuation",
                        help="Output directory for trained model")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    args = parser.parse_args()

    print(f"=== Training Momentum Continuation: {args.symbol} ===")
    print(f"Data: {args.data_dir}")
    print(f"C={args.C}, n_splits={args.n_splits}, embargo={args.embargo}")

    # Load and build dataset
    bars = load_bars(args.data_dir, args.symbol)
    X, y, bars_valid = build_dataset(bars)

    if len(y) < 500:
        print(f"ERROR: Only {len(y)} valid samples, need at least 500")
        sys.exit(1)

    # Walk-forward validation
    result = walk_forward_train(X, y, n_splits=args.n_splits,
                                 embargo=args.embargo, C=args.C)

    # Decision gate
    print(f"\n{'='*60}")
    if result["avg_auc_oos"] < 0.52:
        print(f"FAIL: OOS AUC {result['avg_auc_oos']:.4f} < 0.52 threshold")
        print("Model is not better than random. Do not deploy.")
        if not args.dry_run:
            print("Saving anyway for diagnostic purposes...")
    else:
        print(f"PASS: OOS AUC {result['avg_auc_oos']:.4f} >= 0.52")

    # Save model
    if not args.dry_run:
        metadata = ModelMetadata(
            model_type="logistic_regression",
            model_name="momentum_continuation",
            feature_names=FEATURE_COLS,
            hyperparameters={"C": args.C, "penalty": "l2", "max_iter": 1000},
            performance_metrics={
                "avg_auc_oos": result["avg_auc_oos"],
                "avg_auc_is": result["avg_auc_is"],
                "avg_acc_oos": result["avg_acc_oos"],
                "n_samples": result["n_samples"],
                "n_folds": len(result["fold_results"]),
            },
            training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=f"symbol={args.symbol}, embargo={args.embargo}",
        )

        save_sklearn_model(
            model=result["model"],
            scaler=result["scaler"],
            metadata=metadata,
            output_dir=Path(args.output_dir),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

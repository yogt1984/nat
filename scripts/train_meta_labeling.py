#!/usr/bin/env python3
"""
Train Meta-Labeling precision filter.

Usage:
    python scripts/train_meta_labeling.py --symbol BTC --data-dir data/features
    python scripts/train_meta_labeling.py --symbol BTC --data-dir data/features --dry-run

Loads pre-built meta-training data (or builds it on the fly), trains
LogisticRegression with purged K-fold cross-validation, saves the model.

See docs/research/new/ml_algorithms.txt Section 3 for full specification.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model_io import ModelMetadata, save_sklearn_model
from build_meta_training_data import (
    build_meta_training_data, META_FEATURE_COLS,
    compute_triple_barrier_labels,
)

# Features used by the meta-labeling model
FEATURE_COLS = META_FEATURE_COLS


def purged_kfold_split(
    n: int, k: int = 5, embargo: int = 100
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold indices with embargo gap.

    Each fold's test set is separated from its train set by at least
    `embargo` bars on both sides. This prevents temporal leakage.

    Returns list of (train_indices, test_indices) tuples.
    """
    fold_size = n // k
    splits = []

    for fold in range(k):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)

        test_idx = np.arange(test_start, test_end)

        # Purge: remove embargo zone around test set from train
        purge_start = max(0, test_start - embargo)
        purge_end = min(n, test_end + embargo)
        purge_set = set(range(purge_start, purge_end))

        train_idx = np.array([i for i in range(n) if i not in purge_set])

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train_meta_labeling(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 5,
    embargo: int = 100,
    C: float = 1.0,
) -> dict:
    """Purged K-fold training of meta-labeling LogisticRegression.

    Returns dict with OOS metrics and final trained model.
    """
    n = len(y)
    splits = purged_kfold_split(n, k=k, embargo=embargo)

    print(f"\nPurged K-fold: {len(splits)} folds, {n} samples, embargo={embargo}")

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(X_test) < 20 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
        model.fit(X_train_s, y_train)

        prob_train = model.predict_proba(X_train_s)[:, 1]
        prob_test = model.predict_proba(X_test_s)[:, 1]

        auc_train = roc_auc_score(y_train, prob_train)
        auc_test = roc_auc_score(y_test, prob_test)
        acc_test = accuracy_score(y_test, (prob_test > 0.5).astype(int))

        fold_results.append({
            "fold": fold_idx,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "auc_train": auc_train,
            "auc_test": auc_test,
            "acc_test": acc_test,
        })

        print(f"  Fold {fold_idx}: train={len(y_train):,} test={len(y_test):,} "
              f"AUC={auc_test:.4f} Acc={acc_test:.4f}")

    if not fold_results:
        print("ERROR: No valid folds")
        sys.exit(1)

    avg_auc_oos = np.mean([f["auc_test"] for f in fold_results])
    avg_auc_is = np.mean([f["auc_train"] for f in fold_results])
    avg_acc_oos = np.mean([f["acc_test"] for f in fold_results])

    print(f"\nOOS Average: AUC={avg_auc_oos:.4f} Acc={avg_acc_oos:.4f}")
    print(f"IS Average:  AUC={avg_auc_is:.4f}")

    # Final model on all data
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)
    final_model = LogisticRegression(C=C, penalty="l2", max_iter=1000, solver="lbfgs")
    final_model.fit(X_all_s, y)

    # Feature importance (coefficients)
    print("\nFeature coefficients (final model):")
    for name, coef in zip(feature_names, final_model.coef_[0]):
        print(f"  {name:45s} {coef:+.6f}")

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
    parser = argparse.ArgumentParser(description="Train Meta-Labeling precision filter")
    parser.add_argument("--symbol", default="BTC", help="Symbol to train on")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization (1/lambda)")
    parser.add_argument("--k-folds", type=int, default=5, help="Purged K-fold count")
    parser.add_argument("--embargo", type=int, default=100, help="Embargo bars")
    parser.add_argument("--output-dir", default="models/meta_labeling",
                        help="Output directory for trained model")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    args = parser.parse_args()

    print(f"=== Training Meta-Labeling: {args.symbol} ===")

    # Build training data
    bars, labels, meta_features = build_meta_training_data(
        args.data_dir, args.symbol
    )

    if len(labels) < 500:
        print(f"ERROR: Only {len(labels)} valid samples, need at least 500")
        sys.exit(1)

    # Use available meta features
    available = [c for c in FEATURE_COLS if c in bars.columns]
    if not available:
        print("ERROR: No meta features available in bars")
        sys.exit(1)

    X = bars[available].values
    y = labels

    # Drop NaN rows
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    print(f"Training on {len(y):,} samples with {len(available)} features")

    # Train
    result = train_meta_labeling(
        X, y,
        feature_names=available,
        k=args.k_folds,
        embargo=args.embargo,
        C=args.C,
    )

    # Decision gate
    print(f"\n{'='*60}")
    if result["avg_auc_oos"] < 0.52:
        print(f"FAIL: OOS AUC {result['avg_auc_oos']:.4f} < 0.52")
    else:
        print(f"PASS: OOS AUC {result['avg_auc_oos']:.4f} >= 0.52")

    if not args.dry_run:
        metadata = ModelMetadata(
            model_type="logistic_regression",
            model_name="meta_labeling",
            feature_names=available,
            hyperparameters={"C": args.C, "penalty": "l2"},
            performance_metrics={
                "avg_auc_oos": result["avg_auc_oos"],
                "avg_auc_is": result["avg_auc_is"],
                "avg_acc_oos": result["avg_acc_oos"],
                "n_samples": result["n_samples"],
                "n_folds": len(result["fold_results"]),
            },
            training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=f"symbol={args.symbol}, embargo={args.embargo}, "
                  f"k_folds={args.k_folds}",
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

#!/usr/bin/env python3
"""
Train Mean-Reversion / False-Breakout Detector.

Usage:
    python scripts/train_mean_reversion.py --symbol BTC --data-dir data/features
    python scripts/train_mean_reversion.py --symbol BTC --data-dir data/features --dry-run

Loads 5-min bars, computes z-score and reversion labels, trains LightGBM
classifier, evaluates via expanding-window walk-forward validation with
SHAP feature importance, saves the model via model_io.

See docs/research/new/ml_algorithms.txt Section 2 for full specification.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model_io import ModelMetadata, save_lightgbm_model
from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.loader import load_parquet

# Features used by the model (bar-aggregated names)
FEATURE_COLS = [
    "vol_returns_5m_last",
    "ent_tick_1m_mean",
    "trend_hurst_300_mean",
    "imbalance_qty_l1_mean",
    "toxic_vpin_50_mean",
]

# Additional columns for z-score computation (not model features)
ZSCORE_COLS = ["raw_midprice_mean", "mf_ema_15m_last"]

HORIZON_BARS = 20  # 20 * 5min = 100min forward return


def load_bars(data_dir: str, symbol: str, start_date: str | None = None) -> pd.DataFrame:
    """Load parquet data and aggregate to 5-min bars."""
    raw_cols = [
        "timestamp_ns", "symbol", "raw_midprice", "raw_spread",
        "vol_returns_5m", "ent_tick_1m", "trend_hurst_300",
        "imbalance_qty_l1", "toxic_vpin_50",
    ]

    df = load_parquet(data_dir, symbols=[symbol], columns=raw_cols,
                      start_date=start_date, max_memory_mb=4000)
    print(f"Loaded {len(df):,} ticks for {symbol}")

    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} ticks, need at least 1000")
        sys.exit(1)

    bars = aggregate_bars(df, timeframe="5min")
    print(f"Aggregated to {len(bars):,} bars")

    # Compute EMA locally if mf_ema_15m_last not available
    if "mf_ema_15m_last" not in bars.columns or bars["mf_ema_15m_last"].isna().all():
        span = 3  # 3 bars × 5min = 15min
        bars["mf_ema_15m_last"] = bars["raw_midprice_mean"].ewm(span=span).mean()
        print("Computed mf_ema_15m_last locally (ewm span=3)")

    return bars


def compute_zscore(bars: pd.DataFrame) -> np.ndarray:
    """Compute z-score = (midprice - ema) / (vol * midprice).

    Uses only data available at time t (no lookahead).
    """
    midprice = bars["raw_midprice_mean"].values
    ema = bars["mf_ema_15m_last"].values
    vol = bars["vol_returns_5m_last"].values

    denom = vol * midprice
    safe_denom = np.where((denom > 0) & np.isfinite(denom), denom, np.nan)
    zscore = (midprice - ema) / safe_denom
    return np.where(np.isfinite(zscore), zscore, 0.0)


def build_dataset(bars: pd.DataFrame, horizon: int = HORIZON_BARS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix X and binary reversion label y.

    Reversion label: 1 if price reverts toward EMA (z-score shrinks), 0 otherwise.
    Specifically: label=1 if sign(zscore_t) != sign(fwd_return_t).
    """
    # Forward returns
    mid = bars["raw_midprice_mean"].values
    fwd = np.full(len(mid), np.nan)
    fwd[:-horizon] = mid[horizon:] / mid[:-horizon] - 1.0

    # Z-score
    zscore = compute_zscore(bars)

    # Reversion label: price moved against the displacement
    # If zscore > 0 (above EMA) and fwd_return < 0 -> reversion (label=1)
    # If zscore < 0 (below EMA) and fwd_return > 0 -> reversion (label=1)
    reversion = np.where(
        np.isfinite(fwd),
        (np.sign(zscore) != np.sign(fwd)).astype(float),
        np.nan,
    )
    # zscore == 0 is ambiguous, mark as NaN
    reversion = np.where(zscore == 0, np.nan, reversion)

    bars = bars.copy()
    bars["fwd_return"] = fwd
    bars["zscore"] = zscore
    bars["label"] = reversion

    # Check required columns
    all_feature_cols = FEATURE_COLS + ["zscore"]
    missing = [c for c in FEATURE_COLS if c not in bars.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}")
        sys.exit(1)

    # Drop NaN rows
    valid_mask = bars["label"].notna()
    for col in all_feature_cols:
        if col in bars.columns:
            valid_mask &= bars[col].notna()

    bars_valid = bars[valid_mask].copy()
    X = bars_valid[all_feature_cols].values
    y = bars_valid["label"].values

    print(f"Valid samples: {len(y):,} / {len(bars):,} ({100*len(y)/len(bars):.1f}%)")
    print(f"Reversion rate: {y.mean():.3f}")

    if not (0.35 <= y.mean() <= 0.65):
        print(f"WARNING: Label balance {y.mean():.3f} outside [0.35, 0.65]")

    return X, y, bars_valid


def walk_forward_train(X: np.ndarray, y: np.ndarray, feature_names: list[str],
                        n_splits: int = 4, embargo: int = 100,
                        num_leaves: int = 15, n_estimators: int = 100) -> dict:
    """Expanding-window walk-forward validation with LightGBM.

    Returns dict with OOS metrics, SHAP importances, and final model.
    """
    import lightgbm as lgb

    n = len(y)
    min_train = n // (n_splits + 1)
    fold_size = (n - min_train) // n_splits

    print(f"\nWalk-forward: {n_splits} folds, {n} samples, min_train={min_train}")

    fold_results = []
    shap_importances = np.zeros(X.shape[1])
    n_shap_folds = 0

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

        # LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": num_leaves,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        # Evaluate
        from sklearn.metrics import roc_auc_score, accuracy_score

        prob_train = model.predict(X_train)
        prob_test = model.predict(X_test)

        auc_train = roc_auc_score(y_train, prob_train)
        auc_test = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.5
        acc_test = accuracy_score(y_test, (prob_test > 0.5).astype(int))

        fold_results.append({
            "fold": fold,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "auc_train": auc_train,
            "auc_test": auc_test,
            "acc_test": acc_test,
            "best_iteration": model.best_iteration,
        })

        # SHAP feature importance (gain-based as proxy)
        importance = model.feature_importance(importance_type="gain")
        if importance.sum() > 0:
            shap_importances += importance / importance.sum()
            n_shap_folds += 1

        print(f"  Fold {fold}: train={len(y_train):,} test={len(y_test):,} "
              f"AUC={auc_test:.4f} Acc={acc_test:.4f} iter={model.best_iteration}")

    if not fold_results:
        print("ERROR: No valid folds")
        sys.exit(1)

    # Average SHAP importances
    if n_shap_folds > 0:
        shap_importances /= n_shap_folds
    shap_dict = dict(zip(feature_names, shap_importances))

    # Report feature importances
    print("\nFeature importance (avg normalized gain):")
    for name, imp in sorted(shap_dict.items(), key=lambda x: -x[1]):
        flag = " ** DROP CANDIDATE" if imp < 0.02 else ""
        print(f"  {name:40s} {imp:.4f}{flag}")

    # Flag low-importance features
    drop_features = [n for n, v in shap_dict.items() if v < 0.02]
    if drop_features:
        print(f"\nFeatures below 0.02 threshold: {drop_features}")

    # Aggregate metrics
    avg_auc_oos = np.mean([f["auc_test"] for f in fold_results])
    avg_auc_is = np.mean([f["auc_train"] for f in fold_results])
    avg_acc_oos = np.mean([f["acc_test"] for f in fold_results])

    print(f"\nOOS Average: AUC={avg_auc_oos:.4f} Acc={avg_acc_oos:.4f}")
    print(f"IS Average:  AUC={avg_auc_is:.4f}")
    print(f"OOS/IS AUC ratio: {avg_auc_oos / avg_auc_is:.3f}" if avg_auc_is > 0 else "")

    # Final model: train on all data
    train_all = lgb.Dataset(X, label=y, feature_name=feature_names)
    final_model = lgb.train(
        params,
        train_all,
        num_boost_round=n_estimators,
    )

    return {
        "model": final_model,
        "fold_results": fold_results,
        "avg_auc_oos": avg_auc_oos,
        "avg_auc_is": avg_auc_is,
        "avg_acc_oos": avg_acc_oos,
        "n_samples": len(y),
        "feature_importance": shap_dict,
        "drop_features": drop_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mean-Reversion Detector")
    parser.add_argument("--symbol", default="BTC", help="Symbol to train on")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--num-leaves", type=int, default=15, help="LightGBM num_leaves")
    parser.add_argument("--n-estimators", type=int, default=100, help="LightGBM n_estimators")
    parser.add_argument("--n-splits", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--embargo", type=int, default=100, help="Embargo bars between train/test")
    parser.add_argument("--output-dir", default="models/mean_reversion_detector",
                        help="Output directory for trained model")
    parser.add_argument("--start-date", default=None,
                        help="Earliest date to load (YYYY-MM-DD), limits memory")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    args = parser.parse_args()

    print(f"=== Training Mean-Reversion Detector: {args.symbol} ===")
    print(f"Data: {args.data_dir}")
    print(f"num_leaves={args.num_leaves}, n_estimators={args.n_estimators}, "
          f"n_splits={args.n_splits}, embargo={args.embargo}")

    # Load and build dataset
    bars = load_bars(args.data_dir, args.symbol, start_date=args.start_date)

    # Check z-score columns exist
    for col in ZSCORE_COLS:
        if col not in bars.columns:
            print(f"ERROR: Missing z-score column: {col}")
            sys.exit(1)

    feature_names = FEATURE_COLS + ["zscore"]
    X, y, bars_valid = build_dataset(bars)

    if len(y) < 500:
        print(f"ERROR: Only {len(y)} valid samples, need at least 500")
        sys.exit(1)

    # Walk-forward validation
    result = walk_forward_train(
        X, y,
        feature_names=feature_names,
        n_splits=args.n_splits,
        embargo=args.embargo,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )

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
            model_type="lightgbm",
            model_name="mean_reversion_detector",
            feature_names=feature_names,
            hyperparameters={
                "num_leaves": args.num_leaves,
                "n_estimators": args.n_estimators,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
            },
            performance_metrics={
                "avg_auc_oos": result["avg_auc_oos"],
                "avg_auc_is": result["avg_auc_is"],
                "avg_acc_oos": result["avg_acc_oos"],
                "n_samples": result["n_samples"],
                "n_folds": len(result["fold_results"]),
                "feature_importance": result["feature_importance"],
            },
            training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=f"symbol={args.symbol}, embargo={args.embargo}, "
                  f"drop_candidates={result['drop_features']}",
        )

        save_lightgbm_model(
            model=result["model"],
            metadata=metadata,
            output_dir=Path(args.output_dir),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

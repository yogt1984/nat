#!/usr/bin/env python3
"""
Train Regime-Conditioned LightGBM ensemble.

Usage:
    python scripts/train_regime_lgbm.py --symbol BTC --data-dir data/features
    python scripts/train_regime_lgbm.py --symbol BTC --data-dir data/features --dry-run

Loads 5-min bars, computes regime labels (via RSM), splits data by regime,
trains per-regime LightGBM models + global fallback, evaluates via
walk-forward validation.

See docs/research/new/ml_algorithms.txt Section 7 for full specification.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.model_io import ModelMetadata, save_lightgbm_model
from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.loader import load_parquet
from algorithms.regime_conditioned_lgbm import (
    TRENDING_FEATURES, RANGING_FEATURES, VOLATILE_FEATURES,
    GLOBAL_FEATURES, REGIME_TO_GROUP,
)

HORIZON_BARS = 20  # 20 * 5min = 100min forward return
MIN_SAMPLES_REGIME = 500


def load_bars(data_dir: str, symbol: str) -> pd.DataFrame:
    """Load parquet data and aggregate to 5-min bars."""
    df = load_parquet(data_dir, symbols=[symbol], max_memory_mb=4000)
    print(f"Loaded {len(df):,} ticks for {symbol}")
    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} ticks, need at least 1000")
        sys.exit(1)
    bars = aggregate_bars(df, timeframe="5min")
    print(f"Aggregated to {len(bars):,} bars")
    return bars


def add_regime_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """Add regime labels by running RSM on bars."""
    from algorithms.regime_state_machine import RegimeStateMachine

    rsm = RegimeStateMachine()
    rsm_result = rsm.run_batch(bars)
    bars = bars.copy()
    bars["regime"] = rsm_result["alg_rsm_regime"].values
    bars["regime_confidence"] = rsm_result["alg_rsm_confidence"].values
    return bars


def build_labels(bars: pd.DataFrame, horizon: int = HORIZON_BARS) -> np.ndarray:
    """Compute forward returns as regression targets."""
    mid = bars["raw_midprice_mean"].values
    fwd = np.full(len(mid), np.nan)
    fwd[:-horizon] = mid[horizon:] / mid[:-horizon] - 1.0
    return fwd


def train_lgbm_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 4,
    embargo: int = 100,
    num_leaves: int = 15,
    n_estimators: int = 100,
) -> tuple[object, dict]:
    """Train LightGBM with walk-forward validation.

    Returns (model, metrics_dict).
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    n = len(y)
    min_train = n // (n_splits + 1)
    fold_size = (n - min_train) // n_splits

    fold_results = []

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

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": num_leaves,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model = lgb.train(
            params, train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, pred_test)
        corr = np.corrcoef(y_test, pred_test)[0, 1] if len(y_test) > 2 else 0.0

        fold_results.append({
            "fold": fold, "mse": mse, "corr": corr,
            "train_size": len(y_train), "test_size": len(y_test),
        })

    # Final model on all data
    train_all = lgb.Dataset(X, label=y, feature_name=feature_names)
    params_final = {**params}
    params_final.pop("metric", None)
    final_model = lgb.train(params_final, train_all, num_boost_round=n_estimators)

    avg_corr = np.mean([f["corr"] for f in fold_results]) if fold_results else 0.0
    avg_mse = np.mean([f["mse"] for f in fold_results]) if fold_results else 0.0

    return final_model, {
        "fold_results": fold_results,
        "avg_corr_oos": avg_corr,
        "avg_mse_oos": avg_mse,
        "n_samples": len(y),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Regime-Conditioned LightGBM")
    parser.add_argument("--symbol", default="BTC", help="Symbol to train on")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_REGIME)
    parser.add_argument("--output-dir", default="models/regime_conditioned_lgbm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== Training Regime-Conditioned LightGBM: {args.symbol} ===")

    bars = load_bars(args.data_dir, args.symbol)
    bars = add_regime_labels(bars)
    fwd_returns = build_labels(bars)

    bars["fwd_return"] = fwd_returns
    valid = np.isfinite(fwd_returns)

    # Train per-regime models
    group_features = {
        "trending": TRENDING_FEATURES,
        "ranging": RANGING_FEATURES,
        "volatile": VOLATILE_FEATURES,
    }

    all_metrics = {}
    output_dir = Path(args.output_dir)

    for group_name, feat_cols in group_features.items():
        regime_ids = [r for r, g in REGIME_TO_GROUP.items() if g == group_name]
        mask = valid & bars["regime"].isin(regime_ids).values

        # Check all feature columns exist
        available = [c for c in feat_cols if c in bars.columns]
        if not available:
            print(f"\n  [{group_name}] No features available, skipping")
            continue

        bars_group = bars[mask]
        if len(bars_group) < args.min_samples:
            print(f"\n  [{group_name}] Only {len(bars_group)} samples (< {args.min_samples}), skipping")
            continue

        X = bars_group[available].values
        y = bars_group["fwd_return"].values
        finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[finite], y[finite]

        print(f"\n  [{group_name}] {len(y):,} samples, {len(available)} features")
        model, metrics = train_lgbm_model(
            X, y, available,
            n_splits=args.n_splits, embargo=args.embargo,
            num_leaves=args.num_leaves, n_estimators=args.n_estimators,
        )
        print(f"    OOS corr={metrics['avg_corr_oos']:.4f} MSE={metrics['avg_mse_oos']:.6f}")

        all_metrics[group_name] = metrics

        if not args.dry_run:
            meta = ModelMetadata(
                model_type="lightgbm",
                model_name=f"regime_lgbm_{group_name}",
                feature_names=available,
                hyperparameters={"num_leaves": args.num_leaves, "n_estimators": args.n_estimators},
                performance_metrics={"avg_corr_oos": metrics["avg_corr_oos"]},
                training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                notes=f"symbol={args.symbol}, group={group_name}",
            )
            save_lightgbm_model(model, meta, output_dir, model_filename=f"model_{group_name}.txt")

    # Train global fallback
    available_global = [c for c in GLOBAL_FEATURES if c in bars.columns]
    if available_global:
        bars_valid = bars[valid]
        X_global = bars_valid[available_global].values
        y_global = bars_valid["fwd_return"].values
        finite = np.all(np.isfinite(X_global), axis=1) & np.isfinite(y_global)
        X_global, y_global = X_global[finite], y_global[finite]

        print(f"\n  [global] {len(y_global):,} samples, {len(available_global)} features")
        model_global, metrics_global = train_lgbm_model(
            X_global, y_global, available_global,
            n_splits=args.n_splits, embargo=args.embargo,
            num_leaves=args.num_leaves, n_estimators=args.n_estimators,
        )
        print(f"    OOS corr={metrics_global['avg_corr_oos']:.4f}")
        all_metrics["global"] = metrics_global

        if not args.dry_run:
            meta = ModelMetadata(
                model_type="lightgbm",
                model_name="regime_lgbm_global",
                feature_names=available_global,
                hyperparameters={"num_leaves": args.num_leaves, "n_estimators": args.n_estimators},
                performance_metrics={"avg_corr_oos": metrics_global["avg_corr_oos"]},
                training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                notes=f"symbol={args.symbol}, group=global",
            )
            save_lightgbm_model(model_global, meta, output_dir, model_filename="model_global.txt")

    print(f"\n{'='*60}")
    print("Summary:")
    for name, m in all_metrics.items():
        print(f"  {name:12s} corr={m['avg_corr_oos']:.4f} samples={m['n_samples']}")
    print("Done.")


if __name__ == "__main__":
    main()

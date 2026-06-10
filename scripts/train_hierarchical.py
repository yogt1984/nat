#!/usr/bin/env python3
"""
Train Hierarchical Signal Combiner weights.

Usage:
    python scripts/train_hierarchical.py --symbol BTC --data-dir data/features
    python scripts/train_hierarchical.py --symbol BTC --data-dir data/features --dry-run

Loads 5-min bars, computes rolling IC per feature layer, trains IC-weighted
combination via walk-forward validation, saves weights to JSON.

See docs/research/new/9_6/ for the IC scan and validation reports that
motivate the three-layer architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.loader import load_parquet
from algorithms.hierarchical_combiner import (
    L1_FEATURES, L2_FEATURES, L3_FEATURES,
    DEFAULT_L1_WEIGHTS, DEFAULT_L2_WEIGHTS, DEFAULT_L3_WEIGHTS,
    _rolling_zscore, _ic_weighted_composite,
)

HORIZON_BARS = 60  # 60 * 5min = 5h forward return
VALID_MODES = ("walk_forward", "purged_kfold")

ABLATION_MODES = ["full", "l1_only", "no_l3", "no_l2"]


def _compute_composite(
    l1_z: pd.Series,
    l2_z: pd.Series,
    l3_z: pd.Series,
    l1_threshold: float,
    mode: str = "full",
) -> tuple[pd.Series, pd.Series]:
    """Compute hierarchical composite with ablation support.

    Returns (composite, l1_direction).
    """
    # Layer 1: directional bias
    l1_dir = pd.Series(0.0, index=l1_z.index)
    l1_dir[l1_z > l1_threshold] = 1.0
    l1_dir[l1_z < -l1_threshold] = -1.0

    # Layer 2: entry timing
    l2_entry = l2_z.clip(-3, 3) / 3.0

    # Layer 3: vol sizing
    l3_scale = 1.0 / (1.0 + np.exp(l3_z.clip(-5, 5)))

    # Apply ablation overrides
    if mode == "l1_only":
        l2_entry = pd.Series(1.0, index=l1_z.index)
        l3_scale = pd.Series(0.5, index=l1_z.index)
    elif mode == "no_l3":
        l3_scale = pd.Series(0.5, index=l1_z.index)
    elif mode == "no_l2":
        # L2 passthrough (no gating), but keep its magnitude
        pass  # skip gating below

    # Gate L2 by L1 alignment (skip for no_l2 mode)
    if mode != "no_l2" and mode != "l1_only":
        l1_active = l1_dir != 0
        aligned = np.sign(l2_entry) == l1_dir
        l2_entry = l2_entry.copy()
        l2_entry[l1_active & ~aligned] = 0.0

    # Composite assembly
    composite = l1_dir * l2_entry.abs() * l3_scale
    neutral = l1_dir == 0
    composite[neutral] = l2_entry[neutral] * l3_scale[neutral] * 0.3

    return composite.clip(-1, 1), l1_dir


def load_bars(data_dir: str, symbol: str, start_date: str | None = None) -> pd.DataFrame:
    """Load parquet data and aggregate to 5-min bars."""
    df = load_parquet(data_dir, symbols=[symbol], start_date=start_date, max_memory_mb=4000)
    print(f"Loaded {len(df):,} ticks for {symbol}")
    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} ticks, need at least 1000")
        sys.exit(1)
    bars = aggregate_bars(df, timeframe="5min")
    print(f"Aggregated to {len(bars):,} bars")
    return bars


def build_forward_returns(bars: pd.DataFrame, horizon: int = HORIZON_BARS) -> np.ndarray:
    """Compute forward returns as regression targets."""
    mid = bars["raw_midprice_mean"].values
    fwd = np.full(len(mid), np.nan)
    fwd[:-horizon] = mid[horizon:] / mid[:-horizon] - 1.0
    return fwd


def compute_rolling_ic(
    feature: pd.Series,
    returns: pd.Series,
    window: int = 200,
) -> pd.Series:
    """Rolling Spearman IC between feature and forward returns."""
    ic_values = pd.Series(np.nan, index=feature.index)

    feat_vals = feature.values
    ret_vals = returns.values

    for i in range(window, len(feature)):
        f_win = feat_vals[i - window:i]
        r_win = ret_vals[i - window:i]
        valid = np.isfinite(f_win) & np.isfinite(r_win)
        if valid.sum() < 30:
            continue
        corr, _ = spearmanr(f_win[valid], r_win[valid])
        ic_values.iloc[i] = corr

    return ic_values


def train_layer_weights(
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    features: list[str],
    default_weights: dict[str, float],
    window: int = 200,
    label: str = "",
) -> dict[str, float]:
    """Compute mean IC per feature as weights."""
    returns = pd.Series(fwd_returns, index=bars.index)
    weights = {}

    for feat in features:
        if feat not in bars.columns:
            print(f"  [{label}] {feat}: MISSING, using default")
            weights[feat] = default_weights.get(feat, 0.01)
            continue

        if bars[feat].notna().mean() < 0.1:
            print(f"  [{label}] {feat}: >90% NaN, using default")
            weights[feat] = default_weights.get(feat, 0.01)
            continue

        rolling_ic = compute_rolling_ic(bars[feat], returns, window=window)
        mean_ic = rolling_ic.dropna().mean()
        weights[feat] = float(mean_ic) if np.isfinite(mean_ic) else 0.0
        print(f"  [{label}] {feat}: mean_IC={mean_ic:+.4f}")

    return weights


def _eval_fold_metrics(
    composite: pd.Series,
    l1_dir: pd.Series,
    fwd_return: pd.Series,
    test_start: int,
    test_end: int,
) -> dict | None:
    """Compute OOS metrics for a single fold slice."""
    tc_series = composite.iloc[test_start:test_end]
    tr_series = fwd_return.iloc[test_start:test_end]
    l1_slice = l1_dir.iloc[test_start:test_end]

    valid_mask = np.isfinite(tc_series.values) & np.isfinite(tr_series.values)
    if valid_mask.sum() < 30:
        return None

    tc = tc_series.values[valid_mask]
    tr = tr_series.values[valid_mask]

    composite_ic, _ = spearmanr(tc, tr)
    dir_acc = np.mean(np.sign(tc) == np.sign(tr))

    cost_bps = 11e-4
    signal_returns = np.sign(tc) * tr - cost_bps * (np.abs(np.diff(np.sign(tc), prepend=0)) > 0)
    sharpe_proxy = np.mean(signal_returns) / max(np.std(signal_returns), 1e-8) * np.sqrt(252 * 288)

    l1_active_pct = (l1_slice.values != 0).mean()

    return {
        "test_size": int(valid_mask.sum()),
        "composite_ic": float(composite_ic),
        "dir_accuracy": float(dir_acc),
        "sharpe_proxy": float(sharpe_proxy),
        "l1_active_pct": float(l1_active_pct),
    }


def walk_forward_evaluate(
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    l1_weights: dict[str, float],
    l2_weights: dict[str, float],
    l3_weights: dict[str, float],
    l1_threshold: float = 0.5,
    n_splits: int = 4,
    embargo: int = 100,
    zscore_window: int = 200,
    ablation: bool = False,
) -> list[dict]:
    """Walk-forward evaluation of hierarchical composite signal."""
    n = len(bars)
    min_train = n // (n_splits + 1)
    fold_size = (n - min_train) // n_splits

    bars = bars.copy()
    bars["fwd_return"] = fwd_returns

    modes = ABLATION_MODES if ablation else ["full"]
    fold_results = []

    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start = train_end + embargo
        test_end = min(train_end + fold_size + embargo, n)

        if test_start >= n or test_end <= test_start:
            break

        if test_end - test_start < 50:
            continue

        eval_bars = bars.iloc[:test_end].copy()

        l1_z = _ic_weighted_composite(eval_bars, L1_FEATURES, l1_weights, window=zscore_window)
        l2_z = _ic_weighted_composite(eval_bars, L2_FEATURES, l2_weights, window=zscore_window)
        l3_z = _ic_weighted_composite(eval_bars, L3_FEATURES, l3_weights, window=zscore_window)

        fold_entry = {"fold": fold}

        for mode in modes:
            composite, l1_dir = _compute_composite(l1_z, l2_z, l3_z, l1_threshold, mode=mode)
            metrics = _eval_fold_metrics(
                composite, l1_dir, eval_bars["fwd_return"], test_start, test_end,
            )
            if metrics is None:
                break

            if mode == "full":
                fold_entry.update(metrics)
                print(f"  Fold {fold}: n={metrics['test_size']:,} IC={metrics['composite_ic']:+.4f} "
                      f"DirAcc={metrics['dir_accuracy']:.3f} Sharpe={metrics['sharpe_proxy']:+.2f} "
                      f"L1_active={metrics['l1_active_pct']:.1%}")
            else:
                fold_entry[f"ic_{mode}"] = metrics["composite_ic"]

        if "composite_ic" in fold_entry:
            fold_results.append(fold_entry)

    return fold_results


def purged_kfold_evaluate(
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    l1_weights: dict[str, float],
    l2_weights: dict[str, float],
    l3_weights: dict[str, float],
    l1_threshold: float = 0.5,
    n_splits: int = 5,
    embargo: int = 100,
    zscore_window: int = 200,
    ablation: bool = False,
) -> list[dict]:
    """Purged K-fold evaluation with embargo zones.

    Each fold's test set is separated from train by `embargo` bars on both
    sides, preventing temporal leakage. Unlike walk-forward, all data is
    used for testing (symmetric confidence intervals).
    """
    n = len(bars)
    fold_size = n // n_splits

    bars = bars.copy()
    bars["fwd_return"] = fwd_returns

    modes = ABLATION_MODES if ablation else ["full"]
    fold_results = []

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 50:
            continue

        # Use full data for z-score computation (features are self-normalizing)
        l1_z = _ic_weighted_composite(bars, L1_FEATURES, l1_weights, window=zscore_window)
        l2_z = _ic_weighted_composite(bars, L2_FEATURES, l2_weights, window=zscore_window)
        l3_z = _ic_weighted_composite(bars, L3_FEATURES, l3_weights, window=zscore_window)

        fold_entry = {"fold": fold}

        for mode in modes:
            composite, l1_dir = _compute_composite(l1_z, l2_z, l3_z, l1_threshold, mode=mode)
            metrics = _eval_fold_metrics(
                composite, l1_dir, bars["fwd_return"], test_start, test_end,
            )
            if metrics is None:
                break

            if mode == "full":
                fold_entry.update(metrics)
                print(f"  Fold {fold}: n={metrics['test_size']:,} IC={metrics['composite_ic']:+.4f} "
                      f"DirAcc={metrics['dir_accuracy']:.3f} Sharpe={metrics['sharpe_proxy']:+.2f} "
                      f"L1_active={metrics['l1_active_pct']:.1%}")
            else:
                fold_entry[f"ic_{mode}"] = metrics["composite_ic"]

        if "composite_ic" in fold_entry:
            fold_results.append(fold_entry)

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Signal Combiner")
    parser.add_argument("--symbol", default="BTC", help="Symbol to train on")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--horizon", type=int, default=HORIZON_BARS,
                        help="Forward return horizon in bars (default: 60 = 5h)")
    parser.add_argument("--n-splits", type=int, default=4, help="Walk-forward folds")
    parser.add_argument("--embargo", type=int, default=100, help="Embargo bars between train/test")
    parser.add_argument("--l1-threshold", type=float, default=0.5,
                        help="Z-score threshold for L1 directional activation")
    parser.add_argument("--zscore-window", type=int, default=200,
                        help="Rolling z-score window in bars")
    parser.add_argument("--output-dir", default="models/hierarchical_combiner",
                        help="Output directory for trained weights")
    parser.add_argument("--start-date", default=None,
                        help="Earliest date to load (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    parser.add_argument("--ablation", action="store_true",
                        help="Run layer ablation analysis (auto-enabled with --dry-run)")
    parser.add_argument("--validation-mode", choices=VALID_MODES, default="walk_forward",
                        help="Validation method: walk_forward (expanding) or purged_kfold")
    args = parser.parse_args()
    if args.dry_run:
        args.ablation = True

    print(f"=== Training Hierarchical Signal Combiner: {args.symbol} ===")
    print(f"Data: {args.data_dir}, horizon: {args.horizon} bars ({args.horizon * 5}min)")

    # Load and build dataset
    bars = load_bars(args.data_dir, args.symbol, start_date=args.start_date)
    fwd_returns = build_forward_returns(bars, horizon=args.horizon)

    valid_count = np.isfinite(fwd_returns).sum()
    print(f"Valid forward returns: {valid_count:,} / {len(bars):,}")

    if valid_count < 500:
        print(f"ERROR: Only {valid_count} valid samples, need at least 500")
        sys.exit(1)

    # Feature availability report
    print(f"\nFeature availability:")
    for group_name, features in [("L1", L1_FEATURES), ("L2", L2_FEATURES), ("L3", L3_FEATURES)]:
        for f in features:
            if f in bars.columns:
                nan_pct = bars[f].isna().mean() * 100
                std = bars[f].std()
                print(f"  [{group_name}] {f:40s} nan={nan_pct:4.1f}%  std={std:.6f}")
            else:
                print(f"  [{group_name}] {f:40s} MISSING")

    # Train IC weights per layer
    print(f"\n--- Training Layer 1 (slow directional) ---")
    l1_weights = train_layer_weights(
        bars, fwd_returns, L1_FEATURES, DEFAULT_L1_WEIGHTS,
        window=args.zscore_window, label="L1",
    )

    print(f"\n--- Training Layer 2 (fast entry) ---")
    l2_weights = train_layer_weights(
        bars, fwd_returns, L2_FEATURES, DEFAULT_L2_WEIGHTS,
        window=args.zscore_window, label="L2",
    )

    print(f"\n--- Training Layer 3 (vol sizing) ---")
    l3_weights = train_layer_weights(
        bars, fwd_returns, L3_FEATURES, DEFAULT_L3_WEIGHTS,
        window=args.zscore_window, label="L3",
    )

    # Evaluation
    eval_fn = purged_kfold_evaluate if args.validation_mode == "purged_kfold" else walk_forward_evaluate
    print(f"\n--- {args.validation_mode.replace('_', ' ').title()} Evaluation ({args.n_splits} folds) ---")
    fold_results = eval_fn(
        bars, fwd_returns,
        l1_weights, l2_weights, l3_weights,
        l1_threshold=args.l1_threshold,
        n_splits=args.n_splits,
        embargo=args.embargo,
        zscore_window=args.zscore_window,
        ablation=args.ablation,
    )

    if not fold_results:
        print("ERROR: No valid folds")
        sys.exit(1)

    # Summary
    avg_ic = np.mean([f["composite_ic"] for f in fold_results])
    avg_dir = np.mean([f["dir_accuracy"] for f in fold_results])
    avg_sharpe = np.mean([f["sharpe_proxy"] for f in fold_results])
    avg_l1_active = np.mean([f["l1_active_pct"] for f in fold_results])
    ic_std = np.std([f["composite_ic"] for f in fold_results])

    print(f"\n{'='*60}")
    print(f"OOS Average:")
    print(f"  Composite IC:      {avg_ic:+.4f} +/- {ic_std:.4f}")
    print(f"  Dir Accuracy:      {avg_dir:.4f}")
    print(f"  Sharpe (cost-adj): {avg_sharpe:+.2f}")
    print(f"  L1 Active Rate:    {avg_l1_active:.1%}")

    # IC significance check
    if len(fold_results) >= 2 and avg_ic < 2 * ic_std:
        print(f"\n  WARN: IC not significant (mean {avg_ic:.4f} < 2*std {2*ic_std:.4f})")

    # Monotonicity check
    ics = [f["composite_ic"] for f in fold_results]
    if len(ics) >= 3 and all(ics[i+1] > ics[i] for i in range(len(ics)-1)):
        print(f"\n  WARN: IC monotonically increasing across folds — possible data ordering bias")

    # Ablation summary
    if args.ablation and "ic_l1_only" in fold_results[0]:
        print(f"\n--- Ablation Analysis ---")
        print(f"  {'Mode':<12s} {'Mean IC':>10s} {'Delta':>10s} {'Rel %':>8s}")
        print(f"  {'-'*42}")
        for mode in ABLATION_MODES:
            key = f"ic_{mode}" if mode != "full" else "composite_ic"
            mode_ics = [f.get(key, f.get("composite_ic")) for f in fold_results]
            mode_avg = np.mean(mode_ics)
            delta = mode_avg - avg_ic
            rel = (delta / abs(avg_ic) * 100) if avg_ic != 0 else 0
            marker = "" if mode == "full" else f"  {'<-- L2/L3 not helping' if delta >= -0.005 else ''}"
            print(f"  {mode:<12s} {mode_avg:>+10.4f} {delta:>+10.4f} {rel:>+7.1f}%{marker}")

    # Decision gate
    if avg_ic < 0.02:
        print(f"\nWARN: Composite IC {avg_ic:.4f} < 0.02 — weak signal")
    if avg_dir < 0.51:
        print(f"\nWARN: Dir accuracy {avg_dir:.4f} < 0.51 — near random")
    if avg_sharpe < 0:
        print(f"\nWARN: Cost-adjusted Sharpe {avg_sharpe:.2f} < 0 — not profitable after costs")

    # Save weights
    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "weights.json"

        weights_data = {
            "l1_weights": l1_weights,
            "l2_weights": l2_weights,
            "l3_weights": l3_weights,
            "l1_threshold": args.l1_threshold,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": args.symbol,
            "horizon_bars": args.horizon,
            "zscore_window": args.zscore_window,
            "performance": {
                "avg_composite_ic": float(avg_ic),
                "avg_dir_accuracy": float(avg_dir),
                "avg_sharpe_proxy": float(avg_sharpe),
                "avg_l1_active_pct": float(avg_l1_active),
                "n_folds": len(fold_results),
                "fold_results": fold_results,
            },
        }

        with open(weights_path, "w") as f:
            json.dump(weights_data, f, indent=2)
        print(f"\nSaved weights to {weights_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

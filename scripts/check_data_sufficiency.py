#!/usr/bin/env python3
"""
Data sufficiency checks for ML model training.

Verifies that available data meets minimum thresholds before training.
Run after Wave 0 infrastructure, before any ML model training.

Usage:
    python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features
    python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features --json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Thresholds
MIN_BARS = 4000
MIN_LABEL_BALANCE = 0.40
MAX_LABEL_BALANCE = 0.60
MAX_NAN_RATE = 0.05
MIN_FOLD_SIZE = 500
N_FOLDS = 4

# Key features to check NaN rates for
KEY_FEATURES = [
    "ent_tick_1m",
    "trend_hurst_300",
    "toxic_vpin_50",
    "whale_net_flow_4h",
    "vol_returns_5m",
    "regime_accumulation_score",
    "imbalance_qty_l1",
]


def check_bar_count(n_bars: int) -> dict:
    """Check if bar count meets minimum threshold."""
    passed = n_bars >= MIN_BARS
    return {
        "check": "bar_count",
        "value": n_bars,
        "threshold": MIN_BARS,
        "passed": passed,
        "message": f"Bar count: {n_bars} ({'PASS' if passed else 'FAIL'}, need >= {MIN_BARS})",
    }


def check_label_balance(fwd_returns: pd.Series) -> dict:
    """Check if forward return label balance is within acceptable range."""
    pos_rate = float((fwd_returns > 0).mean())
    in_range = MIN_LABEL_BALANCE <= pos_rate <= MAX_LABEL_BALANCE
    return {
        "check": "label_balance",
        "value": round(pos_rate, 4),
        "range": [MIN_LABEL_BALANCE, MAX_LABEL_BALANCE],
        "passed": True,  # Warning only, doesn't block
        "warning": not in_range,
        "message": f"Label balance: {pos_rate:.1%} positive ({'OK' if in_range else 'WARNING'}, want {MIN_LABEL_BALANCE:.0%}-{MAX_LABEL_BALANCE:.0%})",
    }


def check_nan_rates(df: pd.DataFrame, features: list[str]) -> dict:
    """Check NaN rates for key features. Checks both raw and _mean suffixed."""
    results = {}
    any_fail = False

    for feat in features:
        # Check both raw name and _mean suffixed
        candidates = [feat, f"{feat}_mean"]
        found = False
        for col in candidates:
            if col in df.columns:
                nan_rate = float(df[col].isna().mean())
                passed = nan_rate < MAX_NAN_RATE
                if not passed:
                    any_fail = True
                results[col] = {
                    "nan_rate": round(nan_rate, 4),
                    "passed": passed,
                }
                found = True
                break
        if not found:
            results[feat] = {"nan_rate": None, "passed": False, "missing": True}
            any_fail = True

    return {
        "check": "nan_rates",
        "features": results,
        "passed": not any_fail,
        "message": f"NaN rates: {'PASS' if not any_fail else 'FAIL'} ({sum(1 for r in results.values() if not r['passed'])} features above {MAX_NAN_RATE:.0%})",
    }


def check_fold_sizes(n_bars: int, n_folds: int = N_FOLDS) -> dict:
    """Check if there are enough bars per walk-forward fold."""
    bars_per_fold = n_bars // n_folds if n_folds > 0 else 0
    passed = bars_per_fold >= MIN_FOLD_SIZE
    return {
        "check": "fold_sizes",
        "value": bars_per_fold,
        "n_folds": n_folds,
        "threshold": MIN_FOLD_SIZE,
        "passed": passed,
        "message": f"Fold size: {bars_per_fold} bars/fold ({'PASS' if passed else 'FAIL'}, need >= {MIN_FOLD_SIZE} with {n_folds} folds)",
    }


def run_all_checks(df: pd.DataFrame, midprice_col: str = "raw_midprice_mean") -> dict:
    """Run all 4 sufficiency checks on a bar-aggregated DataFrame."""
    n_bars = len(df)
    results = []

    # 1. Bar count
    results.append(check_bar_count(n_bars))

    # 2. Label balance (forward returns from midprice)
    if midprice_col in df.columns:
        fwd_ret = df[midprice_col].shift(-20) / df[midprice_col] - 1
        fwd_ret = fwd_ret.dropna()
        results.append(check_label_balance(fwd_ret))
    else:
        results.append({
            "check": "label_balance",
            "passed": True,
            "warning": True,
            "message": f"Label balance: SKIPPED (no '{midprice_col}' column)",
        })

    # 3. NaN rates
    results.append(check_nan_rates(df, KEY_FEATURES))

    # 4. Fold sizes
    results.append(check_fold_sizes(n_bars))

    # Overall
    hard_fails = [r for r in results if not r["passed"]]
    warnings = [r for r in results if r.get("warning", False)]
    sufficient = len(hard_fails) == 0

    return {
        "bar_count": n_bars,
        "checks": results,
        "sufficient": sufficient,
        "n_failures": len(hard_fails),
        "n_warnings": len(warnings),
    }


def main():
    parser = argparse.ArgumentParser(description="Check data sufficiency for ML training")
    parser.add_argument("--symbol", required=True, help="Symbol to check (e.g. BTC)")
    parser.add_argument("--data-dir", required=True, help="Path to data/features directory")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    args = parser.parse_args()

    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars

    print(f"Loading data for {args.symbol} from {args.data_dir}...")
    df = load_parquet(args.data_dir, symbols=[args.symbol])
    print(f"  Loaded {len(df)} ticks")

    print(f"Aggregating to 5-min bars...")
    bars = aggregate_bars(df, timeframe="5min")
    print(f"  Produced {len(bars)} bars")

    result = run_all_checks(bars)

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nDATA SUFFICIENCY CHECK — {args.symbol}")
        print("=" * 50)
        for check in result["checks"]:
            print(f"  {check['message']}")
        print("=" * 50)
        status = "DATA SUFFICIENT" if result["sufficient"] else "DATA INSUFFICIENT"
        print(f"  Result: {status}")
        if result["n_warnings"] > 0:
            print(f"  ({result['n_warnings']} warning(s))")

    sys.exit(0 if result["sufficient"] else 1)


if __name__ == "__main__":
    main()

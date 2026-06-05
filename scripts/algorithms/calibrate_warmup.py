#!/usr/bin/env python3
"""
Warmup calibration — empirically measure warmup stability for each algorithm.

For each algorithm:
1. Run on real data (or synthetic if no data available)
2. Compute rolling lag-1 autocorrelation over sliding 100-tick windows
3. Mark warmup as the tick where autocorrelation drops below 0.99 and stays
4. Compare empirical warmup to declared warmup; flag discrepancies > 2x

Usage:
    python scripts/algorithms/calibrate_warmup.py --data-dir data/features --symbol BTC
    python scripts/algorithms/calibrate_warmup.py --synthetic  # use synthetic data

Output: reports/warmup_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.autodiscover import discover_all  # noqa: E402
from algorithms.registry import list_algorithms, get_algorithm  # noqa: E402
from algorithms.runner import AlgorithmRunner  # noqa: E402
from algorithms.tests.conftest import make_synthetic_ticks  # noqa: E402

# Autocorrelation parameters
WINDOW = 100      # rolling window for lag-1 autocorrelation
THRESHOLD = 0.99  # autocorrelation must drop below this
SUSTAIN = 50      # must stay below threshold for this many ticks


def rolling_lag1_autocorrelation(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Compute rolling lag-1 autocorrelation over a sliding window.

    Returns array of same length as x, with NaN for the first `window` ticks.
    """
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(window, n):
        segment = x[i - window:i]
        if np.all(np.isfinite(segment)):
            mean = np.mean(segment)
            var = np.var(segment)
            if var > 1e-20:
                cov = np.mean((segment[1:] - mean) * (segment[:-1] - mean))
                result[i] = cov / var
            else:
                result[i] = 1.0  # constant → perfectly autocorrelated
        # else: leave as NaN

    return result


def find_empirical_warmup(autocorr: np.ndarray, threshold: float = THRESHOLD,
                          sustain: int = SUSTAIN) -> int:
    """Find the first tick where autocorrelation drops below threshold and stays.

    Returns the tick index, or len(autocorr) if it never stabilizes.
    """
    n = len(autocorr)
    for i in range(len(autocorr)):
        if np.isnan(autocorr[i]):
            continue
        if autocorr[i] < threshold:
            # Check if it stays below for `sustain` ticks
            end = min(i + sustain, n)
            window = autocorr[i:end]
            valid = window[np.isfinite(window)]
            if len(valid) > 0 and np.all(valid < threshold):
                return i
    return n  # never stabilized


def calibrate_algorithm(name: str, df: pd.DataFrame) -> dict:
    """Calibrate warmup for a single algorithm."""
    alg = get_algorithm(name)
    runner = AlgorithmRunner(alg)

    try:
        result = runner.run_on_dataframe(df)
    except Exception as e:
        return {
            "algorithm": name,
            "error": str(e),
            "declared_warmup": alg.warmup,
        }

    declared = alg.warmup
    features = {}

    for col in result.features_df.columns:
        values = result.features_df[col].values.astype(np.float64)

        # Skip if mostly NaN
        finite_rate = np.isfinite(values).mean()
        if finite_rate < 0.1:
            features[col] = {
                "empirical_warmup": None,
                "note": f"Only {finite_rate:.0%} finite values",
            }
            continue

        # Fill NaN with 0 for autocorrelation computation
        filled = np.where(np.isfinite(values), values, 0.0)
        autocorr = rolling_lag1_autocorrelation(filled, WINDOW)
        empirical = find_empirical_warmup(autocorr)

        ratio = empirical / declared if declared > 0 else float("inf")
        flag = "OK"
        if ratio > 2.0:
            flag = f"SLOW (empirical {ratio:.1f}x declared)"
        elif empirical < declared * 0.3 and declared > 10:
            flag = f"FAST (empirical {ratio:.1f}x declared, may be over-declared)"

        features[col] = {
            "empirical_warmup": empirical,
            "declared_warmup": declared,
            "ratio": round(ratio, 2),
            "flag": flag,
        }

    return {
        "algorithm": name,
        "declared_warmup": declared,
        "features": features,
        "n_ticks": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Warmup calibration study")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real data")
    parser.add_argument("--n-ticks", type=int, default=10000,
                        help="Number of ticks for synthetic data")
    parser.add_argument("--output", default=str(ROOT / "reports" / "warmup_calibration.json"))
    parser.add_argument("--algorithms", nargs="*", help="Specific algorithms to test")
    args = parser.parse_args()

    discover_all()

    # Load or generate data
    if args.synthetic:
        print(f"Using synthetic data ({args.n_ticks} ticks)")
        # Collect all required columns across algorithms
        algo_names = args.algorithms or list_algorithms()
        all_cols = set()
        for name in algo_names:
            alg = get_algorithm(name)
            if not alg.bar_level:
                all_cols.update(alg.required_columns())
        df = make_synthetic_ticks(args.n_ticks, list(all_cols))
    else:
        print(f"Loading {args.symbol} data from {args.data_dir}...")
        from cluster_pipeline.loader import load_parquet
        df = load_parquet(args.data_dir, symbols=[args.symbol])
        print(f"  Loaded {len(df)} ticks")

    # Calibrate each algorithm
    algo_names = args.algorithms or list_algorithms()
    results = []
    flagged = []

    for name in sorted(algo_names):
        alg = get_algorithm(name)
        if alg.bar_level:
            print(f"  {name}: SKIP (bar-level)")
            continue

        # Check required columns
        missing = [c for c in alg.required_columns() if c not in df.columns]
        if missing:
            print(f"  {name}: SKIP (missing columns: {missing[:3]}...)")
            continue

        print(f"  {name}...", end="", flush=True)
        cal = calibrate_algorithm(name, df)
        results.append(cal)

        # Summarize
        if "error" in cal:
            print(f" ERROR: {cal['error']}")
            continue

        n_flags = sum(1 for f in cal["features"].values()
                      if isinstance(f, dict) and f.get("flag", "OK") != "OK")
        if n_flags > 0:
            print(f" {n_flags} flagged")
            for feat_name, feat_info in cal["features"].items():
                if isinstance(feat_info, dict) and feat_info.get("flag", "OK") != "OK":
                    flagged.append(f"  {name}.{feat_name}: {feat_info['flag']}")
        else:
            print(" OK")

    # Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Algorithms tested: {len(results)}")
    print(f"Flagged features:  {len(flagged)}")
    if flagged:
        print("\nFlagged:")
        for line in flagged:
            print(line)


if __name__ == "__main__":
    main()

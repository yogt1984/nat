#!/usr/bin/env python3
"""Algorithm regression test: snapshot → run → compare against baseline.

Usage:
    python scripts/test_regression.py snapshot          # capture + run + save baseline
    python scripts/test_regression.py check             # run + compare against baseline
    python scripts/test_regression.py snapshot --help    # see options

Not CI-friendly (requires live data). Provides a repeatable regression gate
before deployment — detects if algorithm IC drops or signal counts change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.autodiscover import discover_all  # noqa: E402
from algorithms.registry import get_algorithm  # noqa: E402
from algorithms.runner import AlgorithmRunner  # noqa: E402
from algorithms.evaluate import AlgorithmEvaluator  # noqa: E402
from cluster_pipeline.loader import load_parquet  # noqa: E402

SNAPSHOT_DIR = ROOT / "data" / "test_snapshots"
SNAPSHOT_DATA = SNAPSHOT_DIR / "latest.parquet"
BASELINE_FILE = SNAPSHOT_DIR / "latest_results.json"

WINNERS = ["jump_detector", "optimal_entry", "funding_reversion",
           "surprise_signal", "weighted_ofi"]

# Regression thresholds
IC_DROP_THRESHOLD = 0.01     # flag if IC drops by more than this
SIGNAL_CHANGE_THRESHOLD = 0.20  # flag if signal count changes by > 20%


def snapshot(data_dir: str, symbol: str, hours: int) -> None:
    """Capture a data snapshot and save baseline results."""
    discover_all()

    print(f"Loading {symbol} data from {data_dir}...")
    df = load_parquet(data_dir, symbols=[symbol], max_rows=hours * 36000)
    print(f"  Loaded {len(df)} ticks, {len(df.columns)} columns")

    if len(df) < 1000:
        print(f"  ERROR: Only {len(df)} ticks — need at least 1000")
        sys.exit(1)

    # Save snapshot
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SNAPSHOT_DATA, index=False)
    print(f"  Saved snapshot: {SNAPSHOT_DATA} ({SNAPSHOT_DATA.stat().st_size / 1024:.0f} KB)")

    # Run all winners and save results
    results = _run_all(df)
    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved baseline: {BASELINE_FILE}")

    _print_summary(results)


def check() -> None:
    """Run algorithms on snapshot and compare against baseline."""
    if not SNAPSHOT_DATA.exists():
        print(f"ERROR: No snapshot found at {SNAPSHOT_DATA}")
        print("  Run: python scripts/test_regression.py snapshot")
        sys.exit(1)

    if not BASELINE_FILE.exists():
        print(f"ERROR: No baseline found at {BASELINE_FILE}")
        print("  Run: python scripts/test_regression.py snapshot")
        sys.exit(1)

    discover_all()

    print(f"Loading snapshot: {SNAPSHOT_DATA}")
    df = pd.read_parquet(SNAPSHOT_DATA)
    print(f"  {len(df)} ticks")

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    # Run current
    current = _run_all(df)

    # Compare
    regressions = _compare(baseline, current)

    _print_summary(current)

    if regressions:
        print(f"\n{'='*60}")
        print(f"REGRESSION DETECTED ({len(regressions)} issue(s)):")
        for r in regressions:
            print(f"  - {r}")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print("\nNo regressions detected.")


def _run_all(df: pd.DataFrame) -> dict:
    """Run all winner algorithms and collect results."""
    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "algorithms": {}}

    for name in WINNERS:
        alg = get_algorithm(name)
        runner = AlgorithmRunner(alg)
        result = runner.run_on_dataframe(df)
        evaluator = AlgorithmEvaluator(result)
        report = evaluator.full_report()

        # Extract key metrics
        ic_summary = {}
        for feat_name, ic_dict in report.get("ic", {}).get("features", {}).items():
            for h_key, ic_val in ic_dict.items():
                if ic_val is not None and not (isinstance(ic_val, float) and np.isnan(ic_val)):
                    ic_summary[f"{feat_name}_{h_key}"] = round(float(ic_val), 6)

        # Count non-NaN signals post-warmup
        post_warmup = result.features_df.iloc[alg.warmup:]
        signal_counts = {}
        for col in post_warmup.columns:
            non_nan = int(post_warmup[col].notna().sum())
            non_zero = int((post_warmup[col].abs() > 1e-10).sum())
            signal_counts[col] = {"non_nan": non_nan, "non_zero": non_zero}

        results["algorithms"][name] = {
            "ic": ic_summary,
            "signal_counts": signal_counts,
            "n_ticks": result.n_ticks,
            "elapsed_s": result.elapsed_s,
        }

    return results


def _compare(baseline: dict, current: dict) -> list[str]:
    """Compare current results against baseline. Returns list of regression messages."""
    regressions = []

    for name in WINNERS:
        b_alg = baseline.get("algorithms", {}).get(name, {})
        c_alg = current.get("algorithms", {}).get(name, {})

        if not b_alg or not c_alg:
            continue

        # IC comparison
        b_ic = b_alg.get("ic", {})
        c_ic = c_alg.get("ic", {})
        for key in b_ic:
            if key not in c_ic:
                continue
            b_val = float(b_ic[key])
            c_val = float(c_ic[key])
            drop = b_val - c_val
            if abs(b_val) > 0.005 and drop > IC_DROP_THRESHOLD:
                regressions.append(
                    f"{name} IC regression: {key} dropped {b_val:.4f} → {c_val:.4f} "
                    f"(delta={drop:+.4f}, threshold={IC_DROP_THRESHOLD})"
                )

        # Signal count comparison
        b_counts = b_alg.get("signal_counts", {})
        c_counts = c_alg.get("signal_counts", {})
        for col in b_counts:
            if col not in c_counts:
                continue
            b_nz = b_counts[col].get("non_zero", 0)
            c_nz = c_counts[col].get("non_zero", 0)
            if b_nz > 10:
                change = abs(c_nz - b_nz) / b_nz
                if change > SIGNAL_CHANGE_THRESHOLD:
                    regressions.append(
                        f"{name} signal count change: {col} "
                        f"{b_nz} → {c_nz} ({change:.0%} change, "
                        f"threshold={SIGNAL_CHANGE_THRESHOLD:.0%})"
                    )

    return regressions


def _print_summary(results: dict) -> None:
    """Print a summary of algorithm results."""
    print(f"\n{'Algorithm':<22} {'Elapsed':>8} {'IC keys':>8}")
    print("-" * 42)
    for name in WINNERS:
        alg = results.get("algorithms", {}).get(name, {})
        elapsed = alg.get("elapsed_s", 0)
        n_ic = len(alg.get("ic", {}))
        print(f"{name:<22} {elapsed:>7.2f}s {n_ic:>8}")


def main():
    parser = argparse.ArgumentParser(description="Algorithm regression test")
    sub = parser.add_subparsers(dest="command")

    snap = sub.add_parser("snapshot", help="Capture snapshot + save baseline")
    snap.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
    snap.add_argument("--symbol", default="BTC")
    snap.add_argument("--hours", type=int, default=1)

    sub.add_parser("check", help="Run and compare against baseline")

    args = parser.parse_args()

    if args.command == "snapshot":
        snapshot(args.data_dir, args.symbol, args.hours)
    elif args.command == "check":
        check()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

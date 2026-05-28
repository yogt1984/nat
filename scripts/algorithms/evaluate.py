#!/usr/bin/env python3
"""
Algorithm evaluation harness.

Reuses existing IC, drift, and fill-sim infrastructure from scripts/kalman/.
Provides a unified evaluation interface for any MicrostructureAlgorithm.

Usage:
    python scripts/algorithms/evaluate.py --algorithm regime_gated --symbol BTC
    python scripts/algorithms/evaluate.py --all --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


from kalman.analysis import compute_ic, compute_forward_returns
from kalman.fill_sim import MakerFillSimulator

try:
    from .runner import AlgorithmResult
except ImportError:
    from algorithms.runner import AlgorithmResult


class AlgorithmEvaluator:
    """Evaluate an algorithm's output features."""

    def __init__(self, result: AlgorithmResult):
        self.result = result
        self._midprices = result.base_df["raw_midprice"].values.astype(np.float64)

    def ic_analysis(self, horizons: list[int] = None) -> dict:
        """IC of each alg_feature vs forward returns."""
        if horizons is None:
            horizons = [1, 5, 10, 50, 100]

        fwd_rets = {h: compute_forward_returns(self._midprices, h) for h in horizons}
        results = {}

        for feat_name in self.result.features_df.columns:
            signal = self.result.features_df[feat_name].values.astype(np.float64)
            feat_ics = {}
            for h in horizons:
                feat_ics[f"{h}t"] = compute_ic(signal, fwd_rets[h])
            results[feat_name] = feat_ics

        return {"horizons": horizons, "features": results}

    def ic_by_regime(
        self,
        regime_col: str = "ent_book_shape",
        percentile: float = 30.0,
        horizons: list[int] = None,
    ) -> dict:
        """Regime-stratified IC."""
        if horizons is None:
            horizons = [1, 5, 10, 50, 100]

        if regime_col not in self.result.base_df.columns:
            return {"error": f"{regime_col} not in data"}

        regime_vals = self.result.base_df[regime_col].values.astype(np.float64)
        thresh = np.nanpercentile(regime_vals, percentile)
        mask = regime_vals < thresh

        fwd_rets = {h: compute_forward_returns(self._midprices, h) for h in horizons}
        results = {}

        for feat_name in self.result.features_df.columns:
            signal = self.result.features_df[feat_name].values.astype(np.float64)
            feat_ics = {}
            for h in horizons:
                sig_r = np.where(mask, signal, np.nan)
                fwd_r = np.where(mask, fwd_rets[h], np.nan)
                feat_ics[f"{h}t"] = compute_ic(sig_r, fwd_r)
            results[feat_name] = feat_ics

        return {
            "regime_col": regime_col,
            "percentile": percentile,
            "threshold": float(thresh),
            "horizons": horizons,
            "features": results,
        }

    def drift_analysis(
        self,
        signal_col: str,
        entry_threshold: float = 0.3,
        regime_percentile: float = 30.0,
    ) -> dict:
        """Post-fill drift via MakerFillSimulator."""
        if signal_col not in self.result.features_df.columns:
            return {"error": f"{signal_col} not in features_df"}

        base = self.result.base_df
        if "raw_spread" not in base.columns or "ent_book_shape" not in base.columns:
            return {"error": "raw_spread or ent_book_shape not in base data"}

        signal = self.result.features_df[signal_col].values.astype(np.float64)
        midprices = self._midprices
        spreads = base["raw_spread"].values.astype(np.float64)
        regime = base["ent_book_shape"].values.astype(np.float64)

        sim = MakerFillSimulator(
            entry_threshold=entry_threshold,
            regime_percentile=regime_percentile,
        )
        fills = sim.simulate(midprices, spreads, signal, regime)

        if len(fills) < 10:
            return {"n_fills": len(fills), "error": "too_few_fills"}

        horizons = [1, 5, 10, 50, 100]
        drifts = sim.compute_post_fill_drift(fills, midprices, horizons)

        drift_stats = []
        for h in horizons:
            d = drifts[h]
            valid = d[np.isfinite(d)]
            if len(valid) < 5:
                continue
            drift_stats.append({
                "horizon_ticks": h,
                "mean_bps": float(np.mean(valid)),
                "median_bps": float(np.median(valid)),
                "std_bps": float(np.std(valid)),
                "pct_positive": float(np.mean(valid > 0) * 100),
                "n_fills": len(valid),
            })

        return {
            "signal_col": signal_col,
            "entry_threshold": entry_threshold,
            "n_fills": len(fills),
            "drift_by_horizon": drift_stats,
        }

    def full_report(
        self,
        signal_col: str = None,
        horizons: list[int] = None,
    ) -> dict:
        """Run IC + regime IC + drift analysis. Returns consolidated report."""
        if signal_col is None:
            # Use first feature as signal
            signal_col = self.result.features_df.columns[0]

        report = {
            "algorithm": self.result.algorithm_name,
            "n_ticks": self.result.n_ticks,
            "warmup_ticks": self.result.warmup_ticks,
            "elapsed_s": self.result.elapsed_s,
            "ic": self.ic_analysis(horizons),
            "ic_regime": self.ic_by_regime(horizons=horizons),
        }

        # Only run drift if we have spread data
        if "raw_spread" in self.result.base_df.columns:
            report["drift"] = self.drift_analysis(signal_col)

        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate algorithms")
    parser.add_argument("--algorithm", default=None, help="Algorithm name")
    parser.add_argument("--all", action="store_true", help="Evaluate all registered")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--max-memory-mb", type=float, default=2000.0)
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    # Auto-discover all algorithms
    from algorithms.autodiscover import discover_all
    discover_all()
    from algorithms.registry import list_algorithms, get_algorithm
    from algorithms.runner import AlgorithmRunner

    if args.all:
        alg_names = list_algorithms()
    elif args.algorithm:
        alg_names = [args.algorithm]
    else:
        alg_names = list_algorithms()

    print(f"Evaluating {len(alg_names)} algorithms on {args.symbol}")
    results = []

    for alg_name in alg_names:
        print(f"\n{'='*50}")
        print(f"Algorithm: {alg_name}")
        print(f"{'='*50}")

        alg = get_algorithm(alg_name)
        runner = AlgorithmRunner(alg)

        try:
            result = runner.run_on_parquet(
                args.data_dir, args.symbol,
                max_memory_mb=args.max_memory_mb,
                columns=["ent_book_shape", "raw_spread"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"algorithm": alg_name, "error": str(e)})
            continue

        print(f"  Processed {result.n_ticks:,} ticks in {result.elapsed_s:.1f}s")

        evaluator = AlgorithmEvaluator(result)
        report = evaluator.full_report()

        # Print IC summary
        print(f"\n  IC (full sample):")
        for feat, ics in report["ic"]["features"].items():
            ic_10 = ics.get("10t", 0)
            ic_50 = ics.get("50t", 0)
            print(f"    {feat}: IC@10t={ic_10:+.4f}, IC@50t={ic_50:+.4f}")

        # Print regime IC
        if "features" in report.get("ic_regime", {}):
            print(f"\n  IC (regime-gated):")
            for feat, ics in report["ic_regime"]["features"].items():
                ic_10 = ics.get("10t", 0)
                ic_50 = ics.get("50t", 0)
                print(f"    {feat}: IC@10t={ic_10:+.4f}, IC@50t={ic_50:+.4f}")

        results.append(report)

    # Save report
    report_path = args.json_report or f"reports/algorithms/{args.symbol}_eval.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()

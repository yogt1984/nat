#!/usr/bin/env python3
"""
Phase 2: Post-Fill Drift Analysis

Runs maker fill simulation and analyzes post-fill price drift
stratified by signal strength, regime, and symbol.

Critical go/no-go metric:
- Unconditional drift should be NEGATIVE (confirms adverse selection)
- Conditional drift (strong signal + regime) should be less negative or positive

Usage:
    python scripts/kalman/drift_analysis.py --symbol BTC --data-dir data/features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


from cluster_pipeline.loader import load_parquet
from kalman.fill_sim import MakerFillSimulator


def run_drift_analysis(
    data_dir: str,
    symbol: str,
    entry_thresholds: list[float] = None,
    regime_percentile: float = 30.0,
    latency_ticks: int = 2,
    dt: float = 0.1,
    max_memory_mb: float = None,
) -> dict:
    """Run Phase 2 drift analysis for one symbol."""

    if entry_thresholds is None:
        entry_thresholds = [0.1, 0.2, 0.3, 0.5]

    print(f"\n{'='*60}")
    print(f"Phase 2: Drift Analysis — {symbol}")
    print(f"{'='*60}")

    # Load data
    columns = ["timestamp_ns", "symbol", "raw_midprice", "raw_spread",
               "imbalance_qty_l1", "ent_book_shape"]
    df = load_parquet(
        data_dir,
        symbols=[symbol],
        columns=columns,
        max_memory_mb=max_memory_mb,
    )
    print(f"  Loaded {len(df):,} rows")

    if len(df) < 1000:
        return {"symbol": symbol, "error": "too_few_rows"}

    midprices = df["raw_midprice"].values.astype(np.float64)
    spreads = df["raw_spread"].values.astype(np.float64)
    signal = df["imbalance_qty_l1"].values.astype(np.float64)
    regime = df["ent_book_shape"].values.astype(np.float64)

    print(f"  Spread stats: median={np.nanmedian(spreads):.4f}, "
          f"mean={np.nanmean(spreads):.4f}")
    print(f"  Signal stats: std={np.nanstd(signal):.4f}, "
          f"mean={np.nanmean(signal):.4f}")

    horizons = [1, 5, 10, 50, 100]
    threshold_results = []

    for thresh in entry_thresholds:
        print(f"\n  --- Entry threshold: {thresh:.2f} ---")

        sim = MakerFillSimulator(
            entry_threshold=thresh,
            regime_percentile=regime_percentile,
            latency_ticks=latency_ticks,
        )
        fills = sim.simulate(midprices, spreads, signal, regime)
        n_fills = len(fills)

        if n_fills < 10:
            print(f"    Only {n_fills} fills — skipping")
            threshold_results.append({
                "threshold": thresh,
                "n_fills": n_fills,
                "error": "too_few_fills",
            })
            continue

        n_buy = sum(1 for f in fills if f.side == "buy")
        n_sell = n_fills - n_buy
        fill_rate_pct = n_fills / (len(df) / sim.min_ticks_between_signals) * 100

        print(f"    Fills: {n_fills} ({n_buy} buy, {n_sell} sell), "
              f"rate≈{fill_rate_pct:.1f}%")

        # Fill latency stats
        latencies = [f.fill_tick - f.signal_tick for f in fills]
        print(f"    Fill latency: median={np.median(latencies):.0f}t, "
              f"mean={np.mean(latencies):.1f}t")

        # Post-fill drift
        drifts = sim.compute_post_fill_drift(fills, midprices, horizons)

        print(f"    Post-fill drift (bps, positive=favorable):")
        print(f"    {'Horizon':>10s}  {'Mean':>8s}  {'Median':>8s}  "
              f"{'Std':>8s}  {'%>0':>6s}  {'N':>6s}")
        print(f"    {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

        drift_stats = []
        for h in horizons:
            d = drifts[h]
            valid = d[np.isfinite(d)]
            if len(valid) < 5:
                continue
            mean_d = float(np.mean(valid))
            med_d = float(np.median(valid))
            std_d = float(np.std(valid))
            pct_pos = float(np.mean(valid > 0) * 100)

            print(f"    {h:>8d}t  {mean_d:>+8.2f}  {med_d:>+8.2f}  "
                  f"{std_d:>8.2f}  {pct_pos:>5.1f}%  {len(valid):>6d}")

            drift_stats.append({
                "horizon_ticks": h,
                "horizon_s": h * dt,
                "mean_bps": mean_d,
                "median_bps": med_d,
                "std_bps": std_d,
                "pct_positive": pct_pos,
                "n_fills": len(valid),
            })

        # Stratify by signal strength quintiles
        strengths = np.array([abs(f.signal_strength) for f in fills])
        quintile_edges = np.percentile(strengths, [20, 40, 60, 80])

        print(f"\n    Drift @ 50t by signal strength quintile:")
        print(f"    {'Quintile':>10s}  {'Range':>16s}  {'Mean':>8s}  "
              f"{'Median':>8s}  {'%>0':>6s}  {'N':>6s}")
        print(f"    {'-'*10}  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

        strength_strata = []
        bins = [0] + quintile_edges.tolist() + [np.inf]
        for q in range(5):
            mask = (strengths >= bins[q]) & (strengths < bins[q + 1])
            d50 = drifts[50][mask]
            valid = d50[np.isfinite(d50)]
            if len(valid) < 3:
                continue
            mean_d = float(np.mean(valid))
            med_d = float(np.median(valid))
            pct_pos = float(np.mean(valid > 0) * 100)
            label = f"Q{q+1}"
            rng = f"[{bins[q]:.2f},{bins[q+1]:.2f})"

            print(f"    {label:>10s}  {rng:>16s}  {mean_d:>+8.2f}  "
                  f"{med_d:>+8.2f}  {pct_pos:>5.1f}%  {len(valid):>6d}")

            strength_strata.append({
                "quintile": q + 1,
                "range_low": float(bins[q]),
                "range_high": float(bins[q + 1]),
                "mean_bps": mean_d,
                "median_bps": med_d,
                "pct_positive": pct_pos,
                "n": len(valid),
            })

        threshold_results.append({
            "threshold": thresh,
            "n_fills": n_fills,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "fill_rate_pct": fill_rate_pct,
            "median_latency_ticks": float(np.median(latencies)),
            "drift_by_horizon": drift_stats,
            "drift_by_strength": strength_strata,
        })

    # Go/no-go summary
    print(f"\n  {'='*50}")
    print(f"  GO/NO-GO ASSESSMENT")
    print(f"  {'='*50}")

    viable = False
    for tr in threshold_results:
        if "error" in tr:
            continue
        for ds in tr.get("drift_by_horizon", []):
            if ds["horizon_ticks"] == 50 and ds["mean_bps"] > 0:
                print(f"  VIABLE: thresh={tr['threshold']}, "
                      f"drift@50t={ds['mean_bps']:+.2f}bps, "
                      f"n={ds['n_fills']}")
                viable = True
        # Check if strongest quintile has positive drift
        strata = tr.get("drift_by_strength", [])
        if strata and strata[-1].get("mean_bps", 0) > 0:
            print(f"  CONDITIONAL VIABLE: thresh={tr['threshold']}, "
                  f"Q5 drift={strata[-1]['mean_bps']:+.2f}bps")
            viable = True

    if not viable:
        print("  NO VIABLE CONFIGURATION FOUND")
        print("  All post-fill drifts negative — adverse selection dominates")

    return {
        "symbol": symbol,
        "n_rows": len(df),
        "regime_percentile": regime_percentile,
        "latency_ticks": latency_ticks,
        "threshold_results": threshold_results,
        "viable": viable,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Drift Analysis")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--max-memory-mb", type=float, default=2000.0)
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--regime-percentile", type=float, default=30.0)
    parser.add_argument("--latency-ticks", type=int, default=2)
    args = parser.parse_args()

    symbols = ["BTC", "ETH", "SOL"] if args.all_symbols else [args.symbol]
    results = []

    t0 = time.time()
    for sym in symbols:
        result = run_drift_analysis(
            data_dir=args.data_dir,
            symbol=sym,
            regime_percentile=args.regime_percentile,
            latency_ticks=args.latency_ticks,
            max_memory_mb=args.max_memory_mb,
        )
        results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    report_path = args.json_report or "reports/kalman/phase2_drift.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"results": results, "elapsed_s": round(elapsed, 1)}, f, indent=2)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()

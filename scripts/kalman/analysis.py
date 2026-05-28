#!/usr/bin/env python3
"""
Phase 1: Kalman Filter Offline IC Analysis

Compares information coefficient (IC) of raw vs Kalman-filtered imbalance
signal at multiple forward horizons, stratified by regime gate.

Usage:
    python scripts/kalman/analysis.py --symbol BTC --data-dir data/features
    python scripts/kalman/analysis.py --symbol BTC --data-dir data/features --max-memory-mb 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


from cluster_pipeline.loader import load_parquet
from kalman.ou_filter import OUKalmanFilter, estimate_ou_params, auto_tune_filter


def compute_ic(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Pearson correlation between signal and forward return."""
    mask = np.isfinite(signal) & np.isfinite(forward_ret)
    if mask.sum() < 100:
        return 0.0
    return float(np.corrcoef(signal[mask], forward_ret[mask])[0, 1])


def compute_forward_returns(midprices: np.ndarray, horizon: int) -> np.ndarray:
    """Compute forward return at given horizon (in ticks)."""
    fwd = np.full(len(midprices), np.nan)
    if horizon < len(midprices):
        fwd[:-horizon] = (midprices[horizon:] - midprices[:-horizon]) / midprices[:-horizon]
    return fwd


def run_analysis(
    data_dir: str,
    symbol: str,
    horizons: list[int] = None,
    regime_feature: str = "ent_book_shape",
    regime_percentile: float = 30.0,
    dt: float = 0.1,
    max_memory_mb: Optional[float] = None,
) -> dict:
    """Run full Phase 1 analysis for one symbol."""

    if horizons is None:
        horizons = [1, 5, 10, 50, 100]

    print(f"\n{'='*60}")
    print(f"Phase 1: Kalman IC Analysis — {symbol}")
    print(f"{'='*60}")

    # Load data
    columns = ["timestamp_ns", "symbol", "raw_midprice", "imbalance_qty_l1", regime_feature]
    df = load_parquet(
        data_dir,
        symbols=[symbol],
        columns=columns,
        max_memory_mb=max_memory_mb,
    )
    print(f"  Loaded {len(df):,} rows")

    if len(df) < 1000:
        print("  ERROR: too few rows")
        return {"symbol": symbol, "error": "too_few_rows", "n_rows": len(df)}

    # Extract arrays
    midprices = df["raw_midprice"].values.astype(np.float64)
    raw_imb = df["imbalance_qty_l1"].values.astype(np.float64)

    has_regime = regime_feature in df.columns
    if has_regime:
        regime_vals = df[regime_feature].values.astype(np.float64)
        regime_thresh = np.nanpercentile(regime_vals, regime_percentile)
        regime_mask = regime_vals < regime_thresh
        regime_coverage = float(np.mean(regime_mask[np.isfinite(regime_vals)]))
        print(f"  Regime gate: {regime_feature} < P{regime_percentile:.0f} "
              f"(thresh={regime_thresh:.4f}, coverage={regime_coverage:.1%})")
    else:
        regime_mask = np.ones(len(df), dtype=bool)
        regime_coverage = 1.0
        print(f"  WARNING: {regime_feature} not in data, no regime gate")

    # Estimate OU parameters
    print("\n  OU Parameter Estimation:")
    params = estimate_ou_params(raw_imb, dt=dt)
    print(f"    theta = {params.theta:.4f} (half-life = {params.half_life:.1f}s)")
    print(f"    mu    = {params.mu:.6f}")
    print(f"    sigma = {params.sigma:.6f}")

    # Precompute forward returns
    fwd_rets = {h: compute_forward_returns(midprices, h) for h in horizons}

    # Regime-masked arrays for raw signal
    raw_regime = np.where(regime_mask, raw_imb, np.nan)
    fwd_regime = {h: np.where(regime_mask, fwd_rets[h], np.nan) for h in horizons}

    # Sweep observation noise multipliers to find optimal smoothing
    # Higher R_mult → more smoothing → extracts slower component
    r_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    print("\n  Noise ratio sweep (finding optimal smoothing):")
    print(f"  {'R_mult':>8s}  {'ACF(1)':>8s}  ", end="")
    for h in horizons:
        print(f"  {'IC@'+str(h)+'t':>8s}", end="")
    print(f"  {'IC@'+str(horizons[-2])+'t|R':>10s}")  # regime at key horizon
    print(f"  {'-'*8}  {'-'*8}  ", end="")
    for _ in horizons:
        print(f"  {'-'*8}", end="")
    print(f"  {'-'*10}")

    sweep_results = []
    for r_mult in r_multipliers:
        kf = auto_tune_filter(raw_imb, dt=dt)
        # Scale observation noise to increase smoothing
        kf.sigma_obs *= r_mult
        kf.R = kf.sigma_obs**2
        kf.reset()

        filt, uncerts, innovs = kf.filter_series_full(raw_imb)
        acf1 = float(np.corrcoef(innovs[1:], innovs[:-1])[0, 1])

        ics = {}
        for h in horizons:
            ics[h] = compute_ic(filt, fwd_rets[h])

        # Regime IC at key horizon (50t)
        filt_regime = np.where(regime_mask, filt, np.nan)
        ic_regime_key = compute_ic(filt_regime, fwd_regime[horizons[-2]])

        print(f"  {r_mult:>8.1f}  {acf1:>+8.4f}  ", end="")
        for h in horizons:
            print(f"  {ics[h]:>+8.4f}", end="")
        print(f"  {ic_regime_key:>+10.4f}")

        sweep_results.append({
            "r_mult": r_mult,
            "acf1": acf1,
            "ics": ics,
            "ic_regime_key": ic_regime_key,
            "uncertainty": float(uncerts[-1]),
        })

    # Select best R multiplier: maximize regime IC at 50-tick horizon
    best_sweep = max(sweep_results, key=lambda r: r["ic_regime_key"])
    best_r_mult = best_sweep["r_mult"]
    print(f"\n  Best R multiplier: {best_r_mult}x "
          f"(regime IC@{horizons[-2]}t = {best_sweep['ic_regime_key']:+.4f})")

    # Run final filter with best params
    print(f"\n  Running Kalman filter (R_mult={best_r_mult}x)...")
    kf = auto_tune_filter(raw_imb, dt=dt)
    kf.sigma_obs *= best_r_mult
    kf.R = kf.sigma_obs**2
    kf.reset()
    filtered, uncertainties, innovations = kf.filter_series_full(raw_imb)
    print(f"    Steady-state uncertainty: {uncertainties[-1]:.6f}")

    # Compute IC comparison table
    print("\n  IC Comparison (raw vs filtered, R_mult={:.0f}x):".format(best_r_mult))
    print(f"  {'Horizon':>10s}  {'Raw IC':>10s}  {'Filt IC':>10s}  {'Gain':>10s}"
          f"  {'Raw|Regime':>12s}  {'Filt|Regime':>12s}  {'Gain|Regime':>12s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

    ic_results = []
    for h in horizons:
        fwd_ret = fwd_rets[h]

        # Full sample
        ic_raw = compute_ic(raw_imb, fwd_ret)
        ic_filt = compute_ic(filtered, fwd_ret)
        gain = ic_filt - ic_raw

        # Regime-gated
        filt_reg = np.where(regime_mask, filtered, np.nan)
        ic_raw_regime = compute_ic(raw_regime, fwd_regime[h])
        ic_filt_regime = compute_ic(filt_reg, fwd_regime[h])
        gain_regime = ic_filt_regime - ic_raw_regime

        print(f"  {h:>8d}t  {ic_raw:>+10.4f}  {ic_filt:>+10.4f}  {gain:>+10.4f}"
              f"  {ic_raw_regime:>+12.4f}  {ic_filt_regime:>+12.4f}  {gain_regime:>+12.4f}")

        ic_results.append({
            "horizon_ticks": h,
            "horizon_s": h * dt,
            "ic_raw": ic_raw,
            "ic_filtered": ic_filt,
            "ic_gain": gain,
            "ic_raw_regime": ic_raw_regime,
            "ic_filtered_regime": ic_filt_regime,
            "ic_gain_regime": gain_regime,
        })

    # Innovation statistics
    innov_acf1 = float(np.corrcoef(innovations[1:], innovations[:-1])[0, 1])
    innov_std = float(np.std(innovations))
    print(f"\n  Innovation diagnostics:")
    print(f"    Std: {innov_std:.6f}")
    print(f"    ACF(1): {innov_acf1:.4f} (should be near 0 if well-tuned)")

    # Summary
    best = max(ic_results, key=lambda r: r["ic_filtered_regime"])
    print(f"\n  Best filtered IC: {best['ic_filtered_regime']:+.4f} "
          f"at {best['horizon_ticks']}t ({best['horizon_s']:.1f}s) in regime")

    return {
        "symbol": symbol,
        "n_rows": len(df),
        "ou_params": {
            "theta": params.theta,
            "mu": params.mu,
            "sigma": params.sigma,
            "half_life_s": params.half_life,
        },
        "regime": {
            "feature": regime_feature,
            "percentile": regime_percentile,
            "threshold": float(regime_thresh) if has_regime else None,
            "coverage": regime_coverage,
        },
        "ic_results": ic_results,
        "innovation_acf1": innov_acf1,
        "innovation_std": innov_std,
        "filter_params": {
            "theta": kf.theta,
            "sigma_process": kf.sigma_process,
            "sigma_obs": kf.sigma_obs,
            "dt": kf.dt,
            "mu": kf.mu,
            "r_multiplier": best_r_mult,
        },
        "sweep_results": [
            {"r_mult": s["r_mult"], "acf1": s["acf1"], "ic_regime_key": s["ic_regime_key"]}
            for s in sweep_results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Kalman Filter IC Analysis")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Run for BTC, ETH, SOL")
    parser.add_argument("--max-memory-mb", type=float, default=2000.0)
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--regime-feature", default="ent_book_shape")
    parser.add_argument("--regime-percentile", type=float, default=30.0)
    args = parser.parse_args()

    symbols = ["BTC", "ETH", "SOL"] if args.all_symbols else [args.symbol]
    results = []

    t0 = time.time()
    for sym in symbols:
        result = run_analysis(
            data_dir=args.data_dir,
            symbol=sym,
            regime_feature=args.regime_feature,
            regime_percentile=args.regime_percentile,
            max_memory_mb=args.max_memory_mb,
        )
        results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # Save report
    report_path = args.json_report or "reports/kalman/phase1_ic.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"results": results, "elapsed_s": round(elapsed, 1)}, f, indent=2)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()

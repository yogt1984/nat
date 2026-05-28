#!/usr/bin/env python3
"""
IT Engine — Multi-Day Bar-Level Analysis

Runs KSG MI, CMI, interaction info, and greedy feature selection on 5min
bar-aggregated data across all available dates. Uses the same estimators as
scripts/it_engine/ but at MF timescale with full accumulated data.

Purpose: find cost-viable features beyond spread+depth that the single-day
smoke test couldn't detect.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse estimators from IT engine
from it_engine.estimators import ksg_mi, cmi, interaction_info, linear_te, min_info_bits
from it_engine.feature_selector import greedy_select

log = logging.getLogger(__name__)

BAR_SECONDS = 300  # 5min
MIN_BARS_PER_DATE = 12
HORIZONS_BARS = {"10min": 2, "25min": 5, "50min": 10}

# Columns to skip in aggregation
META_COLS = {"timestamp_ns", "symbol", "date", "hour", "minute", "second"}

FEE_RT_BPS = {
    "binance_vip9": 1.61,
    "binance_vip0": 3.50,
    "hyperliquid": 7.00,
}


def load_and_aggregate(data_dir: Path, symbol: str) -> pd.DataFrame | None:
    """Load all dates, filter to symbol, aggregate to 5min bars."""
    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )

    all_bars = []
    for date_str in all_dates:
        date_path = data_dir / date_str
        files = sorted(f for f in date_path.iterdir() if f.suffix == ".parquet")
        if not files:
            continue

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
                if len(df) > 0:
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            continue

        ticks = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns")
        if len(ticks) < 100:
            continue

        # Aggregate to 5min bars
        bar_ns = BAR_SECONDS * 1_000_000_000
        ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns

        # Identify numeric feature columns
        numeric_cols = [
            c for c in ticks.columns
            if c not in META_COLS and c != "bar_id"
            and ticks[c].dtype.kind in ('f', 'i', 'u')
        ]

        # Build aggregation dict: _last and _std for each feature
        agg_dict = {"timestamp_ns": "first"}
        for col in numeric_cols:
            agg_dict[col] = ["last", "std"]

        bars = ticks.groupby("bar_id").agg(agg_dict)
        # Flatten multi-level columns
        bars.columns = [
            f"{col}_{agg}" if agg != "first" else col
            for col, agg in bars.columns
        ]
        bars = bars.reset_index(drop=True)

        # Drop bars with too few ticks
        n_ticks_col = [c for c in bars.columns if "midprice" in c and c.endswith("_last")]
        if n_ticks_col:
            bars = bars.dropna(subset=n_ticks_col[:1])

        bars["_date"] = date_str

        if len(bars) >= MIN_BARS_PER_DATE:
            all_bars.append(bars)
            log.info("%s: %d bars", date_str, len(bars))
        else:
            log.debug("%s: %d bars — skipped", date_str, len(bars))

    if not all_bars:
        return None

    result = pd.concat(all_bars, ignore_index=True)
    log.info("Total: %d bars across %d dates for %s", len(result), len(all_bars), symbol)
    return result


def compute_forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forward returns in bps at bar horizon. NaN at boundaries between dates.

    Returns are NOT z-scored (sigma_r is needed raw for cost threshold).
    """
    price_col = None
    for candidate in ["raw_midprice_last", "raw_microprice_last"]:
        if candidate in bars.columns:
            price_col = candidate
            break
    if price_col is None:
        return np.full(len(bars), np.nan)

    # Use original (un-z-scored) midprice for return computation
    # raw_midprice_last was z-scored in run_it_analysis, so reconstruct from
    # the original price stored before z-scoring
    prices = bars[price_col].values
    dates = bars["_date"].values
    fwd = np.full(len(prices), np.nan)

    for i in range(len(prices) - horizon):
        # Don't compute across date boundaries
        if dates[i] != dates[i + horizon]:
            continue
        if prices[i] > 0 and np.isfinite(prices[i]) and np.isfinite(prices[i + horizon]):
            fwd[i] = (prices[i + horizon] - prices[i]) / prices[i] * 10000
    return fwd


def _prepare_feature(vals: np.ndarray) -> np.ndarray:
    """Prepare feature array for KSG: fill NaN with median, add jitter to break ties."""
    vals = vals.copy().astype(np.float64)
    nan_mask = ~np.isfinite(vals)
    if nan_mask.any():
        med = np.nanmedian(vals)
        vals[nan_mask] = med if np.isfinite(med) else 0.0
    # Add tiny jitter to break tied values (KSG fails with exact ties)
    std = np.std(vals)
    if std > 0:
        vals += np.random.randn(len(vals)) * std * 1e-6
    return vals


def run_it_analysis(bars: pd.DataFrame, symbol: str, fee_model: str, ksg_k: int = 5):
    """Run full IT analysis on bar-level data."""
    fee_bps = FEE_RT_BPS[fee_model]
    np.random.seed(42)

    # Identify feature columns (everything ending in _last or _std, excluding meta)
    feature_cols = [
        c for c in bars.columns
        if (c.endswith("_last") or c.endswith("_std"))
        and c != "timestamp_ns"
        and c != "_date"
        and bars[c].dtype.kind in ('f', 'i', 'u')
    ]

    # Drop constant or near-constant features
    active_cols = []
    for c in feature_cols:
        vals = bars[c].dropna().values
        if len(vals) > 50 and np.std(vals) > 1e-12:
            active_cols.append(c)
    feature_cols = active_cols

    log.info("%d active feature columns", len(feature_cols))

    # Identify entropy columns for conditioning
    entropy_cols = [c for c in feature_cols if c.startswith("ent_")]
    log.info("%d entropy conditioning columns", len(entropy_cols))

    # Pre-compute forward returns BEFORE z-scoring (needs original prices)
    fwd_returns = {}
    for h_name, h_bars in HORIZONS_BARS.items():
        fwd_returns[h_name] = compute_forward_returns(bars, h_bars)

    # Within-date z-scoring to remove cross-date level shifts
    # Without this, any feature that varies across dates gets inflated MI
    for c in feature_cols:
        bars[c] = bars.groupby("_date")[c].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-10)
        )

    # Re-check for constant features after z-scoring (some dates may have constant values)
    active_after = []
    for c in feature_cols:
        vals = bars[c].dropna().values
        if len(vals) > 50 and np.std(vals) > 1e-12:
            active_after.append(c)
    feature_cols = active_after
    log.info("%d active after within-date z-scoring", len(feature_cols))

    results = {}

    for h_name, h_bars in HORIZONS_BARS.items():
        log.info("── Horizon: %s (%d bars) ──", h_name, h_bars)
        fwd = fwd_returns[h_name]
        valid_mask = np.isfinite(fwd)
        n_valid = valid_mask.sum()

        if n_valid < 100:
            log.warning("Only %d valid samples at %s, skipping", n_valid, h_name)
            continue

        r = _prepare_feature(fwd[valid_mask])
        sigma_r = float(np.std(r))
        i_min = min_info_bits(fee_bps, sigma_r)
        log.info("N=%d, σ_r=%.2f bps, I_min=%.6f bits (fee=%.2f bps)", n_valid, sigma_r, i_min, fee_bps)

        # Phase 1: MI for each feature
        mi_results = {}
        t0 = time.time()
        for i, col in enumerate(feature_cols):
            vals = bars[col].values[valid_mask]
            if np.sum(np.isfinite(vals)) < 50 or np.nanstd(vals) < 1e-12:
                continue
            vals = _prepare_feature(vals)

            mi_val = ksg_mi(vals, r, k=ksg_k)
            mi_results[col] = {
                "mi": mi_val,
                "cost_viable": mi_val > i_min,
            }

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                log.info("  MI: %d/%d features (%.1fs)", i + 1, len(feature_cols), elapsed)

        elapsed = time.time() - t0
        log.info("MI computed for %d features in %.1fs", len(mi_results), elapsed)

        # Phase 2: CMI for top features (expensive, limit to top 30)
        top_by_mi = sorted(mi_results.items(), key=lambda x: x[1]["mi"], reverse=True)[:30]
        Z_cols = [c for c in entropy_cols if c in bars.columns]

        if Z_cols and top_by_mi:
            Z_parts = [_prepare_feature(bars[c].values[valid_mask]) for c in Z_cols]
            Z = np.column_stack(Z_parts)

            t0 = time.time()
            for col, info in top_by_mi:
                vals = _prepare_feature(bars[col].values[valid_mask])
                cmi_val = cmi(vals, r, Z, k=ksg_k)
                ii_val = cmi_val - info["mi"]
                mi_results[col]["cmi"] = cmi_val
                mi_results[col]["ii"] = ii_val

            elapsed = time.time() - t0
            log.info("CMI computed for top %d features in %.1fs", len(top_by_mi), elapsed)

        # Phase 3: Greedy selection at this horizon
        feat_arrays = {}
        for col in mi_results:
            feat_arrays[col] = _prepare_feature(bars[col].values[valid_mask])

        selected = greedy_select(
            features=feat_arrays,
            returns=r,
            fee_rt_bps=fee_bps,
            sigma_r_bps=sigma_r,
            max_features=10,
            k=ksg_k,
        )

        results[h_name] = {
            "n_valid": int(n_valid),
            "sigma_r_bps": round(sigma_r, 2),
            "i_min_bits": round(i_min, 6),
            "n_features_tested": len(mi_results),
            "n_cost_viable": sum(1 for v in mi_results.values() if v.get("cost_viable")),
            "top_mi": [
                {
                    "feature": col,
                    "mi": round(info["mi"], 6),
                    "cmi": round(info.get("cmi", 0), 6),
                    "ii": round(info.get("ii", 0), 6),
                    "cost_viable": info.get("cost_viable", False),
                }
                for col, info in top_by_mi
            ],
            "greedy_selected": selected,
        }

    return results


def print_results(results: dict, symbol: str):
    """Pretty-print IT analysis results."""
    print(f"\n{'═' * 70}")
    print(f"  IT Analysis — {symbol}")
    print(f"{'═' * 70}\n")

    for h_name, h_data in results.items():
        viable = h_data["n_cost_viable"]
        tested = h_data["n_features_tested"]
        sigma = h_data["sigma_r_bps"]
        i_min = h_data["i_min_bits"]

        print(f"── {h_name} (N={h_data['n_valid']}, σ_r={sigma:.1f} bps, "
              f"I_min={i_min:.4f} bits) ──")
        print(f"   {tested} features tested, {viable} cost-viable\n")

        # Top 15 by MI
        print(f"   {'Feature':<40s}  {'MI':>8s}  {'CMI':>8s}  {'II':>8s}  {'Viable':>7s}")
        print(f"   {'─' * 75}")
        for entry in h_data["top_mi"][:15]:
            flag = "  ✓" if entry["cost_viable"] else ""
            ii_str = f"{entry['ii']:+.4f}" if entry["ii"] else ""
            print(f"   {entry['feature']:<40s}  {entry['mi']:8.4f}  "
                  f"{entry['cmi']:8.4f}  {ii_str:>8s}{flag}")

        # Greedy selection
        if h_data["greedy_selected"]:
            print(f"\n   Greedy selection (cumulative MI):")
            for step in h_data["greedy_selected"]:
                viable_flag = " ← VIABLE" if step["cost_viable"] else ""
                print(f"     {step['name']:<38s}  gain={step['cmi_gain']:.4f}  "
                      f"cum={step['cumulative_mi']:.4f}{viable_flag}")
        else:
            print(f"\n   Greedy selection: no features above I_min")

        print()


def main():
    parser = argparse.ArgumentParser(description="IT Engine — Multi-Day Bar Analysis")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--fee-model", choices=list(FEE_RT_BPS.keys()), default="binance_vip9")
    parser.add_argument("--ksg-k", type=int, default=5)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    bars = load_and_aggregate(data_dir, args.symbol)
    if bars is None:
        print(f"No data found for {args.symbol}")
        sys.exit(1)

    results = run_it_analysis(bars, args.symbol, args.fee_model, args.ksg_k)
    print_results(results, args.symbol)

    if args.save:
        report = {
            "title": f"IT Multi-Day Bar Analysis — {args.symbol}",
            "generated": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "fee_model": args.fee_model,
            "fee_bps_rt": FEE_RT_BPS[args.fee_model],
            "horizons": results,
        }
        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"it_multiday_{args.symbol.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()

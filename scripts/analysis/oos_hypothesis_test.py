#!/usr/bin/env python3
"""
OOS Hypothesis Test: Is spread IC=0.54 at 50min real?

Three tests:
  1. Permutation test (BTC): shuffle returns within each date 1000x
  2. Cross-symbol OOS (ETH, SOL): IC scan was only run on BTC
  3. Temporal split: first-half calibration, second-half test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BAR_SECONDS = 300
MIN_BARS = 20
HORIZON_BARS = 10  # 50min
N_PERMUTATIONS = 2000

# Top features to test (from BTC IC scan)
FEATURES_TO_TEST = [
    "raw_spread_bps",
    "raw_ask_depth_5",
    "raw_ask_depth_10",
    "flow_vwap_deviation",
    "trend_momentum_60",
    "vol_returns_1m",
    "imbalance_pressure_bid",
    "ent_book_shape",
]


def load_symbol_bars(data_dir: Path, symbol: str) -> list[tuple[str, pd.DataFrame]]:
    """Load all dates for a symbol, aggregate to 5min bars."""
    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )
    date_bars = []
    for date_str in all_dates:
        dp = data_dir / date_str
        files = sorted(f for f in dp.iterdir() if f.suffix == ".parquet")
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
        bar_ns = BAR_SECONDS * 1_000_000_000
        ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns

        # Aggregate key features
        agg = {"raw_midprice": "last"}
        for feat in FEATURES_TO_TEST:
            if feat in ticks.columns:
                agg[feat] = ["last", "std"]
        bars = ticks.groupby("bar_id").agg(agg)
        bars.columns = [
            f"{c}_{a}" if a != "last" or c != "raw_midprice" else "mid"
            for c, a in bars.columns
        ]
        bars = bars.reset_index(drop=True)
        bars = bars[bars["mid"].notna()].reset_index(drop=True)
        if len(bars) >= MIN_BARS:
            date_bars.append((date_str, bars))
    return date_bars


def compute_date_ic(bars: pd.DataFrame, feat_col: str) -> float | None:
    """Compute Spearman IC of feat vs 50min forward return for one date."""
    prices = bars["mid"].values
    n = len(prices)
    if n <= HORIZON_BARS + 5:
        return None
    fwd = np.full(n, np.nan)
    for i in range(n - HORIZON_BARS):
        if prices[i] > 0:
            fwd[i] = (prices[i + HORIZON_BARS] - prices[i]) / prices[i] * 1e4
    valid = np.isfinite(fwd)
    if feat_col not in bars.columns:
        return None
    x = bars[feat_col].values
    both_valid = valid & np.isfinite(x)
    if both_valid.sum() < 15 or np.std(x[both_valid]) < 1e-12:
        return None
    ic, _ = spearmanr(x[both_valid], fwd[both_valid])
    return float(ic) if np.isfinite(ic) else None


def run_permutation_test(date_bars: list[tuple[str, pd.DataFrame]], feat_col: str):
    """Test 1: permutation test. Shuffle returns within each date."""
    # Observed ICs
    obs_ics = []
    for _, bars in date_bars:
        ic = compute_date_ic(bars, feat_col)
        if ic is not None:
            obs_ics.append(ic)
    if not obs_ics:
        return None
    obs_mean = np.mean(obs_ics)

    # Permutation distribution
    rng = np.random.default_rng(42)
    perm_means = []
    for _ in range(N_PERMUTATIONS):
        perm_ics = []
        for _, bars in date_bars:
            prices = bars["mid"].values
            n = len(prices)
            if n <= HORIZON_BARS + 5:
                continue
            fwd = np.full(n, np.nan)
            for i in range(n - HORIZON_BARS):
                if prices[i] > 0:
                    fwd[i] = (prices[i + HORIZON_BARS] - prices[i]) / prices[i] * 1e4
            valid = np.isfinite(fwd)
            if feat_col not in bars.columns:
                continue
            x = bars[feat_col].values
            both_valid = valid & np.isfinite(x)
            if both_valid.sum() < 15:
                continue

            # Shuffle returns (preserving feature values)
            fwd_shuffled = fwd.copy()
            idx = np.where(both_valid)[0]
            fwd_shuffled[idx] = rng.permutation(fwd[idx])

            ic, _ = spearmanr(x[both_valid], fwd_shuffled[both_valid])
            if np.isfinite(ic):
                perm_ics.append(ic)

        if perm_ics:
            perm_means.append(np.mean(perm_ics))

    perm_arr = np.array(perm_means)
    p_value = np.mean(perm_arr >= obs_mean)

    return {
        "observed_mean_ic": round(obs_mean, 4),
        "n_dates": len(obs_ics),
        "perm_mean": round(np.mean(perm_arr), 4),
        "perm_std": round(np.std(perm_arr), 4),
        "perm_p95": round(np.percentile(perm_arr, 95), 4),
        "perm_p99": round(np.percentile(perm_arr, 99), 4),
        "p_value": round(p_value, 4),
        "n_permutations": N_PERMUTATIONS,
    }


def run_cross_symbol(data_dir: Path, feat_col: str):
    """Test 2: cross-symbol OOS. Run on ETH and SOL."""
    results = {}
    for symbol in ["ETH", "SOL"]:
        date_bars = load_symbol_bars(data_dir, symbol)
        ics = []
        per_date = {}
        for d, bars in date_bars:
            ic = compute_date_ic(bars, feat_col)
            if ic is not None:
                ics.append(ic)
                per_date[d] = round(ic, 4)
        if ics:
            arr = np.array(ics)
            results[symbol] = {
                "mean_ic": round(np.mean(arr), 4),
                "std_ic": round(np.std(arr), 4),
                "n_positive": int(np.sum(arr > 0)),
                "n_dates": len(arr),
                "t_stat": round(np.mean(arr) / (np.std(arr) / np.sqrt(len(arr))), 2)
                if np.std(arr) > 0 else 0.0,
                "per_date": per_date,
            }
    return results


def run_temporal_split(date_bars: list[tuple[str, pd.DataFrame]], feat_col: str):
    """Test 3: first half calibration, second half OOS."""
    n = len(date_bars)
    mid = n // 2

    first_ics, second_ics = [], []
    first_dates, second_dates = {}, {}
    for i, (d, bars) in enumerate(date_bars):
        ic = compute_date_ic(bars, feat_col)
        if ic is None:
            continue
        if i < mid:
            first_ics.append(ic)
            first_dates[d] = round(ic, 4)
        else:
            second_ics.append(ic)
            second_dates[d] = round(ic, 4)

    result = {}
    if first_ics:
        arr = np.array(first_ics)
        result["first_half"] = {
            "dates": list(first_dates.keys()),
            "mean_ic": round(np.mean(arr), 4),
            "std": round(np.std(arr), 4),
            "n_positive": int(np.sum(arr > 0)),
            "n": len(arr),
            "per_date": first_dates,
        }
    if second_ics:
        arr = np.array(second_ics)
        result["second_half"] = {
            "dates": list(second_dates.keys()),
            "mean_ic": round(np.mean(arr), 4),
            "std": round(np.std(arr), 4),
            "n_positive": int(np.sum(arr > 0)),
            "n": len(arr),
            "per_date": second_dates,
        }
    return result


def main():
    data_dir = Path("data/features")

    # Load BTC
    print("Loading BTC data...")
    btc_bars = load_symbol_bars(data_dir, "BTC")
    print(f"  {len(btc_bars)} dates loaded\n")

    # Test the key features
    features = ["raw_spread_bps_last", "raw_ask_depth_5_std", "flow_vwap_deviation_std"]

    for feat in features:
        print(f"{'═' * 70}")
        print(f"  Feature: {feat}")
        print(f"{'═' * 70}\n")

        # Test 1: Permutation
        print("  ── Test 1: Permutation (BTC, 2000 shuffles) ──")
        perm = run_permutation_test(btc_bars, feat)
        if perm:
            print(f"    Observed mean IC:  {perm['observed_mean_ic']:+.4f} ({perm['n_dates']} dates)")
            print(f"    Null distribution: mean={perm['perm_mean']:+.4f}, "
                  f"std={perm['perm_std']:.4f}")
            print(f"    Null P95/P99:      {perm['perm_p95']:+.4f} / {perm['perm_p99']:+.4f}")
            print(f"    p-value:           {perm['p_value']:.4f}")
            sig = "***" if perm["p_value"] < 0.001 else "**" if perm["p_value"] < 0.01 else "*" if perm["p_value"] < 0.05 else "ns"
            print(f"    Significance:      {sig}")
        print()

        # Test 2: Cross-symbol
        print("  ── Test 2: Cross-Symbol OOS (ETH, SOL) ──")
        cross = run_cross_symbol(data_dir, feat)
        for sym, res in cross.items():
            sign_pct = res['n_positive'] / res['n_dates'] * 100
            print(f"    {sym}: IC={res['mean_ic']:+.4f} ± {res['std_ic']:.4f}, "
                  f"t={res['t_stat']:+.2f}, "
                  f"positive {res['n_positive']}/{res['n_dates']} ({sign_pct:.0f}%)")
            for d, ic in sorted(res["per_date"].items()):
                tag = "+" if ic > 0 else " "
                print(f"      {d}: {tag}{ic:+.4f}")
        print()

        # Test 3: Temporal split
        print("  ── Test 3: Temporal Split (BTC first/second half) ──")
        split = run_temporal_split(btc_bars, feat)
        for half_name, half in split.items():
            n_pos = half["n_positive"]
            n_tot = half["n"]
            print(f"    {half_name}: IC={half['mean_ic']:+.4f} ± {half['std']:.4f}, "
                  f"positive {n_pos}/{n_tot} "
                  f"({half['dates'][0]}..{half['dates'][-1]})")
        print()


if __name__ == "__main__":
    main()

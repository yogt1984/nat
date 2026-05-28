#!/usr/bin/env python3
"""
Signal correlation analysis between winning algorithms.

For each (date, symbol), generate the direction vector from each algorithm,
then compute pairwise correlations and trade overlap statistics.

Winning signals:
  1. 3f liquidity (spread + depth + vwap_dev composite)
  2. jump_detector (post-jump reversion)
  3. optimal_entry (SPRT on Kalman innovation)
  4. funding_reversion (funding rate z-score mean-reversion)
  5. surprise_signal (entropy regime transition)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent

from algorithms import get_algorithm
from algorithms.surprise_signal import SurpriseSignal

# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300
HORIZON_BARS = 20
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12

# Columns needed per algorithm
ALGO_SPECS = {
    "3f_liquidity": {
        "columns": ["raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation"],
        "type": "custom_3f",
    },
    "jump_detector": {
        "columns": ["raw_midprice"],
        "primary": "alg_post_jump_reversion",
        "polarity": "low_long",
    },
    "optimal_entry": {
        "columns": ["imbalance_qty_l1"],
        "primary": "alg_entry_signal",
        "polarity": "high_long",
    },
    "funding_reversion": {
        "columns": ["ctx_funding_rate", "ctx_funding_zscore", "ctx_premium_bps"],
        "primary": "alg_funding_signal",
        "polarity": "high_long",
    },
    "surprise_signal": {
        "columns": ["ent_book_shape", "ent_tick_5s"],
        "type": "custom_surprise",
    },
}

P_HIGH = 80
P_LOW = 20


# ── Data loading ────────────────────────────────────────────────────────

def discover_dates(data_dir: Path) -> list[str]:
    return sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )


def load_date_ticks(data_dir: Path, date_str: str, symbol: str,
                    extra_cols: list[str]) -> pd.DataFrame | None:
    date_path = data_dir / date_str
    if not date_path.is_dir():
        return None
    files = sorted(f for f in date_path.iterdir() if f.suffix == ".parquet")
    if not files:
        return None

    base = ["timestamp_ns", "symbol", "raw_midprice"]
    load_cols = list(set(base + extra_cols))

    dfs = []
    for f in files:
        try:
            tbl = pq.read_table(str(f))
            df = tbl.to_pandas()
            cols = [c for c in load_cols if c in df.columns]
            df = df[cols]
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].copy()
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns").reset_index(drop=True)


# ── Bar aggregation ─────────────────────────────────────────────────────

def make_bars_3f(ticks: pd.DataFrame) -> pd.DataFrame:
    bar_ns = BAR_SECONDS * 1_000_000_000
    t = ticks.copy()
    t["bar_id"] = t["timestamp_ns"].values // bar_ns
    agg = {
        "midprice_last": ("raw_midprice", "last"),
        "spread_bps_last": ("raw_spread_bps", "last"),
        "depth_5_std": ("raw_ask_depth_5", "std"),
        "n_ticks": ("raw_midprice", "count"),
    }
    if "flow_vwap_deviation" in t.columns:
        agg["vwap_deviation_std"] = ("flow_vwap_deviation", "std")
    bars = t.groupby("bar_id").agg(**agg).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    if "vwap_deviation_std" in bars.columns:
        bars["vwap_deviation_std"] = bars["vwap_deviation_std"].fillna(0.0)
    return bars


def make_bars_algo(ticks: pd.DataFrame, algo_name: str, primary: str,
                   agg_method: str = "mean") -> pd.DataFrame:
    algo = get_algorithm(algo_name)
    features = algo.run_batch(ticks)
    algo.reset()

    bar_ns = BAR_SECONDS * 1_000_000_000
    t = ticks.copy()
    t["bar_id"] = t["timestamp_ns"].values // bar_ns
    t["_signal"] = features[primary].values

    agg_fn = "last" if agg_method == "last" else "mean"
    bars = t.groupby("bar_id").agg(
        midprice_last=("raw_midprice", "last"),
        n_ticks=("raw_midprice", "count"),
        signal=("_signal", agg_fn),
    ).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    return bars


def make_bars_surprise(ticks: pd.DataFrame) -> pd.DataFrame:
    algo = SurpriseSignal(roc_window=50, transition_threshold=2.0)
    features = algo.run_batch(ticks)

    bar_ns = BAR_SECONDS * 1_000_000_000
    t = ticks.copy()
    t["bar_id"] = t["timestamp_ns"].values // bar_ns
    t["_signal"] = features["alg_entropy_surprise"].values

    bars = t.groupby("bar_id").agg(
        midprice_last=("raw_midprice", "last"),
        n_ticks=("raw_midprice", "count"),
        signal=("_signal", "mean"),
    ).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    return bars


# ── Signal generation ───────────────────────────────────────────────────

def direction_3f(train_bars: list[pd.DataFrame], test_bars: pd.DataFrame) -> np.ndarray:
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars])
    vwap_arrs = [b["vwap_deviation_std"].values for b in train_bars
                 if "vwap_deviation_std" in b.columns]
    if not vwap_arrs:
        return np.zeros(len(test_bars))
    vwap = np.concatenate(vwap_arrs)
    n = min(len(spread), len(depth), len(vwap))
    spread, depth, vwap = spread[:n], depth[:n], vwap[:n]
    mask = np.isfinite(spread) & np.isfinite(depth) & np.isfinite(vwap)
    spread, depth, vwap = spread[mask], depth[mask], vwap[mask]
    if len(spread) < 20:
        return np.zeros(len(test_bars))

    params = {
        "s_m": np.mean(spread), "s_s": max(np.std(spread), 1e-10),
        "d_m": np.mean(depth), "d_s": max(np.std(depth), 1e-10),
        "v_m": np.mean(vwap), "v_s": max(np.std(vwap), 1e-10),
    }
    z_s = (spread - params["s_m"]) / params["s_s"]
    z_d = (depth - params["d_m"]) / params["d_s"]
    z_v = (vwap - params["v_m"]) / params["v_s"]
    composite_train = (z_s + z_d + z_v) / 3.0
    p_long = np.percentile(composite_train, P_HIGH)
    p_short = np.percentile(composite_train, P_LOW)

    tb = test_bars
    z_s = (tb["spread_bps_last"] - params["s_m"]) / params["s_s"]
    z_d = (tb["depth_5_std"] - params["d_m"]) / params["d_s"]
    z_v = (tb["vwap_deviation_std"] - params["v_m"]) / params["v_s"]
    composite = (z_s + z_d + z_v) / 3.0

    dirs = np.zeros(len(tb))
    dirs[composite.values >= p_long] = 1
    dirs[composite.values <= p_short] = -1
    return dirs


def direction_generic(train_bars: list[pd.DataFrame], test_bars: pd.DataFrame,
                      polarity: str) -> np.ndarray:
    vals = np.concatenate([b["signal"].values for b in train_bars])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return np.zeros(len(test_bars))

    mean = np.mean(vals)
    std = max(np.std(vals), 1e-10)
    z_train = (vals - mean) / std

    if polarity == "high_long":
        p_long = np.percentile(z_train, P_HIGH)
        p_short = np.percentile(z_train, P_LOW)
    else:
        p_long = np.percentile(z_train, P_LOW)
        p_short = np.percentile(z_train, P_HIGH)

    z_test = (test_bars["signal"].values - mean) / std
    dirs = np.zeros(len(test_bars))

    if polarity == "high_long":
        dirs[z_test >= p_long] = 1
        dirs[z_test <= p_short] = -1
    else:
        dirs[z_test <= p_long] = 1
        dirs[z_test >= p_short] = -1

    return dirs


# ── Main ────────────────────────────────────────────────────────────────

def main():
    data_dir = ROOT / "data" / "features"
    symbols = ["BTC", "ETH", "SOL"]
    all_dates = discover_dates(data_dir)
    algo_names = list(ALGO_SPECS.keys())

    # Collect all needed columns
    all_cols = set()
    for spec in ALGO_SPECS.values():
        all_cols.update(spec["columns"])

    print(f"Signal Correlation Analysis")
    print(f"Dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")
    print(f"Algorithms: {', '.join(algo_names)}")
    print(f"{'=' * 70}\n")

    results = {}

    for symbol in symbols:
        print(f"═══ {symbol} ═══")

        # Load all dates
        date_data: dict[str, dict] = {}  # date -> {algo_name -> bars}
        for date_str in all_dates:
            ticks = load_date_ticks(data_dir, date_str, symbol, list(all_cols))
            if ticks is None or len(ticks) < 200:
                continue

            entry = {}

            # 3f bars
            if all(c in ticks.columns for c in ALGO_SPECS["3f_liquidity"]["columns"]):
                bars = make_bars_3f(ticks)
                if len(bars) >= MIN_BARS_PER_DATE:
                    entry["3f_liquidity"] = bars

            # Algorithm bars
            for aname in ["jump_detector", "optimal_entry", "funding_reversion"]:
                spec = ALGO_SPECS[aname]
                if all(c in ticks.columns for c in spec["columns"]):
                    try:
                        agg = "last" if aname in ("optimal_entry", "funding_reversion") else "mean"
                        bars = make_bars_algo(ticks, aname, spec["primary"], agg)
                        if len(bars) >= MIN_BARS_PER_DATE:
                            entry[aname] = bars
                    except Exception as e:
                        pass

            # Surprise bars
            if all(c in ticks.columns for c in ALGO_SPECS["surprise_signal"]["columns"]):
                try:
                    bars = make_bars_surprise(ticks)
                    if len(bars) >= MIN_BARS_PER_DATE:
                        entry["surprise_signal"] = bars
                except Exception:
                    pass

            if len(entry) >= 2:
                date_data[date_str] = entry

        dates_sorted = sorted(date_data.keys())
        print(f"  {len(dates_sorted)} usable dates\n")

        if len(dates_sorted) < TRAIN_WINDOW + 1:
            continue

        # Walk-forward: generate direction vectors per OOS date
        all_directions: dict[str, list[np.ndarray]] = {a: [] for a in algo_names}
        oos_bar_counts = []

        for i in range(TRAIN_WINDOW, len(dates_sorted)):
            test_date = dates_sorted[i]
            train_dates = dates_sorted[i - TRAIN_WINDOW:i]

            # Check which algos have data for all train + test dates
            available = set(algo_names)
            for d in train_dates + [test_date]:
                if d in date_data:
                    available &= set(date_data[d].keys())

            if len(available) < 2:
                continue

            # Common bar count = min across algos for this test date
            n_bars = min(len(date_data[test_date][a]) for a in available)
            oos_bar_counts.append(n_bars)

            for aname in algo_names:
                if aname not in available:
                    all_directions[aname].append(np.zeros(n_bars))
                    continue

                train_bars = [date_data[d][aname] for d in train_dates]
                test_bars = date_data[test_date][aname].iloc[:n_bars].copy()

                if aname == "3f_liquidity":
                    dirs = direction_3f(train_bars, test_bars)
                elif aname == "surprise_signal":
                    dirs = direction_generic(train_bars, test_bars, "low_long")
                else:
                    dirs = direction_generic(train_bars, test_bars,
                                             ALGO_SPECS[aname]["polarity"])

                all_directions[aname].append(dirs[:n_bars])

        # Concatenate all OOS directions
        dir_concat = {}
        for aname in algo_names:
            if all_directions[aname]:
                dir_concat[aname] = np.concatenate(all_directions[aname])

        n_total = len(dir_concat[algo_names[0]]) if algo_names[0] in dir_concat else 0
        print(f"  Total OOS bars: {n_total}")

        # ── Pairwise correlation of direction vectors ──
        print(f"\n  Direction Correlation (Spearman):")
        print(f"  {'':22s}", end="")
        for a in algo_names:
            print(f" {a[:10]:>10s}", end="")
        print()

        corr_matrix = {}
        for a1 in algo_names:
            corr_matrix[a1] = {}
            print(f"  {a1:22s}", end="")
            for a2 in algo_names:
                if a1 in dir_concat and a2 in dir_concat:
                    v1, v2 = dir_concat[a1], dir_concat[a2]
                    # Only compare where both are non-zero or both have signal
                    mask = np.isfinite(v1) & np.isfinite(v2)
                    if mask.sum() > 20:
                        from scipy.stats import spearmanr
                        rho, _ = spearmanr(v1[mask], v2[mask])
                    else:
                        rho = np.nan
                else:
                    rho = np.nan
                corr_matrix[a1][a2] = round(rho, 3) if np.isfinite(rho) else None
                if a1 == a2:
                    print(f"      1.000", end="")
                elif np.isfinite(rho):
                    print(f" {rho:+10.3f}", end="")
                else:
                    print(f"        N/A", end="")
            print()

        # ── Trade overlap analysis ──
        print(f"\n  Trade Overlap (% of bars where both fire same direction):")
        print(f"  {'':22s}", end="")
        for a in algo_names:
            print(f" {a[:10]:>10s}", end="")
        print()

        overlap_matrix = {}
        for a1 in algo_names:
            overlap_matrix[a1] = {}
            print(f"  {a1:22s}", end="")
            for a2 in algo_names:
                if a1 in dir_concat and a2 in dir_concat:
                    d1, d2 = dir_concat[a1], dir_concat[a2]
                    both_active = (d1 != 0) & (d2 != 0)
                    if both_active.sum() > 0:
                        same_dir = ((d1 == d2) & both_active).sum()
                        overlap_pct = same_dir / both_active.sum() * 100
                    else:
                        overlap_pct = 0
                else:
                    overlap_pct = np.nan
                overlap_matrix[a1][a2] = round(overlap_pct, 1) if np.isfinite(overlap_pct) else None
                if a1 == a2:
                    print(f"     100.0%", end="")
                elif np.isfinite(overlap_pct):
                    print(f"  {overlap_pct:7.1f}%", end="")
                else:
                    print(f"        N/A", end="")
            print()

        # ── Signal activity ──
        print(f"\n  Signal Activity:")
        for aname in algo_names:
            if aname in dir_concat:
                d = dir_concat[aname]
                n_long = (d == 1).sum()
                n_short = (d == -1).sum()
                n_flat = (d == 0).sum()
                pct_active = (n_long + n_short) / len(d) * 100
                print(f"    {aname:22s}: {n_long:4d} long | {n_short:4d} short | "
                      f"{n_flat:4d} flat | {pct_active:.1f}% active")

        # ── Simultaneous firing ──
        print(f"\n  Simultaneous Firing (bars where N algorithms fire):")
        if all(a in dir_concat for a in algo_names):
            active_count = np.zeros(n_total)
            for aname in algo_names:
                active_count += (dir_concat[aname] != 0).astype(float)
            for k in range(6):
                cnt = (active_count == k).sum()
                if cnt > 0:
                    print(f"    {k} signals active: {cnt:5d} bars ({cnt / n_total * 100:.1f}%)")

        results[symbol] = {
            "n_oos_bars": n_total,
            "correlation": corr_matrix,
            "overlap": overlap_matrix,
        }
        print()

    # Save report
    report = {
        "title": "Signal Correlation Analysis — Winning Algorithms",
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "algorithms": algo_names,
        "data_interval": f"{all_dates[0]} to {all_dates[-1]}",
        "results": results,
    }
    out = ROOT / "reports" / "signal_correlation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()

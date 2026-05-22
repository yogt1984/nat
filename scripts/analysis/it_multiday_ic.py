#!/usr/bin/env python3
"""
Multi-Day IC Scan — Feature Discovery at MF Timescale

For each bar-aggregated feature, computes within-date Spearman IC against
forward returns, then averages across dates. This avoids the cross-date
confound that inflates pooled MI estimates.

Output: ranked feature list with mean IC, IC std, IC IR (stability), and
number of positive-IC dates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BAR_SECONDS = 300
MIN_BARS_PER_DATE = 20
META_COLS = {"timestamp_ns", "symbol", "date", "hour", "minute", "second"}
HORIZONS = {"10min": 2, "25min": 5, "50min": 10}


def load_and_aggregate(data_dir: Path, symbol: str) -> list[tuple[str, pd.DataFrame]]:
    """Load all dates, filter to symbol, aggregate to 5min bars. Returns list of (date, bars)."""
    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )

    date_bars = []
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

        bar_ns = BAR_SECONDS * 1_000_000_000
        ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns

        # Identify numeric feature columns
        numeric_cols = [
            c for c in ticks.columns
            if c not in META_COLS and c != "bar_id"
            and ticks[c].dtype.kind in ('f', 'i', 'u')
        ]

        # Aggregate: last + std for each feature
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = ["last", "std"]

        bars = ticks.groupby("bar_id").agg(agg_dict)
        bars.columns = [f"{col}_{agg}" for col, agg in bars.columns]
        bars = bars.reset_index(drop=True)

        if len(bars) >= MIN_BARS_PER_DATE:
            date_bars.append((date_str, bars))

    return date_bars


def compute_ic_table(date_bars: list[tuple[str, pd.DataFrame]], horizons: dict) -> dict:
    """Compute per-date Spearman IC for each feature × horizon."""
    # Get union of all feature columns across dates
    all_feat_cols = set()
    for _, bars in date_bars:
        cols = [c for c in bars.columns if bars[c].dtype.kind in ('f', 'i', 'u')]
        all_feat_cols.update(cols)

    # Exclude midprice/microprice _last (price level, not feature)
    all_feat_cols = sorted(c for c in all_feat_cols
                           if c != "raw_midprice_last" and c != "raw_microprice_last"
                           and c != "sequence_id_last" and c != "sequence_id_std")

    results = {}

    for h_name, h_bars in horizons.items():
        print(f"\n── {h_name} ({h_bars} bars forward) ──")

        feat_ics: dict[str, list[float]] = {c: [] for c in all_feat_cols}

        for date_str, bars in date_bars:
            # Price for forward returns (use un-aggregated or _last)
            price_col = None
            for candidate in ["raw_midprice_last", "raw_microprice_last"]:
                if candidate in bars.columns:
                    price_col = candidate
                    break
            if price_col is None:
                continue

            prices = bars[price_col].values
            n = len(prices)
            if n <= h_bars + 5:
                continue

            # Forward returns
            fwd = np.full(n, np.nan)
            for i in range(n - h_bars):
                if prices[i] > 0 and np.isfinite(prices[i]) and np.isfinite(prices[i + h_bars]):
                    fwd[i] = (prices[i + h_bars] - prices[i]) / prices[i] * 10000
            valid_fwd = np.isfinite(fwd)

            if valid_fwd.sum() < 15:
                continue

            r = fwd[valid_fwd]

            for col in all_feat_cols:
                if col not in bars.columns:
                    continue
                x = bars[col].values[valid_fwd]
                x_valid = np.isfinite(x)
                if x_valid.sum() < 15 or np.std(x[x_valid]) < 1e-12:
                    continue

                # Replace NaN for spearman
                x_clean = x.copy()
                x_clean[~x_valid] = np.nanmedian(x)

                try:
                    ic, _ = spearmanr(x_clean, r)
                    if np.isfinite(ic):
                        feat_ics[col].append(ic)
                except Exception:
                    continue

        # Aggregate: mean IC, std, IC IR, sign consistency
        table = []
        for col, ics in feat_ics.items():
            if len(ics) < 3:
                continue
            arr = np.array(ics)
            mean_ic = float(np.mean(arr))
            std_ic = float(np.std(arr))
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
            n_positive = int(np.sum(arr > 0))
            n_total = len(arr)
            # t-stat for mean IC != 0
            t_stat = mean_ic / (std_ic / np.sqrt(n_total)) if std_ic > 0 else 0.0

            table.append({
                "feature": col,
                "mean_ic": round(mean_ic, 4),
                "std_ic": round(std_ic, 4),
                "ic_ir": round(ic_ir, 3),
                "t_stat": round(t_stat, 2),
                "n_positive": n_positive,
                "n_dates": n_total,
                "sign_rate": round(n_positive / n_total, 2),
            })

        # Sort by |mean_ic| * sign_consistency
        table.sort(key=lambda x: abs(x["mean_ic"]) * x["sign_rate"], reverse=True)
        results[h_name] = table

        # Print top 30
        print(f"  {'Feature':<45} {'IC':>7} {'std':>7} {'IR':>6} {'t':>6} "
              f"{'sign':>5} {'N':>3}")
        print(f"  {'─' * 85}")
        for entry in table[:30]:
            flag = " ←" if abs(entry["mean_ic"]) > 0.1 and entry["sign_rate"] >= 0.7 else ""
            print(f"  {entry['feature']:<45} {entry['mean_ic']:>+.4f} "
                  f"{entry['std_ic']:>7.4f} {entry['ic_ir']:>+6.3f} "
                  f"{entry['t_stat']:>+6.2f} "
                  f"{entry['n_positive']:>2}/{entry['n_dates']:<2} {flag}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Day IC Scan")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    date_bars = load_and_aggregate(data_dir, args.symbol)
    print(f"Loaded {len(date_bars)} dates for {args.symbol}")
    for d, b in date_bars:
        print(f"  {d}: {len(b)} bars")

    results = compute_ic_table(date_bars, HORIZONS)

    # Summary: features with |IC| > 0.1 and sign_rate >= 0.7 at 50min
    if "50min" in results:
        strong = [e for e in results["50min"]
                  if abs(e["mean_ic"]) > 0.1 and e["sign_rate"] >= 0.7]
        print(f"\n══ Strong signals at 50min: {len(strong)} features with "
              f"|IC| > 0.1 and sign rate >= 70% ══")
        for e in strong[:20]:
            print(f"  {e['feature']:<45} IC={e['mean_ic']:+.4f}  "
                  f"IR={e['ic_ir']:+.3f}  t={e['t_stat']:+.2f}  "
                  f"sign={e['n_positive']}/{e['n_dates']}")

    if args.save:
        report = {
            "title": f"Multi-Day IC Scan — {args.symbol}",
            "generated": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "n_dates": len(date_bars),
            "horizons": {h: [e for e in table[:50]] for h, table in results.items()},
        }
        out_path = Path(f"reports/ic_scan_{args.symbol.lower()}.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MF Liquidity Signal — Walk-Forward Backtest

Supports 2-feature (spread+depth) and 3-feature (spread+depth+vwap_deviation) composites.
Horizons from 10min to 200min. Walk-forward: train on prior dates, test on next.

Usage:
  python scripts/analysis/mf_liquidity_backtest.py --features both --save
  python scripts/analysis/mf_liquidity_backtest.py --features 3 --horizons 100 200
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
import pyarrow.parquet as pq


# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300  # 5min
HORIZONS_BARS_DEFAULT = [2, 5, 10, 20, 40]  # 10/25/50/100/200 min
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12  # need at least 60min of data
P_LONG = 80
P_SHORT = 20

FEE_MODELS = {
    "binance_vip9": 1.61,
    "binance_vip0": 3.50,
    "hyperliquid": 7.00,
}


# ── Data loading ──────────────────────────────────────────────────────────

def load_date(data_dir: Path, date_str: str, symbol: str) -> pd.DataFrame | None:
    """Load all parquet files for a date, filter to symbol, return tick df."""
    date_path = data_dir / date_str
    if not date_path.is_dir():
        return None

    files = sorted(f for f in date_path.iterdir() if f.suffix == ".parquet")
    if not files:
        return None

    want_cols = [
        "timestamp_ns", "symbol", "raw_midprice",
        "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation",
    ]
    dfs = []
    for f in files:
        try:
            tbl = pq.read_table(str(f))
            df = tbl.to_pandas()
            cols = [c for c in want_cols if c in df.columns]
            df = df[cols]
            df = df[df["symbol"] == symbol].copy() if "symbol" in df.columns else df
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns").reset_index(drop=True)


def aggregate_to_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 100ms ticks to 5min bars."""
    ts = ticks["timestamp_ns"].values
    # Floor to 5min buckets (nanoseconds)
    bar_ns = BAR_SECONDS * 1_000_000_000
    bar_id = ts // bar_ns

    ticks = ticks.copy()
    ticks["bar_id"] = bar_id

    agg = {
        "timestamp_ns": ("timestamp_ns", "first"),
        "midprice_last": ("raw_midprice", "last"),
        "spread_bps_last": ("raw_spread_bps", "last"),
        "depth_5_std": ("raw_ask_depth_5", "std"),
        "n_ticks": ("raw_midprice", "count"),
    }
    if "flow_vwap_deviation" in ticks.columns:
        agg["vwap_deviation_std"] = ("flow_vwap_deviation", "std")

    bars = ticks.groupby("bar_id").agg(**agg).reset_index(drop=True)

    # Drop bars with too few ticks (partial bars at edges)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    if "vwap_deviation_std" in bars.columns:
        bars["vwap_deviation_std"] = bars["vwap_deviation_std"].fillna(0.0)
    return bars


# ── Signal construction ───────────────────────────────────────────────────

def compute_zscore_params_2f(train_bars_list: list[pd.DataFrame]) -> dict:
    """2-feature z-score params: spread + depth."""
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    mask = np.isfinite(spread) & np.isfinite(depth)
    spread, depth = spread[mask], depth[mask]
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    composite = (z_s + z_d) / 2.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def compute_zscore_params_3f(train_bars_list: list[pd.DataFrame]) -> dict | None:
    """3-feature z-score params: spread + depth + vwap_deviation."""
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    vwap_arrs = [b["vwap_deviation_std"].values for b in train_bars_list
                 if "vwap_deviation_std" in b.columns]
    if not vwap_arrs:
        return None
    vwap = np.concatenate(vwap_arrs)
    n = min(len(spread), len(depth), len(vwap))
    spread, depth, vwap = spread[:n], depth[:n], vwap[:n]
    mask = np.isfinite(spread) & np.isfinite(depth) & np.isfinite(vwap)
    spread, depth, vwap = spread[mask], depth[mask], vwap[mask]
    if len(spread) < 20:
        return None
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
        "vwap_mean": np.mean(vwap), "vwap_std": max(np.std(vwap), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    z_v = (vwap - params["vwap_mean"]) / params["vwap_std"]
    composite = (z_s + z_d + z_v) / 3.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def apply_signal_2f(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply 2-feature composite signal."""
    bars = bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    bars["composite"] = (z_s + z_d) / 2.0
    bars["direction"] = 0
    bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
    bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    return bars


def apply_signal_3f(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply 3-feature composite signal (spread + depth + vwap_deviation)."""
    bars = bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    z_v = (bars["vwap_deviation_std"] - params["vwap_mean"]) / params["vwap_std"]
    bars["composite"] = (z_s + z_d + z_v) / 3.0
    bars["direction"] = 0
    bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
    bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    return bars


# ── PnL computation ───────────────────────────────────────────────────────

def compute_trades(bars: pd.DataFrame, horizon: int, fee_bps: float) -> dict:
    """Compute trade-level PnL for a test date.

    Returns dict with gross/net stats or None if no trades.
    """
    prices = bars["midprice_last"].values
    directions = bars["direction"].values
    n = len(prices)

    trade_pnls_gross = []
    trade_pnls_net = []

    for i in range(n - horizon):
        d = directions[i]
        if d == 0:
            continue
        if prices[i] <= 0 or np.isnan(prices[i]) or np.isnan(prices[i + horizon]):
            continue
        ret_bps = (prices[i + horizon] - prices[i]) / prices[i] * 1e4
        gross = d * ret_bps
        net = gross - fee_bps
        trade_pnls_gross.append(gross)
        trade_pnls_net.append(net)

    if not trade_pnls_gross:
        return {"n_trades": 0, "gross_bps": 0.0, "net_bps": -fee_bps,
                "total_net_bps": -fee_bps, "std_bps": 0.0,
                "win_rate": 0.0}

    gross_arr = np.array(trade_pnls_gross)
    net_arr = np.array(trade_pnls_net)

    return {
        "n_trades": len(gross_arr),
        "gross_bps": float(np.mean(gross_arr)),
        "net_bps": float(np.mean(net_arr)),
        "total_net_bps": float(np.sum(net_arr)),
        "std_bps": float(np.std(gross_arr)),
        "win_rate": float(np.mean(net_arr > 0)),
    }


# ── Main backtest ─────────────────────────────────────────────────────────

def _aggregate_horizon(daily_results: list[dict], fee_bps: float) -> dict:
    """Aggregate daily trade results into summary stats."""
    total_trades = sum(r["n_trades"] for r in daily_results)
    all_total_net = [r["total_net_bps"] for r in daily_results]
    n_oos = len(daily_results)
    n_positive = sum(1 for r in daily_results if r["total_net_bps"] > 0)

    if total_trades > 0:
        weighted_gross = sum(r["gross_bps"] * r["n_trades"] for r in daily_results) / total_trades
        weighted_net = sum(r["net_bps"] * r["n_trades"] for r in daily_results) / total_trades
        weighted_std = sum(r["std_bps"] * r["n_trades"] for r in daily_results if r["n_trades"] > 0) / total_trades
        total_pnl = sum(all_total_net)
        daily_pnl_arr = np.array(all_total_net)
        daily_std = np.std(daily_pnl_arr)
        sharpe = (np.mean(daily_pnl_arr) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    else:
        weighted_gross, weighted_net, weighted_std = 0.0, -fee_bps, 0.0
        total_pnl, sharpe = -fee_bps * n_oos, 0.0

    return {
        "n_oos_dates": n_oos,
        "n_trades": total_trades,
        "gross_bps": round(weighted_gross, 3),
        "net_bps": round(weighted_net, 3),
        "sharpe_ann": round(sharpe, 2),
        "daily_win_rate": round(n_positive / n_oos, 2) if n_oos > 0 else 0.0,
        "total_pnl_bps": round(total_pnl, 1),
        "std_bps": round(weighted_std, 3),
        "daily_pnl": {r["date"]: round(r["total_net_bps"], 2) for r in daily_results},
    }


def run_backtest(
    data_dir: Path,
    symbols: list[str],
    fee_model: str = "binance_vip9",
    feature_modes: list[str] | None = None,
    horizons_bars: list[int] | None = None,
):
    fee_bps = FEE_MODELS[fee_model]
    if feature_modes is None:
        feature_modes = ["2f"]
    if horizons_bars is None:
        horizons_bars = HORIZONS_BARS_DEFAULT

    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )
    horizons_min = [h * BAR_SECONDS // 60 for h in horizons_bars]
    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")
    print(f"Fee model: {fee_model} ({fee_bps} bps RT)")
    print(f"Features: {', '.join(feature_modes)}")
    print(f"Horizons: {horizons_min} min\n")

    results = {}

    for symbol in symbols:
        print(f"═══ {symbol} ═══")

        date_bars: list[tuple[str, pd.DataFrame]] = []
        for date_str in all_dates:
            ticks = load_date(data_dir, date_str, symbol)
            if ticks is None or len(ticks) < 100:
                continue
            bars = aggregate_to_bars(ticks)
            if len(bars) >= MIN_BARS_PER_DATE:
                date_bars.append((date_str, bars))
                print(f"  {date_str}: {len(bars)} bars ({len(bars)*5:.0f} min)")
            else:
                print(f"  {date_str}: {len(bars)} bars — skipped (< {MIN_BARS_PER_DATE})")

        print(f"  → {len(date_bars)} usable dates\n")

        if len(date_bars) < TRAIN_WINDOW + 1:
            print(f"  Not enough dates for walk-forward (need {TRAIN_WINDOW + 1})\n")
            continue

        sym_results = {}

        for mode in feature_modes:
            if mode == "3f":
                compute_params = compute_zscore_params_3f
                apply_sig = apply_signal_3f
            else:
                compute_params = compute_zscore_params_2f
                apply_sig = apply_signal_2f

            for horizon in horizons_bars:
                horizon_min = horizon * BAR_SECONDS // 60
                daily_results = []

                for i in range(TRAIN_WINDOW, len(date_bars)):
                    train_bar_list = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
                    test_date_str, test_bars = date_bars[i]

                    params = compute_params(train_bar_list)
                    if params is None:
                        continue
                    scored = apply_sig(test_bars, params)
                    trades = compute_trades(scored, horizon, fee_bps)
                    trades["date"] = test_date_str
                    daily_results.append(trades)

                if not daily_results:
                    continue

                summary = _aggregate_horizon(daily_results, fee_bps)
                key = f"{mode}_{horizon_min}min"
                sym_results[key] = summary

                tag = "▶" if summary["net_bps"] > 0 else "✗"
                print(f"  {tag} {mode} {horizon_min:>3}min | OOS {summary['n_oos_dates']}d | "
                      f"{summary['n_trades']} trades | "
                      f"gross {summary['gross_bps']:+.2f} net {summary['net_bps']:+.2f} bps | "
                      f"Sharpe {summary['sharpe_ann']:+.1f} | "
                      f"WR {int(summary['daily_win_rate'] * summary['n_oos_dates'])}/{summary['n_oos_dates']} | "
                      f"PnL {summary['total_pnl_bps']:+.0f} bps")

        results[symbol] = sym_results
        print()

    return results, all_dates, date_bars


def compare_with_original(results: dict, orig_path: Path):
    """Compare 2f results against the original report."""
    if not orig_path.exists():
        return

    with open(orig_path) as f:
        orig = json.load(f)

    print("\n═══ Comparison vs Original (2026-05-20) — 2f baseline ═══\n")
    print(f"  {'Symbol':<6} {'Horizon':<10} {'Orig Net':>10} {'New Net':>10} {'Orig Sharpe':>12} {'New Sharpe':>12} {'Orig OOS':>9} {'New OOS':>9}")
    print("  " + "─" * 80)

    for symbol in ["BTC", "ETH", "SOL"]:
        if symbol not in results or symbol not in orig.get("composite", {}):
            continue
        for horizon_key in ["10min", "25min", "50min"]:
            if horizon_key not in orig["composite"][symbol]:
                continue
            o = orig["composite"][symbol][horizon_key]
            new_key = f"2f_{horizon_key}"
            n = results[symbol].get(new_key, {})
            if not n:
                continue
            print(f"  {symbol:<6} {new_key:<10} "
                  f"{o['net_bps']:>+10.2f} {n['net_bps']:>+10.2f} "
                  f"{o['sharpe_ann']:>+12.1f} {n['sharpe_ann']:>+12.1f} "
                  f"{o['n_dates']:>9d} {n['n_oos_dates']:>9d}")


def compare_2f_vs_3f(results: dict):
    """Print side-by-side 2f vs 3f comparison."""
    print("\n═══ 2-Feature vs 3-Feature Comparison ═══\n")
    print(f"  {'Symbol':<6} {'Horizon':>7} {'2f Net':>8} {'3f Net':>8} {'2f Sharpe':>10} {'3f Sharpe':>10} {'Δ Net':>7} {'Δ Sharpe':>9}")
    print("  " + "─" * 70)

    for symbol in ["BTC", "ETH", "SOL"]:
        if symbol not in results:
            continue
        for key_2f in sorted(k for k in results[symbol] if k.startswith("2f_")):
            horizon = key_2f.removeprefix("2f_")
            key_3f = f"3f_{horizon}"
            r2 = results[symbol].get(key_2f)
            r3 = results[symbol].get(key_3f)
            if not r2 or not r3:
                continue
            d_net = r3["net_bps"] - r2["net_bps"]
            d_sharpe = r3["sharpe_ann"] - r2["sharpe_ann"]
            tag = "+" if d_net > 0 else " "
            print(f"  {symbol:<6} {horizon:>7} "
                  f"{r2['net_bps']:>+8.2f} {r3['net_bps']:>+8.2f} "
                  f"{r2['sharpe_ann']:>+10.1f} {r3['sharpe_ann']:>+10.1f} "
                  f"{tag}{d_net:>+6.2f} {d_sharpe:>+9.1f}")


def main():
    parser = argparse.ArgumentParser(description="MF Liquidity Signal Walk-Forward Backtest")
    parser.add_argument("--data-dir", default="data/features", help="Features directory")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--fee-model", choices=list(FEE_MODELS.keys()),
                        default="binance_vip9")
    parser.add_argument("--features", choices=["2", "3", "both"], default="both",
                        help="Feature set: 2 (spread+depth), 3 (+vwap_deviation), both")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Horizons in minutes (default: 10 25 50 100 200)")
    parser.add_argument("--save", action="store_true", help="Save JSON report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Map feature flag to mode list
    if args.features == "2":
        feature_modes = ["2f"]
    elif args.features == "3":
        feature_modes = ["3f"]
    else:
        feature_modes = ["2f", "3f"]

    # Map minute horizons to bar counts
    if args.horizons:
        horizons_bars = [h * 60 // BAR_SECONDS for h in args.horizons]
    else:
        horizons_bars = HORIZONS_BARS_DEFAULT

    results, all_dates, _ = run_backtest(
        data_dir, args.symbols, args.fee_model, feature_modes, horizons_bars,
    )

    # Compare with original 2f report
    if "2f" in feature_modes:
        orig_path = Path("reports/best__mf_liquidity_signal.json")
        compare_with_original(results, orig_path)

    # Compare 2f vs 3f
    if "2f" in feature_modes and "3f" in feature_modes:
        compare_2f_vs_3f(results)

    if args.save:
        report = {
            "title": "MF Liquidity Signal: Walk-Forward Backtest (2f vs 3f)",
            "generated": datetime.now(timezone.utc).isoformat(),
            "data": {
                "dates": all_dates,
                "n_dates": len(all_dates),
                "symbols": args.symbols,
                "timeframe": "5min bars",
                "train_window": TRAIN_WINDOW,
                "horizons_min": [h * BAR_SECONDS // 60 for h in horizons_bars],
            },
            "feature_modes": feature_modes,
            "fee_model": args.fee_model,
            "fee_bps_rt": FEE_MODELS[args.fee_model],
            "results": results,
        }
        out_path = Path("reports/mf_liquidity_updated.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()

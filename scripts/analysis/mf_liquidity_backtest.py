#!/usr/bin/env python3
"""
MF Liquidity Signal — Walk-Forward Backtest

Replicates the spread+depth composite signal from reports/best__mf_liquidity_signal.json
on all available data. Designed to be re-run as new dates accumulate.

Signal: composite = (zscore(spread_bps_last) + zscore(depth_5_std)) / 2
Entry:  long when composite > P80, short when composite < P20
Exit:   fixed horizon (default 50min = 10 bars)
Fees:   Binance VIP9 1.61 bps RT (default)

Walk-forward: train on prior `train_window` dates, test on next date.
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
HORIZONS_BARS = [2, 5, 10]  # 10min, 25min, 50min
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

    dfs = []
    for f in files:
        try:
            tbl = pq.read_table(
                str(f),
                columns=["timestamp_ns", "symbol", "raw_midprice",
                         "raw_spread_bps", "raw_ask_depth_5"],
            )
            df = tbl.to_pandas()
            df = df[df["symbol"] == symbol].copy()
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

    bars = ticks.groupby("bar_id").agg(
        timestamp_ns=("timestamp_ns", "first"),
        midprice_last=("raw_midprice", "last"),
        spread_bps_last=("raw_spread_bps", "last"),
        depth_5_std=("raw_ask_depth_5", "std"),
        n_ticks=("raw_midprice", "count"),
    ).reset_index(drop=True)

    # Drop bars with too few ticks (partial bars at edges)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    # Fill NaN std (single-tick bars already dropped)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    return bars


# ── Signal construction ───────────────────────────────────────────────────

def compute_zscore_params(train_bars_list: list[pd.DataFrame]) -> dict:
    """Compute z-score mean/std and composite percentile thresholds from training dates."""
    all_spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    all_depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])

    # Remove NaN/inf
    mask = np.isfinite(all_spread) & np.isfinite(all_depth)
    all_spread = all_spread[mask]
    all_depth = all_depth[mask]

    params = {
        "spread_mean": np.mean(all_spread),
        "spread_std": max(np.std(all_spread), 1e-10),
        "depth_mean": np.mean(all_depth),
        "depth_std": max(np.std(all_depth), 1e-10),
    }

    # Compute composite on training data to get thresholds
    z_spread = (all_spread - params["spread_mean"]) / params["spread_std"]
    z_depth = (all_depth - params["depth_mean"]) / params["depth_std"]
    composite = (z_spread + z_depth) / 2.0

    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)

    return params


def apply_signal(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute composite signal and entry direction on test bars."""
    bars = bars.copy()
    z_spread = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_depth = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    bars["composite"] = (z_spread + z_depth) / 2.0

    # Direction: +1 long, -1 short, 0 no trade
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

def run_backtest(data_dir: Path, symbols: list[str], fee_model: str = "binance_vip9"):
    fee_bps = FEE_MODELS[fee_model]

    # Discover dates
    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )
    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")
    print(f"Fee model: {fee_model} ({fee_bps} bps RT)")
    print(f"Horizons: {[h * BAR_SECONDS // 60 for h in HORIZONS_BARS]} min")
    print()

    results = {}

    for symbol in symbols:
        print(f"═══ {symbol} ═══")

        # Load and aggregate all dates
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

        # Walk-forward
        sym_results = {}
        for horizon in HORIZONS_BARS:
            horizon_min = horizon * BAR_SECONDS // 60
            daily_results = []

            for i in range(TRAIN_WINDOW, len(date_bars)):
                train_dates = date_bars[i - TRAIN_WINDOW:i]
                test_date_str, test_bars = date_bars[i]

                # Train: compute z-score params + thresholds
                train_bar_list = [b for _, b in train_dates]
                params = compute_zscore_params(train_bar_list)

                # Test: apply signal and compute PnL
                test_bars = apply_signal(test_bars, params)
                trades = compute_trades(test_bars, horizon, fee_bps)
                trades["date"] = test_date_str
                daily_results.append(trades)

            # Aggregate
            total_trades = sum(r["n_trades"] for r in daily_results)
            all_net = [r["net_bps"] for r in daily_results if r["n_trades"] > 0]
            all_total_net = [r["total_net_bps"] for r in daily_results]
            n_oos = len(daily_results)
            n_positive = sum(1 for r in daily_results if r["total_net_bps"] > 0)

            if total_trades > 0 and len(all_net) > 0:
                # Weight average by trade count
                weighted_gross = sum(r["gross_bps"] * r["n_trades"] for r in daily_results) / total_trades
                weighted_net = sum(r["net_bps"] * r["n_trades"] for r in daily_results) / total_trades
                weighted_std = sum(r["std_bps"] * r["n_trades"] for r in daily_results if r["n_trades"] > 0) / total_trades
                total_pnl = sum(r["total_net_bps"] for r in daily_results)

                # Sharpe: annualize from daily PnL
                daily_pnl_arr = np.array(all_total_net)
                daily_mean = np.mean(daily_pnl_arr)
                daily_std = np.std(daily_pnl_arr)
                sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            else:
                weighted_gross = 0.0
                weighted_net = -fee_bps
                weighted_std = 0.0
                total_pnl = -fee_bps * n_oos
                sharpe = 0.0

            sym_results[f"{horizon_min}min"] = {
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

            # Print
            tag = "▶" if weighted_net > 0 else "✗"
            print(f"  {tag} {horizon_min}min | OOS {n_oos}d | {total_trades} trades | "
                  f"gross {weighted_gross:+.2f} net {weighted_net:+.2f} bps | "
                  f"Sharpe {sharpe:+.1f} | WR {n_positive}/{n_oos} | "
                  f"PnL {total_pnl:+.0f} bps")

        results[symbol] = sym_results
        print()

    return results, all_dates, date_bars


def compare_with_original(results: dict, orig_path: Path):
    """Compare new results against the original report."""
    if not orig_path.exists():
        return

    with open(orig_path) as f:
        orig = json.load(f)

    print("\n═══ Comparison vs Original (2026-05-20) ═══\n")
    print(f"  {'Symbol':<6} {'Horizon':<8} {'Orig Net':>10} {'New Net':>10} {'Orig Sharpe':>12} {'New Sharpe':>12} {'Orig OOS':>9} {'New OOS':>9}")
    print("  " + "─" * 80)

    for symbol in ["BTC", "ETH", "SOL"]:
        if symbol not in results or symbol not in orig.get("composite", {}):
            continue
        for horizon_key in ["10min", "25min", "50min"]:
            orig_key = horizon_key
            if orig_key not in orig["composite"][symbol]:
                continue
            o = orig["composite"][symbol][orig_key]
            n = results[symbol].get(horizon_key, {})
            if not n:
                continue

            print(f"  {symbol:<6} {horizon_key:<8} "
                  f"{o['net_bps']:>+10.2f} {n['net_bps']:>+10.2f} "
                  f"{o['sharpe_ann']:>+12.1f} {n['sharpe_ann']:>+12.1f} "
                  f"{o['n_dates']:>9d} {n['n_oos_dates']:>9d}")

    # Highlight new OOS dates
    orig_dates = set(orig["data"]["dates"])
    for symbol in results:
        for horizon_key, h_result in results[symbol].items():
            new_dates = [d for d in h_result["daily_pnl"] if d not in orig_dates]
            if new_dates and horizon_key == "50min":
                print(f"\n  New OOS dates for {symbol} {horizon_key}:")
                for d in new_dates:
                    pnl = h_result["daily_pnl"][d]
                    tag = "+" if pnl > 0 else " "
                    print(f"    {d}: {tag}{pnl:.1f} bps")


def main():
    parser = argparse.ArgumentParser(description="MF Liquidity Signal Walk-Forward Backtest")
    parser.add_argument("--data-dir", default="data/features", help="Features directory")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--fee-model", choices=list(FEE_MODELS.keys()),
                        default="binance_vip9")
    parser.add_argument("--save", action="store_true", help="Save JSON report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    results, all_dates, _ = run_backtest(data_dir, args.symbols, args.fee_model)

    # Compare with original
    orig_path = Path("reports/best__mf_liquidity_signal.json")
    compare_with_original(results, orig_path)

    if args.save:
        report = {
            "title": "MF Liquidity Signal: Walk-Forward Backtest (updated)",
            "generated": datetime.now(timezone.utc).isoformat(),
            "data": {
                "dates": all_dates,
                "n_dates": len(all_dates),
                "symbols": args.symbols,
                "timeframe": "5min bars",
                "train_window": TRAIN_WINDOW,
            },
            "fee_model": args.fee_model,
            "fee_bps_rt": FEE_MODELS[args.fee_model],
            "composite": results,
        }
        out_path = Path("reports/mf_liquidity_updated.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()

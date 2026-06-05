#!/usr/bin/env python3
"""
Daily Paper Trader — 6-hour OOS snapshot with rolling accumulation.

Runs the 5 winning algorithms on today's data (or a specified date) using
the prior 3 days as training. Stores results as 6h__YYYY-MM-DD.json.
Rolling 7-day and 30-day stats are computed across all daily files.

Usage:
  python scripts/alpha/paper_trader_daily.py                  # test today
  python scripts/alpha/paper_trader_daily.py --date 2026-05-23
  python scripts/alpha/paper_trader_daily.py --min-hours 4    # require 4h minimum
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

from algorithms import get_algorithm, discover_all
from alpha.paper_trader_generic import (
    ALGO_CONFIG,
    BAR_SECONDS,
    COST_PRESETS,
    HORIZON_BARS,
    MIN_BARS_PER_DATE,
    TRAIN_WINDOW,
    aggregate_to_bars,
    apply_signal,
    compute_params,
    discover_dates,
    generate_trades,
    load_date_ticks,
    summarize_day,
)
from backtest.costs import CostModel

# ── Config ────────────────────────────────────────────────────────────────

DAILY_ALGOS = [
    "jump_detector", "optimal_entry", "funding_reversion",
    "surprise_signal", "bipower_jump", "cascade_probability", "convolver",
    "entropy_momentum", "hawkes_intensity", "kalman_imbalance",
    "oi_divergence", "propagator", "regime_gated", "spread_decomp",
    "switching_ou", "trade_through", "vpin_regime", "weighted_ofi",
    "change_point_detector",
    "momentum_continuation",
    "regime_state_machine",
    "mean_reversion_detector",
    "meta_labeling",
    "regime_conditioned_lgbm",
    "knn_retrieval",
]
SURPRISE_ALGO = None  # now included in DAILY_ALGOS
SYMBOLS = ["BTC", "ETH", "SOL"]
MIN_HOURS = 4.0
REPORTS_DIR = ROOT / "reports"
DAILY_PREFIX = "6h__"

# Default: Binance VIP9 taker (matches paper_trader_generic experiment reports)
from utils.costs import binance_vip9_rt_bps as _vip9_bps
DEFAULT_COST = CostModel(fee_bps=_vip9_bps() / 2, slippage_bps=0.0)  # half RT, from config/costs.toml


# ── 3f liquidity signal (inline — avoids importing full paper_trader.py) ──

def run_3f_liquidity(data_dir: Path, train_dates: list[str],
                     test_date: str, symbols: list[str],
                     cost_model: CostModel = DEFAULT_COST) -> dict:
    """Run the 3-feature liquidity composite on one test date."""
    from data.features import load_features

    results = {}
    want_cols = ["timestamp_ns", "symbol", "raw_midprice",
                 "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation"]

    for symbol in symbols:
        # Load training data
        train_vals = []
        for d in train_dates:
            df = load_features(
                symbols=[symbol], date_range=(d, d),
                columns=want_cols, data_dir=data_dir, validate=False,
            )
            if df.empty:
                continue
            bar_ns = BAR_SECONDS * 1_000_000_000
            df["bar_id"] = df["timestamp_ns"].values // bar_ns
            bars = df.groupby("bar_id").agg(
                spread=("raw_spread_bps", "mean"),
                depth=("raw_ask_depth_5", "mean"),
                vwap_dev=("flow_vwap_deviation", "mean"),
                midprice=("raw_midprice", "last"),
            ).reset_index(drop=True)
            if len(bars) >= MIN_BARS_PER_DATE:
                train_vals.append(bars)

        if not train_vals:
            continue

        # Compute training stats and IC-based weights
        all_train = __import__("pandas").concat(train_vals, ignore_index=True)
        stats = {}
        factors = ["spread", "depth", "vwap_dev"]
        for col in factors:
            vals = all_train[col].dropna()
            stats[col] = {"mean": float(vals.mean()), "std": float(max(vals.std(), 1e-10))}

        # Estimate factor weights via rank IC on training forward returns
        from scipy.stats import spearmanr
        fwd_ret = all_train["midprice"].pct_change(HORIZON_BARS).shift(-HORIZON_BARS).values
        ic_abs = []
        for col in factors:
            z = (all_train[col].values - stats[col]["mean"]) / stats[col]["std"]
            mask = np.isfinite(z) & np.isfinite(fwd_ret)
            if mask.sum() > 30:
                rho, _ = spearmanr(z[mask], fwd_ret[mask])
                ic_abs.append(max(abs(rho), 0.0))
            else:
                ic_abs.append(1.0 / len(factors))
        ic_sum = sum(ic_abs) or 1.0
        weights = {col: ic / ic_sum for col, ic in zip(factors, ic_abs)}

        # Load test data
        test_df = load_features(
            symbols=[symbol], date_range=(test_date, test_date),
            columns=want_cols, data_dir=data_dir, validate=False,
        )
        if test_df.empty:
            continue

        bar_ns = BAR_SECONDS * 1_000_000_000
        test_df["bar_id"] = test_df["timestamp_ns"].values // bar_ns
        test_bars = test_df.groupby("bar_id").agg(
            spread=("raw_spread_bps", "mean"),
            depth=("raw_ask_depth_5", "mean"),
            vwap_dev=("flow_vwap_deviation", "mean"),
            midprice=("raw_midprice", "last"),
            n_ticks=("raw_midprice", "count"),
        ).reset_index(drop=True)
        test_bars = test_bars[test_bars["n_ticks"] >= 10].reset_index(drop=True)

        if len(test_bars) < MIN_BARS_PER_DATE:
            continue

        # IC-weighted z-score composite
        composite = np.zeros(len(test_bars))
        for col in factors:
            z = (test_bars[col].values - stats[col]["mean"]) / stats[col]["std"]
            composite += weights[col] * z

        # Training composite for percentiles (same weights)
        train_composite = np.zeros(len(all_train))
        for col in factors:
            z = (all_train[col].values - stats[col]["mean"]) / stats[col]["std"]
            train_composite += weights[col] * z
        p20 = float(np.nanpercentile(train_composite, 20))
        p80 = float(np.nanpercentile(train_composite, 80))

        # Generate trades
        prices = test_bars["midprice"].values
        n = len(prices)
        rt_cost = cost_model.round_trip_cost_bps
        trades_pnl = []
        for i in range(n - HORIZON_BARS):
            if composite[i] <= p20:
                d = 1
            elif composite[i] >= p80:
                d = -1
            else:
                continue
            entry_p, exit_p = prices[i], prices[i + HORIZON_BARS]
            if entry_p <= 0 or not np.isfinite(entry_p) or not np.isfinite(exit_p):
                continue
            gross = d * (exit_p - entry_p) / entry_p * 1e4
            trades_pnl.append(gross - rt_cost)

        if trades_pnl:
            pnl_arr = np.array(trades_pnl)
            results[symbol] = {
                "trades": len(trades_pnl),
                "total_net_bps": round(float(np.sum(pnl_arr)), 2),
                "net_bps_per_trade": round(float(np.mean(pnl_arr)), 3),
                "win_rate": round(float(np.mean(pnl_arr > 0)), 3),
            }
        else:
            results[symbol] = {"trades": 0, "total_net_bps": 0.0,
                               "net_bps_per_trade": 0.0, "win_rate": 0.0}

    return results


# ── Generic algorithm runner (single test date) ──────────────────────────

def run_algo_single_date(algo_name: str, data_dir: Path,
                         train_dates: list[str], test_date: str,
                         symbols: list[str],
                         cost_model: CostModel = DEFAULT_COST) -> dict:
    """Run one algorithm on one test date with given training dates."""
    config = ALGO_CONFIG.get(algo_name)
    if config is None:
        return {}

    algo = get_algorithm(algo_name)
    required = algo.required_columns()
    primary = config["primary"]
    polarity = config["polarity"]
    agg_method = config["bar_agg"]

    results = {}
    for symbol in symbols:
        # Load training bars
        train_bar_list = []
        for d in train_dates:
            ticks = load_date_ticks(data_dir, d, symbol, required)
            if ticks is None or len(ticks) < 200:
                continue
            missing = [c for c in required if c not in ticks.columns]
            if missing:
                continue
            try:
                features = algo.run_batch(ticks)
                algo.reset()
            except Exception:
                continue
            if primary not in features.columns:
                continue
            bars = aggregate_to_bars(ticks, features, primary, agg_method)
            if len(bars) >= MIN_BARS_PER_DATE:
                train_bar_list.append(bars)

        if not train_bar_list:
            continue

        # Load test date
        ticks = load_date_ticks(data_dir, test_date, symbol, required)
        if ticks is None or len(ticks) < 200:
            continue
        missing = [c for c in required if c not in ticks.columns]
        if missing:
            continue
        try:
            features = algo.run_batch(ticks)
            algo.reset()
        except Exception:
            continue
        if primary not in features.columns:
            continue
        test_bars = aggregate_to_bars(ticks, features, primary, agg_method)
        if len(test_bars) < MIN_BARS_PER_DATE:
            continue

        params = compute_params(train_bar_list, polarity)
        if params is None:
            continue

        scored = apply_signal(test_bars, params)
        trades = generate_trades(scored, test_date, symbol, cost_model=cost_model)
        summary = summarize_day(trades, test_date, symbol)

        results[symbol] = {
            "trades": summary.n_trades,
            "total_net_bps": summary.total_net_bps,
            "net_bps_per_trade": summary.net_bps,
            "win_rate": summary.win_rate,
        }

    return results


# ── Rolling stats across daily files ─────────────────────────────────────

def compute_rolling_stats(reports_dir: Path, current_date: str) -> dict:
    """Compute rolling 7-day and 30-day stats from all 6h__ files."""
    daily_files = sorted(reports_dir.glob(f"{DAILY_PREFIX}*.json"))
    if not daily_files:
        return {}

    all_days = []
    for f in daily_files:
        try:
            with open(f) as fh:
                day = json.load(fh)
            all_days.append(day)
        except (json.JSONDecodeError, KeyError):
            continue

    if not all_days:
        return {}

    rolling = {}
    for window_name, window_size in [("7d", 7), ("30d", 30)]:
        recent = all_days[-window_size:]
        if not recent:
            continue

        algo_stats = {}
        # Collect all algorithm names
        algo_names = set()
        for day in recent:
            algo_names.update(day.get("algorithms", {}).keys())

        for algo in sorted(algo_names):
            sym_pnl = {}
            for sym in SYMBOLS:
                daily_pnl = []
                for day in recent:
                    algo_data = day.get("algorithms", {}).get(algo, {})
                    sym_data = algo_data.get(sym, {})
                    daily_pnl.append(sym_data.get("total_net_bps", 0.0))

                pnl_arr = np.array(daily_pnl)
                total = float(np.sum(pnl_arr))
                from utils.metrics import sharpe_daily
                sharpe = sharpe_daily(pnl_arr)
                sym_pnl[sym] = {
                    "total_bps": round(total, 1),
                    "sharpe": round(sharpe, 2),
                    "days": len(recent),
                }

            algo_stats[algo] = sym_pnl
        rolling[window_name] = algo_stats

    return rolling


# ── Terminal report ──────────────────────────────────────────────────────

def print_report(report: dict):
    """Print compact terminal report."""
    date = report["date"]
    hours = report["hours_collected"]
    bars = report["bars_per_symbol"]

    print(f"\n{'=' * 70}")
    print(f"  NAT Daily Report -- {date}")
    print(f"  Data: {hours:.1f}h collected, {bars} bars/symbol")
    print(f"  Cost: {report.get('cost_bps_rt', '?')} bps RT")
    print(f"{'=' * 70}")

    print(f"\n  {'Algorithm':<20s} {'BTC':>12s} {'ETH':>12s} {'SOL':>12s} {'Trades':>8s}")
    print(f"  {'-' * 64}")

    algos = report.get("algorithms", {})
    for algo_name, sym_data in algos.items():
        cols = []
        total_trades = 0
        for sym in SYMBOLS:
            d = sym_data.get(sym, {})
            pnl = d.get("total_net_bps", 0.0)
            trades = d.get("trades", 0)
            total_trades += trades
            cols.append(f"{pnl:+8.1f} bps")
        print(f"  {algo_name:<20s} {cols[0]:>12s} {cols[1]:>12s} {cols[2]:>12s} {total_trades:>8d}")

    # Rolling stats
    rolling = report.get("rolling", {})
    for window_name in ["7d", "30d"]:
        window_data = rolling.get(window_name)
        if not window_data:
            continue
        # Check if there's actual data
        first_algo = next(iter(window_data.values()), {})
        first_sym = next(iter(first_algo.values()), {})
        days = first_sym.get("days", 0)
        if days < 2:
            continue

        print(f"\n  Rolling {window_name} Sharpe ({days} days):")
        print(f"  {'Algorithm':<20s} {'BTC':>8s} {'ETH':>8s} {'SOL':>8s}")
        print(f"  {'-' * 46}")
        for algo_name, sym_data in window_data.items():
            cols = []
            for sym in SYMBOLS:
                s = sym_data.get(sym, {}).get("sharpe", 0.0)
                cols.append(f"{s:+5.1f}")
            print(f"  {algo_name:<20s} {cols[0]:>8s} {cols[1]:>8s} {cols[2]:>8s}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily 6-hour OOS Paper Trading")
    parser.add_argument("--date", type=str, default=None,
                        help="Test date (YYYY-MM-DD). Default: latest available.")
    parser.add_argument("--data-dir", default="data/features",
                        help="Feature data directory")
    parser.add_argument("--min-hours", type=float, default=MIN_HOURS,
                        help=f"Minimum hours of data required (default: {MIN_HOURS})")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Symbols to test")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save report file")
    parser.add_argument("--cost-mode", choices=["binance_vip9", "taker", "maker", "config"],
                        default="binance_vip9",
                        help="Cost model (default: binance_vip9 = 1.61 bps RT)")
    args = parser.parse_args()

    cost_presets = {
        "binance_vip9": DEFAULT_COST,
        **COST_PRESETS,
    }
    cost_model = cost_presets[args.cost_mode]

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Auto-discover algorithms
    discover_all()

    all_dates = discover_dates(data_dir)
    if len(all_dates) < TRAIN_WINDOW + 1:
        print(f"Need at least {TRAIN_WINDOW + 1} dates, have {len(all_dates)}")
        sys.exit(1)

    # Determine test date
    if args.date:
        test_date = args.date
        if test_date not in all_dates:
            print(f"Date {test_date} not found in data. Available: {all_dates[-5:]}")
            sys.exit(1)
    else:
        test_date = all_dates[-1]

    # Find training dates (the 3 dates before test_date)
    test_idx = all_dates.index(test_date)
    if test_idx < TRAIN_WINDOW:
        print(f"Not enough prior dates for training. Test date {test_date} is at index {test_idx}, need {TRAIN_WINDOW}")
        sys.exit(1)
    train_dates = all_dates[test_idx - TRAIN_WINDOW:test_idx]

    # Check data volume for test date
    from data.features import load_features
    test_check = load_features(
        symbols=[args.symbols[0]], date_range=(test_date, test_date),
        columns=["timestamp_ns", "symbol"], data_dir=data_dir, validate=False,
    )
    if test_check.empty:
        print(f"No data for {test_date}")
        sys.exit(1)

    ts = test_check["timestamp_ns"].values
    hours = (ts[-1] - ts[0]) / 1e9 / 3600
    n_bars = int((ts[-1] - ts[0]) / 1e9 / BAR_SECONDS)

    if hours < args.min_hours:
        print(f"Only {hours:.1f}h of data for {test_date} (minimum: {args.min_hours}h)")
        sys.exit(1)

    print(f"Test date: {test_date} ({hours:.1f}h, ~{n_bars} bars)")
    print(f"Training: {train_dates}")
    print(f"Symbols: {args.symbols}")
    print(f"Cost: {cost_model.round_trip_cost_bps:.2f} bps RT ({args.cost_mode})")

    # Run all algorithms
    t0 = time.time()
    algo_results = {}

    # 1. 3f liquidity
    print("\n  Running 3f_liquidity...")
    r = run_3f_liquidity(data_dir, train_dates, test_date, args.symbols, cost_model=cost_model)
    if r:
        algo_results["3f_liquidity"] = r

    # 2. Generic algorithms
    for algo_name in DAILY_ALGOS:
        print(f"  Running {algo_name}...")
        r = run_algo_single_date(algo_name, data_dir, train_dates, test_date, args.symbols, cost_model=cost_model)
        if r:
            algo_results[algo_name] = r

    # 3. Surprise signal (now included in DAILY_ALGOS, skip if None)
    if SURPRISE_ALGO:
        print(f"  Running {SURPRISE_ALGO}...")
        r = run_algo_single_date(SURPRISE_ALGO, data_dir, train_dates, test_date, args.symbols, cost_model=cost_model)
        if r:
            algo_results[SURPRISE_ALGO] = r

    elapsed = time.time() - t0

    # Build report
    report = {
        "date": test_date,
        "generated": datetime.now(timezone.utc).isoformat(),
        "hours_collected": round(hours, 1),
        "bars_per_symbol": n_bars,
        "train_dates": train_dates,
        "cost_bps_rt": cost_model.round_trip_cost_bps,
        "elapsed_s": round(elapsed, 1),
        "algorithms": algo_results,
    }

    # Rolling stats
    report["rolling"] = compute_rolling_stats(REPORTS_DIR, test_date)

    # Print
    print_report(report)

    # Save
    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / f"{DAILY_PREFIX}{test_date}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"  Saved: {out_path}")

    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main() or 0)

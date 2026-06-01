#!/usr/bin/env python3
"""
Overnight OOS Sweep — run all algorithms across all available dates.

Tests every algorithm on every date that has ≥4h of data, using the prior
3 dates as walk-forward training. Produces a per-date report and a final
aggregate summary with cumulative PnL and Sharpe per algorithm per symbol.

Usage:
  # Full sweep (all dates with enough data):
  nohup python scripts/alpha/overnight_sweep.py > reports/overnight.log 2>&1 &

  # Last N days only:
  python scripts/alpha/overnight_sweep.py --last 7

  # Specific date range:
  python scripts/alpha/overnight_sweep.py --from 2026-05-20 --to 2026-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms import discover_all
from alpha.paper_trader_daily import (
    DAILY_ALGOS,
    DEFAULT_COST,
    SYMBOLS,
    run_3f_liquidity,
    run_algo_single_date,
)
from alpha.paper_trader_generic import (
    BAR_SECONDS,
    TRAIN_WINDOW,
    discover_dates,
)
from backtest.costs import CostModel

REPORTS_DIR = ROOT / "reports"
MIN_HOURS = 4.0


def check_date_hours(data_dir: Path, date: str, symbol: str) -> float:
    """Return hours of data for a date, or 0 if insufficient."""
    from data.features import load_features
    df = load_features(
        symbols=[symbol], date_range=(date, date),
        columns=["timestamp_ns", "symbol"], data_dir=data_dir, validate=False,
    )
    if df.empty:
        return 0.0
    ts = df["timestamp_ns"].values
    return float((ts[-1] - ts[0]) / 1e9 / 3600)


def run_one_date(data_dir: Path, all_dates: list[str], test_date: str,
                 symbols: list[str], cost_model: CostModel) -> dict | None:
    """Run all algorithms on a single test date. Returns report dict or None."""
    test_idx = all_dates.index(test_date)
    if test_idx < TRAIN_WINDOW:
        return None

    train_dates = all_dates[test_idx - TRAIN_WINDOW:test_idx]

    hours = check_date_hours(data_dir, test_date, symbols[0])
    if hours < MIN_HOURS:
        print(f"  [{test_date}] Skipping — only {hours:.1f}h (need {MIN_HOURS}h)")
        return None

    n_bars = int(hours * 3600 / BAR_SECONDS)
    print(f"\n  [{test_date}] {hours:.1f}h, ~{n_bars} bars | train: {train_dates}")

    t0 = time.time()
    algo_results = {}

    # 3f liquidity
    r = run_3f_liquidity(data_dir, train_dates, test_date, symbols, cost_model=cost_model)
    if r:
        algo_results["3f_liquidity"] = r

    # All generic algorithms
    for algo_name in DAILY_ALGOS:
        r = run_algo_single_date(algo_name, data_dir, train_dates, test_date, symbols, cost_model=cost_model)
        if r:
            algo_results[algo_name] = r

    elapsed = time.time() - t0
    print(f"  [{test_date}] Done in {elapsed:.0f}s — {len(algo_results)} algos produced results")

    return {
        "date": test_date,
        "hours": round(hours, 1),
        "bars": n_bars,
        "train_dates": train_dates,
        "elapsed_s": round(elapsed, 1),
        "algorithms": algo_results,
    }


def print_summary(daily_reports: list[dict], symbols: list[str]):
    """Print aggregate summary across all dates."""
    # Collect per-algo, per-symbol daily PnL series
    algo_names = set()
    for report in daily_reports:
        algo_names.update(report["algorithms"].keys())
    algo_names = sorted(algo_names)

    dates = [r["date"] for r in daily_reports]

    print(f"\n{'=' * 80}")
    print(f"  OVERNIGHT SWEEP SUMMARY")
    print(f"  {len(daily_reports)} test dates: {dates[0]} → {dates[-1]}")
    print(f"{'=' * 80}")

    # Build PnL matrix: algo -> symbol -> [daily_bps]
    pnl = {}
    for algo in algo_names:
        pnl[algo] = {}
        for sym in symbols:
            daily_bps = []
            for report in daily_reports:
                algo_data = report["algorithms"].get(algo, {})
                sym_data = algo_data.get(sym, {})
                daily_bps.append(sym_data.get("total_net_bps", 0.0))
            pnl[algo][sym] = np.array(daily_bps)

    # Summary table
    header = f"  {'Algorithm':<24s}"
    for sym in symbols:
        header += f" {'Σ'+sym:>10s} {'SR'+sym:>7s}"
    header += f" {'Σ Total':>10s} {'SR Tot':>7s} {'Win%':>6s}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    rows = []
    for algo in algo_names:
        total_all = 0.0
        all_daily = np.zeros(len(daily_reports))
        cols = []
        for sym in symbols:
            arr = pnl[algo][sym]
            cumul = float(np.sum(arr))
            total_all += cumul
            all_daily += arr
            sr = _sharpe(arr)
            cols.append(f"{cumul:>+10.1f} {sr:>+6.2f}")
        total_sr = _sharpe(all_daily)
        win_pct = float(np.mean(all_daily > 0) * 100)
        line = f"  {algo:<24s} {' '.join(cols)} {total_all:>+10.1f} {total_sr:>+6.2f} {win_pct:>5.0f}%"
        rows.append((total_sr, line))

    # Sort by total Sharpe descending
    rows.sort(key=lambda x: -x[0])
    for _, line in rows:
        print(line)

    # Per-date breakdown
    print(f"\n  Per-date totals (sum across all algos, all symbols):")
    print(f"  {'Date':<14s} {'Hours':>6s} {'Total bps':>10s} {'Algos':>6s}")
    print(f"  {'-' * 40}")
    for report in daily_reports:
        date = report["date"]
        hours = report["hours"]
        total = 0.0
        n_algos = len(report["algorithms"])
        for algo_data in report["algorithms"].values():
            for sym_data in algo_data.values():
                total += sym_data.get("total_net_bps", 0.0)
        print(f"  {date:<14s} {hours:>5.1f}h {total:>+10.1f} {n_algos:>6d}")

    print()


def _sharpe(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from daily PnL array."""
    if len(daily_pnl) < 2:
        return 0.0
    mu = np.mean(daily_pnl)
    sigma = np.std(daily_pnl, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


def main():
    parser = argparse.ArgumentParser(description="Overnight OOS Sweep")
    parser.add_argument("--data-dir", default="data/features",
                        help="Feature data directory")
    parser.add_argument("--last", type=int, default=None,
                        help="Only test the last N dates")
    parser.add_argument("--from", dest="date_from", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--cost-mode", choices=["binance_vip9", "taker", "maker"],
                        default="binance_vip9")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    discover_all()
    all_dates = discover_dates(data_dir)

    if len(all_dates) < TRAIN_WINDOW + 1:
        print(f"Need at least {TRAIN_WINDOW + 1} dates, have {len(all_dates)}")
        sys.exit(1)

    # Determine testable dates
    testable = all_dates[TRAIN_WINDOW:]

    if args.date_from:
        testable = [d for d in testable if d >= args.date_from]
    if args.date_to:
        testable = [d for d in testable if d <= args.date_to]
    if args.last:
        testable = testable[-args.last:]

    if not testable:
        print("No testable dates in range.")
        sys.exit(1)

    cost_model = DEFAULT_COST
    print(f"Overnight OOS Sweep")
    print(f"  Dates to test: {len(testable)} ({testable[0]} → {testable[-1]})")
    print(f"  Algorithms: {len(DAILY_ALGOS) + 1} (18 generic + 3f_liquidity)")
    print(f"  Symbols: {args.symbols}")
    print(f"  Cost: {cost_model.round_trip_cost_bps:.2f} bps RT ({args.cost_mode})")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    daily_reports = []
    t_total = time.time()

    for test_date in testable:
        report = run_one_date(data_dir, all_dates, test_date, args.symbols, cost_model)
        if report is not None:
            daily_reports.append(report)

    elapsed_total = time.time() - t_total
    print(f"\n  Total elapsed: {elapsed_total / 60:.1f} min")

    if not daily_reports:
        print("No dates had sufficient data.")
        sys.exit(1)

    # Print summary
    print_summary(daily_reports, args.symbols)

    # Save full results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"overnight_{testable[0]}_{testable[-1]}.json"
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dates_tested": len(daily_reports),
        "cost_bps_rt": cost_model.round_trip_cost_bps,
        "elapsed_min": round(elapsed_total / 60, 1),
        "daily": daily_reports,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"  Full results saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

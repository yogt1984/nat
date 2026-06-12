#!/usr/bin/env python3
"""
Gauntlet — multi-day OOS sweep across all algorithms.

Subcommands:
  run         Run the sweep (saves incrementally after each date)
  stop        Kill a running gauntlet, print partial results
  report      Print the latest gauntlet report
  report_all  Merge all gauntlet reports into one combined summary

Usage:
  python scripts/alpha/overnight_sweep.py run                    # full sweep
  python scripts/alpha/overnight_sweep.py run --last 7           # last 7 days
  python scripts/alpha/overnight_sweep.py stop                   # kill + report
  python scripts/alpha/overnight_sweep.py report                 # latest report
  python scripts/alpha/overnight_sweep.py report_all             # merge all runs
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

REPORTS_DIR = ROOT / "reports"
PID_FILE = REPORTS_DIR / ".gauntlet.pid"
LATEST_FILE = REPORTS_DIR / "gauntlet_latest.json"
MIN_HOURS = 4.0
SYMBOLS_DEFAULT = ["BTC", "ETH", "SOL"]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def _sharpe(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from daily PnL array."""
    if len(daily_pnl) < 2:
        return 0.0
    mu = np.mean(daily_pnl)
    sigma = np.std(daily_pnl, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


# ── Shared report printing ────────────────────────────────────────────────

def print_summary(daily_reports: list[dict], symbols: list[str], title: str = "GAUNTLET SUMMARY"):
    """Print aggregate summary across all dates."""
    algo_names = set()
    for report in daily_reports:
        algo_names.update(report["algorithms"].keys())
    algo_names = sorted(algo_names)

    dates = [r["date"] for r in daily_reports]
    total_hours = sum(r.get("hours", 0) for r in daily_reports)

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"  {len(daily_reports)} test dates: {dates[0]} -> {dates[-1]}  ({total_hours:.1f}h total)")
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
        header += f" {'S'+sym:>10s} {'SR'+sym:>7s}"
    header += f" {'S Total':>10s} {'SR Tot':>7s} {'Win%':>6s}"
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

    rows.sort(key=lambda x: -x[0])
    for _, line in rows:
        print(line)

    # Per-date breakdown
    print(f"\n  Per-date breakdown:")
    print(f"  {'Date':<14s} {'Hours':>6s} {'Total bps':>10s} {'Algos':>6s}")
    print(f"  {'-' * 40}")
    for report in daily_reports:
        date = report["date"]
        hours = report.get("hours", 0)
        total = 0.0
        n_algos = len(report["algorithms"])
        for algo_data in report["algorithms"].values():
            for sym_data in algo_data.values():
                total += sym_data.get("total_net_bps", 0.0)
        print(f"  {date:<14s} {hours:>5.1f}h {total:>+10.1f} {n_algos:>6d}")

    print()


def _save_report(report: dict, path: Path):
    """Save report JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    tmp.rename(path)


def _load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── run ───────────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run the full gauntlet sweep with incremental saves."""
    from algorithms import discover_all
    from alpha.paper_trader_daily import (
        DAILY_ALGOS,
        DEFAULT_COST,
        run_3f_liquidity,
        run_algo_single_date,
    )
    from alpha.paper_trader_generic import (
        BAR_SECONDS,
        TRAIN_WINDOW,
        discover_dates,
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    discover_all()
    all_dates = discover_dates(data_dir)
    symbols = args.symbols

    if len(all_dates) < TRAIN_WINDOW + 1:
        print(f"Need at least {TRAIN_WINDOW + 1} dates, have {len(all_dates)}")
        return 1

    testable = all_dates[TRAIN_WINDOW:]
    if args.date_from:
        testable = [d for d in testable if d >= args.date_from]
    if args.date_to:
        testable = [d for d in testable if d <= args.date_to]
    if args.last:
        testable = testable[-args.last:]

    if not testable:
        print("No testable dates in range.")
        return 1

    # Algo filtering (--algos = explicit include list, --exclude-algos = skip list)
    active_algos = list(DAILY_ALGOS)
    run_3f = True
    if args.algos:
        active_algos = [a for a in active_algos if a in args.algos]
        run_3f = "3f_liquidity" in args.algos
    if args.exclude_algos:
        active_algos = [a for a in active_algos if a not in args.exclude_algos]
        run_3f = run_3f and "3f_liquidity" not in args.exclude_algos
    n_active = len(active_algos) + (1 if run_3f else 0)
    excluded = sorted(set(DAILY_ALGOS + ["3f_liquidity"])
                      - set(active_algos) - ({"3f_liquidity"} if run_3f else set()))

    cost_model = DEFAULT_COST

    # Write PID file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    print(f"Gauntlet Sweep")
    print(f"  PID: {os.getpid()}")
    print(f"  Dates to test: {len(testable)} ({testable[0]} -> {testable[-1]})")
    print(f"  Algorithms: {n_active}"
          + (f" (excluded: {', '.join(excluded)})" if excluded else ""))
    print(f"  Symbols: {symbols}")
    print(f"  Cost: {cost_model.round_trip_cost_bps:.2f} bps RT ({args.cost_mode})")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    sys.stdout.flush()

    daily_reports = []
    t_total = time.time()

    def _save_incremental():
        """Save current progress to gauntlet_latest.json."""
        elapsed = time.time() - t_total
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "dates_tested": len(daily_reports),
            "dates_total": len(testable),
            "cost_bps_rt": cost_model.round_trip_cost_bps,
            "symbols": symbols,
            "elapsed_min": round(elapsed / 60, 1),
            "daily": daily_reports,
        }
        _save_report(report, LATEST_FILE)

    for test_date in testable:
        # Check hours
        from data.features import load_features
        df = load_features(
            symbols=[symbols[0]], date_range=(test_date, test_date),
            columns=["timestamp_ns", "symbol"], data_dir=data_dir, validate=False,
        )
        if df.empty:
            print(f"  [{test_date}] Skipping — no data")
            sys.stdout.flush()
            continue
        ts = df["timestamp_ns"].values
        hours = float((ts[-1] - ts[0]) / 1e9 / 3600)
        if hours < MIN_HOURS:
            print(f"  [{test_date}] Skipping — only {hours:.1f}h (need {MIN_HOURS}h)")
            sys.stdout.flush()
            continue

        test_idx = all_dates.index(test_date)
        train_dates = all_dates[test_idx - TRAIN_WINDOW:test_idx]
        n_bars = int(hours * 3600 / BAR_SECONDS)

        print(f"\n  [{test_date}] {hours:.1f}h, ~{n_bars} bars | train: {train_dates}")
        sys.stdout.flush()

        t0 = time.time()
        algo_results = {}
        algo_elapsed = {}

        if run_3f:
            t_a = time.time()
            r = run_3f_liquidity(data_dir, train_dates, test_date, symbols, cost_model=cost_model)
            algo_elapsed["3f_liquidity"] = round(time.time() - t_a, 1)
            if r:
                algo_results["3f_liquidity"] = r

        for algo_name in active_algos:
            t_a = time.time()
            r = run_algo_single_date(algo_name, data_dir, train_dates, test_date, symbols, cost_model=cost_model)
            algo_elapsed[algo_name] = round(time.time() - t_a, 1)
            if r:
                algo_results[algo_name] = r

        elapsed = time.time() - t0
        slowest = sorted(algo_elapsed.items(), key=lambda kv: -kv[1])[:3]
        print(f"  [{test_date}] Done in {elapsed:.0f}s — {len(algo_results)} algos produced results"
              f" (slowest: {', '.join(f'{a} {s:.0f}s' for a, s in slowest)})")
        sys.stdout.flush()

        daily_reports.append({
            "date": test_date,
            "hours": round(hours, 1),
            "bars": n_bars,
            "train_dates": train_dates,
            "elapsed_s": round(elapsed, 1),
            "algo_elapsed_s": algo_elapsed,
            "algorithms": algo_results,
        })

        # Save after every date
        _save_incremental()

    elapsed_total = time.time() - t_total
    print(f"\n  Total elapsed: {elapsed_total / 60:.1f} min")

    if not daily_reports:
        print("No dates had sufficient data.")
        PID_FILE.unlink(missing_ok=True)
        return 1

    # Final save with status=complete
    final = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "dates_tested": len(daily_reports),
        "dates_total": len(testable),
        "cost_bps_rt": cost_model.round_trip_cost_bps,
        "symbols": symbols,
        "elapsed_min": round(elapsed_total / 60, 1),
        "daily": daily_reports,
    }
    _save_report(final, LATEST_FILE)

    # Also save a timestamped copy
    ts_path = REPORTS_DIR / f"gauntlet_{testable[0]}_{testable[-1]}.json"
    _save_report(final, ts_path)

    print_summary(daily_reports, symbols)
    print(f"  Saved: {LATEST_FILE}")
    print(f"  Saved: {ts_path}")

    PID_FILE.unlink(missing_ok=True)
    return 0


# ── stop ──────────────────────────────────────────────────────────────────

def cmd_stop(args):
    """Kill a running gauntlet and print partial results."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Stopped gauntlet (PID {pid})")
            # Give it a moment to flush
            time.sleep(1)
        except ProcessLookupError:
            print(f"  Gauntlet (PID {pid}) was not running")
        PID_FILE.unlink(missing_ok=True)
    else:
        # Try to find it by process name
        import subprocess
        r = subprocess.run(["pgrep", "-f", "overnight_sweep.py run"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            for pid_str in r.stdout.strip().split("\n"):
                pid = int(pid_str)
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Stopped gauntlet (PID {pid})")
            time.sleep(1)
        else:
            print("  No running gauntlet found")

    # Print latest report
    return cmd_report(args)


# ── report ────────────────────────────────────────────────────────────────

def cmd_report(args):
    """Print the latest gauntlet report."""
    report = _load_report(LATEST_FILE)
    if report is None:
        print("  No gauntlet report found. Run `nat gauntlet` first.")
        return 1

    daily = report.get("daily", [])
    if not daily:
        print("  Report exists but has no completed dates yet.")
        return 1

    symbols = report.get("symbols", SYMBOLS_DEFAULT)
    status = report.get("status", "unknown")
    tested = report.get("dates_tested", len(daily))
    total = report.get("dates_total", tested)
    elapsed = report.get("elapsed_min", 0)
    cost = report.get("cost_bps_rt", 0)

    title = f"GAUNTLET REPORT ({status})"
    if status == "running":
        title += f" — {tested}/{total} dates, {elapsed:.0f} min elapsed"

    print(f"\n  Generated: {report.get('generated', '?')}")
    print(f"  Cost: {cost} bps RT")

    print_summary(daily, symbols, title=title)
    return 0


# ── report_all ────────────────────────────────────────────────────────────

def cmd_report_all(args):
    """Merge all gauntlet reports into one combined summary."""
    if not REPORTS_DIR.exists():
        print("  No reports directory found.")
        return 1

    # Collect all gauntlet_*.json files (timestamped runs) + latest
    report_files = sorted(REPORTS_DIR.glob("gauntlet_*.json"))
    # Also include overnight_*.json from the old naming convention
    report_files += sorted(REPORTS_DIR.glob("overnight_*.json"))

    if not report_files:
        print("  No gauntlet reports found.")
        return 1

    # Gather all daily entries, dedup by date (keep latest run's version)
    date_to_daily = {}
    all_symbols = set()
    total_cost = None

    for rf in report_files:
        report = _load_report(rf)
        if report is None:
            continue
        total_cost = report.get("cost_bps_rt", total_cost)
        for sym in report.get("symbols", SYMBOLS_DEFAULT):
            all_symbols.add(sym)
        for daily in report.get("daily", []):
            date = daily.get("date")
            if date:
                date_to_daily[date] = daily

    if not date_to_daily:
        print("  Reports found but no daily data inside them.")
        return 1

    # Sort by date
    merged_daily = [date_to_daily[d] for d in sorted(date_to_daily.keys())]
    symbols = sorted(all_symbols)

    n_files = len(report_files)
    n_dates = len(merged_daily)
    total_hours = sum(d.get("hours", 0) for d in merged_daily)

    print(f"\n  Merged {n_files} report files -> {n_dates} unique dates ({total_hours:.1f}h total)")
    if total_cost:
        print(f"  Cost: {total_cost} bps RT")

    print_summary(merged_daily, symbols, title=f"GAUNTLET COMBINED — {n_dates} dates")

    # Save combined report
    combined = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "status": "combined",
        "source_files": [str(f.name) for f in report_files],
        "dates_tested": n_dates,
        "cost_bps_rt": total_cost,
        "symbols": symbols,
        "daily": merged_daily,
    }
    out_path = REPORTS_DIR / "gauntlet_combined.json"
    _save_report(combined, out_path)
    print(f"  Saved: {out_path}")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gauntlet — multi-day OOS sweep")
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run the gauntlet sweep")
    run_p.add_argument("--data-dir", default="data/features")
    run_p.add_argument("--last", type=int, default=None, help="Only test the last N dates")
    run_p.add_argument("--from", dest="date_from", type=str, default=None)
    run_p.add_argument("--to", dest="date_to", type=str, default=None)
    run_p.add_argument("--symbols", nargs="+", default=SYMBOLS_DEFAULT)
    run_p.add_argument("--algos", nargs="+", default=None,
                       help="Only run these algorithms (names from DAILY_ALGOS / 3f_liquidity)")
    run_p.add_argument("--exclude-algos", nargs="+", default=None,
                       help="Run all algorithms except these")
    run_p.add_argument("--cost-mode", choices=["binance_vip9", "taker", "maker"],
                       default="binance_vip9")
    run_p.set_defaults(func=cmd_run)

    # stop
    stop_p = sub.add_parser("stop", help="Stop a running gauntlet, print partial results")
    stop_p.set_defaults(func=cmd_stop)

    # report
    report_p = sub.add_parser("report", help="Print the latest gauntlet report")
    report_p.set_defaults(func=cmd_report)

    # report_all
    report_all_p = sub.add_parser("report_all", help="Merge all reports into combined summary")
    report_all_p.set_defaults(func=cmd_report_all)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        # Default to run if no subcommand given (backwards compat)
        # Re-parse with "run" prepended
        args = parser.parse_args(["run"] + sys.argv[1:])

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

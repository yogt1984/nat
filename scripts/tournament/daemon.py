#!/usr/bin/env python3
"""Tournament daemon — continuous algorithm evaluation engine.

Cycle: DISCOVER → COLLECT → EVALUATE → RANK → DETECT → REPORT → SLEEP

Usage:
  python scripts/tournament/daemon.py run          # single cycle
  python scripts/tournament/daemon.py start        # daemon mode
  python scripts/tournament/daemon.py stop         # kill daemon
  python scripts/tournament/daemon.py status       # show state
  python scripts/tournament/daemon.py rankings     # current leaderboard
  python scripts/tournament/daemon.py history <algo>
  python scripts/tournament/daemon.py compare <a> <b>
  python scripts/tournament/daemon.py report       # generate markdown
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from tournament.db import TournamentDB
from tournament.lifecycle import run_lifecycle_checks
from tournament.report import (
    compute_rankings,
    format_rankings_table,
    format_telegram_summary,
    generate_markdown_report,
    save_report,
    send_telegram,
)

log = logging.getLogger("nat.tournament")

DATA_DIR = ROOT / "data" / "features"
PID_FILE = ROOT / "reports" / ".tournament.pid"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load tournament config from config/tournament.toml."""
    config_path = ROOT / "config" / "tournament.toml"
    if not config_path.exists():
        log.warning("No tournament config at %s, using defaults", config_path)
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return raw.get("tournament", {})


# ---------------------------------------------------------------------------
# Algorithm pool discovery
# ---------------------------------------------------------------------------

def collect_algorithm_pool() -> list[dict]:
    """Discover all testable algorithms: hand-coded + signal-adapted.

    Returns list of dicts: {name, source, algo_obj, config}.
    """
    from algorithms import discover_all, list_algorithms, get_algorithm
    from alpha.paper_trader_generic import ALGO_CONFIG

    discover_all()
    pool = []

    # Hand-coded algorithms with paper trader config
    for algo_name in list_algorithms():
        if algo_name in ALGO_CONFIG:
            algo = get_algorithm(algo_name)
            pool.append({
                "name": algo_name,
                "source": "hand_coded",
                "signal_id": None,
                "algo": algo,
                "config": ALGO_CONFIG[algo_name],
            })

    # Signal-adapted algorithms from agent registries
    try:
        from algorithms.signal_adapter import load_signal_algorithms
        signal_algos = load_signal_algorithms()
        for sa in signal_algos:
            gated_name = f"alg_{sa.name()}_gated"
            pool.append({
                "name": sa.name(),
                "source": f"agent_{sa.source_agent}",
                "signal_id": sa.signal_id,
                "algo": sa,
                "config": {
                    "primary": gated_name,
                    "polarity": sa.polarity,
                    "bar_agg": "mean",
                },
            })
    except Exception as e:
        log.warning("Failed to load signal algorithms: %s", e)

    log.info("Algorithm pool: %d total (%d hand-coded, %d signal-adapted)",
             len(pool),
             sum(1 for p in pool if p["source"] == "hand_coded"),
             sum(1 for p in pool if p["source"] != "hand_coded"))
    return pool


# ---------------------------------------------------------------------------
# Single algorithm evaluation (runs in worker process)
# ---------------------------------------------------------------------------

def evaluate_algorithm_date(algo_entry: dict, date_str: str,
                            train_dates: list[str], symbol: str,
                            data_dir: Path) -> dict | None:
    """Evaluate one algorithm on one (symbol, date) with walk-forward.

    Returns a dict suitable for db.insert_run(), or None on failure.
    """
    from alpha.paper_trader_generic import (
        load_date_ticks, aggregate_to_bars, compute_params,
        apply_signal, generate_trades, summarize_day,
    )
    from backtest.costs import CostModel
    from config_utils import load_cost_config

    algo = algo_entry["algo"]
    config = algo_entry["config"]
    primary = config["primary"]
    polarity = config["polarity"]
    agg_method = config["bar_agg"]
    required = algo.required_columns()

    cost_cfg = load_cost_config()
    cost_model = CostModel(fee_bps=cost_cfg["fee_bps"],
                           slippage_bps=cost_cfg["slippage_bps"])

    # Load training bars
    train_bars = []
    for td in train_dates:
        ticks = load_date_ticks(data_dir, td, symbol, required)
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
        if len(bars) >= 12:
            train_bars.append(bars)

    if not train_bars:
        return None

    # Load test date
    ticks = load_date_ticks(data_dir, date_str, symbol, required)
    if ticks is None or len(ticks) < 200:
        return None
    missing = [c for c in required if c not in ticks.columns]
    if missing:
        return None
    try:
        features = algo.run_batch(ticks)
        algo.reset()
    except Exception:
        return None
    if primary not in features.columns:
        return None

    bars = aggregate_to_bars(ticks, features, primary, agg_method)
    if len(bars) < 12:
        return None

    params = compute_params(train_bars, polarity)
    if params is None:
        return None

    scored = apply_signal(bars, params)
    trades = generate_trades(scored, date_str, symbol, cost_model=cost_model)
    summary = summarize_day(trades, date_str, symbol)

    return {
        "algo_name": algo_entry["name"],
        "algo_source": algo_entry["source"],
        "symbol": symbol,
        "date": date_str,
        "n_trades": summary.n_trades,
        "total_net_bps": summary.total_net_bps,
        "net_bps_per_trade": summary.net_bps,
        "win_rate": summary.win_rate,
        "max_loss_bps": summary.max_loss_bps,
    }


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

def run_cycle(config: dict) -> dict:
    """Execute one tournament evaluation cycle.

    Returns: {dates_evaluated, algos_evaluated, rankings, lifecycle}
    """
    db = TournamentDB()
    symbols = config.get("symbols", ["BTC", "ETH", "SOL"])
    min_hours = config.get("min_hours_per_date", 4.0)
    train_window = config.get("train_window", 3)

    # 1. DISCOVER — find new dates
    from data.features import available_dates
    all_dates = [d for d in available_dates(data_dir=DATA_DIR) if "clean" not in d]
    evaluated = db.get_evaluated_dates()
    new_dates = sorted(set(all_dates) - evaluated)

    if not new_dates:
        log.info("No new dates to evaluate")
        # Still run rankings + lifecycle on existing data
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rankings = compute_rankings(db, today, config)
        lifecycle = run_lifecycle_checks(db, config)
        return {
            "dates_evaluated": 0,
            "algos_evaluated": 0,
            "rankings": rankings,
            "lifecycle": lifecycle,
        }

    log.info("DISCOVER: %d new dates to evaluate: %s", len(new_dates), new_dates)

    # 2. COLLECT — gather algorithm pool
    pool = collect_algorithm_pool()
    if not pool:
        log.error("No algorithms in pool")
        return {"dates_evaluated": 0, "algos_evaluated": 0, "rankings": [], "lifecycle": {}}

    log.info("COLLECT: %d algorithms in pool", len(pool))

    # 3. EVALUATE — run each (algo, symbol, date) combination
    n_evaluated = 0
    for date_str in new_dates:
        date_idx = all_dates.index(date_str) if date_str in all_dates else -1
        if date_idx < train_window:
            log.info("  Skipping %s — not enough training dates", date_str)
            continue

        train_dates = all_dates[date_idx - train_window:date_idx]
        log.info("  EVALUATE %s (train: %s)", date_str, train_dates)

        for algo_entry in pool:
            for symbol in symbols:
                try:
                    result = evaluate_algorithm_date(
                        algo_entry, date_str, train_dates, symbol, DATA_DIR,
                    )
                except Exception as e:
                    log.warning("    %s/%s/%s failed: %s",
                                algo_entry["name"], symbol, date_str, e)
                    continue

                if result:
                    db.insert_run(**result)
                    n_evaluated += 1

        log.info("  %s complete — %d runs stored", date_str, n_evaluated)

    # 4. RANK
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rankings = compute_rankings(db, today, config)
    log.info("RANK: %d algorithms ranked", len(rankings))

    # 5. DETECT — lifecycle checks
    lifecycle = run_lifecycle_checks(db, config)

    # 6. REPORT
    if rankings:
        print(f"\n{format_rankings_table(rankings)}\n")

        telegram_msg = format_telegram_summary(
            today, rankings, lifecycle, n_evaluated,
        )
        send_telegram(telegram_msg)

        md = generate_markdown_report(db, today, rankings, lifecycle)
        save_report(today, md)

    # Register new algorithms as candidates
    for algo_entry in pool:
        existing = db.get_status(algo_entry["name"])
        if not existing:
            db.upsert_status(
                algo_name=algo_entry["name"],
                status="candidate",
                source=algo_entry["source"],
                signal_id=algo_entry.get("signal_id"),
                days_tested=db.count_dates_tested(algo_entry["name"]),
            )

    db.close()

    return {
        "dates_evaluated": len(new_dates),
        "algos_evaluated": n_evaluated,
        "rankings": rankings,
        "lifecycle": lifecycle,
    }


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def daemon_loop(config: dict) -> None:
    """Run tournament cycles on schedule."""
    sleep_s = config.get("sleep_interval_s", 3600)

    # Write PID
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False
        log.info("Received signal %d, stopping", signum)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info("Tournament daemon started (PID %d, sleep %ds)", os.getpid(), sleep_s)

    while running:
        try:
            result = run_cycle(config)
            log.info("Cycle complete: %d dates, %d evals",
                     result["dates_evaluated"], result["algos_evaluated"])
        except Exception as e:
            log.exception("Cycle failed: %s", e)

        # Sleep in short intervals to allow signal handling
        for _ in range(sleep_s):
            if not running:
                break
            time.sleep(1)

    PID_FILE.unlink(missing_ok=True)
    log.info("Tournament daemon stopped")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Run a single tournament cycle."""
    config = load_config()
    result = run_cycle(config)
    print(f"\nDates evaluated: {result['dates_evaluated']}")
    print(f"Runs stored: {result['algos_evaluated']}")
    if result["rankings"]:
        print(f"\n{format_rankings_table(result['rankings'])}")


def cmd_start(args):
    """Start the tournament daemon."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Tournament daemon already running (PID {pid})")
            return
        except OSError:
            PID_FILE.unlink()

    config = load_config()
    print("Starting tournament daemon...")
    daemon_loop(config)


def cmd_stop(args):
    """Stop the tournament daemon."""
    if not PID_FILE.exists():
        print("No tournament daemon running")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
    except OSError as e:
        print(f"Failed to stop PID {pid}: {e}")
        PID_FILE.unlink()


def cmd_status(args):
    """Show tournament status."""
    db = TournamentDB()

    # Daemon status
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Daemon: running (PID {pid})")
        except OSError:
            print("Daemon: not running (stale PID file)")
    else:
        print("Daemon: not running")

    # DB stats
    all_algos = db.get_all_algo_names()
    dates = db.get_evaluated_dates()
    statuses = db.get_all_statuses()

    print(f"\nAlgorithms tracked: {len(all_algos)}")
    print(f"Dates evaluated: {len(dates)}")

    status_counts = {}
    for s in statuses:
        st = s["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
    for st, n in sorted(status_counts.items()):
        print(f"  {st}: {n}")

    # Latest rankings
    rankings = db.get_latest_rankings()
    if rankings:
        print(f"\nLatest rankings ({rankings[0]['date'] if rankings else '?'}):")
        print(format_rankings_table(rankings[:10]))

    db.close()


def cmd_rankings(args):
    """Show current leaderboard."""
    db = TournamentDB()
    rankings = db.get_latest_rankings()
    if rankings:
        print(format_rankings_table(rankings))
    else:
        print("No rankings available. Run 'nat tournament run' first.")
    db.close()


def cmd_history(args):
    """Show per-day history for one algorithm."""
    db = TournamentDB()
    runs = db.get_daily_pnl(args.algorithm, last_n_days=args.days)
    if not runs:
        print(f"No runs found for '{args.algorithm}'")
        db.close()
        return

    print(f"{'Date':<12} {'Daily BPS':>10} {'Win Rate':>9} {'Trades':>7}")
    print("-" * 42)
    cumulative = 0.0
    for r in runs:
        cumulative += r["daily_bps"]
        print(f"{r['date']:<12} {r['daily_bps']:>+10.1f} "
              f"{r['avg_win_rate']:>9.1%} {r['total_trades']:>7}")
    print("-" * 42)
    print(f"{'Total':<12} {cumulative:>+10.1f}")

    sharpe_7d = db.compute_rolling_sharpe(args.algorithm, 7)
    sharpe_30d = db.compute_rolling_sharpe(args.algorithm, 30)
    print(f"\nSharpe: 7d={sharpe_7d:.2f}  30d={sharpe_30d:.2f}")
    db.close()


def cmd_compare(args):
    """Head-to-head comparison of two algorithms."""
    db = TournamentDB()
    a, b = args.algo_a, args.algo_b

    pnl_a = db.get_daily_pnl(a, last_n_days=30)
    pnl_b = db.get_daily_pnl(b, last_n_days=30)

    if not pnl_a or not pnl_b:
        print(f"Insufficient data for comparison")
        db.close()
        return

    dates_a = {r["date"] for r in pnl_a}
    dates_b = {r["date"] for r in pnl_b}
    common = sorted(dates_a & dates_b)

    if not common:
        print("No common evaluation dates")
        db.close()
        return

    map_a = {r["date"]: r for r in pnl_a}
    map_b = {r["date"]: r for r in pnl_b}

    print(f"{'Date':<12} {a:>15} {b:>15} {'Winner':>10}")
    print("-" * 55)
    a_wins = b_wins = 0
    cum_a = cum_b = 0.0
    for d in common:
        va = map_a[d]["daily_bps"]
        vb = map_b[d]["daily_bps"]
        cum_a += va
        cum_b += vb
        winner = a if va > vb else b if vb > va else "tie"
        if va > vb:
            a_wins += 1
        elif vb > va:
            b_wins += 1
        print(f"{d:<12} {va:>+15.1f} {vb:>+15.1f} {winner:>10}")

    print("-" * 55)
    print(f"{'Total':<12} {cum_a:>+15.1f} {cum_b:>+15.1f}")
    print(f"\n{a} wins: {a_wins} | {b} wins: {b_wins} | "
          f"ties: {len(common) - a_wins - b_wins}")

    print(f"\nSharpe 7d:  {a}={db.compute_rolling_sharpe(a, 7):.2f}  "
          f"{b}={db.compute_rolling_sharpe(b, 7):.2f}")
    print(f"Sharpe 30d: {a}={db.compute_rolling_sharpe(a, 30):.2f}  "
          f"{b}={db.compute_rolling_sharpe(b, 30):.2f}")
    db.close()


def cmd_report(args):
    """Generate markdown report."""
    config = load_config()
    db = TournamentDB()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rankings = compute_rankings(db, today, config)
    lifecycle = run_lifecycle_checks(db, config)
    md = generate_markdown_report(db, today, rankings, lifecycle)
    path = save_report(today, md)
    print(f"Report saved to {path}")
    db.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tournament — continuous algorithm testing")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run a single evaluation cycle")
    sub.add_parser("start", help="Start daemon")
    sub.add_parser("stop", help="Stop daemon")
    sub.add_parser("status", help="Show status")
    sub.add_parser("rankings", help="Current leaderboard")

    p_hist = sub.add_parser("history", help="Per-day history for one algorithm")
    p_hist.add_argument("algorithm")
    p_hist.add_argument("--days", type=int, default=30)

    p_cmp = sub.add_parser("compare", help="Head-to-head comparison")
    p_cmp.add_argument("algo_a")
    p_cmp.add_argument("algo_b")

    sub.add_parser("report", help="Generate markdown report")

    args = parser.parse_args()

    commands = {
        "run": cmd_run,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "rankings": cmd_rankings,
        "history": cmd_history,
        "compare": cmd_compare,
        "report": cmd_report,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

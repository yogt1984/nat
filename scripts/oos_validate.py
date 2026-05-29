"""
Continuous OOS Validation Runner

Runs the 4 winning algorithms (3f, jump_detector, funding_reversion, optimal_entry)
on available data, computes rolling validation metrics, writes consolidated state
for the terminal dashboard (oos_terminal.py).

Usage:
  python scripts/oos_validate.py batch            # Run all algos, write state
  python scripts/oos_validate.py watch [--poll N]  # Poll for new data, re-run
  python scripts/oos_validate.py report            # Show terminal dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data" / "oos_validation"
STATE_FILE = STATE_DIR / "state.json"

PYTHON = sys.executable

# Baseline Sharpe ratios from initial validation runs
BASELINES = {
    "3f": {"BTC": 9.25, "ETH": 7.8, "SOL": 5.1},
    "jump_detector": {"BTC": 1.62, "ETH": 6.23, "SOL": 6.22},
    "funding_reversion": {"BTC": 0.38, "ETH": 6.05, "SOL": 1.69},
    "optimal_entry": {"BTC": 1.12, "ETH": 5.23, "SOL": 0.98},
}

GENERIC_ALGOS = ["jump_detector", "funding_reversion", "optimal_entry"]
SYMBOLS = ["BTC", "ETH", "SOL"]


# ── State I/O ───────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_updated": None, "algos": {}}


def save_state(state: dict) -> None:
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ── Algo runners (subprocess) ───────────────────────────────────────────

def run_3f(data_dir: Path) -> dict | None:
    """Run 3f paper trader via subprocess, return results dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = tf.name
    try:
        cmd = [
            PYTHON, str(ROOT / "scripts" / "alpha" / "paper_trader.py"),
            "batch", "--data-dir", str(data_dir), "--save",
            "--json-output", tmp_path,
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}
        print(f"  Running 3f signal...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
        if result.returncode != 0:
            print(f"  3f failed: {result.stderr[-500:]}")
            return None
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_generic(algo_name: str, data_dir: Path) -> dict | None:
    """Run a generic algo via subprocess, return results dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = tf.name
    try:
        cmd = [
            PYTHON, str(ROOT / "scripts" / "alpha" / "paper_trader_generic.py"),
            "--algorithms", algo_name,
            "--data-dir", str(data_dir),
            "--json-output", tmp_path,
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}
        print(f"  Running {algo_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
        if result.returncode != 0:
            print(f"  {algo_name} failed: {result.stderr[-500:]}")
            return None
        with open(tmp_path) as f:
            raw = json.load(f)
        # generic output is {algo_name: {elapsed_s, results: {symbol: {...}}}}
        if algo_name in raw:
            return raw[algo_name].get("results", {})
        return raw
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(daily: list[dict], baseline_sharpe: float) -> dict:
    """Compute rolling validation metrics from daily summaries."""
    if not daily:
        return {
            "current_sharpe": 0.0,
            "cumulative_pnl_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "n_days": 0,
            "rolling_sharpe_7d": [],
            "cumulative_pnl_series": [],
            "win_rate_trend": [],
            "degradation": False,
        }

    pnl = np.array([d["total_net_bps"] for d in daily])
    n = len(pnl)

    from utils.metrics import sharpe_daily

    # Overall Sharpe
    current_sharpe = sharpe_daily(pnl)

    # Cumulative PnL
    cum_pnl = np.cumsum(pnl)
    cumulative_pnl_bps = float(cum_pnl[-1])

    # Max drawdown
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_drawdown_bps = float(np.min(drawdown))

    # Rolling 7-day Sharpe
    window = 7
    rolling_sharpe = []
    for i in range(window - 1, n):
        w = pnl[i - window + 1:i + 1]
        s = sharpe_daily(w)
        rolling_sharpe.append({
            "date": daily[i]["date"],
            "sharpe": round(s, 2),
        })

    # Cumulative PnL series
    cum_series = [
        {"date": daily[i]["date"], "cum_bps": round(float(cum_pnl[i]), 1)}
        for i in range(n)
    ]

    # Win rate trend (rolling 7d)
    win_rates = np.array([d.get("win_rate", 0) for d in daily])
    wr_trend = []
    for i in range(window - 1, n):
        wr = float(np.mean(win_rates[i - window + 1:i + 1]))
        wr_trend.append({"date": daily[i]["date"], "wr": round(wr, 3)})

    # Degradation: rolling Sharpe < 50% of baseline for 3+ consecutive days
    degradation = False
    if baseline_sharpe > 0 and len(rolling_sharpe) >= 3:
        threshold = baseline_sharpe * 0.5
        consecutive = 0
        for rs in rolling_sharpe:
            if rs["sharpe"] < threshold:
                consecutive += 1
                if consecutive >= 3:
                    degradation = True
                    break
            else:
                consecutive = 0

    return {
        "current_sharpe": round(current_sharpe, 2),
        "cumulative_pnl_bps": round(cumulative_pnl_bps, 1),
        "max_drawdown_bps": round(max_drawdown_bps, 1),
        "n_days": n,
        "rolling_sharpe_7d": rolling_sharpe,
        "cumulative_pnl_series": cum_series,
        "win_rate_trend": wr_trend,
        "degradation": degradation,
    }


def _extract_daily(results: dict, symbol: str) -> list[dict]:
    """Extract daily summary list from algo results for a symbol."""
    sym_data = results.get(symbol, {})
    return sym_data.get("daily", [])


# ── Build state ─────────────────────────────────────────────────────────

def build_algo_state(algo_name: str, results: dict | None) -> dict:
    """Build state entry for one algorithm from its results."""
    baseline = BASELINES.get(algo_name, {})
    entry = {"baseline_sharpe": baseline, "symbols": {}}

    if results is None:
        return entry

    for symbol in SYMBOLS:
        daily = _extract_daily(results, symbol)
        bl_sharpe = baseline.get(symbol, 0.0)
        metrics = compute_metrics(daily, bl_sharpe)
        entry["symbols"][symbol] = {
            "daily": daily,
            "metrics": metrics,
        }

    return entry


# ── Commands ────────────────────────────────────────────────────────────

def cmd_batch(data_dir: Path):
    """Run all 4 algos, compute metrics, save state."""
    print("OOS Validation — Batch Run")
    print(f"Data: {data_dir}\n")

    state = load_state()

    # 3f signal
    results_3f = run_3f(data_dir)
    state["algos"]["3f"] = build_algo_state("3f", results_3f)

    # Generic algos
    for algo in GENERIC_ALGOS:
        results = run_generic(algo, data_dir)
        state["algos"][algo] = build_algo_state(algo, results)

    save_state(state)
    print(f"\nState saved: {STATE_FILE}\n")
    sys.stdout.flush()

    # Show dashboard
    _show_dashboard()


def cmd_watch(data_dir: Path, poll_seconds: int):
    """Watch for new data and re-run validation."""
    print(f"OOS Validation — Watch Mode (poll every {poll_seconds}s)")
    print(f"Data: {data_dir}\n")

    last_dates: set[str] = set()

    while True:
        # Discover available dates
        current_dates = set()
        if data_dir.exists():
            for d in sorted(data_dir.iterdir()):
                if d.is_dir() and len(d.name) == 10:  # YYYY-MM-DD
                    current_dates.add(d.name)

        new_dates = current_dates - last_dates
        if new_dates or not last_dates:
            if new_dates:
                print(f"\nNew dates detected: {sorted(new_dates)}")
            cmd_batch(data_dir)
            last_dates = current_dates
        else:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] No new data, sleeping {poll_seconds}s...")

        time.sleep(poll_seconds)


def _show_dashboard():
    """Call the terminal viewer."""
    cmd = [PYTHON, str(ROOT / "scripts" / "oos_terminal.py")]
    subprocess.run(cmd, cwd=str(ROOT))


def cmd_report():
    """Just show the dashboard from existing state."""
    if not STATE_FILE.exists():
        print(f"No state file found at {STATE_FILE}")
        print("Run 'python scripts/oos_validate.py batch' first.")
        sys.exit(1)
    _show_dashboard()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continuous OOS Validation Runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("batch", help="Run all algos, write state")
    p_batch.add_argument("--data-dir", default="data/features")

    p_watch = sub.add_parser("watch", help="Poll for new data, re-run")
    p_watch.add_argument("--data-dir", default="data/features")
    p_watch.add_argument("--poll", type=int, default=300, help="Poll interval seconds")

    sub.add_parser("report", help="Show terminal dashboard from existing state")

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") else None

    if args.command == "batch":
        cmd_batch(data_dir)
    elif args.command == "watch":
        cmd_watch(data_dir, args.poll)
    elif args.command == "report":
        cmd_report()


if __name__ == "__main__":
    main()

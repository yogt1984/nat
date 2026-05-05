"""
Experiment runner for NAT profiling pipeline.

Commands:
    start   — Start ingestor in tmux background session
    stop    — Stop ingestor gracefully
    status  — Check ingestor health + data stats
    check   — Run daily validation on recent data
    midweek — Full validation + schema scan (day 3)
    analyze — Stop ingestor, validate, run profiling + quality gates
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"
LOG_DIR = PROJECT_ROOT / "logs"
BINARY = PROJECT_ROOT / "rust" / "target" / "release" / "ing"
CONFIG = PROJECT_ROOT / "config" / "ing.toml"
TMUX_SESSION = "ingestor"


def _run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=check, capture_output=capture, text=True)


def _tmux_exists() -> bool:
    r = _run(f"tmux has-session -t {TMUX_SESSION} 2>/dev/null", check=False)
    return r.returncode == 0


def _ingestor_pid() -> int | None:
    r = _run("pgrep -x ing", check=False, capture=True)
    if r.returncode == 0 and r.stdout.strip():
        return int(r.stdout.strip().split("\n")[0])
    return None


def _data_stats() -> dict:
    stats = {"total_files": 0, "total_size": "0", "days": [], "today_files": 0}
    if not DATA_DIR.exists():
        return stats

    parquets = list(DATA_DIR.rglob("*.parquet"))
    stats["total_files"] = len(parquets)

    r = _run(f"du -sh {DATA_DIR}", check=False, capture=True)
    if r.returncode == 0:
        stats["total_size"] = r.stdout.split()[0]

    days = sorted(set(p.parent.name for p in parquets))
    stats["days"] = days

    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = DATA_DIR / today
    if today_dir.exists():
        stats["today_files"] = len(list(today_dir.glob("*.parquet")))

    return stats


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_start(args):
    """Start the ingestor in a tmux session."""
    if _ingestor_pid():
        print(f"[!] Ingestor already running (PID {_ingestor_pid()})")
        print("    Use 'make exp_stop' first if you want to restart.")
        return

    if not BINARY.exists():
        print("[!] Binary not found. Building release...")
        _run(f"cd {PROJECT_ROOT} && make release")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"ingestor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = (
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"'cd {PROJECT_ROOT / 'rust'} && "
        f"./target/release/ing ../config/ing.toml 2>&1 | tee {logfile}'"
    )
    _run(cmd)

    # Wait and verify
    time.sleep(3)
    pid = _ingestor_pid()
    if pid:
        print(f"[OK] Ingestor started (PID {pid})")
        print(f"     Log: {logfile}")
        print(f"     Tmux: tmux attach -t {TMUX_SESSION}")
        print(f"     Detach: Ctrl+B then D")
    else:
        print("[FAIL] Ingestor did not start. Check the log:")
        print(f"       cat {logfile}")


def cmd_stop(args):
    """Stop the ingestor gracefully."""
    pid = _ingestor_pid()
    if not pid:
        print("[!] Ingestor is not running.")
        if _tmux_exists():
            _run(f"tmux kill-session -t {TMUX_SESSION}", check=False)
        return

    print(f"[..] Stopping ingestor (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    time.sleep(2)

    if _ingestor_pid():
        print("[..] Still alive, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)

    if _tmux_exists():
        _run(f"tmux kill-session -t {TMUX_SESSION}", check=False)

    print("[OK] Ingestor stopped.")


def cmd_status(args):
    """Check ingestor health and data stats."""
    pid = _ingestor_pid()
    stats = _data_stats()

    print("=" * 60)
    print("  NAT Experiment Status")
    print("=" * 60)
    print()

    # Ingestor
    if pid:
        print(f"  Ingestor:     RUNNING (PID {pid})")
    else:
        print("  Ingestor:     STOPPED")

    # Tmux
    if _tmux_exists():
        print(f"  Tmux session: {TMUX_SESSION} (active)")
    else:
        print("  Tmux session: none")

    print()

    # Data
    print(f"  Data dir:     {DATA_DIR}")
    print(f"  Total size:   {stats['total_size']}")
    print(f"  Parquet files:{stats['total_files']}")
    print(f"  Days:         {len(stats['days'])}")
    if stats["days"]:
        print(f"  Range:        {stats['days'][0]} to {stats['days'][-1]}")
    print(f"  Today files:  {stats['today_files']}")

    # Row count (quick, only if loader available)
    print()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from cluster_pipeline.loader import load_parquet
        df = load_parquet(str(DATA_DIR))
        print(f"  Total rows:   {len(df):,}")
        bars_15m = len(df) // (30 * 60 * 15)  # rough estimate
        print(f"  ~15min bars:  ~{bars_15m}")
    except Exception as e:
        print(f"  Row count:    (could not load: {e})")

    print()
    print("=" * 60)


def cmd_check(args):
    """Daily validation check on recent data."""
    hours = args.hours if hasattr(args, "hours") else 24
    print(f"[..] Validating last {hours} hours of data...\n")
    _run(f"cd {PROJECT_ROOT} && make validate_data_recent HOURS={hours}")


def cmd_midweek(args):
    """Mid-week deep validation."""
    print("[..] Running full validation...\n")
    _run(f"cd {PROJECT_ROOT} && make validate_data")
    print()
    print("[..] Scanning schema...\n")
    _run(f"cd {PROJECT_ROOT} && make scan_schema")


def cmd_analyze(args):
    """End-of-experiment: stop ingestor, validate, profile, evaluate."""
    # Step 1: Stop ingestor
    pid = _ingestor_pid()
    if pid:
        print("[1/5] Stopping ingestor...")
        cmd_stop(args)
    else:
        print("[1/5] Ingestor already stopped.")

    # Step 2: Final validation
    print("\n[2/5] Running final validation...")
    _run(f"cd {PROJECT_ROOT} && make validate_data", check=False)

    # Step 3: Schema scan
    print("\n[3/5] Scanning schema...")
    _run(f"cd {PROJECT_ROOT} && make scan_schema", check=False)

    # Step 4: Profile
    print("\n[4/5] Running profiling pipeline...")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    try:
        import numpy as np
        from cluster_pipeline.loader import load_parquet
        from cluster_pipeline.preprocess import aggregate_bars
        from cluster_pipeline.derivatives import generate_derivatives
        from cluster_pipeline.hierarchy import profile
        from cluster_pipeline.validate import validate

        df = load_parquet(str(DATA_DIR))
        print(f"       Loaded {len(df):,} rows")

        bars = aggregate_bars(df, bar_minutes=15)
        print(f"       Aggregated to {len(bars)} bars")

        deriv = generate_derivatives(bars, vector="entropy", include_spectral=True)
        print(f"       {deriv.derivatives.shape[1]} derivative features, warmup={deriv.warmup_rows}")

        result = profile(deriv.derivatives, warmup_rows=deriv.warmup_rows)
        print(f"       Macro k={result.macro.k}")
        print(f"       Silhouette={result.macro.quality.silhouette:.3f}")
        print(f"       Bootstrap ARI={result.macro.stability.mean_ari:.3f}")

        # Step 5: Validate
        print("\n[5/5] Running quality gates...")
        prices = bars["raw_midprice_mean"].values if "raw_midprice_mean" in bars.columns else np.ones(len(bars))
        verdict = validate(result, prices)

        print()
        print("=" * 60)
        print("  EXPERIMENT RESULT")
        print("=" * 60)
        print()
        print(f"  {verdict.summary}")
        print()
        print(f"  *** VERDICT: {verdict.overall} ***")
        print()

        if verdict.overall == "GO":
            print("  All gates passed. Proceed to strategy development.")
        elif verdict.overall == "PIVOT":
            print("  Clusters exist and predict returns but aren't tradeable yet.")
            print("  Try: different bar sizes (5min, 1h), different feature vectors.")
        elif verdict.overall == "COLLECT":
            print("  Insufficient evidence. Extend data collection to 14+ days.")
        elif verdict.overall == "DROP":
            print("  No structure found. Try different feature subsets.")

        print()
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Profiling failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NAT Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start     Start ingestor in tmux background
  stop      Stop ingestor gracefully
  status    Show ingestor health + data stats
  check     Daily validation (last 24h by default)
  midweek   Full validation + schema scan
  analyze   Stop, validate, profile, evaluate quality gates
""",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start ingestor in tmux")
    sub.add_parser("stop", help="Stop ingestor")
    sub.add_parser("status", help="Check health + data stats")

    p_check = sub.add_parser("check", help="Daily validation")
    p_check.add_argument("--hours", type=int, default=24)

    sub.add_parser("midweek", help="Full validation + schema scan")
    sub.add_parser("analyze", help="End-of-experiment analysis")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "check": cmd_check,
        "midweek": cmd_midweek,
        "analyze": cmd_analyze,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

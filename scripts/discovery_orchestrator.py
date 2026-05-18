#!/usr/bin/env python3
"""
Alpha Discovery Orchestrator

Continuous daemon that sweeps (symbol, horizon) combos for alpha signals,
then pipelines winners through train → backtest → validate → paper trade.

All child scripts are called via subprocess to prevent OOM from accumulated
memory across steps.

Usage:
    python scripts/discovery_orchestrator.py --config config/discovery.toml start
    python scripts/discovery_orchestrator.py --config config/discovery.toml once
    python scripts/discovery_orchestrator.py --config config/discovery.toml status
    python scripts/discovery_orchestrator.py --config config/discovery.toml stop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal as signal_mod
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    IDLE = "IDLE"
    DATA_HEALTH = "DATA_HEALTH"
    SIGNAL_SWEEP = "SIGNAL_SWEEP"
    TRAINING = "TRAINING"
    BACKTESTING = "BACKTESTING"
    ALPHA_PIPELINE = "ALPHA_PIPELINE"
    REPORTING = "REPORTING"
    SLEEPING = "SLEEPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Discovery config not found: {config_path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# State persistence (follows AlphaPipelineState pattern)
# ---------------------------------------------------------------------------


class DiscoveryState:
    """Persistent orchestrator state — survives restarts."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "phase": Phase.IDLE.value,
            "cycle_number": 0,
            "started_at": None,
            "last_cycle_at": None,
            "current_combo": None,
            "winners": [],
            "gates": {},
            "artifacts": {},
            "step_outputs": {},
            "error": None,
            "history": [],
        }

    def save(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def transition(self, phase: Phase, message: str = "") -> None:
        old = self._data["phase"]
        self._data["phase"] = phase.value
        self._data["history"].append({
            "from": old,
            "to": phase.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })
        if len(self._data["history"]) > 200:
            self._data["history"] = self._data["history"][-100:]
        self.save()

    @property
    def current(self) -> Phase:
        return Phase(self._data["phase"])

    def record_gate(self, name: str, verdict: str, metrics: dict, advice: str = "") -> None:
        self._data["gates"][name] = {
            "verdict": verdict,
            "metrics": metrics,
            "advice": advice,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def set_artifact(self, name: str, path: str) -> None:
        self._data["artifacts"][name] = path
        self.save()

    def set_output(self, name: str, data: dict) -> None:
        self._data["step_outputs"][name] = data
        self.save()

    def reset_cycle(self) -> None:
        """Reset per-cycle state but keep cycle_number and history."""
        self._data["current_combo"] = None
        self._data["winners"] = []
        self._data["gates"] = {}
        self._data["artifacts"] = {}
        self._data["step_outputs"] = {}
        self._data["error"] = None
        self.save()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: str) -> logging.Logger:
    log = logging.getLogger("discovery")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


@dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    report: Optional[dict] = None


def run_subprocess(
    cmd: List[str],
    log: logging.Logger,
    timeout_s: int = 600,
    report_path: Optional[Path] = None,
) -> SubprocessResult:
    """Run a child script and optionally parse a JSON report it writes."""
    log.info("  CMD: %s", " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
        )
    except subprocess.TimeoutExpired:
        log.error("  TIMEOUT after %ds", timeout_s)
        return SubprocessResult(-1, "", f"Timeout after {timeout_s}s", timeout_s)

    duration = time.time() - t0
    report = None
    if report_path and report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
        except Exception as e:
            log.warning("  Failed to parse report %s: %s", report_path, e)

    if proc.returncode != 0:
        log.warning("  EXIT %d (%.1fs)", proc.returncode, duration)
        # Log last few lines of stderr
        for line in proc.stderr.strip().split("\n")[-5:]:
            if line.strip():
                log.warning("    %s", line.strip())
    else:
        log.info("  OK (%.1fs)", duration)

    return SubprocessResult(proc.returncode, proc.stdout, proc.stderr, duration, report)


# ---------------------------------------------------------------------------
# Step 1: Data Health
# ---------------------------------------------------------------------------


def step_data_health(config: dict, state: DiscoveryState, log: logging.Logger) -> str:
    log.info("=" * 60)
    log.info("  STEP 1: Data Health Check")
    log.info("=" * 60)

    data_dir = config["discovery"]["data_dir"]
    cmd = [sys.executable, "scripts/validate_data.py", data_dir, "--verbose"]
    result = run_subprocess(cmd, log, timeout_s=300)

    # Exit 0 = all checks pass, exit 1 = some checks failed (NaN in optional
    # features is expected), exit 2+ = hard failure (no data, corrupt files)
    if result.returncode == 0:
        verdict = "PASS"
    elif result.returncode == 1:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    state.record_gate("data_health", verdict, {"exit_code": result.returncode})
    log.info("  Data health: %s", verdict)
    return verdict


# ---------------------------------------------------------------------------
# Step 2: Signal Sweep
# ---------------------------------------------------------------------------


def step_signal_sweep(
    config: dict, state: DiscoveryState, log: logging.Logger, cycle_dir: Path,
) -> tuple[str, list[dict]]:
    log.info("=" * 60)
    log.info("  STEP 2: Signal Sweep")
    log.info("=" * 60)

    disc = config["discovery"]
    gates = config["gates"]
    symbols = disc["sweep"]["symbols"]
    horizons = disc["sweep"]["horizons"]
    data_dir = disc["data_dir"]
    start_date = disc.get("start_date") or None
    end_date = disc.get("end_date") or None
    max_mem = disc.get("max_memory_mb", 2000.0)

    all_results = []
    winners = []

    for symbol in symbols:
        for horizon in horizons:
            combo_name = f"{symbol}_{horizon}"
            report_path = cycle_dir / f"signal_{combo_name}.json"

            cmd = [
                sys.executable, "scripts/phase1_signal_test.py",
                "--symbol", symbol,
                "--horizon", str(horizon),
                "--data-dir", data_dir,
                "--remove-leaky",
                "--max-memory-mb", str(max_mem),
                "--json-report", str(report_path),
            ]
            if start_date:
                cmd += ["--start-date", start_date]
            if end_date:
                cmd += ["--end-date", end_date]

            result = run_subprocess(cmd, log, timeout_s=600, report_path=report_path)

            if result.report is None:
                log.warning("  %s: no report (exit %d)", combo_name, result.returncode)
                continue

            r = result.report
            edge = r.get("test2_avg_edge", 0.0)
            best_gross = r.get("test3_best_gross_bps", 0.0)

            all_results.append({
                "symbol": symbol,
                "horizon": horizon,
                "edge": edge,
                "gross_bps": best_gross,
                "accuracy": r.get("test2_avg_accuracy", 0.0),
                "sharpe": r.get("test2_avg_sharpe", 0.0),
            })

            log.info("  %s: edge=%.4f gross=%.2f bps", combo_name, edge, best_gross)

            if edge >= gates["min_wf_edge"] and best_gross >= gates["min_gross_bps"]:
                winners.append({
                    "symbol": symbol,
                    "horizon": horizon,
                    "edge": edge,
                    "gross_bps": best_gross,
                })

    # Sort winners by edge descending
    winners.sort(key=lambda w: w["edge"], reverse=True)
    state.set("winners", winners)
    state.set_output("signal_sweep", {"all_results": all_results, "n_winners": len(winners)})

    verdict = "PASS" if len(winners) > 0 else "FAIL"
    state.record_gate("signal_sweep", verdict, {
        "n_tested": len(all_results),
        "n_winners": len(winners),
    }, "Lower min_wf_edge or min_gross_bps, or collect more data" if verdict == "FAIL" else "")
    log.info("  Signal sweep: %s (%d winners / %d tested)", verdict, len(winners), len(all_results))

    return verdict, winners


# ---------------------------------------------------------------------------
# Step 3: Train Model
# ---------------------------------------------------------------------------


def step_train(
    config: dict, state: DiscoveryState, combo: dict, cycle_num: int,
    cycle_dir: Path, log: logging.Logger,
) -> tuple[str, Optional[str]]:
    symbol = combo["symbol"]
    horizon = combo["horizon"]
    combo_name = f"{symbol}_{horizon}"

    log.info("-" * 40)
    log.info("  STEP 3: Train Model — %s", combo_name)
    log.info("-" * 40)

    snapshot_name = f"discovery_c{cycle_num}_{combo_name}"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--data-dir", config["discovery"]["data_dir"],
        "--model", "lightgbm",
        "--symbol", symbol,
        "--horizon", str(horizon),
        "--output-dir", str(models_dir),
        "--no-tracking",
    ]

    result = run_subprocess(cmd, log, timeout_s=600)

    if result.returncode != 0:
        log.warning("  Train FAIL for %s", combo_name)
        state.record_gate(f"train_{combo_name}", "FAIL", {"exit_code": result.returncode})
        return "FAIL", None

    # Find model path from stdout or by globbing newest file
    model_path = None
    for line in result.stdout.split("\n"):
        if "Model saved to:" in line:
            model_path = line.split("Model saved to:")[-1].strip()
            break

    if not model_path:
        # Glob for most recent model
        candidates = sorted(models_dir.glob("lightgbm_*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            model_path = str(candidates[-1])

    if not model_path or not Path(model_path).exists():
        log.warning("  Train: no model file found for %s", combo_name)
        state.record_gate(f"train_{combo_name}", "FAIL", {"reason": "no model file"})
        return "FAIL", None

    # Parse R² from stdout
    test_r2 = 0.0
    for line in result.stdout.split("\n"):
        if "Test R" in line and ":" in line:
            try:
                test_r2 = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    min_r2 = config["gates"].get("min_test_r2", 0.0)
    verdict = "PASS" if test_r2 > min_r2 else "FAIL"
    state.record_gate(f"train_{combo_name}", verdict, {"test_r2": test_r2, "model_path": model_path})
    state.set_artifact(f"model_{combo_name}", model_path)
    log.info("  Train %s: R²=%.4f model=%s", verdict, test_r2, model_path)

    return verdict, model_path


# ---------------------------------------------------------------------------
# Step 4: Score + Backtest
# ---------------------------------------------------------------------------


def step_backtest(
    config: dict, state: DiscoveryState, combo: dict, model_path: str,
    cycle_num: int, cycle_dir: Path, log: logging.Logger,
) -> tuple[str, dict]:
    symbol = combo["symbol"]
    horizon = combo["horizon"]
    combo_name = f"{symbol}_{horizon}"

    log.info("-" * 40)
    log.info("  STEP 4: Score + Backtest — %s", combo_name)
    log.info("-" * 40)

    data_dir = config["discovery"]["data_dir"]
    predictions_path = cycle_dir / f"predictions_{combo_name}.parquet"

    # Score
    cmd_score = [
        sys.executable, "scripts/score_data.py",
        "--model", model_path,
        "--data", data_dir,
        "--output", str(predictions_path),
        "--evaluate",
    ]
    result_score = run_subprocess(cmd_score, log, timeout_s=600)

    if result_score.returncode != 0 or not predictions_path.exists():
        log.warning("  Score FAIL for %s", combo_name)
        state.record_gate(f"backtest_{combo_name}", "FAIL", {"reason": "scoring failed"})
        return "FAIL", {}

    # Backtest
    cost_model = config["gates"].get("cost_model", "taker")
    cmd_bt = [
        sys.executable, "scripts/run_backtest.py",
        "--data-dir", data_dir,
        "--symbol", symbol,
        "--ml-predictions", str(predictions_path),
        "--walk-forward",
        "--cost-model", cost_model,
    ]
    result_bt = run_subprocess(cmd_bt, log, timeout_s=600)

    # Parse metrics from stdout
    sharpe = 0.0
    max_dd = 100.0
    for line in result_bt.stdout.split("\n"):
        lower = line.lower()
        if "sharpe" in lower and ":" in line:
            try:
                sharpe = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "drawdown" in lower and "%" in line:
            try:
                # Extract percentage value
                import re
                m = re.search(r"([\d.]+)\s*%", line)
                if m:
                    max_dd = float(m.group(1))
            except (ValueError, IndexError):
                pass

    gates = config["gates"]
    metrics = {"sharpe": sharpe, "max_drawdown_pct": max_dd}
    passed = sharpe >= gates["min_oos_sharpe"] and max_dd <= gates["max_drawdown_pct"]
    verdict = "PASS" if passed else "FAIL"

    state.record_gate(f"backtest_{combo_name}", verdict, metrics)
    state.set_output(f"backtest_{combo_name}", metrics)
    log.info("  Backtest %s: Sharpe=%.2f DD=%.1f%%", verdict, sharpe, max_dd)

    return verdict, metrics


# ---------------------------------------------------------------------------
# Step 5: Alpha Pipeline
# ---------------------------------------------------------------------------


def step_alpha_pipeline(
    config: dict, state: DiscoveryState, combo: dict, log: logging.Logger,
) -> str:
    combo_name = f"{combo['symbol']}_{combo['horizon']}"

    log.info("-" * 40)
    log.info("  STEP 5: Alpha Pipeline — %s", combo_name)
    log.info("-" * 40)

    # Use a separate state file to avoid conflicting with manual alpha pipeline runs
    alpha_state_file = ROOT / "data" / "discovery" / "alpha_state.json"
    # Remove stale state so pipeline starts fresh
    if alpha_state_file.exists():
        alpha_state_file.unlink()

    cmd = [
        sys.executable, "scripts/alpha/alpha_pipeline.py",
        "--config", "config/alpha.toml",
        "start",
    ]
    result = run_subprocess(cmd, log, timeout_s=1800)

    # Read alpha pipeline state to check how far it got
    reached_phase = "IDLE"
    alpha_state_path = ROOT / "data" / "alpha" / "pipeline_state.json"
    if alpha_state_path.exists():
        try:
            with open(alpha_state_path) as f:
                alpha_state = json.load(f)
            reached_phase = alpha_state.get("phase", "IDLE")
        except Exception:
            pass

    min_phase = config["gates"].get("min_alpha_phase", "VALIDATING")
    # Simple phase ordering check
    phase_order = [
        "IDLE", "SCREENING", "COMBINING", "SIZING", "VALIDATING",
        "REGIME", "MULTI_FREQ", "PORTFOLIO", "PAPER", "DEPLOYING", "DONE",
    ]
    reached_idx = phase_order.index(reached_phase) if reached_phase in phase_order else 0
    min_idx = phase_order.index(min_phase) if min_phase in phase_order else 4

    verdict = "PASS" if reached_idx >= min_idx else "FAIL"
    state.record_gate(f"alpha_{combo_name}", verdict, {"reached_phase": reached_phase})
    log.info("  Alpha pipeline %s: reached %s (need %s)", verdict, reached_phase, min_phase)

    return verdict


# ---------------------------------------------------------------------------
# Step 6: Report
# ---------------------------------------------------------------------------


def step_report(
    config: dict, state: DiscoveryState, cycle_num: int, cycle_dir: Path,
    cycle_start: float, log: logging.Logger,
) -> None:
    log.info("=" * 60)
    log.info("  STEP 6: Report")
    log.info("=" * 60)

    summary = {
        "cycle": cycle_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - cycle_start, 1),
        "winners": state.get("winners", []),
        "gates": state.get("gates", {}),
        "step_outputs": state.get("step_outputs", {}),
    }

    summary_path = cycle_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    state.set_artifact("cycle_summary", str(summary_path))
    log.info("  Report: %s", summary_path)

    # Print summary
    n_winners = len(summary["winners"])
    n_gates = sum(1 for g in summary["gates"].values() if g["verdict"] == "PASS")
    n_total = len(summary["gates"])
    log.info("  Cycle %d complete: %d winners, %d/%d gates passed, %.0fs",
             cycle_num, n_winners, n_gates, n_total, summary["duration_s"])


# ---------------------------------------------------------------------------
# Cycle runner
# ---------------------------------------------------------------------------


def run_cycle(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Execute a single discovery cycle (steps 1-6)."""
    cycle_start = time.time()
    cycle_num = state.get("cycle_number", 0) + 1
    state.set("cycle_number", cycle_num)
    state.set("started_at", datetime.now(timezone.utc).isoformat())
    state.reset_cycle()

    report_dir = config["discovery"].get("report_dir", "reports/discovery")
    cycle_dir = ROOT / report_dir / f"cycle_{cycle_num:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  DISCOVERY CYCLE %d", cycle_num)
    log.info("=" * 60)

    # Step 1: Data Health
    state.transition(Phase.DATA_HEALTH, f"cycle {cycle_num} — data health")
    verdict = step_data_health(config, state, log)
    if verdict == "FAIL":
        log.error("  Data health failed — skipping cycle")
        state.transition(Phase.ERROR, "data health failed")
        step_report(config, state, cycle_num, cycle_dir, cycle_start, log)
        return
    if verdict == "WEAK":
        log.warning("  Data health warnings — continuing anyway")

    # Step 2: Signal Sweep
    state.transition(Phase.SIGNAL_SWEEP, f"cycle {cycle_num} — signal sweep")
    verdict, winners = step_signal_sweep(config, state, log, cycle_dir)
    if verdict == "FAIL":
        log.info("  No winners — cycle complete (normal)")
        state.transition(Phase.REPORTING, "no winners found")
        step_report(config, state, cycle_num, cycle_dir, cycle_start, log)
        return

    # Steps 3-5: Per winner
    for i, combo in enumerate(winners):
        combo_name = f"{combo['symbol']}_{combo['horizon']}"
        log.info("  Processing winner %d/%d: %s (edge=%.4f, gross=%.2f bps)",
                 i + 1, len(winners), combo_name, combo["edge"], combo["gross_bps"])
        state.set("current_combo", combo)

        # Step 3: Train
        state.transition(Phase.TRAINING, f"training {combo_name}")
        verdict, model_path = step_train(config, state, combo, cycle_num, cycle_dir, log)
        if verdict == "FAIL" or model_path is None:
            log.info("  Skipping %s — training failed", combo_name)
            continue

        # Step 4: Backtest
        state.transition(Phase.BACKTESTING, f"backtesting {combo_name}")
        verdict, metrics = step_backtest(config, state, combo, model_path, cycle_num, cycle_dir, log)
        if verdict == "FAIL":
            log.info("  Skipping %s — backtest failed", combo_name)
            continue

        # Step 5: Alpha Pipeline
        state.transition(Phase.ALPHA_PIPELINE, f"alpha pipeline {combo_name}")
        step_alpha_pipeline(config, state, combo, log)
        # Alpha pipeline result is logged but doesn't block the cycle

    # Step 6: Report
    state.transition(Phase.REPORTING, f"cycle {cycle_num} — reporting")
    step_report(config, state, cycle_num, cycle_dir, cycle_start, log)
    state.set("last_cycle_at", datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------


def run_daemon(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Continuous daemon — cycles until SIGTERM/SIGINT."""
    _shutdown = False

    def handle_signal(signum, frame):
        nonlocal _shutdown
        log.info("Received signal %d — shutting down after current step", signum)
        _shutdown = True

    signal_mod.signal(signal_mod.SIGTERM, handle_signal)
    signal_mod.signal(signal_mod.SIGINT, handle_signal)

    log.info("Discovery daemon started (cycle interval: %ds)",
             config["discovery"]["cycle_interval_s"])

    while not _shutdown:
        try:
            run_cycle(config, state, log)
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)
            state.transition(Phase.ERROR, str(e))

        if _shutdown:
            break

        interval = config["discovery"]["cycle_interval_s"]
        state.transition(Phase.SLEEPING, f"sleeping {interval}s")
        log.info("Sleeping %ds until next cycle...", interval)
        for _ in range(interval):
            if _shutdown:
                break
            time.sleep(1)

    state.transition(Phase.STOPPED, "graceful shutdown")
    log.info("Discovery daemon stopped")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_start(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Start the discovery daemon."""
    if state.current not in (Phase.IDLE, Phase.STOPPED, Phase.ERROR, Phase.SLEEPING):
        log.error("Cannot start — current phase is %s (use 'stop' first)", state.current.value)
        return
    state.transition(Phase.IDLE, "starting daemon")
    run_daemon(config, state, log)


def cmd_once(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Run a single discovery cycle."""
    state.transition(Phase.IDLE, "single cycle")
    run_cycle(config, state, log)
    state.transition(Phase.IDLE, "single cycle complete")


def cmd_status(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Display current orchestrator status."""
    print(f"Phase:        {state.current.value}")
    print(f"Cycle:        {state.get('cycle_number', 0)}")
    print(f"Started:      {state.get('started_at', 'never')}")
    print(f"Last cycle:   {state.get('last_cycle_at', 'never')}")
    print(f"Current:      {state.get('current_combo', 'none')}")

    winners = state.get("winners", [])
    if winners:
        print(f"\nWinners ({len(winners)}):")
        for w in winners:
            print(f"  {w['symbol']} h={w['horizon']}: edge={w['edge']:.4f} gross={w['gross_bps']:.2f}bp")

    gates = state.get("gates", {})
    if gates:
        print(f"\nGates:")
        for name, g in gates.items():
            print(f"  {name}: {g['verdict']}")

    error = state.get("error")
    if error:
        print(f"\nError: {error}")


def cmd_stop(config: dict, state: DiscoveryState, log: logging.Logger) -> None:
    """Signal the daemon to stop."""
    state.transition(Phase.STOPPED, "stop requested")
    print("Stop signal written to state file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NAT Alpha Discovery Orchestrator")
    parser.add_argument("--config", default="config/discovery.toml",
                        help="Config file (default: config/discovery.toml)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="Start discovery daemon")
    sub.add_parser("once", help="Run single cycle (for testing)")
    sub.add_parser("status", help="Show orchestrator status")
    sub.add_parser("stop", help="Signal daemon to stop")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = load_config(args.config)
    disc = config["discovery"]
    state = DiscoveryState(disc["state_file"])
    log = setup_logging(disc.get("log_file", "logs/discovery_orchestrator.log"))

    commands = {
        "start": cmd_start,
        "once": cmd_once,
        "status": cmd_status,
        "stop": cmd_stop,
    }
    commands[args.command](config, state, log)


if __name__ == "__main__":
    main()

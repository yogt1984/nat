"""
Live Deployment Framework (Alpha Roadmap Step 9).

Manages capital scaling, kill switches, and position limits for
transitioning from paper trading to live execution.

Scale-up schedule:
  - Week 1-2: 1% of capital (maker orders only)
  - Week 3-4: 5% of capital
  - Month 2-3: 10% of capital
  - Maximum: 25% of capital

Kill switches:
  - Daily loss > 1%: halt for 24 hours
  - Weekly DD > 2%: halt and review
  - Monthly DD > 5%: kill strategy
  - IC < 0 for 5 days: halt

Usage:
    python -m alpha.deployer --status          # show current state
    python -m alpha.deployer --check           # run kill switch checks
    python -m alpha.deployer --paper-report reports/alpha_paper.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = ROOT / "data" / "deploy_state.json"
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScaleSchedule:
    """Capital allocation schedule."""
    week: int  # current week since deployment start
    max_capital_pct: float  # max % of total capital
    order_type: str  # "maker" or "taker"
    description: str

    @staticmethod
    def get_schedule(weeks_deployed: int) -> "ScaleSchedule":
        if weeks_deployed <= 2:
            return ScaleSchedule(weeks_deployed, 1.0, "maker", "Week 1-2: minimal size, maker only")
        elif weeks_deployed <= 4:
            return ScaleSchedule(weeks_deployed, 5.0, "maker", "Week 3-4: small size, maker only")
        elif weeks_deployed <= 12:
            return ScaleSchedule(weeks_deployed, 10.0, "taker", "Month 2-3: moderate size")
        else:
            return ScaleSchedule(weeks_deployed, 25.0, "taker", "Month 4+: full size (capped 25%)")


@dataclass
class KillSwitch:
    """Kill switch state."""
    name: str
    threshold: float
    current_value: float
    triggered: bool
    action: str  # "halt_24h", "halt_review", "kill", "halt"
    description: str


@dataclass
class PositionLimits:
    """Per-symbol position limits."""
    symbol: str
    max_position_usd: float
    max_notional_pct: float  # % of account
    current_position_usd: float = 0.0
    utilization_pct: float = 0.0


@dataclass
class DeploymentState:
    """Complete deployment state snapshot."""
    is_live: bool
    start_date: Optional[str]
    weeks_deployed: int
    scale: ScaleSchedule
    kill_switches: List[KillSwitch]
    position_limits: List[PositionLimits]
    total_pnl_pct: float
    daily_pnl_pct: float
    weekly_pnl_pct: float
    monthly_pnl_pct: float
    any_kill_triggered: bool
    status: str  # "running", "halted", "killed", "not_started"


@dataclass
class DeploymentReadiness:
    """Pre-deployment readiness check."""
    paper_sharpe_ok: bool
    paper_max_loss_ok: bool
    paper_ic_ok: bool
    paper_infra_ok: bool
    paper_duration_ok: bool  # >= 14 days
    overall_ready: bool
    blockers: List[str]


# ---------------------------------------------------------------------------
# Kill switch evaluation
# ---------------------------------------------------------------------------


def evaluate_kill_switches(
    daily_pnl_pct: float = 0.0,
    weekly_dd_pct: float = 0.0,
    monthly_dd_pct: float = 0.0,
    ic_negative_days: int = 0,
) -> List[KillSwitch]:
    """Evaluate all kill switches against current metrics."""
    switches = [
        KillSwitch(
            name="daily_loss",
            threshold=1.0,
            current_value=abs(min(daily_pnl_pct, 0)),
            triggered=daily_pnl_pct < -1.0,
            action="halt_24h",
            description="Daily loss > 1%: halt trading for 24 hours",
        ),
        KillSwitch(
            name="weekly_drawdown",
            threshold=2.0,
            current_value=weekly_dd_pct,
            triggered=weekly_dd_pct > 2.0,
            action="halt_review",
            description="Weekly DD > 2%: halt and manual review required",
        ),
        KillSwitch(
            name="monthly_drawdown",
            threshold=5.0,
            current_value=monthly_dd_pct,
            triggered=monthly_dd_pct > 5.0,
            action="kill",
            description="Monthly DD > 5%: kill strategy permanently",
        ),
        KillSwitch(
            name="ic_decay",
            threshold=5.0,
            current_value=float(ic_negative_days),
            triggered=ic_negative_days >= 5,
            action="halt",
            description="IC < 0 for 5 consecutive days: halt and investigate",
        ),
    ]

    return switches


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------


def compute_position_limits(
    symbols: List[str],
    account_value_usd: float,
    max_capital_pct: float,
    per_symbol_max_pct: float = 40.0,
) -> List[PositionLimits]:
    """Compute per-symbol position limits based on scale schedule."""
    total_deployable = account_value_usd * max_capital_pct / 100.0
    per_symbol = total_deployable * per_symbol_max_pct / 100.0

    limits = []
    for sym in symbols:
        limits.append(PositionLimits(
            symbol=sym,
            max_position_usd=per_symbol,
            max_notional_pct=max_capital_pct * per_symbol_max_pct / 100.0,
        ))

    return limits


# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------


def check_readiness(
    paper_report_path: str = "reports/alpha_paper.json",
) -> DeploymentReadiness:
    """Check if paper trading results meet deployment criteria."""
    path = Path(paper_report_path)
    blockers = []

    if not path.exists():
        return DeploymentReadiness(
            paper_sharpe_ok=False, paper_max_loss_ok=False,
            paper_ic_ok=False, paper_infra_ok=False,
            paper_duration_ok=False, overall_ready=False,
            blockers=["Paper trading report not found"],
        )

    with open(path) as f:
        report = json.load(f)

    # G8 checks
    sharpe_ok = report.get("gate_sharpe_within_2x", False)
    if not sharpe_ok:
        blockers.append(f"Paper Sharpe ratio too low: {report.get('sharpe_ratio', 0):.2f}")

    loss_ok = report.get("gate_no_big_daily_loss", False)
    if not loss_ok:
        blockers.append(f"Max daily loss exceeded: {report.get('max_daily_loss_pct', 0):.2f}%")

    ic_ok = report.get("gate_ic_stable", False)
    if not ic_ok:
        blockers.append(f"IC decay detected: {report.get('ic_decay_pct', 0):.1f}%")

    infra_ok = report.get("gate_infra_stable", False)
    if not infra_ok:
        blockers.append("Infrastructure errors detected")

    duration_ok = report.get("n_days", 0) >= 14
    if not duration_ok:
        blockers.append(f"Paper trading duration: {report.get('n_days', 0)} days (need >= 14)")

    overall = sharpe_ok and loss_ok and ic_ok and infra_ok and duration_ok

    return DeploymentReadiness(
        paper_sharpe_ok=sharpe_ok,
        paper_max_loss_ok=loss_ok,
        paper_ic_ok=ic_ok,
        paper_infra_ok=infra_ok,
        paper_duration_ok=duration_ok,
        overall_ready=overall,
        blockers=blockers,
    )


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def get_deployment_state(
    symbols: Optional[List[str]] = None,
    account_value_usd: float = 10000.0,
) -> DeploymentState:
    """Load or initialize deployment state."""
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            saved = json.load(f)
        weeks = saved.get("weeks_deployed", 0)
        is_live = saved.get("is_live", False)
        start_date = saved.get("start_date")
    else:
        weeks = 0
        is_live = False
        start_date = None

    scale = ScaleSchedule.get_schedule(weeks)
    kills = evaluate_kill_switches()
    limits = compute_position_limits(symbols, account_value_usd, scale.max_capital_pct)
    any_triggered = any(k.triggered for k in kills)

    status = "not_started"
    if is_live:
        status = "killed" if any(k.action == "kill" and k.triggered for k in kills) else \
                 "halted" if any_triggered else "running"

    return DeploymentState(
        is_live=is_live,
        start_date=start_date,
        weeks_deployed=weeks,
        scale=scale,
        kill_switches=kills,
        position_limits=limits,
        total_pnl_pct=0.0,
        daily_pnl_pct=0.0,
        weekly_pnl_pct=0.0,
        monthly_pnl_pct=0.0,
        any_kill_triggered=any_triggered,
        status=status,
    )


def save_deployment_state(state: DeploymentState):
    """Persist deployment state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Live deployment framework (Step 9)")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--check", action="store_true", help="Run readiness check")
    parser.add_argument("--paper-report", default="reports/alpha_paper.json")
    parser.add_argument("--account-value", type=float, default=10000.0)
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.check:
        readiness = check_readiness(args.paper_report)
        print("\n=== Deployment Readiness Check ===")
        print(f"  Paper Sharpe OK:    {'PASS' if readiness.paper_sharpe_ok else 'FAIL'}")
        print(f"  Max Daily Loss OK:  {'PASS' if readiness.paper_max_loss_ok else 'FAIL'}")
        print(f"  IC Stability OK:    {'PASS' if readiness.paper_ic_ok else 'FAIL'}")
        print(f"  Infrastructure OK:  {'PASS' if readiness.paper_infra_ok else 'FAIL'}")
        print(f"  Duration OK (14d):  {'PASS' if readiness.paper_duration_ok else 'FAIL'}")
        print(f"\n  Overall: {'READY' if readiness.overall_ready else 'NOT READY'}")
        if readiness.blockers:
            print(f"\n  Blockers:")
            for b in readiness.blockers:
                print(f"    - {b}")
        return

    # Default: show status
    state = get_deployment_state(args.symbols, args.account_value)
    print("\n=== Deployment Status ===")
    print(f"  Status:   {state.status.upper()}")
    print(f"  Live:     {state.is_live}")
    print(f"  Week:     {state.weeks_deployed}")
    print(f"  Scale:    {state.scale.max_capital_pct}% capital ({state.scale.order_type})")
    print(f"  Schedule: {state.scale.description}")

    print(f"\n  Kill Switches:")
    for k in state.kill_switches:
        status = "TRIGGERED" if k.triggered else "OK"
        print(f"    {k.name}: {k.current_value:.2f} / {k.threshold:.1f} [{status}] → {k.action}")

    print(f"\n  Position Limits:")
    for p in state.position_limits:
        print(f"    {p.symbol}: max ${p.max_position_usd:.0f} ({p.max_notional_pct:.1f}%)")


if __name__ == "__main__":
    main()

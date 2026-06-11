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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


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
class PortfolioConstraints:
    """Portfolio-level risk constraint snapshot."""
    # Leverage
    gross_leverage: float
    max_leverage: float
    leverage_breached: bool
    leverage_action: str
    # Concentration
    herfindahl: float
    effective_n: float
    min_effective_n: float
    concentration_breached: bool
    # Portfolio drawdown
    equity_peak: float
    current_equity: float
    portfolio_dd_pct: float
    max_portfolio_dd_pct: float
    dd_breached: bool
    dd_action: str
    # VaR (parametric, 1-day)
    var_99_pct: float = 0.0  # 99% 1-day VaR as % of equity
    cvar_99_pct: float = 0.0  # 99% 1-day CVaR as % of equity
    # Aggregate
    any_breached: bool = False
    scale_factor: float = 1.0  # 1.0 if clean, max_leverage/gross_leverage if scaling


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
    portfolio_constraints: Optional[PortfolioConstraints] = None


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
# Portfolio-level risk constraints
# ---------------------------------------------------------------------------


def _load_portfolio_risk_config() -> dict:
    """Load [portfolio_risk] from config/alpha.toml with defaults."""
    defaults = {
        "max_leverage": 3.0,
        "min_effective_n": 1.5,
        "max_portfolio_dd_pct": 5.0,
        "dd_action": "halt_review",
        "leverage_action": "scale_down",
    }
    config_path = ROOT / "config" / "alpha.toml"
    if not config_path.exists():
        return defaults
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(config_path, "rb") as f:
        section = tomllib.load(f).get("portfolio_risk", {})
    return {**defaults, **section}


def check_portfolio_constraints(
    positions_usd: Dict[str, float],
    account_equity: float,
    equity_peak: float,
    max_leverage: float = 3.0,
    min_effective_n: float = 1.5,
    max_portfolio_dd_pct: float = 5.0,
    dd_action: str = "halt_review",
    leverage_action: str = "scale_down",
    symbol_vol_daily: Optional[Dict[str, float]] = None,
) -> PortfolioConstraints:
    """Evaluate portfolio-level risk constraints.

    Args:
        positions_usd: Symbol → signed position USD (positive=long, negative=short).
        account_equity: Current account equity in USD.
        equity_peak: Historical equity high-water mark.
        max_leverage: Gross leverage cap.
        min_effective_n: Minimum Herfindahl effective N (diversification).
        max_portfolio_dd_pct: Portfolio drawdown trigger (%).
        dd_action: Action on DD breach ("halt_24h", "halt_review", "kill").
        leverage_action: Action on leverage breach ("scale_down", "reject").
        symbol_vol_daily: Symbol → daily return volatility (decimal). Used for VaR.
    """
    # Leverage
    gross_notional = sum(abs(v) for v in positions_usd.values())
    gross_leverage = gross_notional / account_equity if account_equity > 0 else 0.0
    leverage_breached = gross_leverage > max_leverage

    # Concentration (Herfindahl)
    if gross_notional > 0:
        weights = [abs(v) / gross_notional for v in positions_usd.values()]
        herfindahl = sum(w * w for w in weights)
        effective_n = 1.0 / herfindahl if herfindahl > 0 else 0.0
    else:
        herfindahl = 0.0
        effective_n = float(len(positions_usd)) if positions_usd else 0.0
    concentration_breached = effective_n < min_effective_n and len(positions_usd) > 1

    # Portfolio drawdown
    current_equity = account_equity
    equity_peak = max(equity_peak, current_equity)
    portfolio_dd_pct = (
        (equity_peak - current_equity) / equity_peak * 100
        if equity_peak > 0
        else 0.0
    )
    dd_breached = portfolio_dd_pct > max_portfolio_dd_pct

    # Scale factor for leverage management
    if leverage_breached and leverage_action == "scale_down":
        scale_factor = max_leverage / gross_leverage
    else:
        scale_factor = 1.0

    # Parametric VaR/CVaR (1-day, 99%)
    # Assumes independent positions (conservative for hedged portfolios)
    var_99_pct = 0.0
    cvar_99_pct = 0.0
    if symbol_vol_daily and account_equity > 0:
        var_usd_sq = 0.0
        for sym, pos in positions_usd.items():
            vol = symbol_vol_daily.get(sym, 0.02)  # default 2% daily vol
            var_usd_sq += (pos * vol) ** 2
        portfolio_vol_usd = np.sqrt(var_usd_sq)
        # 99% VaR: z = 2.326
        var_99_pct = 2.326 * portfolio_vol_usd / account_equity * 100
        # 99% CVaR (Gaussian): E[X | X > z] = φ(z) / (1-Φ(z))
        # For 99%: CVaR multiplier ≈ 2.665
        cvar_99_pct = 2.665 * portfolio_vol_usd / account_equity * 100

    any_breached = leverage_breached or concentration_breached or dd_breached

    return PortfolioConstraints(
        gross_leverage=gross_leverage,
        max_leverage=max_leverage,
        leverage_breached=leverage_breached,
        leverage_action=leverage_action,
        herfindahl=herfindahl,
        effective_n=effective_n,
        min_effective_n=min_effective_n,
        concentration_breached=concentration_breached,
        equity_peak=equity_peak,
        current_equity=current_equity,
        portfolio_dd_pct=portfolio_dd_pct,
        max_portfolio_dd_pct=max_portfolio_dd_pct,
        dd_breached=dd_breached,
        dd_action=dd_action,
        var_99_pct=var_99_pct,
        cvar_99_pct=cvar_99_pct,
        any_breached=any_breached,
        scale_factor=scale_factor,
    )


def enforce_portfolio_constraints(
    proposed_positions: Dict[str, float],
    account_equity: float,
    equity_peak: float,
    symbol_vol_daily: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, float], PortfolioConstraints]:
    """Apply portfolio risk constraints to proposed positions.

    Returns adjusted positions (scaled/zeroed) and the constraint snapshot.
    This is the single enforcement point for all portfolio-level risk.

    Enforcement rules:
    1. DD breached + action "kill"/"halt_review" → zero all positions
    2. DD breached + action "halt_24h" → zero all positions
    3. Leverage breached + action "scale_down" → pro-rata scale to max
    4. Leverage breached + action "reject" → zero all positions
    5. Concentration breached → scale down largest position until effective_n OK
    """
    risk_cfg = _load_portfolio_risk_config()
    constraints = check_portfolio_constraints(
        positions_usd=proposed_positions,
        account_equity=account_equity,
        equity_peak=equity_peak,
        symbol_vol_daily=symbol_vol_daily,
        **risk_cfg,
    )

    adjusted = dict(proposed_positions)

    if not constraints.any_breached:
        return adjusted, constraints

    # DD breach → halt all trading
    if constraints.dd_breached:
        log.warning(
            "Portfolio DD %.1f%% > %.1f%% — zeroing all positions (action=%s)",
            constraints.portfolio_dd_pct,
            constraints.max_portfolio_dd_pct,
            constraints.dd_action,
        )
        adjusted = {sym: 0.0 for sym in adjusted}
        return adjusted, constraints

    # Leverage breach
    if constraints.leverage_breached:
        if constraints.leverage_action == "scale_down":
            log.warning(
                "Leverage %.2f > %.2f — scaling positions by %.2f",
                constraints.gross_leverage,
                constraints.max_leverage,
                constraints.scale_factor,
            )
            adjusted = {sym: pos * constraints.scale_factor for sym, pos in adjusted.items()}
        else:  # "reject"
            log.warning(
                "Leverage %.2f > %.2f — rejecting all positions",
                constraints.gross_leverage,
                constraints.max_leverage,
            )
            adjusted = {sym: 0.0 for sym in adjusted}
            return adjusted, constraints

    # Concentration breach — scale down largest until effective_n passes
    if constraints.concentration_breached and len(adjusted) > 1:
        gross = sum(abs(v) for v in adjusted.values())
        if gross > 0:
            sorted_syms = sorted(adjusted, key=lambda s: abs(adjusted[s]), reverse=True)
            largest = sorted_syms[0]
            # Scale largest position to equalize with second-largest
            second = abs(adjusted[sorted_syms[1]]) if len(sorted_syms) > 1 else gross / 2
            cap = max(second, gross * 0.5)  # at most 50% of gross in one name
            if abs(adjusted[largest]) > cap:
                sign = 1.0 if adjusted[largest] > 0 else -1.0
                log.warning(
                    "Concentration breach (eff_n=%.2f) — capping %s from %.0f to %.0f",
                    constraints.effective_n, largest, abs(adjusted[largest]), cap,
                )
                adjusted[largest] = sign * cap

    return adjusted, constraints


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
    positions_usd: Optional[Dict[str, float]] = None,
    equity_peak: Optional[float] = None,
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
        saved_peak = saved.get("equity_peak", account_value_usd)
    else:
        weeks = 0
        is_live = False
        start_date = None
        saved_peak = account_value_usd

    scale = ScaleSchedule.get_schedule(weeks)
    kills = evaluate_kill_switches()
    limits = compute_position_limits(symbols, account_value_usd, scale.max_capital_pct)
    any_triggered = any(k.triggered for k in kills)

    # Portfolio constraints (when position data is available)
    port_constraints = None
    if positions_usd is not None:
        risk_cfg = _load_portfolio_risk_config()
        peak = equity_peak if equity_peak is not None else saved_peak
        port_constraints = check_portfolio_constraints(
            positions_usd=positions_usd,
            account_equity=account_value_usd,
            equity_peak=peak,
            **risk_cfg,
        )

    port_dd_kill = (
        port_constraints is not None
        and port_constraints.dd_breached
        and port_constraints.dd_action == "kill"
    )
    port_halted = port_constraints is not None and port_constraints.dd_breached

    status = "not_started"
    if is_live:
        status = (
            "killed" if (any(k.action == "kill" and k.triggered for k in kills) or port_dd_kill)
            else "halted" if (any_triggered or port_halted)
            else "running"
        )

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
        portfolio_constraints=port_constraints,
    )


def save_deployment_state(state: DeploymentState):
    """Persist deployment state (includes equity_peak for DD tracking)."""
    data = asdict(state)
    # Persist equity_peak at top level for easy reload
    if state.portfolio_constraints is not None:
        data["equity_peak"] = state.portfolio_constraints.equity_peak
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


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

    if state.portfolio_constraints is not None:
        pc = state.portfolio_constraints
        print(f"\n  Portfolio Constraints:")
        print(f"    Leverage:      {pc.gross_leverage:.2f}x / {pc.max_leverage:.1f}x "
              f"[{'BREACH' if pc.leverage_breached else 'OK'}]")
        print(f"    Effective N:   {pc.effective_n:.1f} / {pc.min_effective_n:.1f} "
              f"[{'BREACH' if pc.concentration_breached else 'OK'}]")
        print(f"    Portfolio DD:  {pc.portfolio_dd_pct:.1f}% / {pc.max_portfolio_dd_pct:.1f}% "
              f"[{'BREACH' if pc.dd_breached else 'OK'}]")
        print(f"    Scale Factor:  {pc.scale_factor:.2f}")


if __name__ == "__main__":
    main()

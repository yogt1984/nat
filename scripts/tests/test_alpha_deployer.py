"""Tests for alpha.deployer — kill switches, position limits, readiness."""

import json
import tempfile
from pathlib import Path


import numpy as np
import pytest
from alpha.deployer import (
    ScaleSchedule,
    KillSwitch,
    PositionLimits,
    PortfolioConstraints,
    DeploymentReadiness,
    evaluate_kill_switches,
    compute_position_limits,
    check_portfolio_constraints,
    check_readiness,
)


# ---------------------------------------------------------------------------
# ScaleSchedule
# ---------------------------------------------------------------------------


class TestScaleSchedule:
    def test_week_1(self):
        s = ScaleSchedule.get_schedule(1)
        assert s.max_capital_pct == 1.0
        assert s.order_type == "maker"

    def test_week_2(self):
        s = ScaleSchedule.get_schedule(2)
        assert s.max_capital_pct == 1.0
        assert s.order_type == "maker"

    def test_week_3(self):
        s = ScaleSchedule.get_schedule(3)
        assert s.max_capital_pct == 5.0
        assert s.order_type == "maker"

    def test_week_4(self):
        s = ScaleSchedule.get_schedule(4)
        assert s.max_capital_pct == 5.0

    def test_month_2(self):
        s = ScaleSchedule.get_schedule(8)
        assert s.max_capital_pct == 10.0
        assert s.order_type == "taker"

    def test_month_4_plus(self):
        s = ScaleSchedule.get_schedule(20)
        assert s.max_capital_pct == 25.0
        assert s.order_type == "taker"

    def test_week_0(self):
        s = ScaleSchedule.get_schedule(0)
        assert s.max_capital_pct == 1.0  # still in week 1-2 bracket


# ---------------------------------------------------------------------------
# evaluate_kill_switches
# ---------------------------------------------------------------------------


class TestEvaluateKillSwitches:
    def test_no_triggers(self):
        switches = evaluate_kill_switches(
            daily_pnl_pct=0.5, weekly_dd_pct=0.5,
            monthly_dd_pct=1.0, ic_negative_days=0,
        )
        assert len(switches) == 4
        assert not any(k.triggered for k in switches)

    def test_daily_loss_triggers(self):
        switches = evaluate_kill_switches(daily_pnl_pct=-1.5)
        daily = next(k for k in switches if k.name == "daily_loss")
        assert daily.triggered is True
        assert daily.action == "halt_24h"

    def test_weekly_dd_triggers(self):
        switches = evaluate_kill_switches(weekly_dd_pct=2.5)
        weekly = next(k for k in switches if k.name == "weekly_drawdown")
        assert weekly.triggered is True
        assert weekly.action == "halt_review"

    def test_monthly_dd_triggers(self):
        switches = evaluate_kill_switches(monthly_dd_pct=6.0)
        monthly = next(k for k in switches if k.name == "monthly_drawdown")
        assert monthly.triggered is True
        assert monthly.action == "kill"

    def test_ic_decay_triggers(self):
        switches = evaluate_kill_switches(ic_negative_days=5)
        ic = next(k for k in switches if k.name == "ic_decay")
        assert ic.triggered is True
        assert ic.action == "halt"

    def test_ic_decay_below_threshold(self):
        switches = evaluate_kill_switches(ic_negative_days=4)
        ic = next(k for k in switches if k.name == "ic_decay")
        assert ic.triggered is False

    def test_boundary_values(self):
        # Exactly at threshold should NOT trigger (daily_pnl > -1.0)
        switches = evaluate_kill_switches(daily_pnl_pct=-1.0)
        daily = next(k for k in switches if k.name == "daily_loss")
        assert daily.triggered is False

        # Just below threshold triggers
        switches = evaluate_kill_switches(daily_pnl_pct=-1.01)
        daily = next(k for k in switches if k.name == "daily_loss")
        assert daily.triggered is True


# ---------------------------------------------------------------------------
# compute_position_limits
# ---------------------------------------------------------------------------


class TestComputePositionLimits:
    def test_basic(self):
        limits = compute_position_limits(
            ["BTC", "ETH", "SOL"],
            account_value_usd=10000,
            max_capital_pct=5.0,
            per_symbol_max_pct=40.0,
        )
        assert len(limits) == 3
        # total deployable = 10000 * 5% = 500
        # per symbol = 500 * 40% = 200
        assert limits[0].max_position_usd == pytest.approx(200.0)
        assert limits[0].max_notional_pct == pytest.approx(2.0)  # 5% * 40%

    def test_single_symbol(self):
        limits = compute_position_limits(
            ["BTC"], account_value_usd=100000, max_capital_pct=25.0,
        )
        assert len(limits) == 1
        # 100000 * 25% = 25000; per symbol = 25000 * 40% = 10000
        assert limits[0].max_position_usd == pytest.approx(10000.0)

    def test_zero_capital(self):
        limits = compute_position_limits(["BTC"], account_value_usd=0, max_capital_pct=5.0)
        assert limits[0].max_position_usd == 0.0


# ---------------------------------------------------------------------------
# check_readiness
# ---------------------------------------------------------------------------


class TestCheckReadiness:
    def test_report_not_found(self):
        readiness = check_readiness("/nonexistent/report.json")
        assert readiness.overall_ready is False
        assert len(readiness.blockers) > 0

    def test_all_gates_pass(self):
        report = {
            "gate_sharpe_within_2x": True,
            "gate_no_big_daily_loss": True,
            "gate_ic_stable": True,
            "gate_infra_stable": True,
            "n_days": 14,
            "sharpe_ratio": 1.2,
            "max_daily_loss_pct": -0.5,
            "ic_decay_pct": 10,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            f.flush()
            readiness = check_readiness(f.name)

        assert readiness.overall_ready is True
        assert readiness.paper_sharpe_ok is True
        assert readiness.paper_duration_ok is True
        assert len(readiness.blockers) == 0

    def test_duration_too_short(self):
        report = {
            "gate_sharpe_within_2x": True,
            "gate_no_big_daily_loss": True,
            "gate_ic_stable": True,
            "gate_infra_stable": True,
            "n_days": 7,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            f.flush()
            readiness = check_readiness(f.name)

        assert readiness.overall_ready is False
        assert readiness.paper_duration_ok is False
        assert any("duration" in b.lower() or "days" in b.lower() for b in readiness.blockers)

    def test_sharpe_fails(self):
        report = {
            "gate_sharpe_within_2x": False,
            "gate_no_big_daily_loss": True,
            "gate_ic_stable": True,
            "gate_infra_stable": True,
            "n_days": 14,
            "sharpe_ratio": 0.3,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            f.flush()
            readiness = check_readiness(f.name)

        assert readiness.overall_ready is False
        assert readiness.paper_sharpe_ok is False


# ---------------------------------------------------------------------------
# check_portfolio_constraints
# ---------------------------------------------------------------------------


class TestCheckPortfolioConstraints:
    def test_no_breach(self):
        """Normal positions, all within limits."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 5000, "ETH": 3000, "SOL": 2000},
            account_equity=10000,
            equity_peak=10000,
        )
        assert result.gross_leverage == pytest.approx(1.0)
        assert not result.leverage_breached
        assert not result.concentration_breached
        assert not result.dd_breached
        assert not result.any_breached
        assert result.scale_factor == pytest.approx(1.0)

    def test_leverage_breach_scale_down(self):
        """Gross leverage exceeds 3x, scale_down mode."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 20000, "ETH": 15000},
            account_equity=10000,
            equity_peak=10000,
            max_leverage=3.0,
            leverage_action="scale_down",
        )
        assert result.gross_leverage == pytest.approx(3.5)
        assert result.leverage_breached
        assert result.scale_factor == pytest.approx(3.0 / 3.5)

    def test_leverage_breach_reject(self):
        """Gross leverage exceeds limit, reject mode — scale_factor stays 1.0."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 20000, "ETH": 15000},
            account_equity=10000,
            equity_peak=10000,
            leverage_action="reject",
        )
        assert result.leverage_breached
        assert result.scale_factor == pytest.approx(1.0)

    def test_concentration_breach(self):
        """One position dominates."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 9000, "ETH": 500, "SOL": 500},
            account_equity=10000,
            equity_peak=10000,
            min_effective_n=2.0,
        )
        assert result.concentration_breached
        assert result.effective_n < 2.0

    def test_concentration_single_symbol_not_breached(self):
        """Single symbol should not trigger concentration breach."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 5000},
            account_equity=10000,
            equity_peak=10000,
            min_effective_n=2.0,
        )
        assert not result.concentration_breached

    def test_portfolio_dd_breach(self):
        """Equity drawdown from peak triggers circuit breaker."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 3000},
            account_equity=9000,
            equity_peak=10000,
            max_portfolio_dd_pct=5.0,
        )
        assert result.portfolio_dd_pct == pytest.approx(10.0)
        assert result.dd_breached

    def test_empty_positions(self):
        """No positions — everything clean."""
        result = check_portfolio_constraints(
            positions_usd={},
            account_equity=10000,
            equity_peak=10000,
        )
        assert result.gross_leverage == pytest.approx(0.0)
        assert not result.any_breached

    def test_equity_peak_updates(self):
        """equity_peak should be max(passed_peak, current_equity)."""
        result = check_portfolio_constraints(
            positions_usd={"BTC": 3000},
            account_equity=11000,
            equity_peak=10000,
        )
        assert result.equity_peak == 11000.0
        assert result.portfolio_dd_pct == pytest.approx(0.0)

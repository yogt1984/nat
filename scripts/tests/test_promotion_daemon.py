"""Planted tests for the T14 promotion daemon.

Test-first (red). The daemon drives the signal_lifecycle state machine through
automatic transitions, gating each on imported thresholds (G4 = config g4_*,
G8 = g8_*, g8_min_days=14) and a data-sufficiency + >=7-clean-day guard. The
expensive steps are seams (_run_g4 / _run_paper / _count_clean_days /
_days_in_paper / _data_sufficient / _check_g8 / _check_decay) overridable per
test, so transitions are exercised deterministically without shelling out.

Non-negotiable: the daemon NEVER promotes APPROVAL_PENDING -> LIVE (the sole
human gate). It stops at APPROVAL_PENDING.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from signal_lifecycle import (
    SignalLifecycle,
    DISCOVERED, VALIDATED, PAPER_TRADING, APPROVAL_PENDING, LIVE, MONITORING, RETIRED,
)
from promotion_daemon import PromotionDaemon, build_paper_report, healthy


@pytest.fixture
def lc(tmp_path):
    s = SignalLifecycle(db_path=tmp_path / "nat.db")
    yield s
    s.close()


def _seed(lc, sid, state):
    """Walk a signal from DISCOVERED up to the requested state."""
    lc.discover(sid, name=sid)
    if state == DISCOVERED:
        return
    lc.validate(sid)
    if state == VALIDATED:
        return
    lc.start_paper(sid)
    if state == PAPER_TRADING:
        return
    lc.request_approval(sid)
    if state == APPROVAL_PENDING:
        return
    lc.approve(sid)
    if state == LIVE:
        return
    lc.monitor(sid)  # MONITORING


def _daemon(lc, tmp_path, *, dry_run=False, clock=None):
    """Daemon with all seams defaulted to the happy path; override per test."""
    d = PromotionDaemon(
        lc=lc,
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb",
        pid_file=tmp_path / "pid",
        dry_run=dry_run,
        clock=clock,
    )
    d._count_clean_days = lambda: 10
    d._data_sufficient = lambda row: True
    d._run_g4 = lambda row: {"gate_pass": True, "metrics": {"oos_sharpe": 0.7}}
    d._run_paper = lambda row: {"ok": True}
    d._days_in_paper = lambda row: 20
    d._check_g8 = lambda row: (True, {"n_days": 20})
    d._check_decay = lambda row: False
    return d


def _state(lc, sid):
    return lc.get_signal(sid)["state"]


# ---------------------------------------------------------------------------
# DISCOVERED -> VALIDATED (G4 gate + data guards)
# ---------------------------------------------------------------------------


class TestDiscoveredToValidated:
    def test_promotes_on_g4_pass(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path)
        assert d.process_signal(lc.get_signal("algoA")) == "validated"
        assert _state(lc, "algoA") == VALIDATED

    def test_blocked_under_7_clean_days(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path)
        d._count_clean_days = lambda: 3
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == DISCOVERED

    def test_blocked_when_data_insufficient(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path)
        d._data_sufficient = lambda row: False
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == DISCOVERED

    def test_blocked_on_g4_fail(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path)
        d._run_g4 = lambda row: {"gate_pass": False, "metrics": {}}
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == DISCOVERED

    def test_g4_timeout_skips_cleanly(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path)
        d._run_g4 = lambda row: None  # subprocess timeout/error
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == DISCOVERED


# ---------------------------------------------------------------------------
# VALIDATED -> PAPER_TRADING
# ---------------------------------------------------------------------------


class TestValidatedToPaper:
    def test_starts_paper(self, lc, tmp_path):
        _seed(lc, "algoA", VALIDATED)
        d = _daemon(lc, tmp_path)
        assert d.process_signal(lc.get_signal("algoA")) == "started_paper"
        assert _state(lc, "algoA") == PAPER_TRADING

    def test_no_transition_when_paper_launch_fails(self, lc, tmp_path):
        _seed(lc, "algoA", VALIDATED)
        d = _daemon(lc, tmp_path)
        d._run_paper = lambda row: None
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == VALIDATED


# ---------------------------------------------------------------------------
# PAPER_TRADING -> APPROVAL_PENDING (>=14 days + G8)
# ---------------------------------------------------------------------------


class TestPaperToApproval:
    def test_promotes_after_14d_and_g8(self, lc, tmp_path):
        _seed(lc, "algoA", PAPER_TRADING)
        d = _daemon(lc, tmp_path)
        assert d.process_signal(lc.get_signal("algoA")) == "approval_pending"
        assert _state(lc, "algoA") == APPROVAL_PENDING

    def test_blocked_under_14_days(self, lc, tmp_path):
        _seed(lc, "algoA", PAPER_TRADING)
        d = _daemon(lc, tmp_path)
        d._days_in_paper = lambda row: 5
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == PAPER_TRADING

    def test_blocked_on_g8_fail(self, lc, tmp_path):
        _seed(lc, "algoA", PAPER_TRADING)
        d = _daemon(lc, tmp_path)
        d._check_g8 = lambda row: (False, {"n_days": 20})
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == PAPER_TRADING


# ---------------------------------------------------------------------------
# Human gate + decay
# ---------------------------------------------------------------------------


class TestHumanGateAndDecay:
    def test_never_promotes_to_live(self, lc, tmp_path):
        # APPROVAL_PENDING -> LIVE is the sole HUMAN gate; the daemon must not do it.
        _seed(lc, "algoA", APPROVAL_PENDING)
        d = _daemon(lc, tmp_path)
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == APPROVAL_PENDING

    def test_live_retires_on_decay(self, lc, tmp_path):
        _seed(lc, "algoA", LIVE)
        d = _daemon(lc, tmp_path)
        d._check_decay = lambda row: True
        assert d.process_signal(lc.get_signal("algoA")) == "retired"
        assert _state(lc, "algoA") == RETIRED

    def test_live_stays_when_healthy(self, lc, tmp_path):
        _seed(lc, "algoA", LIVE)
        d = _daemon(lc, tmp_path)
        assert d.process_signal(lc.get_signal("algoA")) is None
        assert _state(lc, "algoA") == LIVE


# ---------------------------------------------------------------------------
# Dry-run + cycle
# ---------------------------------------------------------------------------


class TestDryRunAndCycle:
    def test_dry_run_reports_but_does_not_mutate(self, lc, tmp_path):
        _seed(lc, "algoA", DISCOVERED)
        d = _daemon(lc, tmp_path, dry_run=True)
        action = d.process_signal(lc.get_signal("algoA"))
        assert action == "validated"          # intended action reported
        assert _state(lc, "algoA") == DISCOVERED  # but no transition applied

    def test_run_cycle_processes_all_and_summarizes(self, lc, tmp_path):
        _seed(lc, "a", DISCOVERED)
        _seed(lc, "b", VALIDATED)
        _seed(lc, "c", APPROVAL_PENDING)  # untouched (human gate)
        d = _daemon(lc, tmp_path)
        summary = d.run_cycle()
        assert _state(lc, "a") == VALIDATED
        assert _state(lc, "b") == PAPER_TRADING
        assert _state(lc, "c") == APPROVAL_PENDING
        assert summary["transitions"] == 2


# ---------------------------------------------------------------------------
# build_paper_report — G8 boolean mapping (imported thresholds)
# ---------------------------------------------------------------------------


_GATES = {
    "g8_min_sharpe_ratio": 0.5,
    "g8_max_daily_loss_pct": 2.0,
    "g8_max_ic_decay_pct": 50.0,
    "g8_min_days": 14,
}


class TestBuildPaperReport:
    def test_all_pass(self):
        r = build_paper_report(
            {"paper_sharpe": 1.0, "baseline_sharpe": 1.0, "max_daily_loss_pct": 0.5,
             "ic_decay_pct": 10.0, "infra_stable": True, "n_days": 20},
            _GATES,
        )
        assert r["gate_sharpe_within_2x"] is True
        assert r["gate_no_big_daily_loss"] is True
        assert r["gate_ic_stable"] is True
        assert r["gate_infra_stable"] is True
        assert r["n_days"] == 20

    def test_sharpe_ratio_below_min_fails(self):
        r = build_paper_report(
            {"paper_sharpe": 0.2, "baseline_sharpe": 1.0, "max_daily_loss_pct": 0.5,
             "ic_decay_pct": 10.0, "infra_stable": True, "n_days": 20},
            _GATES,
        )
        assert r["gate_sharpe_within_2x"] is False

    def test_big_daily_loss_fails(self):
        r = build_paper_report(
            {"paper_sharpe": 1.0, "baseline_sharpe": 1.0, "max_daily_loss_pct": 3.5,
             "ic_decay_pct": 10.0, "infra_stable": True, "n_days": 20},
            _GATES,
        )
        assert r["gate_no_big_daily_loss"] is False


# ---------------------------------------------------------------------------
# Healthcheck heartbeat
# ---------------------------------------------------------------------------


class TestHealthcheck:
    def test_missing_heartbeat_unhealthy(self, tmp_path):
        assert healthy({"heartbeat_path": str(tmp_path / "hb"), "poll_interval_s": 300}) is False

    def test_fresh_heartbeat_healthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("now")
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 300}) is True

    def test_stale_heartbeat_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("old")
        os.utime(hb, (1_700_000_000.0, 1_700_000_000.0))
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 300}) is False

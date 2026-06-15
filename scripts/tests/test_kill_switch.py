"""Planted tests for risk.kill_switch — the T16 kill-switch daemon.

Test-first (red) per METHODOLOGY: these encode the contract before any
implementation exists. Thresholds are imported from ROADMAP Step 9 (via
alpha.deployer.evaluate_kill_switches) — never invented here:

    daily loss  > 1%  -> halt_24h      (auto-resume after 24h)
    weekly DD   > 2%  -> halt_review    (manual `nat risk resume --confirm`)
    monthly DD  > 5%  -> kill_strategy  (refuse resume; pipeline re-run required)
    IC < 0 for 5d     -> halt           (manual resume after investigation)

The synthetic series in each test isolate a single threshold so the planted
breach can only trip the gate under test.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from risk.kill_switch import (
    KillSwitch,
    HaltState,
    max_drawdown_pct,
    consecutive_negative,
    daily_ic,
    compute_kill_metrics,
    read_halt_state,
    write_halt_state,
    effective_level,
    load_pnl_history,
)


# ---------------------------------------------------------------------------
# Pure estimator helpers (the bug-prone bits — planted tests pin them)
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown_monotonic_up(self):
        assert max_drawdown_pct([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_empty_is_zero(self):
        assert max_drawdown_pct([]) == pytest.approx(0.0)

    def test_window_relative_single_loss(self):
        # window opens at 0 equity; a single -1.5% day is a 1.5% drawdown
        assert max_drawdown_pct([-1.5]) == pytest.approx(1.5)

    def test_peak_to_trough(self):
        # equity 0 -> +1 -> -2 -> -1 ; peak 1, trough -2 => DD 3
        assert max_drawdown_pct([1.0, -3.0, 1.0]) == pytest.approx(3.0)

    def test_always_non_negative(self):
        assert max_drawdown_pct([-0.5, -0.5, -0.5]) >= 0.0


class TestConsecutiveNegative:
    def test_all_negative(self):
        assert consecutive_negative([-0.1, -0.2, -0.3]) == 3

    def test_counts_only_trailing(self):
        assert consecutive_negative([-0.1, 0.2, -0.3, -0.4]) == 2

    def test_positive_tail_is_zero(self):
        assert consecutive_negative([-0.1, -0.2, 0.05]) == 0

    def test_nan_breaks_streak(self):
        # missing IC must not count as a negative day
        assert consecutive_negative([-0.1, float("nan"), -0.2, -0.3]) == 2

    def test_empty(self):
        assert consecutive_negative([]) == 0


class TestDailyIC:
    def test_perfect_negative_correlation(self):
        sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        fwd = [-x for x in sig]
        assert daily_ic(sig, fwd, min_obs=5) == pytest.approx(-1.0)

    def test_perfect_positive_correlation(self):
        sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert daily_ic(sig, list(sig), min_obs=5) == pytest.approx(1.0)

    def test_below_min_obs_is_nan(self):
        assert np.isnan(daily_ic([1.0, 2.0], [1.0, 2.0], min_obs=5))


# ---------------------------------------------------------------------------
# Threshold metrics -> evaluate_kill_switches (imported thresholds)
# ---------------------------------------------------------------------------


def _pnl(days):
    """Build pnl_history records from a list of daily pct returns."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    return [
        {"date": (base + timedelta(days=i)).strftime("%Y-%m-%d"), "pnl_pct": float(v)}
        for i, v in enumerate(days)
    ]


class TestKillMetricsThresholds:
    def test_daily_loss_triggers_halt_24h(self):
        m = compute_kill_metrics(_pnl([-1.5]), ic_series=[])
        assert m["daily_pnl_pct"] == pytest.approx(-1.5)
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert sw["daily_loss"].triggered
        assert sw["daily_loss"].action == "halt_24h"

    def test_daily_loss_below_threshold_no_trigger(self):
        m = compute_kill_metrics(_pnl([-0.5]), ic_series=[])
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert not sw["daily_loss"].triggered

    def test_weekly_drawdown_triggers_halt_review(self):
        # 5 days of -0.5% => 2.5% window drawdown (>2), each day -0.5 (>-1 daily)
        m = compute_kill_metrics(_pnl([-0.5, -0.5, -0.5, -0.5, -0.5]), ic_series=[])
        assert m["weekly_dd_pct"] == pytest.approx(2.5)
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert sw["weekly_drawdown"].triggered
        assert sw["weekly_drawdown"].action == "halt_review"
        assert not sw["daily_loss"].triggered

    def test_monthly_drawdown_triggers_kill(self):
        m = compute_kill_metrics(_pnl([-0.6] * 10), ic_series=[])
        assert m["monthly_dd_pct"] == pytest.approx(6.0)
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert sw["monthly_drawdown"].triggered

    def test_ic_decay_triggers_halt(self):
        m = compute_kill_metrics(_pnl([0.1, 0.1, 0.1]), ic_series=[-0.05] * 5)
        assert m["ic_negative_days"] == 5
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert sw["ic_decay"].triggered
        assert sw["ic_decay"].action == "halt"

    def test_ic_decay_four_days_no_trigger(self):
        m = compute_kill_metrics(_pnl([0.1, 0.1, 0.1]), ic_series=[-0.05] * 4)
        sw = {s.name: s for s in evaluate_kill_switches_local(m)}
        assert not sw["ic_decay"].triggered

    def test_empty_data_no_trigger(self):
        m = compute_kill_metrics([], ic_series=[])
        sw = evaluate_kill_switches_local(m)
        assert not any(s.triggered for s in sw)


# ---------------------------------------------------------------------------
# effective_level — severity ordering
# ---------------------------------------------------------------------------


class TestEffectiveLevel:
    def test_kill_strategy_dominates(self):
        assert effective_level(["halt_24h", "kill_strategy", "halt_review"]) == "kill_strategy"

    def test_review_over_auto(self):
        assert effective_level(["halt_24h", "halt_review"]) == "halt_review"

    def test_none_when_empty(self):
        assert effective_level([]) is None


# ---------------------------------------------------------------------------
# Halt-state persistence (the IPC contract T17 reads)
# ---------------------------------------------------------------------------


class TestHaltStatePersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "halt_state.json"
        st = HaltState(
            halted=True, level="halt_review", reason="weekly DD 2.5%",
            triggered=["weekly_drawdown"], metrics={"weekly_dd_pct": 2.5},
            halted_at="2026-05-10T00:00:00+00:00", resume_at=None, git_sha="abc123",
        )
        write_halt_state(path, st)
        loaded = read_halt_state(path)
        assert loaded.halted
        assert loaded.level == "halt_review"
        assert loaded.triggered == ["weekly_drawdown"]

    def test_absent_file_is_not_halted(self, tmp_path):
        st = read_halt_state(tmp_path / "missing.json")
        assert not st.halted
        assert st.level is None


# ---------------------------------------------------------------------------
# KillSwitch controller — check() writes halt state; resume rules enforced
# ---------------------------------------------------------------------------


def _ks(tmp_path, clock=None):
    return KillSwitch(halt_path=tmp_path / "halt_state.json",
                      audit_path=tmp_path / "halt_history.jsonl",
                      db_path=None, clock=clock)


class TestCheckWritesHaltState:
    def test_daily_loss_writes_halt_24h(self, tmp_path):
        ks = _ks(tmp_path)
        st = ks.check(pnl_history=_pnl([-1.5]), ic_series=[])
        assert st.halted
        assert st.level == "halt_24h"
        assert st.resume_at is not None  # auto-resume scheduled
        # persisted to disk
        assert read_halt_state(tmp_path / "halt_state.json").level == "halt_24h"

    def test_monthly_writes_kill_strategy(self, tmp_path):
        ks = _ks(tmp_path)
        st = ks.check(pnl_history=_pnl([-0.6] * 10), ic_series=[])
        assert st.halted
        assert st.level == "kill_strategy"

    def test_clean_data_no_halt(self, tmp_path):
        ks = _ks(tmp_path)
        st = ks.check(pnl_history=_pnl([0.2, 0.3, 0.1]), ic_series=[0.05, 0.04])
        assert not st.halted


class TestResumeRules:
    def test_resume_when_not_halted(self, tmp_path):
        ks = _ks(tmp_path)
        ok, _ = ks.resume(confirm=True)
        assert not ok  # nothing to resume

    def test_kill_strategy_refuses_resume(self, tmp_path):
        ks = _ks(tmp_path)
        ks.check(pnl_history=_pnl([-0.6] * 10), ic_series=[])
        ok, msg = ks.resume(confirm=True)
        assert not ok
        assert "pipeline" in msg.lower()

    def test_halt_review_requires_confirm(self, tmp_path):
        ks = _ks(tmp_path)
        ks.check(pnl_history=_pnl([-0.5] * 5), ic_series=[])
        ok, _ = ks.resume(confirm=False)
        assert not ok
        ok, _ = ks.resume(confirm=True)
        assert ok
        assert not read_halt_state(tmp_path / "halt_state.json").halted

    def test_halt_24h_auto_resume_after_window(self, tmp_path):
        t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        clk = {"now": t0}
        ks = _ks(tmp_path, clock=lambda: clk["now"])
        ks.check(pnl_history=_pnl([-1.5]), ic_series=[])
        # before 24h: still halted even on clean data (cooldown)
        clk["now"] = t0 + timedelta(hours=12)
        st = ks.check(pnl_history=_pnl([0.2]), ic_series=[])
        assert st.halted
        # after 24h: auto-resumes
        clk["now"] = t0 + timedelta(hours=24, minutes=1)
        st = ks.check(pnl_history=_pnl([0.2]), ic_series=[])
        assert not st.halted


# ---------------------------------------------------------------------------
# PnL loader — prefer exec log (pct); fall back to paper trades (bps -> pct)
# ---------------------------------------------------------------------------


class TestLoadPnlHistory:
    def test_prefers_exec_log_pct(self, tmp_path):
        exec_log = tmp_path / "daily_pnl.json"
        exec_log.write_text(json.dumps([
            {"date": "2026-05-01", "pnl_pct": 0.5},
            {"date": "2026-05-02", "pnl_pct": -0.3},
        ]))
        hist = load_pnl_history(exec_log=exec_log, paper_report=tmp_path / "nope.json")
        assert [h["pnl_pct"] for h in hist] == [0.5, -0.3]

    def test_falls_back_to_paper_bps_to_pct(self, tmp_path):
        # paper batch report in bps; 100 bps == 1.0 pct
        report = tmp_path / "batch_report.json"
        report.write_text(json.dumps({
            "results": {
                "BTC": {"daily": [
                    {"date": "2026-05-01", "total_net_bps": 150.0},
                    {"date": "2026-05-02", "total_net_bps": -50.0},
                ]},
            }
        }))
        hist = load_pnl_history(exec_log=tmp_path / "absent.json", paper_report=report)
        by_date = {h["date"]: h["pnl_pct"] for h in hist}
        assert by_date["2026-05-01"] == pytest.approx(1.5)
        assert by_date["2026-05-02"] == pytest.approx(-0.5)


class TestPaperFallbackGating:
    """The paper-trade feed (sum-of-trade-returns) must NOT drive the daemon by
    default — it would false-trip the drawdown gates. Off by default; opt-in."""

    def _breach_report(self, tmp_path):
        report = tmp_path / "batch_report.json"
        report.write_text(json.dumps({"results": {"BTC": {"daily": [
            {"date": f"2026-05-{d:02d}", "total_net_bps": -800.0}
            for d in range(1, 11)
        ]}}}))
        return report

    def _cfg(self, tmp_path, report, use_fallback):
        return {
            "exec_log": str(tmp_path / "absent_daily_pnl.json"),
            "paper_report": str(report),
            "paper_trades_dir": str(tmp_path / "no_trades"),
            "use_paper_fallback": use_fallback,
            "ic_min_obs": 10,
        }

    def _ks(self, tmp_path, use_fallback):
        return KillSwitch(
            config=self._cfg(tmp_path, self._breach_report(tmp_path), use_fallback),
            halt_path=tmp_path / "h.json", audit_path=tmp_path / "a.jsonl",
            db_path=None, notify=False,
        )

    def test_fallback_off_does_not_halt(self, tmp_path):
        # exec log absent + fallback off => no data => no halt (pre-T17 reality)
        st = self._ks(tmp_path, use_fallback=False).check(ic_series=[])
        assert not st.halted

    def test_fallback_on_halts_on_breach(self, tmp_path):
        # opt-in: the breaching paper report now drives the gates
        st = self._ks(tmp_path, use_fallback=True).check(ic_series=[])
        assert st.halted


# Helper: import evaluate_kill_switches from the module under test (re-exported
# from alpha.deployer) without colliding with the deployer's KillSwitch name.
def evaluate_kill_switches_local(metrics):
    from risk.kill_switch import evaluate_kill_switches
    return evaluate_kill_switches(**metrics)

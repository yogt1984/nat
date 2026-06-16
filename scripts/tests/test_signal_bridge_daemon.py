"""Planted tests for the T17 signal-bridge daemon.

Test-first (red). The bridge daemon executes LIVE lifecycle signals under risk
gating. The non-negotiables are exercised deterministically via seams + a temp
lifecycle + a spy client (NO real orders, NO live mode anywhere here):

- the halt check at the top of every cycle CANNOT be skipped;
- sizing is portfolio-level (risk parity), never independent per-signal;
- dry-run places no orders;
- fills log to fills_*.jsonl and the daily P&L rollup writes the exact shape the
  kill-switch reads back (loop closure).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from signal_lifecycle import SignalLifecycle, LIVE
from risk.kill_switch import HaltState, write_halt_state, load_pnl_history
from execution.signal_bridge import SignalBridgeDaemon, healthy


class SpyClient:
    """Records any order attempt; the daemon must never call it in dry-run."""
    def __init__(self):
        self.orders = []

    def place_order(self, *a, **k):
        self.orders.append((a, k))
        return {"status": "filled"}


@pytest.fixture
def lc(tmp_path):
    s = SignalLifecycle(db_path=tmp_path / "nat.db")
    yield s
    s.close()


def _seed_live(lc, sid, ic_history):
    lc.discover(sid, name=sid, metadata={"ic_history": ic_history})
    lc.validate(sid)
    lc.start_paper(sid)
    lc.request_approval(sid)
    lc.approve(sid)  # APPROVAL_PENDING → LIVE


def _daemon(lc, tmp_path, *, dry_run=True, client=None, clock=None):
    cfg = {
        "poll_interval_s": 300,
        "account_value_usd": 10000.0,
        "mode": "dry-run",
        "db_path": str(tmp_path / "nat.db"),
        "halt_state_path": str(tmp_path / "halt_state.json"),
        "daily_pnl_path": str(tmp_path / "daily_pnl.json"),
        "fills_dir": str(tmp_path / "exec"),
        "pid_file": str(tmp_path / "bridge.pid"),
        "heartbeat_path": str(tmp_path / "bridge.heartbeat"),
    }
    d = SignalBridgeDaemon(config=cfg, lc=lc, client=client, clock=clock, dry_run=dry_run)
    d._signal_value = lambda sig: 1.0   # signal present & long, so a trade is intended
    d._mid = lambda sig: 100.0          # deterministic sim price
    return d


# ---------------------------------------------------------------------------
# Halt gate — cannot be skipped
# ---------------------------------------------------------------------------


class TestHaltGate:
    def test_halt_skips_cycle_entirely(self, lc, tmp_path):
        write_halt_state(tmp_path / "halt_state.json", HaltState(halted=True, level="halt_review"))
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        d = _daemon(lc, tmp_path)
        sized = []
        d._size = lambda sigs: sized.append(sigs) or []
        summary = d.run_cycle()
        assert summary["halted"] is True
        assert summary["fills"] == 0
        assert sized == []  # sizing/execution never reached under halt

    def test_no_halt_proceeds(self, lc, tmp_path):
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        summary = _daemon(lc, tmp_path).run_cycle()
        assert summary["halted"] is False


# ---------------------------------------------------------------------------
# LIVE-signal pickup
# ---------------------------------------------------------------------------


class TestLivePickup:
    def test_approved_signal_picked_up(self, lc, tmp_path):
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        summary = _daemon(lc, tmp_path).run_cycle()
        assert "jump_detector" in summary["picked_up"]

    def test_no_live_signals_is_noop(self, lc, tmp_path):
        # signals exist but none LIVE (left VALIDATED)
        lc.discover("x", name="x"); lc.validate("x")
        summary = _daemon(lc, tmp_path).run_cycle()
        assert summary["picked_up"] == [] and summary["fills"] == 0 and summary["halted"] is False


# ---------------------------------------------------------------------------
# Portfolio sizing — risk parity, never independent
# ---------------------------------------------------------------------------


class TestSizing:
    def test_risk_parity_weights_sum_to_one(self, lc, tmp_path):
        _seed_live(lc, "a", [0.10, 0.12, 0.08, 0.11])
        _seed_live(lc, "b", [0.01, 0.30, -0.05, 0.20])  # higher IC variance
        d = _daemon(lc, tmp_path)
        sigs = d._live_signals()
        w = d._size(sigs)
        assert len(w) == 2
        assert sum(w) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 < x < 1.0 for x in w)  # weighted, not each full-size


# ---------------------------------------------------------------------------
# Fills + daily P&L rollup (loop closure with the kill-switch)
# ---------------------------------------------------------------------------


class TestFillsAndRollup:
    def test_fills_jsonl_populates(self, lc, tmp_path):
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        d = _daemon(lc, tmp_path)
        d.run_cycle()
        fills_files = list((tmp_path / "exec").glob("fills_*.jsonl"))
        assert fills_files, "no fills file written"
        rows = [json.loads(ln) for ln in fills_files[0].read_text().splitlines() if ln.strip()]
        assert rows
        for r in rows:
            for k in ("timestamp", "signal_id", "signal_name", "symbol", "side",
                      "entry_price", "entry_size", "pnl_pct"):
                assert k in r

    def test_daily_pnl_shape_roundtrips_through_killswitch(self, lc, tmp_path):
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        d = _daemon(lc, tmp_path)
        d.run_cycle()
        dp = tmp_path / "daily_pnl.json"
        assert dp.exists()
        data = json.loads(dp.read_text())
        assert isinstance(data, list) and data
        assert all(set(("date", "pnl_pct")).issubset(r) for r in data)
        # the kill-switch must be able to read exactly this file
        hist = load_pnl_history(exec_log=dp, paper_report=None)
        assert [r["date"] for r in hist] == [r["date"] for r in data]


# ---------------------------------------------------------------------------
# Dry-run never trades
# ---------------------------------------------------------------------------


class TestDryRunSafety:
    def test_dry_run_places_no_orders(self, lc, tmp_path):
        _seed_live(lc, "jump_detector", [0.1, 0.12, 0.11])
        spy = SpyClient()
        d = _daemon(lc, tmp_path, dry_run=True, client=spy)
        d.run_cycle()
        assert spy.orders == []  # not one order placed in dry-run


# ---------------------------------------------------------------------------
# Healthcheck heartbeat
# ---------------------------------------------------------------------------


class TestHealthcheck:
    def test_missing_heartbeat_unhealthy(self, tmp_path):
        assert healthy({"heartbeat_path": str(tmp_path / "hb"), "poll_interval_s": 300}) is False

    def test_fresh_heartbeat_healthy(self, tmp_path):
        hb = tmp_path / "hb"; hb.write_text("now")
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 300}) is True

    def test_stale_heartbeat_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"; hb.write_text("old")
        os.utime(hb, (1_700_000_000.0, 1_700_000_000.0))
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 300}) is False

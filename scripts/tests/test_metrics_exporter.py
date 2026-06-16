"""Planted tests for monitoring.metrics_exporter — the T18 keystone.

Test-first (red). The exporter turns SQLite/JSON state (which Grafana can't scrape
directly) into Prometheus-shaped numbers: lifecycle state counts, live P&L from
data/execution/daily_pnl.json, and per-signal paper metrics. The reductions are
pure (take explicit paths / a dict), so they're tested without prometheus_client
and without the filesystem layout — the exposition layer is thin glue on top.
"""

import json

import pytest

from signal_lifecycle import SignalLifecycle
from monitoring.metrics_exporter import (
    lifecycle_counts,
    live_pnl,
    paper_metrics,
    collect,
)


# ---------------------------------------------------------------------------
# lifecycle funnel — counts by state
# ---------------------------------------------------------------------------


class TestLifecycleCounts:
    def test_counts_by_state(self, tmp_path):
        lc = SignalLifecycle(db_path=tmp_path / "nat.db")
        lc.discover("a", name="a"); lc.validate("a")
        lc.discover("b", name="b"); lc.validate("b")
        lc.discover("c", name="c")  # stays DISCOVERED
        lc.close()
        counts = lifecycle_counts(tmp_path / "nat.db")
        assert counts.get("VALIDATED") == 2
        assert counts.get("DISCOVERED") == 1

    def test_empty_db_is_empty(self, tmp_path):
        SignalLifecycle(db_path=tmp_path / "nat.db").close()
        assert lifecycle_counts(tmp_path / "nat.db") == {}

    def test_missing_db_is_empty(self, tmp_path):
        assert lifecycle_counts(tmp_path / "absent.db") == {}


# ---------------------------------------------------------------------------
# live P&L — from data/execution/daily_pnl.json
# ---------------------------------------------------------------------------


class TestLivePnl:
    def test_aggregates(self, tmp_path):
        p = tmp_path / "daily_pnl.json"
        p.write_text(json.dumps([
            {"date": "2026-06-14", "pnl_pct": 0.5},
            {"date": "2026-06-15", "pnl_pct": -0.2},
            {"date": "2026-06-16", "pnl_pct": 0.3},
        ]))
        m = live_pnl(p)
        assert m["cum_pnl_pct"] == pytest.approx(0.6)
        assert m["last_daily_pnl_pct"] == pytest.approx(0.3)
        assert m["n_days"] == 3

    def test_missing_file_is_zeros(self, tmp_path):
        m = live_pnl(tmp_path / "absent.json")
        assert m == {"cum_pnl_pct": 0.0, "last_daily_pnl_pct": 0.0, "n_days": 0}


# ---------------------------------------------------------------------------
# paper metrics — per-signal, reusing viz.approval over OOS state
# ---------------------------------------------------------------------------


def _oos_state():
    daily = [{"date": f"2026-05-{i:02d}", "symbol": "BTC", "total_net_bps": v, "max_loss_bps": -5.0}
             for i, v in zip(range(10, 15), [10, -5, 8, 12, -3])]
    return {
        "algos": {
            "jump_detector": {
                "baseline_sharpe": {"BTC": 1.2},
                "symbols": {"BTC": {"daily": daily, "metrics": {
                    "current_sharpe": 1.4, "max_drawdown_bps": -30.0, "n_days": 5,
                    "cumulative_pnl_bps": 22.0, "degradation": 5.0,
                    "rolling_sharpe_7d": [], "cumulative_pnl_series": [],
                }}},
            }
        }
    }


class TestPaperMetrics:
    def test_per_signal_sharpe(self, tmp_path):
        sp = tmp_path / "oos.json"
        sp.write_text(json.dumps(_oos_state()))
        m = paper_metrics(sp)
        assert "jump_detector" in m
        assert m["jump_detector"]["sharpe"] == pytest.approx(1.4)
        assert m["jump_detector"]["n_days"] == 5

    def test_missing_state_is_empty(self, tmp_path):
        assert paper_metrics(tmp_path / "absent.json") == {}


# ---------------------------------------------------------------------------
# collect() aggregation
# ---------------------------------------------------------------------------


class TestCollect:
    def test_collect_has_all_sections(self, tmp_path):
        lc = SignalLifecycle(db_path=tmp_path / "nat.db")
        lc.discover("a", name="a"); lc.validate("a"); lc.close()
        (tmp_path / "daily_pnl.json").write_text(json.dumps([{"date": "2026-06-16", "pnl_pct": 0.4}]))
        (tmp_path / "oos.json").write_text(json.dumps(_oos_state()))
        cfg = {
            "db_path": str(tmp_path / "nat.db"),
            "daily_pnl_path": str(tmp_path / "daily_pnl.json"),
            "oos_state_path": str(tmp_path / "oos.json"),
        }
        out = collect(cfg)
        assert out["lifecycle"].get("VALIDATED") == 1
        assert out["live_pnl"]["n_days"] == 1
        assert "jump_detector" in out["paper"]

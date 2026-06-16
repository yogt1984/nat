"""T18 follow-on slices: agent-dashboard lifecycle endpoint + Grafana dashboards.

- read_lifecycle() funnel over a temp lifecycle DB (the /api/lifecycle backend).
- the 3 provisioned Grafana dashboards are valid JSON and reference the exporter's
  metric names (so the panels actually have a datasource).
"""

import json
from pathlib import Path

import pytest

from signal_lifecycle import SignalLifecycle

ROOT = Path(__file__).resolve().parents[2]
DASH_DIR = ROOT / "docker" / "grafana" / "dashboards"


# ---------------------------------------------------------------------------
# agent_dashboard.read_lifecycle (the /api/lifecycle backend)
# ---------------------------------------------------------------------------


class TestLifecycleEndpoint:
    def test_funnel_and_approval_pending(self, tmp_path, monkeypatch):
        db = tmp_path / "nat.db"
        lc = SignalLifecycle(db_path=db)
        lc.discover("a", name="a"); lc.validate("a")
        lc.discover("b", name="b"); lc.validate("b"); lc.start_paper("b"); lc.request_approval("b")
        lc.close()

        import agent_dashboard as ad
        monkeypatch.setattr(ad, "DB_PATH", db)
        out = ad.read_lifecycle()
        assert out["total"] == 2
        assert out["by_state"].get("VALIDATED") == 1
        assert out["by_state"].get("APPROVAL_PENDING") == 1
        assert out["approval_pending"] == ["b"]

    def test_missing_db_is_empty(self, tmp_path, monkeypatch):
        import agent_dashboard as ad
        monkeypatch.setattr(ad, "DB_PATH", tmp_path / "absent.db")
        assert ad.read_lifecycle() == {"by_state": {}, "total": 0,
                                       "approval_pending": [], "signals": []}


# ---------------------------------------------------------------------------
# Grafana dashboards — valid JSON + reference exporter metrics
# ---------------------------------------------------------------------------

_EXPECTED = {
    "lifecycle_funnel.json": "nat_lifecycle_signals",
    "paper_performance.json": "nat_paper_sharpe",
    "live_pnl.json": "nat_live_cum_pnl_pct",
}


@pytest.mark.parametrize("fname,metric", list(_EXPECTED.items()))
def test_dashboard_valid_and_references_exporter_metric(fname, metric):
    path = DASH_DIR / fname
    assert path.exists(), f"missing dashboard {fname}"
    d = json.loads(path.read_text())  # valid JSON
    assert d.get("uid") and d.get("title") and isinstance(d.get("panels"), list) and d["panels"]
    exprs = " ".join(t.get("expr", "") for p in d["panels"] for t in p.get("targets", []))
    assert metric in exprs, f"{fname} panels do not reference {metric}"

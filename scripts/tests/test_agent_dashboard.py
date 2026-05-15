"""Tests for agent_dashboard — agent web dashboard data readers and API.

Tests data readers (pure functions) and HTTP handler separately.
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent_dashboard import (
    read_state, read_hypotheses, read_registry, read_gen_stats,
    get_queue, get_graveyard, get_tested, build_heatmap_data,
    get_cache_stats, get_summary,
    AgentDashboardHandler, DASHBOARD_HTML,
)
from http.server import HTTPServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR to a temporary directory."""
    import agent_dashboard
    monkeypatch.setattr(agent_dashboard, "STATE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_state():
    return {
        "phase": "IDLE",
        "cycle_count": 5,
        "total_hypotheses_tested": 42,
        "total_signals_registered": 3,
    }


@pytest.fixture
def sample_hypotheses():
    return [
        {
            "id": "HYP-SYS-001",
            "claim": "imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns",
            "generator": "systematic",
            "priority": 0.9,
            "status": "replicated",
            "failure_reason": None,
            "thresholds": {"regime_gate": "ent_book_shape<P20"},
            "results": {"gate_results": [{"msg": "IC=0.5689 [gated] p=0.00e+00 PASS"}]},
        },
        {
            "id": "HYP-SYS-002",
            "claim": "imbalance_qty_l1 gated by ent_book_shape<P40 predicts 5s returns",
            "generator": "systematic",
            "priority": 0.8,
            "status": "failed",
            "failure_reason": "no_replication",
            "thresholds": {"regime_gate": "ent_book_shape<P40"},
            "results": {"gate_results": [{"msg": "IC=0.5437 [gated] p=0.00e+00 PASS"}]},
        },
        {
            "id": "HYP-SYS-003",
            "claim": "imbalance_qty_l5 gated by ent_book_shape<P20 predicts 5s returns",
            "generator": "systematic",
            "priority": 0.7,
            "status": "failed",
            "failure_reason": "redundant",
            "thresholds": {"regime_gate": "ent_book_shape<P20"},
            "results": {"gate_results": [{"msg": "IC=0.5423 [gated] p=0.00e+00 PASS"}],
                        "correlation_check": "max_corr=+0.948 vs imbalance_qty_l1 REDUNDANT"},
        },
        {
            "id": "HYP-SYS-004",
            "claim": "imbalance_qty_l1 gated by ent_book_shape>P20 predicts 5s returns",
            "generator": "systematic",
            "priority": 0.5,
            "status": "failed",
            "failure_reason": "no_effect",
            "thresholds": {"regime_gate": "ent_book_shape>P20"},
            "results": {"gate_results": [{"msg": "IC=0.4200 [gated] p=1.00e-10 FAIL"}]},
        },
        {
            "id": "HYP-SYS-005",
            "claim": "imbalance_qty_l10 gated by toxic_vpin_50<P40 predicts 5s returns",
            "generator": "systematic",
            "priority": 0.6,
            "status": "queued",
            "failure_reason": None,
            "thresholds": {"regime_gate": "toxic_vpin_50<P40"},
            "results": None,
        },
    ]


@pytest.fixture
def sample_registry():
    return [
        {
            "name": "imbalance_qty_l1 gated by ent_book_shape<P20",
            "features": ["imbalance_qty_l1"],
            "regime_gate": "ent_book_shape<P20",
            "expected_ic": 0.569,
            "symbols": ["BTC", "ETH", "SOL"],
            "discovery_date": "2026-05-15",
            "hypothesis_id": "HYP-SYS-001",
        },
    ]


# ---------------------------------------------------------------------------
# read_state
# ---------------------------------------------------------------------------

class TestReadState:
    def test_reads_existing_state(self, state_dir, sample_state):
        (state_dir / "agent_state.json").write_text(json.dumps(sample_state))
        result = read_state()
        assert result["phase"] == "IDLE"
        assert result["cycle_count"] == 5

    def test_returns_default_when_missing(self, state_dir):
        result = read_state()
        assert result["phase"] == "UNKNOWN"
        assert result["cycle_count"] == 0


# ---------------------------------------------------------------------------
# read_hypotheses
# ---------------------------------------------------------------------------

class TestReadHypotheses:
    def test_reads_existing(self, state_dir, sample_hypotheses):
        (state_dir / "hypotheses.json").write_text(json.dumps(sample_hypotheses))
        result = read_hypotheses()
        assert len(result) == 5

    def test_returns_empty_when_missing(self, state_dir):
        assert read_hypotheses() == []


# ---------------------------------------------------------------------------
# read_registry
# ---------------------------------------------------------------------------

class TestReadRegistry:
    def test_reads_existing(self, state_dir, sample_registry):
        (state_dir / "registry.json").write_text(json.dumps(sample_registry))
        result = read_registry()
        assert len(result) == 1
        assert result[0]["expected_ic"] == 0.569

    def test_returns_empty_when_missing(self, state_dir):
        assert read_registry() == []


# ---------------------------------------------------------------------------
# get_queue
# ---------------------------------------------------------------------------

class TestGetQueue:
    def test_returns_only_queued(self, sample_hypotheses):
        queue = get_queue(sample_hypotheses)
        assert all(h["status"] == "queued" for h in queue)
        assert len(queue) == 1

    def test_sorted_by_priority_descending(self):
        hyps = [
            {"status": "queued", "priority": 0.3, "claim": "low"},
            {"status": "queued", "priority": 0.9, "claim": "high"},
            {"status": "queued", "priority": 0.6, "claim": "mid"},
        ]
        queue = get_queue(hyps)
        assert [h["claim"] for h in queue] == ["high", "mid", "low"]

    def test_respects_limit(self):
        hyps = [{"status": "queued", "priority": i} for i in range(100)]
        queue = get_queue(hyps, limit=5)
        assert len(queue) == 5

    def test_empty_input(self):
        assert get_queue([]) == []

    def test_no_queued_hypotheses(self, sample_hypotheses):
        # Remove the queued one
        no_queued = [h for h in sample_hypotheses if h["status"] != "queued"]
        assert get_queue(no_queued) == []


# ---------------------------------------------------------------------------
# get_graveyard
# ---------------------------------------------------------------------------

class TestGetGraveyard:
    def test_returns_only_failed(self, sample_hypotheses):
        grave = get_graveyard(sample_hypotheses)
        assert all(h["status"] == "failed" for h in grave)
        assert len(grave) == 3

    def test_includes_all_failure_reasons(self, sample_hypotheses):
        grave = get_graveyard(sample_hypotheses)
        reasons = {h["failure_reason"] for h in grave}
        assert reasons == {"no_replication", "redundant", "no_effect"}

    def test_empty_input(self):
        assert get_graveyard([]) == []


# ---------------------------------------------------------------------------
# get_tested
# ---------------------------------------------------------------------------

class TestGetTested:
    def test_excludes_queued(self, sample_hypotheses):
        tested = get_tested(sample_hypotheses)
        assert len(tested) == 4  # 5 total - 1 queued
        assert all(h["status"] != "queued" for h in tested)

    def test_empty_input(self):
        assert get_tested([]) == []


# ---------------------------------------------------------------------------
# build_heatmap_data
# ---------------------------------------------------------------------------

class TestBuildHeatmapData:
    def test_extracts_signals_and_gates(self, sample_hypotheses):
        data = build_heatmap_data(sample_hypotheses)
        assert "imbalance_qty_l1" in data["signals"]
        assert "imbalance_qty_l5" in data["signals"]
        assert "ent_book_shape<P20" in data["gates"]
        assert "ent_book_shape<P40" in data["gates"]

    def test_cells_have_required_fields(self, sample_hypotheses):
        data = build_heatmap_data(sample_hypotheses)
        for cell in data["cells"]:
            assert "signal" in cell
            assert "gate" in cell
            assert "ic" in cell
            assert "status" in cell

    def test_extracts_ic_from_results(self, sample_hypotheses):
        data = build_heatmap_data(sample_hypotheses)
        l1_p20 = [c for c in data["cells"]
                   if c["signal"] == "imbalance_qty_l1" and c["gate"] == "ent_book_shape<P20"]
        assert len(l1_p20) == 1
        assert l1_p20[0]["ic"] == pytest.approx(0.5689, abs=0.001)

    def test_skips_queued_hypotheses(self, sample_hypotheses):
        data = build_heatmap_data(sample_hypotheses)
        assert "toxic_vpin_50<P40" not in data["gates"]

    def test_empty_input(self):
        data = build_heatmap_data([])
        assert data["signals"] == []
        assert data["gates"] == []
        assert data["cells"] == []

    def test_hypothesis_without_gated_by(self):
        hyps = [{"status": "failed", "claim": "plain signal test", "thresholds": {}, "results": {}}]
        data = build_heatmap_data(hyps)
        assert data["cells"] == []

    def test_signals_and_gates_sorted(self, sample_hypotheses):
        data = build_heatmap_data(sample_hypotheses)
        assert data["signals"] == sorted(data["signals"])
        assert data["gates"] == sorted(data["gates"])


# ---------------------------------------------------------------------------
# get_cache_stats
# ---------------------------------------------------------------------------

class TestGetCacheStats:
    def test_no_cache_dir(self, state_dir):
        stats = get_cache_stats()
        assert stats["entries"] == 0

    def test_with_cache_entries(self, state_dir):
        cache_dir = state_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "abc123.json").write_text('{"data": true}')
        (cache_dir / "abc123.meta.json").write_text('{"cached_at": 0}')
        stats = get_cache_stats()
        assert stats["entries"] == 1
        assert stats["size_kb"] >= 0


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_computes_all_fields(self, sample_hypotheses, sample_registry, sample_state):
        summary = get_summary(sample_hypotheses, sample_registry, sample_state)
        assert summary["phase"] == "IDLE"
        assert summary["cycle_count"] == 5
        assert summary["total_tested"] == 4
        assert summary["total_queued"] == 1
        assert summary["total_registered"] == 1

    def test_status_breakdown(self, sample_hypotheses, sample_registry, sample_state):
        summary = get_summary(sample_hypotheses, sample_registry, sample_state)
        bd = summary["status_breakdown"]
        assert bd["replicated"] == 1
        assert bd["failed(no_replication)"] == 1
        assert bd["failed(redundant)"] == 1
        assert bd["failed(no_effect)"] == 1

    def test_empty_inputs(self):
        summary = get_summary([], [], {"phase": "IDLE"})
        assert summary["total_tested"] == 0
        assert summary["total_queued"] == 0
        assert summary["total_registered"] == 0


# ---------------------------------------------------------------------------
# read_gen_stats
# ---------------------------------------------------------------------------

class TestReadGenStats:
    def test_reads_existing(self, state_dir):
        stats = {"systematic": {"attempts": 10, "successes": 3}}
        (state_dir / "generator_stats.json").write_text(json.dumps(stats))
        result = read_gen_stats()
        assert result["systematic"]["attempts"] == 10

    def test_returns_empty_when_missing(self, state_dir):
        assert read_gen_stats() == {}


# ---------------------------------------------------------------------------
# HTTP handler (integration)
# ---------------------------------------------------------------------------

class TestHTTPEndpoints:
    """Start a real server and hit each endpoint."""

    @pytest.fixture(autouse=True)
    def server(self, state_dir, sample_state, sample_hypotheses, sample_registry):
        """Write sample data and start server on a random port."""
        (state_dir / "agent_state.json").write_text(json.dumps(sample_state))
        (state_dir / "hypotheses.json").write_text(json.dumps(sample_hypotheses))
        (state_dir / "registry.json").write_text(json.dumps(sample_registry))

        self._server = HTTPServer(("127.0.0.1", 0), AgentDashboardHandler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        yield
        self._server.shutdown()

    def _get(self, path: str) -> dict | str:
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode()
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                return json.loads(data)
            return data

    def test_root_returns_html(self):
        html = self._get("/")
        assert "<title>NAT Agent Dashboard</title>" in html

    def test_api_state(self):
        data = self._get("/api/state")
        assert data["phase"] == "IDLE"
        assert data["cycle_count"] == 5
        assert "_queue_depth" in data

    def test_api_registry(self):
        data = self._get("/api/registry")
        assert len(data) == 1
        assert data[0]["expected_ic"] == 0.569

    def test_api_queue(self):
        data = self._get("/api/queue")
        assert isinstance(data, list)
        assert all(h["status"] == "queued" for h in data)

    def test_api_graveyard(self):
        data = self._get("/api/graveyard")
        assert len(data) == 3
        assert all(h["status"] == "failed" for h in data)

    def test_api_heatmap(self):
        data = self._get("/api/heatmap")
        assert "signals" in data
        assert "gates" in data
        assert "cells" in data
        assert len(data["cells"]) > 0

    def test_api_cache(self):
        data = self._get("/api/cache")
        assert "entries" in data
        assert "size_kb" in data

    def test_404_on_unknown_path(self):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._get("/api/nonexistent")
        assert exc_info.value.code == 404

    def test_state_includes_queue_depth(self):
        data = self._get("/api/state")
        assert data["_queue_depth"] == 1  # 1 queued hypothesis

    def test_state_includes_gen_stats(self):
        data = self._get("/api/state")
        assert "_gen_stats" in data


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

class TestDashboardHTML:
    def test_html_is_valid_string(self):
        assert isinstance(DASHBOARD_HTML, str)
        assert len(DASHBOARD_HTML) > 1000

    def test_contains_all_panels(self):
        assert "Registry" in DASHBOARD_HTML
        assert "Heatmap" in DASHBOARD_HTML
        assert "Graveyard" in DASHBOARD_HTML
        assert "Queue" in DASHBOARD_HTML
        assert "Generator Performance" in DASHBOARD_HTML
        assert "Cache" in DASHBOARD_HTML

    def test_contains_refresh_logic(self):
        assert "refreshAll" in DASHBOARD_HTML
        assert "setInterval" in DASHBOARD_HTML

    def test_contains_api_endpoints(self):
        assert "/api/state" in DASHBOARD_HTML
        assert "/api/registry" in DASHBOARD_HTML
        assert "/api/heatmap" in DASHBOARD_HTML
        assert "/api/graveyard" in DASHBOARD_HTML
        assert "/api/queue" in DASHBOARD_HTML
        assert "/api/cache" in DASHBOARD_HTML

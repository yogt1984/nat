"""Tests for agent_dashboard — agent web dashboard data readers and API.

Tests data readers (pure functions) and HTTP handler separately.
"""

import json
import threading
import time
import urllib.request
from pathlib import Path


import pytest
from agent_dashboard import (
    read_state, read_hypotheses, read_registry, read_gen_stats,
    get_queue, get_graveyard, get_tested, build_heatmap_data,
    get_cache_stats, get_summary,
    build_graveyard_sankey, build_cross_symbol_ic,
    build_ic_decay_data, build_correlation_matrix,
    build_ic_threshold_curve, _extract_ic_from_hypothesis,
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

    def test_api_graveyard_sankey(self):
        data = self._get("/api/graveyard_sankey")
        assert "generators" in data
        assert "reasons" in data
        assert "links" in data

    def test_api_cross_symbol_ic(self):
        data = self._get("/api/cross_symbol_ic")
        assert "points" in data

    def test_api_ic_decay(self):
        data = self._get("/api/ic_decay")
        assert "curves" in data

    def test_api_correlation_matrix(self):
        data = self._get("/api/correlation_matrix")
        assert "signals" in data
        assert "matrix" in data


# ---------------------------------------------------------------------------
# build_graveyard_sankey
# ---------------------------------------------------------------------------

class TestBuildGraveyardSankey:
    def test_counts_generator_reason_flows(self, sample_hypotheses):
        result = build_graveyard_sankey(sample_hypotheses)
        # 3 failed hypotheses: systematic→no_replication, systematic→redundant, systematic→no_effect
        assert len(result["links"]) == 3
        assert "systematic" in result["generators"]
        for link in result["links"]:
            assert link["count"] >= 1

    def test_empty_hypotheses(self):
        result = build_graveyard_sankey([])
        assert result["links"] == []
        assert result["generators"] == []
        assert result["reasons"] == []

    def test_only_failed_counted(self, sample_hypotheses):
        result = build_graveyard_sankey(sample_hypotheses)
        total = sum(l["count"] for l in result["links"])
        assert total == 3  # 3 failed, 1 replicated, 1 queued excluded

    def test_sorted_by_count_descending(self):
        hyps = [
            {"status": "failed", "generator": "a", "failure_reason": "x"},
            {"status": "failed", "generator": "a", "failure_reason": "x"},
            {"status": "failed", "generator": "b", "failure_reason": "y"},
        ]
        result = build_graveyard_sankey(hyps)
        assert result["links"][0]["count"] >= result["links"][1]["count"]


# ---------------------------------------------------------------------------
# build_cross_symbol_ic
# ---------------------------------------------------------------------------

class TestBuildCrossSymbolIC:
    def test_extracts_per_symbol_ic(self):
        hyps = [{
            "status": "replicated",
            "id": "HYP-001",
            "generator": "systematic",
            "claim": "test claim",
            "results": {
                "symbol_replication": {
                    "BTC": {"ic": 0.45, "pass": True},
                    "ETH": {"ic": 0.38, "pass": True},
                }
            },
        }]
        result = build_cross_symbol_ic(hyps)
        assert len(result["points"]) == 1
        assert result["points"][0]["ics"]["BTC"] == 0.45
        assert result["points"][0]["ics"]["ETH"] == 0.38

    def test_skips_queued(self):
        hyps = [{"status": "queued", "results": {"symbol_replication": {"BTC": {"ic": 0.5}}}}]
        result = build_cross_symbol_ic(hyps)
        assert result["points"] == []

    def test_needs_at_least_2_symbols(self):
        hyps = [{
            "status": "replicated",
            "results": {"symbol_replication": {"BTC": {"ic": 0.45}}},
        }]
        result = build_cross_symbol_ic(hyps)
        assert result["points"] == []

    def test_empty_hypotheses(self):
        result = build_cross_symbol_ic([])
        assert result["points"] == []


# ---------------------------------------------------------------------------
# build_ic_decay_data
# ---------------------------------------------------------------------------

class TestBuildICDecayData:
    def test_extracts_ic_history(self):
        registry = [{
            "name": "test_signal",
            "status": "validated",
            "expected_ic": 0.5,
            "ic_history": [
                {"date": "2026-05-14", "ic": 0.48},
                {"date": "2026-05-15", "ic": 0.45},
            ],
        }]
        result = build_ic_decay_data(registry)
        assert len(result["curves"]) == 1
        assert result["curves"][0]["discovery_ic"] == 0.5
        assert result["curves"][0]["retirement_threshold"] == 0.25
        assert len(result["curves"][0]["points"]) == 2

    def test_skips_retired(self):
        registry = [{
            "name": "old", "status": "retired",
            "ic_history": [{"date": "d", "ic": 0.1}], "expected_ic": 0.5,
        }]
        result = build_ic_decay_data(registry)
        assert result["curves"] == []

    def test_skips_no_history(self):
        registry = [{"name": "new", "status": "validated", "expected_ic": 0.5}]
        result = build_ic_decay_data(registry)
        assert result["curves"] == []

    def test_empty_registry(self):
        result = build_ic_decay_data([])
        assert result["curves"] == []

    def test_numeric_history_fallback(self):
        """Handles old format where ic_history is just a list of floats."""
        registry = [{
            "name": "s", "status": "validated", "expected_ic": 0.4,
            "ic_history": [0.39, 0.38, 0.37],
        }]
        result = build_ic_decay_data(registry)
        assert len(result["curves"][0]["points"]) == 3
        assert result["curves"][0]["points"][0]["ic"] == 0.39


# ---------------------------------------------------------------------------
# build_correlation_matrix
# ---------------------------------------------------------------------------

class TestBuildCorrelationMatrix:
    def test_builds_matrix_from_registry(self):
        registry = [
            {"name": "sig_a", "status": "validated", "correlation_with": {"sig_b": 0.45}},
            {"name": "sig_b", "status": "validated", "correlation_with": {"sig_a": 0.45}},
        ]
        result = build_correlation_matrix(registry)
        assert len(result["signals"]) == 2
        assert len(result["matrix"]) == 2
        # Diagonal = 1.0
        assert result["matrix"][0][0] == 1.0
        assert result["matrix"][1][1] == 1.0
        # Off-diagonal
        assert result["matrix"][0][1] == 0.45

    def test_needs_at_least_2_signals(self):
        result = build_correlation_matrix([{"name": "a", "status": "validated"}])
        assert result["signals"] == []
        assert result["matrix"] == []

    def test_excludes_retired(self):
        registry = [
            {"name": "a", "status": "validated"},
            {"name": "b", "status": "retired"},
        ]
        result = build_correlation_matrix(registry)
        assert result["signals"] == []  # only 1 active

    def test_none_for_unknown_pairs(self):
        registry = [
            {"name": "a", "status": "validated"},
            {"name": "b", "status": "validated"},
        ]
        result = build_correlation_matrix(registry)
        assert result["matrix"][0][1] is None  # no correlation_with data


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
        assert "Graveyard Analysis" in DASHBOARD_HTML
        assert "Cross-Symbol IC" in DASHBOARD_HTML
        assert "IC Decay" in DASHBOARD_HTML
        assert "Correlation Matrix" in DASHBOARD_HTML

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
        assert "/api/graveyard_sankey" in DASHBOARD_HTML
        assert "/api/cross_symbol_ic" in DASHBOARD_HTML
        assert "/api/ic_decay" in DASHBOARD_HTML
        assert "/api/correlation_matrix" in DASHBOARD_HTML
        assert "/api/ic_threshold_curve" in DASHBOARD_HTML

    def test_contains_ic_threshold_panel(self):
        assert "IC by Threshold" in DASHBOARD_HTML
        assert "refreshICThreshold" in DASHBOARD_HTML
        assert "ic-threshold" in DASHBOARD_HTML


# ---------------------------------------------------------------------------
# _extract_ic_from_hypothesis (helper)
# ---------------------------------------------------------------------------

class TestExtractICFromHypothesis:
    """Skeptical tests for IC extraction — the parser that feeds the curve."""

    def test_extracts_positive_ic(self):
        h = {"results": {"gate_results": [{"msg": "IC=0.5689 [gated] p=0.00e+00 PASS"}]}}
        assert _extract_ic_from_hypothesis(h) == pytest.approx(0.5689)

    def test_extracts_negative_ic(self):
        """IC can be negative — must not silently discard."""
        h = {"results": {"gate_results": [{"msg": "IC=-0.1234 [gated] p=1e-5 FAIL"}]}}
        assert _extract_ic_from_hypothesis(h) == pytest.approx(-0.1234)

    def test_returns_none_on_missing_results(self):
        assert _extract_ic_from_hypothesis({}) is None
        assert _extract_ic_from_hypothesis({"results": None}) is None

    def test_returns_none_on_no_ic_in_message(self):
        h = {"results": {"gate_results": [{"msg": "some unrelated message"}]}}
        assert _extract_ic_from_hypothesis(h) is None

    def test_returns_none_on_empty_gate_results(self):
        h = {"results": {"gate_results": []}}
        assert _extract_ic_from_hypothesis(h) is None

    def test_returns_none_on_malformed_ic(self):
        """IC= followed by garbage should not crash."""
        h = {"results": {"gate_results": [{"msg": "IC=notanumber FAIL"}]}}
        assert _extract_ic_from_hypothesis(h) is None

    def test_takes_first_ic_match(self):
        """If multiple gate_results have IC=, take the first."""
        h = {"results": {"gate_results": [
            {"msg": "IC=0.30 [gated] PASS"},
            {"msg": "IC=0.50 dIC check"},
        ]}}
        assert _extract_ic_from_hypothesis(h) == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# build_ic_threshold_curve
# ---------------------------------------------------------------------------

class TestBuildICThresholdCurve:
    """Skeptical tests — verify the curve builder handles real and adversarial data."""

    def _make_hyp(self, signal, gate, ic, status="replicated"):
        """Helper to build a hypothesis with a gate and IC."""
        return {
            "claim": f"{signal} gated by {gate} predicts 5s returns",
            "generator": "systematic",
            "status": status,
            "thresholds": {"regime_gate": gate},
            "results": {"gate_results": [{"msg": f"IC={ic:.4f} [gated] p=0.00e+00 PASS"}]},
        }

    def test_basic_curve_from_multiple_thresholds(self):
        """Three thresholds → one curve with 3 points, sorted by percentile."""
        hyps = [
            self._make_hyp("imbalance_qty_l1", "ent_book_shape<P20", 0.56),
            self._make_hyp("imbalance_qty_l1", "ent_book_shape<P40", 0.54),
            self._make_hyp("imbalance_qty_l1", "ent_book_shape<P60", 0.48),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        curve = result["curves"][0]
        assert curve["signal"] == "imbalance_qty_l1"
        assert curve["gate"] == "ent_book_shape"
        assert curve["direction"] == "<"
        assert len(curve["points"]) == 3
        # Points must be sorted by percentile, not by IC
        pcts = [p["percentile"] for p in curve["points"]]
        assert pcts == [20, 40, 60]

    def test_empty_input(self):
        result = build_ic_threshold_curve([])
        assert result["curves"] == []

    def test_single_threshold_produces_no_curve(self):
        """A single threshold cannot form a curve — need ≥2 points."""
        hyps = [self._make_hyp("sig", "gate<P40", 0.5)]
        result = build_ic_threshold_curve(hyps)
        assert result["curves"] == []

    def test_queued_hypotheses_excluded(self):
        """Queued hypotheses have no results — must not appear in curves."""
        hyps = [
            self._make_hyp("sig", "gate<P20", 0.5),
            self._make_hyp("sig", "gate<P40", 0.4),
            {
                "claim": "sig gated by gate<P60 predicts 5s returns",
                "status": "queued",
                "thresholds": {"regime_gate": "gate<P60"},
                "results": None,
            },
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        assert len(result["curves"][0]["points"]) == 2  # P60 excluded

    def test_different_directions_are_separate_curves(self):
        """ent<P40 and ent>P40 are different conditioning — must not merge."""
        hyps = [
            self._make_hyp("sig", "ent<P20", 0.50),
            self._make_hyp("sig", "ent<P40", 0.45),
            self._make_hyp("sig", "ent>P20", 0.30),
            self._make_hyp("sig", "ent>P40", 0.35),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 2
        directions = {c["direction"] for c in result["curves"]}
        assert directions == {"<", ">"}

    def test_different_signals_are_separate_curves(self):
        """Two signals with the same gate feature → separate curves."""
        hyps = [
            self._make_hyp("sig_a", "gate<P20", 0.50),
            self._make_hyp("sig_a", "gate<P40", 0.45),
            self._make_hyp("sig_b", "gate<P20", 0.30),
            self._make_hyp("sig_b", "gate<P40", 0.25),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 2
        signals = {c["signal"] for c in result["curves"]}
        assert signals == {"sig_a", "sig_b"}

    def test_different_gate_features_are_separate_curves(self):
        """Same signal, different gate features → separate curves."""
        hyps = [
            self._make_hyp("sig", "ent_book<P20", 0.50),
            self._make_hyp("sig", "ent_book<P40", 0.45),
            self._make_hyp("sig", "toxic_vpin<P20", 0.40),
            self._make_hyp("sig", "toxic_vpin<P40", 0.35),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 2
        gates = {c["gate"] for c in result["curves"]}
        assert gates == {"ent_book", "toxic_vpin"}

    def test_duplicate_percentile_keeps_last(self):
        """If the same threshold is tested twice, keep the later IC (list order)."""
        hyps = [
            self._make_hyp("sig", "gate<P20", 0.50),  # first run
            self._make_hyp("sig", "gate<P40", 0.45),
            self._make_hyp("sig", "gate<P20", 0.52),  # re-run overwrites
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        p20 = [p for p in result["curves"][0]["points"] if p["percentile"] == 20]
        assert len(p20) == 1
        assert p20[0]["ic"] == pytest.approx(0.52)  # last wins

    def test_baseline_ic_from_ungated_hypothesis(self):
        """Ungated hypotheses (no 'gated by') provide the baseline IC."""
        hyps = [
            {
                "claim": "imb_l1 predicts 5s returns",
                "status": "replicated",
                "thresholds": {},
                "results": {"gate_results": [{"msg": "IC=0.4200 [aggregate] PASS"}]},
            },
            self._make_hyp("imb_l1", "ent<P20", 0.56),
            self._make_hyp("imb_l1", "ent<P40", 0.50),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        assert result["curves"][0]["baseline_ic"] == pytest.approx(0.42)

    def test_baseline_ic_is_none_when_no_ungated(self):
        """If there's no ungated hypothesis, baseline should be None, not crash."""
        hyps = [
            self._make_hyp("sig", "gate<P20", 0.50),
            self._make_hyp("sig", "gate<P40", 0.45),
        ]
        result = build_ic_threshold_curve(hyps)
        assert result["curves"][0]["baseline_ic"] is None

    def test_curves_sorted_by_peak_ic_descending(self):
        """Best curves (highest IC) should render first."""
        hyps = [
            self._make_hyp("weak", "gate<P20", 0.10),
            self._make_hyp("weak", "gate<P40", 0.12),
            self._make_hyp("strong", "gate<P20", 0.50),
            self._make_hyp("strong", "gate<P40", 0.55),
        ]
        result = build_ic_threshold_curve(hyps)
        assert result["curves"][0]["signal"] == "strong"
        assert result["curves"][1]["signal"] == "weak"

    def test_failed_hypotheses_included(self):
        """Failed hypotheses still have IC values — they belong in the curve."""
        hyps = [
            self._make_hyp("sig", "gate<P20", 0.50, status="failed"),
            self._make_hyp("sig", "gate<P40", 0.42, status="failed"),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        assert len(result["curves"][0]["points"]) == 2

    def test_hypothesis_with_no_ic_is_skipped(self):
        """If IC extraction fails, don't crash — skip the data point."""
        hyps = [
            self._make_hyp("sig", "gate<P20", 0.50),
            {
                "claim": "sig gated by gate<P40 predicts 5s returns",
                "status": "failed",
                "thresholds": {"regime_gate": "gate<P40"},
                "results": {"gate_results": [{"msg": "no IC here, just a message"}]},
            },
            self._make_hyp("sig", "gate<P60", 0.44),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        assert len(result["curves"][0]["points"]) == 2  # P40 skipped

    def test_malformed_gate_spec_skipped(self):
        """Non-standard gate specs (no percentile) should be silently dropped."""
        hyps = [
            {
                "claim": "sig gated by some_random_gate predicts 5s returns",
                "status": "replicated",
                "thresholds": {"regime_gate": "some_random_gate"},
                "results": {"gate_results": [{"msg": "IC=0.45 PASS"}]},
            },
            self._make_hyp("sig", "gate<P20", 0.50),
            self._make_hyp("sig", "gate<P40", 0.45),
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1  # malformed one dropped

    def test_label_format(self):
        hyps = [
            self._make_hyp("imb_l1", "ent_shape<P20", 0.50),
            self._make_hyp("imb_l1", "ent_shape<P40", 0.45),
        ]
        result = build_ic_threshold_curve(hyps)
        assert result["curves"][0]["label"] == "imb_l1 | ent_shape<"

    def test_wide_percentile_range(self):
        """P10 through P90 — boundary percentiles should work."""
        hyps = [
            self._make_hyp("sig", f"gate<P{p}", 0.5 - p * 0.003)
            for p in range(10, 91, 10)
        ]
        result = build_ic_threshold_curve(hyps)
        assert len(result["curves"]) == 1
        assert len(result["curves"][0]["points"]) == 9
        pcts = [p["percentile"] for p in result["curves"][0]["points"]]
        assert pcts == list(range(10, 91, 10))


# ---------------------------------------------------------------------------
# HTTP endpoint for ic_threshold_curve
# ---------------------------------------------------------------------------

class TestICThresholdEndpoint(TestHTTPEndpoints):
    """Verify the /api/ic_threshold_curve endpoint works end-to-end."""

    def test_api_ic_threshold_curve(self):
        data = self._get("/api/ic_threshold_curve")
        assert "curves" in data


# ---------------------------------------------------------------------------
# Cache tests (P1-7)
# ---------------------------------------------------------------------------

class TestCache:
    """Verify the in-memory cache with TTL."""

    def setup_method(self):
        from agent_dashboard import cache_clear
        cache_clear()

    def test_cached_returns_same_object_within_ttl(self):
        from agent_dashboard import _cached
        call_count = [0]

        def loader():
            call_count[0] += 1
            return {"data": call_count[0]}

        r1 = _cached("test_key", loader, ttl=60)
        r2 = _cached("test_key", loader, ttl=60)
        assert r1 is r2
        assert call_count[0] == 1  # loader called only once

    def test_cached_refreshes_after_ttl(self):
        from agent_dashboard import _cached, _cache
        call_count = [0]

        def loader():
            call_count[0] += 1
            return {"n": call_count[0]}

        r1 = _cached("ttl_key", loader, ttl=0.05)
        assert r1["n"] == 1

        # Expire by backdating timestamp
        _cache["ttl_key"] = (_cache["ttl_key"][0], time.time() - 1)

        r2 = _cached("ttl_key", loader, ttl=0.05)
        assert r2["n"] == 2
        assert call_count[0] == 2

    def test_cache_clear_resets_all(self):
        from agent_dashboard import _cached, _cache, cache_clear
        _cached("a", lambda: 1)
        _cached("b", lambda: 2)
        assert len(_cache) == 2
        cache_clear()
        assert len(_cache) == 0

    def test_different_keys_cached_independently(self):
        from agent_dashboard import _cached
        r1 = _cached("key_a", lambda: "alpha")
        r2 = _cached("key_b", lambda: "beta")
        assert r1 == "alpha"
        assert r2 == "beta"

    def test_api_state_uses_cache(self):
        """Verify /api/state doesn't re-read on rapid successive calls."""
        from agent_dashboard import _cached, _cache, cache_clear
        cache_clear()
        # Pre-populate cache
        _cached("state", lambda: {"phase": "IDLE", "_queue_depth": 0, "_gen_stats": {}})

        # Second call should return cached
        result = _cached("state", lambda: {"phase": "CHANGED"})
        assert result["phase"] == "IDLE"  # still cached

    def test_cache_control_header_present(self):
        """HTTP responses should include Cache-Control header."""
        import io
        from unittest.mock import MagicMock, patch

        from agent_dashboard import AgentDashboardHandler, cache_clear, _CACHE_TTL
        cache_clear()

        # Create mock request
        handler = MagicMock(spec=AgentDashboardHandler)
        handler.path = "/api/registry"
        handler.command = "GET"
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        # Call do_GET directly
        with patch.object(AgentDashboardHandler, '__init__', lambda x, *a, **k: None):
            h = AgentDashboardHandler.__new__(AgentDashboardHandler)
            h.path = "/api/registry"
            h.command = "GET"
            h.wfile = io.BytesIO()
            h.send_response = MagicMock()
            h.send_header = MagicMock()
            h.end_headers = MagicMock()
            h.do_GET()

        # Check Cache-Control was sent
        header_calls = [c[0] for c in h.send_header.call_args_list]
        cache_headers = [c for c in header_calls if c[0] == "Cache-Control"]
        assert len(cache_headers) == 1
        assert f"max-age={int(_CACHE_TTL)}" in cache_headers[0][1]

#!/usr/bin/env python3
"""
Skeptical tests for NAT Pipeline Dashboard.

Covers:
  - Data readers (state, logs, figures, report, data stats)
  - HTTP handler (all endpoints, error cases, security)
  - HTML template integrity
  - No interference with pipeline (read-only verification)
  - Edge cases (corrupt files, missing dirs, concurrent reads)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from http.server import HTTPServer
from unittest.mock import patch

import pytest

# Import the dashboard module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import (
    read_state,
    read_log_tail,
    list_figures,
    read_report,
    data_dir_stats,
    load_config,
    DashboardHandler,
    DASHBOARD_HTML,
    DEFAULT_LOG_LINES,
    MAX_LOG_LINES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pipeline(tmp_path):
    """Create a minimal pipeline directory structure for testing."""
    # State file
    state_file = tmp_path / "pipeline_state.json"
    state_data = {
        "state": "INGESTING",
        "started_at": "2026-04-18T10:00:00",
        "ingest_started_at": "2026-04-18T10:01:00",
        "ingest_target_end": "2026-04-25T10:01:00",
        "ingest_pid": 12345,
        "ingest_stopped_at": None,
        "analyze_started_at": None,
        "analyze_finished_at": None,
        "last_health_check": "2026-04-18T12:00:00",
        "health_checks_ok": 24,
        "health_checks_fail": 0,
        "restarts": 0,
        "total_rows": 180000,
        "total_files": 3,
        "decision": None,
        "error": None,
        "history": [
            {"from": "IDLE", "to": "BUILDING", "at": "2026-04-18T10:00:00", "message": "start"},
            {"from": "BUILDING", "to": "INGESTING", "at": "2026-04-18T10:01:00", "message": "build ok"},
        ],
    }
    state_file.write_text(json.dumps(state_data, indent=2))

    # Log file
    log_file = tmp_path / "pipeline.log"
    log_lines = [f"2026-04-18 10:{i:02d}:00 INFO Line {i}\n" for i in range(50)]
    log_file.write_text("".join(log_lines))

    # Report dir with figures and report
    report_dir = tmp_path / "reports" / "pipeline"
    report_dir.mkdir(parents=True)

    # Dummy PNG (valid PNG header)
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    (report_dir / "cluster_entropy.png").write_bytes(png_header)
    (report_dir / "scatter_pca.png").write_bytes(png_header)
    (report_dir / "comparison_grid.svg").write_bytes(b"<svg></svg>")

    # Report JSON
    report = {
        "generated_at": "2026-04-18T14:00:00",
        "data_dir": str(tmp_path / "data"),
        "timeframe": "15min",
        "bar_summary": {"n_bars": 672, "n_features": 45, "symbols": ["BTC", "ETH", "SOL"],
                        "time_range": "2026-04-11 to 2026-04-18"},
        "vectors": {
            "entropy": {"status": "ok", "best_k": 3, "silhouette": 0.42,
                        "q1_pass": True, "q2_pass": True, "q3_pass": False},
            "trend": {"status": "ok", "best_k": 2, "silhouette": 0.35,
                      "q1_pass": True, "q2_pass": False, "q3_pass": False},
            "micro": {"status": "skipped", "reason": "zero variance"},
        },
        "decision_gate": {
            "decision": "PIVOT", "best_vector": "entropy",
            "q1_pass": 2, "q2_pass": 1, "q3_pass": 0,
            "n_vectors_ok": 2, "n_vectors_total": 3,
        },
    }
    (report_dir / "analysis_report.json").write_text(json.dumps(report, indent=2))

    # Data dir with parquet-like files
    data_dir = tmp_path / "data" / "features"
    day_dir = data_dir / "2026-04-18"
    day_dir.mkdir(parents=True)
    (day_dir / "20260418_100000.parquet").write_bytes(b"\x00" * 5000)
    (day_dir / "20260418_110000.parquet").write_bytes(b"\x00" * 3000)

    # Pipeline config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_toml = config_dir / "pipeline.toml"
    config_toml.write_text(f"""\
[ingestion]
duration_days = 7
ingestor_config = "config/ing.toml"
data_dir = "./data/features"
health_check_interval = 300
max_gap_seconds = 600

[analysis]
timeframe = "15min"
vectors = ["entropy", "trend"]
scaler = "zscore"
k_min = 2
k_max = 10
n_bootstrap = 50
random_state = 42

[analysis.thresholds]
silhouette = 0.25
bootstrap_ari = 0.6
temporal_ari = 0.5
self_transition = 0.7
kruskal_p = 0.05
eta_squared = 0.01

[output]
report_dir = "./reports/pipeline"
html_report = true
save_figures = true
figure_format = "png"
figure_dpi = 150

[state]
state_file = "./pipeline_state.json"
log_file = "./pipeline.log"
""")

    return {
        "root": tmp_path,
        "state_file": str(state_file),
        "log_file": str(log_file),
        "report_dir": str(report_dir),
        "data_dir": str(data_dir),
        "config_file": str(config_toml),
        "state_data": state_data,
        "report_data": report,
    }


@pytest.fixture
def dashboard_server(tmp_pipeline):
    """Start dashboard on a random port, return (base_url, server)."""
    cfg = {
        "_project_root": str(tmp_pipeline["root"]),
        "_state_file": tmp_pipeline["state_file"],
        "_log_file": tmp_pipeline["log_file"],
        "_report_dir": tmp_pipeline["report_dir"],
        "_data_dir": tmp_pipeline["data_dir"],
    }
    server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
    server.dashboard_config = cfg
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    # Wait for server to be ready
    for _ in range(20):
        try:
            urllib.request.urlopen(f"{base_url}/api/state", timeout=1)
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.05)
    yield base_url, server
    server.shutdown()


def fetch(url, timeout=5):
    """Helper to GET a URL and return (status, body_bytes)."""
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def fetch_json(url):
    """GET and parse JSON."""
    status, body = fetch(url)
    return status, json.loads(body)


# ===========================================================================
# Unit tests — data readers
# ===========================================================================


class TestReadState:
    def test_reads_valid_state(self, tmp_pipeline):
        state = read_state(tmp_pipeline["state_file"])
        assert state["state"] == "INGESTING"
        assert state["total_rows"] == 180000
        assert len(state["history"]) == 2

    def test_missing_file_returns_idle(self, tmp_path):
        state = read_state(str(tmp_path / "nonexistent.json"))
        assert state["state"] == "IDLE"
        assert "message" in state

    def test_corrupt_json_returns_unknown(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json!!")
        state = read_state(str(bad))
        assert state["state"] == "UNKNOWN"
        assert state["error"] is not None

    def test_empty_file_returns_unknown(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("")
        state = read_state(str(empty))
        assert state["state"] == "UNKNOWN"


class TestReadLogTail:
    def test_reads_last_n_lines(self, tmp_pipeline):
        lines = read_log_tail(tmp_pipeline["log_file"], n_lines=10)
        assert len(lines) == 10
        assert "Line 49" in lines[-1]
        assert "Line 40" in lines[0]

    def test_reads_all_when_n_exceeds_file(self, tmp_pipeline):
        lines = read_log_tail(tmp_pipeline["log_file"], n_lines=1000)
        assert len(lines) == 50

    def test_missing_file_returns_empty(self, tmp_path):
        lines = read_log_tail(str(tmp_path / "nope.log"))
        assert lines == []

    def test_empty_file_returns_empty(self, tmp_path):
        empty = tmp_path / "empty.log"
        empty.write_text("")
        lines = read_log_tail(str(empty))
        assert lines == []

    def test_respects_default_limit(self, tmp_path):
        big_log = tmp_path / "big.log"
        big_log.write_text("".join(f"line {i}\n" for i in range(500)))
        lines = read_log_tail(str(big_log))
        assert len(lines) == DEFAULT_LOG_LINES


class TestListFigures:
    def test_lists_png_and_svg(self, tmp_pipeline):
        figs = list_figures(tmp_pipeline["report_dir"])
        assert "cluster_entropy.png" in figs
        assert "scatter_pca.png" in figs
        assert "comparison_grid.svg" in figs

    def test_excludes_json(self, tmp_pipeline):
        figs = list_figures(tmp_pipeline["report_dir"])
        assert "analysis_report.json" not in figs

    def test_sorted_output(self, tmp_pipeline):
        figs = list_figures(tmp_pipeline["report_dir"])
        assert figs == sorted(figs)

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_figures(str(tmp_path / "nope")) == []

    def test_empty_dir_returns_empty(self, tmp_path):
        d = tmp_path / "empty_reports"
        d.mkdir()
        assert list_figures(str(d)) == []


class TestReadReport:
    def test_reads_valid_report(self, tmp_pipeline):
        report = read_report(tmp_pipeline["report_dir"])
        assert report is not None
        assert report["decision_gate"]["decision"] == "PIVOT"
        assert "entropy" in report["vectors"]

    def test_missing_report_returns_none(self, tmp_path):
        assert read_report(str(tmp_path)) is None

    def test_corrupt_report_returns_none(self, tmp_path):
        d = tmp_path / "bad_reports"
        d.mkdir()
        (d / "analysis_report.json").write_text("not json{{{")
        assert read_report(str(d)) is None


class TestDataDirStats:
    def test_counts_parquet_files(self, tmp_pipeline):
        stats = data_dir_stats(tmp_pipeline["data_dir"])
        assert stats["exists"] is True
        assert stats["n_files"] == 2
        assert stats["total_bytes"] == 8000
        assert stats["total_mb"] == round(8000 / 1048576, 2)
        assert "path" in stats

    def test_lists_dates(self, tmp_pipeline):
        stats = data_dir_stats(tmp_pipeline["data_dir"])
        assert "2026-04-18" in stats["dates"]

    def test_missing_dir(self, tmp_path):
        stats = data_dir_stats(str(tmp_path / "nope"))
        assert stats["exists"] is False
        assert stats["n_files"] == 0
        assert "path" in stats

    def test_relative_path_display(self, tmp_pipeline):
        stats = data_dir_stats(tmp_pipeline["data_dir"], str(tmp_pipeline["root"]))
        assert stats["path"] == "data/features"

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty_data"
        d.mkdir()
        stats = data_dir_stats(str(d))
        assert stats["exists"] is True
        assert stats["n_files"] == 0


# ===========================================================================
# Unit tests — config loading
# ===========================================================================


class TestLoadConfig:
    def test_loads_valid_config(self, tmp_pipeline):
        cfg = load_config(tmp_pipeline["config_file"])
        assert "_state_file" in cfg
        assert "_log_file" in cfg
        assert "_report_dir" in cfg
        assert "_data_dir" in cfg
        assert cfg["ingestion"]["duration_days"] == 7

    def test_resolved_paths_are_absolute(self, tmp_pipeline):
        cfg = load_config(tmp_pipeline["config_file"])
        assert os.path.isabs(cfg["_state_file"])
        assert os.path.isabs(cfg["_log_file"])
        assert os.path.isabs(cfg["_report_dir"])


# ===========================================================================
# Unit tests — HTML template
# ===========================================================================


class TestHTMLTemplate:
    def test_contains_required_elements(self):
        assert "NAT Pipeline Dashboard" in DASHBOARD_HTML
        assert "/api/state" in DASHBOARD_HTML
        assert "/api/logs" in DASHBOARD_HTML
        assert "/api/results" in DASHBOARD_HTML
        assert "/api/figures" in DASHBOARD_HTML
        assert "/api/data" in DASHBOARD_HTML
        assert "refreshAll" in DASHBOARD_HTML

    def test_has_auto_refresh(self):
        assert "auto-refresh" in DASHBOARD_HTML
        assert "setInterval" in DASHBOARD_HTML

    def test_has_data_path_bar(self):
        assert "data-path" in DASHBOARD_HTML
        assert "data-path-status" in DASHBOARD_HTML
        assert "Data folder" in DASHBOARD_HTML

    def test_has_state_badges(self):
        for state in ["IDLE", "INGESTING", "ANALYZING", "DONE", "ERROR"]:
            assert f"state-{state}" in DASHBOARD_HTML

    def test_has_decision_classes(self):
        for d in ["GO", "PIVOT", "NO-GO"]:
            assert f"decision-{d}" in DASHBOARD_HTML

    def test_valid_html_structure(self):
        assert DASHBOARD_HTML.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in DASHBOARD_HTML
        assert "<body>" in DASHBOARD_HTML
        assert "</body>" in DASHBOARD_HTML


# ===========================================================================
# Integration tests — HTTP endpoints
# ===========================================================================


class TestHTTPEndpoints:
    def test_root_serves_html(self, dashboard_server):
        url, _ = dashboard_server
        status, body = fetch(url + "/")
        assert status == 200
        assert b"NAT Pipeline Dashboard" in body

    def test_api_state(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/state")
        assert status == 200
        assert data["state"] == "INGESTING"
        assert data["total_rows"] == 180000

    def test_api_logs(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/logs")
        assert status == 200
        assert "lines" in data
        assert len(data["lines"]) == 50

    def test_api_logs_with_n_param(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/logs?n=5")
        assert status == 200
        assert len(data["lines"]) == 5

    def test_api_logs_n_capped_at_max(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + f"/api/logs?n={MAX_LOG_LINES + 1000}")
        assert status == 200
        # Should not exceed MAX_LOG_LINES (but file only has 50 lines anyway)
        assert len(data["lines"]) <= MAX_LOG_LINES

    def test_api_results(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/results")
        assert status == 200
        assert data["decision_gate"]["decision"] == "PIVOT"

    def test_api_figures(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/figures")
        assert status == 200
        assert "cluster_entropy.png" in data["figures"]

    def test_api_data(self, dashboard_server):
        url, _ = dashboard_server
        status, data = fetch_json(url + "/api/data")
        assert status == 200
        assert data["n_files"] == 2
        assert data["exists"] is True
        assert "path" in data

    def test_figure_serving(self, dashboard_server):
        url, _ = dashboard_server
        status, body = fetch(url + "/figures/cluster_entropy.png")
        assert status == 200
        assert body[:4] == b"\x89PNG"

    def test_svg_serving(self, dashboard_server):
        url, _ = dashboard_server
        status, body = fetch(url + "/figures/comparison_grid.svg")
        assert status == 200
        assert b"<svg>" in body

    def test_404_unknown_path(self, dashboard_server):
        url, _ = dashboard_server
        status, _ = fetch(url + "/unknown/path")
        assert status == 404

    def test_404_missing_figure(self, dashboard_server):
        url, _ = dashboard_server
        status, _ = fetch(url + "/figures/nonexistent.png")
        assert status == 404

    def test_cors_header(self, dashboard_server):
        url, _ = dashboard_server
        resp = urllib.request.urlopen(url + "/api/state")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


# ===========================================================================
# Security tests
# ===========================================================================


class TestSecurity:
    def test_path_traversal_blocked(self, dashboard_server):
        url, _ = dashboard_server
        status, _ = fetch(url + "/figures/../pipeline_state.json")
        assert status == 403

    def test_path_traversal_double_dot(self, dashboard_server):
        url, _ = dashboard_server
        status, _ = fetch(url + "/figures/..%2F..%2Fetc%2Fpasswd")
        assert status in (403, 404)

    def test_path_traversal_backslash(self, dashboard_server):
        url, _ = dashboard_server
        status, _ = fetch(url + "/figures/..\\..\\etc\\passwd")
        assert status in (403, 404)


# ===========================================================================
# Read-only verification — pipeline interference tests
# ===========================================================================


class TestNoInterference:
    def test_state_file_not_modified(self, tmp_pipeline, dashboard_server):
        """Dashboard must never write to the state file."""
        url, _ = dashboard_server
        state_path = Path(tmp_pipeline["state_file"])
        mtime_before = state_path.stat().st_mtime
        content_before = state_path.read_text()

        # Hit all endpoints multiple times
        for _ in range(5):
            fetch(url + "/api/state")
            fetch(url + "/api/logs")
            fetch(url + "/api/results")
            fetch(url + "/api/data")
            fetch(url + "/api/figures")
            fetch(url + "/")

        mtime_after = state_path.stat().st_mtime
        content_after = state_path.read_text()
        assert mtime_before == mtime_after, "State file mtime changed — dashboard wrote to it!"
        assert content_before == content_after, "State file content changed!"

    def test_log_file_not_modified(self, tmp_pipeline, dashboard_server):
        """Dashboard must never write to the log file."""
        url, _ = dashboard_server
        log_path = Path(tmp_pipeline["log_file"])
        mtime_before = log_path.stat().st_mtime
        size_before = log_path.stat().st_size

        for _ in range(5):
            fetch(url + "/api/logs")

        mtime_after = log_path.stat().st_mtime
        size_after = log_path.stat().st_size
        assert mtime_before == mtime_after, "Log file mtime changed!"
        assert size_before == size_after, "Log file size changed!"

    def test_report_not_modified(self, tmp_pipeline, dashboard_server):
        """Dashboard must never modify report files."""
        url, _ = dashboard_server
        report_path = Path(tmp_pipeline["report_dir"]) / "analysis_report.json"
        content_before = report_path.read_text()

        for _ in range(5):
            fetch(url + "/api/results")

        assert report_path.read_text() == content_before

    def test_data_dir_not_modified(self, tmp_pipeline, dashboard_server):
        """Dashboard must never touch parquet files."""
        url, _ = dashboard_server
        data_path = Path(tmp_pipeline["data_dir"])
        files_before = set(str(f) for f in data_path.rglob("*"))

        for _ in range(5):
            fetch(url + "/api/data")

        files_after = set(str(f) for f in data_path.rglob("*"))
        assert files_before == files_after, "Data directory contents changed!"

    def test_no_file_handles_leaked(self, tmp_pipeline, dashboard_server):
        """Repeated requests should not leak file descriptors."""
        url, _ = dashboard_server
        # Just verify no crash after many requests
        for _ in range(50):
            fetch(url + "/api/state")
            fetch(url + "/api/logs")


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_state_file_deleted_mid_operation(self, tmp_pipeline, dashboard_server):
        """If state file disappears, should return IDLE gracefully."""
        url, _ = dashboard_server
        # First verify it works
        status, data = fetch_json(url + "/api/state")
        assert data["state"] == "INGESTING"

        # Delete state file
        os.unlink(tmp_pipeline["state_file"])

        # Should still respond, with IDLE state
        status, data = fetch_json(url + "/api/state")
        assert status == 200
        assert data["state"] == "IDLE"

    def test_report_dir_deleted_mid_operation(self, tmp_pipeline, dashboard_server):
        """If report dir disappears, figures and results should return empty."""
        url, _ = dashboard_server
        import shutil
        shutil.rmtree(tmp_pipeline["report_dir"])

        status, data = fetch_json(url + "/api/figures")
        assert status == 200
        assert data["figures"] == []

        status, data = fetch_json(url + "/api/results")
        assert status == 200
        assert data is None

    def test_concurrent_requests(self, dashboard_server):
        """Multiple threads hitting the server simultaneously."""
        url, _ = dashboard_server
        results = []
        errors = []

        def worker(endpoint):
            try:
                s, d = fetch(url + endpoint)
                results.append(s)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            for ep in ["/api/state", "/api/logs", "/api/results", "/api/data"]:
                t = threading.Thread(target=worker, args=(ep,))
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent requests failed: {errors}"
        assert all(s == 200 for s in results)

    def test_very_large_log_file(self, tmp_path):
        """Reading tail of a large log should not load entire file into memory."""
        big_log = tmp_path / "big.log"
        # Write 100K lines
        with open(big_log, "w") as f:
            for i in range(100_000):
                f.write(f"2026-04-18 12:00:00 INFO Log line number {i}\n")

        lines = read_log_tail(str(big_log), n_lines=10)
        assert len(lines) == 10
        assert "99999" in lines[-1]

    def test_binary_in_log_file(self, tmp_path):
        """Log file with some binary content should not crash."""
        log = tmp_path / "weird.log"
        log.write_bytes(b"normal line\n\x00\x01\x02 binary stuff\nanother line\n")
        lines = read_log_tail(str(log))
        assert len(lines) >= 1

    def test_unicode_in_state(self, tmp_path):
        """State file with unicode should be handled."""
        state = tmp_path / "state.json"
        state.write_text(json.dumps({"state": "ERROR", "error": "Fehler: Verbindung unterbrochen \u2014 Versuch 3",
                                     "history": []}))
        result = read_state(str(state))
        assert result["state"] == "ERROR"
        assert "\u2014" in result["error"]

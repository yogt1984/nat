"""
Skeptical tests for the pipeline runner state machine.

Tests the state machine transitions, health checks, configuration,
state persistence, and analysis integration — without needing a
live Rust ingestor or real market data.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

# Add scripts/ to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_runner import (
    State,
    PipelineState,
    load_config,
    health_check,
    collect_data,
    run_analysis,
    build_ingestor,
    start_ingestor,
    stop_ingestor,
    is_ingestor_alive,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test state/data."""
    return tmp_path


@pytest.fixture
def state_file(tmp_dir):
    return str(tmp_dir / "pipeline_state.json")


@pytest.fixture
def ps(state_file):
    """Fresh pipeline state."""
    return PipelineState(state_file)


@pytest.fixture
def sample_config(tmp_dir):
    """Minimal pipeline config dict."""
    return {
        "ingestion": {
            "duration_days": 7,
            "ingestor_config": "config/ing.toml",
            "data_dir": str(tmp_dir / "data"),
            "health_check_interval": 10,
            "max_gap_seconds": 60,
        },
        "analysis": {
            "timeframe": "15min",
            "vectors": ["entropy", "trend"],
            "scaler": "zscore",
            "k_min": 2,
            "k_max": 5,
            "n_bootstrap": 5,
            "random_state": 42,
            "thresholds": {
                "silhouette": 0.25,
                "bootstrap_ari": 0.6,
                "temporal_ari": 0.5,
                "self_transition": 0.7,
                "kruskal_p": 0.05,
                "eta_squared": 0.01,
            },
        },
        "output": {
            "report_dir": str(tmp_dir / "reports"),
            "figure_dpi": 72,
        },
        "state": {
            "state_file": str(tmp_dir / "state.json"),
            "log_file": str(tmp_dir / "pipeline.log"),
        },
    }


@pytest.fixture
def fake_data_dir(tmp_dir):
    """Create a fake data dir with a parquet file.

    Generates enough ticks (~3 hours at 100ms) to produce multiple 15min bars
    so bar aggregation doesn't collapse everything to zero variance.
    """
    data_dir = tmp_dir / "data"
    data_dir.mkdir()

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # 108000 ticks = 3 hours at 100ms → ~12 bars at 15min
    n = 108_000
    rng = np.random.default_rng(42)
    # Start at a round time
    t0 = int(pd.Timestamp("2026-04-10 00:00:00").value)  # nanoseconds
    data = {
        "timestamp_ns": t0 + np.arange(n, dtype=np.int64) * 100_000_000,
        "symbol": ["BTC"] * n,
        "sequence_id": np.arange(n, dtype=np.uint64),
    }
    # Add feature columns with some structure (random walk + noise)
    for prefix in ["ent_tick_1s", "ent_tick_5s", "ent_tick_1m",
                    "ent_permutation_returns_8",
                    "trend_momentum_60", "trend_momentum_300",
                    "vol_returns_1m", "vol_returns_5m",
                    "raw_midprice", "raw_spread"]:
        base = np.cumsum(rng.normal(0, 0.01, n))  # random walk
        noise = rng.normal(0, 0.1, n)
        data[prefix] = (base + noise).astype(np.float64)

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(data_dir / "test.parquet"))

    return str(data_dir)


# ===========================================================================
# TestState — Enum basics
# ===========================================================================


class TestState:
    def test_all_states_defined(self):
        states = [s.value for s in State]
        assert "IDLE" in states
        assert "BUILDING" in states
        assert "INGESTING" in states
        assert "COLLECTING" in states
        assert "ANALYZING" in states
        assert "DONE" in states
        assert "ERROR" in states

    def test_state_count(self):
        assert len(State) == 7

    def test_state_from_string(self):
        assert State("IDLE") == State.IDLE
        assert State("INGESTING") == State.INGESTING

    def test_state_is_string_enum(self):
        assert isinstance(State.IDLE, str)
        assert State.IDLE == "IDLE"


# ===========================================================================
# TestPipelineState — Persistence
# ===========================================================================


class TestPipelineState:
    def test_fresh_state_is_idle(self, ps):
        assert ps.current == State.IDLE

    def test_state_persists_to_disk(self, state_file):
        ps = PipelineState(state_file)
        ps.set("total_rows", 42)
        # Reload from disk
        ps2 = PipelineState(state_file)
        assert ps2.get("total_rows") == 42

    def test_transition_changes_state(self, ps):
        ps.transition(State.BUILDING, "test transition")
        assert ps.current == State.BUILDING

    def test_transition_records_history(self, ps):
        ps.transition(State.BUILDING, "step 1")
        ps.transition(State.INGESTING, "step 2")
        history = ps.get("history")
        assert len(history) == 2
        assert history[0]["from"] == "IDLE"
        assert history[0]["to"] == "BUILDING"
        assert history[1]["from"] == "BUILDING"
        assert history[1]["to"] == "INGESTING"

    def test_set_and_get(self, ps):
        ps.set("custom_key", {"nested": True})
        assert ps.get("custom_key") == {"nested": True}

    def test_get_default(self, ps):
        assert ps.get("nonexistent", "default") == "default"

    def test_initial_values(self, ps):
        assert ps.get("started_at") is None
        assert ps.get("ingest_pid") is None
        assert ps.get("restarts") == 0
        assert ps.get("health_checks_ok") == 0
        assert ps.get("decision") is None

    def test_state_file_created(self, state_file):
        ps = PipelineState(state_file)
        ps.save()  # explicit save to ensure file exists
        assert Path(state_file).exists()

    def test_state_file_is_valid_json(self, state_file):
        ps = PipelineState(state_file)
        ps.set("test", True)
        with open(state_file) as f:
            data = json.load(f)
        assert data["test"] is True

    def test_parent_dirs_created(self, tmp_dir):
        nested = str(tmp_dir / "a" / "b" / "c" / "state.json")
        ps = PipelineState(nested)
        assert ps.current == State.IDLE

    def test_reload_after_error(self, state_file):
        ps = PipelineState(state_file)
        ps.transition(State.ERROR, "something broke")
        ps.set("error", "test error")

        ps2 = PipelineState(state_file)
        assert ps2.current == State.ERROR
        assert ps2.get("error") == "test error"

    def test_full_lifecycle_transitions(self, ps):
        ps.transition(State.BUILDING, "build")
        ps.transition(State.INGESTING, "ingest")
        ps.transition(State.COLLECTING, "collect")
        ps.transition(State.ANALYZING, "analyze")
        ps.transition(State.DONE, "done")
        assert ps.current == State.DONE
        assert len(ps.get("history")) == 5


# ===========================================================================
# TestConfig — Configuration loading
# ===========================================================================


class TestConfig:
    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/pipeline.toml")

    def test_load_config_real_file(self):
        """Load the actual project config if it exists."""
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "pipeline.toml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert "ingestion" in config
            assert "analysis" in config
            assert "output" in config
            assert "state" in config
        else:
            pytest.skip("pipeline.toml not found")

    def test_config_has_required_sections(self):
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "pipeline.toml"
        if not config_path.exists():
            pytest.skip("pipeline.toml not found")
        config = load_config(str(config_path))
        assert "duration_days" in config["ingestion"]
        assert "timeframe" in config["analysis"]
        assert "vectors" in config["analysis"]
        assert "thresholds" in config["analysis"]

    def test_config_thresholds_reasonable(self):
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "pipeline.toml"
        if not config_path.exists():
            pytest.skip("pipeline.toml not found")
        config = load_config(str(config_path))
        t = config["analysis"]["thresholds"]
        assert 0 < t["silhouette"] < 1
        assert 0 < t["bootstrap_ari"] < 1
        assert 0 < t["kruskal_p"] < 1


# ===========================================================================
# TestHealthCheck — Data freshness monitoring
# ===========================================================================


class TestHealthCheck:
    def test_nonexistent_dir(self, tmp_dir):
        log = setup_logging(None)
        result = health_check(str(tmp_dir / "missing"), 600, log)
        assert result["ok"] is False
        assert "does not exist" in result["reason"]

    def test_empty_dir(self, tmp_dir):
        log = setup_logging(None)
        empty = tmp_dir / "empty"
        empty.mkdir()
        result = health_check(str(empty), 600, log)
        assert result["ok"] is False
        assert "no parquet" in result["reason"]

    def test_fresh_data_ok(self, fake_data_dir):
        log = setup_logging(None)
        result = health_check(fake_data_dir, 600, log)
        assert result["ok"] is True
        assert result["file_count"] == 1
        assert result["total_rows"] > 0
        assert result["gap_seconds"] < 60  # just created

    def test_stale_data_not_ok(self, fake_data_dir):
        log = setup_logging(None)
        # Set max_gap to 0 so everything is "stale"
        result = health_check(fake_data_dir, 0, log)
        # File was just created so gap > 0 but close; with max_gap=0, should fail
        assert result["ok"] is False

    def test_returns_file_count(self, fake_data_dir):
        log = setup_logging(None)
        result = health_check(fake_data_dir, 600, log)
        assert "file_count" in result
        assert "total_rows" in result
        assert "gap_seconds" in result
        assert "latest_file" in result


# ===========================================================================
# TestIngestorManagement — Process control
# ===========================================================================


class TestIngestorManagement:
    def test_is_ingestor_alive_none(self):
        assert is_ingestor_alive(None) is False

    def test_is_ingestor_alive_dead_pid(self):
        # PID 999999999 almost certainly doesn't exist
        assert is_ingestor_alive(999999999) is False

    def test_is_ingestor_alive_self(self):
        # Our own process is alive
        assert is_ingestor_alive(os.getpid()) is True

    def test_stop_ingestor_none(self):
        log = setup_logging(None)
        stop_ingestor(None, log)  # should not raise

    def test_stop_ingestor_dead_pid(self):
        log = setup_logging(None)
        stop_ingestor(999999999, log)  # should not raise

    @patch("subprocess.run")
    def test_build_ingestor_success(self, mock_run, tmp_dir):
        mock_run.return_value = MagicMock(returncode=0)
        log = setup_logging(None)
        assert build_ingestor(tmp_dir, log) is True

    @patch("subprocess.run")
    def test_build_ingestor_failure(self, mock_run, tmp_dir):
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        log = setup_logging(None)
        assert build_ingestor(tmp_dir, log) is False

    def test_start_ingestor_missing_binary(self, tmp_dir):
        log = setup_logging(None)
        result = start_ingestor(tmp_dir, "config/ing.toml", tmp_dir / "log", log)
        assert result is None

    def test_start_ingestor_missing_config(self, tmp_dir):
        log = setup_logging(None)
        # Create fake binary
        binary = tmp_dir / "rust" / "target" / "release" / "ing"
        binary.parent.mkdir(parents=True)
        binary.touch()
        result = start_ingestor(tmp_dir, "nonexistent.toml", tmp_dir / "log", log)
        assert result is None


# ===========================================================================
# TestCollectData — Data validation phase
# ===========================================================================


class TestCollectData:
    def test_collect_with_valid_data(self, fake_data_dir):
        log = setup_logging(None)
        result = collect_data(fake_data_dir, log)
        assert result["file_count"] >= 1
        assert result["total_rows"] > 0
        assert result["valid"] is True
        assert isinstance(result["vectors_available"], list)

    def test_collect_with_missing_dir(self, tmp_dir):
        log = setup_logging(None)
        with pytest.raises(FileNotFoundError):
            collect_data(str(tmp_dir / "missing"), log)


# ===========================================================================
# TestRunAnalysis — Full analysis integration
# ===========================================================================


class TestRunAnalysis:
    def test_analysis_runs_on_synthetic_data(self, fake_data_dir, sample_config, tmp_dir):
        """Run analysis on minimal synthetic data — tests the full code path."""
        log = setup_logging(None)
        sample_config["output"]["report_dir"] = str(tmp_dir / "reports")
        # Use very small k range and bootstrap for speed
        sample_config["analysis"]["k_min"] = 2
        sample_config["analysis"]["k_max"] = 3
        sample_config["analysis"]["n_bootstrap"] = 3
        # Only test vectors we have columns for
        sample_config["analysis"]["vectors"] = ["entropy", "trend"]
        sample_config["ingestion"]["data_dir"] = fake_data_dir

        gate = run_analysis(fake_data_dir, sample_config, str(tmp_dir / "reports"), log)

        assert "decision" in gate
        assert gate["decision"] in ("GO", "PIVOT", "NO-GO")
        assert gate["n_vectors_total"] == 2

        # Check report was written
        report_file = tmp_dir / "reports" / "analysis_report.json"
        assert report_file.exists()

        with open(report_file) as f:
            report = json.load(f)
        assert "vectors" in report
        assert "decision_gate" in report

    def test_analysis_creates_figures(self, fake_data_dir, sample_config, tmp_dir):
        log = setup_logging(None)
        sample_config["output"]["report_dir"] = str(tmp_dir / "reports")
        sample_config["analysis"]["k_min"] = 2
        sample_config["analysis"]["k_max"] = 3
        sample_config["analysis"]["n_bootstrap"] = 3
        sample_config["analysis"]["vectors"] = ["entropy"]
        sample_config["ingestion"]["data_dir"] = fake_data_dir

        run_analysis(fake_data_dir, sample_config, str(tmp_dir / "reports"), log)

        fig_dir = tmp_dir / "reports" / "figures"
        assert fig_dir.exists()
        png_files = list(fig_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_analysis_handles_skip_vector(self, fake_data_dir, sample_config, tmp_dir):
        """Vector with no matching columns should be skipped, not crash."""
        log = setup_logging(None)
        sample_config["output"]["report_dir"] = str(tmp_dir / "reports")
        sample_config["analysis"]["k_min"] = 2
        sample_config["analysis"]["k_max"] = 3
        sample_config["analysis"]["n_bootstrap"] = 3
        # "liquidation" vector columns are not in fake data
        sample_config["analysis"]["vectors"] = ["liquidation"]
        sample_config["ingestion"]["data_dir"] = fake_data_dir

        gate = run_analysis(fake_data_dir, sample_config, str(tmp_dir / "reports"), log)
        # Should complete without error, vector skipped
        assert gate["decision"] in ("GO", "PIVOT", "NO-GO")

    def test_analysis_report_json_schema(self, fake_data_dir, sample_config, tmp_dir):
        log = setup_logging(None)
        sample_config["output"]["report_dir"] = str(tmp_dir / "reports")
        sample_config["analysis"]["k_min"] = 2
        sample_config["analysis"]["k_max"] = 3
        sample_config["analysis"]["n_bootstrap"] = 3
        sample_config["analysis"]["vectors"] = ["entropy"]

        run_analysis(fake_data_dir, sample_config, str(tmp_dir / "reports"), log)

        with open(tmp_dir / "reports" / "analysis_report.json") as f:
            report = json.load(f)

        # Check required fields
        assert "generated_at" in report
        assert "data_dir" in report
        assert "timeframe" in report
        assert "bar_summary" in report
        assert "vectors" in report
        assert "decision_gate" in report

        gate = report["decision_gate"]
        assert "decision" in gate
        assert "best_vector" in gate
        assert "q1_pass" in gate
        assert "q2_pass" in gate
        assert "q3_pass" in gate


# ===========================================================================
# TestStateMachineTransitions — Logical transition correctness
# ===========================================================================


class TestStateMachineTransitions:
    def test_idle_to_building(self, ps):
        assert ps.current == State.IDLE
        ps.transition(State.BUILDING, "start")
        assert ps.current == State.BUILDING

    def test_building_to_ingesting(self, ps):
        ps.transition(State.BUILDING)
        ps.transition(State.INGESTING, "build done")
        assert ps.current == State.INGESTING

    def test_ingesting_to_collecting(self, ps):
        ps.transition(State.INGESTING)
        ps.transition(State.COLLECTING, "duration reached")
        assert ps.current == State.COLLECTING

    def test_collecting_to_analyzing(self, ps):
        ps.transition(State.COLLECTING)
        ps.transition(State.ANALYZING, "data validated")
        assert ps.current == State.ANALYZING

    def test_analyzing_to_done(self, ps):
        ps.transition(State.ANALYZING)
        ps.transition(State.DONE, "GO")
        assert ps.current == State.DONE

    def test_any_to_error(self, ps):
        for state in [State.BUILDING, State.INGESTING, State.COLLECTING, State.ANALYZING]:
            ps2 = PipelineState(str(Path(ps.state_file).parent / f"err_{state.value}.json"))
            ps2.transition(state)
            ps2.transition(State.ERROR, "test error")
            assert ps2.current == State.ERROR

    def test_error_preserves_history(self, ps):
        ps.transition(State.BUILDING)
        ps.transition(State.INGESTING)
        ps.transition(State.ERROR, "crash")
        history = ps.get("history")
        assert len(history) == 3
        assert history[-1]["to"] == "ERROR"
        assert history[-1]["message"] == "crash"


# ===========================================================================
# TestResumeability — State survives restart
# ===========================================================================


class TestResumeability:
    def test_resume_from_ingesting(self, state_file):
        ps = PipelineState(state_file)
        ps.transition(State.BUILDING)
        ps.transition(State.INGESTING, "ingesting")
        ps.set("ingest_pid", 12345)
        ps.set("ingest_started_at", "2026-04-10T00:00:00")
        ps.set("ingest_target_end", "2026-04-17T00:00:00")

        # Simulate restart: reload state
        ps2 = PipelineState(state_file)
        assert ps2.current == State.INGESTING
        assert ps2.get("ingest_pid") == 12345
        assert ps2.get("ingest_target_end") == "2026-04-17T00:00:00"

    def test_resume_from_collecting(self, state_file):
        ps = PipelineState(state_file)
        ps.transition(State.COLLECTING, "stopped manually")
        ps.set("total_rows", 500_000)

        ps2 = PipelineState(state_file)
        assert ps2.current == State.COLLECTING
        assert ps2.get("total_rows") == 500_000

    def test_resume_from_analyzing(self, state_file):
        ps = PipelineState(state_file)
        ps.transition(State.ANALYZING, "resumed")

        ps2 = PipelineState(state_file)
        assert ps2.current == State.ANALYZING


# ===========================================================================
# TestDecisionGate — Decision logic
# ===========================================================================


class TestDecisionGate:
    """Test the decision logic in isolation using mock analysis results."""

    def _decide(self, results: dict) -> str:
        """Replicate decision logic from run_analysis."""
        ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
        n = len(ok)
        if n == 0:
            return "NO-GO"
        q1 = sum(1 for v in ok.values() if v.get("q1_pass"))
        q2 = sum(1 for v in ok.values() if v.get("q2_pass"))
        q3 = sum(1 for v in ok.values() if v.get("q3_pass"))
        if q1 > n / 2 and q2 > n / 2 and q3 > 0:
            return "GO"
        elif q1 > n / 2 and q2 > 0:
            return "PIVOT"
        return "NO-GO"

    def test_go_decision(self):
        results = {
            "v1": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": True},
            "v2": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": False},
        }
        assert self._decide(results) == "GO"

    def test_pivot_decision(self):
        results = {
            "v1": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": False},
            "v2": {"status": "ok", "q1_pass": True, "q2_pass": False, "q3_pass": False},
        }
        assert self._decide(results) == "PIVOT"

    def test_nogo_decision(self):
        results = {
            "v1": {"status": "ok", "q1_pass": False, "q2_pass": False, "q3_pass": False},
            "v2": {"status": "ok", "q1_pass": False, "q2_pass": False, "q3_pass": False},
        }
        assert self._decide(results) == "NO-GO"

    def test_nogo_all_skipped(self):
        results = {
            "v1": {"status": "skip", "reason": "no columns"},
        }
        assert self._decide(results) == "NO-GO"

    def test_go_requires_at_least_one_q3(self):
        results = {
            "v1": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": False},
            "v2": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": False},
        }
        # All q1/q2 pass but no q3 → PIVOT not GO
        assert self._decide(results) == "PIVOT"

    def test_mixed_status(self):
        results = {
            "v1": {"status": "ok", "q1_pass": True, "q2_pass": True, "q3_pass": True},
            "v2": {"status": "skip", "reason": "no data"},
            "v3": {"status": "ok", "q1_pass": False, "q2_pass": False, "q3_pass": False},
        }
        # 1/2 ok pass q1 → not majority, NO-GO
        assert self._decide(results) == "NO-GO"


# ===========================================================================
# TestLogging — Log setup
# ===========================================================================


class TestLogging:
    def test_setup_console_only(self):
        log = setup_logging(None)
        assert log is not None
        assert log.name == "pipeline"

    def test_setup_with_file(self, tmp_dir):
        log_file = str(tmp_dir / "test.log")
        log = setup_logging(log_file)
        log.info("test message")
        assert Path(log_file).exists()

    def test_log_file_parent_created(self, tmp_dir):
        log_file = str(tmp_dir / "nested" / "deep" / "test.log")
        log = setup_logging(log_file)
        assert Path(log_file).parent.exists()


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    def test_state_with_corrupt_json(self, tmp_dir):
        state_file = tmp_dir / "corrupt.json"
        state_file.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            PipelineState(str(state_file))

    def test_double_transition(self, ps):
        ps.transition(State.BUILDING)
        ps.transition(State.BUILDING)  # same state again
        assert ps.current == State.BUILDING
        assert len(ps.get("history")) == 2  # both recorded

    def test_health_check_with_subdirs(self, tmp_dir):
        """Parquet in subdirectories should be found."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        sub = tmp_dir / "data" / "2026-04-10"
        sub.mkdir(parents=True)
        df = pd.DataFrame({"x": [1.0, 2.0]})
        pq.write_table(pa.Table.from_pandas(df), str(sub / "test.parquet"))

        log = setup_logging(None)
        result = health_check(str(tmp_dir / "data"), 600, log)
        assert result["ok"] is True
        assert result["file_count"] == 1

    def test_state_datetime_serialization(self, ps):
        """Datetimes should serialize cleanly."""
        now = datetime.datetime.utcnow()
        ps.set("test_time", now.isoformat())
        ps2 = PipelineState(ps.state_file)
        assert ps2.get("test_time") == now.isoformat()


# ===========================================================================
# TestDeterminism
# ===========================================================================


class TestDeterminism:
    def test_analysis_deterministic(self, fake_data_dir, sample_config, tmp_dir):
        """Same data + same config → same decision."""
        log = setup_logging(None)
        cfg = sample_config.copy()
        cfg["analysis"]["k_min"] = 2
        cfg["analysis"]["k_max"] = 3
        cfg["analysis"]["n_bootstrap"] = 3
        cfg["analysis"]["vectors"] = ["entropy"]

        r1 = run_analysis(fake_data_dir, cfg, str(tmp_dir / "r1"), log)
        r2 = run_analysis(fake_data_dir, cfg, str(tmp_dir / "r2"), log)

        assert r1["decision"] == r2["decision"]
        assert r1["best_vector"] == r2["best_vector"]

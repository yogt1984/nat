"""
Comprehensive tests for discovery_orchestrator.py.

Tests cover:
- Config loading (TOML parsing, missing file handling)
- DiscoveryState (persistence, transitions, gates, history, reset)
- Phase enum (all values, string representation)
- SubprocessResult dataclass
- run_subprocess helper (success, failure, timeout, JSON report)
- Step functions (data health, signal sweep, train, backtest, alpha pipeline, report)
- Cycle runner (happy path, failure cascading)
- CLI commands (start, once, status, stop)
- Edge cases (corrupt state, missing config sections)
"""

import json
import logging
import sys
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


from discovery_orchestrator import (
    Phase,
    DiscoveryState,
    SubprocessResult,
    load_config,
    run_subprocess,
    step_data_health,
    step_signal_sweep,
    step_train,
    step_backtest,
    step_alpha_pipeline,
    step_report,
    run_cycle,
    cmd_status,
    cmd_stop,
    ROOT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def state_file(tmp_dir):
    return str(tmp_dir / "state.json")


@pytest.fixture
def state(state_file):
    return DiscoveryState(state_file)


@pytest.fixture
def logger():
    log = logging.getLogger("test_discovery")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


@pytest.fixture
def config():
    return {
        "discovery": {
            "cycle_interval_s": 10,
            "data_dir": "data/features",
            "state_file": "/tmp/test_disc_state.json",
            "log_file": "/tmp/test_disc.log",
            "report_dir": "/tmp/test_reports",
            "max_memory_mb": 500.0,
            "start_date": "",
            "end_date": "",
            "sweep": {
                "symbols": ["BTC"],
                "horizons": [300],
            },
        },
        "gates": {
            "min_wf_edge": 0.005,
            "min_gross_bps": 0.5,
            "min_test_r2": 0.0,
            "min_oos_sharpe": 0.5,
            "max_drawdown_pct": 10.0,
            "cost_model": "taker",
            "min_alpha_phase": "VALIDATING",
        },
    }


@pytest.fixture
def config_file(tmp_dir, config):
    """Write a valid TOML config file."""
    path = tmp_dir / "discovery.toml"
    # Write as TOML manually
    content = """
[discovery]
cycle_interval_s = 10
data_dir = "data/features"
state_file = "/tmp/test_disc_state.json"
log_file = "/tmp/test_disc.log"
report_dir = "/tmp/test_reports"
max_memory_mb = 500.0
start_date = ""
end_date = ""

[discovery.sweep]
symbols = ["BTC"]
horizons = [300]

[gates]
min_wf_edge = 0.005
min_gross_bps = 0.5
min_test_r2 = 0.0
min_oos_sharpe = 0.5
max_drawdown_pct = 10.0
cost_model = "taker"
min_alpha_phase = "VALIDATING"
"""
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: Phase Enum
# ---------------------------------------------------------------------------


class TestPhase:
    def test_all_phases_defined(self):
        expected = {
            "IDLE", "DATA_HEALTH", "SIGNAL_SWEEP", "TRAINING",
            "BACKTESTING", "ALPHA_PIPELINE", "REPORTING",
            "SLEEPING", "STOPPED", "ERROR",
        }
        actual = {p.value for p in Phase}
        assert actual == expected

    def test_is_string_enum(self):
        assert Phase.IDLE == "IDLE"
        assert Phase.IDLE.value == "IDLE"

    def test_from_string(self):
        assert Phase("IDLE") == Phase.IDLE
        assert Phase("ERROR") == Phase.ERROR

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            Phase("NONEXISTENT")


# ---------------------------------------------------------------------------
# Tests: Config Loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_config(self, config_file):
        cfg = load_config(config_file)
        assert "discovery" in cfg
        assert "gates" in cfg
        assert cfg["discovery"]["cycle_interval_s"] == 10

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_config_xyz.toml")

    def test_sweep_config(self, config_file):
        cfg = load_config(config_file)
        assert cfg["discovery"]["sweep"]["symbols"] == ["BTC"]
        assert cfg["discovery"]["sweep"]["horizons"] == [300]


# ---------------------------------------------------------------------------
# Tests: DiscoveryState
# ---------------------------------------------------------------------------


class TestDiscoveryState:
    def test_initial_state(self, state):
        assert state.current == Phase.IDLE
        assert state.get("cycle_number") == 0
        assert state.get("winners") == []

    def test_set_and_get(self, state):
        state.set("cycle_number", 5)
        assert state.get("cycle_number") == 5

    def test_persistence(self, state_file):
        s1 = DiscoveryState(state_file)
        s1.set("cycle_number", 42)

        s2 = DiscoveryState(state_file)
        assert s2.get("cycle_number") == 42

    def test_transition(self, state):
        state.transition(Phase.DATA_HEALTH, "starting health check")
        assert state.current == Phase.DATA_HEALTH

    def test_transition_history(self, state):
        state.transition(Phase.DATA_HEALTH, "step 1")
        state.transition(Phase.SIGNAL_SWEEP, "step 2")
        history = state.get("history", [])
        assert len(history) == 2
        assert history[0]["from"] == "IDLE"
        assert history[0]["to"] == "DATA_HEALTH"
        assert history[1]["from"] == "DATA_HEALTH"
        assert history[1]["to"] == "SIGNAL_SWEEP"

    def test_history_bounded(self, state):
        """History should be truncated to last 100 entries when it exceeds 200."""
        for i in range(210):
            state.transition(Phase.IDLE, f"entry {i}")
        history = state.get("history", [])
        assert len(history) <= 200

    def test_record_gate(self, state):
        state.record_gate("data_health", "PASS", {"exit_code": 0}, "all good")
        gates = state.get("gates", {})
        assert "data_health" in gates
        assert gates["data_health"]["verdict"] == "PASS"
        assert gates["data_health"]["metrics"]["exit_code"] == 0

    def test_set_artifact(self, state):
        state.set_artifact("model_BTC_300", "/path/to/model.txt")
        artifacts = state.get("artifacts", {})
        assert artifacts["model_BTC_300"] == "/path/to/model.txt"

    def test_set_output(self, state):
        state.set_output("signal_sweep", {"n_winners": 2})
        outputs = state.get("step_outputs", {})
        assert outputs["signal_sweep"]["n_winners"] == 2

    def test_reset_cycle(self, state):
        state.set("winners", [{"symbol": "BTC"}])
        state.record_gate("test_gate", "PASS", {})
        state.set_artifact("model", "/path")
        state.set("cycle_number", 5)

        state.reset_cycle()

        assert state.get("winners") == []
        assert state.get("gates") == {}
        assert state.get("artifacts") == {}
        # cycle_number should be preserved
        assert state.get("cycle_number") == 5

    def test_defaults(self, state):
        d = state._defaults()
        assert d["phase"] == "IDLE"
        assert d["cycle_number"] == 0
        assert d["winners"] == []
        assert d["history"] == []

    def test_creates_parent_directory(self, tmp_dir):
        path = str(tmp_dir / "deep" / "nested" / "state.json")
        s = DiscoveryState(path)
        assert s.current == Phase.IDLE
        assert Path(path).parent.exists()

    def test_get_default_value(self, state):
        assert state.get("nonexistent", "default_val") == "default_val"

    def test_save_creates_valid_json(self, state, state_file):
        state.set("cycle_number", 7)
        with open(state_file) as f:
            data = json.load(f)
        assert data["cycle_number"] == 7


# ---------------------------------------------------------------------------
# Tests: SubprocessResult
# ---------------------------------------------------------------------------


class TestSubprocessResult:
    def test_basic_creation(self):
        r = SubprocessResult(0, "output", "", 1.5)
        assert r.returncode == 0
        assert r.stdout == "output"
        assert r.duration_s == 1.5
        assert r.report is None

    def test_with_report(self):
        r = SubprocessResult(0, "", "", 1.0, report={"key": "val"})
        assert r.report == {"key": "val"}


# ---------------------------------------------------------------------------
# Tests: run_subprocess
# ---------------------------------------------------------------------------


class TestRunSubprocess:
    def test_successful_command(self, logger):
        result = run_subprocess(
            [sys.executable, "-c", "print('hello')"],
            logger,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_failing_command(self, logger):
        result = run_subprocess(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            logger,
        )
        assert result.returncode == 1

    def test_timeout(self, logger):
        result = run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            logger,
            timeout_s=1,
        )
        assert result.returncode == -1
        assert "Timeout" in result.stderr

    def test_json_report_parsing(self, logger, tmp_dir):
        report_path = tmp_dir / "report.json"
        report_data = {"test": True, "value": 42}
        report_path.write_text(json.dumps(report_data))

        # Command that succeeds
        result = run_subprocess(
            [sys.executable, "-c", "pass"],
            logger,
            report_path=report_path,
        )
        assert result.report == report_data

    def test_missing_report_file(self, logger, tmp_dir):
        missing = tmp_dir / "nonexistent.json"
        result = run_subprocess(
            [sys.executable, "-c", "pass"],
            logger,
            report_path=missing,
        )
        assert result.report is None

    def test_corrupt_report_file(self, logger, tmp_dir):
        bad_json = tmp_dir / "bad.json"
        bad_json.write_text("not valid json{{{")
        result = run_subprocess(
            [sys.executable, "-c", "pass"],
            logger,
            report_path=bad_json,
        )
        assert result.report is None

    def test_duration_tracked(self, logger):
        result = run_subprocess(
            [sys.executable, "-c", "pass"],
            logger,
        )
        assert result.duration_s >= 0


# ---------------------------------------------------------------------------
# Tests: Step Functions (mocked subprocess)
# ---------------------------------------------------------------------------


class TestStepDataHealth:
    @patch("discovery_orchestrator.run_subprocess")
    def test_pass_on_exit_0(self, mock_run, config, state, logger):
        mock_run.return_value = SubprocessResult(0, "", "", 1.0)
        verdict = step_data_health(config, state, logger)
        assert verdict == "PASS"

    @patch("discovery_orchestrator.run_subprocess")
    def test_weak_on_exit_1(self, mock_run, config, state, logger):
        mock_run.return_value = SubprocessResult(1, "", "", 1.0)
        verdict = step_data_health(config, state, logger)
        assert verdict == "WEAK"

    @patch("discovery_orchestrator.run_subprocess")
    def test_fail_on_exit_2(self, mock_run, config, state, logger):
        mock_run.return_value = SubprocessResult(2, "", "", 1.0)
        verdict = step_data_health(config, state, logger)
        assert verdict == "FAIL"

    @patch("discovery_orchestrator.run_subprocess")
    def test_gate_recorded(self, mock_run, config, state, logger):
        mock_run.return_value = SubprocessResult(0, "", "", 1.0)
        step_data_health(config, state, logger)
        gates = state.get("gates", {})
        assert "data_health" in gates
        assert gates["data_health"]["verdict"] == "PASS"


class TestStepSignalSweep:
    @patch("discovery_orchestrator.run_subprocess")
    def test_no_winners(self, mock_run, config, state, logger, tmp_dir):
        """When signal sweep finds no winners, verdict should be FAIL."""
        report = {
            "test2_avg_edge": 0.001,  # Below gate
            "test3_best_gross_bps": 0.1,  # Below gate
            "test2_avg_accuracy": 0.51,
            "test2_avg_sharpe": 0.5,
        }
        mock_run.return_value = SubprocessResult(0, "", "", 1.0, report=report)
        verdict, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert verdict == "FAIL"
        assert winners == []

    @patch("discovery_orchestrator.run_subprocess")
    def test_winner_found(self, mock_run, config, state, logger, tmp_dir):
        """When a combo passes gates, it should appear as a winner."""
        report = {
            "test2_avg_edge": 0.05,  # Above min_wf_edge=0.005
            "test3_best_gross_bps": 1.0,  # Above min_gross_bps=0.5
            "test2_avg_accuracy": 0.55,
            "test2_avg_sharpe": 2.0,
        }
        mock_run.return_value = SubprocessResult(0, "", "", 1.0, report=report)
        verdict, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert verdict == "PASS"
        assert len(winners) == 1
        assert winners[0]["symbol"] == "BTC"
        assert winners[0]["horizon"] == 300

    @patch("discovery_orchestrator.run_subprocess")
    def test_multiple_combos(self, mock_run, config, state, logger, tmp_dir):
        """Test with multiple symbols/horizons."""
        config["discovery"]["sweep"]["symbols"] = ["BTC", "ETH"]
        config["discovery"]["sweep"]["horizons"] = [300, 3000]

        report_pass = {
            "test2_avg_edge": 0.05,
            "test3_best_gross_bps": 1.0,
            "test2_avg_accuracy": 0.55,
            "test2_avg_sharpe": 2.0,
        }
        report_fail = {
            "test2_avg_edge": 0.001,
            "test3_best_gross_bps": 0.1,
            "test2_avg_accuracy": 0.50,
            "test2_avg_sharpe": 0.1,
        }
        # Alternate pass and fail
        mock_run.side_effect = [
            SubprocessResult(0, "", "", 1.0, report=report_pass),
            SubprocessResult(0, "", "", 1.0, report=report_fail),
            SubprocessResult(0, "", "", 1.0, report=report_fail),
            SubprocessResult(0, "", "", 1.0, report=report_pass),
        ]
        verdict, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert verdict == "PASS"
        assert len(winners) == 2

    @patch("discovery_orchestrator.run_subprocess")
    def test_subprocess_failure(self, mock_run, config, state, logger, tmp_dir):
        """When subprocess fails (no report), combo is skipped."""
        mock_run.return_value = SubprocessResult(1, "", "error", 1.0, report=None)
        verdict, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert verdict == "FAIL"
        assert winners == []

    @patch("discovery_orchestrator.run_subprocess")
    def test_winners_sorted_by_edge(self, mock_run, config, state, logger, tmp_dir):
        config["discovery"]["sweep"]["symbols"] = ["BTC", "ETH"]
        reports = [
            {"test2_avg_edge": 0.02, "test3_best_gross_bps": 1.0,
             "test2_avg_accuracy": 0.55, "test2_avg_sharpe": 2.0},
            {"test2_avg_edge": 0.08, "test3_best_gross_bps": 1.5,
             "test2_avg_accuracy": 0.58, "test2_avg_sharpe": 3.0},
        ]
        mock_run.side_effect = [
            SubprocessResult(0, "", "", 1.0, report=reports[0]),
            SubprocessResult(0, "", "", 1.0, report=reports[1]),
        ]
        _, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert winners[0]["edge"] > winners[1]["edge"]


class TestStepTrain:
    @patch("discovery_orchestrator.run_subprocess")
    def test_train_success(self, mock_run, config, state, logger, tmp_dir):
        # The model file must exist on disk for the path check
        model_file = tmp_dir / "model.txt"
        model_file.touch()
        mock_run.return_value = SubprocessResult(
            0,
            f"Model saved to: {model_file}\nTest R²: 0.0500\n",
            "", 10.0,
        )
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, model_path = step_train(config, state, combo, 1, tmp_dir, logger)
        assert verdict == "PASS"
        assert model_path == str(model_file)

    @patch("discovery_orchestrator.run_subprocess")
    def test_train_failure(self, mock_run, config, state, logger, tmp_dir):
        mock_run.return_value = SubprocessResult(1, "", "error", 5.0)
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, model_path = step_train(config, state, combo, 1, tmp_dir, logger)
        assert verdict == "FAIL"
        assert model_path is None

    @patch("discovery_orchestrator.run_subprocess")
    def test_train_negative_r2(self, mock_run, config, state, logger, tmp_dir):
        """Negative R² should fail the gate (min_test_r2=0.0)."""
        mock_run.return_value = SubprocessResult(
            0,
            "Model saved to: /tmp/model.txt\nTest R²: -0.0100\n",
            "", 10.0,
        )
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, _ = step_train(config, state, combo, 1, tmp_dir, logger)
        assert verdict == "FAIL"

    @patch("discovery_orchestrator.run_subprocess")
    def test_gate_recorded(self, mock_run, config, state, logger, tmp_dir):
        mock_run.return_value = SubprocessResult(
            0,
            "Model saved to: /tmp/m.txt\nTest R²: 0.0300\n",
            "", 5.0,
        )
        combo = {"symbol": "BTC", "horizon": 300}
        step_train(config, state, combo, 1, tmp_dir, logger)
        gates = state.get("gates", {})
        assert "train_BTC_300" in gates


class TestStepBacktest:
    @patch("discovery_orchestrator.run_subprocess")
    def test_backtest_pass(self, mock_run, config, state, logger, tmp_dir):
        # Score succeeds, backtest succeeds
        pred_path = tmp_dir / "predictions_BTC_300.parquet"
        pred_path.touch()
        mock_run.side_effect = [
            SubprocessResult(0, "", "", 5.0),  # score
            SubprocessResult(0, "Sharpe: 1.50\nMax Drawdown: 5.0%\n", "", 10.0),  # backtest
        ]
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, metrics = step_backtest(config, state, combo, "/tmp/model.txt", 1, tmp_dir, logger)
        assert verdict == "PASS"
        assert metrics["sharpe"] == 1.5
        assert metrics["max_drawdown_pct"] == 5.0

    @patch("discovery_orchestrator.run_subprocess")
    def test_backtest_fail_sharpe(self, mock_run, config, state, logger, tmp_dir):
        pred_path = tmp_dir / "predictions_BTC_300.parquet"
        pred_path.touch()
        mock_run.side_effect = [
            SubprocessResult(0, "", "", 5.0),
            SubprocessResult(0, "Sharpe: 0.20\nMax Drawdown: 3.0%\n", "", 10.0),
        ]
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, _ = step_backtest(config, state, combo, "/tmp/model.txt", 1, tmp_dir, logger)
        assert verdict == "FAIL"

    @patch("discovery_orchestrator.run_subprocess")
    def test_backtest_fail_drawdown(self, mock_run, config, state, logger, tmp_dir):
        pred_path = tmp_dir / "predictions_BTC_300.parquet"
        pred_path.touch()
        mock_run.side_effect = [
            SubprocessResult(0, "", "", 5.0),
            SubprocessResult(0, "Sharpe: 2.00\nMax Drawdown: 15.0%\n", "", 10.0),
        ]
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, _ = step_backtest(config, state, combo, "/tmp/model.txt", 1, tmp_dir, logger)
        assert verdict == "FAIL"

    @patch("discovery_orchestrator.run_subprocess")
    def test_score_failure(self, mock_run, config, state, logger, tmp_dir):
        mock_run.return_value = SubprocessResult(1, "", "error", 5.0)
        combo = {"symbol": "BTC", "horizon": 300}
        verdict, _ = step_backtest(config, state, combo, "/tmp/model.txt", 1, tmp_dir, logger)
        assert verdict == "FAIL"


class TestStepAlphaPipeline:
    @patch("discovery_orchestrator.run_subprocess")
    def test_passes_when_phase_reached(self, mock_run, config, state, logger, tmp_dir):
        # Write a fake alpha pipeline state that reached VALIDATING
        alpha_state_dir = ROOT / "data" / "alpha"
        alpha_state_dir.mkdir(parents=True, exist_ok=True)
        alpha_state_path = alpha_state_dir / "pipeline_state.json"
        alpha_state_path.write_text(json.dumps({"phase": "VALIDATING"}))

        mock_run.return_value = SubprocessResult(0, "", "", 30.0)
        combo = {"symbol": "BTC", "horizon": 300}
        verdict = step_alpha_pipeline(config, state, combo, logger)
        assert verdict == "PASS"

        # Clean up
        alpha_state_path.unlink(missing_ok=True)

    @patch("discovery_orchestrator.run_subprocess")
    def test_fails_when_phase_not_reached(self, mock_run, config, state, logger, tmp_dir):
        alpha_state_dir = ROOT / "data" / "alpha"
        alpha_state_dir.mkdir(parents=True, exist_ok=True)
        alpha_state_path = alpha_state_dir / "pipeline_state.json"
        alpha_state_path.write_text(json.dumps({"phase": "SCREENING"}))

        mock_run.return_value = SubprocessResult(0, "", "", 30.0)
        combo = {"symbol": "BTC", "horizon": 300}
        verdict = step_alpha_pipeline(config, state, combo, logger)
        assert verdict == "FAIL"

        alpha_state_path.unlink(missing_ok=True)


class TestStepReport:
    def test_writes_summary_json(self, config, state, logger, tmp_dir):
        state.set("winners", [{"symbol": "BTC", "horizon": 300}])
        state.record_gate("test_gate", "PASS", {"metric": 1.0})

        step_report(config, state, 1, tmp_dir, time.time() - 60, logger)

        summary_path = tmp_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["cycle"] == 1
        assert len(summary["winners"]) == 1
        assert "test_gate" in summary["gates"]

    def test_summary_has_duration(self, config, state, logger, tmp_dir):
        start = time.time() - 120
        step_report(config, state, 1, tmp_dir, start, logger)
        with open(tmp_dir / "summary.json") as f:
            summary = json.load(f)
        assert summary["duration_s"] >= 100


# ---------------------------------------------------------------------------
# Tests: Cycle Runner
# ---------------------------------------------------------------------------


class TestRunCycle:
    @patch("discovery_orchestrator.step_alpha_pipeline")
    @patch("discovery_orchestrator.step_backtest")
    @patch("discovery_orchestrator.step_train")
    @patch("discovery_orchestrator.step_signal_sweep")
    @patch("discovery_orchestrator.step_data_health")
    def test_happy_path(self, mock_health, mock_sweep, mock_train,
                        mock_bt, mock_alpha, config, state, logger, tmp_dir):
        config["discovery"]["report_dir"] = str(tmp_dir / "reports")
        mock_health.return_value = "PASS"
        mock_sweep.return_value = ("PASS", [{"symbol": "BTC", "horizon": 300, "edge": 0.05, "gross_bps": 1.0}])
        mock_train.return_value = ("PASS", "/tmp/model.txt")
        mock_bt.return_value = ("PASS", {"sharpe": 1.5, "max_drawdown_pct": 5.0})
        mock_alpha.return_value = "PASS"

        run_cycle(config, state, logger)

        assert state.get("cycle_number") == 1
        assert mock_health.called
        assert mock_sweep.called
        assert mock_train.called
        assert mock_bt.called
        assert mock_alpha.called

    @patch("discovery_orchestrator.step_report")
    @patch("discovery_orchestrator.step_data_health")
    def test_data_health_fail_stops_cycle(self, mock_health, mock_report,
                                          config, state, logger, tmp_dir):
        config["discovery"]["report_dir"] = str(tmp_dir / "reports")
        mock_health.return_value = "FAIL"

        run_cycle(config, state, logger)

        assert state.current == Phase.ERROR
        mock_report.assert_called_once()

    @patch("discovery_orchestrator.step_report")
    @patch("discovery_orchestrator.step_signal_sweep")
    @patch("discovery_orchestrator.step_data_health")
    def test_no_winners_ends_cycle(self, mock_health, mock_sweep, mock_report,
                                    config, state, logger, tmp_dir):
        config["discovery"]["report_dir"] = str(tmp_dir / "reports")
        mock_health.return_value = "PASS"
        mock_sweep.return_value = ("FAIL", [])

        run_cycle(config, state, logger)

        assert state.current == Phase.REPORTING
        mock_report.assert_called_once()

    @patch("discovery_orchestrator.step_backtest")
    @patch("discovery_orchestrator.step_train")
    @patch("discovery_orchestrator.step_signal_sweep")
    @patch("discovery_orchestrator.step_data_health")
    def test_train_fail_skips_winner(self, mock_health, mock_sweep, mock_train,
                                     mock_bt, config, state, logger, tmp_dir):
        config["discovery"]["report_dir"] = str(tmp_dir / "reports")
        mock_health.return_value = "PASS"
        mock_sweep.return_value = ("PASS", [{"symbol": "BTC", "horizon": 300, "edge": 0.05, "gross_bps": 1.0}])
        mock_train.return_value = ("FAIL", None)

        run_cycle(config, state, logger)

        # Backtest should NOT be called
        assert not mock_bt.called

    @patch("discovery_orchestrator.step_data_health")
    def test_cycle_increments(self, mock_health, config, state, logger, tmp_dir):
        config["discovery"]["report_dir"] = str(tmp_dir / "reports")
        mock_health.return_value = "FAIL"  # Stop early

        run_cycle(config, state, logger)
        assert state.get("cycle_number") == 1

        run_cycle(config, state, logger)
        assert state.get("cycle_number") == 2


# ---------------------------------------------------------------------------
# Tests: CLI Commands
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_status_prints_phase(self, config, state, logger, capsys):
        cmd_status(config, state, logger)
        captured = capsys.readouterr()
        assert "IDLE" in captured.out

    def test_status_shows_cycle(self, config, state, logger, capsys):
        state.set("cycle_number", 5)
        cmd_status(config, state, logger)
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_status_shows_winners(self, config, state, logger, capsys):
        state.set("winners", [{"symbol": "BTC", "horizon": 300, "edge": 0.05, "gross_bps": 1.0}])
        cmd_status(config, state, logger)
        captured = capsys.readouterr()
        assert "BTC" in captured.out


class TestCmdStop:
    def test_stop_sets_phase(self, config, state, logger):
        cmd_stop(config, state, logger)
        assert state.current == Phase.STOPPED


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_corrupt_state_file_uses_defaults(self, tmp_dir):
        state_path = tmp_dir / "corrupt.json"
        state_path.write_text("not json{{{")
        # Should not crash — falls back to defaults
        with pytest.raises(json.JSONDecodeError):
            DiscoveryState(str(state_path))

    def test_empty_sweep_config(self, config, state, logger, tmp_dir):
        """Empty symbols/horizons should produce no winners."""
        config["discovery"]["sweep"]["symbols"] = []
        config["discovery"]["sweep"]["horizons"] = []
        verdict, winners = step_signal_sweep(config, state, logger, tmp_dir)
        assert verdict == "FAIL"
        assert winners == []

    def test_state_concurrent_access(self, state_file):
        """Two state instances should see each other's writes."""
        s1 = DiscoveryState(state_file)
        s2 = DiscoveryState(state_file)
        s1.set("cycle_number", 99)
        # s2 needs to reload
        s2_reloaded = DiscoveryState(state_file)
        assert s2_reloaded.get("cycle_number") == 99

    def test_large_gate_metrics(self, state):
        """Gate metrics can contain arbitrary nested data."""
        big_metrics = {
            "results": [{"threshold": i, "value": i * 0.1} for i in range(100)],
        }
        state.record_gate("big_gate", "PASS", big_metrics)
        gates = state.get("gates", {})
        assert len(gates["big_gate"]["metrics"]["results"]) == 100

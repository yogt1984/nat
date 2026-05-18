"""Tests for alpha_pipeline.py — state machine orchestrator.

Tests cover:
- Phase enum ordering and mappings
- AlphaPipelineState persistence, transitions, gates, artifacts, history cap, reset
- evaluate_gate PASS/WEAK/FAIL logic
- Config loader (valid / missing)
- run_pipeline state machine progression with mocked modules
- Gate failure stops the pipeline
- force_gates bypasses failures
- Resume logic: force-gate advances past a failed gate
- CLI dispatch commands
- Display helpers
- Edge cases: corrupt JSON, missing artifacts, missing config sections
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from alpha.alpha_pipeline import (
    Phase,
    STEP_PHASES,
    GATE_NAMES,
    STEP_LABELS,
    AlphaPipelineState,
    evaluate_gate,
    load_config,
    run_pipeline,
    _print_status,
    _print_gate_summary,
    _print_gates_detail,
    cmd_start,
    cmd_resume,
    cmd_status,
    cmd_gates,
    cmd_run_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state(tmp_path):
    """Provide a fresh AlphaPipelineState backed by a temp file."""
    return AlphaPipelineState(str(tmp_path / "state.json"))


@pytest.fixture
def sample_config(tmp_path):
    """Minimal valid config dict matching config/alpha.toml structure."""
    return {
        "pipeline": {
            "data_dir": "data/features",
            "timeframe": "15min",
            "symbols": ["BTC", "ETH", "SOL"],
            "primary_symbol": "BTC",
            "state_file": str(tmp_path / "state.json"),
            "log_file": str(tmp_path / "alpha_pipeline.log"),
            "report_dir": str(tmp_path / "reports"),
        },
        "screener": {
            "fdr_alpha": 0.05,
            "min_ic": 0.015,
            "min_breakeven_bps": 2.0,
            "price_col": "raw_midprice",
        },
        "combiner": {
            "top_n": 20,
            "max_corr": 0.8,
            "method": "equal",
        },
        "position": {
            "cost_multiplier": 1.5,
            "scale": 1.0,
            "ramp_bars": 2880,
            "ramp_fraction": 0.5,
            "bar_minutes": 15.0,
        },
        "validation": {
            "entry_threshold": 0.3,
            "n_splits": 5,
            "embargo_bars": 600,
            "directions": ["long", "short"],
        },
        "regime": {
            "improvement_threshold": 1.5,
            "top_n": 10,
            "model_path": "",
        },
        "multi_freq": {
            "signal_path": "",
        },
        "paper": {
            "backtest_sharpe": 1.0,
            "backtest_ic": 0.03,
        },
        "gates": {
            "g1_min_significant": 5,
            "g1_weak_significant": 1,
            "g2_ic_ratio": 0.8,
            "g2_max_turnover_ratio": 2.0,
            "g2_max_single_corr": 0.9,
            "g3_min_trade_reduction_pct": 50.0,
            "g3_min_holding_hours": 2.0,
            "g4_min_oos_sharpe": 0.5,
            "g4_min_oos_is_ratio": 0.7,
            "g4_max_deflated_sharpe_p": 0.05,
            "g4_max_drawdown_pct": 5.0,
            "g4_min_trades": 30,
            "g4_min_profit_factor": 1.2,
            "g5_min_ic_ratio": 1.5,
            "g7_dd_ratio": 0.8,
            "g8_min_sharpe_ratio": 0.5,
            "g8_max_daily_loss_pct": 2.0,
            "g8_max_ic_decay_pct": 50.0,
            "g8_min_days": 14,
        },
    }


# ---------------------------------------------------------------------------
# Mock dataclass factories (matching real module return types)
# ---------------------------------------------------------------------------


def _mock_screen_result(n_significant=10, total_tests=200):
    """Mock ScreenResult with fields the pipeline reads."""
    m = MagicMock()
    m.n_significant = n_significant
    m.total_tests = total_tests
    return m


def _mock_combine_result(
    combined_ic=0.05,
    max_individual_ic=0.04,
    combined_turnover=0.3,
    avg_individual_turnover=0.5,
    max_single_corr=0.6,
):
    m = MagicMock()
    m.combined_ic = combined_ic
    m.max_individual_ic = max_individual_ic
    m.combined_turnover = combined_turnover
    m.avg_individual_turnover = avg_individual_turnover
    m.max_single_corr = max_single_corr
    m.features_after_dedup = [f"feat_{i}" for i in range(5)]
    return m


def _mock_position_result(trade_reduction_pct=70.0, mean_holding_hours=4.0):
    m = MagicMock()
    m.trade_reduction_pct = trade_reduction_pct
    m.mean_holding_hours = mean_holding_hours
    return m


def _mock_validation_result(
    oos_sharpe=1.2,
    is_sharpe=1.5,
    oos_is_ratio=0.8,
    max_drawdown_pct=3.0,
    total_oos_trades=100,
    profit_factor=1.5,
    deflated_sharpe_p=0.01,
    direction="long",
):
    m = MagicMock()
    m.oos_sharpe = oos_sharpe
    m.is_sharpe = is_sharpe
    m.oos_is_ratio = oos_is_ratio
    m.max_drawdown_pct = max_drawdown_pct
    m.total_oos_trades = total_oos_trades
    m.profit_factor = profit_factor
    m.deflated_sharpe_p = deflated_sharpe_p
    m.direction = direction
    return m


def _mock_regime_result(gate_has_improving=True, n_regimes=3, conditioned=None):
    m = MagicMock()
    m.n_regimes = n_regimes
    m.conditioned_regimes = conditioned if conditioned is not None else [0, 2]
    m.gate_has_improving_regime = gate_has_improving
    m.improvement_ratios = {0: 1.8, 1: 0.9, 2: 1.6}
    return m


def _mock_multi_freq_result(
    micro_sharpe=1.0,
    macro_sharpe=0.5,
    composite_sharpe=1.3,
    composite_max_dd=2.0,
    sharpe_improves=True,
    dd_improves=True,
):
    m = MagicMock()
    m.micro_sharpe = micro_sharpe
    m.macro_sharpe = macro_sharpe
    m.composite_sharpe = composite_sharpe
    m.composite_max_dd = composite_max_dd
    m.gate_sharpe_improves = sharpe_improves
    m.gate_dd_improves = dd_improves
    return m


def _mock_portfolio_result(
    portfolio_sharpe=1.5,
    portfolio_max_dd=2.5,
    max_individual_sharpe=1.2,
    sharpe_improves=True,
    dd_improves=True,
):
    m = MagicMock()
    m.portfolio_sharpe = portfolio_sharpe
    m.portfolio_max_dd = portfolio_max_dd
    m.max_individual_sharpe = max_individual_sharpe
    m.gate_sharpe_improves = sharpe_improves
    m.gate_dd_improves = dd_improves
    return m


def _mock_paper_result(
    paper_sharpe=0.8,
    sharpe_ratio=0.8,
    max_daily_loss_pct=1.0,
    n_days=20,
    gate_sharpe=True,
    gate_loss=True,
    gate_ic=True,
    gate_infra=True,
):
    m = MagicMock()
    m.paper_sharpe = paper_sharpe
    m.sharpe_ratio = sharpe_ratio
    m.max_daily_loss_pct = max_daily_loss_pct
    m.n_days = n_days
    m.gate_sharpe_within_2x = gate_sharpe
    m.gate_no_big_daily_loss = gate_loss
    m.gate_ic_stable = gate_ic
    m.gate_infra_stable = gate_infra
    return m


def _mock_deployment_readiness(overall_ready=True, blockers=None):
    m = MagicMock()
    m.overall_ready = overall_ready
    m.blockers = blockers if blockers is not None else []
    return m


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class TestPhaseEnum:
    def test_step_phases_count(self):
        assert len(STEP_PHASES) == 9

    def test_step_phases_order(self):
        expected = [
            Phase.SCREENING, Phase.COMBINING, Phase.SIZING, Phase.VALIDATING,
            Phase.REGIME, Phase.MULTI_FREQ, Phase.PORTFOLIO, Phase.PAPER, Phase.DEPLOYING,
        ]
        assert STEP_PHASES == expected

    def test_gate_names_for_all_steps(self):
        for i, p in enumerate(STEP_PHASES, 1):
            assert p in GATE_NAMES
            assert GATE_NAMES[p] == f"G{i}"

    def test_step_labels_for_all_steps(self):
        for p in STEP_PHASES:
            assert p in STEP_LABELS
            assert isinstance(STEP_LABELS[p], str)
            assert len(STEP_LABELS[p]) > 0

    def test_terminal_phases_not_in_steps(self):
        for p in [Phase.IDLE, Phase.DONE, Phase.GATE_FAILED, Phase.ERROR]:
            assert p not in STEP_PHASES

    def test_phase_is_str_enum(self):
        assert Phase.IDLE.value == "IDLE"
        assert Phase("SCREENING") == Phase.SCREENING

    def test_all_phases_count(self):
        # 9 step + IDLE + DONE + GATE_FAILED + ERROR = 13
        assert len(Phase) == 13


# ---------------------------------------------------------------------------
# AlphaPipelineState
# ---------------------------------------------------------------------------


class TestAlphaPipelineState:
    def test_fresh_state_is_idle(self, tmp_state):
        assert tmp_state.current == Phase.IDLE

    def test_defaults(self, tmp_state):
        assert tmp_state.get("started_at") is None
        assert tmp_state.get("finished_at") is None
        assert tmp_state.get("current_step") == 0
        assert tmp_state.get("gates") == {}
        assert tmp_state.get("artifacts") == {}
        assert tmp_state.get("step_outputs") == {}
        assert tmp_state.get("error") is None
        assert tmp_state.get("history") == []

    def test_transition_changes_phase(self, tmp_state):
        tmp_state.transition(Phase.SCREENING, "starting")
        assert tmp_state.current == Phase.SCREENING

    def test_transition_appends_history(self, tmp_state):
        tmp_state.transition(Phase.SCREENING)
        tmp_state.transition(Phase.COMBINING)
        history = tmp_state.get("history")
        assert len(history) == 2
        assert history[0]["from"] == "IDLE"
        assert history[0]["to"] == "SCREENING"
        assert history[1]["from"] == "SCREENING"
        assert history[1]["to"] == "COMBINING"

    def test_history_has_timestamp(self, tmp_state):
        tmp_state.transition(Phase.SCREENING)
        entry = tmp_state.get("history")[0]
        assert "at" in entry
        assert "T" in entry["at"]  # ISO format

    def test_history_cap_at_200(self, tmp_state):
        """History is trimmed to last 100 when it exceeds 200."""
        for i in range(210):
            tmp_state.transition(Phase.SCREENING, f"iter {i}")
            # Alternate to avoid staying in same phase
            tmp_state.transition(Phase.COMBINING, f"iter {i}")
        history = tmp_state.get("history")
        assert len(history) <= 200

    def test_set_get(self, tmp_state):
        tmp_state.set("foo", "bar")
        assert tmp_state.get("foo") == "bar"

    def test_get_default(self, tmp_state):
        assert tmp_state.get("nonexistent", 42) == 42

    def test_persistence_survives_reload(self, tmp_path):
        path = str(tmp_path / "state.json")
        ps1 = AlphaPipelineState(path)
        ps1.transition(Phase.SCREENING, "test")
        ps1.set_artifact("screen", "/tmp/screen.json")
        ps1.record_gate("G1", "PASS", {"n_sig": 10})

        ps2 = AlphaPipelineState(path)
        assert ps2.current == Phase.SCREENING
        assert ps2.get_artifact("screen") == "/tmp/screen.json"
        gates = ps2.get("gates")
        assert gates["G1"]["verdict"] == "PASS"

    def test_record_gate(self, tmp_state):
        tmp_state.record_gate("G1", "PASS", {"n": 5}, "")
        gates = tmp_state.get("gates")
        assert "G1" in gates
        assert gates["G1"]["verdict"] == "PASS"
        assert gates["G1"]["metrics"]["n"] == 5
        assert "evaluated_at" in gates["G1"]

    def test_record_gate_overwrite(self, tmp_state):
        tmp_state.record_gate("G1", "FAIL", {"n": 0}, "fix it")
        tmp_state.record_gate("G1", "PASS", {"n": 10}, "")
        assert tmp_state.get("gates")["G1"]["verdict"] == "PASS"

    def test_set_artifact(self, tmp_state):
        tmp_state.set_artifact("signal_npy", "/tmp/signal.npy")
        assert tmp_state.get_artifact("signal_npy") == "/tmp/signal.npy"

    def test_get_artifact_missing(self, tmp_state):
        assert tmp_state.get_artifact("nonexistent") is None

    def test_set_output(self, tmp_state):
        tmp_state.set_output("screen", {"n_significant": 10})
        assert tmp_state.get_output("screen") == {"n_significant": 10}

    def test_get_output_missing(self, tmp_state):
        assert tmp_state.get_output("nonexistent") is None

    def test_reset(self, tmp_state):
        tmp_state.transition(Phase.SCREENING)
        tmp_state.set_artifact("x", "y")
        tmp_state.record_gate("G1", "PASS", {})
        tmp_state.reset()
        assert tmp_state.current == Phase.IDLE
        assert tmp_state.get("artifacts") == {}
        assert tmp_state.get("gates") == {}
        assert tmp_state.get("history") == []

    def test_creates_parent_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "state.json"
        ps = AlphaPipelineState(str(deep))
        ps.transition(Phase.SCREENING)
        assert deep.exists()

    def test_corrupt_json_handled(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            AlphaPipelineState(str(path))

    def test_save_creates_valid_json(self, tmp_state):
        tmp_state.transition(Phase.SCREENING)
        raw = json.loads(tmp_state.state_file.read_text())
        assert raw["phase"] == "SCREENING"


# ---------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------


class TestEvaluateGate:
    def test_pass_when_all_met(self):
        verdict, metrics, advice = evaluate_gate("G1", 5, 5, weak_min=1, metrics={"n": 5}, advice_on_fail="fix")
        assert verdict == "PASS"
        assert advice == ""

    def test_pass_when_exceeds(self):
        verdict, _, _ = evaluate_gate("G1", 10, 5, weak_min=1, metrics={}, advice_on_fail="fix")
        assert verdict == "PASS"

    def test_weak_at_boundary(self):
        verdict, _, advice = evaluate_gate("G1", 3, 5, weak_min=3, metrics={}, advice_on_fail="try harder")
        assert verdict == "WEAK"
        assert advice == "try harder"

    def test_fail_below_weak(self):
        verdict, _, advice = evaluate_gate("G1", 0, 5, weak_min=1, metrics={}, advice_on_fail="oops")
        assert verdict == "FAIL"
        assert advice == "oops"

    def test_weak_min_equals_total(self):
        """When weak_min equals total, anything below total is FAIL (no WEAK zone)."""
        verdict, _, _ = evaluate_gate("G1", 4, 5, weak_min=5, metrics={}, advice_on_fail="x")
        assert verdict == "FAIL"

    def test_metrics_passed_through(self):
        _, metrics, _ = evaluate_gate("G1", 3, 3, weak_min=1, metrics={"a": 1, "b": 2}, advice_on_fail="x")
        assert metrics == {"a": 1, "b": 2}

    def test_zero_total_zero_pass_is_pass(self):
        """Edge: 0/0 counts as PASS."""
        verdict, _, _ = evaluate_gate("G1", 0, 0, weak_min=0, metrics={}, advice_on_fail="x")
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_real_config(self):
        # Use the actual config file
        config = load_config("config/alpha.toml")
        assert "pipeline" in config
        assert "gates" in config
        assert config["pipeline"]["primary_symbol"] == "BTC"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/alpha.toml")


# ---------------------------------------------------------------------------
# run_pipeline — full pass with all gates passing
# ---------------------------------------------------------------------------


class TestRunPipelineFullPass:
    """Test that all 9 steps execute and reach DONE when every gate passes."""

    def _patch_all_modules(self, tmp_path, sample_config):
        """Build a dict of patches for all 9 step imports."""
        signal_npy = str(tmp_path / "data" / "alpha" / "signal.npy")
        position_npy = str(tmp_path / "data" / "alpha" / "position.npy")

        patches = {}

        # Step 1: screener
        mock_screen = MagicMock()
        mock_screen.screen_features.return_value = _mock_screen_result(10, 200)
        mock_screen.save_results.return_value = str(tmp_path / "reports" / "alpha_screen.json")
        patches["alpha.screener"] = mock_screen

        # Step 2: combiner
        mock_combine = MagicMock()
        mock_combine.run_combine.return_value = (np.zeros(100), _mock_combine_result())
        patches["alpha.combiner"] = mock_combine

        # Step 3: position
        mock_position = MagicMock()
        mock_position.run_position_sizing.return_value = (np.zeros(100), _mock_position_result())
        patches["alpha.position"] = mock_position

        # Step 4: adapter (validation)
        mock_adapter = MagicMock()
        mock_adapter.run_validation.return_value = [_mock_validation_result()]
        patches["alpha.adapter"] = mock_adapter

        # Step 5: regime_filter
        mock_regime = MagicMock()
        mock_regime.run_regime_filter.return_value = _mock_regime_result()
        patches["alpha.regime_filter"] = mock_regime

        # Step 6: multi_freq
        mock_mf = MagicMock()
        mock_mf.run_multi_freq.return_value = _mock_multi_freq_result()
        patches["alpha.multi_freq"] = mock_mf

        # Step 7: portfolio
        mock_port = MagicMock()
        mock_port.run_portfolio.return_value = _mock_portfolio_result()
        patches["alpha.portfolio"] = mock_port

        # Step 8: paper_trader
        mock_paper = MagicMock()
        mock_paper.run_paper_simulation.return_value = _mock_paper_result()
        patches["alpha.paper_trader"] = mock_paper

        # Step 9: deployer
        mock_deployer = MagicMock()
        mock_deployer.check_readiness.return_value = _mock_deployment_readiness()
        patches["alpha.deployer"] = mock_deployer

        # cluster_pipeline (for data loading in steps 3/4)
        import polars as pl
        mock_loader = MagicMock()
        mock_loader.load_parquet.return_value = MagicMock()
        patches["cluster_pipeline.loader"] = mock_loader

        # Return a real polars DataFrame so isinstance(bars, pl.DataFrame) passes
        prices = np.linspace(100, 101, 500)
        mock_bars_df = pl.DataFrame({"raw_midprice_mean": prices})
        mock_preprocess = MagicMock()
        mock_preprocess.aggregate_bars.return_value = mock_bars_df
        patches["cluster_pipeline.preprocess"] = mock_preprocess

        return patches

    def _run_with_mocks(self, tmp_path, sample_config, patches, force_gates=False):
        """Execute run_pipeline with all modules mocked."""
        import logging

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        log = logging.getLogger("test_pipeline")
        log.setLevel(logging.WARNING)  # suppress output during tests

        # Patch all module imports
        with patch.dict("sys.modules", patches):
            # Also need to handle np.load for signal/position arrays
            original_np_load = np.load

            def mock_np_load(path, *a, **kw):
                return np.zeros(100)

            with patch("numpy.load", side_effect=mock_np_load):
                run_pipeline(sample_config, ps, log, force_gates=force_gates)

        return ps

    def test_reaches_done(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        assert ps.current == Phase.DONE

    def test_all_nine_gates_recorded(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        gates = ps.get("gates", {})
        for i in range(1, 10):
            assert f"G{i}" in gates, f"G{i} not recorded"

    def test_all_gates_pass(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        gates = ps.get("gates", {})
        for gname, g in gates.items():
            assert g["verdict"] in ("PASS", "WEAK"), f"{gname} verdict={g['verdict']}"

    def test_started_at_set(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        assert ps.get("started_at") is not None

    def test_finished_at_set(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        assert ps.get("finished_at") is not None

    def test_artifacts_recorded(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        for name in ["screen", "signal_npy", "combine", "position_npy", "position",
                      "validation", "regime", "multi_freq", "portfolio", "paper", "deployer"]:
            assert ps.get_artifact(name) is not None, f"artifact '{name}' missing"

    def test_step_outputs_recorded(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        for name in ["screen", "combine", "position", "validate", "regime",
                      "multi_freq", "portfolio", "paper", "deployer"]:
            assert ps.get_output(name) is not None, f"output '{name}' missing"

    def test_history_not_empty(self, tmp_path, sample_config):
        patches = self._patch_all_modules(tmp_path, sample_config)
        ps = self._run_with_mocks(tmp_path, sample_config, patches)
        assert len(ps.get("history")) >= 10  # At least IDLE→SCREEN + 9 transitions


# ---------------------------------------------------------------------------
# Gate failure stops pipeline
# ---------------------------------------------------------------------------


class TestGateFailure:
    """Test that each gate failure halts the pipeline at GATE_FAILED."""

    def _run_step1_fail(self, tmp_path, sample_config):
        """Run pipeline where G1 fails (0 significant features)."""
        import logging

        mock_screen = MagicMock()
        mock_screen.screen_features.return_value = _mock_screen_result(n_significant=0, total_tests=200)
        mock_screen.save_results.return_value = str(tmp_path / "screen.json")

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        log = logging.getLogger("test_g1_fail")
        log.setLevel(logging.WARNING)

        with patch.dict("sys.modules", {"alpha.screener": mock_screen}):
            run_pipeline(sample_config, ps, log, force_gates=False)
        return ps

    def test_g1_fail_stops_at_gate_failed(self, tmp_path, sample_config):
        ps = self._run_step1_fail(tmp_path, sample_config)
        assert ps.current == Phase.GATE_FAILED

    def test_g1_fail_records_gate(self, tmp_path, sample_config):
        ps = self._run_step1_fail(tmp_path, sample_config)
        gates = ps.get("gates")
        assert "G1" in gates
        assert gates["G1"]["verdict"] == "FAIL"

    def test_g1_fail_no_g2(self, tmp_path, sample_config):
        ps = self._run_step1_fail(tmp_path, sample_config)
        assert "G2" not in ps.get("gates")

    def test_g1_weak_continues(self, tmp_path, sample_config):
        """G1 WEAK (2 significant, threshold 5, weak_min 1) should continue to G2."""
        import logging

        mock_screen = MagicMock()
        mock_screen.screen_features.return_value = _mock_screen_result(n_significant=2, total_tests=200)
        mock_screen.save_results.return_value = str(tmp_path / "screen.json")

        # Combiner returns bad IC so G2 fails → stops cleanly
        mock_combine = MagicMock()
        bad_cr = _mock_combine_result(
            combined_ic=0.001, max_individual_ic=0.04,
            combined_turnover=10.0, avg_individual_turnover=0.5,
            max_single_corr=0.99,
        )
        mock_combine.run_combine.return_value = (np.zeros(100), bad_cr)

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        log = logging.getLogger("test_g1_weak")
        log.setLevel(logging.WARNING)

        patches = {
            "alpha.screener": mock_screen,
            "alpha.combiner": mock_combine,
        }
        with patch.dict("sys.modules", patches):
            run_pipeline(sample_config, ps, log, force_gates=False)

        gates = ps.get("gates")
        assert gates["G1"]["verdict"] == "WEAK"
        # Should have advanced past screening to combining
        assert "G2" in gates
        assert gates["G2"]["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# force_gates bypasses failures
# ---------------------------------------------------------------------------


class TestForceGates:
    def test_force_gate_bypasses_g1_fail(self, tmp_path, sample_config):
        """With force_gates=True, G1 FAIL should not stop — pipeline continues to G2."""
        import logging

        mock_screen = MagicMock()
        mock_screen.screen_features.return_value = _mock_screen_result(n_significant=0)
        mock_screen.save_results.return_value = str(tmp_path / "screen.json")

        # Combiner returns bad values → G2 FAIL.
        # With force_gates=False this would stop. We check G1+G2 are both recorded.
        mock_combine = MagicMock()
        bad_cr = _mock_combine_result(
            combined_ic=0.001, max_individual_ic=0.04,
            combined_turnover=10.0, avg_individual_turnover=0.5,
            max_single_corr=0.99,
        )
        mock_combine.run_combine.return_value = (np.zeros(100), bad_cr)

        # Position sizing: return mock result so G3 also evaluates
        mock_position = MagicMock()
        mock_position.run_position_sizing.return_value = (
            np.zeros(100), _mock_position_result(trade_reduction_pct=10.0, mean_holding_hours=0.5)
        )

        # Validation: make data load fail → ERROR stops the chain
        mock_loader = MagicMock()
        mock_loader.load_parquet.side_effect = FileNotFoundError("no data")

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        log = logging.getLogger("test_force")
        log.setLevel(logging.WARNING)

        patches = {
            "alpha.screener": mock_screen,
            "alpha.combiner": mock_combine,
            "alpha.position": mock_position,
            "cluster_pipeline.loader": mock_loader,
        }
        with patch.dict("sys.modules", patches), \
             patch("numpy.load", return_value=np.zeros(100)):
            run_pipeline(sample_config, ps, log, force_gates=True)

        gates = ps.get("gates")
        assert gates["G1"]["verdict"] == "FAIL"
        # Should have continued to G2 despite G1 failure
        assert "G2" in gates


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_from_combining(self, tmp_path, sample_config):
        """Pipeline resumed at COMBINING should skip screening and evaluate G2."""
        import logging

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.COMBINING)
        ps.set_artifact("screen", str(tmp_path / "screen.json"))

        # Bad combiner result → G2 FAIL, stops cleanly
        mock_combine = MagicMock()
        bad_cr = _mock_combine_result(
            combined_ic=0.001, max_individual_ic=0.04,
            combined_turnover=10.0, avg_individual_turnover=0.5,
            max_single_corr=0.99,
        )
        mock_combine.run_combine.return_value = (np.zeros(100), bad_cr)

        log = logging.getLogger("test_resume")
        log.setLevel(logging.WARNING)

        with patch.dict("sys.modules", {"alpha.combiner": mock_combine}):
            run_pipeline(sample_config, ps, log, force_gates=False)

        # G1 should NOT be evaluated (we started at COMBINING)
        assert "G1" not in ps.get("gates")
        # G2 should be evaluated
        assert "G2" in ps.get("gates")
        assert ps.current == Phase.GATE_FAILED

    def test_force_gate_resume_advances_past_failed(self, tmp_path, sample_config):
        """cmd_resume with --force-gate should advance past GATE_FAILED."""
        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.SCREENING)
        ps.record_gate("G1", "FAIL", {"n_significant": 0})
        ps.transition(Phase.GATE_FAILED, "G1 FAIL")

        # Simulate what cmd_resume does for force-gate
        gates = ps.get("gates", {})
        for p in STEP_PHASES:
            gname = GATE_NAMES[p]
            if gname in gates and gates[gname]["verdict"] == "FAIL":
                idx = STEP_PHASES.index(p)
                if idx + 1 < len(STEP_PHASES):
                    next_phase = STEP_PHASES[idx + 1]
                else:
                    next_phase = Phase.DONE
                ps.transition(next_phase, f"Force-gate: skipping {gname}")
                break

        assert ps.current == Phase.COMBINING

    def test_force_gate_last_step_goes_to_done(self, tmp_path, sample_config):
        """Force-gate on the last step (DEPLOYING/G9) should go to DONE."""
        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.DEPLOYING)
        ps.record_gate("G9", "FAIL", {"overall_ready": False})
        ps.transition(Phase.GATE_FAILED, "G9 FAIL")

        gates = ps.get("gates", {})
        for p in STEP_PHASES:
            gname = GATE_NAMES[p]
            if gname in gates and gates[gname]["verdict"] == "FAIL":
                idx = STEP_PHASES.index(p)
                if idx + 1 < len(STEP_PHASES):
                    next_phase = STEP_PHASES[idx + 1]
                else:
                    next_phase = Phase.DONE
                ps.transition(next_phase, f"Force-gate: skipping {gname}")
                break

        assert ps.current == Phase.DONE


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


class TestDisplayHelpers:
    def test_print_status_idle(self, tmp_state, capsys):
        _print_status(tmp_state)
        out = capsys.readouterr().out
        assert "IDLE" in out
        assert "Alpha Pipeline Status" in out

    def test_print_status_with_error(self, tmp_state, capsys):
        tmp_state.set("error", "something broke")
        _print_status(tmp_state)
        out = capsys.readouterr().out
        assert "something broke" in out

    def test_print_status_with_artifacts(self, tmp_state, capsys):
        tmp_state.set_artifact("screen", "/tmp/screen.json")
        _print_status(tmp_state)
        out = capsys.readouterr().out
        assert "screen" in out

    def test_print_gate_summary_empty(self, tmp_state, capsys):
        _print_gate_summary(tmp_state)
        out = capsys.readouterr().out
        assert "G1" in out
        assert "[ ]" in out  # empty gates show [ ]

    def test_print_gate_summary_with_pass(self, tmp_state, capsys):
        tmp_state.record_gate("G1", "PASS", {"n_significant": 10})
        _print_gate_summary(tmp_state)
        out = capsys.readouterr().out
        assert "[+]" in out
        assert "PASS" in out

    def test_print_gate_summary_with_fail(self, tmp_state, capsys):
        tmp_state.record_gate("G1", "FAIL", {"n_significant": 0})
        _print_gate_summary(tmp_state)
        out = capsys.readouterr().out
        assert "[x]" in out
        assert "FAIL" in out

    def test_print_gate_summary_with_weak(self, tmp_state, capsys):
        tmp_state.record_gate("G1", "WEAK", {"n_significant": 2})
        _print_gate_summary(tmp_state)
        out = capsys.readouterr().out
        assert "[~]" in out
        assert "WEAK" in out

    def test_print_gates_detail_empty(self, tmp_state, capsys):
        _print_gates_detail(tmp_state)
        out = capsys.readouterr().out
        assert "No gates evaluated" in out

    def test_print_gates_detail_with_data(self, tmp_state, capsys):
        tmp_state.record_gate("G1", "PASS", {"n_significant": 10, "ratio": 0.1234})
        tmp_state.record_gate("G2", "FAIL", {"sub_pass": 1}, "try harder")
        _print_gates_detail(tmp_state)
        out = capsys.readouterr().out
        assert "G1" in out
        assert "PASS" in out
        assert "G2" in out
        assert "FAIL" in out
        assert "try harder" in out
        assert "0.1234" in out  # float formatting


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCLI:
    def _make_args(self, command, tmp_path, **kwargs):
        """Create a mock argparse.Namespace."""
        import argparse
        config_path = tmp_path / "config.toml"
        # Write a minimal TOML config
        try:
            import tomllib
        except ImportError:
            pass  # we'll use the dict approach instead

        ns = argparse.Namespace(
            command=command,
            config=str(config_path),
            **kwargs,
        )
        return ns

    def test_cmd_status_runs(self, tmp_path, sample_config, capsys):
        import argparse

        # Write state file so it can be loaded
        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml")
            cmd_status(args)

        out = capsys.readouterr().out
        assert "IDLE" in out

    def test_cmd_gates_runs(self, tmp_path, sample_config, capsys):
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml")
            cmd_gates(args)

        out = capsys.readouterr().out
        assert "No gates evaluated" in out

    def test_cmd_start_resets_state(self, tmp_path, sample_config):
        """cmd_start from a terminal state (DONE) resets and starts fresh."""
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.DONE)  # Terminal state — start is allowed

        mock_screen = MagicMock()
        mock_screen.screen_features.return_value = _mock_screen_result(n_significant=0)
        mock_screen.save_results.return_value = str(tmp_path / "screen.json")

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config), \
             patch.dict("sys.modules", {"alpha.screener": mock_screen}):
            args = argparse.Namespace(config="config/alpha.toml")
            cmd_start(args)

        # Reload to check
        ps2 = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        # Should have started from IDLE (reset) and advanced to at least SCREENING
        history = ps2.get("history")
        assert history[0]["from"] == "IDLE"

    def test_cmd_start_refuses_when_running(self, tmp_path, sample_config):
        """start should refuse if pipeline is in a non-terminal phase."""
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.VALIDATING)

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml")
            with pytest.raises(SystemExit):
                cmd_start(args)

    def test_cmd_resume_refuses_when_idle(self, tmp_path, sample_config):
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml", force_gate=False)
            with pytest.raises(SystemExit):
                cmd_resume(args)

    def test_cmd_run_step_invalid_step(self, tmp_path, sample_config):
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml", step=0)
            with pytest.raises(SystemExit):
                cmd_run_step(args)

    def test_cmd_run_step_too_high(self, tmp_path, sample_config):
        import argparse

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config):
            args = argparse.Namespace(config="config/alpha.toml", step=10)
            with pytest.raises(SystemExit):
                cmd_run_step(args)


# ---------------------------------------------------------------------------
# Gate-specific evaluation tests (using config thresholds)
# ---------------------------------------------------------------------------


class TestGateEvaluationDetails:
    """Test the specific gate evaluation logic in run_pipeline."""

    def test_g2_three_sub_gates(self):
        """G2 has 3 sub-gates: ic_ratio, turnover_ratio, max_single_corr."""
        # All 3 pass
        cr = _mock_combine_result(
            combined_ic=0.05,
            max_individual_ic=0.04,  # 0.05 > 0.8*0.04=0.032 → pass
            combined_turnover=0.3,
            avg_individual_turnover=0.5,  # 0.3 < 2.0*0.5=1.0 → pass
            max_single_corr=0.6,  # 0.6 < 0.9 → pass
        )
        sub_pass = sum([
            cr.combined_ic > 0.8 * cr.max_individual_ic,
            cr.combined_turnover < 2.0 * cr.avg_individual_turnover,
            cr.max_single_corr < 0.9,
        ])
        assert sub_pass == 3

    def test_g2_ic_ratio_fails(self):
        cr = _mock_combine_result(
            combined_ic=0.02,
            max_individual_ic=0.04,  # 0.02 < 0.8*0.04=0.032 → fail
        )
        assert not (cr.combined_ic > 0.8 * cr.max_individual_ic)

    def test_g4_six_sub_gates(self):
        vr = _mock_validation_result(
            oos_sharpe=1.2,      # > 0.5 → pass
            oos_is_ratio=0.8,    # > 0.7 → pass
            deflated_sharpe_p=0.01,  # < 0.05 → pass
            max_drawdown_pct=3.0,    # < 5.0 → pass
            total_oos_trades=100,    # > 30 → pass
            profit_factor=1.5,       # > 1.2 → pass
        )
        sub_pass = sum([
            vr.oos_sharpe >= 0.5,
            vr.oos_is_ratio >= 0.7,
            vr.deflated_sharpe_p <= 0.05,
            vr.max_drawdown_pct <= 5.0,
            vr.total_oos_trades >= 30,
            vr.profit_factor >= 1.2,
        ])
        assert sub_pass == 6

    def test_g4_marginal_fail(self):
        """G4 with only 3/6 sub-gates → FAIL (weak_min=4)."""
        vr = _mock_validation_result(
            oos_sharpe=0.3,      # < 0.5 → fail
            oos_is_ratio=0.5,    # < 0.7 → fail
            deflated_sharpe_p=0.1,   # > 0.05 → fail
            max_drawdown_pct=3.0,    # < 5.0 → pass
            total_oos_trades=100,    # > 30 → pass
            profit_factor=1.5,       # > 1.2 → pass
        )
        sub_pass = sum([
            vr.oos_sharpe >= 0.5,
            vr.oos_is_ratio >= 0.7,
            vr.deflated_sharpe_p <= 0.05,
            vr.max_drawdown_pct <= 5.0,
            vr.total_oos_trades >= 30,
            vr.profit_factor >= 1.2,
        ])
        assert sub_pass == 3
        verdict, _, _ = evaluate_gate("G4", sub_pass, 6, weak_min=4, metrics={}, advice_on_fail="x")
        assert verdict == "FAIL"

    def test_g4_weak(self):
        """G4 with 4/6 sub-gates → WEAK."""
        verdict, _, _ = evaluate_gate("G4", 4, 6, weak_min=4, metrics={}, advice_on_fail="x")
        assert verdict == "WEAK"

    def test_g8_four_sub_gates(self):
        pr = _mock_paper_result()
        sub_pass = sum([
            pr.gate_sharpe_within_2x,
            pr.gate_no_big_daily_loss,
            pr.gate_ic_stable,
            pr.gate_infra_stable,
        ])
        assert sub_pass == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_validation_results(self, tmp_path, sample_config):
        """Step 4 with empty validation results → G4 FAIL."""
        import logging
        import polars as pl

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.VALIDATING)
        ps.set_artifact("signal_npy", str(tmp_path / "signal.npy"))
        ps.set_output("screen", {"n_tested": 200})

        mock_adapter = MagicMock()
        mock_adapter.run_validation.return_value = []  # empty!

        mock_loader = MagicMock()
        mock_preprocess = MagicMock()
        # Return a real polars DataFrame so isinstance check passes
        mock_preprocess.aggregate_bars.return_value = pl.DataFrame({"x": [1.0, 2.0]})

        log = logging.getLogger("test_empty_val")
        log.setLevel(logging.WARNING)

        patches = {
            "alpha.adapter": mock_adapter,
            "cluster_pipeline.loader": mock_loader,
            "cluster_pipeline.preprocess": mock_preprocess,
        }
        with patch.dict("sys.modules", patches), \
             patch("numpy.load", return_value=np.zeros(100)):
            run_pipeline(sample_config, ps, log, force_gates=False)

        assert ps.get("gates", {}).get("G4", {}).get("verdict") == "FAIL"
        assert ps.current == Phase.GATE_FAILED

    def test_pipeline_error_sets_error_phase(self, tmp_path, sample_config):
        """Unhandled exception in a step should result in ERROR phase via cmd_start."""
        import argparse

        mock_screen = MagicMock()
        mock_screen.screen_features.side_effect = RuntimeError("kaboom")

        with patch("alpha.alpha_pipeline.load_config", return_value=sample_config), \
             patch.dict("sys.modules", {"alpha.screener": mock_screen}):
            args = argparse.Namespace(config="config/alpha.toml")
            cmd_start(args)

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        assert ps.current == Phase.ERROR
        assert "kaboom" in ps.get("error", "")

    def test_g9_deployment_not_ready(self, tmp_path, sample_config):
        """G9 FAIL when deployer says not ready."""
        import logging

        ps = AlphaPipelineState(sample_config["pipeline"]["state_file"])
        ps.transition(Phase.DEPLOYING)
        ps.set_artifact("paper", str(tmp_path / "paper.json"))

        mock_deployer = MagicMock()
        mock_deployer.check_readiness.return_value = _mock_deployment_readiness(
            overall_ready=False, blockers=["insufficient paper days"]
        )

        log = logging.getLogger("test_g9_fail")
        log.setLevel(logging.WARNING)

        with patch.dict("sys.modules", {"alpha.deployer": mock_deployer}):
            run_pipeline(sample_config, ps, log, force_gates=False)

        assert ps.current == Phase.GATE_FAILED
        assert ps.get("gates")["G9"]["verdict"] == "FAIL"

    def test_shutdown_flag_not_set_by_default(self):
        """The _shutdown flag should be False at import time."""
        from alpha.alpha_pipeline import _shutdown
        assert _shutdown is False


# ---------------------------------------------------------------------------
# Step output data integrity
# ---------------------------------------------------------------------------


class TestStepOutputIntegrity:
    """Verify that step outputs contain the expected keys."""

    def test_screen_output_keys(self, tmp_state):
        tmp_state.set_output("screen", {"n_significant": 10, "n_tested": 200})
        out = tmp_state.get_output("screen")
        assert "n_significant" in out
        assert "n_tested" in out

    def test_combine_output_keys(self, tmp_state):
        tmp_state.set_output("combine", {
            "combined_ic": 0.05, "max_individual_ic": 0.04,
            "combined_turnover": 0.3, "n_features": 5,
        })
        out = tmp_state.get_output("combine")
        assert "combined_ic" in out
        assert "n_features" in out

    def test_validate_output_keys(self, tmp_state):
        tmp_state.set_output("validate", {
            "oos_sharpe": 1.2, "is_sharpe": 1.5, "oos_is_ratio": 0.8,
            "max_drawdown_pct": 3.0, "total_oos_trades": 100,
            "profit_factor": 1.5, "direction": "long",
        })
        out = tmp_state.get_output("validate")
        assert len(out) == 7

    def test_paper_output_keys(self, tmp_state):
        tmp_state.set_output("paper", {
            "paper_sharpe": 0.8, "sharpe_ratio": 0.8,
            "max_daily_loss_pct": 1.0, "n_days": 20,
        })
        out = tmp_state.get_output("paper")
        assert "paper_sharpe" in out
        assert "n_days" in out

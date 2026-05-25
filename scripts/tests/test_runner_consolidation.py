"""Skeptical tests for P1-3: Runner consolidation.

Verifies that moving gate methods from 3 runner subclasses into BaseRunner
preserves identical behavior. Tests focus on:
- Gate method correctness with mocked external calls
- Subclass isolation (features, horizons, registries, timeframes)
- Timeframe propagation in temporal replication
- dIC exclusion in IC extraction
- 5-gate vs 4-gate protocol structure
- Registry write correctness per runner type
- Backward compatibility of imports and aliases
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent.base import BaseRunner
from agent.hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_runner(cls, claim="test claim", thresholds=None, manifest=None):
    """Helper: create a runner with sensible defaults."""
    h = Hypothesis.create(
        claim, "gen", ["cmd --data data/features/2026-05-12 --symbol BTC"], 1.0,
        thresholds=thresholds or {"min_ic": 0.10},
    )
    return cls(h, manifest or {"dates": {"2026-05-12": {}, "2026-05-13": {}}})


def _ok_result(stdout="ok"):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail_result():
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")


def _good_report(ic=0.15, n_rows=1000):
    return {"baseline_ic_filt_5s": ic, "n_rows": n_rows}


# ===========================================================================
# Gate method correctness
# ===========================================================================

class TestRunDiscovery:
    """Discovery gate: run test protocol, check IC/dIC, pass/fail."""

    @patch("agent.runner.run_nat_cached")
    def test_passes_on_good_report(self, mock_run):
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_ok_result(), _good_report(ic=0.20))
        runner = _make_runner(MicrostructureRunner)
        assert runner.run_discovery() is True
        assert runner.h.status == "passed"

    @patch("agent.runner.run_nat_cached")
    def test_fails_on_bad_returncode(self, mock_run):
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_fail_result(), None)
        runner = _make_runner(MicrostructureRunner)
        assert runner.run_discovery() is False
        assert runner.h.failure_reason == "command_error"

    @patch("agent.runner.run_nat_cached")
    def test_fails_on_low_ic(self, mock_run):
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_ok_result(), _good_report(ic=0.01))
        runner = _make_runner(MicrostructureRunner, thresholds={"min_ic": 0.10})
        assert runner.run_discovery() is False
        assert runner.h.failure_reason == "no_effect"

    @patch("agent.runner.run_nat_cached")
    def test_bar_runner_tries_parse_report_fallback(self, mock_run):
        """MF/macro runners try parse_report with timeframe on cache miss."""
        from agent.mf_runner import MediumFrequencyRunner
        # First call: run_nat_cached returns no report
        mock_run.return_value = (_ok_result(), None)
        with patch("agent.runner.parse_report", return_value=_good_report(0.20)) as mock_parse:
            runner = _make_runner(MediumFrequencyRunner)
            result = runner.run_discovery()
            # parse_report should have been called with timeframe="5min"
            mock_parse.assert_called_once()
            assert mock_parse.call_args[1]["timeframe"] == "5min"
            assert result is True

    @patch("agent.runner.run_nat_cached")
    def test_tick_runner_skips_parse_report_fallback(self, mock_run):
        """Microstructure runner (TIMEFRAME=None) does NOT try parse_report fallback."""
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_ok_result(), None)
        with patch("agent.runner.parse_report") as mock_parse:
            runner = _make_runner(MicrostructureRunner)
            # No report and no fallback → discovery passes with no gate results
            runner.run_discovery()
            mock_parse.assert_not_called()


class TestTemporalReplication:
    """Temporal replication: re-run on other dates, check pass threshold."""

    @patch("agent.runner.run_nat_cached")
    def test_passes_with_enough_dates(self, mock_run):
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.return_value = (_ok_result(), None)
        runner = _make_runner(MediumFrequencyRunner,
                              manifest={"dates": {"d1": {}, "d2": {}, "d3": {}}})
        assert runner.run_replication_temporal() is True

    @patch("agent.runner.run_nat_cached")
    def test_fails_when_commands_fail(self, mock_run):
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.return_value = (_fail_result(), None)
        runner = _make_runner(MediumFrequencyRunner,
                              manifest={"dates": {"d1": {}, "d2": {}, "d3": {}}})
        assert runner.run_replication_temporal() is False

    @patch("agent.runner.run_nat_cached")
    def test_skips_with_single_date(self, mock_run):
        from agent.mf_runner import MediumFrequencyRunner
        runner = _make_runner(MediumFrequencyRunner,
                              manifest={"dates": {"d1": {}}})
        assert runner.run_replication_temporal() is True
        mock_run.assert_not_called()

    @patch("agent.runner.run_nat_cached")
    def test_appends_timeframe_for_bar_runners(self, mock_run):
        """MF runner appends --timeframe 5min to replication commands."""
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.return_value = (_ok_result(), None)
        runner = _make_runner(MediumFrequencyRunner,
                              manifest={"dates": {"d1": {}, "d2": {}}})
        runner.run_replication_temporal()
        called_parts = mock_run.call_args[0][0]
        assert "--timeframe" in called_parts
        assert "5min" in called_parts

    @patch("agent.runner.run_nat_cached")
    def test_no_timeframe_for_tick_runners(self, mock_run):
        """Micro runner (TIMEFRAME=None) does NOT append --timeframe."""
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_ok_result(), None)
        runner = _make_runner(MicrostructureRunner,
                              manifest={"dates": {"d1": {}, "d2": {}}})
        runner.run_replication_temporal()
        called_parts = mock_run.call_args[0][0]
        assert "--timeframe" not in called_parts

    @patch("agent.runner.run_nat_cached")
    def test_micro_default_min_oos_dates_is_1(self, mock_run):
        """Tick-level runners default to min_oos_dates=1 (easier to replicate)."""
        from agent.runner import MicrostructureRunner
        mock_run.return_value = (_ok_result(), None)
        runner = _make_runner(MicrostructureRunner,
                              manifest={"dates": {"d1": {}, "d2": {}, "d3": {}}})
        # 1 of 2 dates passes → should pass for micro (default min=1)
        mock_run.side_effect = [(_ok_result(), None), (_fail_result(), None)]
        assert runner.run_replication_temporal() is True

    @patch("agent.runner.run_nat_cached")
    def test_bar_default_min_oos_dates_is_2(self, mock_run):
        """Bar-resampled runners default to min_oos_dates=2 (stricter)."""
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.side_effect = [(_ok_result(), None), (_fail_result(), None)]
        runner = _make_runner(MediumFrequencyRunner,
                              manifest={"dates": {"d1": {}, "d2": {}, "d3": {}}})
        # 1 of 2 passes → fails for MF (default min=2)
        assert runner.run_replication_temporal() is False


class TestSymbolReplication:
    @patch("agent.runner.run_nat_cached")
    def test_passes_when_symbols_pass_gates(self, mock_run):
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.return_value = (_ok_result(), _good_report(0.20))
        runner = _make_runner(MediumFrequencyRunner)
        assert runner.run_replication_symbol() is True
        assert "symbol_replication" in runner.h.results

    @patch("agent.runner.run_nat_cached")
    def test_fails_when_symbols_fail(self, mock_run):
        from agent.mf_runner import MediumFrequencyRunner
        mock_run.return_value = (_fail_result(), None)
        runner = _make_runner(MediumFrequencyRunner)
        assert runner.run_replication_symbol() is False

    @patch("agent.runner.run_nat_cached")
    def test_stores_passed_and_failed_symbols(self, mock_run):
        from agent.runner import MicrostructureRunner
        # ETH passes, SOL fails
        mock_run.side_effect = [
            (_ok_result(), _good_report(0.20)),
            (_fail_result(), None),
        ]
        runner = _make_runner(MicrostructureRunner)
        runner.run_replication_symbol()
        sr = runner.h.results["symbol_replication"]
        assert "BTC" in sr["passed"]  # primary always in passed
        assert sr["n_total"] == 2


class TestCorrelationCheck:
    def test_passes_with_empty_registry(self):
        from agent.mf_runner import MediumFrequencyRunner
        orig = MediumFrequencyRunner.REGISTRY_PATH
        MediumFrequencyRunner.REGISTRY_PATH = Path("/nonexistent")
        runner = _make_runner(MediumFrequencyRunner)
        assert runner.run_correlation_check() is True
        MediumFrequencyRunner.REGISTRY_PATH = orig

    def test_passes_with_no_registry_file(self, tmp_path):
        from agent.macro_runner import MacroRunner
        orig = MacroRunner.REGISTRY_PATH
        MacroRunner.REGISTRY_PATH = tmp_path / "no_such_file.json"
        runner = _make_runner(MacroRunner)
        assert runner.run_correlation_check() is True
        MacroRunner.REGISTRY_PATH = orig


# ===========================================================================
# Helper method correctness
# ===========================================================================

class TestCheckGates:
    @patch("agent.runner.check_ic_gate", return_value=(True, "IC=0.20 PASS"))
    @patch("agent.runner.check_dIC_gate", return_value=(True, "dIC skipped"))
    def test_passes_when_both_pass(self, mock_dIC, mock_ic):
        from agent.mf_runner import MediumFrequencyRunner
        runner = _make_runner(MediumFrequencyRunner)
        passed, msg = runner._check_gates({"some": "report"})
        assert passed is True
        assert "IC=0.20 PASS" in msg

    @patch("agent.runner.check_ic_gate", return_value=(False, "IC=0.01 FAIL"))
    @patch("agent.runner.check_dIC_gate", return_value=(True, "dIC skipped"))
    def test_fails_when_ic_fails(self, mock_dIC, mock_ic):
        from agent.mf_runner import MediumFrequencyRunner
        runner = _make_runner(MediumFrequencyRunner)
        passed, msg = runner._check_gates({"some": "report"})
        assert passed is False
        assert "FAIL" in msg


class TestExtractFeatures:
    def test_micro_matches_microstructure_features(self):
        from agent.runner import MicrostructureRunner
        h = Hypothesis.create(
            "imbalance_qty_l5 gated by ent_book_shape<P40",
            "gen", ["cmd"], 1.0)
        runner = MicrostructureRunner(h, {})
        feats = runner._extract_features()
        assert "imbalance_qty_l5" in feats
        assert "ent_book_shape" in feats

    def test_mf_matches_mf_features(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create(
            "trend_momentum_300 predicts 5min returns",
            "gen", ["cmd"], 1.0)
        runner = MediumFrequencyRunner(h, {})
        assert runner._extract_features() == ["trend_momentum_300"]

    def test_macro_matches_macro_features(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create(
            "ctx_funding_zscore meanrev predicts 1h returns",
            "gen", ["cmd"], 1.0)
        runner = MacroRunner(h, {})
        assert runner._extract_features() == ["ctx_funding_zscore"]

    def test_micro_default_feature(self):
        from agent.runner import MicrostructureRunner
        h = Hypothesis.create("unknown claim", "gen", ["cmd"], 1.0)
        runner = MicrostructureRunner(h, {})
        assert runner._extract_features() == ["imbalance_qty_l1"]

    def test_mf_default_feature(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("unknown claim", "gen", ["cmd"], 1.0)
        runner = MediumFrequencyRunner(h, {})
        assert runner._extract_features() == ["trend_momentum_300"]

    def test_macro_default_feature(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create("unknown claim", "gen", ["cmd"], 1.0)
        runner = MacroRunner(h, {})
        assert runner._extract_features() == ["ctx_funding_zscore"]

    def test_no_cross_contamination(self):
        """MF features must NOT match in micro runner and vice versa."""
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        # MF feature in micro runner should not match
        h = Hypothesis.create("trend_momentum_300 is great", "gen", ["cmd"], 1.0)
        micro_runner = MicrostructureRunner(h, {})
        assert micro_runner._extract_features() == ["imbalance_qty_l1"]  # default, not MF
        # Micro feature in MF runner should not match
        h2 = Hypothesis.create("imbalance_qty_l1 is great", "gen", ["cmd"], 1.0)
        mf_runner = MediumFrequencyRunner(h2, {})
        # imbalance_qty_l5 is in MF features but imbalance_qty_l1 is not
        assert "imbalance_qty_l1" not in mf_runner.SIGNAL_FEATURES


class TestExtractIC:
    def test_extracts_ic_from_gate_results(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        h.results = {"gate_results": [
            {"msg": "IC=0.150 [aggregate] vs min=0.10 p=1.2e-03 PASS"},
        ]}
        runner = MediumFrequencyRunner(h, {})
        assert abs(runner._extract_ic_from_results() - 0.150) < 1e-6

    def test_excludes_dIC_messages(self):
        """Must not confuse dIC=0.050 with IC=0.050."""
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        h.results = {"gate_results": [
            {"msg": "IC=0.120 PASS"},
            {"msg": "dIC=+0.050 PASS"},
        ]}
        runner = MacroRunner(h, {})
        assert abs(runner._extract_ic_from_results() - 0.120) < 1e-6

    def test_returns_zero_when_no_results(self):
        from agent.runner import MicrostructureRunner
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        runner = MicrostructureRunner(h, {})
        assert runner._extract_ic_from_results() == 0.0

    def test_dIC_only_message_returns_zero(self):
        """If only dIC messages exist (no IC=), return 0.0."""
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        h.results = {"gate_results": [{"msg": "dIC=+0.050 PASS"}]}
        runner = MediumFrequencyRunner(h, {})
        assert runner._extract_ic_from_results() == 0.0


# ===========================================================================
# Subclass isolation — runners must be independent
# ===========================================================================

class TestRunnerIsolation:
    def test_micro_has_5_gates_mf_macro_have_4(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        h = Hypothesis.create("test", "gen",
                              ["cmd --data d --symbol BTC"], 1.0)
        assert len(MicrostructureRunner(h, {}).steps()) == 5
        assert len(MediumFrequencyRunner(h, {}).steps()) == 4
        assert len(MacroRunner(h, {}).steps()) == 4

    def test_micro_extra_gate_is_cost_check(self):
        from agent.runner import MicrostructureRunner
        h = Hypothesis.create("test", "gen",
                              ["cmd --data d --symbol BTC"], 1.0)
        steps = MicrostructureRunner(h, {}).steps()
        names = [s.__name__ for s in steps]
        assert "run_cost_check" in names
        assert names.index("run_cost_check") == 1  # after discovery

    def test_registry_paths_are_distinct(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        paths = {
            str(MicrostructureRunner.REGISTRY_PATH),
            str(MediumFrequencyRunner.REGISTRY_PATH),
            str(MacroRunner.REGISTRY_PATH),
        }
        assert len(paths) == 3, "Registry paths must be distinct"

    def test_signal_features_are_distinct(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        micro = set(MicrostructureRunner.SIGNAL_FEATURES)
        mf = set(MediumFrequencyRunner.SIGNAL_FEATURES)
        macro = set(MacroRunner.SIGNAL_FEATURES)
        # No full overlap between any pair
        assert micro != mf
        assert micro != macro
        assert mf != macro

    def test_default_features_are_distinct(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        defaults = {
            MicrostructureRunner.DEFAULT_FEATURE,
            MediumFrequencyRunner.DEFAULT_FEATURE,
            MacroRunner.DEFAULT_FEATURE,
        }
        assert len(defaults) == 3

    def test_default_horizons_match_expectations(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        assert MicrostructureRunner.DEFAULT_HORIZON_S == 5.0
        assert MediumFrequencyRunner.DEFAULT_HORIZON_S == 300.0
        assert MacroRunner.DEFAULT_HORIZON_S == 3600.0

    def test_timeframes_match_expectations(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner

        assert MicrostructureRunner.TIMEFRAME is None
        assert MediumFrequencyRunner.TIMEFRAME == "5min"
        assert MacroRunner.TIMEFRAME == "1h"


# ===========================================================================
# Registration correctness
# ===========================================================================

class TestRegisterSignal:
    def test_register_writes_correct_horizon(self, tmp_path):
        from agent.macro_runner import MacroRunner
        orig = MacroRunner.REGISTRY_PATH
        MacroRunner.REGISTRY_PATH = tmp_path / "registry.json"

        h = Hypothesis.create("ctx_funding_zscore test", "gen",
                              ["cmd --data d --symbol BTC"], 1.0)
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.10 PASS"}]}
        runner = MacroRunner(h, {})
        sig = runner.register_signal()

        # Macro default horizon is 3600s
        assert sig.horizon_s == 3600.0
        MacroRunner.REGISTRY_PATH = orig

    def test_register_appends_to_existing(self, tmp_path):
        from agent.mf_runner import MediumFrequencyRunner
        orig = MediumFrequencyRunner.REGISTRY_PATH
        reg_path = tmp_path / "registry.json"
        MediumFrequencyRunner.REGISTRY_PATH = reg_path

        # Pre-populate registry
        with open(reg_path, "w") as f:
            json.dump([{"name": "existing_signal", "hypothesis_id": "old"}], f)

        h = Hypothesis.create("trend_momentum_300 test", "gen",
                              ["cmd --data d --symbol BTC"], 1.0)
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.12 PASS"}]}
        runner = MediumFrequencyRunner(h, {})
        runner.register_signal()

        with open(reg_path) as f:
            registry = json.load(f)
        assert len(registry) == 2
        assert registry[0]["hypothesis_id"] == "old"
        assert registry[1]["hypothesis_id"] == h.id

        MediumFrequencyRunner.REGISTRY_PATH = orig

    def test_register_uses_threshold_horizon_over_default(self, tmp_path):
        from agent.runner import MicrostructureRunner
        orig = MicrostructureRunner.REGISTRY_PATH
        MicrostructureRunner.REGISTRY_PATH = tmp_path / "registry.json"

        h = Hypothesis.create("test", "gen",
                              ["cmd --data d --symbol BTC"], 1.0,
                              thresholds={"horizon_s": 10.0})
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.20 PASS"}]}
        runner = MicrostructureRunner(h, {})
        sig = runner.register_signal()
        # Threshold horizon overrides default 5.0
        assert sig.horizon_s == 10.0

        MicrostructureRunner.REGISTRY_PATH = orig


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestRunnerBackwardCompat:
    def test_experiment_runner_alias(self):
        from agent.runner import ExperimentRunner, MicrostructureRunner
        assert ExperimentRunner is MicrostructureRunner

    def test_module_constants_still_accessible(self):
        from agent.runner import REGISTRY_PATH
        from agent.mf_runner import MF_REGISTRY_PATH
        from agent.macro_runner import MACRO_REGISTRY_PATH
        assert "agent" in str(REGISTRY_PATH)
        assert "agent_mf" in str(MF_REGISTRY_PATH)
        assert "agent_macro" in str(MACRO_REGISTRY_PATH)

    def test_gate_functions_importable_from_runner(self):
        from agent.runner import (
            check_ic_gate, check_dIC_gate, check_cost_gate,
            check_correlation_gate, check_coverage_gate,
            check_walkforward_gate, run_nat, run_nat_cached,
            parse_report, apply_fdr,
        )
        assert all(callable(f) for f in [
            check_ic_gate, check_dIC_gate, check_cost_gate,
            check_correlation_gate, check_coverage_gate,
            check_walkforward_gate, run_nat, run_nat_cached,
            parse_report, apply_fdr,
        ])

    def test_mf_signal_features_importable(self):
        from agent.mf_runner import MF_SIGNAL_FEATURES
        assert isinstance(MF_SIGNAL_FEATURES, list)
        assert "trend_momentum_300" in MF_SIGNAL_FEATURES

    def test_macro_signal_features_importable(self):
        from agent.macro_runner import MACRO_SIGNAL_FEATURES
        assert isinstance(MACRO_SIGNAL_FEATURES, list)
        assert "ctx_funding_zscore" in MACRO_SIGNAL_FEATURES

    def test_all_runners_are_base_runner_subclasses(self):
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner
        for cls in [MicrostructureRunner, MediumFrequencyRunner, MacroRunner]:
            assert issubclass(cls, BaseRunner)

    def test_load_registry_callable_on_class(self):
        """_load_registry must be callable as ClassName._load_registry()."""
        from agent.runner import MicrostructureRunner
        from agent.mf_runner import MediumFrequencyRunner
        from agent.macro_runner import MacroRunner
        # Should not raise (returns [] for non-existent paths)
        for cls in [MicrostructureRunner, MediumFrequencyRunner, MacroRunner]:
            result = cls._load_registry()
            assert isinstance(result, list)

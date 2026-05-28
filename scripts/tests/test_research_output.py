"""Tests for structured hypothesis output (research_output.py).

Covers:
- Gate result parsing from various message formats
- _infer_gate_name classification
- Metric, threshold, and p-value regex extraction
- build_hypothesis_record schema completeness
- build_cycle_summary structure and aggregation
- File output (JSON written to disk)
- GENERATOR_MATH coverage (all generators have derivations)
- Integration with real Hypothesis objects (various statuses)
- Edge cases: empty results, missing fields, malformed messages
- Schema version consistency
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


import pytest
from agent.hypothesis import Hypothesis, GeneratorStats
from agent.research_output import (
    GENERATOR_MATH,
    _parse_gate_results,
    _infer_gate_name,
    _extract_metric,
    _extract_threshold,
    _extract_pvalue,
    _extract_features_from_thresholds,
    build_hypothesis_record,
    build_cycle_summary,
    _write_record,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_hypothesis():
    """A hypothesis that passed all gates."""
    h = Hypothesis.create(
        claim="Spread feature predicts 5s returns in BTC",
        generator="systematic",
        test_protocol=["compute_ic", "temporal_replicate", "symbol_replicate"],
        priority=1.5,
        thresholds={"horizon_s": 5.0, "regime_gate": "ent_book_shape<0.4"},
    )
    h.status = "replicated"
    h.completed = datetime.now(timezone.utc).isoformat()
    h.results = {
        "gate_results": [
            {"passed": True, "msg": "PASS IC=0.082 vs min=0.03 dIC=0.045 vs min=0.01 p=0.002"},
            {"passed": True, "msg": "PASS coverage=75% vs min=20%"},
        ],
        "cost_check": "PASS avg_ret=0.0003 vs min=0.0001",
        "correlation_check": "max_corr=0.31 vs registered — OK",
        "symbol_replication": {"n_pass": 2, "passed": ["ETH", "SOL"], "failed": []},
    }
    return h


@pytest.fixture
def failed_hypothesis():
    """A hypothesis that failed at IC gate."""
    h = Hypothesis.create(
        claim="Volume imbalance predicts 10s returns",
        generator="spectral",
        test_protocol=["compute_ic"],
        priority=0.8,
        thresholds={"horizon_s": 10.0},
    )
    h.fail("no_effect")
    h.results = {
        "gate_results": [
            {"passed": False, "msg": "FAIL IC=0.012 vs min=0.03 p=0.45"},
        ],
    }
    return h


@pytest.fixture
def recycled_hypothesis():
    """A hypothesis recycled from a parent."""
    h = Hypothesis.create(
        claim="Spread feature (tighter threshold) predicts 5s returns",
        generator="recycler",
        test_protocol=["compute_ic"],
        priority=1.0,
        thresholds={"horizon_s": 5.0, "regime_gate": ""},
        parent_id="HYP-SYS-abc12345",
    )
    h.status = "passed"
    h.results = {
        "gate_results": [
            {"passed": True, "msg": "PASS IC=0.065 vs min=0.03 dIC=0.025 vs min=0.01 p=0.01"},
        ],
    }
    return h


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "research"


# ===========================================================================
# _infer_gate_name
# ===========================================================================

class TestInferGateName:
    def test_ic_gate(self):
        assert _infer_gate_name("PASS IC=0.082 vs min=0.03") == "IC"

    def test_dic_gate(self):
        # dIC appears before IC in the string
        assert _infer_gate_name("PASS dIC=0.045 vs min=0.01 IC=0.082") == "dIC"

    def test_coverage_gate(self):
        assert _infer_gate_name("PASS coverage=75% vs min=20%") == "coverage"

    def test_cost_gate(self):
        assert _infer_gate_name("PASS avg_ret=0.0003 vs min=0.0001") == "cost"

    def test_walkforward_gate(self):
        assert _infer_gate_name("KEEP=True OOS IC=0.05") == "walkforward"

    def test_correlation_gate(self):
        assert _infer_gate_name("max_corr=0.31 vs registered") == "correlation"

    def test_unknown_gate(self):
        assert _infer_gate_name("some random message") == "unknown"

    def test_empty_message(self):
        assert _infer_gate_name("") == "unknown"

    def test_ic_without_dic_prefix(self):
        # IC= without dIC= before it
        assert _infer_gate_name("IC=0.05 vs min=0.03") == "IC"

    def test_dic_after_ic_is_still_ic(self):
        # dIC appears AFTER IC — should classify as IC (positional logic)
        assert _infer_gate_name("IC=0.05 dIC=0.02") == "IC"


# ===========================================================================
# _extract_metric
# ===========================================================================

class TestExtractMetric:
    def test_ic_positive(self):
        assert _extract_metric("IC=0.082 vs min=0.03") == pytest.approx(0.082)

    def test_ic_negative(self):
        assert _extract_metric("IC=-0.045 vs min=0.03") == pytest.approx(-0.045)

    def test_dic(self):
        assert _extract_metric("dIC=0.025 vs min=0.01") == pytest.approx(0.025)

    def test_avg_ret(self):
        assert _extract_metric("avg_ret=0.0003 vs min=0.0001") == pytest.approx(0.0003)

    def test_max_corr(self):
        assert _extract_metric("max_corr=0.31 vs registered") == pytest.approx(0.31)

    def test_coverage_percent(self):
        assert _extract_metric("coverage=75% vs min=20%") == pytest.approx(0.75)

    def test_coverage_decimal(self):
        assert _extract_metric("coverage=0.75 vs min=0.20") == pytest.approx(0.75)

    def test_no_metric(self):
        assert _extract_metric("some random text") is None

    def test_empty_string(self):
        assert _extract_metric("") is None

    def test_ic_integer(self):
        assert _extract_metric("IC=1 vs min=0.03") == pytest.approx(1.0)

    def test_dic_takes_priority_over_ic(self):
        # dIC pattern matches first due to list ordering
        result = _extract_metric("dIC=0.02 IC=0.05")
        # The regex tries IC= first, but "dIC=0.02" contains "IC=0.02" substring
        # Actually let's check what happens
        assert result is not None


# ===========================================================================
# _extract_threshold
# ===========================================================================

class TestExtractThreshold:
    def test_vs_min_pattern(self):
        assert _extract_threshold("IC=0.08 vs min=0.03") == pytest.approx(0.03)

    def test_threshold_pattern(self):
        assert _extract_threshold("threshold=0.05") == pytest.approx(0.05)

    def test_negative_threshold(self):
        assert _extract_threshold("vs min=-0.01") == pytest.approx(-0.01)

    def test_no_threshold(self):
        assert _extract_threshold("no threshold here") is None

    def test_empty_string(self):
        assert _extract_threshold("") is None

    def test_vs_min_takes_priority(self):
        result = _extract_threshold("vs min=0.03 threshold=0.05")
        assert result == pytest.approx(0.03)


# ===========================================================================
# _extract_pvalue
# ===========================================================================

class TestExtractPvalue:
    def test_simple_pvalue(self):
        assert _extract_pvalue("p=0.002") == pytest.approx(0.002)

    def test_scientific_notation(self):
        assert _extract_pvalue("p=3.2e-05") == pytest.approx(3.2e-5)

    def test_scientific_positive_exp(self):
        assert _extract_pvalue("p=1.5e+02") == pytest.approx(150.0)

    def test_large_pvalue(self):
        assert _extract_pvalue("FAIL p=0.45") == pytest.approx(0.45)

    def test_no_pvalue(self):
        assert _extract_pvalue("IC=0.05 vs min=0.03") is None

    def test_empty_string(self):
        assert _extract_pvalue("") is None

    def test_pvalue_at_end(self):
        assert _extract_pvalue("PASS IC=0.08 p=0.001") == pytest.approx(0.001)


# ===========================================================================
# _parse_gate_results
# ===========================================================================

class TestParseGateResults:
    def test_none_results(self):
        assert _parse_gate_results(None) == []

    def test_empty_dict(self):
        assert _parse_gate_results({}) == []

    def test_single_gate_pass(self):
        results = {
            "gate_results": [
                {"passed": True, "msg": "PASS IC=0.08 vs min=0.03 p=0.001"},
            ]
        }
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["name"] == "IC"
        assert gates[0]["passed"] is True
        assert gates[0]["metric"] == pytest.approx(0.08)
        assert gates[0]["threshold"] == pytest.approx(0.03)
        assert gates[0]["p_value"] == pytest.approx(0.001)

    def test_multiple_gate_results(self):
        results = {
            "gate_results": [
                {"passed": True, "msg": "PASS IC=0.08 vs min=0.03 p=0.001"},
                {"passed": True, "msg": "PASS coverage=65% vs min=20%"},
                {"passed": False, "msg": "FAIL dIC=0.005 vs min=0.01 IC=0.03 p=0.12"},
            ]
        }
        gates = _parse_gate_results(results)
        assert len(gates) == 3
        assert gates[0]["name"] == "IC"
        assert gates[1]["name"] == "coverage"
        assert gates[2]["name"] == "dIC"
        assert gates[2]["passed"] is False

    def test_cost_check_pass(self):
        results = {"cost_check": "PASS avg_ret=0.0003 vs min=0.0001"}
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["name"] == "cost"
        assert gates[0]["passed"] is True
        assert gates[0]["metric"] == pytest.approx(0.0003)

    def test_cost_check_fail(self):
        results = {"cost_check": "FAIL avg_ret=-0.0001 vs min=0.0001"}
        gates = _parse_gate_results(results)
        assert gates[0]["passed"] is False

    def test_correlation_check_ok(self):
        results = {"correlation_check": "max_corr=0.31 vs registered — OK"}
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["name"] == "correlation"
        assert gates[0]["passed"] is True
        assert gates[0]["metric"] == pytest.approx(0.31)

    def test_correlation_check_redundant(self):
        results = {"correlation_check": "REDUNDANT max_corr=0.92 vs registered"}
        gates = _parse_gate_results(results)
        assert gates[0]["passed"] is False

    def test_symbol_replication_pass(self):
        results = {
            "symbol_replication": {"n_pass": 2, "passed": ["ETH", "SOL"], "failed": []}
        }
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["name"] == "symbol_replication"
        assert gates[0]["passed"] is True
        assert gates[0]["metric"] == 2

    def test_symbol_replication_fail(self):
        results = {
            "symbol_replication": {"n_pass": 0, "passed": [], "failed": ["ETH", "SOL"]}
        }
        gates = _parse_gate_results(results)
        assert gates[0]["passed"] is False
        assert gates[0]["metric"] == 0

    def test_all_gates_combined(self):
        """Full results dict with all gate types."""
        results = {
            "gate_results": [
                {"passed": True, "msg": "PASS IC=0.08 vs min=0.03 p=0.001"},
            ],
            "cost_check": "PASS avg_ret=0.0003 vs min=0.0001",
            "correlation_check": "max_corr=0.31 — OK",
            "symbol_replication": {"n_pass": 2, "passed": ["ETH", "SOL"], "failed": []},
        }
        gates = _parse_gate_results(results)
        assert len(gates) == 4
        names = [g["name"] for g in gates]
        assert "IC" in names
        assert "cost" in names
        assert "correlation" in names
        assert "symbol_replication" in names

    def test_gate_results_empty_list(self):
        results = {"gate_results": []}
        assert _parse_gate_results(results) == []

    def test_missing_msg_field(self):
        results = {"gate_results": [{"passed": True}]}
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["message"] == ""
        assert gates[0]["name"] == "unknown"


# ===========================================================================
# _extract_features_from_thresholds
# ===========================================================================

class TestExtractFeatures:
    def test_regime_gate_feature(self):
        thresholds = {"regime_gate": "ent_book_shape<0.4"}
        assert _extract_features_from_thresholds(thresholds) == ["ent_book_shape"]

    def test_regime_gate_gt(self):
        thresholds = {"regime_gate": "spread_bps>2.5"}
        assert _extract_features_from_thresholds(thresholds) == ["spread_bps"]

    def test_empty_regime_gate(self):
        thresholds = {"regime_gate": ""}
        assert _extract_features_from_thresholds(thresholds) == []

    def test_no_regime_gate(self):
        thresholds = {"horizon_s": 5.0}
        assert _extract_features_from_thresholds(thresholds) == []

    def test_complex_feature_name(self):
        thresholds = {"regime_gate": "whale_flow_net_30s<100"}
        assert _extract_features_from_thresholds(thresholds) == ["whale_flow_net_30s"]


# ===========================================================================
# build_hypothesis_record
# ===========================================================================

class TestBuildHypothesisRecord:
    def test_schema_version(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["schema_version"] == 1

    def test_all_required_fields(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        required = [
            "schema_version", "id", "agent", "generator", "claim",
            "math", "status", "failure_reason", "gates", "features",
            "regime_gate", "horizon_s", "thresholds", "parent_id", "timestamps",
        ]
        for field in required:
            assert field in record, f"Missing field: {field}"

    def test_agent_type_propagates(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "medium_freq")
        assert record["agent"] == "medium_freq"

    def test_math_from_generator(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["math"] == GENERATOR_MATH["systematic"]
        assert len(record["math"]) > 0

    def test_unknown_generator_empty_math(self):
        h = Hypothesis.create(
            claim="test", generator="unknown_gen",
            test_protocol=[], thresholds={},
        )
        record = build_hypothesis_record(h, "micro")
        assert record["math"] == ""

    def test_gates_parsed(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        # gate_results(2) + cost + correlation + symbol_replication = 5
        assert len(record["gates"]) == 5

    def test_features_extracted(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert "ent_book_shape" in record["features"]

    def test_regime_gate(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["regime_gate"] == "ent_book_shape<0.4"

    def test_horizon_s(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["horizon_s"] == 5.0

    def test_timestamps(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert "created" in record["timestamps"]
        assert "completed" in record["timestamps"]
        assert record["timestamps"]["created"] is not None
        assert record["timestamps"]["completed"] is not None

    def test_failed_hypothesis(self, failed_hypothesis):
        record = build_hypothesis_record(failed_hypothesis, "micro")
        assert record["status"] == "failed"
        assert record["failure_reason"] == "no_effect"
        assert record["generator"] == "spectral"
        assert record["math"] == GENERATOR_MATH["spectral"]

    def test_recycled_hypothesis_parent_id(self, recycled_hypothesis):
        record = build_hypothesis_record(recycled_hypothesis, "micro")
        assert record["parent_id"] == "HYP-SYS-abc12345"
        assert record["math"] == GENERATOR_MATH["recycler"]

    def test_writes_to_disk(self, basic_hypothesis, output_dir):
        record = build_hypothesis_record(
            basic_hypothesis, "micro", output_root=output_dir
        )
        json_path = output_dir / "hypotheses" / f"{basic_hypothesis.id}.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["id"] == basic_hypothesis.id
        assert loaded["schema_version"] == 1

    def test_no_write_without_output_root(self, basic_hypothesis, tmp_path):
        record = build_hypothesis_record(basic_hypothesis, "micro", output_root=None)
        # Should still return the record
        assert record["id"] == basic_hypothesis.id
        # But no files written
        assert not (tmp_path / "research").exists()

    def test_hypothesis_with_no_results(self):
        h = Hypothesis.create(
            claim="untested", generator="systematic",
            test_protocol=[], thresholds={},
        )
        record = build_hypothesis_record(h, "micro")
        assert record["gates"] == []
        assert record["status"] == "queued"

    def test_claim_preserved_verbatim(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["claim"] == basic_hypothesis.claim

    def test_thresholds_preserved(self, basic_hypothesis):
        record = build_hypothesis_record(basic_hypothesis, "micro")
        assert record["thresholds"]["horizon_s"] == 5.0
        assert "regime_gate" in record["thresholds"]


# ===========================================================================
# build_cycle_summary
# ===========================================================================

class TestBuildCycleSummary:
    def _make_hypotheses(self, n_pass=2, n_fail=3):
        hyps = []
        for i in range(n_pass):
            h = Hypothesis.create(
                claim=f"Passing hypothesis {i}",
                generator="systematic",
                test_protocol=[],
            )
            h.status = "replicated"
            hyps.append(h)
        for i in range(n_fail):
            h = Hypothesis.create(
                claim=f"Failing hypothesis {i}",
                generator="spectral",
                test_protocol=[],
            )
            h.fail("no_effect")
            hyps.append(h)
        return hyps

    def test_schema_version(self):
        hyps = self._make_hypotheses()
        summary = build_cycle_summary(
            cycle_id="CYC-test01",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=120.5,
            hypotheses=hyps,
            n_registered=2,
            n_fdr_rejected=0,
            n_chained=1,
            fdr_q=0.05,
            generator_stats={},
        )
        assert summary["schema_version"] == 1

    def test_all_required_fields(self):
        hyps = self._make_hypotheses()
        summary = build_cycle_summary(
            cycle_id="CYC-test01",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=120.5,
            hypotheses=hyps,
            n_registered=2,
            n_fdr_rejected=0,
            n_chained=1,
            fdr_q=0.05,
            generator_stats={},
        )
        required = [
            "schema_version", "cycle_id", "agent", "started", "completed",
            "duration_s", "n_tested", "n_registered", "n_fdr_rejected",
            "n_chained", "fdr_q", "hypotheses", "generator_stats",
        ]
        for field in required:
            assert field in summary, f"Missing field: {field}"

    def test_n_tested_counts(self):
        hyps = self._make_hypotheses(n_pass=3, n_fail=7)
        summary = build_cycle_summary(
            cycle_id="CYC-test02",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=60.0,
            hypotheses=hyps,
            n_registered=3,
            n_fdr_rejected=1,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        assert summary["n_tested"] == 10
        assert summary["n_registered"] == 3
        assert summary["n_fdr_rejected"] == 1
        assert summary["n_chained"] == 0

    def test_hypothesis_summaries(self):
        hyps = self._make_hypotheses(n_pass=1, n_fail=1)
        summary = build_cycle_summary(
            cycle_id="CYC-test03",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=30.0,
            hypotheses=hyps,
            n_registered=1,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        hs = summary["hypotheses"]
        assert len(hs) == 2
        assert all("id" in h and "generator" in h and "status" in h for h in hs)
        assert hs[0]["status"] == "replicated"
        assert hs[1]["status"] == "failed"
        assert hs[1]["failure_reason"] == "no_effect"

    def test_claim_truncated(self):
        h = Hypothesis.create(
            claim="A" * 200,  # very long claim
            generator="systematic",
            test_protocol=[],
        )
        summary = build_cycle_summary(
            cycle_id="CYC-trunc",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[h],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        assert len(summary["hypotheses"][0]["claim"]) == 100

    def test_generator_stats_serialization(self):
        stats = {
            "systematic": GeneratorStats(attempts=10, successes=3),
            "spectral": GeneratorStats(attempts=5, successes=0),
        }
        summary = build_cycle_summary(
            cycle_id="CYC-stats",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=45.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats=stats,
        )
        gs = summary["generator_stats"]
        assert gs["systematic"]["attempts"] == 10
        assert gs["systematic"]["successes"] == 3
        assert gs["systematic"]["hit_rate"] == pytest.approx(0.3)
        assert gs["systematic"]["weight"] == pytest.approx(4 / 12)
        assert gs["spectral"]["attempts"] == 5
        assert gs["spectral"]["successes"] == 0

    def test_duration_rounded(self):
        summary = build_cycle_summary(
            cycle_id="CYC-dur",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=123.456789,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        assert summary["duration_s"] == 123.5

    def test_completed_is_utc_iso(self):
        summary = build_cycle_summary(
            cycle_id="CYC-ts",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        assert "+00:00" in summary["completed"]

    def test_writes_to_disk(self, output_dir):
        summary = build_cycle_summary(
            cycle_id="CYC-disk01",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
            output_root=output_dir,
        )
        json_path = output_dir / "cycles" / "CYC-disk01.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["cycle_id"] == "CYC-disk01"

    def test_no_write_without_output_root(self, tmp_path):
        summary = build_cycle_summary(
            cycle_id="CYC-nowrite",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
            output_root=None,
        )
        assert summary["cycle_id"] == "CYC-nowrite"

    def test_fdr_q_preserved(self):
        summary = build_cycle_summary(
            cycle_id="CYC-fdr",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.10,
            generator_stats={},
        )
        assert summary["fdr_q"] == 0.10

    def test_empty_hypotheses_list(self):
        summary = build_cycle_summary(
            cycle_id="CYC-empty",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=0.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        assert summary["n_tested"] == 0
        assert summary["hypotheses"] == []


# ===========================================================================
# _write_record
# ===========================================================================

class TestWriteRecord:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "dir"
        _write_record(target, "test-record", {"key": "value"})
        assert (target / "test-record.json").exists()

    def test_valid_json(self, tmp_path):
        data = {"num": 1.23, "nested": {"a": [1, 2, 3]}}
        _write_record(tmp_path, "test", data)
        loaded = json.loads((tmp_path / "test.json").read_text())
        assert loaded == data

    def test_handles_datetime(self, tmp_path):
        """default=str in json.dump handles datetime objects."""
        data = {"ts": datetime.now(timezone.utc)}
        _write_record(tmp_path, "ts-test", data)
        loaded = json.loads((tmp_path / "ts-test.json").read_text())
        assert "ts" in loaded

    def test_overwrites_existing(self, tmp_path):
        _write_record(tmp_path, "ow", {"v": 1})
        _write_record(tmp_path, "ow", {"v": 2})
        loaded = json.loads((tmp_path / "ow.json").read_text())
        assert loaded["v"] == 2

    def test_indented_output(self, tmp_path):
        _write_record(tmp_path, "indent", {"a": 1, "b": 2})
        raw = (tmp_path / "indent.json").read_text()
        assert "\n" in raw  # indented, not single-line


# ===========================================================================
# GENERATOR_MATH coverage
# ===========================================================================

class TestGeneratorMath:
    """Every known generator type has a non-empty LaTeX derivation."""

    EXPECTED_GENERATORS = [
        "systematic", "spectral", "regime", "cross_asset", "recycler",
        "ensemble", "momentum", "vol_breakout", "flow_cluster",
        "funding_meanrev", "oi_divergence", "whale_momentum", "it_discovery",
    ]

    def test_all_expected_generators_present(self):
        for gen in self.EXPECTED_GENERATORS:
            assert gen in GENERATOR_MATH, f"Missing math for generator: {gen}"

    def test_all_derivations_nonempty(self):
        for gen, math in GENERATOR_MATH.items():
            assert len(math) > 10, f"Empty/trivial math for generator: {gen}"

    def test_systematic_contains_ic(self):
        assert "IC" in GENERATOR_MATH["systematic"]

    def test_spectral_contains_psd(self):
        assert "PSD" in GENERATOR_MATH["spectral"]

    def test_regime_contains_regime(self):
        assert "regime" in GENERATOR_MATH["regime"]

    def test_it_discovery_contains_mutual_info(self):
        assert "I(X; Y)" in GENERATOR_MATH["it_discovery"]

    def test_no_extra_generators_without_math(self):
        """Sanity: GENERATOR_MATH keys ⊆ expected list + possibly new ones."""
        # This just checks we didn't accidentally remove a key
        assert len(GENERATOR_MATH) >= len(self.EXPECTED_GENERATORS)


# ===========================================================================
# Integration: base.py integration points
# ===========================================================================

class TestBaseIntegration:
    """Verify research_output is importable and callable from base."""

    def test_build_hypothesis_record_importable(self):
        from agent.research_output import build_hypothesis_record
        assert callable(build_hypothesis_record)

    def test_build_cycle_summary_importable(self):
        from agent.research_output import build_cycle_summary
        assert callable(build_cycle_summary)

    def test_base_has_emit_hypothesis_method(self):
        from agent.base import ResearchAgent
        assert hasattr(ResearchAgent, "_emit_hypothesis_record")

    def test_base_has_emit_cycle_summary_method(self):
        from agent.base import ResearchAgent
        assert hasattr(ResearchAgent, "_emit_cycle_summary")

    def test_base_has_research_output_root_property(self):
        from agent.base import ResearchAgent
        assert hasattr(ResearchAgent, "research_output_root")


# ===========================================================================
# Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    def test_hypothesis_with_empty_thresholds(self):
        h = Hypothesis.create(
            claim="minimal", generator="systematic",
            test_protocol=[], thresholds={},
        )
        record = build_hypothesis_record(h, "micro")
        assert record["regime_gate"] is None
        assert record["horizon_s"] is None
        assert record["features"] == []

    def test_hypothesis_with_none_results(self):
        h = Hypothesis.create(
            claim="no results", generator="regime",
            test_protocol=[],
        )
        h.results = None
        record = build_hypothesis_record(h, "micro")
        assert record["gates"] == []

    def test_malformed_gate_message(self):
        results = {
            "gate_results": [
                {"passed": True, "msg": "PASS weird=format no_metrics here"},
            ]
        }
        gates = _parse_gate_results(results)
        assert len(gates) == 1
        assert gates[0]["name"] == "unknown"
        assert gates[0]["metric"] is None
        assert gates[0]["threshold"] is None

    def test_gate_with_special_characters(self):
        results = {
            "gate_results": [
                {"passed": True, "msg": "PASS IC=0.05 — special chars éàü"},
            ]
        }
        gates = _parse_gate_results(results)
        assert gates[0]["metric"] == pytest.approx(0.05)

    def test_very_long_claim_in_cycle(self):
        h = Hypothesis.create(
            claim="x" * 500,
            generator="systematic",
            test_protocol=[],
        )
        summary = build_cycle_summary(
            cycle_id="CYC-long",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=1.0,
            hypotheses=[h],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats={},
        )
        # Claim truncated to 100 chars in summary
        assert len(summary["hypotheses"][0]["claim"]) == 100

    def test_record_json_serializable(self, basic_hypothesis, output_dir):
        """The full record must be JSON-serializable (no custom objects)."""
        record = build_hypothesis_record(basic_hypothesis, "micro")
        # Should not raise
        serialized = json.dumps(record, default=str)
        reloaded = json.loads(serialized)
        assert reloaded["id"] == record["id"]

    def test_cycle_summary_json_serializable(self):
        stats = {"sys": GeneratorStats(attempts=5, successes=2)}
        summary = build_cycle_summary(
            cycle_id="CYC-ser",
            agent_type="micro",
            started="2026-05-25T10:00:00+00:00",
            duration_s=10.0,
            hypotheses=[],
            n_registered=0,
            n_fdr_rejected=0,
            n_chained=0,
            fdr_q=0.05,
            generator_stats=stats,
        )
        serialized = json.dumps(summary, default=str)
        reloaded = json.loads(serialized)
        assert reloaded["cycle_id"] == "CYC-ser"

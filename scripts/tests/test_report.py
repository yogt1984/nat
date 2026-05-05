"""
Tests for Task 8.2: Automated Report Generator.

Covers: report structure, section presence, markdown format,
different verdicts, cross-symbol inclusion, and edge cases.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional

from cluster_pipeline.report import generate_report
from cluster_pipeline.validate import CrossSymbolResult


# ---------------------------------------------------------------------------
# Helpers — Build mock ProfilingResult
# ---------------------------------------------------------------------------


def _make_structure_test(has_structure=True, hopkins=0.8, dip_p=0.01):
    st = MagicMock()
    st.has_structure = has_structure
    st.hopkins = hopkins
    st.dip_p = dip_p
    st.recommendation = "proceed" if has_structure else "stop"
    return st


def _make_quality(silhouette=0.5, min_cluster_fraction=0.2):
    q = MagicMock()
    q.silhouette = silhouette
    q.min_cluster_fraction = min_cluster_fraction
    q.n_per_cluster = {0: 50, 1: 50}
    return q


def _make_stability(mean_ari=0.8, std_ari=0.05):
    s = MagicMock()
    s.mean_ari = mean_ari
    s.std_ari = std_ari
    s.n_bootstrap = 20
    s.block_size = 10
    return s


def _make_macro(k=2, early_exit=False, silhouette=0.5, mean_ari=0.8, str_val=0.85):
    macro = MagicMock()
    macro.k = k
    macro.early_exit = early_exit
    macro.quality = _make_quality(silhouette=silhouette)
    macro.stability = _make_stability(mean_ari=mean_ari)
    macro.self_transition_rate = str_val
    macro.durations = {i: [5, 6, 7, 8] for i in range(k)}
    macro.structure_test = _make_structure_test(has_structure=not early_exit)
    return macro


def _make_micro(regime_id=0, k=2, n_bars=50, silhouette=0.4, mean_ari=0.7):
    micro = MagicMock()
    micro.regime_id = regime_id
    micro.k = k
    micro.n_bars = n_bars
    micro.quality = _make_quality(silhouette=silhouette)
    micro.stability = _make_stability(mean_ari=mean_ari)
    return micro


def _make_hierarchy(n_regimes=2, micro_per_regime=2):
    h = MagicMock()
    n_total = n_regimes * micro_per_regime
    h.n_micro_total = n_total
    h.micro_labels = np.zeros(100, dtype=int)
    # Assign alternating states
    for i in range(100):
        h.micro_labels[i] = i % n_total
    label_map = {}
    gid = 0
    for r in range(n_regimes):
        for l in range(micro_per_regime):
            label_map[gid] = (r, l)
            gid += 1
    h.label_map = label_map
    return h


def _make_profiling_result(
    n_bars=100, n_regimes=2, micro_per_regime=2,
    early_exit=False, silhouette=0.5, mean_ari=0.8, str_val=0.85,
):
    pr = MagicMock()
    pr.bars = np.zeros((n_bars, 5))  # dummy bars
    pr.macro = _make_macro(
        k=n_regimes, early_exit=early_exit,
        silhouette=silhouette, mean_ari=mean_ari, str_val=str_val,
    )
    pr.hierarchy = _make_hierarchy(n_regimes, micro_per_regime)
    pr.micros = {i: _make_micro(regime_id=i) for i in range(n_regimes)}
    # Remove break_indices attribute to test hasattr path
    if hasattr(pr, 'break_indices'):
        del pr.break_indices
    return pr


def _make_prices(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(rng.standard_normal(n) * 0.01) + 4.0)


# ---------------------------------------------------------------------------
# TestReportStructure
# ---------------------------------------------------------------------------


class TestReportStructure:
    """Tests for overall report structure."""

    def test_returns_string(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices, vector="orderflow", timeframe="15min")
        assert isinstance(report, str)

    def test_starts_with_header(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices, vector="entropy", timeframe="1h")
        assert report.startswith("# Profiling Report — entropy@1h")

    def test_contains_all_core_sections(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "## Data Summary" in report
        assert "## Structural Breaks" in report
        assert "## Structure Test" in report
        assert "## Macro Regimes" in report
        assert "## Micro States" in report
        assert "## Transition Structure" in report
        assert "## Validation Verdict" in report
        assert "## Recommendations" in report

    def test_contains_q_sections(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "### Q1: Structural" in report
        assert "### Q2: Predictive" in report
        assert "### Q3: Operational" in report

    def test_valid_markdown_tables(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        # Tables have header separator
        assert "|-----" in report


# ---------------------------------------------------------------------------
# TestSectionContent
# ---------------------------------------------------------------------------


class TestSectionContent:
    """Tests for section content accuracy."""

    def test_data_summary_bar_count(self):
        pr = _make_profiling_result(n_bars=100)
        prices = _make_prices(n=100)
        report = generate_report(pr, prices)
        assert "100" in report

    def test_macro_k_in_report(self):
        pr = _make_profiling_result(n_regimes=3)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "**k:** 3" in report

    def test_structure_test_hopkins(self):
        pr = _make_profiling_result()
        pr.macro.structure_test.hopkins = 0.7654
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "0.7654" in report

    def test_state_map_table(self):
        pr = _make_profiling_result(n_regimes=2, micro_per_regime=2)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "| Global ID |" in report
        assert "| 0 | 0 | 0 |" in report

    def test_vector_timeframe_in_header(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices, vector="toxicity", timeframe="5min")
        assert "toxicity@5min" in report


# ---------------------------------------------------------------------------
# TestVerdicts
# ---------------------------------------------------------------------------


class TestVerdicts:
    """Tests for different verdict outcomes in recommendations."""

    def test_go_verdict(self):
        pr = _make_profiling_result(silhouette=0.5, mean_ari=0.8, str_val=0.9)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "Deploy" in report or "GO" in report

    def test_drop_verdict_early_exit(self):
        pr = _make_profiling_result(early_exit=True)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "DROP" in report
        assert "Drop" in report or "no meaningful" in report.lower()

    def test_per_state_verdicts_table(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "### Per-State Verdicts" in report
        assert "| State | Verdict |" in report


# ---------------------------------------------------------------------------
# TestCrossSymbol
# ---------------------------------------------------------------------------


class TestCrossSymbol:
    """Tests for cross-symbol section."""

    def test_no_cross_symbol_by_default(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "## Cross-Symbol Consistency" not in report

    def test_cross_symbol_included_when_provided(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        cs = CrossSymbolResult(
            agreement_matrix=np.eye(3),
            mean_agreement=0.85,
            above_random=True,
            consensus_labels=np.zeros(100, dtype=int),
            disagreement_rate=0.05,
            per_symbol_labels={"BTC": np.zeros(100), "ETH": np.zeros(100), "SOL": np.zeros(100)},
            symbol_names=["BTC", "ETH", "SOL"],
        )
        report = generate_report(pr, prices, cross_symbol_result=cs)
        assert "## Cross-Symbol Consistency" in report
        assert "0.8500" in report
        assert "BTC" in report


# ---------------------------------------------------------------------------
# TestDriftBaseline
# ---------------------------------------------------------------------------


class TestDriftBaseline:
    """Tests for drift baseline section."""

    def test_no_drift_by_default(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "## Drift Baseline" not in report

    def test_drift_included_when_provided(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices, training_ll_p10=-5.5, training_ll_p50=-3.2)
        assert "## Drift Baseline" in report
        assert "-5.5000" in report
        assert "-3.2000" in report


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases."""

    def test_early_exit_report(self):
        """Early exit produces valid report without crash."""
        pr = _make_profiling_result(early_exit=True)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "Early exit" in report or "early exit" in report

    def test_single_regime(self):
        pr = _make_profiling_result(n_regimes=1, micro_per_regime=1)
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert "**k:** 1" in report

    def test_report_not_empty(self):
        pr = _make_profiling_result()
        prices = _make_prices()
        report = generate_report(pr, prices)
        assert len(report) > 100

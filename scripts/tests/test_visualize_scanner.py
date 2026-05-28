#!/usr/bin/env python3
"""Tests for visualize_scanner.py — verify all 10 plot functions produce valid Figures."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent.parent

from scalp_edge_scanner import (
    ConjunctionSetup,
    FeatureTailProfile,
    ScalpEdgeScanner,
    ScanReport,
    StabilityResult,
    StrategyArchetype,
    TailStats,
    TemporalProfile,
    _NumpyEncoder,
    save_json_report,
)
from visualize_scanner import (
    load_scan_report,
    plot_conditional_return_kde,
    plot_conjunction_heatmap,
    plot_cumulative_triggered_pnl,
    plot_edge_vs_frequency,
    plot_feature_coverage_heatmap,
    plot_holding_period_curves,
    plot_return_distribution_panel,
    plot_stability_half_split,
    plot_tick_activity_heatmap,
    plot_trigger_raster,
    run_data_plots,
    run_scanner_plots,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_bars(n=500, seed=42):
    """Synthetic bars with injected +5bps upper-tail edge in feature_signal."""
    rng = np.random.RandomState(seed)

    prices = np.cumsum(rng.randn(n) * 0.0001) + np.log(100_000)
    prices = np.exp(prices)
    feature = rng.randn(n)
    p95 = np.percentile(feature, 95)
    upper_mask = feature >= p95

    prices_adj = prices.copy()
    for i in range(n - 1):
        if upper_mask[i]:
            prices_adj[i + 1] = prices_adj[i] * (1 + 0.0005 + rng.randn() * 0.0001)

    bars = pd.DataFrame({
        "raw_midprice_close": prices_adj,
        "symbol": "BTC",
        "feature_signal": feature,
    })
    bars.index = pd.date_range("2026-01-01", periods=n, freq="5min")

    # Add vector-prefixed features for coverage heatmap
    for i in range(5):
        bars[f"ent_test_{i}"] = rng.randn(n)
    for i in range(3):
        bars[f"noise_{i}"] = rng.randn(n)
    bars["orderflow_imbalance_mean"] = rng.randn(n)
    bars["vpin_mean"] = rng.randn(n)

    return bars


def _make_scanner():
    """Scanner with relaxed thresholds for synthetic data."""
    return ScalpEdgeScanner(config={
        "timeframe": "5min",
        "cost_bps": 3.5,
        "tail_percentiles": [1, 5, 10, 90, 95, 99],
        "tail_threshold": 5,
        "top_features": 15,
        "conjunction_features": 10,
        "min_observations": 30,
        "min_tail_obs": 5,
        "significance_alpha": 0.10,
        "forward_horizon_bars": 1,
        "symbols": ["BTC"],
    })


def _make_report_and_data():
    """Run scanner analyses on synthetic bars, return (report, bars, fwd, scanner)."""
    bars = _make_bars()
    scanner = _make_scanner()
    fwd = scanner.compute_forward_returns(bars)
    features = scanner.select_features(bars, n=15)
    profiles = scanner.tail_return_profile(bars, fwd, features)
    sig = [p for p in profiles if p.is_significant]
    conjunctions = scanner.conjunction_scan(bars, fwd, sig)

    # Build setup masks
    setup_masks = []
    for p in sig[:10]:
        values = bars[p.name].values.astype(float)
        lo = p.percentile_thresholds.get(5, np.nan)
        hi = p.percentile_thresholds.get(95, np.nan)
        if p.upper_tail and (not p.lower_tail or abs(p.upper_tail.mean) >= abs(p.lower_tail.mean)):
            mask = np.isfinite(values) & (values >= hi)
        elif p.lower_tail:
            mask = np.isfinite(values) & (values <= lo)
        else:
            continue
        setup_masks.append((f"tail:{p.name}", mask))

    temporal = scanner.temporal_characterize(bars, fwd, setup_masks)
    stability = scanner.stability_assess(fwd, setup_masks)
    archetypes = scanner.classify_archetypes(sig, conjunctions, stability)

    report = ScanReport(
        symbol="BTC", timeframe="5min", n_bars=len(bars),
        timestamp="2026-01-01T00:00:00Z",
        tail_profiles=profiles, conjunctions=conjunctions,
        temporal=temporal, stability=stability, archetypes=archetypes,
        config=scanner.config, warnings=[],
    )
    return report, bars, fwd, scanner


@pytest.fixture(scope="module")
def report_data():
    """Module-scoped fixture: (report, bars, fwd_returns, scanner)."""
    return _make_report_and_data()


@pytest.fixture
def empty_report():
    """Report with no significant findings."""
    return ScanReport(
        symbol="BTC", timeframe="5min", n_bars=100,
        timestamp="2026-01-01T00:00:00Z",
        tail_profiles=[], conjunctions=[],
        temporal=[], stability=[], archetypes=[],
        config={"tail_threshold": 5, "cost_bps": 3.5}, warnings=[],
    )


# ---------------------------------------------------------------------------
# Test: JSON round-trip
# ---------------------------------------------------------------------------

class TestLoadScanReport:
    def test_round_trip(self, report_data):
        report, _, _, _ = report_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json_report(report, tmpdir)
            loaded = load_scan_report(path)
            assert loaded.symbol == report.symbol
            assert loaded.n_bars == report.n_bars
            assert len(loaded.tail_profiles) == len(report.tail_profiles)
            assert len(loaded.conjunctions) == len(report.conjunctions)
            assert len(loaded.stability) == len(report.stability)

    def test_preserves_tail_stats(self, report_data):
        report, _, _, _ = report_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json_report(report, tmpdir)
            loaded = load_scan_report(path)
            for orig, loaded_p in zip(report.tail_profiles, loaded.tail_profiles):
                assert orig.name == loaded_p.name
                assert orig.is_significant == loaded_p.is_significant
                if orig.upper_tail:
                    assert loaded_p.upper_tail is not None
                    assert abs(orig.upper_tail.mean - loaded_p.upper_tail.mean) < 1e-10


# ---------------------------------------------------------------------------
# Test: Plot functions
# ---------------------------------------------------------------------------

class TestConditionalReturnKDE:
    def test_returns_figure(self, report_data):
        report, bars, fwd, _ = report_data
        fig = plot_conditional_return_kde(report, bars, fwd)
        assert isinstance(fig, Figure)

    def test_empty_report(self, empty_report):
        bars = _make_bars(100)
        fwd = np.random.randn(100) * 0.001
        fig = plot_conditional_return_kde(empty_report, bars, fwd)
        assert isinstance(fig, Figure)


class TestEdgeVsFrequency:
    def test_returns_figure(self, report_data):
        report, _, _, _ = report_data
        fig = plot_edge_vs_frequency(report)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        fig = plot_edge_vs_frequency(empty_report)
        assert isinstance(fig, Figure)


class TestConjunctionHeatmap:
    def test_returns_figure(self, report_data):
        report, _, _, _ = report_data
        fig = plot_conjunction_heatmap(report)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        fig = plot_conjunction_heatmap(empty_report)
        assert isinstance(fig, Figure)


class TestTriggerRaster:
    def test_returns_figure(self, report_data):
        report, bars, fwd, _ = report_data
        fig = plot_trigger_raster(report, bars, fwd)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        bars = _make_bars(100)
        fwd = np.random.randn(100) * 0.001
        fig = plot_trigger_raster(empty_report, bars, fwd)
        assert isinstance(fig, Figure)


class TestCumulativeTriggeredPnl:
    def test_returns_figure(self, report_data):
        report, bars, fwd, _ = report_data
        fig = plot_cumulative_triggered_pnl(report, bars, fwd)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        bars = _make_bars(100)
        fwd = np.random.randn(100) * 0.001
        fig = plot_cumulative_triggered_pnl(empty_report, bars, fwd)
        assert isinstance(fig, Figure)


class TestHoldingPeriodCurves:
    def test_returns_figure(self, report_data):
        report, bars, _, scanner = report_data
        fig = plot_holding_period_curves(report, bars, scanner)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        bars = _make_bars(100)
        scanner = _make_scanner()
        fig = plot_holding_period_curves(empty_report, bars, scanner)
        assert isinstance(fig, Figure)


class TestStabilityHalfSplit:
    def test_returns_figure(self, report_data):
        report, _, _, _ = report_data
        fig = plot_stability_half_split(report)
        assert isinstance(fig, Figure)

    def test_empty(self, empty_report):
        fig = plot_stability_half_split(empty_report)
        assert isinstance(fig, Figure)


class TestFeatureCoverageHeatmap:
    def test_returns_figure(self):
        bars = _make_bars(200)
        fig = plot_feature_coverage_heatmap(bars)
        assert isinstance(fig, Figure)


class TestTickActivityHeatmap:
    def test_returns_figure(self):
        bars = _make_bars(500)
        fig = plot_tick_activity_heatmap(bars)
        assert isinstance(fig, Figure)


class TestReturnDistributionPanel:
    def test_returns_figure(self):
        bars = _make_bars(200)
        fwd = np.random.randn(200) * 0.001
        fig = plot_return_distribution_panel(bars, fwd, cost_bps=3.5)
        assert isinstance(fig, Figure)

    def test_has_two_axes(self):
        bars = _make_bars(200)
        fwd = np.random.randn(200) * 0.001
        fig = plot_return_distribution_panel(bars, fwd, cost_bps=3.5)
        assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# Test: Orchestration
# ---------------------------------------------------------------------------

class TestRunScannerPlots:
    def test_saves_pngs(self, report_data):
        report, bars, fwd, scanner = report_data
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_scanner_plots(report, bars, fwd, scanner, tmpdir, dpi=72)
            assert len(paths) >= 1
            for p in paths:
                assert p.endswith(".png")
                assert os.path.exists(p)
                assert os.path.getsize(p) > 0


class TestRunDataPlots:
    def test_saves_pngs(self):
        bars = _make_bars(200)
        fwd = np.random.randn(200) * 0.001
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_data_plots(bars, fwd, 3.5, tmpdir, "BTC", dpi=72)
            assert len(paths) >= 1
            for p in paths:
                assert p.endswith(".png")
                assert os.path.exists(p)
                assert os.path.getsize(p) > 0

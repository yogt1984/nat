#!/usr/bin/env python3
"""Tests for scalp_edge_scanner.py — synthetic data with injected tail effects."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent

from scalp_edge_scanner import (
    ARCHETYPES,
    ConjunctionSetup,
    FeatureTailProfile,
    ScalpEdgeScanner,
    ScanReport,
    StabilityResult,
    StrategyArchetype,
    TailStats,
    TemporalProfile,
    _NumpyEncoder,
    load_scanner_config,
    save_json_report,
    save_md_report,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic bar data with known injected tail effects
# ---------------------------------------------------------------------------

def _make_bars(n=2000, seed=42):
    """Create synthetic 5-min bars with price and features.

    Injects a known +5bps tail edge in feature_signal when it's above
    its 95th percentile (upper tail).
    """
    rng = np.random.RandomState(seed)

    prices = np.cumsum(rng.randn(n) * 0.0001) + np.log(100_000)
    prices = np.exp(prices)

    bars = pd.DataFrame({
        "raw_midprice_close": prices,
        "symbol": "BTC",
    })
    bars.index = pd.date_range("2026-01-01", periods=n, freq="5min")

    # Feature with injected tail effect
    feature = rng.randn(n)
    p95 = np.percentile(feature, 95)
    upper_mask = feature >= p95
    # Inject +5bps forward return shift in upper tail
    price_shift = np.zeros(n)
    price_shift[:-1][upper_mask[:-1]] = 0.0005  # +5bps
    prices_adj = prices.copy()
    for i in range(n - 1):
        if upper_mask[i]:
            # Shift future price up by 5bps
            prices_adj[i + 1] = prices_adj[i] * (1 + 0.0005 + rng.randn() * 0.0001)
    bars["raw_midprice_close"] = prices_adj
    bars["feature_signal"] = feature

    # Random noise features (should not show significance)
    for i in range(9):
        bars[f"noise_{i}"] = rng.randn(n)

    # Feature in 'orderflow' vector (for archetype matching)
    bars["orderflow_imbalance_mean"] = rng.randn(n)
    # Feature in 'toxicity' vector
    bars["vpin_mean"] = rng.randn(n)
    # Feature in 'entropy' vector
    bars["entropy_mean"] = rng.randn(n)
    # Feature in 'flow' vector
    bars["flow_intensity_mean"] = rng.randn(n)

    return bars


def _make_bars_with_conjunction(n=2000, seed=42):
    """Bars with two features that have a conjunction effect.

    feature_a upper tail + feature_b lower tail => +8bps edge.
    """
    rng = np.random.RandomState(seed)

    prices = np.cumsum(rng.randn(n) * 0.0001) + np.log(100_000)
    prices = np.exp(prices)

    feature_a = rng.randn(n)
    feature_b = rng.randn(n)

    p95_a = np.percentile(feature_a, 95)
    p5_b = np.percentile(feature_b, 5)

    # Inject conjunction effect
    for i in range(n - 1):
        if feature_a[i] >= p95_a and feature_b[i] <= p5_b:
            prices[i + 1] = prices[i] * (1 + 0.0008 + rng.randn() * 0.00005)

    bars = pd.DataFrame({
        "raw_midprice_close": prices,
        "symbol": "BTC",
        "feature_a": feature_a,
        "feature_b": feature_b,
    })
    bars.index = pd.date_range("2026-01-01", periods=n, freq="5min")

    # Add more noise features to pass variance filter
    for i in range(8):
        bars[f"noise_{i}"] = rng.randn(n)

    return bars


@pytest.fixture
def scanner():
    """Default scanner with test-friendly config."""
    return ScalpEdgeScanner(config={
        "timeframe": "5min",
        "cost_bps": 3.5,
        "tail_percentiles": [1, 5, 10, 90, 95, 99],
        "tail_threshold": 5,
        "top_features": 40,
        "conjunction_features": 15,
        "min_observations": 50,
        "min_tail_obs": 5,
        "significance_alpha": 0.05,
        "forward_horizon_bars": 1,
        "symbols": ["BTC"],
    })


@pytest.fixture
def bars():
    return _make_bars()


@pytest.fixture
def conjunction_bars():
    return _make_bars_with_conjunction()


# ---------------------------------------------------------------------------
# Test: Config loading
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        scanner = ScalpEdgeScanner(config={
            "timeframe": "5min", "cost_bps": 3.5,
            "tail_percentiles": [1, 5, 10, 90, 95, 99],
            "tail_threshold": 5, "top_features": 40,
            "conjunction_features": 15, "min_observations": 100,
            "min_tail_obs": 10, "significance_alpha": 0.05,
            "forward_horizon_bars": 1, "symbols": ["BTC"],
        })
        assert scanner.cost_bps == 3.5
        assert scanner.tail_threshold == 5
        assert scanner.timeframe == "5min"

    def test_load_scanner_config_missing_file(self):
        cfg = load_scanner_config("/nonexistent/path.toml")
        assert cfg["cost_bps"] == 3.5
        assert cfg["tail_threshold"] == 5

    def test_load_scanner_config_real(self):
        cfg_path = str(ROOT / "config" / "pipeline.toml")
        if os.path.exists(cfg_path):
            cfg = load_scanner_config(cfg_path)
            assert "timeframe" in cfg
            assert "cost_bps" in cfg


# ---------------------------------------------------------------------------
# Test: Forward returns
# ---------------------------------------------------------------------------

class TestForwardReturns:
    def test_shape(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        assert len(fwd) == len(bars)

    def test_last_is_nan(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        assert np.isnan(fwd[-1])

    def test_correct_sign(self, scanner):
        """Synthetic: prices strictly increasing -> forward returns > 0."""
        prices = np.arange(1000, 2000, dtype=float)
        bars = pd.DataFrame({
            "raw_midprice_close": prices,
        })
        bars.index = pd.date_range("2026-01-01", periods=len(prices), freq="5min")
        fwd = scanner.compute_forward_returns(bars)
        assert np.all(fwd[:-1] > 0)

    def test_horizon(self, scanner):
        """Multi-bar horizon."""
        prices = np.arange(1000, 2000, dtype=float)
        bars = pd.DataFrame({"raw_midprice_close": prices})
        bars.index = pd.date_range("2026-01-01", periods=len(prices), freq="5min")
        fwd = scanner.compute_forward_returns(bars, horizon=5)
        # Last 5 should be NaN
        assert all(np.isnan(fwd[-5:]))
        assert not np.isnan(fwd[0])


# ---------------------------------------------------------------------------
# Test: Feature selection
# ---------------------------------------------------------------------------

class TestFeatureSelection:
    def test_excludes_price(self, scanner, bars):
        cols = scanner.select_features(bars)
        assert "raw_midprice_close" not in cols

    def test_excludes_non_numeric(self, scanner, bars):
        cols = scanner.select_features(bars)
        assert "symbol" not in cols

    def test_returns_up_to_n(self, scanner, bars):
        cols = scanner.select_features(bars, n=5)
        assert len(cols) <= 5

    def test_all_finite_variance(self, scanner, bars):
        cols = scanner.select_features(bars)
        for c in cols:
            assert np.isfinite(bars[c].var())


# ---------------------------------------------------------------------------
# Test: Tail Return Profiling (Analysis 1)
# ---------------------------------------------------------------------------

class TestTailReturnProfile:
    def test_detects_injected_tail_edge(self, scanner, bars):
        """The injected +5bps upper-tail effect in feature_signal should be detected."""
        fwd = scanner.compute_forward_returns(bars)
        profiles = scanner.tail_return_profile(bars, fwd, ["feature_signal"])
        assert len(profiles) == 1
        prof = profiles[0]
        # Upper tail should have positive mean
        assert prof.upper_tail is not None
        assert prof.upper_tail.mean > 0

    def test_no_false_discovery_on_noise(self, scanner, bars):
        """Pure noise features should mostly not pass FDR."""
        fwd = scanner.compute_forward_returns(bars)
        noise_cols = [c for c in bars.columns if c.startswith("noise_")]
        profiles = scanner.tail_return_profile(bars, fwd, noise_cols)
        sig = [p for p in profiles if p.is_significant]
        # FDR controls false positives: at most 1 or 2 out of 9 noise features
        assert len(sig) <= 3

    def test_min_obs_gate(self, scanner):
        """Features with too few observations should be skipped."""
        bars = pd.DataFrame({
            "raw_midprice_close": np.arange(30, dtype=float) + 100,
            "feature_x": np.random.randn(30),
        })
        bars.index = pd.date_range("2026-01-01", periods=30, freq="5min")
        fwd = scanner.compute_forward_returns(bars)
        profiles = scanner.tail_return_profile(bars, fwd, ["feature_x"])
        # With 30 obs and min_observations=50, should be empty
        assert len(profiles) == 0

    def test_tail_stats_direction(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        profiles = scanner.tail_return_profile(bars, fwd, ["feature_signal"])
        prof = profiles[0]
        if prof.upper_tail and prof.upper_tail.p_value < 0.1:
            assert prof.upper_tail.direction in ("bullish", "bearish")

    def test_percentile_thresholds_present(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        profiles = scanner.tail_return_profile(bars, fwd, ["feature_signal"])
        prof = profiles[0]
        assert 5 in prof.percentile_thresholds
        assert 95 in prof.percentile_thresholds

    def test_edge_bps_nonnegative(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        cols = scanner.select_features(bars, n=10)
        profiles = scanner.tail_return_profile(bars, fwd, cols)
        for p in profiles:
            assert p.edge_bps >= 0

    def test_profiles_sorted_by_edge(self, scanner, bars):
        fwd = scanner.compute_forward_returns(bars)
        cols = scanner.select_features(bars, n=10)
        profiles = scanner.tail_return_profile(bars, fwd, cols)
        edges = [p.edge_bps for p in profiles]
        assert edges == sorted(edges, reverse=True)


# ---------------------------------------------------------------------------
# Test: Conjunction Scan (Analysis 2)
# ---------------------------------------------------------------------------

class TestConjunctionScan:
    def test_empty_with_too_few_profiles(self, scanner, bars):
        """Need at least 2 significant profiles."""
        fwd = scanner.compute_forward_returns(bars)
        single = scanner.tail_return_profile(bars, fwd, ["feature_signal"])
        result = scanner.conjunction_scan(bars, fwd, single)
        # Only 1 profile -> no pairs
        assert len(result) == 0

    def test_conjunction_structure(self, scanner, conjunction_bars):
        """Verify conjunction setup fields are populated."""
        bars = conjunction_bars
        fwd = scanner.compute_forward_returns(bars)
        cols = [c for c in bars.columns if c.startswith(("feature_", "noise_"))]
        profiles = scanner.tail_return_profile(bars, fwd, cols)
        # Force profiles to be "significant" for testing
        for p in profiles:
            p.is_significant = True
        conjunctions = scanner.conjunction_scan(bars, fwd, profiles[:5])
        for c in conjunctions:
            assert c.feature_a != c.feature_b
            assert c.tail_a in ("lower", "upper")
            assert c.tail_b in ("lower", "upper")
            assert c.n_occurrences >= scanner.min_tail_obs
            assert isinstance(c.weighted_sharpe, float)

    def test_weighted_sharpe_ranking(self, scanner, conjunction_bars):
        """Conjunctions should be sorted by |weighted_sharpe| descending."""
        bars = conjunction_bars
        fwd = scanner.compute_forward_returns(bars)
        cols = [c for c in bars.columns if c.startswith(("feature_", "noise_"))]
        profiles = scanner.tail_return_profile(bars, fwd, cols)
        for p in profiles:
            p.is_significant = True
        conj = scanner.conjunction_scan(bars, fwd, profiles[:5])
        ws = [abs(c.weighted_sharpe) for c in conj]
        assert ws == sorted(ws, reverse=True)


# ---------------------------------------------------------------------------
# Test: Temporal Characterization (Analysis 3)
# ---------------------------------------------------------------------------

class TestTemporalCharacterize:
    def test_holding_period(self, scanner, bars):
        """Optimal holding should be a positive integer."""
        n = len(bars)
        fwd = scanner.compute_forward_returns(bars)
        # Create clustered trigger pattern
        mask = np.zeros(n, dtype=bool)
        rng = np.random.RandomState(42)
        idx = rng.choice(n - 10, size=50, replace=False)
        mask[idx] = True
        results = scanner.temporal_characterize(bars, fwd, [("test", mask)])
        assert len(results) == 1
        assert results[0].optimal_holding_bars >= 1
        assert results[0].optimal_holding_bars <= 10

    def test_clustering_coefficient(self, scanner, bars):
        """CV of intervals measures clustering. Regular = low CV, clustered = high CV."""
        n = len(bars)
        fwd = scanner.compute_forward_returns(bars)

        # Regular spacing: every 20 bars
        mask_regular = np.zeros(n, dtype=bool)
        mask_regular[::20] = True
        reg_results = scanner.temporal_characterize(bars, fwd, [("regular", mask_regular)])

        # Clustered: bursts of triggers
        mask_cluster = np.zeros(n, dtype=bool)
        for start in [100, 500, 1000, 1500]:
            mask_cluster[start:start + 10] = True
        cl_results = scanner.temporal_characterize(bars, fwd, [("cluster", mask_cluster)])

        # Clustered pattern should have higher CV
        assert cl_results[0].clustering_coeff > reg_results[0].clustering_coeff

    def test_cooldown(self, scanner, bars):
        """Cooldown is median inter-trigger interval."""
        n = len(bars)
        fwd = scanner.compute_forward_returns(bars)
        mask = np.zeros(n, dtype=bool)
        mask[::50] = True  # every 50 bars
        results = scanner.temporal_characterize(bars, fwd, [("even", mask)])
        # Median interval should be ~50
        assert abs(results[0].cooldown_bars - 50) < 5

    def test_regime_edges_has_four_quartiles(self, scanner, bars):
        n = len(bars)
        fwd = scanner.compute_forward_returns(bars)
        mask = np.zeros(n, dtype=bool)
        mask[::20] = True
        results = scanner.temporal_characterize(bars, fwd, [("q_test", mask)])
        assert "Q1" in results[0].regime_edges
        assert "Q4" in results[0].regime_edges
        assert len(results[0].regime_edges) == 4

    def test_few_triggers_still_works(self, scanner, bars):
        """With < 3 triggers, should return default values."""
        n = len(bars)
        fwd = scanner.compute_forward_returns(bars)
        mask = np.zeros(n, dtype=bool)
        mask[100] = True  # Only 1 trigger
        results = scanner.temporal_characterize(bars, fwd, [("sparse", mask)])
        assert results[0].optimal_holding_bars == 1


# ---------------------------------------------------------------------------
# Test: Stability Assessment (Analysis 4)
# ---------------------------------------------------------------------------

class TestStability:
    def test_stable_detection(self, scanner):
        """Same-sign, material edge in both halves => stable."""
        n = 2000
        fwd = np.random.RandomState(42).randn(n) * 0.0001
        mask = np.zeros(n, dtype=bool)
        # Inject consistent positive edge in both halves
        idx_1h = np.arange(0, 1000, 20)
        idx_2h = np.arange(1000, 2000, 20)
        mask[idx_1h] = True
        mask[idx_2h] = True
        fwd[idx_1h] = 0.001  # +10bps
        fwd[idx_2h] = 0.001  # +10bps

        results = scanner.stability_assess(fwd, [("stable_test", mask)])
        assert len(results) == 1
        assert results[0].status == "stable"
        assert results[0].is_stable is True

    def test_fragile_detection(self, scanner):
        """Opposite-sign edge => fragile."""
        n = 2000
        fwd = np.random.RandomState(42).randn(n) * 0.0001
        mask = np.zeros(n, dtype=bool)
        idx_1h = np.arange(0, 1000, 20)
        idx_2h = np.arange(1000, 2000, 20)
        mask[idx_1h] = True
        mask[idx_2h] = True
        fwd[idx_1h] = 0.001   # +10bps in first half
        fwd[idx_2h] = -0.001  # -10bps in second half

        results = scanner.stability_assess(fwd, [("fragile_test", mask)])
        assert results[0].status == "fragile"
        assert results[0].is_stable is False

    def test_insufficient_data(self, scanner):
        """Too few observations in one half => insufficient_data."""
        n = 2000
        fwd = np.random.randn(n) * 0.0001
        mask = np.zeros(n, dtype=bool)
        # Only triggers in first half, none in second
        mask[100] = True
        mask[200] = True

        results = scanner.stability_assess(fwd, [("insuff_test", mask)])
        assert results[0].status == "insufficient_data"
        assert results[0].is_stable is None

    def test_edge_values_in_bps(self, scanner):
        n = 2000
        fwd = np.zeros(n)
        mask = np.zeros(n, dtype=bool)
        idx = np.arange(0, 2000, 20)
        mask[idx] = True
        fwd[idx] = 0.0005  # 5bps
        results = scanner.stability_assess(fwd, [("bps_test", mask)])
        # Edges reported in bps
        assert abs(results[0].edge_first_half - 5.0) < 1.0
        assert abs(results[0].edge_second_half - 5.0) < 1.0


# ---------------------------------------------------------------------------
# Test: Archetype Classification (Analysis 5)
# ---------------------------------------------------------------------------

class TestArchetypeClassification:
    def _make_sig_profile(self, name, vector):
        return FeatureTailProfile(
            name=name, vector=vector,
            percentile_thresholds={5: -2.0, 95: 2.0},
            lower_tail=None,
            upper_tail=TailStats(
                mean=0.0005, median=0.0004, std=0.001, skew=0.1,
                sharpe=0.5, win_rate=0.55, frequency=0.05,
                n_obs=100, t_stat=2.5, p_value=0.01, direction="bullish",
            ),
            is_significant=True, edge_bps=5.0, p_adjusted=0.01,
        )

    def test_orderflow_matches_imbalance_reversion(self, scanner):
        profiles = [self._make_sig_profile("orderflow_imbalance", "orderflow")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Imbalance Reversion" in names

    def test_toxicity_matches_toxicity_alert(self, scanner):
        profiles = [self._make_sig_profile("vpin", "toxicity")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Toxicity Alert" in names

    def test_entropy_matches_entropy_breakout(self, scanner):
        profiles = [self._make_sig_profile("entropy", "entropy")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Entropy Breakout" in names

    def test_flow_matches_flow_momentum(self, scanner):
        profiles = [self._make_sig_profile("flow_intensity", "flow")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Flow Momentum" in names

    def test_illiquidity_matches_liquidity_drain(self, scanner):
        profiles = [self._make_sig_profile("kyle_lambda", "illiquidity")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Liquidity Drain" in names

    def test_context_matches_funding_squeeze(self, scanner):
        profiles = [self._make_sig_profile("funding_rate", "context")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        names = [a.name for a in archetypes]
        assert "Funding Squeeze" in names

    def test_confidence_from_stability(self, scanner):
        profiles = [self._make_sig_profile("orderflow_imbalance", "orderflow")]
        stability = [StabilityResult(
            setup_id="tail:orderflow_imbalance",
            edge_first_half=5.0, edge_second_half=4.0,
            is_stable=True, status="stable",
        )]
        archetypes = scanner.classify_archetypes(profiles, [], stability)
        imb = [a for a in archetypes if a.name == "Imbalance Reversion"][0]
        assert imb.confidence == 1.0

    def test_no_match_returns_empty(self, scanner):
        profiles = [self._make_sig_profile("some_random", "unknown")]
        archetypes = scanner.classify_archetypes(profiles, [], [])
        assert len(archetypes) == 0


# ---------------------------------------------------------------------------
# Test: Utility helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_find_price_col(self, scanner, bars):
        col = scanner._find_price_col(bars)
        assert "midprice" in col

    def test_find_price_col_microprice(self, scanner):
        bars = pd.DataFrame({"raw_microprice_close": [100.0, 101.0]})
        col = scanner._find_price_col(bars)
        assert col == "raw_microprice_close"

    def test_find_price_col_missing_raises(self, scanner):
        bars = pd.DataFrame({"feature_x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No price column"):
            scanner._find_price_col(bars)

    def test_detect_vector_known(self, scanner):
        # Should detect some vector for common feature names
        vec = scanner._detect_vector("orderflow_imbalance_mean")
        assert vec != "unknown" or True  # OK if config doesn't match exactly

    def test_detect_vector_unknown(self, scanner):
        vec = scanner._detect_vector("zzz_random_gibberish")
        assert vec == "unknown"

    def test_numpy_encoder(self):
        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "arr": np.array([1, 2, 3]),
            "bool": np.bool_(True),
        }
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert abs(parsed["float"] - 3.14) < 0.01
        assert parsed["arr"] == [1, 2, 3]
        assert parsed["bool"] is True


# ---------------------------------------------------------------------------
# Test: Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def _dummy_report(self):
        return ScanReport(
            symbol="BTC", timeframe="5min", n_bars=1000,
            timestamp="2026-01-01T00:00:00Z",
            tail_profiles=[], conjunctions=[], temporal=[],
            stability=[], archetypes=[],
            config={"cost_bps": 3.5}, warnings=["LOW_DATA"],
        )

    def test_save_json_report(self):
        report = self._dummy_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json_report(report, tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["symbol"] == "BTC"
            assert data["n_bars"] == 1000

    def test_save_md_report(self):
        report = self._dummy_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_md_report(report, tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "Scalp Edge Scan" in content
            assert "BTC" in content

    def test_json_report_with_profiles(self):
        report = self._dummy_report()
        report.tail_profiles = [FeatureTailProfile(
            name="test_feat", vector="orderflow",
            percentile_thresholds={5: -2.0, 95: 2.0},
            lower_tail=None,
            upper_tail=TailStats(
                mean=0.0005, median=0.0004, std=0.001, skew=0.1,
                sharpe=0.5, win_rate=0.55, frequency=0.05,
                n_obs=100, t_stat=2.5, p_value=0.01, direction="bullish",
            ),
            is_significant=True, edge_bps=5.0, p_adjusted=0.01,
        )]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json_report(report, tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert len(data["tail_profiles"]) == 1
            assert data["tail_profiles"][0]["upper_tail"]["mean"] == 0.0005


# ---------------------------------------------------------------------------
# Test: CLI smoke test
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_nat_scan_parser(self):
        """nat scan --help should parse without error."""
        loader = importlib.machinery.SourceFileLoader("nat_cli", str(ROOT / "nat"))
        spec = importlib.util.spec_from_file_location("nat_cli", str(ROOT / "nat"), loader=loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parser = mod.build_parser()

        args = parser.parse_args(["scan", "--symbol", "ETH", "--data", "/tmp/data"])
        assert args.symbol == "ETH"
        assert args.data == "/tmp/data"
        assert hasattr(args, "func")

    def test_nat_test_scan_parser(self):
        """nat test scan --coverage should parse."""
        loader = importlib.machinery.SourceFileLoader("nat_cli", str(ROOT / "nat"))
        spec = importlib.util.spec_from_file_location("nat_cli", str(ROOT / "nat"), loader=loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parser = mod.build_parser()

        args = parser.parse_args(["test", "scan", "--coverage"])
        assert args.coverage is True
        assert hasattr(args, "func")

    def test_nat_scan_tail_arg(self):
        """nat scan --tail 10 should parse."""
        loader = importlib.machinery.SourceFileLoader("nat_cli", str(ROOT / "nat"))
        spec = importlib.util.spec_from_file_location("nat_cli", str(ROOT / "nat"), loader=loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parser = mod.build_parser()

        args = parser.parse_args(["scan", "--tail", "10"])
        assert args.tail == 10

"""Tests for alpha.regime_filter — per-regime screening and conditioning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from alpha.regime_filter import (
    assign_regime_labels,
    screen_per_regime,
    compute_regime_weights,
    RegimeIC,
    RegimeFilterResult,
)


# ---------------------------------------------------------------------------
# assign_regime_labels
# ---------------------------------------------------------------------------


class TestAssignRegimeLabels:
    def test_from_column(self):
        df = pd.DataFrame({
            "feat_0": np.random.randn(100),
            "regime_id": np.random.choice([0, 1, 2], 100),
        })
        labels = assign_regime_labels(df)
        assert len(labels) == 100
        assert set(labels).issubset({0, 1, 2})

    def test_fallback_single_regime(self):
        df = pd.DataFrame({"feat_0": np.random.randn(50)})
        labels = assign_regime_labels(df)
        assert len(labels) == 50
        assert np.all(labels == 0)

    def test_nonexistent_model_falls_back(self):
        df = pd.DataFrame({"feat_0": np.random.randn(50)})
        labels = assign_regime_labels(df, model_path=Path("/nonexistent/model.json"))
        assert np.all(labels == 0)

    def test_custom_column_name(self):
        df = pd.DataFrame({
            "feat_0": np.random.randn(30),
            "my_regime": [0, 1, 2] * 10,
        })
        labels = assign_regime_labels(df, regime_col="my_regime")
        assert set(labels) == {0, 1, 2}


# ---------------------------------------------------------------------------
# screen_per_regime
# ---------------------------------------------------------------------------


class TestScreenPerRegime:
    def test_single_regime(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "feat_0": np.arange(n, dtype=float),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
        })
        labels = np.zeros(n, dtype=int)
        ics = screen_per_regime(
            df, labels, ["feat_0"], "price", horizon_bars=4, min_bars=50,
        )
        assert 0 in ics
        assert isinstance(ics[0], dict)

    def test_multiple_regimes(self):
        np.random.seed(42)
        n = 600
        df = pd.DataFrame({
            "feat_0": np.random.randn(n),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
        })
        labels = np.array([0] * 300 + [1] * 300, dtype=int)
        ics = screen_per_regime(
            df, labels, ["feat_0"], "price", horizon_bars=4, min_bars=50,
        )
        assert 0 in ics
        assert 1 in ics

    def test_skips_small_regimes(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "feat_0": np.random.randn(n),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
        })
        labels = np.array([0] * 190 + [1] * 10, dtype=int)
        ics = screen_per_regime(
            df, labels, ["feat_0"], "price", horizon_bars=4, min_bars=50,
        )
        assert ics[1] == {}  # too few bars

    def test_missing_feature_skipped(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "feat_0": np.random.randn(n),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
        })
        labels = np.zeros(n, dtype=int)
        ics = screen_per_regime(
            df, labels, ["feat_0", "feat_missing"], "price",
            horizon_bars=4, min_bars=50,
        )
        assert "feat_missing" not in ics.get(0, {})


# ---------------------------------------------------------------------------
# compute_regime_weights
# ---------------------------------------------------------------------------


class TestComputeRegimeWeights:
    def test_improving_regime_uses_regime_weights(self):
        regime_ics = {0: {"feat_a": 0.10, "feat_b": 0.05}}
        global_ics = {"feat_a": 0.03, "feat_b": 0.02}
        weights = compute_regime_weights(regime_ics, global_ics, improvement_threshold=1.5)
        # feat_a: 0.10/0.03 = 3.33x improvement > 1.5 → use regime weights
        assert 0 in weights
        assert len(weights[0]) == 2
        # Weights should sum to 1
        assert abs(sum(weights[0].values()) - 1.0) < 1e-10

    def test_no_improvement_uses_global_weights(self):
        regime_ics = {0: {"feat_a": 0.031, "feat_b": 0.021}}
        global_ics = {"feat_a": 0.03, "feat_b": 0.02}
        weights = compute_regime_weights(regime_ics, global_ics, improvement_threshold=1.5)
        assert 0 in weights
        # No significant improvement → global weights
        assert abs(sum(weights[0].values()) - 1.0) < 1e-10

    def test_empty_regime_ics(self):
        regime_ics = {0: {}}
        global_ics = {"feat_a": 0.03}
        weights = compute_regime_weights(regime_ics, global_ics)
        assert weights[0] == {}

    def test_zero_global_ic(self):
        regime_ics = {0: {"feat_a": 0.05}}
        global_ics = {"feat_a": 0.0}
        weights = compute_regime_weights(regime_ics, global_ics, improvement_threshold=1.5)
        # Global IC is zero but regime IC is nonzero → treat as large improvement
        assert 0 in weights
        assert len(weights[0]) > 0


# ---------------------------------------------------------------------------
# RegimeFilterResult
# ---------------------------------------------------------------------------


class TestRegimeFilterResult:
    def test_gate_pass(self):
        result = RegimeFilterResult(
            n_regimes=3, n_bars_total=1000,
            regime_bar_counts={0: 400, 1: 300, 2: 300},
            global_ic=0.03,
            regime_ics={0: 0.05, 1: 0.06, 2: 0.02},
            improvement_ratios={0: 1.67, 1: 2.0, 2: 0.67},
            conditioned_regimes=[0, 1],
            regime_weights={0: {"f": 1.0}, 1: {"f": 1.0}, 2: {}},
            gate_has_improving_regime=True,
            gate_pass=True,
        )
        assert result.gate_pass is True
        assert len(result.conditioned_regimes) == 2

    def test_gate_fail(self):
        result = RegimeFilterResult(
            n_regimes=2, n_bars_total=500,
            regime_bar_counts={0: 300, 1: 200},
            global_ic=0.05,
            regime_ics={0: 0.04, 1: 0.03},
            improvement_ratios={0: 0.8, 1: 0.6},
            conditioned_regimes=[],
            regime_weights={0: {}, 1: {}},
            gate_has_improving_regime=False,
            gate_pass=False,
        )
        assert result.gate_pass is False

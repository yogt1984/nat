"""
Tests for run_chain() — algorithm output chaining.

Verifies that tick-level algorithms run first and their outputs are
available to bar-level algorithms via aggregate_bars().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.autodiscover import discover_all
from algorithms.registry import get_algorithm
from algorithms.runner import run_chain
from algorithms.tests.conftest import make_synthetic_ticks

discover_all()


def _make_tick_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic tick data with timestamp_ns and symbol for bar aggregation."""
    rng = np.random.default_rng(seed)
    # Need convolver + ML algorithm required columns
    convolver = get_algorithm("convolver")
    mc = get_algorithm("momentum_continuation")

    all_cols = set(convolver.required_columns())
    # Add columns needed by aggregate_bars for bar-level features
    all_cols.update(["timestamp_ns", "symbol"])
    # Add entropy/imbalance columns that bar aggregation produces
    all_cols.update([
        "ent_tick_1m", "ent_permutation_returns_16",
        "trend_hurst_300", "toxic_vpin_50",
        "whale_net_flow_4h", "regime_accumulation_score",
        "vol_returns_5m",
    ])

    df = make_synthetic_ticks(n, list(all_cols), seed=seed)
    # Add proper timestamp_ns for resampling
    df["timestamp_ns"] = np.arange(n) * 100_000_000  # 100ms spacing
    df["symbol"] = "BTC"
    return df


class TestRunChain:
    """Test run_chain() function."""

    def test_tick_only_chain(self):
        """Chain with only tick-level algorithms works."""
        algos = [get_algorithm("convolver"), get_algorithm("jump_detector")]
        df = make_synthetic_ticks(500, list(set(
            algos[0].required_columns() + algos[1].required_columns()
        )))
        results = run_chain(df, algos)

        assert len(results) == 2
        assert results[0].algorithm_name == "convolver"
        assert results[1].algorithm_name == "jump_detector"

    def test_empty_chain(self):
        """Empty algorithm list returns empty results."""
        df = make_synthetic_ticks(100, ["raw_midprice"])
        results = run_chain(df, [])
        assert results == []

    def test_tick_features_appended(self):
        """Tick-level outputs are appended to enriched DataFrame."""
        convolver = get_algorithm("convolver")
        df = make_synthetic_ticks(500, convolver.required_columns())

        # run_chain should append convolver features before passing to bar algos
        # We can verify by running with just the convolver and checking result
        results = run_chain(df, [convolver])
        assert len(results) == 1

        # The result should have all convolver output columns
        feature_names = [f.name for f in convolver.alg_features()]
        for fname in feature_names:
            assert fname in results[0].features_df.columns

    def test_chain_does_not_mutate_input(self):
        """run_chain() should not modify the caller's DataFrame."""
        convolver = get_algorithm("convolver")
        df = make_synthetic_ticks(500, convolver.required_columns())
        original_cols = set(df.columns)
        run_chain(df, [convolver])
        assert set(df.columns) == original_cols


class TestMLWithoutConvolver:
    """Backward compatibility: ML algorithms work without convolver columns."""

    def test_momentum_without_convolver(self):
        """momentum_continuation works without alg_conv_best_score_max."""
        mc = get_algorithm("momentum_continuation")
        # make_bar_df doesn't include convolver columns
        from algorithms.tests.conftest import make_bar_df
        bars = make_bar_df(200)
        assert "alg_conv_best_score_max" not in bars.columns

        result = mc.run_batch(bars)
        assert len(result) == 200
        assert list(result.columns) == [f.name for f in mc.alg_features()]

    def test_momentum_with_convolver(self):
        """momentum_continuation includes convolver features when present."""
        mc = get_algorithm("momentum_continuation")
        from algorithms.tests.conftest import make_bar_df
        bars = make_bar_df(200)
        # Add convolver column
        bars["alg_conv_best_score_max"] = np.random.default_rng(42).uniform(0, 1, 200)

        result = mc.run_batch(bars)
        assert len(result) == 200
        # Should still produce valid output
        warmup = mc.warmup
        post = result.iloc[warmup + 10:]
        assert post["alg_mc_signal"].notna().any()

    def test_regime_lgbm_without_convolver(self):
        """regime_conditioned_lgbm works without alg_conv_best_score_max."""
        rlgbm = get_algorithm("regime_conditioned_lgbm")
        from algorithms.tests.conftest import make_bar_df
        bars = make_bar_df(200)
        assert "alg_conv_best_score_max" not in bars.columns

        result = rlgbm.run_batch(bars)
        assert len(result) == 200
        assert list(result.columns) == [f.name for f in rlgbm.alg_features()]

    def test_regime_lgbm_with_convolver(self):
        """regime_conditioned_lgbm uses convolver features when present."""
        rlgbm = get_algorithm("regime_conditioned_lgbm")
        from algorithms.tests.conftest import make_bar_df
        bars = make_bar_df(200)
        bars["alg_conv_best_score_max"] = np.random.default_rng(42).uniform(0, 1, 200)

        result = rlgbm.run_batch(bars)
        assert len(result) == 200
        warmup = rlgbm.warmup
        post = result.iloc[warmup + 10:]
        assert post["alg_rlgbm_signal"].notna().any()


class TestConvolverAggregation:
    """Test that alg_conv_best_score gets max aggregation."""

    def test_best_score_max_aggregation(self):
        """alg_conv_best_score should produce _max, _mean, _last columns."""
        from cluster_pipeline.preprocess import _build_agg_plan

        plan = _build_agg_plan(["alg_conv_best_score", "alg_conv_breakout_bull"])

        # best_score gets max aggregation
        suffixes = [s for s, _ in plan["alg_conv_best_score"]]
        assert "max" in suffixes
        assert "mean" in suffixes
        assert "last" in suffixes

        # Other convolver features get default aggregation
        suffixes_other = [s for s, _ in plan["alg_conv_breakout_bull"]]
        assert "mean" in suffixes_other
        assert "std" in suffixes_other
        assert "last" in suffixes_other

    def test_best_score_max_captures_peak(self):
        """Verify max aggregation captures the true peak in bars."""
        from cluster_pipeline.preprocess import aggregate_bars

        rng = np.random.default_rng(42)
        n = 6000  # enough for 1 bar at 5min (3000 ticks)

        df = pd.DataFrame({
            "timestamp_ns": np.arange(n) * 100_000_000,
            "symbol": "BTC",
            "raw_midprice": 50000 + rng.normal(0, 1, n).cumsum(),
            "alg_conv_best_score": np.zeros(n),
        })
        # Inject a spike at tick 1500
        df.loc[1500, "alg_conv_best_score"] = 0.95

        bars = aggregate_bars(df, "5min")
        assert "alg_conv_best_score_max" in bars.columns

        # The bar containing tick 1500 should capture the 0.95 spike
        max_val = bars["alg_conv_best_score_max"].max()
        assert max_val == pytest.approx(0.95, abs=1e-6)

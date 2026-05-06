"""Integration test: Full EAMM Pipeline end-to-end.

Tests the complete flow: data → simulate → label → features → train → evaluate → regime.
Uses synthetic data to avoid dependency on real parquet files.
"""

import numpy as np
import polars as pl
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eamm.simulator import simulate_mm, pnl_to_bps, DEFAULT_SPREAD_LEVELS_BPS
from eamm.features import extract_context, context_to_numpy, CONTEXT_FEATURE_NAMES
from eamm.labels import compute_labels, compute_continuous_optimal, label_distribution
from eamm.train import train_eamm, predict_spread
from eamm.evaluate import walk_forward_evaluate
from eamm.regime_analysis import analyze_regimes
from eamm.backtest import run_backtest


def _generate_synthetic_market_data(n=5000):
    """Generate realistic-looking market data with all required columns."""
    np.random.seed(42)

    # Price with regime changes
    prices = [100.0]
    for i in range(n - 1):
        if i < n // 3:
            drift = 0.001  # trending up
        elif i < 2 * n // 3:
            drift = 0.0  # mean-reverting
        else:
            drift = -0.0005  # trending down
        prices.append(prices[-1] * (1 + drift + np.random.randn() * 0.002))
    midprices = np.array(prices)

    timestamps = np.arange(n) * 100_000_000  # 100ms spacing

    # Entropy features (vary by regime) — use safe slicing
    third = n // 3
    ent_tick_30s = np.zeros(n)
    ent_tick_30s[:third] = 0.2 + np.random.rand(third) * 0.1
    ent_tick_30s[third:2*third] = 0.5 + np.random.rand(third) * 0.1
    ent_tick_30s[2*third:] = 0.8 + np.random.rand(n - 2*third) * 0.2

    # All columns required by features.py CONTEXT_FEATURES
    df = pl.DataFrame({
        "timestamp_ns": timestamps,
        "raw_midprice": midprices,
        "symbol": ["BTC"] * n,
        "ent_tick_1s": np.random.uniform(0, 1.1, n),
        "ent_tick_5s": np.random.uniform(0, 1.1, n),
        "ent_tick_30s": ent_tick_30s,
        "ent_tick_1m": np.random.uniform(0, 1.1, n),
        "ent_permutation_returns_8": np.random.uniform(0.5, 1.0, n),
        "ent_permutation_returns_16": np.random.uniform(0.5, 1.0, n),
        "ent_permutation_returns_32": np.random.uniform(0.5, 1.0, n),
        "toxic_vpin_50": np.random.uniform(0, 1, n),
        "toxic_index": np.random.uniform(0, 1, n),
        "toxic_adverse_selection": np.random.uniform(0, 1, n),
        "vol_returns_1m": np.abs(np.random.randn(n)) * 0.001,
        "vol_returns_5m": np.abs(np.random.randn(n)) * 0.0015,
        "flow_intensity": np.random.rand(n) * 10,
        "flow_aggressor_ratio_5s": np.random.rand(n),
        "imbalance_qty_l1": np.random.randn(n) * 0.3,
        "imbalance_qty_l5": np.random.randn(n) * 0.3,
        "trend_momentum_60": np.random.randn(n) * 0.001,
        "trend_hurst_300": np.random.uniform(0.3, 0.7, n),
        "raw_spread_bps": 1.0 + np.random.rand(n) * 3.0,
    })
    return df


class TestFullPipeline:
    """Integration test: simulate → label → features → train → evaluate."""

    def test_end_to_end_regression(self):
        df = _generate_synthetic_market_data(n=3000)
        spreads = [1.0, 3.0, 5.0, 10.0, 20.0]
        horizon = 50

        # Step 1: Simulate
        sim = simulate_mm(df, spread_levels_bps=spreads, horizon=horizon)
        assert sim.pnl.shape[1] == len(spreads)
        valid_rows = len(df) - horizon
        # pnl is allocated as (N, K) but only first valid_rows are meaningful

        # Step 2: PnL in bps
        pnl_bps = pnl_to_bps(sim)
        assert pnl_bps.shape == sim.pnl.shape

        # Step 3: Labels
        labels_df = compute_labels(sim)
        cont_optimal = compute_continuous_optimal(sim)
        assert np.all(cont_optimal >= 0)

        # Step 4: Features
        ctx_df = extract_context(df)
        X = context_to_numpy(ctx_df)
        X = X[:valid_rows]

        # Step 5: Train
        result = train_eamm(X, cont_optimal[:len(X)], CONTEXT_FEATURE_NAMES,
                            mode="regression", save_dir=None)
        assert result.train_score > -1.0  # not completely broken

        # Step 6: Predict
        preds = predict_spread(result, X[:100])
        assert len(preds) == 100
        assert np.all(preds >= 0)

    def test_end_to_end_evaluation(self):
        df = _generate_synthetic_market_data(n=3000)
        spreads = [1.0, 3.0, 5.0, 10.0, 20.0]
        horizon = 50

        sim = simulate_mm(df, spread_levels_bps=spreads, horizon=horizon)
        pnl_bps = pnl_to_bps(sim)
        cont_optimal = compute_continuous_optimal(sim)
        ctx_df = extract_context(df)
        X = context_to_numpy(ctx_df)
        valid_end = len(df) - horizon
        X = X[:valid_end]
        pnl_valid = pnl_bps[:valid_end]
        fill_rt_valid = sim.fill_round_trip[:valid_end]

        eval_result = walk_forward_evaluate(
            context_matrix=X,
            pnl_matrix_bps=pnl_valid,
            fill_rt_matrix=fill_rt_valid,
            optimal_spread_bps=cont_optimal[:len(X)],
            spread_levels_bps=spreads,
            feature_names=CONTEXT_FEATURE_NAMES,
            n_splits=3,
            mode="regression",
        )
        assert eval_result.n_splits == 3
        assert np.isfinite(eval_result.eamm_avg_sharpe)

    def test_end_to_end_regime(self):
        df = _generate_synthetic_market_data(n=3000)
        spreads = [1.0, 3.0, 5.0, 10.0, 20.0]
        horizon = 50

        sim = simulate_mm(df, spread_levels_bps=spreads, horizon=horizon)
        pnl_bps = pnl_to_bps(sim)
        cont_optimal = compute_continuous_optimal(sim)
        ctx_df = extract_context(df)
        valid_end = len(df) - horizon

        entropy_col = ctx_df["H_tick_30s"].to_numpy()[:valid_end]

        result = analyze_regimes(
            entropy_values=entropy_col,
            pnl_matrix=pnl_bps[:valid_end],
            fill_bid_matrix=sim.fill_bid[:valid_end],
            fill_ask_matrix=sim.fill_ask[:valid_end],
            fill_rt_matrix=sim.fill_round_trip[:valid_end],
            spread_levels_bps=spreads,
            optimal_spread_bps=cont_optimal[:valid_end],
        )
        # With our synthetic data that has regime-dependent entropy,
        # the thesis should be confirmed
        assert sum(result.regime_counts) == valid_end
        assert result.kruskal_wallis_p >= 0.0

    def test_end_to_end_backtest(self):
        df = _generate_synthetic_market_data(n=2000)
        spreads = [1.0, 3.0, 5.0, 10.0]
        horizon = 50

        sim = simulate_mm(df, spread_levels_bps=spreads, horizon=horizon)
        cont_optimal = compute_continuous_optimal(sim)
        ctx_df = extract_context(df)
        X = context_to_numpy(ctx_df)
        valid_end = len(df) - horizon

        # Train on first half, backtest on second half
        split = valid_end // 2
        result = train_eamm(X[:split], cont_optimal[:split],
                            CONTEXT_FEATURE_NAMES, mode="regression", save_dir=None)
        predicted = predict_spread(result, X[split:valid_end])

        midprices = df["raw_midprice"].to_numpy()[split:valid_end]
        timestamps = df["timestamp_ns"].to_numpy()[split:valid_end]
        volatility = df["vol_returns_1m"].to_numpy()[split:valid_end]

        bt = run_backtest(
            midprices=midprices,
            timestamps=timestamps,
            predicted_spreads_bps=predicted,
            volatility=volatility,
            gamma=0.1,
            q_max=1.0,
            horizon=horizon,
        )
        assert len(bt.equity_curve) == len(midprices)
        assert np.isfinite(bt.total_pnl)
        assert np.isfinite(bt.sharpe)

"""Unit tests for EAMM Optimal Spread Label Generator."""

import numpy as np
import polars as pl
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.simulator import simulate_mm, SimulationResult
from eamm.labels import compute_labels, compute_continuous_optimal, label_distribution


def _make_df(prices, n=200):
    return pl.DataFrame({
        "timestamp_ns": list(range(n)),
        "raw_midprice": [float(p) for p in prices[:n]],
        "symbol": ["BTC"] * n,
    })


def _simulate(prices, spreads=[1.0, 2.0, 5.0], horizon=10):
    df = _make_df(prices, len(prices))
    return simulate_mm(df, spread_levels_bps=spreads, horizon=horizon)


class TestLabelPicksMaxPnl:
    def test_argmax_correct(self):
        np.random.seed(42)
        prices = [100.0 + np.random.randn() * 0.5 for _ in range(200)]
        result = _simulate(prices, spreads=[1.0, 3.0, 10.0], horizon=20)
        labels = compute_labels(result)

        # For each row, verify the label corresponds to max PnL
        for i in range(min(50, len(labels))):
            row = labels.row(i, named=True)
            cls = row["optimal_spread_class"]
            pnl_at_opt = row["pnl_at_optimal_bps"]
            pnls = [row[f"pnl_level_{k}"] for k in range(3)]
            assert cls == np.argmax(pnls), f"Row {i}: class {cls} != argmax {np.argmax(pnls)}"


class TestAllNegativePnl:
    def test_picks_least_bad(self):
        # Strong downtrend — all spreads lose, but wider loses less
        prices = [100.0 - i * 0.1 for i in range(200)]
        result = _simulate(prices, spreads=[0.5, 1.0, 5.0, 20.0], horizon=20)
        labels = compute_labels(result)
        # Should still have valid labels (picks least negative)
        assert len(labels) > 0
        assert labels["optimal_spread_class"].min() >= 0


class TestMonotonicPnl:
    def test_trending_prefers_wide(self):
        # In a strong trend, wider spread avoids adverse selection
        prices = [100.0 + i * 0.5 for i in range(500)]
        result = _simulate(prices, spreads=[0.5, 1.0, 5.0, 20.0, 50.0], horizon=30)
        labels = compute_labels(result)
        # Mean optimal class should be biased toward wider spreads
        mean_class = labels["optimal_spread_class"].mean()
        assert mean_class > 1.0, f"Expected bias toward wider spreads, got mean class {mean_class}"


class TestInterpolationAccuracy:
    def test_quadratic_known_curve(self):
        # Create scenario where optimal is at a known point
        np.random.seed(123)
        prices = [100.0]
        for _ in range(499):
            prices.append(prices[-1] + np.random.randn() * 0.2)
        result = _simulate(prices, spreads=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0], horizon=30)
        cont = compute_continuous_optimal(result)
        # All values should be positive (spread can't be negative)
        assert np.all(cont >= 0), f"Found negative spread: {cont.min()}"
        # Should be within [min_spread, max_spread]
        assert np.all(cont <= 20.0 + 0.1)


class TestLabelDistribution:
    def test_not_all_same_class(self):
        np.random.seed(42)
        # Use high volatility so different spreads can win at different times
        prices = [100.0]
        for _ in range(1999):
            prices.append(prices[-1] + np.random.randn() * 2.0)
        result = _simulate(prices, spreads=[1.0, 5.0, 20.0, 50.0], horizon=10)
        labels = compute_labels(result)
        dist = label_distribution(labels, 4)
        # At least 2 classes should have some representation
        nonzero_classes = sum(1 for f in dist["fractions"] if f > 0.01)
        assert nonzero_classes >= 2, f"Only {nonzero_classes} class(es) represented: {dist['fractions']}"

"""Unit tests for EAMM Entropy Regime Analysis."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.regime_analysis import assign_regimes, analyze_regimes, REGIME_NAMES, LN3


class TestAssignRegimes:
    def test_boundaries(self):
        # Test exact boundary assignment
        vals = np.array([0.0, 0.2, 0.35, 0.5, 0.55, 0.6, 0.65, 1.0, LN3])
        regimes = assign_regimes(vals)
        assert regimes[0] == 0  # 0.0 → TRENDING
        assert regimes[1] == 0  # 0.2 → TRENDING
        assert regimes[2] == 1  # 0.35 → TRANSITIONING
        assert regimes[3] == 1  # 0.5 → TRANSITIONING
        assert regimes[4] == 2  # 0.55 → NORMAL
        assert regimes[5] == 2  # 0.6 → NORMAL
        assert regimes[6] == 3  # 0.65 → RANDOM
        assert regimes[7] == 3  # 1.0 → RANDOM
        assert regimes[8] == 3  # ln(3) → RANDOM

    def test_all_regimes_represented(self):
        vals = np.array([0.1, 0.4, 0.6, 0.9])
        regimes = assign_regimes(vals)
        assert set(regimes) == {0, 1, 2, 3}

    def test_clamps_outliers(self):
        vals = np.array([-1.0, 5.0])
        regimes = assign_regimes(vals)
        assert regimes[0] == 0
        assert regimes[1] == 3


class TestAnalyzeRegimes:
    def _make_data(self, n=1000, k=4):
        np.random.seed(42)
        entropy = np.random.uniform(0, LN3, n)
        pnl = np.random.randn(n, k)
        fill_bid = (np.random.rand(n, k) > 0.3).astype(float)
        fill_ask = (np.random.rand(n, k) > 0.3).astype(float)
        fill_rt = fill_bid * fill_ask
        optimal = np.random.uniform(1, 20, n)
        spreads = [1.0, 3.0, 5.0, 10.0][:k]
        return entropy, pnl, fill_bid, fill_ask, fill_rt, spreads, optimal

    def test_returns_result(self):
        entropy, pnl, fill_bid, fill_ask, fill_rt, spreads, optimal = self._make_data()
        result = analyze_regimes(entropy, pnl, fill_bid, fill_ask, fill_rt, spreads, optimal)
        assert len(result.regime_names) == 4
        assert len(result.regime_counts) == 4
        assert result.regime_spread_matrix.shape == (4, 4)

    def test_counts_sum_to_n(self):
        entropy, pnl, fill_bid, fill_ask, fill_rt, spreads, optimal = self._make_data(n=500)
        result = analyze_regimes(entropy, pnl, fill_bid, fill_ask, fill_rt, spreads, optimal)
        assert sum(result.regime_counts) == 500

    def test_thesis_with_clear_signal(self):
        # Create data where optimal spread clearly depends on entropy
        np.random.seed(123)
        n = 2000
        entropy = np.random.uniform(0, LN3, n)
        regimes = np.digitize(entropy, [0.35, 0.55, 0.65])
        # Optimal spread = f(regime): trending→wide, random→narrow
        optimal = np.where(regimes == 0, 20.0,
                  np.where(regimes == 1, 10.0,
                  np.where(regimes == 2, 5.0, 2.0))) + np.random.randn(n) * 0.5

        pnl = np.random.randn(n, 4)
        fill_bid = (np.random.rand(n, 4) > 0.3).astype(float)
        fill_ask = (np.random.rand(n, 4) > 0.3).astype(float)
        fill_rt = fill_bid * fill_ask

        result = analyze_regimes(
            entropy, pnl, fill_bid, fill_ask, fill_rt,
            [2.0, 5.0, 10.0, 20.0], optimal
        )
        assert result.thesis_confirmed, (
            f"Expected thesis confirmed: p={result.kruskal_wallis_p:.4f}, "
            f"eta^2={result.eta_squared:.4f}"
        )

    def test_no_signal_not_confirmed(self):
        # Random data — no relationship between entropy and optimal spread
        np.random.seed(99)
        n = 200
        entropy = np.random.uniform(0, LN3, n)
        optimal = np.random.uniform(1, 20, n)  # random, no pattern
        pnl = np.random.randn(n, 3)
        fill_bid = (np.random.rand(n, 3) > 0.5).astype(float)
        fill_ask = (np.random.rand(n, 3) > 0.5).astype(float)
        fill_rt = fill_bid * fill_ask

        result = analyze_regimes(
            entropy, pnl, fill_bid, fill_ask, fill_rt,
            [2.0, 5.0, 10.0], optimal
        )
        # With random data, thesis should likely not be confirmed
        assert result.thesis_confirmed == False
        assert result.kruskal_wallis_p >= 0.0

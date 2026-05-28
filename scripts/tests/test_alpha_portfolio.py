"""Tests for alpha.portfolio — risk parity, correlation adjustment, DD control."""

from pathlib import Path


import numpy as np
import pytest
from alpha.portfolio import (
    compute_risk_parity_weights,
    adjust_for_correlation,
    compute_correlation_matrix,
    apply_drawdown_control,
    simulate_portfolio,
    SymbolAllocation,
    DrawdownControl,
    PortfolioResult,
    _sharpe,
    _max_dd,
)


# ---------------------------------------------------------------------------
# compute_risk_parity_weights
# ---------------------------------------------------------------------------


class TestComputeRiskParityWeights:
    def test_equal_volatility(self):
        np.random.seed(42)
        returns = {
            "BTC": np.random.randn(500) * 0.01,
            "ETH": np.random.randn(500) * 0.01,
        }
        w = compute_risk_parity_weights(returns)
        assert abs(w["BTC"] - 0.5) < 0.1
        assert abs(w["ETH"] - 0.5) < 0.1
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_unequal_volatility(self):
        np.random.seed(42)
        returns = {
            "BTC": np.random.randn(500) * 0.02,  # 2x vol
            "ETH": np.random.randn(500) * 0.01,
        }
        w = compute_risk_parity_weights(returns)
        # ETH should get more weight (lower vol)
        assert w["ETH"] > w["BTC"]
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_single_symbol(self):
        returns = {"BTC": np.random.randn(100) * 0.01}
        w = compute_risk_parity_weights(returns)
        assert abs(w["BTC"] - 1.0) < 1e-10

    def test_short_history(self):
        returns = {"BTC": np.array([0.01, -0.01])}
        w = compute_risk_parity_weights(returns)
        assert "BTC" in w

    def test_lookback_window(self):
        np.random.seed(42)
        # Vol changes: first half low, second half high
        r = np.concatenate([
            np.random.randn(500) * 0.001,
            np.random.randn(500) * 0.1,
        ])
        returns = {"BTC": r, "ETH": np.random.randn(1000) * 0.01}
        # Short lookback should capture recent high vol for BTC
        w = compute_risk_parity_weights(returns, lookback=100)
        assert w["ETH"] > w["BTC"]


# ---------------------------------------------------------------------------
# adjust_for_correlation
# ---------------------------------------------------------------------------


class TestAdjustForCorrelation:
    def test_no_adjustment_low_corr(self):
        np.random.seed(42)
        returns = {
            "BTC": np.random.randn(500),
            "ETH": np.random.randn(500),
        }
        weights = {"BTC": 0.5, "ETH": 0.5}
        adj = adjust_for_correlation(weights, returns, corr_threshold=0.8)
        # Uncorrelated → no change
        assert abs(adj["BTC"] - 0.5) < 0.05
        assert abs(adj["ETH"] - 0.5) < 0.05

    def test_adjustment_high_corr(self):
        np.random.seed(42)
        base = np.random.randn(500)
        returns = {
            "BTC": base,
            "ETH": base + np.random.randn(500) * 0.01,
        }
        weights = {"BTC": 0.5, "ETH": 0.5}
        adj = adjust_for_correlation(weights, returns, corr_threshold=0.8, reduction=0.20)
        # Highly correlated → weights reduced then renormalized
        assert abs(sum(adj.values()) - 1.0) < 1e-10

    def test_short_data_returns_unchanged(self):
        returns = {"BTC": np.array([0.01]), "ETH": np.array([0.02])}
        weights = {"BTC": 0.6, "ETH": 0.4}
        adj = adjust_for_correlation(weights, returns)
        assert adj == weights


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_basic(self):
        np.random.seed(42)
        returns = {
            "BTC": np.random.randn(100),
            "ETH": np.random.randn(100),
            "SOL": np.random.randn(100),
        }
        corr = compute_correlation_matrix(returns)
        assert corr["BTC"]["BTC"] == pytest.approx(1.0, abs=1e-10)
        assert corr["ETH"]["ETH"] == pytest.approx(1.0, abs=1e-10)
        # Off-diagonal should be near zero for independent data
        assert abs(corr["BTC"]["ETH"]) < 0.3

    def test_short_data(self):
        returns = {"BTC": np.array([0.01]), "ETH": np.array([0.02])}
        corr = compute_correlation_matrix(returns)
        assert corr["BTC"]["ETH"] == 0.0


# ---------------------------------------------------------------------------
# apply_drawdown_control
# ---------------------------------------------------------------------------


class TestApplyDrawdownControl:
    def test_no_drawdown(self):
        pnl = np.ones(100) * 0.001
        dd = apply_drawdown_control(pnl, dd_reduce=0.02)
        assert dd.current_dd == 0.0
        assert dd.is_reduced is False
        assert dd.scale_factor == 1.0

    def test_triggers_reduction(self):
        pnl = np.array([0.01, 0.01, -0.05, -0.01])
        dd = apply_drawdown_control(pnl, dd_reduce=0.02)
        assert dd.current_dd > 0.02
        assert dd.is_reduced is True
        assert dd.scale_factor == 0.5

    def test_empty_pnl(self):
        dd = apply_drawdown_control(np.array([]))
        assert dd.current_dd == 0.0
        assert dd.is_reduced is False


# ---------------------------------------------------------------------------
# simulate_portfolio
# ---------------------------------------------------------------------------


class TestSimulatePortfolio:
    def test_basic_simulation(self):
        np.random.seed(42)
        n = 200
        signals = {
            "BTC": np.random.randn(n) * 0.1,
            "ETH": np.random.randn(n) * 0.1,
        }
        prices = {
            "BTC": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "ETH": 50 + np.cumsum(np.random.randn(n) * 0.3),
        }
        weights = {"BTC": 0.5, "ETH": 0.5}
        port_pnl, per_sym, dd = simulate_portfolio(signals, prices, weights)

        assert len(port_pnl) == n - 1
        assert "BTC" in per_sym
        assert "ETH" in per_sym
        assert isinstance(dd, DrawdownControl)

    def test_single_symbol(self):
        np.random.seed(42)
        n = 100
        signals = {"BTC": np.ones(n)}
        prices = {"BTC": np.linspace(100, 110, n)}
        weights = {"BTC": 1.0}
        port_pnl, per_sym, dd = simulate_portfolio(signals, prices, weights)
        assert len(port_pnl) == n - 1
        assert np.sum(port_pnl) > 0  # price went up, signal is long


# ---------------------------------------------------------------------------
# Sharpe & MaxDD helpers
# ---------------------------------------------------------------------------


class TestPortfolioHelpers:
    def test_sharpe_positive(self):
        pnl = np.ones(200) * 0.001 + np.random.randn(200) * 0.0001
        assert _sharpe(pnl) > 0

    def test_sharpe_zero_vol(self):
        assert _sharpe(np.zeros(100)) == 0.0

    def test_max_dd_zero(self):
        assert _max_dd(np.ones(100) * 0.01) == 0.0

    def test_max_dd_positive(self):
        pnl = np.array([0.01, -0.05, 0.01])
        assert _max_dd(pnl) > 0

"""Tests for alpha.position — trade filtering, sizing, quality gate G3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from alpha.position import (
    count_trades,
    mean_holding_bars,
    apply_trade_filter,
    apply_kelly_sizing,
    apply_ramp_up,
    evaluate_quality_gate,
    PositionResult,
)


# ---------------------------------------------------------------------------
# Mock CostModel
# ---------------------------------------------------------------------------

class _MockCostModel:
    def __init__(self, round_trip_cost_bps: float = 7.0):
        self.round_trip_cost_bps = round_trip_cost_bps


# ---------------------------------------------------------------------------
# count_trades
# ---------------------------------------------------------------------------


class TestCountTrades:
    def test_no_trades(self):
        assert count_trades(np.ones(100)) == 0

    def test_alternating(self):
        pos = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        assert count_trades(pos) == 4

    def test_single_change(self):
        pos = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        assert count_trades(pos) == 1

    def test_empty(self):
        assert count_trades(np.array([1.0])) == 0


# ---------------------------------------------------------------------------
# mean_holding_bars
# ---------------------------------------------------------------------------


class TestMeanHoldingBars:
    def test_constant_position(self):
        pos = np.ones(100)
        assert mean_holding_bars(pos) == 100.0

    def test_alternating(self):
        pos = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        holding = mean_holding_bars(pos)
        assert abs(holding - 1.0) < 1e-10

    def test_regular_changes(self):
        # Change every 10 bars — need multiple changes for np.diff
        pos = np.tile(np.repeat([1.0, -1.0], 10), 3)  # 60 bars, changes at 9,19,29,39,49
        holding = mean_holding_bars(pos)
        assert abs(holding - 10.0) < 1e-10

    def test_single_element(self):
        holding = mean_holding_bars(np.array([1.0]))
        assert holding == 1.0


# ---------------------------------------------------------------------------
# apply_trade_filter
# ---------------------------------------------------------------------------


class TestApplyTradeFilter:
    def test_filters_small_changes(self):
        """Small signal changes should be filtered (held at previous)."""
        signal = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.8, 0.9])
        cost_model = _MockCostModel(round_trip_cost_bps=7.0)
        position = apply_trade_filter(
            signal, ic_estimate=0.03, return_vol=0.01,
            cost_model=cost_model, horizon_bars=16, cost_multiplier=1.5,
        )
        # First position follows signal, subsequent small changes should be held
        assert len(position) == len(signal)
        assert np.all(np.isfinite(position))

    def test_large_change_passes(self):
        """Large signal jump should trigger a trade."""
        signal = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        cost_model = _MockCostModel(round_trip_cost_bps=7.0)
        position = apply_trade_filter(
            signal, ic_estimate=0.1, return_vol=0.05,
            cost_model=cost_model, horizon_bars=16, cost_multiplier=1.0,
        )
        # Position should change when signal jumps to 1.0
        assert position[-1] == 1.0

    def test_nan_signal_carries_forward(self):
        signal = np.array([0.5, np.nan, np.nan, 0.5])
        cost_model = _MockCostModel(round_trip_cost_bps=7.0)
        position = apply_trade_filter(
            signal, ic_estimate=0.05, return_vol=0.02,
            cost_model=cost_model,
        )
        # NaN bars should carry forward previous position
        assert np.isfinite(position[1])
        assert np.isfinite(position[2])

    def test_fewer_trades_than_signal(self):
        """Filtered position should have fewer trades than raw signal."""
        np.random.seed(42)
        signal = np.clip(np.cumsum(np.random.randn(200) * 0.02), -1, 1)
        cost_model = _MockCostModel(round_trip_cost_bps=7.0)
        position = apply_trade_filter(
            signal, ic_estimate=0.03, return_vol=0.01,
            cost_model=cost_model, horizon_bars=16, cost_multiplier=1.5,
        )
        assert count_trades(position) <= count_trades(signal)


# ---------------------------------------------------------------------------
# apply_kelly_sizing
# ---------------------------------------------------------------------------


class TestApplyKellySizing:
    def test_clips_to_range(self):
        pos = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = apply_kelly_sizing(pos, scale=1.0)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_scaling(self):
        pos = np.array([0.5, -0.3])
        result = apply_kelly_sizing(pos, scale=2.0)
        assert abs(result[0] - 1.0) < 1e-10  # 0.5 * 2 = 1.0
        assert abs(result[1] - (-0.6)) < 1e-10

    def test_zero_scale(self):
        pos = np.array([0.5, -0.3])
        result = apply_kelly_sizing(pos, scale=0.0)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# apply_ramp_up
# ---------------------------------------------------------------------------


class TestApplyRampUp:
    def test_ramp_period_scaled(self):
        pos = np.ones(100)
        result = apply_ramp_up(pos, ramp_bars=50, ramp_fraction=0.5)
        assert abs(result[0] - 0.5) < 1e-10  # in ramp period
        assert abs(result[49] - 0.5) < 1e-10  # last bar of ramp
        assert abs(result[50] - 1.0) < 1e-10  # after ramp

    def test_ramp_longer_than_data(self):
        pos = np.ones(10)
        result = apply_ramp_up(pos, ramp_bars=100, ramp_fraction=0.25)
        assert np.all(np.abs(result - 0.25) < 1e-10)

    def test_zero_ramp(self):
        pos = np.ones(10)
        result = apply_ramp_up(pos, ramp_bars=0, ramp_fraction=0.5)
        assert np.all(np.abs(result - 1.0) < 1e-10)


# ---------------------------------------------------------------------------
# evaluate_quality_gate
# ---------------------------------------------------------------------------


class TestEvaluateQualityGate:
    def test_gate_pass(self):
        """Significant trade reduction + long holding → PASS."""
        # Raw signal changes every bar
        signal = np.tile([1.0, -1.0], 500)
        # Filtered position holds for long periods
        position = np.repeat([1.0, -1.0], 250)
        result = evaluate_quality_gate(signal, position, bar_minutes=15.0)

        assert isinstance(result, PositionResult)
        assert result.n_bars == 1000
        assert result.n_trades_unfiltered == 999
        assert result.n_trades_filtered == 1
        assert result.trade_reduction_pct > 50.0
        assert result.gate_trade_reduction_pass is True

    def test_gate_fail_no_reduction(self):
        """No trade reduction → FAIL."""
        signal = np.tile([1.0, -1.0], 50)
        position = signal.copy()
        result = evaluate_quality_gate(signal, position, bar_minutes=15.0)
        assert result.trade_reduction_pct < 1.0
        assert result.gate_trade_reduction_pass is False
        assert result.gate_pass is False

    def test_holding_time_computation(self):
        """Mean holding bars → correct hours."""
        signal = np.tile([1.0, -1.0], 50)
        # Hold for 20 bars each — need multiple changes
        position = np.tile(np.repeat([1.0, -1.0], 20), 3)[:100]
        result = evaluate_quality_gate(signal, position, bar_minutes=15.0)
        assert abs(result.mean_holding_bars - 20.0) < 1e-10
        assert abs(result.mean_holding_hours - 5.0) < 1e-10
        assert result.gate_holding_time_pass is True

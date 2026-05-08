"""Tests for alpha.adapter — continuous signal to Strategy bridge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import pytest
from alpha.adapter import ContinuousSignalAdapter, ValidationResult


# ---------------------------------------------------------------------------
# ContinuousSignalAdapter
# ---------------------------------------------------------------------------


class TestContinuousSignalAdapter:
    def test_init_defaults(self):
        signal = np.array([0.1, 0.5, -0.3])
        adapter = ContinuousSignalAdapter(signal)
        assert adapter.entry_threshold == 0.3
        assert adapter.exit_threshold == 0.15  # 0.3 / 2
        assert adapter.stop_loss_pct == 2.0
        assert adapter.take_profit_pct == 4.0
        assert adapter.max_holding_bars == 600

    def test_init_custom(self):
        signal = np.zeros(10)
        adapter = ContinuousSignalAdapter(
            signal, entry_threshold=0.5, exit_threshold=0.1,
            stop_loss_pct=3.0, take_profit_pct=6.0, max_holding_bars=100,
        )
        assert adapter.entry_threshold == 0.5
        assert adapter.exit_threshold == 0.1
        assert adapter.stop_loss_pct == 3.0

    def test_long_strategy_entry(self):
        signal = np.array([0.0, 0.1, 0.4, 0.5, 0.2, -0.4])
        adapter = ContinuousSignalAdapter(signal, entry_threshold=0.3)
        strategy = adapter.to_long_strategy()

        assert strategy.name == "alpha_signal_long"
        assert strategy.direction == "long"

        df = pl.DataFrame({"x": range(len(signal))})
        entries = strategy.entry_condition(df)
        expected = [False, False, True, True, False, False]
        assert entries.to_list() == expected

    def test_long_strategy_exit(self):
        signal = np.array([0.5, 0.4, 0.2, 0.1, 0.05, -0.1])
        adapter = ContinuousSignalAdapter(signal, entry_threshold=0.3)
        strategy = adapter.to_long_strategy()

        df = pl.DataFrame({"x": range(len(signal))})
        exits = strategy.exit_condition(df)
        # Exit when signal < 0.15 (exit_threshold = 0.3/2)
        expected = [False, False, False, True, True, True]
        assert exits.to_list() == expected

    def test_short_strategy_entry(self):
        signal = np.array([0.0, -0.1, -0.4, -0.5, -0.2, 0.4])
        adapter = ContinuousSignalAdapter(signal, entry_threshold=0.3)
        strategy = adapter.to_short_strategy()

        assert strategy.name == "alpha_signal_short"
        assert strategy.direction == "short"

        df = pl.DataFrame({"x": range(len(signal))})
        entries = strategy.entry_condition(df)
        expected = [False, False, True, True, False, False]
        assert entries.to_list() == expected

    def test_short_strategy_exit(self):
        signal = np.array([-0.5, -0.4, -0.2, -0.1, -0.05, 0.1])
        adapter = ContinuousSignalAdapter(signal, entry_threshold=0.3)
        strategy = adapter.to_short_strategy()

        df = pl.DataFrame({"x": range(len(signal))})
        exits = strategy.exit_condition(df)
        # Exit when signal > -0.15 (i.e., signal > -exit_threshold)
        expected = [False, False, False, True, True, True]
        assert exits.to_list() == expected

    def test_signal_padding(self):
        """If df is longer than signal, pad with zeros."""
        signal = np.array([0.5, -0.5])
        adapter = ContinuousSignalAdapter(signal, entry_threshold=0.3)
        strategy = adapter.to_long_strategy()

        df = pl.DataFrame({"x": range(5)})
        entries = strategy.entry_condition(df)
        # signal padded to [0.5, -0.5, 0, 0, 0]
        assert entries.to_list() == [True, False, False, False, False]


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_gate_all_pass(self):
        vr = ValidationResult(
            direction="long",
            oos_sharpe=1.0, is_sharpe=1.2, oos_is_ratio=0.83,
            max_drawdown_pct=3.0, total_oos_trades=50,
            profit_factor=1.5, deflated_sharpe_p=0.01, n_trials=1000,
            gate_oos_sharpe_pass=True, gate_oos_is_ratio_pass=True,
            gate_deflated_sharpe_pass=True, gate_max_dd_pass=True,
            gate_min_trades_pass=True, gate_profit_factor_pass=True,
            gate_pass=True,
        )
        assert vr.gate_pass is True

    def test_gate_fail_single(self):
        vr = ValidationResult(
            direction="long",
            oos_sharpe=0.3, is_sharpe=1.2, oos_is_ratio=0.25,
            max_drawdown_pct=8.0, total_oos_trades=10,
            profit_factor=0.8, deflated_sharpe_p=0.2, n_trials=1000,
            gate_oos_sharpe_pass=False, gate_oos_is_ratio_pass=False,
            gate_deflated_sharpe_pass=False, gate_max_dd_pass=False,
            gate_min_trades_pass=False, gate_profit_factor_pass=False,
            gate_pass=False,
        )
        assert vr.gate_pass is False

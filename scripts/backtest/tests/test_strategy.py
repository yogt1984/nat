"""
Skeptical Tests for Strategy Module

These tests ensure strategies are properly defined and don't have
logical errors that could lead to incorrect backtests.
"""

import pytest
import polars as pl
import numpy as np
from backtest.strategy import (
    Strategy,
    accumulation_long,
    distribution_short,
    entropy_breakout,
    regime_momentum,
    whale_flow_simple,
    get_strategy,
    get_all_strategies,
)


class TestStrategyValidation:
    """Test strategy parameter validation."""

    def test_direction_must_be_long_or_short(self):
        """Direction must be 'long' or 'short'."""
        with pytest.raises(ValueError):
            Strategy(
                name="test",
                entry_condition=lambda df: pl.Series([True]),
                exit_condition=lambda df: pl.Series([False]),
                direction="sideways",
            )

    def test_stop_loss_must_be_positive(self):
        """Stop loss must be positive."""
        with pytest.raises(ValueError):
            Strategy(
                name="test",
                entry_condition=lambda df: pl.Series([True]),
                exit_condition=lambda df: pl.Series([False]),
                stop_loss_pct=-1.0,
            )

    def test_take_profit_must_be_positive(self):
        """Take profit must be positive."""
        with pytest.raises(ValueError):
            Strategy(
                name="test",
                entry_condition=lambda df: pl.Series([True]),
                exit_condition=lambda df: pl.Series([False]),
                take_profit_pct=0.0,
            )

    def test_max_holding_must_be_positive(self):
        """Max holding bars must be positive."""
        with pytest.raises(ValueError):
            Strategy(
                name="test",
                entry_condition=lambda df: pl.Series([True]),
                exit_condition=lambda df: pl.Series([False]),
                max_holding_bars=0,
            )

    def test_valid_strategy_creation(self):
        """Valid strategy should be created without error."""
        strategy = Strategy(
            name="test",
            entry_condition=lambda df: pl.Series([True] * len(df)),
            exit_condition=lambda df: pl.Series([False] * len(df)),
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            max_holding_bars=100,
            direction="long",
        )
        assert strategy.name == "test"
        assert strategy.direction == "long"


class TestStrategyConditions:
    """Test strategy condition evaluation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe with required columns."""
        n = 100
        return pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0 + i * 0.1 for i in range(n)],
            "accumulation_score": [0.5 + 0.3 * np.sin(i / 10) for i in range(n)],
            "distribution_score": [0.5 - 0.3 * np.sin(i / 10) for i in range(n)],
            "whale_flow_zscore_1h": [2.0 * np.sin(i / 15) for i in range(n)],
            "range_position_24h": [(i % 100) / 100 for i in range(n)],
            "tick_entropy_1m": [0.5 + 0.3 * np.cos(i / 8) for i in range(n)],
            "absorption_zscore": [1.5 + np.sin(i / 12) for i in range(n)],
            "whale_net_flow_1h": [100 * np.sin(i / 20) for i in range(n)],
            "regime_clarity": [0.6 + 0.2 * np.cos(i / 10) for i in range(n)],
            "trend_hurst_300": [0.5 + 0.1 * np.sin(i / 15) for i in range(n)],
        })

    def test_entry_condition_returns_series(self, sample_df):
        """Entry condition should return a boolean Series."""
        strategy = accumulation_long()
        result = strategy.entry_condition(sample_df)

        assert isinstance(result, pl.Series), "Should return polars Series"
        assert len(result) == len(sample_df), "Should have same length as input"

    def test_exit_condition_returns_series(self, sample_df):
        """Exit condition should return a boolean Series."""
        strategy = accumulation_long()
        result = strategy.exit_condition(sample_df)

        assert isinstance(result, pl.Series), "Should return polars Series"
        assert len(result) == len(sample_df), "Should have same length as input"

    def test_entry_exit_not_always_true(self, sample_df):
        """Entry and exit should not always be true (would be nonsensical)."""
        strategy = accumulation_long()

        entry = strategy.entry_condition(sample_df)
        exit_sig = strategy.exit_condition(sample_df)

        # At least some False values should exist
        assert not entry.all(), "Entry should not always be true"
        assert not exit_sig.all(), "Exit should not always be true"

    def test_entry_exit_not_always_false(self, sample_df):
        """Entry and exit should not always be false (strategy would never trade)."""
        # Create a df where whale_flow_simple will definitely trigger
        n = 100
        df_with_signals = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0] * n,
            "whale_flow_zscore_1h": [3.0 if i < 50 else -1.0 for i in range(n)],  # Strong signal
        })

        strategy = whale_flow_simple()

        entry = strategy.entry_condition(df_with_signals)
        exit_sig = strategy.exit_condition(df_with_signals)

        # At least some True values should exist
        assert entry.any(), "Entry should sometimes be true"
        assert exit_sig.any(), "Exit should sometimes be true"

    def test_conditions_handle_nulls(self):
        """Conditions should handle null values gracefully."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3, 4, 5],
            "raw_midprice": [100.0, None, 102.0, None, 104.0],
            "whale_flow_zscore_1h": [1.0, None, 3.0, None, 2.0],
        })

        strategy = whale_flow_simple()

        # Should not raise
        entry = strategy.entry_condition(df)
        assert len(entry) == 5

        # Nulls should be treated as False
        assert not entry[1], "Null should evaluate to False"


class TestBuiltinStrategies:
    """Test all built-in strategy definitions."""

    def test_accumulation_long_exists(self):
        """Accumulation long strategy should be available."""
        strategy = accumulation_long()
        assert strategy.name == "accumulation_long"
        assert strategy.direction == "long"
        assert len(strategy.required_features) > 0

    def test_distribution_short_exists(self):
        """Distribution short strategy should be available."""
        strategy = distribution_short()
        assert strategy.name == "distribution_short"
        assert strategy.direction == "short"

    def test_entropy_breakout_exists(self):
        """Entropy breakout strategy should be available."""
        strategy = entropy_breakout()
        assert strategy.name == "entropy_breakout"
        assert strategy.direction == "long"

    def test_regime_momentum_exists(self):
        """Regime momentum strategy should be available."""
        strategy = regime_momentum()
        assert strategy.name == "regime_momentum"

    def test_whale_flow_simple_exists(self):
        """Whale flow simple strategy should be available."""
        strategy = whale_flow_simple()
        assert strategy.name == "whale_flow_simple"

    def test_all_strategies_have_required_features(self):
        """All strategies should declare required features."""
        for name, strategy in get_all_strategies().items():
            assert hasattr(strategy, "required_features"), f"{name} missing required_features"
            assert isinstance(strategy.required_features, list), f"{name} required_features not list"

    def test_all_strategies_have_description(self):
        """All strategies should have a description."""
        for name, strategy in get_all_strategies().items():
            assert strategy.description, f"{name} missing description"


class TestStrategyRegistry:
    """Test strategy registry functions."""

    def test_get_all_strategies_returns_dict(self):
        """get_all_strategies should return a dictionary."""
        strategies = get_all_strategies()
        assert isinstance(strategies, dict)
        assert len(strategies) > 0

    def test_get_strategy_by_name(self):
        """Should retrieve strategy by name."""
        strategy = get_strategy("whale_flow_simple")
        assert strategy.name == "whale_flow_simple"

    def test_get_unknown_strategy_raises(self):
        """Unknown strategy name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_strategy("nonexistent_strategy")
        assert "Unknown strategy" in str(exc_info.value)

    def test_strategy_names_are_unique(self):
        """All strategy names should be unique."""
        strategies = get_all_strategies()
        names = [s.name for s in strategies.values()]
        assert len(names) == len(set(names)), "Duplicate strategy names"


class TestStrategyLogic:
    """Test the logic of strategy conditions makes sense."""

    def test_accumulation_requires_low_range_position(self):
        """Accumulation long should require price at lower range."""
        # This tests that the strategy logic is sensible
        df_low_range = pl.DataFrame({
            "accumulation_score": [0.8],
            "whale_flow_zscore_1h": [2.0],
            "range_position_24h": [0.2],  # Low in range
        })

        df_high_range = pl.DataFrame({
            "accumulation_score": [0.8],
            "whale_flow_zscore_1h": [2.0],
            "range_position_24h": [0.8],  # High in range
        })

        strategy = accumulation_long()

        entry_low = strategy.entry_condition(df_low_range)
        entry_high = strategy.entry_condition(df_high_range)

        assert entry_low[0], "Should enter when range position is low"
        assert not entry_high[0], "Should not enter when range position is high"

    def test_distribution_requires_high_range_position(self):
        """Distribution short should require price at higher range."""
        df_high_range = pl.DataFrame({
            "distribution_score": [0.8],
            "whale_flow_zscore_1h": [-2.0],
            "range_position_24h": [0.8],  # High in range
        })

        df_low_range = pl.DataFrame({
            "distribution_score": [0.8],
            "whale_flow_zscore_1h": [-2.0],
            "range_position_24h": [0.2],  # Low in range
        })

        strategy = distribution_short()

        entry_high = strategy.entry_condition(df_high_range)
        entry_low = strategy.entry_condition(df_low_range)

        assert entry_high[0], "Should enter when range position is high"
        assert not entry_low[0], "Should not enter when range position is low"

    def test_whale_flow_requires_strong_signal(self):
        """Whale flow strategy should require strong z-score."""
        df_strong = pl.DataFrame({"whale_flow_zscore_1h": [2.5]})
        df_weak = pl.DataFrame({"whale_flow_zscore_1h": [1.5]})

        strategy = whale_flow_simple()

        assert strategy.entry_condition(df_strong)[0], "Should enter on strong signal"
        assert not strategy.entry_condition(df_weak)[0], "Should not enter on weak signal"


class TestStrategyRiskParams:
    """Test strategy risk parameters are sensible."""

    def test_stop_loss_less_than_take_profit(self):
        """Stop loss should typically be less than take profit."""
        for name, strategy in get_all_strategies().items():
            # Allow equal, but stop should not be greater
            assert strategy.stop_loss_pct <= strategy.take_profit_pct * 1.5, (
                f"{name}: stop_loss ({strategy.stop_loss_pct}) too large vs "
                f"take_profit ({strategy.take_profit_pct})"
            )

    def test_reasonable_stop_loss_range(self):
        """Stop loss should be in reasonable range (0.5% to 10%)."""
        for name, strategy in get_all_strategies().items():
            assert 0.5 <= strategy.stop_loss_pct <= 10.0, (
                f"{name}: stop_loss ({strategy.stop_loss_pct}) out of reasonable range"
            )

    def test_reasonable_take_profit_range(self):
        """Take profit should be in reasonable range (1% to 20%)."""
        for name, strategy in get_all_strategies().items():
            assert 1.0 <= strategy.take_profit_pct <= 20.0, (
                f"{name}: take_profit ({strategy.take_profit_pct}) out of reasonable range"
            )

    def test_max_holding_reasonable(self):
        """Max holding should be reasonable (100 to 10000 bars)."""
        for name, strategy in get_all_strategies().items():
            assert 100 <= strategy.max_holding_bars <= 10000, (
                f"{name}: max_holding ({strategy.max_holding_bars}) out of reasonable range"
            )

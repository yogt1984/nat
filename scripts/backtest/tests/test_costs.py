"""
Skeptical Tests for Cost Model

These tests ensure the cost model behaves correctly and doesn't
underestimate transaction costs (a common source of backtest fantasy).
"""

import pytest
import math
from backtest.costs import (
    CostModel,
    hyperliquid_taker,
    hyperliquid_maker,
    conservative,
    zero_cost,
)


class TestCostModelBasics:
    """Basic cost model functionality tests."""

    def test_default_costs_are_positive(self):
        """Default costs should not be zero."""
        model = CostModel()
        assert model.fee_bps > 0, "Default fee should be positive"
        assert model.slippage_bps >= 0, "Slippage should be non-negative"

    def test_round_trip_is_double_one_way(self):
        """Round trip cost = 2 * one-way cost."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        assert model.round_trip_cost_bps == 2 * model.one_way_cost_bps

    def test_cost_fraction_conversion(self):
        """10 bps = 0.001 = 0.1%."""
        model = CostModel(fee_bps=10.0, slippage_bps=0.0)
        assert abs(model.one_way_cost_fraction - 0.001) < 1e-10

    def test_negative_fee_rejected(self):
        """Negative fees should raise error."""
        with pytest.raises(ValueError):
            CostModel(fee_bps=-1.0)

    def test_negative_slippage_rejected(self):
        """Negative slippage should raise error."""
        with pytest.raises(ValueError):
            CostModel(slippage_bps=-1.0)


class TestCostApplication:
    """Tests for applying costs to prices."""

    def test_long_entry_costs_more(self):
        """Long entry should pay more than quoted price."""
        model = CostModel(fee_bps=10.0, slippage_bps=0.0)
        price = 100.0
        entry = model.apply_entry_cost(price, "long")
        assert entry > price, "Long entry should cost more than quote"

    def test_long_exit_receives_less(self):
        """Long exit should receive less than quoted price."""
        model = CostModel(fee_bps=10.0, slippage_bps=0.0)
        price = 100.0
        exit_price = model.apply_exit_cost(price, "long")
        assert exit_price < price, "Long exit should receive less than quote"

    def test_short_entry_receives_less(self):
        """Short entry should receive less than quoted price."""
        model = CostModel(fee_bps=10.0, slippage_bps=0.0)
        price = 100.0
        entry = model.apply_entry_cost(price, "short")
        assert entry < price, "Short entry should receive less than quote"

    def test_short_exit_costs_more(self):
        """Short exit (buy to cover) should cost more than quoted price."""
        model = CostModel(fee_bps=10.0, slippage_bps=0.0)
        price = 100.0
        exit_price = model.apply_exit_cost(price, "short")
        assert exit_price > price, "Short exit should cost more than quote"

    def test_symmetric_cost_application(self):
        """Cost application should be symmetric for long and short."""
        model = CostModel(fee_bps=10.0, slippage_bps=5.0)
        price = 100.0

        long_entry = model.apply_entry_cost(price, "long")
        short_exit = model.apply_exit_cost(price, "short")

        # Both should be price * (1 + cost)
        assert abs(long_entry - short_exit) < 1e-10

        long_exit = model.apply_exit_cost(price, "long")
        short_entry = model.apply_entry_cost(price, "short")

        # Both should be price * (1 - cost)
        assert abs(long_exit - short_entry) < 1e-10


class TestPnLCalculation:
    """Tests for P&L calculation with costs."""

    def test_zero_move_loses_costs(self):
        """If price doesn't move, you lose the round-trip cost."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        price = 100.0

        pnl = model.compute_pnl(price, price, "long", include_costs=True)

        # Should lose round-trip cost (~0.14%)
        expected_loss = -model.round_trip_cost_bps / 100
        assert abs(pnl - expected_loss) < 0.01, f"Expected ~{expected_loss}%, got {pnl}%"

    def test_pnl_without_costs_is_raw_return(self):
        """P&L without costs should be raw percentage return."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        entry = 100.0
        exit_price = 102.0

        pnl_no_cost = model.compute_pnl(entry, exit_price, "long", include_costs=False)

        # Raw return = (102/100 - 1) * 100 = 2%
        assert abs(pnl_no_cost - 2.0) < 1e-10

    def test_pnl_with_costs_less_than_without(self):
        """P&L with costs should always be less than without."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        entry = 100.0
        exit_price = 105.0

        pnl_with = model.compute_pnl(entry, exit_price, "long", include_costs=True)
        pnl_without = model.compute_pnl(entry, exit_price, "long", include_costs=False)

        assert pnl_with < pnl_without, "Costs should reduce P&L"

    def test_short_profit_on_price_drop(self):
        """Short position should profit when price drops."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        entry = 100.0
        exit_price = 95.0

        pnl = model.compute_pnl(entry, exit_price, "short", include_costs=True)

        # Should be positive (profitable short)
        assert pnl > 0, "Short should profit on price drop"

    def test_short_loss_on_price_rise(self):
        """Short position should lose when price rises."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        entry = 100.0
        exit_price = 105.0

        pnl = model.compute_pnl(entry, exit_price, "short", include_costs=True)

        # Should be negative (losing short)
        assert pnl < 0, "Short should lose on price rise"

    def test_breakeven_calculation(self):
        """Breakeven move should equal round-trip cost."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        breakeven = model.breakeven_move_pct()

        # Should be 0.14% (14 bps round trip)
        expected = model.round_trip_cost_bps / 100
        assert abs(breakeven - expected) < 1e-10


class TestCostPresets:
    """Test preset cost models."""

    def test_hyperliquid_taker_costs(self):
        """Hyperliquid taker preset should match expected values."""
        model = hyperliquid_taker()
        assert model.fee_bps == 5.0, "Taker fee should be 5 bps"
        assert model.slippage_bps == 2.0, "Default slippage should be 2 bps"

    def test_hyperliquid_maker_cheaper_than_taker(self):
        """Maker fees should be less than taker fees."""
        maker = hyperliquid_maker()
        taker = hyperliquid_taker()
        assert maker.fee_bps < taker.fee_bps, "Maker should be cheaper"

    def test_conservative_more_expensive(self):
        """Conservative model should have higher costs."""
        cons = conservative()
        taker = hyperliquid_taker()
        assert cons.round_trip_cost_bps > taker.round_trip_cost_bps

    def test_zero_cost_is_zero(self):
        """Zero cost model should have no costs."""
        model = zero_cost()
        assert model.fee_bps == 0.0
        assert model.slippage_bps == 0.0
        assert model.round_trip_cost_bps == 0.0


class TestCostEdgeCases:
    """Edge case tests for cost model."""

    def test_very_small_price(self):
        """Cost model should handle very small prices."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        price = 0.00001

        entry = model.apply_entry_cost(price, "long")
        assert entry > price
        assert math.isfinite(entry)

    def test_very_large_price(self):
        """Cost model should handle very large prices."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        price = 1_000_000.0

        entry = model.apply_entry_cost(price, "long")
        assert entry > price
        assert math.isfinite(entry)

    def test_zero_price_returns_zero(self):
        """Zero price should return zero (edge case)."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        price = 0.0

        entry = model.apply_entry_cost(price, "long")
        assert entry == 0.0

    def test_cost_model_repr(self):
        """Cost model should have readable string representation."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        s = repr(model)
        assert "5" in s
        assert "2" in s
        assert "14" in s or "round_trip" in s


class TestCostModelRealism:
    """Tests to ensure cost model is realistic."""

    def test_minimum_realistic_costs(self):
        """Warn if costs are suspiciously low."""
        # Any real exchange has at least 1bp fee
        model = CostModel(fee_bps=0.1, slippage_bps=0.0)

        # This should work but is unrealistic
        # In a real system, we might want to warn
        assert model.one_way_cost_bps < 1.0, "Cost under 1bp is unrealistic"

    def test_costs_accumulate_with_trades(self):
        """Multiple round trips should accumulate costs."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)
        capital = 10000.0

        # 10 round trips
        for _ in range(10):
            capital *= (1 - model.round_trip_cost_fraction)

        # Should have lost significant amount to fees
        loss_pct = (1 - capital / 10000.0) * 100
        assert loss_pct > 1.0, "10 round trips should cost >1%"

    def test_high_frequency_trading_not_free(self):
        """High frequency trading accumulates significant costs."""
        model = CostModel(fee_bps=5.0, slippage_bps=2.0)

        # 100 trades per day for a year
        trades_per_year = 100 * 252
        capital = 10000.0

        for _ in range(trades_per_year):
            capital *= (1 - model.round_trip_cost_fraction)

        # Should be nearly wiped out
        assert capital < 1000.0, "HFT without edge should lose to costs"

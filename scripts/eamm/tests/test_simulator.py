"""Unit tests for EAMM MM Fill Simulator."""

import numpy as np
import polars as pl
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.simulator import simulate_mm, pnl_to_bps, SimulationResult


def _make_df(prices: list, n_repeat: int = 1) -> pl.DataFrame:
    """Create a minimal DataFrame from a price series."""
    if n_repeat > 1:
        prices = prices * n_repeat
    return pl.DataFrame({
        "timestamp_ns": list(range(len(prices))),
        "raw_midprice": [float(p) for p in prices],
        "symbol": ["BTC"] * len(prices),
    })


class TestBothSidesFilled:
    """When price swings wide enough to cross both quotes, PnL = 2*delta."""

    def test_round_trip_pnl(self):
        # Price starts at 100, drops to 95 (fills bid), rises to 105 (fills ask)
        prices = [100.0] + [95.0] * 5 + [105.0] * 5 + [100.0] * 89
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[100.0], horizon=15)
        # 100 bps = 1% spread. Bid at 99, ask at 101.
        # Min future price = 95 < 99 → bid fills
        # Max future price = 105 > 101 → ask fills
        # PnL should be 2 * 1% * 100 = 2.0
        assert result.fill_bid[0, 0] == 1.0
        assert result.fill_ask[0, 0] == 1.0
        assert result.fill_round_trip[0, 0] == 1.0
        assert result.pnl[0, 0] == pytest.approx(2.0, abs=0.01)


class TestBidOnlyFill:
    """Price drops, only bid fills. PnL = mark-to-market from long position."""

    def test_bid_fill_adverse_selection(self):
        # Price at 100, drops to 98 and stays there
        prices = [100.0] + [98.0] * 99
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[50.0], horizon=20)
        # 50 bps = 0.5%. Bid at 99.5, ask at 100.5
        # Min = 98 < 99.5 → bid fills
        # Max = 98 < 100.5 → ask does NOT fill (future max is 98)
        # future_price = prices[20] = 98
        # PnL = (future_price - bid_price) = (98 - 99.5) = -1.5
        assert result.fill_bid[0, 0] == 1.0
        assert result.fill_ask[0, 0] == 0.0
        assert result.pnl[0, 0] < 0  # adverse selection loss


class TestAskOnlyFill:
    """Price rises, only ask fills. PnL = mark-to-market from short position."""

    def test_ask_fill_adverse_selection(self):
        # Price at 100, rises to 102 and stays there
        prices = [100.0] + [102.0] * 99
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[50.0], horizon=20)
        # Bid at 99.5, ask at 100.5
        # Min = 102 > 99.5 → bid does NOT fill
        # Max = 102 > 100.5 → ask fills
        # PnL = (ask_price - future_price) = (100.5 - 102) = -1.5
        assert result.fill_bid[0, 0] == 0.0
        assert result.fill_ask[0, 0] == 1.0
        assert result.pnl[0, 0] < 0  # adverse selection loss


class TestNoFill:
    """Price stays within quotes. PnL = 0."""

    def test_no_fill_zero_pnl(self):
        # Price perfectly flat at 100
        prices = [100.0] * 100
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[100.0], horizon=20)
        # 100 bps spread. Bid at 99, ask at 101.
        # Price never leaves 100 → no fills
        assert result.fill_bid[0, 0] == 0.0
        assert result.fill_ask[0, 0] == 0.0
        assert result.pnl[0, 0] == 0.0


class TestWiderSpreadFewerFills:
    """Wider spreads must have equal or fewer fills than narrow spreads."""

    def test_fill_monotonicity(self):
        # Random walk with enough volatility to fill some spreads
        np.random.seed(42)
        prices = [100.0]
        for _ in range(999):
            prices.append(prices[-1] + np.random.randn() * 0.1)
        df = _make_df(prices)

        spreads = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        result = simulate_mm(df, spread_levels_bps=spreads, horizon=50)

        # Sum fills over all valid rows
        valid = ~np.isnan(result.fill_bid[:, 0])
        for k in range(len(spreads) - 1):
            fills_narrow = np.nansum(result.fill_round_trip[valid, k])
            fills_wide = np.nansum(result.fill_round_trip[valid, k + 1])
            assert fills_wide <= fills_narrow, (
                f"Wider spread {spreads[k+1]} has more round-trip fills "
                f"({fills_wide}) than narrower {spreads[k]} ({fills_narrow})"
            )


class TestZeroSpread:
    """Delta=0 means every trade crosses both quotes. PnL = adverse selection."""

    def test_zero_spread_all_fill(self):
        # Any price movement fills both sides at delta=0
        np.random.seed(42)
        prices = [100.0]
        for _ in range(199):
            prices.append(prices[-1] + np.random.randn() * 0.5)
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[0.0001], horizon=30)
        # With essentially zero spread and volatile prices, most should fill
        valid = ~np.isnan(result.fill_round_trip[:, 0])
        fill_rate = np.nanmean(result.fill_round_trip[valid, 0])
        assert fill_rate > 0.6, f"Expected >60% fill rate at zero spread, got {fill_rate:.2%}"


class TestOutputShape:
    """Verify output dimensions match input."""

    def test_shapes(self):
        prices = [100.0] * 200
        df = _make_df(prices)
        spreads = [1.0, 2.0, 3.0]
        result = simulate_mm(df, spread_levels_bps=spreads, horizon=10)

        N = len(prices)
        K = len(spreads)
        assert result.pnl.shape == (N, K)
        assert result.fill_bid.shape == (N, K)
        assert result.fill_ask.shape == (N, K)
        assert result.fill_round_trip.shape == (N, K)
        assert result.midprice.shape == (N,)
        assert result.midprice_at_horizon.shape == (N,)
        assert len(result.timestamps) == N


class TestPnlToBps:
    """Test PnL conversion to basis points."""

    def test_conversion(self):
        prices = [100.0] + [95.0] * 5 + [105.0] * 5 + [100.0] * 89
        df = _make_df(prices)
        result = simulate_mm(df, spread_levels_bps=[100.0], horizon=15)
        pnl_bps = pnl_to_bps(result)
        # PnL = 2.0 on midprice 100 → 200 bps
        assert pnl_bps[0, 0] == pytest.approx(200.0, abs=1.0)


class TestInsufficientData:
    """Horizon larger than data should raise."""

    def test_raises_on_short_data(self):
        df = _make_df([100.0] * 5)
        with pytest.raises(ValueError, match="Need at least"):
            simulate_mm(df, spread_levels_bps=[1.0], horizon=10)

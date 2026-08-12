"""
Planted tests for funding accrual in the sim cost path (COST-9).

Perp funding is a forced payment at every settlement while a position is held.
Before COST-9 no backtest debited or credited it: `funding_enabled` defaulted to
False everywhere and the settlement interval was a hardcoded `8.0` — while the
venue actually settles EVERY HOUR (verified live 2026-08-12: `fundingHistory`
entries for BTC are spaced 1.0000 h apart and `assetCtx.funding` is the hourly
rate, e.g. 1.25e-5 = the 0.00125 %/h baseline). An 8 h interval therefore
understates accrual 8x even when enabled.

Every expected number below is hand-computable: constant rate, flat price,
zero fee/slippage — accrual = rate x hours held, nothing else.
"""

import numpy as np
import polars as pl
import pytest

from backtest.costs import CostModel, zero_cost
from backtest.engine import run_backtest
from backtest.strategy import Strategy
from utils.costs import funding_interval_hours, load_costs


HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# SSOT plumbing: the interval comes from config/costs.toml, never a literal
# ---------------------------------------------------------------------------

class TestFundingIntervalSSOT:
    def test_accessor_reads_the_toml(self):
        """utils.costs.funding_interval_hours() is the SSOT accessor."""
        toml_value = load_costs()["hyperliquid"]["funding_interval_hours"]
        assert funding_interval_hours() == toml_value

    def test_interval_is_hourly_venue_fact(self):
        """Hyperliquid settles funding every 1 h (live fundingHistory, 2026-08-12).

        If this fails because the venue changed its clock, update costs.toml WITH
        new live evidence — never to silence the test.
        """
        assert funding_interval_hours() == 1.0

    def test_cost_model_sources_interval_from_ssot(self):
        """CostModel's default interval is the config value, not a hardcoded 8."""
        model = CostModel()
        assert model.funding_interval_hours == funding_interval_hours()


# ---------------------------------------------------------------------------
# Charged by default: the omission WAS the bug
# ---------------------------------------------------------------------------

class TestFundingChargedByDefault:
    def test_funding_enabled_by_default(self):
        """A held-position sim must charge funding unless explicitly opted out."""
        assert CostModel().funding_enabled is True

    def test_zero_cost_preset_stays_zero(self):
        """The explicit-opt-in zero preset is genuinely cost-free, funding included."""
        assert zero_cost().funding_enabled is False


# ---------------------------------------------------------------------------
# Accrual arithmetic (hand-computed)
# ---------------------------------------------------------------------------

class TestFundingArithmetic:
    def _model(self, **kw):
        return CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=True,
                         funding_interval_hours=1.0, **kw)

    def test_accrual_is_rate_times_settlements(self):
        """1 bp per settlement, held 5 h at 1 h interval -> 5 bps = 0.05 %."""
        assert self._model().compute_funding_cost(5.0, 1.0) == pytest.approx(0.05)

    def test_default_interval_accrues_hourly(self):
        """With SSOT defaults, 8 h at 1 bp/settlement = 8 settlements = 0.08 %.

        Under the old hardcoded 8 h interval this was ONE settlement (0.01 %) —
        the 8x understatement this test exists to keep dead.
        """
        model = CostModel(fee_bps=0.0, slippage_bps=0.0)
        assert model.compute_funding_cost(8.0, 1.0) == pytest.approx(0.08)

    def test_disabled_accrues_nothing(self):
        model = CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=False)
        assert model.compute_funding_cost(5.0, 1.0) == 0.0

    def test_long_pays_short_receives(self):
        """Flat price, zero fees: funding is the ONLY P&L term, sign by side."""
        model = self._model()
        funding = model.compute_funding_cost(5.0, 1.0)  # 0.05 %
        long_pnl = model.compute_pnl(100.0, 100.0, "long", funding_cost_pct=funding)
        short_pnl = model.compute_pnl(100.0, 100.0, "short", funding_cost_pct=funding)
        assert long_pnl == pytest.approx(-0.05)
        assert short_pnl == pytest.approx(+0.05)

    def test_negative_rate_flips_the_sign(self):
        """Negative funding: shorts pay, longs receive."""
        model = self._model()
        funding = model.compute_funding_cost(5.0, -1.0)  # -0.05 %
        long_pnl = model.compute_pnl(100.0, 100.0, "long", funding_cost_pct=funding)
        assert long_pnl == pytest.approx(+0.05)


# ---------------------------------------------------------------------------
# Engine integration: planted constant-rate data through run_backtest
# ---------------------------------------------------------------------------

def _flat_df(n_bars: int, funding_rate: float | None) -> pl.DataFrame:
    """Flat price at 100.0, hourly bars, optional constant ctx_funding_rate."""
    cols = {
        "timestamp_ms": np.arange(n_bars, dtype=np.int64) * HOUR_MS,
        "raw_midprice": np.full(n_bars, 100.0),
    }
    if funding_rate is not None:
        cols["ctx_funding_rate"] = np.full(n_bars, funding_rate)
    return pl.DataFrame(cols)


def _hold_n_bars(n: int, direction: str) -> Strategy:
    """Enter on bar 0, exit only by timeout after exactly n bars."""
    return Strategy(
        name=f"planted_hold_{n}",
        entry_condition=lambda df: pl.Series([True] + [False] * (len(df) - 1)),
        exit_condition=lambda df: pl.Series([False] * len(df)),
        stop_loss_pct=50.0,
        take_profit_pct=50.0,
        max_holding_bars=n,
        direction=direction,
    )


class TestEngineFundingAccrual:
    """Rate 1e-4/h (10 bps/h — planted, deliberately large), held 10 h."""

    RATE = 1e-4
    EXPECTED_PCT = 0.1  # 1e-4 x 10 h = 1e-3 = 0.1 %

    def _run(self, direction: str, funding_rate: float | None,
             enabled: bool = True):
        model = CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=enabled,
                          funding_interval_hours=1.0)
        return run_backtest(_flat_df(20, funding_rate), _hold_n_bars(10, direction), model)

    def test_long_debited_exactly(self):
        result = self._run("long", self.RATE)
        assert result.total_trades == 1
        assert result.trades[0].pnl_pct == pytest.approx(-self.EXPECTED_PCT)

    def test_short_credited_exactly(self):
        result = self._run("short", self.RATE)
        assert result.trades[0].pnl_pct == pytest.approx(+self.EXPECTED_PCT)

    def test_equity_curve_carries_the_debit(self):
        result = self._run("long", self.RATE)
        assert result.equity_curve[-1] == pytest.approx(10000.0 * (1 - 0.001))

    def test_opt_out_restores_zero(self):
        result = self._run("long", self.RATE, enabled=False)
        assert result.trades[0].pnl_pct == pytest.approx(0.0)

    def test_no_funding_column_accrues_nothing(self):
        """Data without ctx_funding_rate must not invent a charge."""
        result = self._run("long", None)
        assert result.trades[0].pnl_pct == pytest.approx(0.0)

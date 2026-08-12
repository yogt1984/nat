"""Planted tests for the LF8 funding-carry study (pre-registered driver).

Constant planted rates, flat prices, zero-fee funding model — every expectation
below is hand arithmetic. The three legs are separable by construction: flat
prices zero the price leg, constant membership zeroes steady-state turnover,
so the funding leg is the only survivor and its value is rate x hours.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.costs import CostModel
from exploration.lf8_funding_carry_study import (
    GATE0_MIN_RATIO,
    carry_scores,
    carry_weights,
    evaluate_criteria,
    funding_leg_fraction,
    gate0,
    run_config,
)
from utils.costs import load_costs, slippage_bps, taker_bps

RATE = 1e-4  # 1 bp/h — deliberately large, keeps the arithmetic visible


def _panel(n_hours: int, rates: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({c: np.full(n_hours, r) for c, r in rates.items()})


def _zero_fee_model() -> CostModel:
    return CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=True,
                     funding_interval_hours=1.0)


class TestConstruction:
    def test_weights_long_negative_short_positive(self):
        scores = pd.Series({"A": +RATE, "B": -RATE, "C": 0.0})
        wts = carry_weights(scores, k=1)
        assert wts["B"] == pytest.approx(+0.5)   # most negative -> long (gets paid)
        assert wts["A"] == pytest.approx(-0.5)   # most positive -> short (gets paid)
        assert wts["C"] == 0.0
        assert wts.sum() == pytest.approx(0.0)   # dollar-neutral
        assert wts.abs().sum() == pytest.approx(1.0)

    def test_too_few_names_returns_none(self):
        assert carry_weights(pd.Series({"A": RATE}), k=1) is None

    def test_scores_use_no_future_settlement(self):
        """A spike one hour after the rebalance must not move the score."""
        panel = _panel(50, {"A": 0.0, "B": 0.0})
        panel.loc[41, "A"] = 1.0                 # the future
        scores = carry_scores(panel, t_idx=40, window_h=24)
        assert scores["A"] == pytest.approx(0.0)


class TestFundingLeg:
    def test_both_legs_collect_hand_computed(self):
        """w=±0.5, ±1 bp/h, 24 h: each leg earns 0.5 x 24 bp -> 24 bp total."""
        wts = pd.Series({"A": -0.5, "B": +0.5})
        held = _panel(24, {"A": +RATE, "B": -RATE})
        pnl = funding_leg_fraction(wts, held, _zero_fee_model())
        assert pnl == pytest.approx(0.0024)      # 24 bp as a fraction

    def test_long_pays_positive_funding(self):
        wts = pd.Series({"A": +1.0})
        held = _panel(24, {"A": +RATE})
        assert funding_leg_fraction(wts, held, _zero_fee_model()) \
            == pytest.approx(-0.0024)

    def test_interval_comes_from_the_model_not_a_literal(self):
        """Doubling the model's settlement interval must halve the accrual —
        proof the leg routes through CostModel (COST-9), not inline arithmetic."""
        wts = pd.Series({"A": +1.0})
        held = _panel(24, {"A": +RATE})
        slow = CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=True,
                         funding_interval_hours=2.0)
        assert funding_leg_fraction(wts, held, slow) \
            == pytest.approx(-0.0012)


class TestGate0:
    def _spread_universe(self):
        # 4 coins so k=1 has distinct tails; dispersion only between A and B
        rates = {"A": +RATE, "B": -RATE, "C": +1e-6, "D": -1e-6}
        panel = _panel(30 * 24, rates)
        hs = pd.Series(1.0, index=list(rates))
        return panel, hs

    def test_persistent_dispersion_passes(self):
        panel, hs = self._spread_universe()
        g = gate0(panel, hs, load_costs(), k=1, window_h=24)
        # carry = 0.5 x (1e-4 - (-1e-4)) x 24 h = 24 bps/day; churn after day 1 = 0
        assert g["carry_bps_day"] == pytest.approx(24.0, rel=1e-6)
        assert g["cost_bps_day"] == pytest.approx(0.0, abs=1e-9)
        assert g["ratio"] >= GATE0_MIN_RATIO

    def test_zero_dispersion_fails(self):
        """A common funding level is uncollectable by a dollar-neutral book."""
        panel = _panel(30 * 24, {c: +1e-5 for c in "ABCD"})
        hs = pd.Series(1.0, index=list("ABCD"))
        g = gate0(panel, hs, load_costs(), k=1, window_h=24)
        assert g["carry_bps_day"] == pytest.approx(0.0, abs=1e-9)
        assert g["ratio"] < GATE0_MIN_RATIO


class TestRunConfig:
    def test_planted_carry_survives_and_is_exact(self):
        """30 planted days: price leg 0, funding leg 24 bp/day, cost day 1 only."""
        rates = {"A": +RATE, "B": -RATE, "C": +1e-6, "D": -1e-6}
        n_hours = 30 * 24
        panel = _panel(n_hours, rates)
        prices = pd.DataFrame({c: np.full(n_hours, 100.0) for c in rates})
        hs = pd.Series(1.0, index=list(rates))
        r = run_config(k=1, window_h=24, stress=1.0, funding_wide=panel,
                       price_wide=prices, half_spread_bps=hs,
                       costs=load_costs(), cost_model=_zero_fee_model())
        n = r["n"]
        assert n >= 27                            # 30 days minus warmup/tail
        assert r["gross_price_pct"] == pytest.approx(0.0, abs=1e-12)
        assert r["funding_pct"] == pytest.approx(0.24 * n, rel=1e-6)
        # entry cost charged once: turnover 1.0 x (hs + taker + slip), SSOT-sourced
        expected_entry_bps = 1.0 + taker_bps() + slippage_bps()
        assert r["cost_total_pct"] == pytest.approx(expected_entry_bps * 1e-2,
                                                    rel=1e-6)
        assert r["positive_share"] == pytest.approx(1.0)

    def test_criteria_evaluator_needs_the_stressed_twin(self):
        """(f) is unverifiable without the 2x run — the config must then FAIL."""
        base = {"k": 1, "window_h": 24, "stress": 1.0, "n": 30,
                "sharpe_net": 5.0, "dsr_p": 0.001, "positive_share": 1.0,
                "max_day_share": 0.05, "oos_is_ratio": 1.0, "net_total_pct": 7.0}
        survivors = evaluate_criteria([dict(base)])
        assert survivors == []
        stressed = dict(base, stress=2.0, net_total_pct=6.0)
        survivors = evaluate_criteria([dict(base), stressed])
        assert survivors == ["k=1 w=24h"]

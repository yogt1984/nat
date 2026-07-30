"""Planted test for HF5 — Avellaneda–Stoikov maker (sim-first).

Composes the maker line: HF1 microprice center (fair value), A-S reservation price
r = s − q·γ·σ²·τ (inventory skew), optimal spread δ = γ·σ²·τ + (2/γ)·ln(1+γ/κ),
HF4 toxicity gate (pull on toxic flow), A4-calibrated fill intensities
λ(d) = A·exp(−κ·d), and honest accounting (maker rebate per fill, terminal
inventory liquidated at taker cost — both via load_costs()).

Contract:
  (a) reservation skew: long inventory lowers both quotes by exactly q·γ·σ²·τ;
      short raises them; flat leaves r = s;
  (b) spread formula exact; widens in σ; the ln term vanishes as κ → ∞;
  (c) the HF1 center shifts both quotes (fair value ≠ mid);
  (d) gate closed → no quotes; inventory cap → reduce-only quoting;
  (e) κ recovered from a planted exponential fill-intensity profile;
  (f) driftless market + symmetric fills → positive expected PnL (spread + rebate)
      with mean-reverting bounded inventory;
  (g) one-way drift → the skew cuts terminal inventory vs a no-skew baseline;
  (h) terminal liquidation is charged at taker cost (SSOT), seeded runs reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.avellaneda_stoikov import ASParams, ASQuoter, ASSim, calibrate_kappa
from utils.costs import maker_bps, taker_bps


def _params(**over):
    p = dict(gamma=0.1, kappa=1.5, tau_ticks=100, q_max=5.0)
    p.update(over)
    return ASParams(**p)


class TestReservationPrice:
    def test_flat_inventory_reservation_is_fair_value(self):
        q = ASQuoter(_params())
        r = q.reservation_bps(fair_dev_bps=0.3, inventory=0.0, sigma_bps=1.0)
        assert r == pytest.approx(0.3)

    def test_long_inventory_lowers_reservation_exactly(self):
        p = _params(gamma=0.2)
        q = ASQuoter(p)
        r = q.reservation_bps(fair_dev_bps=0.0, inventory=2.0, sigma_bps=1.5)
        assert r == pytest.approx(-2.0 * 0.2 * 1.5**2 * p.tau_ticks)

    def test_short_inventory_raises_reservation(self):
        q = ASQuoter(_params())
        r_short = q.reservation_bps(0.0, inventory=-1.0, sigma_bps=1.0)
        r_long = q.reservation_bps(0.0, inventory=+1.0, sigma_bps=1.0)
        assert r_short > 0 > r_long
        assert r_short == pytest.approx(-r_long)


class TestOptimalSpread:
    def test_formula_exact(self):
        p = _params(gamma=0.1, kappa=1.5, tau_ticks=100)
        q = ASQuoter(p)
        expected = 0.1 * 2.0**2 * 100 + (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
        assert q.total_spread_bps(sigma_bps=2.0) == pytest.approx(expected)

    def test_widens_with_vol(self):
        q = ASQuoter(_params())
        assert q.total_spread_bps(2.0) > q.total_spread_bps(1.0)

    def test_ln_term_vanishes_as_kappa_grows(self):
        p_lo = _params(kappa=0.5)
        p_hi = _params(kappa=500.0)
        s_lo = ASQuoter(p_lo).total_spread_bps(1.0)
        s_hi = ASQuoter(p_hi).total_spread_bps(1.0)
        risk_term = p_lo.gamma * 1.0 * p_lo.tau_ticks
        assert s_hi == pytest.approx(risk_term, rel=1e-2)
        assert s_lo > s_hi


class TestQuoting:
    def test_center_follows_hf1_dev(self):
        q = ASQuoter(_params())
        lo = q.quotes_bps(fair_dev_bps=-0.5, inventory=0.0, sigma_bps=1.0, gate_open=True)
        hi = q.quotes_bps(fair_dev_bps=+0.5, inventory=0.0, sigma_bps=1.0, gate_open=True)
        assert hi.bid_bps > lo.bid_bps and hi.ask_bps > lo.ask_bps
        assert (hi.bid_bps - lo.bid_bps) == pytest.approx(1.0)

    def test_gate_closed_pulls_both(self):
        q = ASQuoter(_params())
        out = q.quotes_bps(0.0, 0.0, 1.0, gate_open=False)
        assert out.bid_bps is None and out.ask_bps is None

    def test_inventory_cap_reduce_only(self):
        p = _params(q_max=2.0)
        q = ASQuoter(p)
        long_capped = q.quotes_bps(0.0, inventory=2.0, sigma_bps=1.0, gate_open=True)
        assert long_capped.bid_bps is None and long_capped.ask_bps is not None
        short_capped = q.quotes_bps(0.0, inventory=-2.0, sigma_bps=1.0, gate_open=True)
        assert short_capped.ask_bps is None and short_capped.bid_bps is not None


class TestKappaCalibration:
    def test_recovers_planted_exponential(self):
        rng = np.random.default_rng(3)
        A_true, k_true = 0.8, 0.9
        offsets = np.linspace(0.2, 4.0, 12)
        n_post = 800
        fills = [(d, int(rng.binomial(n_post, min(A_true * np.exp(-k_true * d), 1.0))))
                 for d in offsets]
        A_hat, k_hat = calibrate_kappa(
            [(d, k, n_post) for d, k in fills])
        assert k_hat == pytest.approx(k_true, rel=0.15)
        assert A_hat == pytest.approx(A_true, rel=0.25)


class TestSimulation:
    def _run(self, n=4000, seed=0, drift=0.0, gamma=0.05, sigma=1.0):
        rng = np.random.default_rng(seed)
        mid = 100.0 * np.exp(np.cumsum(drift * 1e-4 + sigma * 1e-4 * rng.standard_normal(n)))
        sim = ASSim(_params(gamma=gamma), fill_A=0.3, fill_kappa=0.8, seed=seed + 1)
        return sim.run(
            mid=mid,
            fair_dev_bps=np.zeros(n),
            sigma_bps=np.full(n, sigma),
            gate_open=np.ones(n, dtype=bool),
        )

    def test_driftless_symmetric_market_is_profitable(self):
        res = self._run(seed=1)
        assert res["n_fills"] > 100
        assert res["pnl_bps"] > 0, res
        assert abs(res["terminal_inventory"]) <= 5.0     # q_max respected

    def test_inventory_mean_reverts(self):
        res = self._run(seed=2)
        assert abs(res["mean_inventory"]) < 1.0
        assert res["max_abs_inventory"] <= 5.0

    def test_skew_cuts_inventory_under_drift(self):
        strong = self._run(seed=3, drift=0.5, gamma=0.3)
        none = self._run(seed=3, drift=0.5, gamma=1e-9)
        assert strong["max_abs_inventory"] <= none["max_abs_inventory"]
        assert abs(strong["terminal_inventory"]) <= abs(none["terminal_inventory"]) + 1e-9

    def test_terminal_liquidation_charged_at_taker(self):
        res = self._run(seed=4)
        # the report must expose the liquidation cost, priced via the SSOT
        assert "liquidation_cost_bps" in res
        if res["terminal_inventory"] != 0:
            assert res["liquidation_cost_bps"] > 0
        assert res["taker_bps_used"] == pytest.approx(taker_bps())
        assert res["maker_bps_used"] == pytest.approx(maker_bps())

    def test_seeded_reproducible(self):
        r1 = self._run(seed=5)
        r2 = self._run(seed=5)
        assert r1["pnl_bps"] == pytest.approx(r2["pnl_bps"])
        assert r1["n_fills"] == r2["n_fills"]

    def test_gate_closed_everywhere_no_fills(self):
        rng = np.random.default_rng(6)
        n = 1000
        mid = 100.0 * np.exp(np.cumsum(1e-4 * rng.standard_normal(n)))
        sim = ASSim(_params(), fill_A=0.3, fill_kappa=0.8, seed=7)
        res = sim.run(mid=mid, fair_dev_bps=np.zeros(n),
                      sigma_bps=np.ones(n), gate_open=np.zeros(n, dtype=bool))
        assert res["n_fills"] == 0
        assert res["pnl_bps"] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

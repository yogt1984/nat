"""Planted test for the touch-maker experiment (FINDINGS §4.8 → §4.9).

The one maker posture both honest instruments point at: quotes PEGGED AT THE TOUCH
(never spread-derived — width is what killed the textbook config), with three
switchable layers on top:

    side-selection : HF1 dev > +θ → bid only; dev < −θ → ask only; else both;
    inventory skew : |q| ≥ q_soft → reduce-only (suppress the increasing side);
    HF4 gate       : toxic flow → no quotes;
    EV gate (A4)   : post only while capture (half-spread + rebate) exceeds the
                     causal adverse-selection estimate (EWMA of OWN past fills'
                     markout, initialized at the §4.7-measured prior).

Fill mechanics inherited unchanged from ASQueueSim (HF5b): conservative joins,
priority lost on requote, touch-zone depletion, price-through, no crossing quotes,
deterministic (no RNG), SSOT costs.

Pinned:
  (a) quotes sit at the touch — a single planted fill executes at best_bid exactly;
  (b) side-selection suppresses the disfavored side; |dev| ≤ θ posts both;
  (c) reduce-only at q_soft: relentless sell flow cannot push q past the bound;
  (d) HF4 gate closed → nothing; (e) EV gate: adverse prior above capture → no
      postings; below capture → posts;
  (f) no flow + no cross → zero fills, zero PnL; (g) deterministic; SSOT costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.touch_maker import TouchMakerSim, TouchParams
from utils.costs import maker_bps, taker_bps


def _params(**over):
    p = dict(theta_dev_bps=0.05, q_soft=3.0, requote_every=5, l1_fraction=0.5,
             use_hf1_side=False, use_inv_skew=False, use_hf4_gate=False,
             use_ev_gate=False, adverse_prior_bps=0.25)
    p.update(over)
    return TouchParams(**p)


def _mkt(n, mid=100.0, spread_bps=1.0):
    m = np.full(n, float(mid))
    half = mid * spread_bps * 1e-4 / 2.0
    return {
        "mid": m, "best_bid": m - half, "best_ask": m + half,
        "sell_exec": np.zeros(n), "buy_exec": np.zeros(n),
        "depth_bid": np.full(n, 10.0), "depth_ask": np.full(n, 10.0),
        "fair_dev_bps": np.zeros(n), "gate_open": np.ones(n, dtype=bool),
    }


def _run(mkt, params=None):
    return TouchMakerSim(params or _params()).run(**mkt)


class TestTouchPeg:
    def test_single_fill_executes_at_best_bid(self):
        n = 200
        m = _mkt(n)
        m["sell_exec"][:20] = 3.0                     # deplete ahead=5 fast, once
        res = _run(m, _params(requote_every=200))
        assert res["n_fills"] == 1
        # bought 1 unit at best_bid, +rebate; terminal liq of q=1 at taker
        bb = float(m["best_bid"][0]); mid0 = 100.0
        expected = (-bb + maker_bps() * 1e-4 * bb            # buy + rebate
                    + 1.0 * mid0                              # MTM at flat mid
                    - mid0 * taker_bps() * 1e-4) / mid0 * 1e4
        assert res["pnl_bps"] == pytest.approx(expected, abs=1e-6)

    def test_no_flow_no_cross_zero(self):
        res = _run(_mkt(300))
        assert res["n_fills"] == 0
        assert res["pnl_bps"] == pytest.approx(0.0)


class TestSideSelection:
    def _one_sided_flow(self, n=300, sell=True):
        m = _mkt(n)
        (m["sell_exec"] if sell else m["buy_exec"])[:] = 3.0
        return m

    def test_positive_dev_bids_only(self):
        m = self._one_sided_flow(sell=False)          # buy flow fills ASKS only
        m["fair_dev_bps"][:] = +0.2                   # dev > θ → ask suppressed
        res = _run(m, _params(use_hf1_side=True))
        assert res["n_fills"] == 0
        off = _run(m, _params(use_hf1_side=False))    # control: asks post and fill
        assert off["n_fills"] > 0

    def test_negative_dev_asks_only(self):
        m = self._one_sided_flow(sell=True)           # sell flow fills BIDS only
        m["fair_dev_bps"][:] = -0.2                   # dev < −θ → bid suppressed
        res = _run(m, _params(use_hf1_side=True))
        assert res["n_fills"] == 0

    def test_small_dev_posts_both(self):
        m = self._one_sided_flow(sell=True)
        m["fair_dev_bps"][:] = 0.01                   # |dev| ≤ θ
        res = _run(m, _params(use_hf1_side=True))
        assert res["n_fills"] > 0


class TestInventoryAndGates:
    def test_reduce_only_at_q_soft(self):
        m = _mkt(2000)
        m["sell_exec"][:] = 5.0                       # relentless sell flow
        res = _run(m, _params(use_inv_skew=True, q_soft=2.0, requote_every=10))
        assert res["max_abs_inventory"] <= 2.0

    def test_without_skew_inventory_exceeds_bound(self):
        m = _mkt(2000)
        m["sell_exec"][:] = 5.0
        res = _run(m, _params(use_inv_skew=False, q_soft=2.0, requote_every=10))
        assert res["max_abs_inventory"] > 2.0         # the control proves the bound bites

    def test_hf4_gate_closed_nothing(self):
        m = _mkt(500)
        m["sell_exec"][:] = 5.0
        m["gate_open"][:] = False
        res = _run(m, _params(use_hf4_gate=True))
        assert res["n_fills"] == 0

    def test_ev_gate_prior_above_capture_blocks(self):
        # capture = half-spread 0.5 + rebate 0.2 = 0.7 bps; prior 5 bps ≫ capture
        m = _mkt(500)
        m["sell_exec"][:] = 5.0
        res = _run(m, _params(use_ev_gate=True, adverse_prior_bps=5.0))
        assert res["n_fills"] == 0

    def test_ev_gate_prior_below_capture_posts(self):
        m = _mkt(500)
        m["sell_exec"][:] = 5.0
        res = _run(m, _params(use_ev_gate=True, adverse_prior_bps=0.1))
        assert res["n_fills"] > 0


class TestDiscipline:
    def test_deterministic(self):
        m = _mkt(1000)
        m["sell_exec"][::3] = 2.0
        m["buy_exec"][::4] = 2.0
        r1, r2 = _run(m), _run(m)
        assert r1["pnl_bps"] == r2["pnl_bps"] and r1["n_fills"] == r2["n_fills"]

    def test_costs_from_ssot(self):
        m = _mkt(300)
        m["sell_exec"][:] = 3.0
        res = _run(m, _params(requote_every=300))
        assert res["maker_bps_used"] == pytest.approx(maker_bps())
        assert res["taker_bps_used"] == pytest.approx(taker_bps())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Planted test for HF5b — ASQueueSim: A-S quotes filled through the queue engine.

§4.8's standing verdict: exogenous λ(d) fills carry no adverse selection, so absolute
ASSim PnL is fantasy. ASQueueSim couples the A-S quoter to A4's conservative queue
rules — fills happen ONLY by FIFO depletion (side flow consuming the queue while the
quote sits in the touch zone) or price-through. No RNG anywhere: fills are earned
from the tape, not drawn.

Zone rules (per tick, conservative):
  through : opposite best crosses the resting price → fill;
  touch   : best_bid ≤ price < best_ask (bid side) → side flow depletes `ahead`,
            initialized at l1_fraction·depth on EVERY post (even inside the spread —
            latent competition assumed; there is no "alone at the level" free lunch);
  behind  : price outside the touch zone → nothing (cancellations never advance).
  Requoting cancels: queue position resets, priority is never carried across posts.
  A quote that would cross at post time is NOT placed (no fantasy taker fills).

Pinned consequences:
  (a) no flow + no cross → zero fills, zero PnL;
  (b) touch-zone bid fills exactly when planted sell flow depletes ahead;
  (c) inside-spread post still waits behind l1_fraction·depth (conservative join);
  (d) behind-touch bid fills only on price-through;
  (e) requoting resets the queue — insufficient flow between requotes never fills;
  (f) ADVERSE SELECTION EMERGES: sell flow preceding a mid drop fills the bid into
      the fall → the episode LOSES money (the failure the exogenous sim can't show);
  (g) symmetric flow on a flat mid captures spread + rebate;
  (h) gate closed → nothing; one-sided flow is capped at q_max;
  (i) fully deterministic; (j) costs from the SSOT.
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.avellaneda_stoikov import ASParams, ASQueueSim
from utils.costs import maker_bps, taker_bps


def _params(**over):
    # kappa=10 → tight optimal spread → quotes land INSIDE the market spread (touch
    # zone); the behind-touch tests lower kappa / raise gamma to push quotes out.
    p = dict(gamma=0.02, kappa=10.0, tau_ticks=50, q_max=3.0)
    p.update(over)
    return ASParams(**p)


def _mkt(n, mid=100.0, spread_bps=1.0):
    m = np.full(n, float(mid))
    half = mid * spread_bps * 1e-4 / 2.0
    return {
        "mid": m,
        "best_bid": m - half,
        "best_ask": m + half,
        "sell_exec": np.zeros(n),      # sell-aggressor volume per tick (hits bids)
        "buy_exec": np.zeros(n),       # buy-aggressor volume per tick (hits asks)
        "depth_bid": np.full(n, 10.0),
        "depth_ask": np.full(n, 10.0),
        "fair_dev_bps": np.zeros(n),
        "sigma_bps": np.full(n, 0.05),
        "gate_open": np.ones(n, dtype=bool),
    }


def _run(mkt, params=None, **kw):
    sim = ASQueueSim(params or _params(), requote_every=kw.pop("requote_every", 5),
                     l1_fraction=kw.pop("l1_fraction", 0.5))
    return sim.run(**mkt)


class TestNoFreeMoney:
    def test_no_flow_no_cross_zero_fills(self):
        res = _run(_mkt(500))
        assert res["n_fills"] == 0
        assert res["pnl_bps"] == pytest.approx(0.0)

    def test_gate_closed_nothing_happens(self):
        m = _mkt(500)
        m["sell_exec"][:] = 5.0
        m["gate_open"][:] = False
        res = _run(m)
        assert res["n_fills"] == 0

    def test_inside_spread_post_still_joins_conservatively(self):
        # (c): ahead = 0.5·10 = 5 even inside the spread; zero flow → never fills.
        res = _run(_mkt(500), requote_every=500)
        assert res["n_fills"] == 0


class TestFillMechanics:
    def test_touch_zone_fill_by_depletion(self):
        # ahead = 0.5·10 = 5; sell flow 1/tick → fill once 5 units execute.
        m = _mkt(400)
        m["sell_exec"][:] = 1.0
        res = _run(m, requote_every=400)
        assert res["n_fills"] >= 1
        assert res["first_fill_tick"] is not None
        assert 3 <= res["first_fill_tick"] <= 7

    def test_requote_resets_queue_position(self):
        m = _mkt(400)
        m["sell_exec"][:] = 1.0                       # needs ~5 ticks; requote every 3
        res = _run(m, requote_every=3)
        assert res["n_fills"] == 0

    def test_behind_touch_no_depletion_fill(self):
        # wide optimal spread (small kappa, big gamma) → quotes behind the touch
        m = _mkt(300)
        m["sell_exec"][:] = 50.0
        res = _run(m, params=_params(gamma=0.5, kappa=1.0, tau_ticks=200))
        assert res["n_fills"] == 0

    def test_behind_touch_fills_on_price_through(self):
        m = _mkt(300)
        drop = np.linspace(0.0, -30e-4, 300)          # −30 bps drift
        m["mid"] = 100.0 * (1.0 + drop)
        m["best_bid"] = m["mid"] * (1 - 0.5e-4)
        m["best_ask"] = m["mid"] * (1 + 0.5e-4)
        res = _run(m, params=_params(gamma=0.5, kappa=1.0, tau_ticks=200),
                   requote_every=300)
        assert res["n_fills"] >= 1


class TestAdverseSelectionEmerges:
    def test_sell_flow_before_drop_loses_money(self):
        """The honest failure: our bid is consumed by the flow that moves mid down."""
        n = 600
        m = _mkt(n)
        mid = np.full(n, 100.0)
        mid[200:] = 100.0 * (1 - 20e-4)               # −20 bps step after the flow
        m["mid"] = mid
        m["best_bid"] = mid * (1 - 0.5e-4)
        m["best_ask"] = mid * (1 + 0.5e-4)
        m["sell_exec"][150:200] = 2.0                 # sell burst right before the drop
        res = _run(m, requote_every=600)
        assert res["n_fills"] >= 1
        assert res["pnl_bps"] < 0, res                # filled long into the drop

    def test_symmetric_flow_flat_mid_is_profitable(self):
        n = 2000
        m = _mkt(n)
        m["sell_exec"][:] = 1.5
        m["buy_exec"][:] = 1.5
        res = _run(m, requote_every=50)
        assert res["n_fills"] >= 10
        assert res["pnl_bps"] > 0, res                # spread + rebate, no drift


class TestInventoryAndDeterminism:
    def test_one_sided_flow_capped_at_q_max(self):
        m = _mkt(3000)
        m["sell_exec"][:] = 5.0                       # relentless sell flow
        res = _run(m, params=_params(q_max=2.0), requote_every=20)
        assert res["max_abs_inventory"] <= 2.0

    def test_deterministic_no_rng(self):
        m = _mkt(800)
        m["sell_exec"][::3] = 2.0
        m["buy_exec"][::4] = 2.0
        r1 = _run(m)
        r2 = _run(m)
        assert r1["pnl_bps"] == r2["pnl_bps"]
        assert r1["n_fills"] == r2["n_fills"]

    def test_costs_from_ssot(self):
        m = _mkt(400)
        m["sell_exec"][:] = 2.0
        res = _run(m, requote_every=400)
        assert res["maker_bps_used"] == pytest.approx(maker_bps())
        assert res["taker_bps_used"] == pytest.approx(taker_bps())
        if res["terminal_inventory"] != 0:
            assert res["liquidation_cost_bps"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

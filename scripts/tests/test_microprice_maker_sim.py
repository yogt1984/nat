"""Reliability suite for GAP-04 `microprice_maker_sim` (docs/GAP__26_7_26.md/04).

Coverage:
  - quote math (skew + half-width) — exact values
  - conservative fill model (crossing-only, latency, timeout, no optimistic touches)
  - two-sided quoting + inventory cap + skew
  - the toxicity gate + NaN / degenerate-input handling + decision spacing
  - markout() semantics
  - determinism
  - the two planted mechanisms (microprice-centering, toxicity-gating) on a synthetic
    replay with informed bursts.

Invariants are parametrized over several seeds so each holds across many random replays.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from execution.microprice_maker_sim import MicropriceMakerSim
from kalman.fill_sim import FillEvent

SEEDS = [1, 2, 3, 5, 7, 11, 13, 17]


# --------------------------------------------------------------------------- #
# Synthetic replays
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _replay(seed: int, n: int = 4000, drift: float = 0.03, sigma: float = 0.05, ar: float = 0.6):
    """(mid, micro, spread, vol, informed). `micro` = fair-value level (forward-looking);
    `mid` = a fast mean-reverting oscillation around it that ramps up during informed bursts."""
    rng = np.random.default_rng(seed)
    informed = (np.arange(n) % 500) < 100  # 20% informed, in 100-tick bursts
    eps = rng.standard_normal(n)
    mid = np.empty(n)
    micro = np.empty(n)
    level, osc = 100.0, 0.0
    for i in range(n):
        if informed[i]:
            level += drift
        osc = ar * osc + sigma * eps[i]
        micro[i] = level
        mid[i] = level + osc
    spread = np.full(n, 0.10)
    vol = np.full(n, sigma)
    return mid, micro, spread, vol, informed


def _flat(n, mid, center=100.0, spread=2.0, vol=1.0):
    """Constant center/spread/vol with an explicit mid array (for exact-fill tests)."""
    mid = np.asarray(mid, dtype=float)
    return (mid, np.full(n, center), np.full(n, spread), np.full(n, vol))


def _mean_markout(sim, mid, center, spread, vol, gate_open, horizons=(1, 5, 30, 100)):
    fills = sim.simulate(mid, center, spread, vol, gate_open)
    mo = sim.markout(fills, mid, horizons)
    vals = [v for v in mo.values() if np.isfinite(v)]
    return fills, (float(np.mean(vals)) if vals else 0.0)


def _running_inventory(fills, size=1.0):
    inv, peak = 0.0, 0.0
    path = []
    for f in fills:
        inv += size if f.side == "buy" else -size
        path.append(inv)
        peak = max(peak, abs(inv))
    return path, peak


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_defaults(self):
        s = MicropriceMakerSim()
        assert (s.eta, s.c_vol, s.i_max, s.size) == (0.0, 0.0, 5.0, 1.0)
        assert (s.latency, s.timeout, s.min_gap) == (1, 50, 5)

    def test_custom_params_stored(self):
        s = MicropriceMakerSim(eta=0.2, c_vol=0.3, i_max=10, size=2, latency=3, timeout=20, min_gap=7)
        assert (s.eta, s.c_vol, s.i_max, s.size, s.latency, s.timeout, s.min_gap) == \
            (0.2, 0.3, 10, 2, 3, 20, 7)


# --------------------------------------------------------------------------- #
# Quote math (exact)
# --------------------------------------------------------------------------- #
class TestQuoteMath:
    def test_no_skew_no_cushion(self):
        s = MicropriceMakerSim(eta=0.0, c_vol=0.0)
        bid, ask = s.quotes(center=100.0, spread=2.0, vol=1.0, inv=0.0)
        assert (bid, ask) == (99.0, 101.0)

    def test_symmetric_around_center_when_flat(self):
        s = MicropriceMakerSim(eta=0.5, c_vol=0.5)
        bid, ask = s.quotes(100.0, 2.0, 1.0, inv=0.0)  # inv 0 -> no skew
        assert bid == pytest.approx(100 - (1.0 + 0.5))
        assert ask == pytest.approx(100 + (1.0 + 0.5))

    def test_long_inventory_skews_quotes_down(self):
        s = MicropriceMakerSim(eta=0.5, c_vol=0.0)
        bid, ask = s.quotes(100.0, 2.0, 1.0, inv=2.0)  # reservation = 100 - 0.5*2*1 = 99
        assert (bid, ask) == pytest.approx((98.0, 100.0))

    def test_short_inventory_skews_quotes_up(self):
        s = MicropriceMakerSim(eta=0.5, c_vol=0.0)
        bid, ask = s.quotes(100.0, 2.0, 1.0, inv=-2.0)  # reservation = 101
        assert (bid, ask) == pytest.approx((100.0, 102.0))

    def test_vol_cushion_widens_half_spread(self):
        s = MicropriceMakerSim(eta=0.0, c_vol=1.0)
        bid, ask = s.quotes(100.0, 2.0, 3.0, inv=0.0)  # delta = 1 + 1*3 = 4
        assert (bid, ask) == pytest.approx((96.0, 104.0))

    def test_midpoint_is_reservation(self):
        s = MicropriceMakerSim(eta=0.3, c_vol=0.2)
        bid, ask = s.quotes(100.0, 2.0, 1.5, inv=1.0)
        assert (bid + ask) / 2 == pytest.approx(100 - 0.3 * 1 * 1.5)


# --------------------------------------------------------------------------- #
# Invariants over random replays
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
class TestInvariants:
    def _fills(self, seed):
        mid, micro, spread, vol, _ = _replay(seed)
        sim = MicropriceMakerSim(i_max=5.0)
        return sim, mid, sim.simulate(mid, micro, spread, vol, np.ones(len(mid), bool))

    def test_has_fills(self, seed):
        _, _, fills = self._fills(seed)
        assert len(fills) > 30  # the replay must actually exercise the sim

    def test_no_optimistic_fills(self, seed):
        _, _, fills = self._fills(seed)
        for f in fills:
            if f.side == "buy":
                assert f.midprice_at_fill <= f.fill_price + 1e-9
            else:
                assert f.midprice_at_fill >= f.fill_price - 1e-9

    def test_inventory_within_cap(self, seed):
        _, _, fills = self._fills(seed)
        _, peak = _running_inventory(fills)
        assert peak <= 5.0 + 1e-9

    def test_causal_fill_after_order(self, seed):
        sim, _, fills = self._fills(seed)
        for f in fills:
            assert f.fill_tick >= f.signal_tick + sim.latency + 1

    def test_fill_within_timeout(self, seed):
        sim, _, fills = self._fills(seed)
        for f in fills:
            assert f.fill_tick < f.signal_tick + sim.latency + sim.timeout

    def test_midprice_at_fill_matches_series(self, seed):
        _, mid, fills = self._fills(seed)
        for f in fills:
            assert f.midprice_at_fill == pytest.approx(mid[f.fill_tick])

    def test_fill_price_finite_and_sided(self, seed):
        _, _, fills = self._fills(seed)
        for f in fills:
            assert np.isfinite(f.fill_price)
            assert f.side in ("buy", "sell")

    def test_deterministic(self, seed):
        mid, micro, spread, vol, _ = _replay(seed)
        sim = MicropriceMakerSim(i_max=5.0)
        gate = np.ones(len(mid), bool)
        a = sim.simulate(mid, micro, spread, vol, gate)
        b = sim.simulate(mid, micro, spread, vol, gate)
        key = lambda fs: [(f.signal_tick, f.fill_tick, f.fill_price, f.side) for f in fs]
        assert key(a) == key(b)


# --------------------------------------------------------------------------- #
# Conservative fill model (exact)
# --------------------------------------------------------------------------- #
class TestFillModelExact:
    def test_single_buy_fill_at_bid(self):
        n = 20
        mid = np.full(n, 100.0)
        mid[2] = 99.0  # dips to the bid
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid), np.ones(n, bool))
        buys = [f for f in fills if f.side == "buy"]
        assert len(buys) == 1
        assert buys[0].fill_tick == 2
        assert buys[0].fill_price == pytest.approx(99.0)
        assert buys[0].midprice_at_fill == pytest.approx(99.0)

    def test_single_sell_fill_at_ask(self):
        n = 20
        mid = np.full(n, 100.0)
        mid[2] = 101.0  # spikes to the ask
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid), np.ones(n, bool))
        sells = [f for f in fills if f.side == "sell"]
        assert len(sells) == 1 and sells[0].fill_tick == 2
        assert sells[0].fill_price == pytest.approx(101.0)

    def test_no_fill_when_mid_stays_inside_spread(self):
        n = 30
        mid = np.full(n, 100.0)  # never reaches 99 or 101
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        assert sim.simulate(*_flat(n, mid), np.ones(n, bool)) == []

    def test_boundary_touch_fills(self):
        # Exactly at the quote (mid == bid) counts as a fill (<=).
        n = 20
        mid = np.full(n, 100.0)
        mid[2] = 99.0
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid, spread=2.0), np.ones(n, bool))
        assert any(f.fill_tick == 2 for f in fills)

    def test_latency_delays_order(self):
        # A dip before the order rests (tick 2, latency 3 -> order at tick 3) must NOT fill.
        n = 30
        mid = np.full(n, 100.0)
        mid[2] = 99.0
        sim = MicropriceMakerSim(latency=3, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid), np.ones(n, bool))
        assert all(f.fill_tick != 2 for f in fills)

    def test_nan_mid_does_not_fill_but_later_cross_does(self):
        n = 20
        mid = np.full(n, 100.0)
        mid[2] = np.nan   # not a fill
        mid[3] = 99.0     # real cross
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid), np.ones(n, bool))
        buys = [f for f in fills if f.side == "buy"]
        assert len(buys) == 1 and buys[0].fill_tick == 3


# --------------------------------------------------------------------------- #
# Inventory cap under directional pressure
# --------------------------------------------------------------------------- #
class TestInventoryCap:
    def test_falling_market_caps_long_inventory(self):
        n = 400
        mid = np.linspace(100.0, 90.0, n)  # monotone down -> only bid fills
        sim = MicropriceMakerSim(i_max=5.0, latency=1, timeout=20, min_gap=1)
        fills = sim.simulate(mid, np.full(n, 100.0), np.full(n, 2.0), np.full(n, 1.0), np.ones(n, bool))
        path, peak = _running_inventory(fills)
        assert all(f.side == "buy" for f in fills)
        assert peak == pytest.approx(5.0)     # reaches the cap
        assert max(path) <= 5.0 + 1e-9        # never exceeds it

    def test_rising_market_caps_short_inventory(self):
        n = 400
        mid = np.linspace(100.0, 110.0, n)  # monotone up -> only ask fills
        sim = MicropriceMakerSim(i_max=4.0, latency=1, timeout=20, min_gap=1)
        fills = sim.simulate(mid, np.full(n, 100.0), np.full(n, 2.0), np.full(n, 1.0), np.ones(n, bool))
        path, peak = _running_inventory(fills)
        assert all(f.side == "sell" for f in fills)
        assert peak == pytest.approx(4.0)
        assert min(path) >= -4.0 - 1e-9

    def test_size_scales_inventory_step(self):
        n = 400
        mid = np.linspace(100.0, 90.0, n)
        sim = MicropriceMakerSim(i_max=6.0, size=2.0, latency=1, timeout=20, min_gap=1)
        fills = sim.simulate(mid, np.full(n, 100.0), np.full(n, 2.0), np.full(n, 1.0), np.ones(n, bool))
        _, peak = _running_inventory(fills, size=2.0)
        assert peak <= 6.0 + 1e-9

    @pytest.mark.parametrize("cap", [1.0, 3.0, 7.0])
    def test_cap_respected_across_values(self, cap):
        mid, micro, spread, vol, _ = _replay(3)
        sim = MicropriceMakerSim(i_max=cap)
        fills = sim.simulate(mid, micro, spread, vol, np.ones(len(mid), bool))
        _, peak = _running_inventory(fills)
        assert peak <= cap + 1e-9


# --------------------------------------------------------------------------- #
# Toxicity gate
# --------------------------------------------------------------------------- #
class TestGate:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_closed_no_fills(self, seed):
        mid, micro, spread, vol, _ = _replay(seed)
        sim = MicropriceMakerSim()
        assert sim.simulate(mid, micro, spread, vol, np.zeros(len(mid), bool)) == []

    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_open_has_fills(self, seed):
        mid, micro, spread, vol, _ = _replay(seed)
        sim = MicropriceMakerSim()
        assert len(sim.simulate(mid, micro, spread, vol, np.ones(len(mid), bool))) > 0

    def test_no_posting_in_closed_window(self):
        mid, micro, spread, vol, _ = _replay(7)
        gate = np.ones(len(mid), bool)
        gate[1000:2000] = False
        sim = MicropriceMakerSim()
        fills = sim.simulate(mid, micro, spread, vol, gate)
        assert all(not (1000 <= f.signal_tick < 2000) for f in fills)

    def test_resting_order_survives_gate_close(self):
        # A quote posted while the gate is open may still fill after it closes
        # (the gate blocks NEW postings, it does not cancel a resting order).
        n = 20
        mid = np.full(n, 100.0)
        mid[2] = 99.0
        gate = np.ones(n, bool)
        gate[1:] = False  # open only at tick 0
        sim = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = sim.simulate(*_flat(n, mid), gate)
        assert len(fills) == 1 and fills[0].signal_tick == 0 and fills[0].fill_tick == 2


# --------------------------------------------------------------------------- #
# Degenerate / NaN inputs
# --------------------------------------------------------------------------- #
class TestDegenerateInputs:
    @pytest.mark.parametrize("field", ["center", "spread", "vol"])
    def test_all_nan_field_no_fills(self, field):
        n = 100
        mid = np.full(n, 100.0)
        mid[::5] = 98.0
        arrs = dict(center=np.full(n, 100.0), spread=np.full(n, 2.0), vol=np.full(n, 1.0))
        arrs[field] = np.full(n, np.nan)
        sim = MicropriceMakerSim(latency=1, timeout=10, min_gap=1)
        assert sim.simulate(mid, arrs["center"], arrs["spread"], arrs["vol"], np.ones(n, bool)) == []

    def test_nonpositive_spread_no_fills(self):
        n = 100
        mid = np.full(n, 100.0)
        mid[::5] = 98.0
        sim = MicropriceMakerSim(latency=1, timeout=10, min_gap=1)
        assert sim.simulate(mid, np.full(n, 100.0), np.zeros(n), np.full(n, 1.0), np.ones(n, bool)) == []

    def test_empty_input(self):
        s = MicropriceMakerSim()
        e = np.array([])
        assert s.simulate(e, e, e, e, np.array([], bool)) == []

    def test_too_short_for_a_window(self):
        n = 4  # < latency + timeout
        z = np.full(n, 100.0)
        s = MicropriceMakerSim(latency=1, timeout=50)
        assert s.simulate(z, z, np.full(n, 2.0), np.full(n, 1.0), np.ones(n, bool)) == []

    def test_python_list_inputs(self):
        n = 20
        mid = [100.0] * n
        mid[2] = 99.0
        s = MicropriceMakerSim(latency=1, timeout=5, min_gap=1)
        fills = s.simulate(mid, [100.0] * n, [2.0] * n, [1.0] * n, [True] * n)
        assert any(f.side == "buy" for f in fills)


# --------------------------------------------------------------------------- #
# Decision spacing
# --------------------------------------------------------------------------- #
class TestSpacing:
    def test_tighter_spacing_gives_at_least_as_many_fills(self):
        mid, micro, spread, vol, _ = _replay(7)
        gate = np.ones(len(mid), bool)
        n_tight = len(MicropriceMakerSim(min_gap=1).simulate(mid, micro, spread, vol, gate))
        n_wide = len(MicropriceMakerSim(min_gap=50).simulate(mid, micro, spread, vol, gate))
        assert n_tight >= n_wide

    def test_min_gap_zero_and_large_both_run(self):
        mid, micro, spread, vol, _ = _replay(2)
        gate = np.ones(len(mid), bool)
        for mg in (1, 500):
            MicropriceMakerSim(min_gap=mg).simulate(mid, micro, spread, vol, gate)  # no crash


# --------------------------------------------------------------------------- #
# markout()
# --------------------------------------------------------------------------- #
class TestMarkout:
    def _fe(self, side, fill_tick, fill_price):
        return FillEvent(0, fill_tick, fill_price, side, 0.0, np.nan, fill_price)

    def test_keys_match_horizons(self):
        mid = np.full(50, 100.0)
        mo = MicropriceMakerSim.markout([self._fe("buy", 5, 99.0)], mid, horizons=(1, 10))
        assert set(mo.keys()) == {1, 10}

    def test_empty_fills_all_nan(self):
        mo = MicropriceMakerSim.markout([], np.full(10, 100.0), horizons=(1, 5))
        assert all(np.isnan(v) for v in mo.values())

    def test_buy_favourable_positive(self):
        mid = np.full(20, 99.0)
        mid[6] = 100.0  # price rose one horizon after a buy at 99
        mo = MicropriceMakerSim.markout([self._fe("buy", 5, 99.0)], mid, horizons=(1,))
        assert mo[1] > 0

    def test_sell_favourable_positive(self):
        mid = np.full(20, 101.0)
        mid[6] = 100.0  # price fell after a sell at 101
        mo = MicropriceMakerSim.markout([self._fe("sell", 5, 101.0)], mid, horizons=(1,))
        assert mo[1] > 0

    def test_buy_adverse_negative(self):
        mid = np.full(20, 99.0)
        mid[6] = 98.0  # price kept falling after a buy -> adverse
        mo = MicropriceMakerSim.markout([self._fe("buy", 5, 99.0)], mid, horizons=(1,))
        assert mo[1] < 0

    def test_horizon_past_end_is_nan(self):
        mo = MicropriceMakerSim.markout([self._fe("buy", 8, 99.0)], np.full(10, 100.0), horizons=(100,))
        assert np.isnan(mo[100])


# --------------------------------------------------------------------------- #
# Planted mechanisms (the GAP-04 thesis)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
class TestMechanisms:
    def test_microprice_centering_beats_mid(self, seed):
        # Isolated on PURE NOISE (drift=0): quoting around the stable microprice captures
        # the spread, quoting around the noisy mid is adversely selected. (Under informed
        # bursts the two centers are both run over, which confounds this specific effect —
        # that regime is what the toxicity gate below is for.)
        mid, micro, spread, vol, _ = _replay(seed, 4000, 0.0)
        sim = MicropriceMakerSim(i_max=5.0)
        allq = np.ones(len(mid), bool)
        _, m_mid = _mean_markout(sim, mid, mid, spread, vol, allq)
        _, m_micro = _mean_markout(sim, mid, micro, spread, vol, allq)
        assert m_micro > m_mid

    def test_toxicity_gate_lifts_markout_positive(self, seed):
        mid, micro, spread, vol, informed = _replay(seed)
        sim = MicropriceMakerSim(i_max=5.0)
        _, m_off = _mean_markout(sim, mid, micro, spread, vol, np.ones(len(mid), bool))
        _, m_gate = _mean_markout(sim, mid, micro, spread, vol, ~informed)
        assert m_gate > m_off
        assert m_gate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

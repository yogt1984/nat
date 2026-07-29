"""Planted (Level-1) test for GAP-04 `microprice_maker_sim` — RED before the sim exists.

Two mechanisms, both testable on a synthetic replay:
  1. CENTERING — quoting around the noisy *mid* is adversely selected (you fill when the
     mid moves against you, then it reverts), so markout is negative. Quoting around the
     *microprice* (a forward-looking fair value the mid reverts TO) removes that drag ->
     positive markout. This is why microprice replaces mid/VWAP for maker designs.
  2. TOXICITY GATE — during an informed burst the ask is adversely selected (price keeps
     rising through it). Pulling quotes on toxicity removes those fills and lifts markout.

Sim-first (docs/GAP__26_7_26.md/04). Plus sanity: inventory stays within +/-I_max, and
fills are conservative (only when the mid trades THROUGH the resting quote).
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.microprice_maker_sim import MicropriceMakerSim


def _replay(seed: int, *, n: int = 8000, drift: float = 0.03, sigma: float = 0.05, ar: float = 0.6):
    """Deterministic replay. `micro` = the fair-value level (forward-looking); `mid` = a
    noisy oscillation around it that ramps up only during informed bursts.
    Returns (mid, micro, spread, vol, informed)."""
    rng = np.random.default_rng(seed)
    informed = (np.arange(n) % 500) < 100  # 20% informed, in 100-tick bursts
    eps = rng.standard_normal(n)
    mid = np.empty(n)
    micro = np.empty(n)
    level, osc = 100.0, 0.0
    for i in range(n):
        if informed[i]:
            level += drift                 # informed flow pushes the fair value up (trend)
        osc = ar * osc + sigma * eps[i]    # fast zero-mean oscillation (mid reverts to micro)
        micro[i] = level                   # forward-looking fair value (microprice proxy)
        mid[i] = level + osc               # noisy observed mid
    spread = np.full(n, 0.10)              # constant spread (price units) -> half-spread 0.05
    vol = np.full(n, sigma)
    return mid, micro, spread, vol, informed


def _mean_markout(sim, mid, center, spread, vol, gate_open, horizons=(1, 5, 30, 100)):
    """Mean post-fill markout in bps: (mid[fill+h]-fill_price)/fill_price * dir; positive=good."""
    fills = sim.simulate(mid, center, spread, vol, gate_open)
    mid = np.asarray(mid, dtype=np.float64)
    n = len(mid)
    vals = []
    for f in fills:
        direction = 1.0 if f.side == "buy" else -1.0
        for h in horizons:
            t = f.fill_tick + h
            if t < n and np.isfinite(mid[t]):
                vals.append((mid[t] - f.fill_price) / f.fill_price * direction * 10000)
    return fills, (float(np.mean(vals)) if vals else 0.0)


class TestFillModel:
    def test_no_optimistic_fills(self):
        mid, micro, spread, vol, _ = _replay(3)
        sim = MicropriceMakerSim(i_max=5.0)
        fills = sim.simulate(mid, micro, spread, vol, np.ones(len(mid), bool))
        assert len(fills) > 50
        for f in fills:
            if f.side == "buy":
                assert f.midprice_at_fill <= f.fill_price + 1e-9  # mid crossed DOWN to the bid
            else:
                assert f.midprice_at_fill >= f.fill_price - 1e-9  # mid crossed UP to the ask

    def test_inventory_bounded(self):
        mid, micro, spread, vol, _ = _replay(3)
        cap = 3.0
        sim = MicropriceMakerSim(i_max=cap, size=1.0)
        fills = sim.simulate(mid, micro, spread, vol, np.ones(len(mid), bool))
        inv, peak = 0.0, 0.0
        for f in fills:
            inv += 1.0 if f.side == "buy" else -1.0
            peak = max(peak, abs(inv))
        assert peak <= cap + 1e-9


class TestCentering:
    def test_microprice_beats_mid_centering(self):
        # Quoting around the microprice (fair value) must beat quoting around the noisy mid.
        mid, micro, spread, vol, _ = _replay(7)
        sim = MicropriceMakerSim(i_max=5.0)
        allq = np.ones(len(mid), bool)
        _, m_mid = _mean_markout(sim, mid, mid, spread, vol, allq)
        _, m_micro = _mean_markout(sim, mid, micro, spread, vol, allq)
        assert m_micro > m_mid, f"microprice centering must beat mid: micro={m_micro:.3f} vs mid={m_mid:.3f} bps"


class TestToxicityGateMarkout:
    def test_gate_lifts_markout_positive(self):
        # With microprice centering, pulling quotes on toxicity must lift markout and clear zero.
        mid, micro, spread, vol, informed = _replay(7)
        sim = MicropriceMakerSim(i_max=5.0)
        _, m_off = _mean_markout(sim, mid, micro, spread, vol, np.ones(len(mid), bool))  # quote always
        _, m_gate = _mean_markout(sim, mid, micro, spread, vol, ~informed)               # pull on toxicity
        assert m_gate > m_off, f"the gate must lift markout: gated={m_gate:.3f} vs off={m_off:.3f} bps"
        assert m_gate > 0, f"microprice + gate markout should be positive (got {m_gate:.3f} bps)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

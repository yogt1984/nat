"""Planted test for A4 — the queue-value execution model (GAP spec 07, sim-first).

The micro-decision under every maker quote: is this resting limit order +EV after
adverse selection? EV = P(fill) · spread_capture − P(fill) · E[adverse | fill].
Fill probability comes from queue dynamics (FIFO depletion by executed volume,
price-through); adverse cost from post-fill drift. When EV < 0: don't post.

Planted contract (from docs/GAP__26_7_26.md/07_a4_queue_value.md):
  (a) certain fill + zero drift  -> EV == spread_capture (incl. maker rebate);
  (b) certain adverse fill       -> EV < 0 and the order is suppressed;
  (c) no volume, no cross        -> never fills, EV contribution 0;
  (d) price-through              -> immediate fill regardless of queue;
  (e) FIFO: executed volume depletes `ahead` exactly; book shrinkage that is NOT
      trades never advances the queue (conservative cancel rule);
  (f) costs come from load_costs() — never hardcoded;
  (g) analytic fill predictor calibrates to the replay simulator on synthetic flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.queue_value import (
    QueueSim,
    QueueValueModel,
    queue_ev,
)
from utils.costs import maker_bps


def _series(n, best_bid=100.0, best_ask=100.02):
    return {
        "best_bid": np.full(n, best_bid),
        "best_ask": np.full(n, best_ask),
        "exec_volume": np.zeros(n),        # volume executed at our side's best, per tick
        "level_qty": np.full(n, 50.0),     # visible queue at our level, per tick
    }


class TestReplayFills:
    def test_certain_fill_by_depletion(self):
        s = _series(20)
        s["exec_volume"][:] = 3.0                      # 3 units/tick vs ahead=10
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=10.0, horizon=20, **s)
        assert out.filled
        assert out.fill_tick == 3                      # 3+3+3 <10, 12 >= 10 at tick 3 (0-based)

    def test_no_volume_never_fills(self):
        s = _series(50)
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=5.0, horizon=50, **s)
        assert not out.filled

    def test_price_through_fills_immediately(self):
        s = _series(10)
        s["best_ask"][4:] = 99.99                      # ask crosses our bid price
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=1e9, horizon=10, **s)
        assert out.filled
        assert out.fill_tick == 4

    def test_fifo_partial_depletion_tracks_ahead(self):
        s = _series(10)
        s["exec_volume"][:] = 2.0
        sim = QueueSim()
        out = sim.simulate_post(side="bid", price=100.0, queue_ahead=9.0, horizon=4, **s)
        assert not out.filled                          # 2*4=8 < 9 within horizon
        assert out.remaining_ahead == pytest.approx(1.0)

    def test_conservative_cancels_dont_advance_queue(self):
        # level_qty collapses (cancellations) but NO executed volume -> ahead unchanged.
        s = _series(30)
        s["level_qty"][10:] = 1.0
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=5.0, horizon=30, **s)
        assert not out.filled
        assert out.remaining_ahead == pytest.approx(5.0)

    def test_best_moves_away_no_execution_at_our_price(self):
        # best bid drops below our price: executed volume at the (new) best is NOT ours.
        s = _series(30)
        s["best_bid"][5:] = 99.98
        s["exec_volume"][:] = 10.0
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=8.0, horizon=30, **s)
        # only ticks 0-4 deplete: 5*10 >= 8 -> actually fills at tick 0
        assert out.filled and out.fill_tick == 0
        # now make early volume too small to fill before the move-away
        s = _series(30)
        s["best_bid"][2:] = 99.98
        s["exec_volume"][:] = 1.0
        out = QueueSim().simulate_post(
            side="bid", price=100.0, queue_ahead=8.0, horizon=30, **s)
        assert not out.filled
        assert out.remaining_ahead == pytest.approx(6.0)   # only 2 ticks of depletion


class TestEv:
    def test_certain_fill_zero_drift_ev_is_spread_capture(self):
        half_spread = 1.0                              # bps
        capture = half_spread + maker_bps()            # SSOT rebate — never hardcoded
        ev = queue_ev(p_fill=1.0, spread_capture_bps=capture,
                      adverse_bps_given_fill=0.0)
        assert ev == pytest.approx(capture)

    def test_certain_adverse_fill_is_suppressed(self):
        capture = 1.0 + maker_bps()
        ev = queue_ev(p_fill=1.0, spread_capture_bps=capture,
                      adverse_bps_given_fill=5.0)      # drift >> capture
        assert ev < 0

    def test_never_fill_ev_zero(self):
        assert queue_ev(p_fill=0.0, spread_capture_bps=2.0,
                        adverse_bps_given_fill=99.0) == pytest.approx(0.0)

    def test_ev_monotone_in_fill_prob_when_positive_edge(self):
        evs = [queue_ev(p, 2.0, 0.5) for p in (0.1, 0.5, 0.9)]
        assert evs[0] < evs[1] < evs[2]

    def test_model_post_decision(self):
        m = QueueValueModel()
        assert m.should_post(p_fill=0.8, spread_capture_bps=2.0, adverse_bps_given_fill=0.5)
        assert not m.should_post(p_fill=0.8, spread_capture_bps=1.0, adverse_bps_given_fill=3.0)


class TestCalibration:
    def test_analytic_predictor_matches_replay_on_poisson_flow(self):
        """(g) the analytic P(fill | ahead) must calibrate to replay outcomes."""
        rng = np.random.default_rng(7)
        n_trials, horizon, lam = 400, 50, 1.0
        for ahead, tol in ((20.0, 0.12), (50.0, 0.12), (80.0, 0.12)):
            fills = 0
            samples = []
            for _ in range(n_trials):
                s = _series(horizon)
                s["exec_volume"] = rng.poisson(lam, horizon).astype(float)
                samples.append(s["exec_volume"].copy())
                out = QueueSim().simulate_post(
                    side="bid", price=100.0, queue_ahead=ahead, horizon=horizon, **s)
                fills += bool(out.filled)
            realized = fills / n_trials
            m = QueueValueModel()
            m.fit_exec_distribution(np.concatenate(samples), horizon=horizon)
            predicted = m.p_fill(queue_ahead=ahead)
            assert abs(predicted - realized) < tol, (
                f"ahead={ahead}: predicted {predicted:.3f} vs realized {realized:.3f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

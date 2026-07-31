"""Touch-maker experiment engine (FINDINGS §4.8 → §4.9, pre-registered design).

The one maker posture both honest instruments point at (§4.7 touch replay: marginal
+EV, rebate-carried; §4.8 queue-coupled A-S: −1.5..−1.9 bps/fill because risk-widened
quotes harvest price-throughs): **peg quotes at the touch — width is never derived**.
Three switchable layers, each an experiment cell, none re-tuned after results:

    side-selection (HF1) : dev > +θ → bid only; dev < −θ → ask only; else both;
    inventory skew       : |q| ≥ q_soft → reduce-only;
    HF4 gate             : toxic flow → no quotes;
    EV gate (A4)         : post only while capture (live half-spread + SSOT rebate)
                           exceeds a CAUSAL adverse-selection estimate — an EWMA of
                           this sim's own past fills' markout (updated markout_ticks
                           after each fill; initialized at the §4.7-measured prior).

Fill mechanics inherited verbatim from ``ASQueueSim`` (HF5b): conservative joins
(l1_fraction·depth on every post), priority lost on requote, touch-zone FIFO
depletion by side aggressor flow, price-through, never a crossing quote, no RNG.
Costs via ``load_costs()`` only. Sim-first: no live path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.costs import maker_bps, taker_bps


@dataclass(frozen=True)
class TouchParams:
    theta_dev_bps: float = 0.05      # HF1 side-selection threshold on alg_mp_dev_ema
    q_soft: float = 3.0              # reduce-only inventory bound
    requote_every: int = 5           # ticks between cancel+repost cycles
    l1_fraction: float = 0.4         # conservative queue join: ahead = frac · depth
    use_hf1_side: bool = False
    use_inv_skew: bool = False
    use_hf4_gate: bool = False
    use_ev_gate: bool = False
    adverse_prior_bps: float = 0.25  # §4.7-measured E[adverse|fill] prior (0.22–0.26)
    markout_ticks: int = 50          # causal markout delay for the adverse EWMA
    adverse_ewma_alpha: float = 0.05


@dataclass
class _Order:
    price: float
    ahead: float


class TouchMakerSim:
    """Deterministic touch-pegged maker episode over explicit per-tick series."""

    def __init__(self, params: TouchParams):
        self.p = params

    def run(self, mid, best_bid, best_ask, sell_exec, buy_exec,
            depth_bid, depth_ask, fair_dev_bps, gate_open) -> dict:
        p = self.p
        n = len(mid)
        cash = 0.0
        q = 0.0
        n_fills = 0
        n_postings = 0
        inv_path = np.empty(n)
        rebate = maker_bps() * 1e-4
        adverse_est = p.adverse_prior_bps
        pending: list[tuple[int, str, float]] = []    # (fill_tick, side, fill_mid)
        bid_o: Optional[_Order] = None
        ask_o: Optional[_Order] = None
        first_fill: Optional[int] = None

        def _record_fill(t: int, side: str) -> None:
            nonlocal n_fills, first_fill
            n_fills += 1
            first_fill = t if first_fill is None else first_fill
            pending.append((t, side, float(mid[t])))

        for t in range(n):
            # causal adverse-EWMA update from own past fills, markout_ticks later
            while pending and t - pending[0][0] >= p.markout_ticks:
                ft, side, fm = pending.pop(0)
                if fm > 0 and np.isfinite(mid[t]):
                    drift_bps = (mid[t] / fm - 1.0) * 1e4
                    adv = -drift_bps if side == "bid" else drift_bps
                    adverse_est = ((1.0 - p.adverse_ewma_alpha) * adverse_est
                                   + p.adverse_ewma_alpha * adv)

            if t % p.requote_every == 0:
                bid_o = ask_o = None                  # cancel: priority never carries
                want_bid = want_ask = True
                if p.use_hf4_gate and not bool(gate_open[t]):
                    want_bid = want_ask = False
                if p.use_ev_gate and (want_bid or want_ask):
                    half_spread_bps = (best_ask[t] - best_bid[t]) / 2.0 / mid[t] * 1e4
                    capture = half_spread_bps + maker_bps()
                    if capture <= adverse_est:
                        want_bid = want_ask = False
                if p.use_hf1_side:
                    dev = float(fair_dev_bps[t])
                    if dev > p.theta_dev_bps:
                        want_ask = False              # upward pressure: buy side only
                    elif dev < -p.theta_dev_bps:
                        want_bid = False
                if p.use_inv_skew:
                    if q >= p.q_soft:
                        want_bid = False              # long: reduce-only
                    if q <= -p.q_soft:
                        want_ask = False
                if want_bid and np.isfinite(best_bid[t]):
                    bid_o = _Order(float(best_bid[t]),
                                   p.l1_fraction * float(depth_bid[t]))
                    n_postings += 1
                if want_ask and np.isfinite(best_ask[t]):
                    ask_o = _Order(float(best_ask[t]),
                                   p.l1_fraction * float(depth_ask[t]))
                    n_postings += 1

            if bid_o is not None:
                if best_ask[t] <= bid_o.price:                      # price-through
                    cash -= bid_o.price
                    cash += rebate * bid_o.price
                    q += 1.0
                    _record_fill(t, "bid")
                    bid_o = None
                elif best_bid[t] <= bid_o.price:                    # touch zone
                    bid_o.ahead -= float(sell_exec[t])
                    if bid_o.ahead <= 0.0:
                        cash -= bid_o.price
                        cash += rebate * bid_o.price
                        q += 1.0
                        _record_fill(t, "bid")
                        bid_o = None

            if ask_o is not None:
                if best_bid[t] >= ask_o.price:                      # price-through
                    cash += ask_o.price
                    cash += rebate * ask_o.price
                    q -= 1.0
                    _record_fill(t, "ask")
                    ask_o = None
                elif best_ask[t] >= ask_o.price:                    # touch zone
                    ask_o.ahead -= float(buy_exec[t])
                    if ask_o.ahead <= 0.0:
                        cash += ask_o.price
                        cash += rebate * ask_o.price
                        q -= 1.0
                        _record_fill(t, "ask")
                        ask_o = None

            inv_path[t] = q

        mid_end = float(mid[-1])
        liq_cost = abs(q) * mid_end * taker_bps() * 1e-4
        pnl = cash + q * mid_end - liq_cost
        mid0 = float(mid[0])
        return {
            "pnl_bps": pnl / mid0 * 1e4,
            "n_fills": n_fills,
            "n_postings": n_postings,
            "first_fill_tick": first_fill,
            "terminal_inventory": q,
            "mean_inventory": float(inv_path.mean()),
            "max_abs_inventory": float(np.abs(inv_path).max()),
            "liquidation_cost_bps": liq_cost / mid0 * 1e4,
            "adverse_est_final_bps": adverse_est,
            "taker_bps_used": taker_bps(),
            "maker_bps_used": maker_bps(),
        }

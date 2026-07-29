"""
GAP-04: microprice-centered two-sided maker simulation (HF1, seed of HF5).

Quote BOTH sides around a forward-looking fair price (the microprice), skew on inventory,
pull quotes when toxicity is high, cap inventory. Sim-first — this measures fill MARKOUT
(post-fill drift), the direct test of whether the microstructure edge is maker-capturable:
does microprice-centering + toxicity-pulling move markout positive where naive maker
quoting is adversely selected? No paper/live path (docs/GAP__26_7_26.md/04).

Degenerate Avellaneda-Stoikov (fixed risk aversion, no terminal time):
    reservation = center - eta * inventory * sigma      # inventory skew
    delta       = 0.5 * spread + c_vol * sigma          # half-width: spread floor + vol cushion
    bid = reservation - delta        ask = reservation + delta
    quote a side only if inventory has room (|inv| < i_max); pull both when the gate is closed.

Reuses the conservative fill model + markout of scripts/kalman/fill_sim.py: a resting
quote fills ONLY when the mid trades THROUGH it (no optimistic touches), and markout is
`compute_post_fill_drift` (positive = price moved in our favour after the fill).
"""

from __future__ import annotations

import numpy as np

from kalman.fill_sim import FillEvent


class MicropriceMakerSim:
    """Two-sided, inventory-capped, toxicity-gated maker quoting around a fair-value center."""

    def __init__(self, eta: float = 0.0, c_vol: float = 0.0, i_max: float = 5.0,
                 size: float = 1.0, latency: int = 1, timeout: int = 50, min_gap: int = 5):
        self.eta = eta          # inventory-skew coefficient
        self.c_vol = c_vol      # vol cushion added to the half-spread
        self.i_max = i_max      # inventory cap (absolute)
        self.size = size        # fill size per execution
        self.latency = latency  # ticks between quote decision and the order resting
        self.timeout = timeout  # ticks a quote rests before it is pulled
        self.min_gap = min_gap  # min ticks between successive quote decisions

    def simulate(self, mid, center, spread, vol, gate_open) -> list[FillEvent]:
        """Replay two-sided quoting over a mid-price path; return the fills.

        Args:
            mid:       mid-price series.
            center:    quote center (e.g. microprice, mid, or VWAP — the ablation axis).
            spread:    spread in price units (half-spread = spread/2 is the floor).
            vol:       volatility series (units of price) for skew + cushion.
            gate_open: per-tick bool — quote only where True (pull on toxicity).
        """
        mid = np.asarray(mid, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        spread = np.asarray(spread, dtype=np.float64)
        vol = np.asarray(vol, dtype=np.float64)
        gate_open = np.asarray(gate_open, dtype=bool)
        n = len(mid)

        fills: list[FillEvent] = []
        inv = 0.0
        last = -self.min_gap
        i = 0
        while i < n - self.timeout - self.latency:
            if not gate_open[i] or (i - last) < self.min_gap:
                i += 1
                continue
            if not (np.isfinite(center[i]) and np.isfinite(spread[i]) and np.isfinite(vol[i])) \
                    or spread[i] <= 0:
                i += 1
                continue

            reservation = center[i] - self.eta * inv * vol[i]
            delta = 0.5 * spread[i] + self.c_vol * vol[i]
            bid = reservation - delta
            ask = reservation + delta
            post_bid = inv < self.i_max - 1e-9      # room to buy
            post_ask = inv > -self.i_max + 1e-9     # room to sell
            if not (post_bid or post_ask):
                i += 1
                continue

            order_tick = i + self.latency
            last = i
            filled_at = None
            for j in range(order_tick + 1, min(order_tick + self.timeout, n)):
                # Conservative: fill only when the mid trades THROUGH the resting quote.
                if post_bid and mid[j] <= bid:
                    inv += self.size
                    fills.append(FillEvent(i, j, float(bid), "buy", 0.0, np.nan, float(mid[j])))
                    filled_at = j
                    break
                if post_ask and mid[j] >= ask:
                    inv -= self.size
                    fills.append(FillEvent(i, j, float(ask), "sell", 0.0, np.nan, float(mid[j])))
                    filled_at = j
                    break

            i = (filled_at + 1) if filled_at is not None else (order_tick + self.timeout)

        return fills

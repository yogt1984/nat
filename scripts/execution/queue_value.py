"""A4: queue-value execution model — is this resting limit order +EV? (sim-first)

The micro-decision underneath every maker quote (GAP spec 07; Huang, Lehalle &
Rosenbaum 2015 lineage): an order posted at level ``L`` with ``k`` volume ahead
earns the spread iff it fills, and fills are adversely selected. The model:

    EV(post) = P(fill | k, flow) · spread_capture − P(fill) · E[adverse | fill]
             = P(fill) · (spread_capture_bps − adverse_bps_given_fill)

When EV < 0: don't post (or cross instead). This gives HF5 per-order granularity
and gates any maker profit claim (Q4 kill-gate finding: maker economics must be
proven through queue position, never assumed).

Components
----------
``QueueSim``          replay engine over per-tick series (input-agnostic):
                      FIFO depletion of the volume ahead by executed volume at
                      our price while we are at the best; price-through fills
                      immediately. CONSERVATIVE by construction: cancellations
                      (visible level shrinkage that is not trades) never advance
                      the queue — realized fill rates are a lower bound.
``QueueValueModel``   analytic P(fill | ahead) from a fitted executed-volume
                      distribution (normal approximation of the horizon sum,
                      continuity-corrected) + the EV rule ``should_post``.
``queue_ev``          the EV formula (bps).

The engine takes explicit series (best_bid/best_ask/exec_volume/level_qty) so the
planted contract is exact; the parquet adapter (``replay_from_frame``) derives
PROXIES from the 100 ms snapshot columns — documented as such, because the
ingestor emits depth aggregates, not L1 queue sizes or order-by-order events.

Costs: maker rebate via ``utils.costs`` (``load_costs()``) — never hardcoded.
Sim-first: no live-order path exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.costs import maker_bps  # SSOT — spread-capture helpers only


@dataclass(frozen=True)
class FillOutcome:
    """Result of replaying one hypothetical resting order."""
    filled: bool
    fill_tick: Optional[int]           # tick index of the fill (None if unfilled)
    remaining_ahead: float             # volume still ahead at the end (0 if filled)


class QueueSim:
    """Conservative FIFO queue replay for one resting order.

    Rules (all deliberately pessimistic — realized fills are a lower bound):
      1. price-through: the opposite best crossing our price fills us immediately
         (someone traded through the level);
      2. depletion: executed volume at our price advances the queue ONLY while our
         side's best equals our price (volume at a different best is not ours);
      3. cancellations never advance the queue: visible level shrinkage beyond
         trades is assumed to happen entirely BEHIND us.
    """

    def simulate_post(
        self,
        side: str,
        price: float,
        queue_ahead: float,
        horizon: int,
        best_bid: np.ndarray,
        best_ask: np.ndarray,
        exec_volume: np.ndarray,
        level_qty: np.ndarray,          # accepted for future pro-rata modes; unused
    ) -> FillOutcome:
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
        ahead = float(queue_ahead)
        n = min(int(horizon), len(best_bid))
        for t in range(n):
            # 1. price-through
            if side == "bid" and best_ask[t] <= price:
                return FillOutcome(True, t, 0.0)
            if side == "ask" and best_bid[t] >= price:
                return FillOutcome(True, t, 0.0)
            # 2. FIFO depletion while at the best
            at_best = (best_bid[t] == price) if side == "bid" else (best_ask[t] == price)
            if at_best:
                ahead -= float(exec_volume[t])
                if ahead <= 0.0:
                    return FillOutcome(True, t, 0.0)
            # 3. cancellations: never advance (nothing to do)
        return FillOutcome(False, None, ahead)


def queue_ev(
    p_fill: float,
    spread_capture_bps: float,
    adverse_bps_given_fill: float,
) -> float:
    """Expected value (bps) of posting: P(fill)·(capture − adverse-selection cost).

    ``spread_capture_bps`` should already include the maker rebate (see
    ``QueueValueModel.spread_capture_bps``); ``adverse_bps_given_fill`` is the
    expected post-fill drift AGAINST the order, in bps.
    """
    return float(p_fill) * (float(spread_capture_bps) - float(adverse_bps_given_fill))


class QueueValueModel:
    """Analytic fill-probability + EV posting rule.

    ``fit_exec_distribution`` learns the per-tick executed-volume distribution
    (mean/var); ``p_fill(ahead)`` is P(sum over the horizon ≥ ahead) under a
    continuity-corrected normal approximation of the horizon sum — the same
    event the replay engine realizes (non-negative increments: the cumulative
    sum reaches ``ahead`` within H iff the H-sum is ≥ ahead).
    """

    def __init__(self, min_ev_bps: float = 0.0):
        self.min_ev_bps = float(min_ev_bps)
        self._mu: Optional[float] = None
        self._var: Optional[float] = None
        self._horizon: Optional[int] = None

    # ── fill probability ─────────────────────────────────────────────────────
    def fit_exec_distribution(self, exec_samples: np.ndarray, horizon: int) -> None:
        x = np.asarray(exec_samples, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            raise ValueError("need >= 2 finite executed-volume samples to fit")
        self._mu = float(x.mean())
        self._var = float(x.var(ddof=1))
        self._horizon = int(horizon)

    def p_fill(self, queue_ahead: float, horizon: Optional[int] = None) -> float:
        if self._mu is None:
            raise RuntimeError("call fit_exec_distribution first")
        h = int(horizon) if horizon is not None else self._horizon
        mean_sum = h * self._mu
        sd_sum = float(np.sqrt(max(h * self._var, 1e-20)))
        # continuity-corrected normal tail: P(S_H >= ahead)
        from scipy.stats import norm
        z = (float(queue_ahead) - 0.5 - mean_sum) / sd_sum
        return float(norm.sf(z))

    # ── EV rule ──────────────────────────────────────────────────────────────
    @staticmethod
    def spread_capture_bps(half_spread_bps: float) -> float:
        """Half-spread earned at the touch plus the SSOT maker rebate (one-way)."""
        return float(half_spread_bps) + maker_bps()

    def should_post(
        self,
        p_fill: float,
        spread_capture_bps: float,
        adverse_bps_given_fill: float,
    ) -> bool:
        return queue_ev(p_fill, spread_capture_bps, adverse_bps_given_fill) > self.min_ev_bps


# ───────────────────────── parquet adapter (proxy-based) ─────────────────────

#: Columns the replay adapter needs from the 100 ms snapshot parquet.
ADAPTER_COLUMNS = [
    "raw_midprice", "raw_spread", "raw_spread_bps",
    "raw_bid_depth_5", "raw_ask_depth_5",
    "flow_volume_1s", "flow_aggressor_ratio_5s",
]


def replay_from_frame(
    df,
    side: str = "bid",
    post_every: int = 100,
    horizon: int = 300,
    l1_fraction: float = 0.4,
    markout_ticks: int = 50,
):
    """Replay hypothetical postings over a snapshot frame; returns outcome records.

    PROXIES (the ingestor emits aggregates, not L1 queue sizes / MBO events —
    every claim downstream must carry this caveat):
      - best bid/ask     = mid ∓ raw_spread/2;
      - exec volume/tick = flow_volume_1s / 10 (10 Hz ticks) split by the 5 s
        aggressor ratio (sell-aggressor volume hits bids);
      - queue ahead      = l1_fraction · depth_5 of our side (join at the back).

    Each record: {tick, filled, fill_tick, queue_ahead0, adverse_bps} where
    adverse_bps is the post-fill mid drift AGAINST the order over
    ``markout_ticks`` (the adverse-selection cost the EV rule needs).
    """
    mid = df["raw_midprice"].to_numpy(dtype=np.float64, na_value=np.nan)
    spread = df["raw_spread"].to_numpy(dtype=np.float64, na_value=np.nan)
    depth = df["raw_bid_depth_5" if side == "bid" else "raw_ask_depth_5"].to_numpy(
        dtype=np.float64, na_value=np.nan)
    vol1s = df["flow_volume_1s"].to_numpy(dtype=np.float64, na_value=np.nan)
    aggr = df["flow_aggressor_ratio_5s"].to_numpy(dtype=np.float64, na_value=np.nan)

    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    side_ratio = (1.0 - aggr) if side == "bid" else aggr   # sell flow hits bids
    exec_vol = np.nan_to_num(vol1s / 10.0 * side_ratio, nan=0.0)

    sim = QueueSim()
    records = []
    n = len(df)
    for t0 in range(0, n - horizon - markout_ticks, post_every):
        if not (np.isfinite(best_bid[t0]) and np.isfinite(depth[t0])):
            continue
        price = best_bid[t0] if side == "bid" else best_ask[t0]
        ahead0 = l1_fraction * depth[t0]
        sl = slice(t0, t0 + horizon)
        out = sim.simulate_post(
            side=side, price=price, queue_ahead=ahead0, horizon=horizon,
            best_bid=best_bid[sl], best_ask=best_ask[sl],
            exec_volume=exec_vol[sl], level_qty=depth[sl],
        )
        adverse = np.nan
        if out.filled:
            tf = t0 + out.fill_tick
            m0, m1 = mid[tf], mid[min(tf + markout_ticks, n - 1)]
            if np.isfinite(m0) and np.isfinite(m1) and m0 > 0:
                drift_bps = (m1 / m0 - 1.0) * 1e4
                adverse = -drift_bps if side == "bid" else drift_bps
        records.append({
            "tick": t0, "filled": out.filled, "fill_tick": out.fill_tick,
            "queue_ahead0": ahead0, "adverse_bps": adverse,
        })
    return records

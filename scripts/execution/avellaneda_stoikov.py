"""HF5: Avellaneda–Stoikov maker — the maker line composed (sim-first).

Avellaneda & Stoikov (2008): quote around a reservation price that skews with
inventory, at the risk-optimal spread, with fills arriving at intensity
λ(d) = A·exp(−κ·d) in the distance d from mid:

    r(t)      = s(t) − q·γ·σ²·τ                       reservation price (bps space)
    δ_total   = γ·σ²·τ + (2/γ)·ln(1 + γ/κ)            optimal bid+ask spread
    bid/ask   = r ∓ δ_total/2

Composition of the maker line built before it:
  - **HF1**: the fair value s = mid + microprice deviation (``alg_mp_dev_bps``) —
    quotes center on the calibrated expected mid, not the mid itself;
  - **HF4**: the toxicity gate — both quotes pulled when flow is toxic;
  - **A4**: κ (and A) calibrated from replayed fill rates vs posting distance
    (``calibrate_kappa``), instead of invented;
  - honest accounting: maker rebate per fill and terminal inventory liquidation
    at taker cost, both via ``load_costs()`` (never hardcoded).

Units: quotes and deviations in **bps offsets from mid**; σ in bps/√tick; τ in
ticks (rolling-horizon practical form — τ fixed per decision, the standard
industry degeneration of the finite-T model).

Sim-first: no live-order path exists here. **No live capital before G8 + a
healthy kill-switch** (the HF5 dependency that gates any graduation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.costs import maker_bps, taker_bps


@dataclass(frozen=True)
class ASParams:
    gamma: float = 0.1          # risk aversion
    kappa: float = 1.5          # fill-intensity decay (calibrate via calibrate_kappa)
    tau_ticks: float = 100.0    # rolling horizon (ticks)
    q_max: float = 5.0          # inventory hard cap (units); breach → reduce-only


@dataclass(frozen=True)
class Quotes:
    """Quote offsets from mid in bps; None = side pulled."""
    bid_bps: Optional[float]
    ask_bps: Optional[float]


class ASQuoter:
    """Pure quoting rule: reservation price + optimal spread + gate + cap."""

    def __init__(self, params: ASParams):
        self.p = params

    def reservation_bps(self, fair_dev_bps: float, inventory: float,
                        sigma_bps: float) -> float:
        return fair_dev_bps - inventory * self.p.gamma * sigma_bps**2 * self.p.tau_ticks

    def total_spread_bps(self, sigma_bps: float) -> float:
        g, k = self.p.gamma, self.p.kappa
        return g * sigma_bps**2 * self.p.tau_ticks + (2.0 / g) * np.log(1.0 + g / k)

    def quotes_bps(self, fair_dev_bps: float, inventory: float, sigma_bps: float,
                   gate_open: bool) -> Quotes:
        if not gate_open:
            return Quotes(None, None)                  # HF4: pull on toxicity
        r = self.reservation_bps(fair_dev_bps, inventory, sigma_bps)
        half = self.total_spread_bps(sigma_bps) / 2.0
        bid: Optional[float] = r - half
        ask: Optional[float] = r + half
        if inventory >= self.p.q_max:
            bid = None                                 # long-capped: reduce-only
        elif inventory <= -self.p.q_max:
            ask = None                                 # short-capped: reduce-only
        return Quotes(bid, ask)


def calibrate_kappa(observations) -> tuple[float, float]:
    """Fit λ(d) = A·exp(−κ·d) from replayed fill counts (the A4 instrument).

    ``observations``: iterable of (distance_bps, n_fills, n_postings). Log-linear
    least squares on the empirical per-posting fill rate; zero-fill rows dropped.
    Returns (A_hat, kappa_hat).
    """
    d, r = [], []
    for dist, fills, posts in observations:
        if posts > 0 and fills > 0:
            d.append(float(dist))
            r.append(float(fills) / float(posts))
    if len(d) < 2:
        raise ValueError("need >= 2 nonzero fill-rate observations to calibrate")
    slope, intercept = np.polyfit(np.asarray(d), np.log(np.asarray(r)), 1)
    return float(np.exp(intercept)), float(-slope)


class ASSim:
    """Seeded Monte-Carlo maker episode: A-S quotes + λ(d) fills + honest accounting.

    Per tick and side: fill with probability min(1, A·exp(−κ·d)), d ≥ 0 the quote's
    distance from mid (a quote at/through mid fills with probability A capped at 1).
    Each fill trades 1 unit at the quoted price and earns the SSOT maker rebate.
    Terminal inventory is liquidated at the last mid, charged the SSOT taker fee.
    """

    def __init__(self, params: ASParams, fill_A: float, fill_kappa: float, seed: int = 0):
        self.params = params
        self.quoter = ASQuoter(params)
        self.fill_A = float(fill_A)
        self.fill_kappa = float(fill_kappa)
        self._rng = np.random.default_rng(seed)

    def _p_fill(self, dist_bps: float) -> float:
        return min(1.0, self.fill_A * np.exp(-self.fill_kappa * max(dist_bps, 0.0)))

    def run(self, mid: np.ndarray, fair_dev_bps: np.ndarray, sigma_bps: np.ndarray,
            gate_open: np.ndarray) -> dict:
        n = len(mid)
        cash = 0.0
        q = 0.0
        n_fills = 0
        inv_path = np.empty(n)
        rebate = maker_bps() * 1e-4

        for t in range(n):
            quotes = self.quoter.quotes_bps(
                float(fair_dev_bps[t]), q, float(sigma_bps[t]), bool(gate_open[t]))
            m = mid[t]
            if quotes.bid_bps is not None:
                price = m * (1.0 + quotes.bid_bps * 1e-4)
                if self._rng.random() < self._p_fill(-quotes.bid_bps):
                    q += 1.0
                    cash -= price
                    cash += rebate * price
                    n_fills += 1
            if quotes.ask_bps is not None:
                price = m * (1.0 + quotes.ask_bps * 1e-4)
                if self._rng.random() < self._p_fill(quotes.ask_bps):
                    q -= 1.0
                    cash += price
                    cash += rebate * price
                    n_fills += 1
            inv_path[t] = q

        mid_end = float(mid[-1])
        liq_cost = abs(q) * mid_end * taker_bps() * 1e-4
        pnl = cash + q * mid_end - liq_cost
        mid0 = float(mid[0])
        return {
            "pnl_bps": pnl / mid0 * 1e4,
            "n_fills": n_fills,
            "terminal_inventory": q,
            "mean_inventory": float(inv_path.mean()),
            "max_abs_inventory": float(np.abs(inv_path).max()),
            "liquidation_cost_bps": liq_cost / mid0 * 1e4,
            "taker_bps_used": taker_bps(),
            "maker_bps_used": maker_bps(),
        }


@dataclass
class _RestingOrder:
    price: float
    ahead: float


class ASQueueSim:
    """HF5b: the A-S quoter coupled to A4's conservative queue rules — deterministic.

    §4.8 established that exogenous λ(d) fills carry no adverse selection, so
    absolute ``ASSim`` PnL is fantasy. Here fills are EARNED from the tape:

      through : the opposite best crossing the resting price fills it;
      touch   : while the price sits in the touch zone (best_bid ≤ p < best_ask for
                bids), side aggressor flow depletes ``ahead`` — initialized at
                ``l1_fraction · depth`` on EVERY post, even inside the spread
                (latent competition assumed: no "alone at the level" free lunch);
      behind  : outside the touch zone nothing advances (cancellations never do);
      requote : cancel + repost every ``requote_every`` ticks — priority is never
                carried across posts; a quote that would cross at post time is NOT
                placed (no fantasy taker fills).

    Adverse selection therefore emerges structurally: a bid is consumed exactly by
    the sell flow that tends to precede down-moves. No RNG — identical inputs give
    identical episodes. Accounting: SSOT maker rebate per fill; terminal inventory
    liquidated at the SSOT taker fee. Sim-only.
    """

    def __init__(self, params: ASParams, requote_every: int = 5,
                 l1_fraction: float = 0.4):
        self.params = params
        self.quoter = ASQuoter(params)
        self.requote_every = int(requote_every)
        self.l1_fraction = float(l1_fraction)

    def run(self, mid, best_bid, best_ask, sell_exec, buy_exec,
            depth_bid, depth_ask, fair_dev_bps, sigma_bps, gate_open) -> dict:
        n = len(mid)
        cash = 0.0
        q = 0.0
        n_fills = 0
        first_fill: Optional[int] = None
        inv_path = np.empty(n)
        rebate = maker_bps() * 1e-4
        bid_o: Optional[_RestingOrder] = None
        ask_o: Optional[_RestingOrder] = None

        for t in range(n):
            if t % self.requote_every == 0:
                bid_o = ask_o = None                   # cancel: priority never carries
                quotes = self.quoter.quotes_bps(
                    float(fair_dev_bps[t]), q, float(sigma_bps[t]), bool(gate_open[t]))
                if quotes.bid_bps is not None:
                    p = mid[t] * (1.0 + quotes.bid_bps * 1e-4)
                    if p < best_ask[t]:                # never place a crossing quote
                        bid_o = _RestingOrder(p, self.l1_fraction * float(depth_bid[t]))
                if quotes.ask_bps is not None:
                    p = mid[t] * (1.0 + quotes.ask_bps * 1e-4)
                    if p > best_bid[t]:
                        ask_o = _RestingOrder(p, self.l1_fraction * float(depth_ask[t]))

            if bid_o is not None:
                if best_ask[t] <= bid_o.price:                      # price-through
                    cash -= bid_o.price
                    cash += rebate * bid_o.price
                    q += 1.0
                    n_fills += 1
                    first_fill = t if first_fill is None else first_fill
                    bid_o = None
                elif best_bid[t] <= bid_o.price:                    # touch zone
                    bid_o.ahead -= float(sell_exec[t])
                    if bid_o.ahead <= 0.0:
                        cash -= bid_o.price
                        cash += rebate * bid_o.price
                        q += 1.0
                        n_fills += 1
                        first_fill = t if first_fill is None else first_fill
                        bid_o = None

            if ask_o is not None:
                if best_bid[t] >= ask_o.price:                      # price-through
                    cash += ask_o.price
                    cash += rebate * ask_o.price
                    q -= 1.0
                    n_fills += 1
                    first_fill = t if first_fill is None else first_fill
                    ask_o = None
                elif best_ask[t] >= ask_o.price:                    # touch zone
                    ask_o.ahead -= float(buy_exec[t])
                    if ask_o.ahead <= 0.0:
                        cash += ask_o.price
                        cash += rebate * ask_o.price
                        q -= 1.0
                        n_fills += 1
                        first_fill = t if first_fill is None else first_fill
                        ask_o = None

            inv_path[t] = q

        mid_end = float(mid[-1])
        liq_cost = abs(q) * mid_end * taker_bps() * 1e-4
        pnl = cash + q * mid_end - liq_cost
        mid0 = float(mid[0])
        return {
            "pnl_bps": pnl / mid0 * 1e4,
            "n_fills": n_fills,
            "first_fill_tick": first_fill,
            "terminal_inventory": q,
            "mean_inventory": float(inv_path.mean()),
            "max_abs_inventory": float(np.abs(inv_path).max()),
            "liquidation_cost_bps": liq_cost / mid0 * 1e4,
            "taker_bps_used": taker_bps(),
            "maker_bps_used": maker_bps(),
        }

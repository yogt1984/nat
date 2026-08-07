"""A5 — hysteresis no-trade bands and TWAP/VWAP slicing.

**Maturity: PRELIM.** Two execution primitives that act on *cost* rather than on signal.
Neither invents an edge; they stop paying for trades that do not earn their fee.

**Hysteresis.** Rebalancing to target every period pays the full spread on every drift,
including drifts too small to be worth correcting. Under *proportional* transaction costs
the optimal policy is not "trade when the drift is large enough" but a **no-trade region
with trading to its boundary** (Constantinides 1986; Davis & Norman 1990): once outside the
band you move only to the edge, keeping the last increment untraded because correcting it
costs more than it earns. Both are provided — `no_trade_band` (the simple form, trade fully
or not at all) and `trade_to_edge` (the theoretically optimal one).

Relevance here: the XS rotation turns over 0.199 of gross per rebalance and spends 1.10 %
of an 8.18 % gross return on cost (FINDINGS §7.8). That is the line these act on.

**TWAP / VWAP slicing.** Split a parent order across a window, evenly or in proportion to
expected volume. Provided as primitives, with an explicit warning:

    Slicing exists to reduce **market impact**. NAT's cost model is
    spread + fee + slippage per unit of turnover, with **no impact term**, so in every
    simulation currently in this repo slicing measures as **exactly zero benefit**. It is
    not a free win being left on the table — it is unpriceable until the F-task fill data
    (`X-3`) exists. Do not report a slicing improvement from the present harness.

Sizing a band: `band_from_cost` uses the standard proportional-cost intuition that the
no-trade width scales with cost and shrinks as the expected edge grows. It is a heuristic,
not an imported gate — the operating point belongs to whatever study uses it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["no_trade_band", "trade_to_edge", "band_from_cost",
           "twap_slices", "vwap_slices"]


def _as_band(band, index) -> pd.Series:
    """Broadcast a scalar band, or align a per-position one."""
    if isinstance(band, pd.Series):
        return band.reindex(index).astype(float)
    return pd.Series(float(band), index=index)


def no_trade_band(target: pd.Series, current: pd.Series, band) -> pd.Series:
    """Hold `current` where the drift is within `band`; otherwise move fully to `target`.

    The band is applied **per position**: one large drift must not drag small ones along
    with it, which is what a portfolio-level trigger would do.

    `band` may be a scalar or a Series — a per-pair band is the honest form, since a cheap
    liquid pair deserves a tighter no-trade region than an expensive thin one.
    """
    tgt = pd.Series(target, dtype=float)
    cur = pd.Series(current, dtype=float).reindex(tgt.index).fillna(0.0)
    b = _as_band(band, tgt.index)
    move = (tgt - cur).abs() > b
    return tgt.where(move, cur)


def trade_to_edge(target: pd.Series, current: pd.Series, band) -> pd.Series:
    """Move only to the **boundary** of the no-trade region — the optimal proportional-cost
    policy (Constantinides 1986).

    Inside the band: no trade. Outside: close the gap to within `band` of the target, not
    all the way to it. The final increment is deliberately left untraded because correcting
    it costs more than it earns, and rebalancing fully throws that away every period.
    """
    tgt = pd.Series(target, dtype=float)
    cur = pd.Series(current, dtype=float).reindex(tgt.index).fillna(0.0)
    b = _as_band(band, tgt.index)

    delta = tgt - cur
    excess = delta.abs() - b                     # how far outside the band
    step = np.sign(delta) * excess.clip(lower=0.0)
    return cur + step


def band_from_cost(cost_bps: float, alpha_bps: float, multiple: float = 2.0,
                   position_scale: float = 1.0) -> float:
    """Heuristic no-trade half-width.

    `multiple * cost_bps / alpha_bps` is **dimensionless** — a ratio of cost to expected
    edge — so it must be scaled by a characteristic position size to become a band in
    position units. `position_scale` does that; leaving it at 1.0 returns the raw ratio.

    Getting this wrong is easy and expensive: for a unit-gross book of ~120 names the
    typical weight is ~0.008, while the raw ratio at NAT's costs is ~1.4. Applied directly
    as a weight band that means *never trade*, which would look like a strategy that
    mysteriously stopped rebalancing rather than a units error.

    Scales with round-trip cost and shrinks as the expected per-period edge grows: a
    stronger signal justifies trading through more cost. `multiple` is the "~2x round-trip
    cost" convention.

    A non-positive expected edge raises: with no edge *no* trade is justified, and
    returning an infinitely tight band would say the opposite.
    """
    if alpha_bps <= 0:
        raise ValueError(
            f"alpha_bps must be positive (got {alpha_bps}): with no expected edge no "
            "rebalancing is justified, which is not the same as a zero-width band"
        )
    return float(multiple) * float(cost_bps) / float(alpha_bps) * float(position_scale)


def twap_slices(quantity: float, n_slices: int) -> list[float]:
    """Split `quantity` into `n_slices` equal parts. Conserves quantity exactly."""
    if n_slices <= 0:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    return [float(quantity) / n_slices] * int(n_slices)


def vwap_slices(quantity: float, volume_profile: Sequence[float]) -> list[float]:
    """Split `quantity` in proportion to an expected volume profile.

    Degrades to TWAP on a flat profile, which is the sanity check that the weighting is
    doing what it claims. Rejects an empty, all-zero or negative profile rather than
    silently falling back — a bad profile should be a loud error, not an invisible TWAP.
    """
    v = np.asarray(list(volume_profile), dtype=float)
    if v.size == 0:
        raise ValueError("volume_profile is empty")
    if np.any(v < 0):
        raise ValueError("volume_profile contains negative volume")
    total = v.sum()
    if total <= 0:
        raise ValueError("volume_profile sums to zero — cannot weight by it")
    return [float(quantity) * float(x) / total for x in v]

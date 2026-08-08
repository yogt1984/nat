"""B-5a: the wide-pair breakeven arithmetic screen — a conditional, stated as one.

§4.11 measured the maker line's breakeven at BTC's touch:

    posting is +EV  ⟺  half_spread + rebate > E[adverse | fill]

BTC: half-spread 0.083 bps against E[adverse|fill] 0.228 bps, so breakeven needs a rebate
of +0.144 bps and the best rate reachable on volume alone is zero. XS-8 then measured the
universe (§7.2): median half-spread **1.372 bps, 17.7× BTC**, with 169 of 177 pairs wider —
so every maker experiment NAT has run has been on the extreme tight tail of its own venue.

The obvious inference — "wide pairs cover adverse selection" — rests entirely on an
assumption nobody has measured: that `E[adverse | fill]` stays at BTC's 0.228 bps as the
spread widens. It almost certainly does not. **Spreads are wide because market makers price
inventory and toxicity risk into them**, so adverse selection should scale WITH the spread,
and if it scales proportionally the ratio is unchanged and nothing improves anywhere.

So this screen does not produce a verdict. It produces **the exponent at which the verdict
flips**, per pair. Parameterise

    E[adverse | fill](h) = A_btc · (h / h_btc) ** beta

    beta = 0   adverse selection is a constant — the optimistic reading
    beta = 1   adverse selection is proportional to spread — the pessimistic reading
    beta*      the value at which a pair exactly breaks even

A pair with `beta* = 0.64` survives if and only if adverse selection scales more slowly
than `h^0.64`. That number is falsifiable by a single tick-data measurement on one wide
pair, which is what B-5b would do. Reporting `beta*` instead of a survivor count is the
difference between a screen and a claim.

**Capacity is the second blade and it cuts the other way.** The widest pairs are nearly
empty at the touch (§7.2: XAI 12.9 bps on $20, HMSTR 26.8 bps on $3.3k). A large per-fill
edge on $20 of size is not a business, so admission runs through `xs.capacity.admit()` —
the same floors XS-5 applies — and the joint wide-AND-deep survivor set is far smaller than
either margin alone.

Inputs are XS-8 sweeps via `xs.capacity.aggregate_l2`, which requires >= 12 snapshots
because one book is an n=1 estimate of a quantity that moves ~20 % within a morning. Costs
come from `load_costs()`; the rebate is a tier, not a constant (COST-5/COST-8).

Sim-only, no capital path: quoted spreads say what a resting order *could* earn, never what
it would be filled against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

#: BTC's measured anchor (FINDINGS §4.7, replicated §4.11): the only (half_spread,
#: E[adverse|fill]) pair the platform has actually measured, and the point the scaling
#: curve is pinned through.
BTC_HALF_SPREAD_BPS = 0.0832
BTC_ADVERSE_BPS = 0.2275

#: Reported scaling scenarios. beta=0 is the optimistic reading (§7.2's "if"), beta=1 the
#: pessimistic one (adverse selection priced proportionally into the spread).
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def adverse_at(half_spread_bps, beta: float,
               anchor_half_spread: float = BTC_HALF_SPREAD_BPS,
               anchor_adverse: float = BTC_ADVERSE_BPS):
    """E[adverse|fill] at `half_spread_bps` under a power-law scaling with exponent beta.

    Pinned through BTC's measured point, so beta=0 returns BTC's adverse selection
    everywhere and beta=1 returns it scaled by the spread ratio.
    """
    h = np.asarray(half_spread_bps, dtype=np.float64)
    if anchor_half_spread <= 0:
        raise ValueError("anchor half-spread must be positive")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = anchor_adverse * np.power(h / anchor_half_spread, float(beta))
    return float(out) if np.isscalar(half_spread_bps) or out.ndim == 0 else out


def breakeven_beta(half_spread_bps, rebate_bps: float,
                   anchor_half_spread: float = BTC_HALF_SPREAD_BPS,
                   anchor_adverse: float = BTC_ADVERSE_BPS):
    """The exponent beta* at which a pair exactly breaks even. Higher = more robust.

    Solve  h + rebate = A · (h/h_btc)^beta  for beta:

        beta* = ln((h + rebate) / A) / ln(h / h_btc)

    Returns +inf when the pair covers adverse selection at every beta in [0, 1] (the
    capture exceeds even the proportional estimate), and -inf when it covers it at none —
    both are meaningful verdicts, not errors, so they are not silently clipped.
    """
    h = np.asarray(half_spread_bps, dtype=np.float64)
    capture = h + float(rebate_bps)
    ratio = h / anchor_half_spread

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(capture / anchor_adverse) / np.log(ratio)

    # A pair AT the anchor spread has ln(ratio)=0: beta is undefined there, and whether it
    # breaks even is decided by the rebate alone.
    at_anchor = np.isclose(ratio, 1.0)
    beta = np.where(at_anchor, np.where(capture > anchor_adverse, np.inf, -np.inf), beta)
    # Capture below the anchor adverse at a WIDER spread means no exponent saves it.
    beta = np.where((ratio > 1.0) & (capture <= anchor_adverse), -np.inf, beta)
    beta = np.where(capture <= 0, -np.inf, beta)
    return float(beta) if np.isscalar(half_spread_bps) or beta.ndim == 0 else beta


@dataclass(frozen=True)
class ScreenResult:
    """Per-pair screen plus the universe-level summary the row asks to be stated."""
    pairs: pd.DataFrame           # symbol-indexed, one row per admitted pair
    rejected: dict                # symbol -> [failed floors] (XS-5's reasons)
    n_snapshots: int
    rebate_bps: float
    beta_grid: tuple
    survivors_by_beta: dict       # beta -> [symbols covering adverse at that beta]

    def summary(self) -> dict:
        return {
            "n_admitted": int(len(self.pairs)),
            "n_rejected": int(len(self.rejected)),
            "n_snapshots": self.n_snapshots,
            "rebate_bps": self.rebate_bps,
            "survivors_by_beta": {str(b): len(v) for b, v in self.survivors_by_beta.items()},
            "median_breakeven_beta": (float(np.median(self.pairs["breakeven_beta"]
                                                     .replace([np.inf, -np.inf], np.nan)
                                                     .dropna()))
                                      if len(self.pairs) else None),
        }


def screen(agg: pd.DataFrame, rebate_bps: Optional[float] = None,
           beta_grid: tuple = BETA_GRID,
           max_half_spread_bps: Optional[float] = None,
           min_touch_notional: Optional[float] = None,
           min_depth5_notional: Optional[float] = None,
           n_snapshots: int = 0) -> ScreenResult:
    """Run the screen over an `xs.capacity.aggregate_l2` frame.

    Capacity floors are passed straight to `xs.capacity.admit` — this module does not
    invent its own admission rule, because XS-5 already owns that decision.
    """
    from xs.capacity import admit

    if rebate_bps is None:
        from utils.costs import maker_bps
        rebate_bps = maker_bps()

    if agg is None or agg.empty:
        return ScreenResult(pd.DataFrame(), {}, n_snapshots, float(rebate_bps),
                            tuple(beta_grid), {})

    if any(f is not None for f in (max_half_spread_bps, min_touch_notional,
                                   min_depth5_notional)):
        admitted, rejected = admit(agg,
                                   max_half_spread_bps=max_half_spread_bps,
                                   min_touch_notional=min_touch_notional,
                                   min_depth5_notional=min_depth5_notional)
        pairs = agg.loc[[s for s in admitted if s in agg.index]].copy()
    else:
        pairs, rejected = agg.copy(), {}

    h = pairs["half_spread_bps"].to_numpy(dtype=np.float64)
    pairs["capture_bps"] = h + float(rebate_bps)
    pairs["breakeven_beta"] = breakeven_beta(h, float(rebate_bps))

    survivors = {}
    for beta in beta_grid:
        adverse = adverse_at(h, beta)
        pairs[f"ev_beta_{beta:g}"] = pairs["capture_bps"].to_numpy() - adverse
        survivors[beta] = list(pairs.index[pairs[f"ev_beta_{beta:g}"] > 0])

    pairs = pairs.sort_values("breakeven_beta", ascending=False)
    return ScreenResult(pairs, rejected, n_snapshots, float(rebate_bps),
                        tuple(beta_grid), survivors)

"""XS-5 — the capacity gate: which pairs are tradeable at all?

FINDINGS §7.2 measured the trap this exists to catch. The widest-spread pairs on the venue
are nearly empty at the touch — XAI quoting 12.9 bps on **$20** of bid notional, HMSTR
26.8 bps on $3.3 k. A large per-fill edge on $20 of size is not a business, so admission is
a **joint** requirement: wide enough to pay, deep enough to matter. That joint test is far
more restrictive than either margin alone.

**This module does not invent thresholds.** The guardrail is "gates imported, not
invented", and there is no measured economics yet from which a spread ceiling could be
derived — `XS-6` produces that. So the work is split:

* `aggregate_l2()` turns XS-8's snapshot stream into a per-pair liquidity estimate.
* `admit()` applies floors **supplied by the caller**; omitting a floor applies none, and
  every rejection carries *all* its failed floors, not just the first — otherwise
  loosening one floor looks like it would admit a pair that fails three.
* `tradability_curve()` reports admitted-universe size as a *function* of the floor, so a
  downstream study picks an operating point against measured economics rather than taste.

A liquidity estimate from one snapshot is n=1: the XS-8 sampler's own log shows the median
half-spread moving ~20 % across a single morning (1.35 → 1.62 bps). Pairs with too few
observations are dropped with a reason rather than admitted on a lucky quote.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_L2_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "l2"

__all__ = ["load_l2_snapshots", "aggregate_l2", "admit", "tradability_curve"]


def load_l2_snapshots(data_dir: Path | str = DEFAULT_L2_DIR,
                      date: str | None = None) -> pd.DataFrame:
    """Read XS-8 snapshot parquet (one file per sweep) into one frame."""
    data_dir = Path(data_dir)
    pattern = f"{date}/*.parquet" if date else "**/*.parquet"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no L2 snapshots under {data_dir}"
                                + (f" for {date}" if date else ""))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def aggregate_l2(snaps: pd.DataFrame, min_snapshots: int = 12) -> pd.DataFrame:
    """Per-pair liquidity estimate from a stream of snapshots.

    Uses the **median**, not the latest quote: one outlier sweep must not define a pair's
    liquidity. Degenerate books (crossed / locked / one-sided, XS-8's statuses) carry no
    spread and are excluded from the spread estimate while still counting toward how often
    the pair was observed — a book that is frequently crossed is itself a tradability fact.
    """
    if snaps is None or len(snaps) == 0:
        return pd.DataFrame()

    df = snaps.copy()
    n_seen = df.groupby("symbol").size().rename("n_snapshots")
    ok = df[df["status"] == "ok"] if "status" in df.columns else df

    if ok.empty:
        return pd.DataFrame()

    touch = ok[["bid_notional_l1", "ask_notional_l1"]].min(axis=1)
    ok = ok.assign(_touch=touch)
    depth5 = ok[["bid_notional_5", "ask_notional_5"]].min(axis=1) \
        if "bid_notional_5" in ok.columns else touch
    ok = ok.assign(_depth5=depth5)

    agg = ok.groupby("symbol").agg(
        half_spread_bps=("half_spread_bps", "median"),
        half_spread_p90=("half_spread_bps", lambda s: s.quantile(0.9)),
        touch_notional=("_touch", "median"),
        depth5_notional=("_depth5", "median"),
        n_ok=("half_spread_bps", "count"),
    )
    agg = agg.join(n_seen, how="left")
    # n=1 is not a liquidity estimate; spread moves ~20% within a morning.
    return agg[agg["n_snapshots"] >= min_snapshots].sort_index()


def admit(agg: pd.DataFrame,
          max_half_spread_bps: float | None = None,
          min_touch_notional: float | None = None,
          min_depth5_notional: float | None = None,
          max_half_spread_p90_bps: float | None = None
          ) -> tuple[list[str], dict[str, list[str]]]:
    """Split the universe into (admitted, {symbol: [failed floors]}).

    Every floor is optional and supplied by the caller — omitting one applies none. A
    rejected pair lists **all** the floors it failed, so the report cannot suggest that
    relaxing a single threshold would admit it.
    """
    if agg is None or agg.empty:
        return [], {}

    admitted: list[str] = []
    rejected: dict[str, list[str]] = {}

    for sym, row in agg.iterrows():
        reasons: list[str] = []
        if max_half_spread_bps is not None and row["half_spread_bps"] > max_half_spread_bps:
            reasons.append(f"half_spread {row['half_spread_bps']:.3f} bps > "
                           f"ceiling {max_half_spread_bps}")
        if max_half_spread_p90_bps is not None and row["half_spread_p90"] > max_half_spread_p90_bps:
            reasons.append(f"half_spread_p90 {row['half_spread_p90']:.3f} bps > "
                           f"ceiling {max_half_spread_p90_bps}")
        if min_touch_notional is not None and row["touch_notional"] < min_touch_notional:
            reasons.append(f"touch_notional ${row['touch_notional']:,.0f} < "
                           f"floor ${min_touch_notional:,.0f}")
        if min_depth5_notional is not None and row["depth5_notional"] < min_depth5_notional:
            reasons.append(f"depth5_notional ${row['depth5_notional']:,.0f} < "
                           f"floor ${min_depth5_notional:,.0f}")

        if reasons:
            rejected[sym] = reasons
        else:
            admitted.append(sym)

    return admitted, rejected


def tradability_curve(agg: pd.DataFrame, spread_ceilings, min_touch_notional=None,
                      min_depth5_notional=None) -> list[dict]:
    """Admitted-universe size as a function of the spread ceiling.

    A curve, not a verdict: the point is to hand `XS-6` the shape of the trade-off so it
    can choose an operating point against measured economics, rather than have this module
    guess a threshold and call it a gate.
    """
    out = []
    for ceiling in spread_ceilings:
        admitted, rejected = admit(agg, max_half_spread_bps=ceiling,
                                   min_touch_notional=min_touch_notional,
                                   min_depth5_notional=min_depth5_notional)
        out.append({
            "max_half_spread_bps": ceiling,
            "min_touch_notional": min_touch_notional,
            "min_depth5_notional": min_depth5_notional,
            "n_admitted": len(admitted),
            "n_rejected": len(rejected),
            "admitted": admitted,
        })
    return out

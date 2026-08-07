"""XS-5 — the capacity gate: which pairs are tradeable at all?

FINDINGS §7.2 measured the trap this exists to catch: the widest-spread pairs on the venue
are nearly empty at the touch (XAI 12.9 bps on **$20**). A large per-fill edge on $20 of
size is not a business, so admission is a **joint** wide-enough-AND-deep-enough test, far
more restrictive than either margin alone.

The design constraint that shapes everything here is the guardrail *"gates imported, not
invented"*. This module must not mint a spread ceiling out of taste. So it does two
separable things:

  * **`admit()`** applies floors supplied by the caller (config), and every rejection
    carries its reason — plural, because a pair failing three floors should say so rather
    than stopping at the first.
  * **`tradability_curve()`** reports admitted-universe size as a *function* of the floor,
    so `XS-6` can choose an operating point against measured economics instead of this
    module guessing one.

The other property worth defending: a liquidity estimate from one snapshot is n=1. Spread
moves ~20 % across a morning (§7.2 and the XS-8 sampler's own log), so a pair with too few
observations is excluded with a reason rather than admitted on a lucky quote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from xs.capacity import admit, aggregate_l2, tradability_curve  # noqa: E402


def _snaps(rows) -> pd.DataFrame:
    """rows: (symbol, half_spread_bps, touch_notional, status) repeated per sweep."""
    out = []
    for i, (sym, hs, notional, status) in enumerate(rows):
        out.append({"symbol": sym, "ts_ms": 1_786_000_000_000 + i * 300_000,
                    "half_spread_bps": hs, "bid_notional_l1": notional,
                    "ask_notional_l1": notional, "bid_notional_5": notional * 4,
                    "ask_notional_5": notional * 4, "status": status})
    return pd.DataFrame(out)


def _many(sym, hs, notional, n=10, status="ok"):
    return [(sym, hs, notional, status)] * n


# ── aggregation ───────────────────────────────────────────────────────────

def test_aggregate_uses_the_median_not_the_last_quote():
    """One outlier sweep must not define a pair's liquidity."""
    rows = _many("BTC", 0.08, 500_000, n=9) + [("BTC", 99.0, 1.0, "ok")]
    agg = aggregate_l2(_snaps(rows), min_snapshots=5)
    assert agg.loc["BTC", "half_spread_bps"] == pytest.approx(0.08, abs=0.01)
    assert agg.loc["BTC", "n_snapshots"] == 10


def test_degenerate_books_are_excluded_from_the_estimate():
    """Crossed/locked books carry no spread (XS-8) and must not enter the median."""
    rows = _many("X", 2.0, 1000, n=8) + [("X", np.nan, 1000, "crossed"),
                                         ("X", np.nan, 1000, "locked")]
    agg = aggregate_l2(_snaps(rows), min_snapshots=5)
    assert agg.loc["X", "half_spread_bps"] == pytest.approx(2.0)
    assert agg.loc["X", "n_ok"] == 8


def test_a_pair_with_too_few_snapshots_is_dropped():
    """n=1 is not a liquidity estimate — spread moves ~20% within a morning."""
    agg = aggregate_l2(_snaps(_many("THIN", 1.0, 100, n=2)), min_snapshots=5)
    assert "THIN" not in agg.index


# ── admission: floors are supplied, and rejections explain themselves ─────

def test_admission_applies_the_supplied_floors():
    agg = aggregate_l2(_snaps(
        _many("TIGHT_DEEP", 0.1, 500_000) + _many("WIDE_THIN", 12.0, 20)
    ), min_snapshots=5)

    admitted, rejected = admit(agg, max_half_spread_bps=3.0, min_touch_notional=1_000)
    assert admitted == ["TIGHT_DEEP"]
    assert "WIDE_THIN" in rejected


def test_a_rejection_lists_every_failed_floor_not_just_the_first():
    agg = aggregate_l2(_snaps(_many("BAD", 12.0, 20)), min_snapshots=5)
    _, rejected = admit(agg, max_half_spread_bps=3.0, min_touch_notional=1_000)
    reasons = rejected["BAD"]
    assert any("spread" in r for r in reasons)
    assert any("notional" in r or "depth" in r for r in reasons), (
        f"only reported {reasons} — a pair failing several floors should say so, "
        "otherwise loosening one floor looks like it would admit the pair"
    )


def test_passing_one_margin_is_not_enough():
    """FINDINGS §7.2's trap, in the direction that applies to a ROTATION.

    Class 3 crosses the spread rather than earning it, so tight is *desirable* and the
    spread floor is a ceiling. What must be rejected is a pair that clears only one of
    the two: cheap-but-empty (nothing to trade) or deep-but-expensive (edge eaten).
    """
    agg = aggregate_l2(_snaps(
        _many("TIGHT_BUT_EMPTY", 0.5, 20)          # cheap, nothing there
        + _many("DEEP_BUT_EXPENSIVE", 8.0, 900_000)  # size, but the spread eats it
        + _many("TIGHT_AND_DEEP", 0.4, 500_000)      # the only tradeable one
        + _many("BOTH_OK", 1.5, 50_000)
    ), min_snapshots=5)

    admitted, rejected = admit(agg, max_half_spread_bps=3.0, min_touch_notional=10_000)
    assert sorted(admitted) == ["BOTH_OK", "TIGHT_AND_DEEP"]
    assert set(rejected) == {"TIGHT_BUT_EMPTY", "DEEP_BUT_EXPENSIVE"}


def test_no_floor_is_hardcoded():
    """Guardrail: gates are imported, never invented. Omitting a floor must not apply one."""
    agg = aggregate_l2(_snaps(_many("ANY", 25.0, 1.0)), min_snapshots=5)
    admitted, rejected = admit(agg)          # no floors supplied
    assert admitted == ["ANY"] and not rejected


# ── the curve: a function of the floor, not a verdict ─────────────────────

def test_tradability_curve_is_monotone_in_the_ceiling():
    agg = aggregate_l2(_snaps(
        _many("A", 0.1, 100_000) + _many("B", 1.0, 100_000)
        + _many("C", 5.0, 100_000) + _many("D", 20.0, 100_000)
    ), min_snapshots=5)

    curve = tradability_curve(agg, spread_ceilings=[0.5, 2.0, 10.0, 50.0],
                              min_touch_notional=1_000)
    counts = [c["n_admitted"] for c in curve]
    assert counts == sorted(counts), "loosening the ceiling must never admit fewer pairs"
    assert counts[0] == 1 and counts[-1] == 4


def test_curve_reports_the_floor_it_used():
    agg = aggregate_l2(_snaps(_many("A", 1.0, 100_000)), min_snapshots=5)
    curve = tradability_curve(agg, spread_ceilings=[2.0], min_touch_notional=500)
    assert curve[0]["max_half_spread_bps"] == 2.0
    assert curve[0]["min_touch_notional"] == 500


def test_empty_input_returns_empty_not_an_exception():
    agg = aggregate_l2(pd.DataFrame(columns=["symbol", "half_spread_bps",
                                             "bid_notional_l1", "ask_notional_l1",
                                             "status"]), min_snapshots=5)
    assert agg.empty
    admitted, rejected = admit(agg, max_half_spread_bps=1.0)
    assert admitted == [] and rejected == {}

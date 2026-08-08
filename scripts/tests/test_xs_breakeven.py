"""B-5a — the wide-pair breakeven screen. The tests defend the conditional.

The whole hypothesis rests on one unmeasured quantity: does `E[adverse|fill]` scale with
half-spread? §7.2 spelled out the trap — assume it stays at BTC's 0.228 bps and 156 of 177
pairs "cover it on half-spread alone", which would be a spectacular result produced entirely
by an assumption. So the unit under test must never emit a survivor count without the
exponent it is conditional on, and these tests fail if it does.

Anchoring is the second trap: the curve is pinned through BTC's measured point, so the
screen must reproduce BTC's own arithmetic exactly (§4.11's +0.144 bps breakeven) or the
pin is wrong and every wide-pair number inherits the error.
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

from xs.breakeven import (  # noqa: E402
    BTC_ADVERSE_BPS, BTC_HALF_SPREAD_BPS, adverse_at, breakeven_beta, screen,
)


def _agg(rows) -> pd.DataFrame:
    """An `aggregate_l2`-shaped frame: symbol -> spread + depth."""
    return pd.DataFrame(
        [{"half_spread_bps": h, "half_spread_p90": h * 1.3, "touch_notional": t,
          "depth5_notional": t * 4, "n_ok": 20, "n_snapshots": 20} for _, h, t in rows],
        index=[s for s, _, _ in rows],
    ).rename_axis("symbol")


# ── the anchor must reproduce the measurement it is pinned to ────────────────────
class TestAnchor:
    def test_beta_zero_is_btc_adverse_everywhere(self):
        for h in (0.05, 0.0832, 1.372, 26.81):
            assert adverse_at(h, 0.0) == pytest.approx(BTC_ADVERSE_BPS)

    def test_beta_one_is_proportional_to_the_spread(self):
        assert adverse_at(BTC_HALF_SPREAD_BPS * 10, 1.0) == \
            pytest.approx(BTC_ADVERSE_BPS * 10)

    def test_the_curve_passes_through_btc_at_every_beta(self):
        for beta in (0.0, 0.5, 1.0, 2.0):
            assert adverse_at(BTC_HALF_SPREAD_BPS, beta) == pytest.approx(BTC_ADVERSE_BPS)

    def test_btc_reproduces_section_4_11s_breakeven_rebate(self):
        """§4.11: BTC needs +0.144 bps of rebate to break even. If the screen disagrees,
        the pin is wrong and every wide-pair number inherits the error."""
        needed = BTC_ADVERSE_BPS - BTC_HALF_SPREAD_BPS
        assert needed == pytest.approx(0.1443, abs=5e-4)


# ── beta* is the deliverable ─────────────────────────────────────────────────────
class TestBreakevenExponent:
    def test_a_wide_pair_yields_a_finite_exponent_between_zero_and_one(self):
        """The interesting case: covered when adverse is constant, not when proportional."""
        b = breakeven_beta(1.372, rebate_bps=0.2)
        assert 0.0 < b < 1.0
        # closed form, not a magic constant: beta* = ln(capture/A) / ln(h/h_btc).
        # (Hand-computing this as 0.64 by dropping the rebate is exactly the slip the
        # closed form prevents — the rebate is part of capture.)
        expected = np.log((1.372 + 0.2) / BTC_ADVERSE_BPS) / np.log(1.372 / BTC_HALF_SPREAD_BPS)
        assert b == pytest.approx(expected)
        assert b == pytest.approx(0.690, abs=0.01)

    def test_the_exponent_is_the_indifference_point(self):
        """At beta*, capture equals adverse selection exactly — that is its definition."""
        for h in (0.5, 1.372, 5.0, 12.9):
            b = breakeven_beta(h, rebate_bps=0.2)
            if np.isfinite(b):
                assert adverse_at(h, b) == pytest.approx(h + 0.2, rel=1e-6)

    def test_wider_pairs_are_more_robust(self):
        betas = [breakeven_beta(h, 0.2) for h in (0.3, 1.0, 3.0, 10.0)]
        assert betas == sorted(betas), f"beta* must rise with the spread: {betas}"

    def test_a_pair_that_cannot_be_saved_reports_minus_infinity(self):
        """Wider than BTC but capture below BTC's adverse: no exponent rescues it."""
        assert breakeven_beta(0.1, rebate_bps=0.0) == -np.inf

    def test_a_higher_rebate_raises_the_exponent(self):
        assert breakeven_beta(1.372, 0.3) > breakeven_beta(1.372, 0.2)

    def test_vectorised_and_scalar_agree(self):
        h = np.array([0.5, 1.372, 5.0])
        vec = breakeven_beta(h, 0.2)
        assert [breakeven_beta(float(x), 0.2) for x in h] == pytest.approx(list(vec))


# ── the screen must never state a count without its condition ────────────────────
class TestScreenReportsTheCondition:
    def test_survivors_are_reported_per_beta_never_pooled(self):
        r = screen(_agg([("BTC", 0.083, 50_000), ("WIDE", 3.0, 20_000)]))
        assert set(r.survivors_by_beta) == set(r.beta_grid)
        s = r.summary()
        assert "survivors_by_beta" in s
        assert not any(k in s for k in ("n_survivors", "survivors")), \
            "a single survivor count would hide the assumption it depends on"

    def test_the_optimistic_and_pessimistic_readings_differ(self):
        """If beta made no difference the screen would be measuring nothing."""
        r = screen(_agg([("A", 1.4, 9_000), ("B", 3.0, 9_000), ("C", 8.0, 9_000)]))
        assert len(r.survivors_by_beta[0.0]) > len(r.survivors_by_beta[1.0])

    def test_no_pair_survives_the_proportional_reading(self):
        """beta=1 means adverse scales with the spread, so the ratio is unchanged and the
        only thing left is the rebate — which cannot cover 17x BTC's adverse selection."""
        r = screen(_agg([("A", 1.4, 9_000), ("B", 5.0, 9_000), ("C", 26.8, 9_000)]))
        assert r.survivors_by_beta[1.0] == []

    def test_ev_columns_exist_for_every_beta(self):
        r = screen(_agg([("A", 2.0, 9_000)]))
        for beta in r.beta_grid:
            assert f"ev_beta_{beta:g}" in r.pairs.columns


# ── capacity is the second blade ─────────────────────────────────────────────────
class TestCapacityJoin:
    def test_a_wide_but_empty_pair_is_rejected(self):
        """§7.2: XAI 12.9 bps on $20. A large per-fill edge on $20 is not a business."""
        r = screen(_agg([("DEEP", 2.0, 50_000), ("XAI", 12.9, 20)]),
                   min_touch_notional=5_000)
        assert "XAI" not in r.pairs.index and "XAI" in r.rejected
        assert "DEEP" in r.pairs.index

    def test_the_joint_requirement_is_tighter_than_either_margin(self):
        rows = [("TIGHT_DEEP", 0.09, 90_000), ("WIDE_EMPTY", 9.0, 30),
                ("WIDE_DEEP", 2.4, 40_000)]
        loose = screen(_agg(rows))
        joint = screen(_agg(rows), min_touch_notional=5_000)
        assert len(joint.pairs) < len(loose.pairs)
        assert list(joint.pairs.index) == ["WIDE_DEEP"] or "WIDE_DEEP" in joint.pairs.index

    def test_rejection_reasons_come_from_xs5(self):
        r = screen(_agg([("X", 9.0, 30)]), min_touch_notional=5_000)
        assert r.rejected["X"] and any("touch" in s.lower() for s in r.rejected["X"])

    def test_admission_is_not_reimplemented_here(self):
        """XS-5 owns the floors; a second copy is a second thing to drift."""
        src = Path(__import__("xs.breakeven", fromlist=["x"]).__file__).read_text()
        assert "from xs.capacity import admit" in src


# ── hygiene ──────────────────────────────────────────────────────────────────────
class TestHygiene:
    def test_empty_input_returns_empty_not_an_error(self):
        r = screen(pd.DataFrame())
        assert r.pairs.empty and r.summary()["n_admitted"] == 0

    def test_the_rebate_comes_from_the_cost_ssot(self):
        from utils.costs import maker_bps
        assert screen(_agg([("A", 2.0, 9_000)])).rebate_bps == pytest.approx(maker_bps())

    def test_a_tier_change_moves_the_screen(self):
        from utils.costs import maker_tier_override
        base = screen(_agg([("A", 2.0, 9_000)])).pairs["capture_bps"].iloc[0]
        with maker_tier_override("base"):        # +1.5 bps FEE, not a rebate
            charged = screen(_agg([("A", 2.0, 9_000)])).pairs["capture_bps"].iloc[0]
        assert charged < base

    def test_snapshot_count_is_carried_into_the_result(self):
        """n is the caveat that killed LF7's priors; it must travel with the numbers."""
        assert screen(_agg([("A", 2.0, 9_000)]), n_snapshots=17).summary()["n_snapshots"] == 17

    def test_no_hardcoded_rebate_literal(self):
        src = Path(__import__("xs.breakeven", fromlist=["x"]).__file__).read_text()
        assert "rebate_bps=0.2" not in src and "= 0.2\n" not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

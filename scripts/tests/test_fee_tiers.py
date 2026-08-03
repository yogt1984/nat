"""X-1 guard — the HYPE-staking fee tier is SSOT state, never a silent default.

`config/costs.toml` gains a `[hyperliquid_staked]` ladder (HYPE staking discounts on
Hyperliquid fees). The failure mode this guards is exactly the one that produced the
five false winners (FINDINGS §4.6): a *favourable* cost tier leaking into harness
defaults, so every experiment silently prices a discount it did not earn. VIP9 was that
bug at the venue level; a staking discount is the same bug one level down.

Contract encoded here:
  (a) the active tier defaults to "none" — an unset environment prices the base tier,
      byte-identical to the pre-X-1 numbers;
  (b) an unknown tier raises loudly (never a silent 0 % discount fallback);
  (c) the discount multiplies FEES PAID; a maker *rebate* is income and is untouched
      unless `rebate_discount_applies` is explicitly turned on (sensitivity mode);
  (d) the ladder is monotone, bounded, and every experiment can stamp the tier it
      priced (`tier_summary()`);
  (e) no module may DEFAULT to a discounted tier (source scan, VIP9-style).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
LADDER = ["none", "wood", "bronze", "silver", "gold", "platinum", "diamond"]


@pytest.fixture(autouse=True)
def _clean_env():
    """Every test starts from an unset tier and restores whatever was there."""
    saved = {k: os.environ.get(k) for k in ("NAT_FEE_TIER", "NAT_FEE_REBATE_DISCOUNT")}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestLadderIsConfigured:
    def test_config_declares_the_staking_ladder(self):
        from utils.costs import load_costs
        staked = load_costs()["hyperliquid_staked"]
        assert staked["tier"] == "none", "the committed config must price the base tier"
        assert staked["rebate_discount_applies"] is False
        assert set(staked["discounts"]) == set(LADDER)

    def test_ladder_is_monotone_and_bounded(self):
        from utils.costs import staking_discount
        vals = [staking_discount(t) for t in LADDER]
        assert vals[0] == 0.0
        assert vals == sorted(vals), "discounts must increase with the tier"
        assert all(0.0 <= v < 0.5 for v in vals), "a >50% fee discount is not a real tier"


class TestDefaultIsBaseTier:
    def test_unset_env_prices_the_base_tier(self):
        from utils.costs import fee_tier, load_costs, maker_bps, taker_bps
        hl = load_costs()["hyperliquid"]
        assert fee_tier() == "none"
        assert taker_bps() == pytest.approx(hl["taker_bps"])
        assert maker_bps() == pytest.approx(hl["maker_bps"])

    def test_round_trip_taker_tracks_the_tier(self):
        from utils.costs import fee_tier_override, round_trip_taker_bps
        base = round_trip_taker_bps()
        with fee_tier_override("diamond"):
            assert round_trip_taker_bps() == pytest.approx(base * 0.60)

    def test_realistic_rt_still_carries_full_slippage(self):
        """The discount is a FEE discount — slippage is a market fact, not a fee."""
        from utils.costs import fee_tier_override, realistic_taker_rt_bps, slippage_bps
        with fee_tier_override("diamond"):
            assert realistic_taker_rt_bps() == pytest.approx(7.0 * 0.60 + 2.0 * slippage_bps())


class TestDiscountAppliesToFeesNotRebates:
    def test_taker_fee_is_discounted(self):
        from utils.costs import fee_tier_override, taker_bps
        with fee_tier_override("diamond"):
            assert taker_bps() == pytest.approx(3.5 * 0.60)
        with fee_tier_override("gold"):
            assert taker_bps() == pytest.approx(3.5 * 0.80)

    def test_maker_rebate_is_untouched_by_default(self):
        """A rebate is income. Staking does not reduce what the venue pays you."""
        from utils.costs import fee_tier_override, maker_bps
        base = maker_bps()
        for tier in LADDER:
            with fee_tier_override(tier):
                assert maker_bps() == pytest.approx(base)

    def test_rebate_discount_is_available_as_explicit_sensitivity(self):
        """The pessimistic reading (venue discounts the rebate too) must be runnable."""
        from utils.costs import fee_tier_override, maker_bps
        base = maker_bps()
        with fee_tier_override("diamond", rebate_discount=True):
            assert maker_bps() == pytest.approx(base * 0.60)

    def test_maker_fee_venue_would_be_discounted(self):
        """If the maker leg is ever a positive FEE, the discount must apply to it."""
        from utils.costs import apply_fee_discount
        assert apply_fee_discount(1.5, "diamond") == pytest.approx(0.90)
        assert apply_fee_discount(-0.2, "diamond") == pytest.approx(-0.2)  # rebate intact


class TestUnknownTierFailsLoudly:
    def test_unknown_tier_name_raises(self):
        from utils.costs import staking_discount
        with pytest.raises(ValueError, match="unknown fee tier"):
            staking_discount("platinum_plus")

    def test_unknown_tier_in_env_raises_on_use(self):
        from utils.costs import taker_bps
        os.environ["NAT_FEE_TIER"] = "diamond_max"
        with pytest.raises(ValueError, match="unknown fee tier"):
            taker_bps()


class TestExperimentsCanStampTheTier:
    def test_tier_summary_is_self_describing(self):
        from utils.costs import fee_tier_override, tier_summary
        with fee_tier_override("platinum"):
            s = tier_summary()
        assert s["tier"] == "platinum"
        assert s["discount"] == pytest.approx(0.30)
        assert s["taker_bps"] == pytest.approx(3.5 * 0.70)
        assert s["maker_bps"] == pytest.approx(0.2)
        assert s["rebate_discount_applies"] is False


class TestRepriceIdentityIsExact:
    """Planted proof of the X-1 driver's shortcut (`fee_tier_reprice.reprice_grid`).

    The claim: with the rebate untouched, the staking discount enters `TouchMakerSim`
    at exactly one place — the terminal liquidation — so the fill path is tier-invariant
    and one base pass reprices the whole ladder exactly:

        pnl(tier) = pnl(base) + liq_cost(base) · (1 − taker(tier)/taker(base))

    If a future edit puts the taker fee anywhere else in the fill loop, this test fails
    and the driver's shortcut must be replaced by re-simulation.
    """

    @staticmethod
    def _planted_inputs(n=4000):
        """Deterministic sawtooth tape: guarantees fills on both sides and q_end ≠ 0."""
        import numpy as np
        t = np.arange(n)
        mid = 100.0 + 0.5 * np.sin(t / 50.0) + 0.002 * t   # oscillation + drift ⇒ inventory
        spread = np.full(n, 0.02)
        depth = np.full(n, 1.0)
        # asymmetric flow: bids fill faster than asks ⇒ terminal inventory ≠ 0, which is
        # what makes the liquidation term (the only taker-fee term) actually bite
        return dict(
            mid=mid, best_bid=mid - spread / 2, best_ask=mid + spread / 2,
            sell_exec=np.full(n, 0.5), buy_exec=np.full(n, 0.2),
            depth_bid=depth, depth_ask=depth,
            fair_dev_bps=np.zeros(n), gate_open=np.ones(n, dtype=bool),
        )

    def _run(self, **flags):
        import sys
        sys.path.insert(0, str(SCRIPTS))
        from execution.touch_maker import TouchMakerSim, TouchParams
        return TouchMakerSim(TouchParams(l1_fraction=0.4, requote_every=10,
                                         **flags)).run(**self._planted_inputs())

    @pytest.mark.parametrize("cell", [{}, {"use_ev_gate": True},
                                      {"use_hf1_side": True, "use_inv_skew": True}])
    @pytest.mark.parametrize("tier", ["gold", "diamond"])
    def test_analytic_reprice_matches_resimulation(self, cell, tier):
        from utils.costs import fee_tier_override, taker_bps
        base = self._run(**cell)
        assert base["n_fills"] > 0 and base["terminal_inventory"] != 0, "planted tape is inert"
        base_taker = taker_bps()

        with fee_tier_override(tier):
            direct = self._run(**cell)
            scale = taker_bps() / base_taker

        predicted = base["pnl_bps"] + base["liquidation_cost_bps"] * (1.0 - scale)
        assert direct["pnl_bps"] == pytest.approx(predicted, rel=1e-12)
        assert direct["n_fills"] == base["n_fills"], "fill path must be tier-invariant"

    def test_rebate_discount_does_move_the_fill_path(self):
        """The sensitivity variant must NOT be repriced analytically — it re-simulates."""
        from utils.costs import fee_tier_override
        base = self._run(use_ev_gate=True)
        with fee_tier_override("diamond", rebate_discount=True):
            sens = self._run(use_ev_gate=True)
        assert sens["maker_bps_used"] < base["maker_bps_used"]
        # a smaller rebate shrinks EV-gate capture ⇒ strictly fewer postings survive
        assert sens["n_postings"] <= base["n_postings"]


class TestNoHarnessDefaultsToADiscountedTier:
    """VIP9-style source scan, one level down: a discount must be opted into."""

    EXEMPT = {"utils/costs.py", "tests"}
    FORBIDDEN = [
        ("argparse default", re.compile(
            r"default\s*=\s*['\"](wood|bronze|silver|gold|platinum|diamond)['\"]")),
        ("env fallback to a discounted tier", re.compile(
            r"environ\.get\(\s*['\"]NAT_FEE_TIER['\"]\s*,\s*"
            r"['\"](wood|bronze|silver|gold|platinum|diamond)['\"]")),
        ("module-level discounted constant", re.compile(
            r"^(FEE_TIER|DEFAULT_TIER)\s*=\s*['\"](wood|bronze|silver|gold|platinum|diamond)",
            re.M)),
    ]

    def test_no_module_defaults_to_a_discounted_tier(self):
        violations = []
        for path in sorted(SCRIPTS.rglob("*.py")):
            rel = path.relative_to(SCRIPTS).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in self.EXEMPT):
                continue
            text = path.read_text(errors="replace")
            for label, pat in self.FORBIDDEN:
                if pat.search(text):
                    violations.append(f"{rel}: {label}")
        assert not violations, (
            "a HYPE-staking discount is used as a DEFAULT — it must be explicit "
            "(the VIP9 lesson, FINDINGS §4.6):\n  " + "\n  ".join(violations))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

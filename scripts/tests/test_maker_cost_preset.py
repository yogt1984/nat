"""COST-8 — the maker preset must price the SSOT, and price it with the right sign.

`backtest/costs.py` built its maker presets as `CostModel(fee_bps=0.2, ...)` literals.
Two defects in one line:

  1. **hardcoded** — it bypasses `load_costs()`, so the rate §4.11 calls the most
     load-bearing unvalidated assumption in the stack (a 0.2 bps rebate presuming
     ≥1.5 % of venue-wide maker volume) could not be re-priced, and the COST-3 literal
     scanner does not catch it because the number is bare;
  2. **sign-inverted** — `CostModel.fee_bps` is a COST while `utils.costs.maker_bps()`
     is a REBATE EARNED, so a 0.2 bps rebate was booked as a 0.2 bps charge. That is a
     0.4 bps error per side against a measured breakeven of +0.144 bps (§4.11), i.e.
     large enough to flip the sign of a maker result on its own.

The contract these tests pin: the presets read the maker-tier ladder, a rebate *lowers*
the round-trip cost, and re-pricing the tier moves the preset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get("NAT_MAKER_TIER")
    os.environ.pop("NAT_MAKER_TIER", None)
    yield
    if saved is None:
        os.environ.pop("NAT_MAKER_TIER", None)
    else:
        os.environ["NAT_MAKER_TIER"] = saved


class TestPresetReadsTheSsot:
    def test_no_bare_rate_literal_remains(self):
        from backtest import costs
        src = Path(costs.__file__).read_text()
        assert "fee_bps=0.2" not in src, \
            "maker rate still hardcoded — route it through utils.costs"

    def test_maker_preset_matches_the_active_tier(self):
        from backtest.costs import hyperliquid_maker
        from utils.costs import maker_bps
        # SSOT: positive = rebate earned. CostModel: positive = cost. Hence the negation.
        assert hyperliquid_maker().fee_bps == pytest.approx(-maker_bps())

    def test_conservative_variant_uses_the_same_rate(self):
        from backtest.costs import hyperliquid_maker, hyperliquid_maker_conservative
        a, b = hyperliquid_maker(), hyperliquid_maker_conservative()
        assert a.fee_bps == pytest.approx(b.fee_bps)
        assert b.fill_probability < a.fill_probability, "only the fill rate should differ"

    def test_repricing_the_tier_moves_the_preset(self):
        """The whole point: a tier change must ripple, not require an edit."""
        from backtest.costs import hyperliquid_maker
        from utils.costs import maker_tier_override
        with maker_tier_override("base"):            # +1.5 bps FEE at the venue's base rate
            assert hyperliquid_maker().fee_bps == pytest.approx(1.5)
        with maker_tier_override("zero_fee"):
            assert hyperliquid_maker().fee_bps == pytest.approx(0.0)
        with maker_tier_override("rebate_t3"):       # -0.003 % => earns 0.3 bps
            assert hyperliquid_maker().fee_bps == pytest.approx(-0.3)

    def test_from_config_agrees_with_the_preset(self):
        from backtest.costs import CostModel, hyperliquid_maker
        assert CostModel.from_config(role="maker").fee_bps == \
            pytest.approx(hyperliquid_maker().fee_bps)


class TestSignSemantics:
    def test_a_rebate_lowers_the_round_trip_cost(self):
        from backtest.costs import CostModel
        rebated = CostModel(fee_bps=-0.2, slippage_bps=0.5)
        free = CostModel(fee_bps=0.0, slippage_bps=0.5)
        assert rebated.round_trip_cost_bps < free.round_trip_cost_bps
        assert rebated.round_trip_cost_bps == pytest.approx(2 * (0.5 - 0.2))

    def test_a_rebate_improves_the_effective_entry_price(self):
        from backtest.costs import CostModel
        m = CostModel(fee_bps=-1.0, slippage_bps=0.0)
        assert m.apply_entry_cost(100.0, "long") < 100.0, \
            "being paid to trade must not make the entry worse"

    def test_the_maker_preset_is_cheaper_than_taker(self):
        from backtest.costs import hyperliquid_maker, hyperliquid_taker
        assert hyperliquid_maker().round_trip_cost_bps < \
            hyperliquid_taker().round_trip_cost_bps


class TestValidationStillCatchesTypos:
    """Relaxing the non-negative rule must not turn it off."""

    def test_a_plausible_rebate_is_accepted(self):
        from backtest.costs import CostModel
        assert CostModel(fee_bps=-0.3).fee_bps == pytest.approx(-0.3)

    @pytest.mark.parametrize("absurd", [-50.0, -1e6])
    def test_an_implausible_rebate_still_raises(self, absurd):
        from backtest.costs import CostModel
        with pytest.raises(ValueError, match="rebate"):
            CostModel(fee_bps=absurd)

    def test_negative_slippage_is_still_rejected(self):
        from backtest.costs import CostModel
        with pytest.raises(ValueError):
            CostModel(slippage_bps=-1.0)

    def test_the_floor_is_a_named_constant_not_a_literal(self):
        from backtest import costs
        assert hasattr(costs, "MAX_REBATE_BPS") and costs.MAX_REBATE_BPS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

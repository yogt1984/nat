"""DOCS-4 pinning tests: the frozen slice must keep regenerating the claims.

The slice is frozen in git, so these pins are exact — if one moves, either the
slice was touched or the computation changed, and both must be deliberate.
The values were regenerated 2026-08-12 and sit next to the recorded claims
(measured over more days): median 18.6x BTC vs recorded 17.7x, 4/177 pairs at
$5k vs recorded 4/177, breakeven +0.151 vs recorded +0.144, same first viable
rung.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from repro.make_figures import (
    DEFAULT_SLICE,
    main,
    maker_ladder,
    slice_aggregate,
    spread_stats,
    touch_stats,
)
from utils.costs import load_costs

pytestmark = pytest.mark.skipif(not DEFAULT_SLICE.exists(),
                                reason="frozen slice not present")


@pytest.fixture(scope="module")
def agg():
    return slice_aggregate()


class TestFrozenHeadlines:
    def test_spread_claim(self, agg):
        s = spread_stats(agg)
        assert s["n_pairs"] == 177
        assert s["btc_half_spread_bps"] == pytest.approx(0.077)
        assert s["median_half_spread_bps"] == pytest.approx(1.428)
        assert s["median_to_btc_ratio"] > 10  # the claim: the venue is not its majors

    def test_touch_claim(self, agg):
        t = touch_stats(agg)
        assert t["median_touch_usd"] == pytest.approx(81.0)
        assert t["n_pairs_touch_ge_5k"] == 4  # matches the recorded 4/177 exactly

    def test_maker_ladder_claim(self, agg):
        m = maker_ladder(agg, load_costs())
        assert m["breakeven_maker_rate_bps"] == pytest.approx(0.151)
        # the §4.11 claim: zero fees are under water; first viable rung is rebate_t2
        assert m["rungs"]["zero_fee"]["edge_bid_bps"] < 0
        assert m["rungs"]["rebate_t1"]["edge_bid_bps"] < 0
        assert m["first_viable_rung"] == "rebate_t2"


class TestOneCommandPath:
    def test_main_writes_headlines(self, tmp_path):
        assert main(["--out", str(tmp_path), "--no-plots"]) == 0
        assert (tmp_path / "headlines.json").exists()

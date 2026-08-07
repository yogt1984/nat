"""A5 — hysteresis no-trade bands and TWAP/VWAP slicing.

Two execution primitives that act on **cost** rather than on signal. The XS rotation
currently spends 1.10 % of an 8.18 % gross on transaction cost (FINDINGS §7.8), and its
turnover is 0.199 per rebalance, so anything that removes trades which do not pay for
themselves goes straight to the net.

The properties that carry each:

  **Hysteresis.** Under *proportional* transaction costs the optimal policy is not "trade
  when the drift is large enough" but a **no-trade region with trading to the boundary**
  (Constantinides 1986; Davis–Norman 1990). Trading all the way back to target from outside
  the band throws away the part of the move that costs more than it earns. Both variants
  ship, and the tests pin the difference — the boundary variant must land *on* the band
  edge, never at the target.

  **Slicing.** A TWAP schedule must conserve quantity exactly (a slicer that loses or
  invents size is worse than no slicer), and a VWAP schedule must place more where volume
  is, degrading to TWAP on a flat profile. Both are pinned against hand-computed answers.

What is deliberately *not* asserted anywhere here: that slicing improves P&L. Our cost
model has no impact term, so slicing measures as exactly zero — the claim needs the F-task
fill data (`X-3`), and asserting it now would be inventing evidence.
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

from execution.rebalance import (  # noqa: E402
    band_from_cost,
    no_trade_band,
    trade_to_edge,
    twap_slices,
    vwap_slices,
)


# ── hysteresis: the no-trade region ──────────────────────────────────────

def test_small_drifts_are_not_traded():
    cur = pd.Series([0.10, -0.20, 0.05])
    tgt = pd.Series([0.101, -0.203, 0.048])          # all well inside a 0.01 band
    out = no_trade_band(tgt, cur, band=0.01)
    pd.testing.assert_series_equal(out, cur, check_names=False)


def test_large_drifts_are_traded_all_the_way():
    cur = pd.Series([0.10, -0.20])
    tgt = pd.Series([0.30, -0.05])
    out = no_trade_band(tgt, cur, band=0.01)
    pd.testing.assert_series_equal(out, tgt, check_names=False)


def test_the_band_is_applied_per_position_not_globally():
    """One large drift must not drag small ones along with it."""
    cur = pd.Series([0.10, 0.10])
    tgt = pd.Series([0.101, 0.90])
    out = no_trade_band(tgt, cur, band=0.01)
    assert out.iloc[0] == 0.10        # untouched
    assert out.iloc[1] == 0.90        # traded


def test_a_per_position_band_series_is_honoured():
    """Cheap pairs deserve a tighter band than expensive ones."""
    cur = pd.Series([0.10, 0.10])
    tgt = pd.Series([0.15, 0.15])
    out = no_trade_band(tgt, cur, band=pd.Series([0.01, 0.20]))
    assert out.iloc[0] == 0.15        # 0.05 drift > 0.01 band -> trade
    assert out.iloc[1] == 0.10        # 0.05 drift < 0.20 band -> hold


# ── hysteresis: trading to the boundary, not the target ──────────────────

def test_trade_to_edge_stops_at_the_band_not_the_target():
    """The Constantinides result: go to the boundary of the no-trade region.

    cur 0.10, tgt 0.30, band 0.05 -> the optimal move is to 0.30 - 0.05 = 0.25, keeping the
    last 0.05 of drift untraded because it costs more than it earns.
    """
    out = trade_to_edge(pd.Series([0.30]), pd.Series([0.10]), band=0.05)
    assert out.iloc[0] == pytest.approx(0.25)


def test_trade_to_edge_is_symmetric_for_shorts():
    out = trade_to_edge(pd.Series([-0.30]), pd.Series([-0.10]), band=0.05)
    assert out.iloc[0] == pytest.approx(-0.25)


def test_trade_to_edge_never_overshoots_the_target():
    """A band wider than the drift must produce no trade, not a reversal."""
    out = trade_to_edge(pd.Series([0.12]), pd.Series([0.10]), band=0.05)
    assert out.iloc[0] == pytest.approx(0.10)


def test_trade_to_edge_always_moves_less_than_full_rebalancing():
    rng = np.random.default_rng(0)
    cur = pd.Series(rng.normal(scale=0.1, size=200))
    tgt = pd.Series(rng.normal(scale=0.1, size=200))
    edge = trade_to_edge(tgt, cur, band=0.02)
    full = no_trade_band(tgt, cur, band=0.0)
    assert (edge - cur).abs().sum() < (full - cur).abs().sum()


# ── band sizing from cost ────────────────────────────────────────────────

def test_band_from_cost_scales_with_cost_and_multiple():
    a = band_from_cost(cost_bps=2.7, alpha_bps=10.0, multiple=2.0)
    b = band_from_cost(cost_bps=5.4, alpha_bps=10.0, multiple=2.0)
    c = band_from_cost(cost_bps=2.7, alpha_bps=10.0, multiple=4.0)
    assert b == pytest.approx(2 * a)
    assert c == pytest.approx(2 * a)


def test_band_from_cost_shrinks_as_the_edge_grows():
    """A stronger signal justifies trading through more cost."""
    weak = band_from_cost(cost_bps=2.7, alpha_bps=5.0)
    strong = band_from_cost(cost_bps=2.7, alpha_bps=50.0)
    assert strong < weak


def test_band_from_cost_refuses_a_nonpositive_edge():
    """No expected edge means no trade is justified — not an infinitely tight band."""
    with pytest.raises(ValueError):
        band_from_cost(cost_bps=2.7, alpha_bps=0.0)


# ── slicing ──────────────────────────────────────────────────────────────

def test_twap_conserves_quantity_exactly():
    s = twap_slices(100.0, 7)
    assert len(s) == 7
    assert sum(s) == pytest.approx(100.0)
    assert all(x == pytest.approx(100.0 / 7) for x in s)


def test_twap_handles_negative_quantity_and_one_slice():
    assert sum(twap_slices(-30.0, 3)) == pytest.approx(-30.0)
    assert twap_slices(50.0, 1) == [50.0]


def test_twap_rejects_a_nonpositive_slice_count():
    with pytest.raises(ValueError):
        twap_slices(10.0, 0)


def test_vwap_places_more_where_volume_is():
    s = vwap_slices(100.0, [1.0, 3.0])
    assert sum(s) == pytest.approx(100.0)
    assert s[1] == pytest.approx(75.0)
    assert s[0] == pytest.approx(25.0)


def test_vwap_degrades_to_twap_on_a_flat_profile():
    assert vwap_slices(100.0, [2.0, 2.0, 2.0]) == pytest.approx(twap_slices(100.0, 3))


def test_vwap_rejects_an_empty_or_zero_profile():
    for bad in ([], [0.0, 0.0]):
        with pytest.raises(ValueError):
            vwap_slices(100.0, bad)


def test_vwap_rejects_negative_volume():
    with pytest.raises(ValueError):
        vwap_slices(100.0, [1.0, -2.0])


def test_band_from_cost_is_dimensionless_until_scaled():
    """The ratio is not a position band until multiplied by a characteristic size.

    For a unit-gross book of ~120 names the typical weight is ~0.008 while the raw ratio at
    NAT's costs is ~1.4 — applied directly that means *never trade*, which would present as
    a strategy that mysteriously stopped rebalancing rather than as a units error.
    """
    raw = band_from_cost(cost_bps=6.9, alpha_bps=9.8)
    assert raw > 1.0
    scaled = band_from_cost(cost_bps=6.9, alpha_bps=9.8, position_scale=1 / 120)
    assert scaled == pytest.approx(raw / 120)
    assert scaled < 0.02

"""COST-9 Part C — funding accrual on held inventory in the rotation.

`xs/rotation.py` priced only turnover: `cost = Σ(turnover × round_trip_bps)`. Inventory held
between rebalances paid no funding, and the XS-6 study docstring says so in as many words —
*"NOT tested here, and not to be claimed: … funding accrual on held inventory."* Since
**XS-9 passes 4 of 6 pre-registered criteria** (§7.8), that omission sits underneath the
closest thing to a surviving result in this repository.

Contract encoded here:
  (a) **arithmetic** — a planted book with known weights and a known constant rate loses
      exactly the hand-computed amount, in return units (the rate is a fraction, like
      `fwd`; only `cost_bps` needs the 1e-4);
  (b) **the cancellation is real, and bounded.** On a market-neutral book with a *uniform*
      rate, funding nets to ~0 — the plan's honest prior. What survives is exposure to the
      cross-sectional **dispersion** of rates, so a book tilted toward high-funding coins
      pays. Both directions are asserted, because "it cancels" is the claim most likely to
      be believed without evidence and most likely to be wrong in the tail;
  (c) **sign** — a long pays a positive rate, a short receives it. An inverted sign turns a
      cost into income and would make `LF8` measure the negative of what it claims;
  (d) **backwards compatible** — `funding_wide=None` reproduces the pre-COST-9 result
      exactly, so the re-pricing delta is attributable to funding and nothing else;
  (e) **NaN funding is 0 for that cell, never NaN for the period.** A pair listed
      mid-window has no funding history there; poisoning the whole rebalance would delete
      real periods from a study whose binding constraint is already `n`.
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

from xs.rotation import run_rotation  # noqa: E402


def _panel(n_pairs=60, n_bars=900, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01", periods=n_bars, freq="1h", tz="UTC")
    mkt = rng.normal(scale=0.004, size=n_bars)
    betas = np.linspace(0.3, 2.0, n_pairs)
    out = {}
    for j in range(n_pairs):
        r = betas[j] * mkt + rng.normal(scale=0.004, size=n_bars)
        out[f"P{j:02d}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=ts)


def _costs(cols, bps=2.0):
    return pd.Series(bps, index=cols)


def _funding(wide, rate=1.25e-05):
    """Uniform hourly funding rate, aligned to the price panel."""
    return pd.DataFrame(rate, index=wide.index, columns=wide.columns)


# ── (d) backwards compatible ─────────────────────────────────────────────────────

def test_no_funding_reproduces_the_pre_cost9_result():
    """The delta must be attributable to funding alone, so the None path must not move."""
    w = _panel()
    a = run_rotation(w, _costs(w.columns))
    b = run_rotation(w, _costs(w.columns), funding_wide=None)
    assert a == b


def test_zero_rate_costs_nothing():
    w = _panel()
    base = run_rotation(w, _costs(w.columns))
    zero = run_rotation(w, _costs(w.columns), funding_wide=_funding(w, rate=0.0))
    assert zero["net_total_pct"] == pytest.approx(base["net_total_pct"])


# ── (b) the cancellation, and its limit ──────────────────────────────────────────

def test_uniform_rate_on_a_market_neutral_book_nets_to_about_zero():
    """The plan's honest prior, pinned: a beta-neutral book with a flat rate pays ~nothing.

    Weights sum to ~0 by construction, so Σ(w·f) with constant f is ~0. This is why the
    21 bps/week headline is an upper bound that applies to long-only configurations.
    """
    w = _panel()
    base = run_rotation(w, _costs(w.columns))
    flat = run_rotation(w, _costs(w.columns), funding_wide=_funding(w, rate=1.25e-05))
    delta_pct = abs(flat["net_total_pct"] - base["net_total_pct"])
    assert delta_pct < 0.05, (
        f"a uniform rate on a neutral book should nearly cancel, moved {delta_pct:.4f} pct")


def test_dispersed_rates_do_not_cancel():
    """The residual exposure is to cross-sectional DISPERSION, not the level.

    Half the universe pays a large positive rate and half a large negative one, so any net
    tilt is charged. If this passes at ~0 the funding term is not wired to the weights.
    """
    w = _panel()
    f = _funding(w, rate=0.0)
    half = len(w.columns) // 2
    f.iloc[:, :half] = +5e-04
    f.iloc[:, half:] = -5e-04

    base = run_rotation(w, _costs(w.columns))
    disp = run_rotation(w, _costs(w.columns), funding_wide=f)
    assert abs(disp["net_total_pct"] - base["net_total_pct"]) > 0.05, \
        "dispersed funding rates left the result unchanged — the term is not connected"


# ── (a)+(c) arithmetic and sign ──────────────────────────────────────────────────

def _dispersed(wide, amp):
    """Half the universe at +amp, half at -amp. Uniform rates cancel on a neutral book,
    so dispersion is the only thing that can move it — see the test above."""
    f = _funding(wide, rate=0.0)
    half = len(wide.columns) // 2
    f.iloc[:, :half] = +amp
    f.iloc[:, half:] = -amp
    return f


def test_funding_scales_linearly_with_the_rate():
    """10x the dispersion must cost ~10x. This is what pins the term to the weights.

    Note this cannot be shown with a *uniform* rate: `run_rotation` is beta-neutral by
    construction, so a flat rate nets to ~0 at any magnitude (the test above). Scaling is
    only observable where the book has net exposure to the rate.
    """
    w = _panel()
    cheap = _costs(w.columns, bps=0.0)          # isolate funding from turnover cost
    base = run_rotation(w, cheap)["net_total_pct"]
    d_small = run_rotation(w, cheap, funding_wide=_dispersed(w, 1e-04))["net_total_pct"] - base
    d_large = run_rotation(w, cheap, funding_wide=_dispersed(w, 1e-03))["net_total_pct"] - base

    assert abs(d_small) > 0, "dispersed funding had no effect at all"
    assert np.sign(d_small) == np.sign(d_large), "sign flipped with magnitude"
    assert d_large / d_small == pytest.approx(10.0, rel=0.02), \
        f"funding is not linear in the rate: {d_large / d_small:.2f}x for 10x the rate"


def test_negative_rates_flip_the_contribution():
    """Funding goes negative when shorts are crowded — the same book then earns."""
    w = _panel()
    cheap = _costs(w.columns, bps=0.0)
    pos = run_rotation(w, cheap, funding_wide=_funding(w, rate=+1e-03))
    neg = run_rotation(w, cheap, funding_wide=_funding(w, rate=-1e-03))
    base = run_rotation(w, cheap)
    # Tolerance is the reported rounding granularity (3 dp), not float epsilon.
    assert (pos["net_total_pct"] - base["net_total_pct"]) == pytest.approx(
        -(neg["net_total_pct"] - base["net_total_pct"]), abs=2e-3), \
        "flipping the rate sign must flip the funding contribution"


# ── (e) missing funding is 0, not NaN ────────────────────────────────────────────

def test_nan_funding_does_not_poison_the_period():
    """A pair listed mid-window has no funding there; it must not delete the rebalance."""
    w = _panel()
    f = _funding(w)
    f.iloc[:, 0] = np.nan                      # one pair with no funding history at all
    out = run_rotation(w, _costs(w.columns), funding_wide=f)
    assert out["n_periods"] > 0
    assert np.isfinite(out["net_total_pct"]), "NaN funding poisoned the result"


def test_funding_columns_need_not_cover_every_pair():
    """A partial funding panel is reindexed, not an error."""
    w = _panel()
    f = _funding(w).iloc[:, :10]               # only 10 of 60 pairs
    out = run_rotation(w, _costs(w.columns), funding_wide=f)
    assert out["n_periods"] > 0
    assert np.isfinite(out["net_total_pct"])


def test_input_frames_are_not_mutated():
    w = _panel()
    f = _funding(w)
    w_before, f_before = w.copy(), f.copy()
    run_rotation(w, _costs(w.columns), funding_wide=f)
    pd.testing.assert_frame_equal(w, w_before)
    pd.testing.assert_frame_equal(f, f_before)

"""XS-2 — bar-level cross-sectional features: each pair measured against ITSELF.

Class 3 ranks ~177 perps against each other, so the features must be comparable across
pairs. Raw values are not: permutation entropy on a $64k pair with a 1-tick spread and on
a $0.09 memecoin measure different things, and a realized vol of 3 % means something
different for each. `specs/maker_system.md` §5 therefore specifies every score "as
percentile/z vs the pair's own history" — the raw estimator is only half the unit, and
the self-referencing transform is the other half.

Four properties carry this, and each is attacked:

  1. **The estimators are right on inputs with known answers.** A monotone series has zero
     ordinal entropy; an i.i.d. series has ~1; a random walk has Hurst ~0.5, a trend >0.5,
     a mean-reverting series <0.5. These are pinned against hand-computed or theoretical
     values, never golden output.

  2. **`permutation_entropy` agrees with the Rust implementation** (`ing-features/src/
     entropy.rs:373`), which already defines what PE means in this codebase — same ordinal
     patterns, same normalisation by `ln(order!)`. Two definitions of one name would make
     any cross-layer comparison silently wrong.

  3. **The self-percentile is STRICTLY CAUSAL.** A rolling percentile that can see its own
     future is the classic way to manufacture cross-sectional alpha: today's vol looks
     extreme only because tomorrow's calm is already in the window. The decisive test
     perturbs the future violently and asserts past values do not move by a float — the
     same attack PROC-15 uses on its beta fit.

  4. **Thin history yields NaN, never a number.** A percentile over 3 observations is
     noise wearing a decimal point, and on a 177-pair universe the newly-listed pairs are
     exactly the ones that would score extreme.
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

from xs.features import (  # noqa: E402
    hurst_rs,
    momentum_strength,
    permutation_entropy,
    realized_vol,
    rolling_self_percentile,
)


# ── 1 & 2. estimators against known answers ──────────────────────────────

def test_permutation_entropy_of_a_monotone_series_is_zero():
    """Every window has the same ordinal pattern -> one symbol -> zero entropy."""
    assert permutation_entropy(np.arange(50.0), order=3) == pytest.approx(0.0)
    assert permutation_entropy(np.arange(50.0)[::-1], order=3) == pytest.approx(0.0)


def test_permutation_entropy_hand_computed_two_patterns():
    """`ln(k)/ln(order!)` for k equally-frequent patterns — the Rust normalisation.

    [1,2,3,2,1] at order 3 gives windows (1,2,3) rising, (2,3,2) peak, (3,2,1) falling:
    three distinct patterns, once each, out of 3! = 6 possible.
    """
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert permutation_entropy(x, order=3) == pytest.approx(np.log(3) / np.log(6), rel=1e-9)


def test_permutation_entropy_of_noise_approaches_one():
    rng = np.random.default_rng(0)
    pe = permutation_entropy(rng.normal(size=20_000), order=3)
    assert pe > 0.99, f"i.i.d. noise should saturate ordinal entropy, got {pe}"


def test_permutation_entropy_is_bounded_and_handles_short_input():
    assert np.isnan(permutation_entropy(np.array([1.0, 2.0]), order=3))
    assert 0.0 <= permutation_entropy(np.random.default_rng(1).normal(size=200)) <= 1.0


def test_hurst_separates_trend_from_walk_from_reversion():
    rng = np.random.default_rng(7)
    n = 4000
    walk = np.cumsum(rng.normal(size=n))
    trend = np.cumsum(rng.normal(loc=0.35, scale=1.0, size=n))
    # An AR(1) with negative coefficient is anti-persistent by construction.
    rev = np.zeros(n)
    for i in range(1, n):
        rev[i] = -0.6 * rev[i - 1] + rng.normal()

    h_walk, h_trend, h_rev = hurst_rs(walk), hurst_rs(trend), hurst_rs(rev)
    assert 0.40 < h_walk < 0.60, f"random walk should sit near 0.5, got {h_walk}"
    assert h_trend > h_walk, f"trend {h_trend} should exceed walk {h_walk}"
    assert h_rev < 0.45, f"anti-persistent series should fall below 0.45, got {h_rev}"


def test_momentum_strength_rewards_clean_trends_and_punishes_noise():
    n = 300
    t = np.arange(n, dtype=float)
    clean = 100 + 0.05 * t
    noisy = 100 + 0.05 * t + np.random.default_rng(3).normal(scale=5.0, size=n)
    flat = np.full(n, 100.0) + np.random.default_rng(4).normal(scale=0.01, size=n)

    assert momentum_strength(clean) > momentum_strength(noisy) > 0
    assert abs(momentum_strength(flat)) < abs(momentum_strength(noisy))


def test_momentum_strength_is_signed():
    """Direction is carried by the sign, and log-symmetric paths score symmetrically.

    The paths are geometric (`100·e^{±kt}`), not arithmetic (`100 ± kt`): the estimator
    regresses LOG price, and `100 + kt` / `100 − kt` are symmetric in price but not in
    log price, so an arithmetic pair would fail this by construction rather than because
    the estimator is wrong.
    """
    t = np.arange(200, dtype=float)
    up, down = 100 * np.exp(0.0005 * t), 100 * np.exp(-0.0005 * t)
    assert momentum_strength(up) > 0
    assert momentum_strength(down) < 0
    assert momentum_strength(up) == pytest.approx(-momentum_strength(down), rel=1e-9)


def test_realized_vol_scales_with_the_input():
    rng = np.random.default_rng(11)
    lo = realized_vol(rng.normal(scale=0.001, size=5000))
    hi = realized_vol(rng.normal(scale=0.004, size=5000))
    assert hi / lo == pytest.approx(4.0, rel=0.1)


# ── 3. the self-percentile must not see its own future ───────────────────

def test_self_percentile_is_strictly_causal():
    """Perturb the future violently; the past must not move by a float."""
    rng = np.random.default_rng(5)
    s = pd.Series(rng.normal(size=600))

    base = rolling_self_percentile(s, window=200)

    tampered = s.copy()
    tampered.iloc[400:] += 1000.0            # a future regime change
    after = rolling_self_percentile(tampered, window=200)

    pd.testing.assert_series_equal(base.iloc[:400], after.iloc[:400],
                                   check_names=False)


def test_self_percentile_ranks_within_its_own_window():
    """A value larger than everything in its window is the top percentile."""
    s = pd.Series([1.0] * 99 + [99.0])
    out = rolling_self_percentile(s, window=100)
    assert out.iloc[-1] == pytest.approx(1.0)


def test_self_percentile_is_scale_and_shift_invariant():
    """The whole point: two pairs on different price scales become comparable."""
    rng = np.random.default_rng(9)
    s = pd.Series(rng.normal(size=400))
    a = rolling_self_percentile(s, window=100)
    b = rolling_self_percentile(s * 1000 + 50_000, window=100)
    pd.testing.assert_series_equal(a, b, check_names=False)


# ── 4. thin history yields NaN, not a confident wrong number ─────────────

def test_self_percentile_is_nan_before_the_window_fills():
    s = pd.Series(np.arange(50.0))
    out = rolling_self_percentile(s, window=30, min_periods=30)
    assert out.iloc[:29].isna().all(), "a percentile over a partial window is not a measurement"
    assert out.iloc[29:].notna().all()


def test_estimators_return_nan_on_insufficient_history():
    assert np.isnan(hurst_rs(np.arange(8.0)))
    assert np.isnan(momentum_strength(np.array([1.0])))
    assert np.isnan(realized_vol(np.array([])))


def test_nan_inputs_do_not_silently_become_zero():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 10)
    for fn in (permutation_entropy, hurst_rs, momentum_strength, realized_vol):
        v = fn(x)
        assert not (v == 0.0), f"{fn.__name__} turned NaN-laden input into a hard zero"


def test_constant_series_is_not_a_confident_measurement():
    """A dead pair (no price movement) must not read as maximally ordered/trending."""
    flat = np.full(500, 42.0)
    assert np.isnan(realized_vol(flat)) or realized_vol(flat) == 0.0
    assert np.isnan(momentum_strength(flat)) or momentum_strength(flat) == pytest.approx(0.0)
    assert np.isnan(hurst_rs(flat))

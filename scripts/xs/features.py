"""XS-2 — bar-level features for cross-sectional ranking.

Class 3 ranks the whole perp universe against itself, which raw feature values cannot
support: permutation entropy on a $64k pair with a one-tick spread and on a $0.09
memecoin are not the same measurement, and 3 % realized vol means different things for
each. `specs/maker_system.md` §5 therefore specifies every score "as percentile/z vs the
pair's own history". So this module has two halves and both matter:

  * **estimators** — `permutation_entropy`, `hurst_rs`, `momentum_strength`,
    `realized_vol`: raw, per-pair, unitless where possible.
  * **`rolling_self_percentile`** — the transform that makes them comparable, and the
    part that is easy to get subtly wrong.

`rolling_self_percentile` is **strictly causal**: the value at bar *t* ranks *t* against
bars ≤ *t* only. A percentile that can see its own future is the classic way to
manufacture cross-sectional alpha — today's vol looks extreme only because tomorrow's
calm is already in the window — and it would flatter every downstream `xs_*` process
without ever failing a smoke test.

**Bar choice is a measured constraint, not a preference.** FINDINGS §7.1: the venue keeps
~5000 bars per interval, so 1m history reaches 3.5 days while 15m reaches 52 and 1h
reaches 90+. These functions are bar-agnostic; callers should default to 15m/1h. PROC-20
independently found 1m/5m momentum anti-persistent with 5m statistically unresolvable, so
the coarser bars are the right entry regardless.

Estimator provenance:
  * `permutation_entropy` mirrors `rust/ing-features/src/entropy.rs:373` — same ordinal
    patterns, same normalisation by `ln(order!)` — so the name means one thing across the
    Rust and Python layers.
  * `hurst_rs` is classical rescaled-range. Deliberately *not* the PSD-slope estimator in
    `exploration/spannung_spectral.py` (`H = -(beta+1)/2`): that one is tied to a spectral
    pipeline and needs a long stationary tick series, where this runs on a few hundred
    bars per pair.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

__all__ = [
    "permutation_entropy",
    "hurst_rs",
    "momentum_strength",
    "realized_vol",
    "rolling_self_percentile",
]


def _clean(x) -> np.ndarray:
    """Finite values only. NaNs are dropped rather than zero-filled: a zero return is a
    claim about the market, a missing bar is not."""
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def permutation_entropy(x, order: int = 3) -> float:
    """Bandt-Pompe ordinal entropy, normalised to [0, 1].

    0 = perfectly ordered (one ordinal pattern), 1 = every pattern equally likely.
    Mirrors the Rust implementation's pattern set and `ln(order!)` normalisation.

    ⚠️ **Measured 2026-08-07: this does NOT discriminate across the perp universe at bar
    scale, and should not be used as a cross-sectional score.** On 177 pairs of 1h log
    returns it saturates — min 0.9960, median 0.9996, IQR 0.0005 — so the middle half of
    the universe is indistinguishable. Raising the order does not help: at order 6 the
    apparent spread comes from undersampling (720 patterns against ~2,156 windows is 3
    windows per pattern), which biases entropy *down* for shorter series, i.e. it ranks by
    history length rather than by disorder. Kept because it is correct, matches the Rust
    layer, and may separate at tick scale where that implementation runs — but for
    `XS-3`'s ranking use `hurst_rs` / `momentum_strength` / `realized_vol`, which do
    spread. Consistent with PROC-20 (FINDINGS §5): bar-scale returns are ordinally
    near-random for everything.

    Returns NaN when there is not enough data for at least one window, rather than the
    Rust version's 0.0 — here 0.0 is a meaningful score (perfect order) and must not be
    overloaded to mean "unknown".
    """
    a = _clean(x)
    if order < 2 or a.size < order:
        return float("nan")

    index = {p: i for i, p in enumerate(permutations(range(order)))}
    counts = np.zeros(len(index), dtype=np.int64)
    for i in range(a.size - order + 1):
        window = a[i:i + order]
        counts[index[tuple(np.argsort(window, kind="stable"))]] += 1

    total = counts.sum()
    if total == 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(len(index)))


def hurst_rs(x, min_len: int = 32) -> float:
    """Hurst exponent by rescaled range (R/S).

    ~0.5 random walk · >0.5 persistent/trending · <0.5 anti-persistent/mean-reverting.
    Operates on the series' own increments, so it is scale-invariant.

    NaN below `min_len`: R/S needs several chunk sizes to regress over, and a two-point
    "Hurst" is a number without a meaning — which on a 177-pair universe would be
    produced precisely for the newly-listed pairs.
    """
    a = _clean(x)
    if a.size < min_len:
        return float("nan")

    d = np.diff(a)
    if d.size < 8 or not np.any(d):        # a flat series has no rescaled range
        return float("nan")

    # Chunk sizes spanning the series, log-spaced.
    sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(d.size // 2), 8)).astype(int))
    sizes = sizes[sizes >= 8]
    if sizes.size < 2:
        return float("nan")

    logs_n, logs_rs = [], []
    for n in sizes:
        n_chunks = d.size // n
        if n_chunks < 1:
            continue
        rs_vals = []
        for c in range(n_chunks):
            chunk = d[c * n:(c + 1) * n]
            sd = chunk.std(ddof=1)
            if sd <= 0:
                continue
            dev = np.cumsum(chunk - chunk.mean())
            rs_vals.append((dev.max() - dev.min()) / sd)
        if rs_vals:
            logs_n.append(np.log(n))
            logs_rs.append(np.log(np.mean(rs_vals)))

    if len(logs_n) < 2:
        return float("nan")
    return float(np.polyfit(logs_n, logs_rs, 1)[0])


def momentum_strength(prices) -> float:
    """Signed trend strength: OLS slope of log-price on bar index, scaled by R².

    The product is the point. Slope alone rewards a steep line drawn through noise; R²
    alone is blind to direction and magnitude. Multiplying them means a score is high
    only when the move is both large *and* clean, which is what "momentum" has to mean
    for a cross-sectional rank to be worth anything.

    Sign carries direction. Returns NaN on insufficient or degenerate input.
    """
    a = _clean(prices)
    if a.size < 3 or np.any(a <= 0):
        return float("nan")

    y = np.log(a)
    if not np.any(y != y[0]):              # perfectly flat: no trend, and R² undefined
        return 0.0

    t = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
    # Per-bar slope scaled to the window so the score is comparable across bar counts.
    return float(slope * y.size * max(r2, 0.0))


def realized_vol(returns) -> float:
    """Standard deviation of the supplied returns (unannualised — the caller owns the clock)."""
    a = _clean(returns)
    if a.size < 2:
        return float("nan")
    return float(a.std(ddof=1))


def rolling_self_percentile(series: pd.Series, window: int = 250,
                            min_periods: int | None = None) -> pd.Series:
    """Rank each observation against the pair's own trailing `window`, in [0, 1].

    **Strictly causal**: bar *t* is ranked against bars ≤ *t*. This is the function that
    makes 177 pairs comparable, and a lookahead here would flatter every downstream
    `xs_*` score invisibly, so the causality is asserted by test (perturb the future,
    the past must not move).

    Scale- and shift-invariant by construction — which is the entire reason the
    cross-sectional layer can compare a $64k pair with a $0.09 one.

    NaN until `min_periods` observations exist: a percentile over a partial window is
    not a measurement, and on this universe the thin-history pairs are exactly the ones
    that would otherwise score extreme.
    """
    s = pd.Series(series, dtype=float)
    mp = window if min_periods is None else min_periods
    # rank(pct=True) on a trailing window puts the current bar last, so it ranks against
    # its own past only.
    return s.rolling(window, min_periods=mp).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
    )

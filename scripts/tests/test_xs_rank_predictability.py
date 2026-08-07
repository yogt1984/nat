"""XS-3 — `xs_rank_predictability`: Track C's kill test.

The question: does any per-pair score rank the universe's *relative* forward returns?
`THREE_CLASS_RESEARCH_PROPOSAL.md` §9 makes the answer terminal — "Track C stops if XS-3
finds no score family significant after FDR on >= 90 d" — so this process has to be
capable of returning nothing, and has to be trusted when it does.

Four properties carry it, and each has a way of being silently wrong:

  1. **A planted signal is recovered, and a planted null is NOT.** The second half is the
     decisive one, and the same guard PROC-3 puts on its combiner: a score unrelated to
     returns must fail the null, or every subsequent number is decoration.

  2. **IC is computed WITHIN each cross-section, never pooled.** Pooling scores and
     returns across timestamps manufactures correlation out of common time variation: if
     scores happen to be high on days when everything rallied, a pooled correlation is
     large while the actual cross-sectional ordering carries nothing. The test plants
     exactly that — a strong time effect with zero within-date ordering — and demands ~0.

  3. **Returns are relative, so a market-wide move creates no IC.** Adding the same return
     to every pair on a date must leave the verdict untouched; otherwise the process
     measures beta to the market and calls it selection skill.

  4. **The null shuffles PAIR LABELS within a cross-section.** Not returns through time —
     that would destroy each pair's own return distribution and the date structure, giving
     a null that answers a different question than the one asked.

Plus: no lookahead (the final `horizon` bars have no forward return and must yield no
cross-section), and thin cross-sections skipped with a reason rather than ranked.
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

from processes.base import ProcessContext  # noqa: E402


def _ctx(**over) -> ProcessContext:
    kw = dict(symbol="UNIVERSE", timeframe="1h", price_col="close",
              horizons={"1d": 24}, costs={}, symbols=[])
    kw.update(over)
    return ProcessContext(**kw)


def _panel(n_pairs=40, n_bars=600, seed=0, signal=0.0, market=None,
           time_effect=False) -> pd.DataFrame:
    """A long OHLCV panel with a controllable planted cross-sectional signal.

    `signal` couples each pair's *pre-assigned rank* to its forward return, so a score
    that recovers that rank should show positive rank-IC. `market` adds a common return
    to every pair on a date. `time_effect` makes the pair ordering irrelevant while the
    date-level mean varies strongly — the pooling trap.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01", periods=n_bars, freq="1h", tz="UTC")
    rows = []
    # A fixed per-pair "quality" the planted signal keys on.
    quality = np.linspace(-1, 1, n_pairs)
    for i in range(n_pairs):
        r = rng.normal(scale=0.01, size=n_bars) + signal * quality[i] * 0.01
        if market is not None:
            r = r + market
        if time_effect:
            # every pair gets the SAME date effect: no within-date ordering at all
            r = rng.normal(scale=1e-6, size=n_bars) + np.sin(np.arange(n_bars) / 20) * 0.02
        close = 100 * np.exp(np.cumsum(r))
        rows.append(pd.DataFrame({
            "timestamp": ts, "symbol": f"P{i:02d}",
            "open": close, "high": close, "low": close, "close": close,
            "volume": 1.0,
        }))
    return (pd.concat(rows, ignore_index=True)
              .sort_values(["timestamp", "symbol"]).reset_index(drop=True))


def _run(frame, **params):
    from processes.xs_rank_predictability import XsRankPredictability
    proc = XsRankPredictability(**params)
    return proc.evaluate(frame, _ctx())


# ── 1. recover a signal; reject a null ────────────────────────────────────

def test_planted_cross_sectional_signal_is_recovered():
    """A score that genuinely orders forward returns must clear the null."""
    frame = _panel(signal=3.0, seed=1)
    res = _run(frame, scores=("momentum",), lookback=120, horizon=24,
               rebalance_every=24, n_shuffles=100, seed=7)

    f = next(f for f in res.findings if "momentum" in f.feature)
    assert f.value > 0.05, f"planted signal not recovered: rank-IC {f.value}"
    assert f.extras["z"] > 3.0, f"z={f.extras['z']} — signal did not clear the null"
    assert f.informative


def test_pure_noise_does_not_clear_the_null():
    """THE guard. Random scores over random returns must return nothing informative."""
    frame = _panel(signal=0.0, seed=2)
    res = _run(frame, scores=("momentum", "hurst", "vol"), lookback=120, horizon=24,
               rebalance_every=24, n_shuffles=200, seed=11)

    assert not any(f.informative for f in res.findings), (
        "noise cleared the null: " + str([(f.feature, f.extras["z"]) for f in res.findings])
    )
    for f in res.findings:
        assert abs(f.value) < 0.1, f"{f.feature} rank-IC {f.value} on pure noise"


def test_null_mean_is_approximately_zero():
    """Shuffling labels inside a cross-section should leave no ordering — IC ~ 0."""
    res = _run(_panel(signal=2.0, seed=3), scores=("momentum",), lookback=120,
               horizon=24, rebalance_every=24, n_shuffles=200, seed=5)
    f = res.findings[0]
    assert abs(f.extras["null_mean"]) < 0.02, f"null is biased: {f.extras['null_mean']}"
    assert f.extras["null_std"] > 0


# ── 2. the pooling trap ───────────────────────────────────────────────────

def test_a_pure_time_effect_produces_no_rank_ic():
    """Every pair moves identically each date: there is no ordering to find.

    A process that pooled scores and returns across dates instead of ranking within each
    one would report a large correlation here, purely from common time variation.
    """
    frame = _panel(signal=0.0, seed=4, time_effect=True)
    res = _run(frame, scores=("momentum",), lookback=120, horizon=24,
               rebalance_every=24, n_shuffles=100, seed=3)

    f = res.findings[0]
    assert abs(f.value) < 0.15, (
        f"rank-IC {f.value} from a pure time effect — scores/returns are being pooled "
        "across dates rather than ranked within them"
    )
    assert not f.informative


# ── 3. relative returns: the market must not count ────────────────────────

def test_a_market_wide_move_does_not_change_the_verdict():
    """Cross-sectional demeaning: adding the same return to every pair changes nothing."""
    base = _panel(signal=2.0, seed=6)
    with_market = _panel(signal=2.0, seed=6, market=0.004)   # every pair, every bar

    a = _run(base, scores=("momentum",), lookback=120, horizon=24,
             rebalance_every=24, n_shuffles=50, seed=2).findings[0]
    b = _run(with_market, scores=("momentum",), lookback=120, horizon=24,
             rebalance_every=24, n_shuffles=50, seed=2).findings[0]

    assert a.value == pytest.approx(b.value, abs=0.02), (
        "a market-wide move moved the rank-IC — returns are not being cross-sectionally "
        "demeaned, so this measures beta rather than selection"
    )


# ── 4. mechanics that would corrupt the verdict quietly ───────────────────

def test_no_forward_return_means_no_cross_section():
    """The last `horizon` bars cannot have a forward return and must not be scored."""
    frame = _panel(n_bars=200, seed=8)
    res = _run(frame, scores=("momentum",), lookback=100, horizon=24,
               rebalance_every=24, n_shuffles=20, seed=1)
    last_used = pd.Timestamp(res.summary["last_rebalance"])
    assert last_used <= frame.timestamp.max() - pd.Timedelta(hours=24)


def test_thin_cross_sections_are_skipped_with_a_reason():
    frame = _panel(n_pairs=5, n_bars=400, seed=9)
    res = _run(frame, scores=("momentum",), lookback=100, horizon=24,
               rebalance_every=24, min_pairs=20, n_shuffles=20, seed=1)
    assert res.summary.get("n_rebalances", 0) == 0
    assert "min_pairs" in str(res.summary.get("skipped_reason", "")).lower() or \
           res.features_skipped, "a skipped run must say why"


def test_fdr_is_applied_across_score_variants():
    """Several scores tested at once is a multiple-comparison problem by construction."""
    res = _run(_panel(signal=0.0, seed=10), scores=("momentum", "hurst", "vol"),
               lookback=120, horizon=24, rebalance_every=24, n_shuffles=100, seed=4)
    assert len(res.findings) == 3
    for f in res.findings:
        assert f.p_adjusted is not None, "no BH-FDR q-value on a multi-score sweep"


def test_result_is_deterministic_for_a_fixed_seed():
    kw = dict(scores=("momentum",), lookback=120, horizon=24, rebalance_every=24,
              n_shuffles=50, seed=42)
    frame = _panel(signal=1.5, seed=12)
    a, b = _run(frame, **kw).findings[0], _run(frame, **kw).findings[0]
    assert a.value == b.value and a.extras["z"] == b.extras["z"]

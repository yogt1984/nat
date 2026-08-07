"""XS-4 — `xs_persistence`: does the ranking outlive the rebalance?

XS-3 established that two scores rank relative forward returns (FINDINGS §7.4). That is
necessary and not sufficient: if a score's *ranking* is reshuffled by the time you next
rebalance, then every rebalance is a fresh draw, turnover is maximal, and the strategy
pays the full spread to chase a signal that has already gone. `TASKS.md` states the bar —
"must exceed the rebalance cadence or the rotation is churn by construction".

So this measures rank autocorrelation as a function of lag, and reports a half-life.

Four properties, each with a way of being quietly wrong:

  1. **A frozen ranking is maximally persistent; a reshuffled one is not.** The two ends
     of the scale, pinned. If a noise score returned a long half-life the process would
     licence exactly the churn it exists to prevent.

  2. **A planted exponential decay is recovered.** Half-life is a fitted quantity; fitting
     it wrong is easy and invisible without a known answer.

  3. **Only pairs present at BOTH ends of a lag are compared.** The panel has holes by
     design (PROC-19): a pair listed on day 40 must not be treated as having "changed
     rank" between day 10 and day 50 — it had no rank on day 10.

  4. **Lag 0 is exactly 1.0.** A drifting normalisation would show up here first.
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


def _score_panel(scores: np.ndarray, symbols, times) -> pd.DataFrame:
    """Long frame of a PRE-COMPUTED score, shape (n_times, n_pairs)."""
    rows = []
    for j, sym in enumerate(symbols):
        rows.append(pd.DataFrame({"timestamp": times, "symbol": sym,
                                  "score": scores[:, j]}))
    return (pd.concat(rows, ignore_index=True)
              .sort_values(["timestamp", "symbol"]).reset_index(drop=True))


def _run(frame, **params):
    from processes.xs_persistence import XsPersistence
    return XsPersistence(**params).evaluate(frame, _ctx())


def _times(n):
    return pd.date_range("2026-05-01", periods=n, freq="1h", tz="UTC")


# ── 1. the two ends of the scale ──────────────────────────────────────────

def test_a_frozen_ranking_is_maximally_persistent():
    n_t, n_p = 200, 40
    fixed = np.tile(np.linspace(-1, 1, n_p), (n_t, 1))     # identical every date
    syms = [f"P{i:02d}" for i in range(n_p)]

    res = _run(_score_panel(fixed, syms, _times(n_t)), score_col="score",
               max_lag=48, cadence=24)
    f = res.findings[0]
    assert f.extras["autocorr"][1] > 0.99, "a frozen ranking must not decay"
    assert f.value > 48, f"half-life {f.value} — a constant ranking never halves"
    assert f.informative, "persistence far beyond the cadence must pass"


def test_a_reshuffled_ranking_is_churn():
    """The verdict that protects the track: no persistence -> not tradeable at any cadence."""
    rng = np.random.default_rng(0)
    n_t, n_p = 300, 40
    noise = rng.normal(size=(n_t, n_p))
    syms = [f"P{i:02d}" for i in range(n_p)]

    res = _run(_score_panel(noise, syms, _times(n_t)), score_col="score",
               max_lag=48, cadence=24)
    f = res.findings[0]
    assert abs(f.extras["autocorr"][1]) < 0.15, "noise ranks should not autocorrelate"
    assert f.value < 24, f"half-life {f.value} on pure noise"
    assert not f.informative, "a churn score must not be reported as tradeable"
    assert "cadence" in f.extras["verdict"].lower()


# ── 2. a planted decay is recovered ───────────────────────────────────────

def test_planted_exponential_decay_is_recovered():
    """AR(1) on the scores gives a known rank half-life: ln(0.5)/ln(phi)."""
    rng = np.random.default_rng(1)
    n_t, n_p, phi = 4000, 60, 0.97
    x = np.zeros((n_t, n_p))
    x[0] = rng.normal(size=n_p)
    for t in range(1, n_t):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_p)

    syms = [f"P{i:02d}" for i in range(n_p)]
    res = _run(_score_panel(x, syms, _times(n_t)), score_col="score",
               max_lag=120, cadence=1, stride=10)

    expected = np.log(0.5) / np.log(phi)          # ~22.8 bars
    got = res.findings[0].value
    assert got == pytest.approx(expected, rel=0.35), f"half-life {got} vs planted {expected}"


def test_lag_zero_autocorrelation_is_one():
    rng = np.random.default_rng(2)
    n_t, n_p = 100, 30
    syms = [f"P{i:02d}" for i in range(n_p)]
    res = _run(_score_panel(rng.normal(size=(n_t, n_p)), syms, _times(n_t)),
               score_col="score", max_lag=10, cadence=5)
    assert res.findings[0].extras["autocorr"][0] == pytest.approx(1.0)


# ── 3. the panel's holes are respected ────────────────────────────────────

def test_pairs_absent_at_one_end_of_the_lag_are_excluded():
    """A pair listed late has no earlier rank — it cannot have 'changed' rank."""
    n_t, n_p = 200, 30
    fixed = np.tile(np.linspace(-1, 1, n_p), (n_t, 1))
    syms = [f"P{i:02d}" for i in range(n_p)]
    frame = _score_panel(fixed, syms, _times(n_t))
    # P00 only exists in the second half
    frame = frame[~((frame.symbol == "P00") & (frame.timestamp < _times(n_t)[100]))]

    res = _run(frame, score_col="score", max_lag=24, cadence=12)
    f = res.findings[0]
    assert f.extras["autocorr"][1] > 0.99, (
        "a late listing perturbed the autocorrelation — pairs absent at one end of the "
        "lag must be dropped from that comparison, not treated as rank changes"
    )


def test_thin_cross_sections_are_skipped(tmp_path):
    n_t, n_p = 100, 4
    rng = np.random.default_rng(3)
    syms = [f"P{i}" for i in range(n_p)]
    res = _run(_score_panel(rng.normal(size=(n_t, n_p)), syms, _times(n_t)),
               score_col="score", max_lag=10, cadence=5, min_pairs=20)
    assert res.summary.get("n_lags", 0) == 0 or not res.findings
    assert res.features_skipped or res.summary.get("skipped_reason")


# ── 4. the verdict is against the cadence, not an absolute ────────────────

def test_the_same_score_passes_at_a_slow_cadence_and_fails_at_a_fast_one():
    """Persistence is only meaningful relative to how often you trade."""
    rng = np.random.default_rng(4)
    n_t, n_p, phi = 3000, 50, 0.95
    x = np.zeros((n_t, n_p))
    x[0] = rng.normal(size=n_p)
    for t in range(1, n_t):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_p)
    syms = [f"P{i:02d}" for i in range(n_p)]
    frame = _score_panel(x, syms, _times(n_t))

    fast = _run(frame, score_col="score", max_lag=100, cadence=60, stride=10).findings[0]
    slow = _run(frame, score_col="score", max_lag=100, cadence=2, stride=10).findings[0]

    assert fast.value == pytest.approx(slow.value, rel=1e-9), "half-life must not depend on cadence"
    assert slow.informative and not fast.informative

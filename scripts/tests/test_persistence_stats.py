"""PROC-20 — `persistence_stats`: does momentum persist, and do band excursions revert?

Two families, one failure mode. Both statistics are *conditional means over selected
events* — `P(continue | run length k)` and `E[markout | touch at k·sigma]` — and selection
plus conditioning is the most reliable way to manufacture a number. A run-length-5 bucket
holds a handful of events; a 3-sigma band holds fewer. Without a null that shuffles the
outcome while keeping the selection fixed, every one of these cells will look significant
somewhere in the grid.

So the mirror test is the important one: on a **random walk** the process must report
`P(continue) ~ 0.5`, no excess over null, and `informative = False` for every k. If that
ever passes by accident, both families become noise generators with a §5 table.

The rest guards the specific ways this unit could lie: look-ahead in the rolling band
(fit windows must be causal), double-counted overlapping touches (embargo), pooled days
hiding an intermittent effect, and grid-wide multiple testing (FDR across k × horizon).
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

DAY_NS = 86_400_000_000_000


def _ctx(**over) -> ProcessContext:
    kw = dict(symbol="BTC", timeframe="5min", price_col="raw_midprice",
              horizons={"h1": 1, "h4": 4}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _prices_from_returns(r: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.exp(np.cumsum(r))


def _frame(returns_by_day, seed=0) -> pd.DataFrame:
    """One block per day; `returns_by_day[i]` is that day's bar-return series."""
    frames, price = [], 100.0
    for d, r in enumerate(returns_by_day):
        p = _prices_from_returns(np.asarray(r, dtype=float), start=price)
        price = float(p[-1])
        frames.append(pd.DataFrame({
            "timestamp_ns": d * DAY_NS + np.arange(len(p), dtype=np.int64) * 300_000_000_000,
            "symbol": "BTC",
            "raw_midprice": p,
            "flow_volume_1s": np.full(len(p), 1.0),
        }))
    return pd.concat(frames, ignore_index=True)


def _ar1_returns(n, phi, scale=0.001, seed=0) -> np.ndarray:
    """AR(1) returns: phi > 0 persists (runs longer than geometric), phi < 0 alternates."""
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=scale, size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def _ou_band_prices(n, theta=0.05, sigma=0.004, seed=0):
    """Price = slow midline + mean-reverting deviation, so k-sigma touches should revert."""
    rng = np.random.default_rng(seed)
    dev = np.zeros(n)
    for t in range(1, n):
        dev[t] = dev[t - 1] * (1 - theta) + rng.normal(scale=sigma)
    mid = 100.0 + np.cumsum(rng.normal(scale=0.0005, size=n))
    return mid * (1.0 + dev)


def _run(df, ctx=None, **params):
    from processes.persistence_stats import PersistenceStatsProcess
    params.setdefault("n_shuffles", 60)
    params.setdefault("day_shuffles", 30)
    return PersistenceStatsProcess(**params).evaluate(df, ctx or _ctx())


def _cells(result, family, metric=None):
    out = [f for f in result.findings if f.extras.get("family") == family]
    if metric:
        out = [f for f in out if f.metric == metric]
    assert out, f"no {family}/{metric} findings: {[(f.metric, f.extras.get('family')) for f in result.findings]}"
    return out


# ── the mirror test: a random walk must produce nothing ──────────────────────────
class TestRandomWalkProducesNothing:
    def test_no_momentum_cell_is_informative(self):
        rng = np.random.default_rng(5)
        df = _frame([rng.normal(scale=0.001, size=900) for _ in range(6)])
        res = _run(df)
        cells = _cells(res, "momentum", "p_continue_excess")
        assert not any(c.informative for c in cells), \
            [(c.extras["run_length"], c.value, c.extras["z"]) for c in cells if c.informative]

    def test_continuation_rate_sits_at_one_half(self):
        """Within a binomial band, not a magic constant: a k-bucket with n events has
        standard error 0.5/sqrt(n), so the tolerance has to scale with the bucket size or
        the test just fails on whichever bucket happens to be smallest."""
        rng = np.random.default_rng(6)
        df = _frame([rng.normal(scale=0.001, size=1200) for _ in range(5)])
        res = _run(df)
        for c in _cells(res, "momentum", "p_continue_excess"):
            n = c.extras["n_events"]
            if n >= 200:
                tol = 3.5 * 0.5 / np.sqrt(n)
                assert abs(c.extras["p_continue"] - 0.5) < tol, (c.extras, tol)

    def test_no_band_cell_is_informative_on_a_random_walk(self):
        rng = np.random.default_rng(7)
        df = _frame([rng.normal(scale=0.001, size=1500) for _ in range(6)])
        res = _run(df)
        assert not any(c.informative for c in _cells(res, "band", "markout_bps"))


# ── family A: momentum persistence ───────────────────────────────────────────────
class TestMomentumPersistence:
    def test_positive_ar1_is_persistent_above_null(self):
        df = _frame([_ar1_returns(1200, phi=0.35, seed=d) for d in range(6)])
        cells = {c.extras["run_length"]: c for c in
                 _cells(_run(df), "momentum", "p_continue_excess")}
        c1 = cells[1]
        assert c1.extras["p_continue"] > 0.55, c1.extras
        assert c1.value > 0, "excess over null must be positive for a persistent series"
        assert c1.informative and c1.extras["z"] > 0

    def test_negative_ar1_is_anti_persistent(self):
        df = _frame([_ar1_returns(1200, phi=-0.35, seed=d) for d in range(6)])
        cells = {c.extras["run_length"]: c for c in
                 _cells(_run(df), "momentum", "p_continue_excess")}
        assert cells[1].extras["p_continue"] < 0.45, cells[1].extras
        assert cells[1].value < 0

    def test_markout_follows_the_run_direction(self):
        df = _frame([_ar1_returns(1200, phi=0.35, seed=d) for d in range(6)])
        mk = [c for c in _cells(_run(df), "momentum", "markout_bps")
              if c.extras["run_length"] == 1 and c.extras["horizon"] == "h1"]
        assert mk and mk[0].value > 0, "persistent runs must mark out positively"

    def test_run_length_distribution_is_reported(self):
        df = _frame([_ar1_returns(900, phi=0.35, seed=d) for d in range(4)])
        res = _run(df)
        dist = res.summary["run_length_distribution"]
        assert sum(dist.values()) > 0 and "1" in dist


# ── family B: band excursion ─────────────────────────────────────────────────────
class TestBandExcursion:
    def test_deep_touches_revert_on_a_planted_ou_band(self):
        p = _ou_band_prices(6000, seed=1)
        df = _frame_from_prices(p, n_days=6)
        res = _run(df, vwap_window=60, k_grid=[0.5, 2.0], embargo_bars=20)
        by_k = {c.extras["k"]: c for c in _cells(res, "band", "markout_bps")
                if c.extras["horizon"] == "h4"}
        assert by_k[2.0].value > by_k[0.5].value, (
            f"deep touches should revert more than shallow ones: {by_k[2.0].value} "
            f"vs {by_k[0.5].value}")

    def test_event_counts_and_embargo_are_reported(self):
        p = _ou_band_prices(4000, seed=2)
        df = _frame_from_prices(p, n_days=4)
        res = _run(df, k_grid=[1.0, 2.0], embargo_bars=25)
        for c in _cells(res, "band", "markout_bps"):
            assert c.extras["n_events"] >= 0 and c.extras["embargo_bars"] == 25

    def test_embargo_removes_overlapping_touches(self):
        p = _ou_band_prices(4000, seed=3)
        df = _frame_from_prices(p, n_days=4)
        loose = _run(df, k_grid=[1.0], embargo_bars=1)
        tight = _run(df, k_grid=[1.0], embargo_bars=100)
        n_loose = _cells(loose, "band", "markout_bps")[0].extras["n_events"]
        n_tight = _cells(tight, "band", "markout_bps")[0].extras["n_events"]
        assert n_tight < n_loose, "a longer embargo must yield strictly fewer events"

    def test_time_to_revert_is_reported(self):
        p = _ou_band_prices(4000, seed=4)
        df = _frame_from_prices(p, n_days=4)
        res = _run(df, k_grid=[2.0])
        tt = _cells(res, "band", "time_to_revert")
        assert tt and (tt[0].value > 0 or tt[0].extras["n_events"] == 0)


# ── causality ────────────────────────────────────────────────────────────────────
class TestCausality:
    def test_corrupting_the_tail_cannot_change_early_events(self):
        """Rolling midline/sigma must be causal: the last day's prices cannot alter the
        statistics of touches that happened on day 1."""
        p = _ou_band_prices(4000, seed=8)
        df = _frame_from_prices(p, n_days=4)
        base = _run(df, k_grid=[2.0], day_shuffles=20)

        poisoned = df.copy()
        cut = int(len(df) * 0.75)
        rng = np.random.default_rng(99)
        poisoned.loc[poisoned.index[cut:], "raw_midprice"] *= (
            1 + rng.normal(scale=0.05, size=len(df) - cut))
        after = _run(poisoned, k_grid=[2.0], day_shuffles=20)

        def day1(res):
            c = _cells(res, "band", "markout_bps")[0]
            per_day = {d["day"]: d for d in c.extras["per_day"]}
            first = min(per_day)
            return per_day[first]

        a, b = day1(base), day1(after)
        assert a["n_events"] == b["n_events"] and a["value"] == pytest.approx(b["value"]), (
            "day-1 band statistics moved when only the tail changed — the rolling "
            "window is looking ahead")

    def test_markout_never_uses_the_event_bar_itself(self):
        """A flat tail after each event must produce zero markout, not the event's own move."""
        n = 1200
        r = np.zeros(n)
        r[::50] = 0.01                       # isolated jumps, flat everywhere else
        df = _frame([r] * 3)
        res = _run(df, k_grid=[1.0])
        for c in _cells(res, "momentum", "markout_bps"):
            if c.extras["n_events"] > 10:
                assert abs(c.value) < 5.0, (c.metric, c.value, c.extras)


# ── day folds and multiple testing ───────────────────────────────────────────────
class TestDayFoldsAndFDR:
    def test_intermittent_persistence_is_not_hidden_by_pooling(self):
        days = [_ar1_returns(1000, phi=0.45, seed=d) for d in range(4)] + \
               [_ar1_returns(1000, phi=0.0, seed=10 + d) for d in range(4)]
        res = _run(_frame(days))
        c = {x.extras["run_length"]: x for x in
             _cells(res, "momentum", "p_continue_excess")}[1]
        assert 0.25 <= c.extras["frac_days_informative"] <= 0.75, c.extras
        assert c.extras["verdict"] == "non_durable"

    def test_constant_persistence_is_durable(self):
        res = _run(_frame([_ar1_returns(1000, phi=0.45, seed=d) for d in range(8)]))
        c = {x.extras["run_length"]: x for x in
             _cells(res, "momentum", "p_continue_excess")}[1]
        assert c.extras["frac_days_informative"] >= 0.7
        assert c.extras["verdict"] == "durable"

    def test_fdr_is_applied_across_the_grid(self):
        res = _run(_frame([_ar1_returns(900, phi=0.3, seed=d) for d in range(5)]))
        assert res.summary.get("fdr", {}).get("n_cells", 0) > 1
        assert any(f.p_adjusted is not None for f in res.findings), \
            "a k x horizon sweep must carry BH q-values"


# ── contract ─────────────────────────────────────────────────────────────────────
class TestContract:
    def test_registered_and_bar_level(self):
        from processes.persistence_stats import PersistenceStatsProcess
        from processes.registry import list_processes
        assert "persistence_stats" in list_processes()
        assert PersistenceStatsProcess.data_level == "bars"

    def test_input_is_not_mutated(self):
        df = _frame([_ar1_returns(700, phi=0.3, seed=d) for d in range(3)])
        before = df.copy()
        _run(df)
        pd.testing.assert_frame_equal(df, before)

    def test_missing_time_column_is_reported(self):
        df = _frame([_ar1_returns(700, phi=0.3, seed=1)]).drop(columns=["timestamp_ns"])
        res = _run(df)
        assert res.summary.get("error")

    def test_too_few_events_is_said_not_averaged(self):
        res = _run(_frame([_ar1_returns(400, phi=0.3, seed=1)]), k_grid=[3.5],
                   min_events=10_000)
        for c in [f for f in res.findings if f.extras.get("family") == "band"]:
            assert c.extras["verdict"] in ("insufficient_events", "insufficient_days") \
                or not c.informative

    def test_thresholds_are_imported_not_written_here(self):
        from processes import persistence_stats
        src = Path(persistence_stats.__file__).read_text()
        assert "load_null_config" in src and "DEFAULT_FDR_ALPHA" in src
        assert "= 3.0" not in src and "= 0.05" not in src

    def test_determinism(self):
        df = _frame([_ar1_returns(800, phi=0.3, seed=d) for d in range(4)])
        a = _cells(_run(df, seed=1), "momentum", "p_continue_excess")[0]
        b = _cells(_run(df, seed=1), "momentum", "p_continue_excess")[0]
        assert a.value == b.value and a.extras["z"] == b.extras["z"]


def _frame_from_prices(p: np.ndarray, n_days: int) -> pd.DataFrame:
    """Split a single price path into `n_days` contiguous day blocks."""
    per = len(p) // n_days
    frames = []
    for d in range(n_days):
        seg = p[d * per:(d + 1) * per]
        frames.append(pd.DataFrame({
            "timestamp_ns": d * DAY_NS + np.arange(len(seg), dtype=np.int64) * 300_000_000_000,
            "symbol": "BTC",
            "raw_midprice": seg,
            "flow_volume_1s": np.full(len(seg), 1.0),
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

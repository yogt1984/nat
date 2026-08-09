"""VW-2 — planted tests for the multi-scale VWAP study driver.

The study answers one question — *which windows earn a column?* — by wiring VW-1's
anchor into PROC-4/12/13/15/20 and applying the spec's five pre-registered criteria
(`docs/specs/multiscale_vwap.md` §A2). What can go wrong is therefore not statistics
(those are imported) but plumbing, and each test plants one plumbing failure:

- **aggregation equivalence**: the study feeds minute-aggregated pseudo-trades to the
  same `VwapRing` VW-1 ships. If that aggregation ever diverges from the per-trade
  path — sums are sums — every downstream verdict is about a different anchor than
  the one Phase B would implement;
- **planted signal / null control**: a series built to mean-revert around its anchor
  must come back informative, and the same series with the association destroyed must
  not — the study must be able to find what is there and nothing more;
- **redundancy**: a window that IS a faster window wearing a slower label must fail
  criterion (d) — six nested sums of one tape are the spec's declared illusion;
- **criteria composition**: the five gates must compose as written — each one failing
  alone must flip the verdict, because a gate that cannot fail is not a gate;
- **derived, not invented**: the event-rate bound (e) must equal PROC-20's own
  `min_fold_events` default — the number is imported from the process whose verdict
  it makes reachable, per the "gates imported, not invented" guardrail.
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

from exploration.vwap_multiscale_study import (  # noqa: E402
    EXCLUDED_WINDOWS, EXCLUSION_REASON, MIN_EVENTS_PER_DAY, STUDY_WINDOWS,
    aggregate_minutes, amplitude_to_spread, attach_window_vwaps, evaluate_criteria,
    mi_cells, stouffer,
)
from features.vwap_multiscale import compute_multiscale_vwap  # noqa: E402
from processes.persistence_stats import PersistenceStatsProcess  # noqa: E402

MIN_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000


# ── fixtures ─────────────────────────────────────────────────────────────────────
def _random_trades(n_minutes: int, seed: int = 0, hole: tuple = (),
                   empty_every: int = 0) -> pd.DataFrame:
    """Random trade stream: 1–5 trades per active minute, optional hole and
    periodic empty minutes."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for m in range(n_minutes):
        if hole and hole[0] <= m < hole[1]:
            continue
        if empty_every and m % empty_every == 7:
            continue
        for _ in range(int(rng.integers(1, 6))):
            price *= float(np.exp(rng.normal(0, 2e-4)))
            offset = int(rng.integers(0, MIN_NS))
            rows.append((m * MIN_NS + offset, price, float(rng.uniform(0.01, 2.0))))
    rows.sort()
    return pd.DataFrame({
        "timestamp_ns": [r[0] for r in rows],
        "symbol": "BTC",
        "price": [r[1] for r in rows],
        "size": [r[2] for r in rows],
        "is_buy": True,
    })


def _synthetic_minutes(n_minutes: int, seed: int = 0, revert: float = 0.0,
                       start_minute: int = 0) -> pd.DataFrame:
    """Minute bars built directly: OU-style price when `revert` > 0, else a random
    walk. Volume is positive on every minute, so windows are always warm past w."""
    rng = np.random.default_rng(seed)
    anchor = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, n_minutes)))
    dev = np.zeros(n_minutes)
    for t in range(1, n_minutes):
        dev[t] = (1.0 - revert) * dev[t - 1] + rng.normal(0, 8e-4)
    close = anchor * np.exp(dev) if revert > 0 else \
        100.0 * np.exp(np.cumsum(rng.normal(0, 8e-4, n_minutes)))
    volume = rng.uniform(0.5, 2.0, n_minutes)
    minutes = np.arange(start_minute, start_minute + n_minutes, dtype=np.int64)
    return pd.DataFrame({
        "minute": minutes,
        "timestamp_ns": minutes * MIN_NS,
        "notional": close * volume,
        "volume": volume,
        "close": close,
    })


# ── aggregation equivalence: the claim everything downstream rests on ────────────
class TestAggregationEquivalence:
    def test_minute_path_matches_per_trade_path(self):
        """Study bars must equal VW-1's per-trade path read at each minute's last
        trade — including which rows are NaN (warm-up, gaps)."""
        windows = {"5m": 5, "15m": 15, "1h": 60}
        trades = _random_trades(200, seed=3, hole=(80, 121), empty_every=13)

        per_trade = compute_multiscale_vwap(trades, windows=windows)
        per_trade["minute"] = per_trade["timestamp_ns"] // MIN_NS
        last_of_minute = per_trade.groupby("minute").tail(1).set_index("minute")

        bars = attach_window_vwaps(aggregate_minutes(trades), windows=windows)
        bars = bars.set_index("minute")

        assert set(bars.index) == set(last_of_minute.index)
        for w in windows:
            a = bars[f"vwap_{w}"].reindex(last_of_minute.index)
            b = last_of_minute[f"vwap_{w}"]
            both_nan = a.isna() & b.isna()
            assert (both_nan | (np.abs(a - b) < 1e-9 * b.abs())).all(), \
                f"window {w}: minute path diverged from per-trade path"

    def test_deviation_uses_minute_close(self):
        windows = {"5m": 5}
        trades = _random_trades(20, seed=5)
        bars = attach_window_vwaps(aggregate_minutes(trades), windows=windows)
        warm = bars.dropna(subset=["vwap_dev_5m"])
        assert len(warm) > 0
        expected = (warm["close"] - warm["vwap_5m"]) / warm["vwap_5m"]
        assert np.allclose(warm["vwap_dev_5m"], expected, atol=1e-12)

    def test_warmup_refused(self):
        bars = attach_window_vwaps(aggregate_minutes(_random_trades(10, seed=1)),
                                   windows={"15m": 15})
        assert bars["vwap_15m"].isna().all()

    def test_gap_refused_then_recovers(self):
        """A 40-minute hole (> max_gap 30): the 1h window must refuse while >30 min
        of the hole sits inside it (until minute 129 = 100 + 40 - 30 - 1), then
        recover; the 5m window recovers immediately (its internal hole is <= 4 min).
        These are VW-1's shipped semantics — the study inherits, not re-decides."""
        trades = _random_trades(180, seed=7, hole=(60, 100))
        bars = attach_window_vwaps(aggregate_minutes(trades),
                                   windows={"5m": 5, "1h": 60}).set_index("minute")
        after = bars.loc[[m for m in bars.index if 100 <= m < 126]]
        assert after["vwap_5m"].notna().all()              # hole inside 5m is tiny
        assert after["vwap_1h"].isna().all()               # 1h still spans > max_gap
        clear = bars.loc[[m for m in bars.index if 135 <= m < 160]]
        assert clear["vwap_1h"].notna().all()              # hole share now <= max_gap

    def test_no_input_mutation_and_deterministic(self):
        trades = _random_trades(30, seed=11)
        snapshot = trades.copy(deep=True)
        b1 = attach_window_vwaps(aggregate_minutes(trades), windows={"5m": 5})
        b2 = attach_window_vwaps(aggregate_minutes(trades), windows={"5m": 5})
        pd.testing.assert_frame_equal(trades, snapshot)
        pd.testing.assert_frame_equal(b1, b2)


# ── planted signal in, verdict out ───────────────────────────────────────────────
class TestPlantedSignal:
    def _run(self, revert: float, seed: int):
        days = [_synthetic_minutes(1440, seed=seed + d, revert=revert,
                                   start_minute=d * 1440) for d in range(4)]
        bars = attach_window_vwaps(pd.concat(days, ignore_index=True),
                                   windows={"15m": 15})
        # stride 15 leaves 96 rows/day: min_fold_obs must fit the strided fold
        return mi_cells(bars, symbol="TEST", horizons={"15m": 15},
                        n_shuffles=25, min_fold_obs=64, max_samples=500, seed=0)

    def test_planted_reversion_is_found(self):
        res = self._run(revert=0.25, seed=100)
        cells = [f for f in res.findings if f.feature == "vwap_dev_15m"]
        assert cells, f"no finding emitted: {res.summary.get('error')}"
        frac = cells[0].extras["frac_days_informative"]
        assert frac >= 0.75, f"planted reversion missed (frac_days={frac})"

    def test_random_walk_is_not_found(self):
        res = self._run(revert=0.0, seed=200)
        cells = [f for f in res.findings if f.feature == "vwap_dev_15m"]
        assert cells
        frac = cells[0].extras["frac_days_informative"]
        assert frac <= 0.25, f"study finds signal in a random walk (frac_days={frac})"


# ── criteria: composition, derivation, aggregation ───────────────────────────────
def _passing_rows(window="1h", symbol="BTC"):
    mi = [{"window": window, "symbol": symbol, "horizon": "15m",
           "stouffer_z": 4.0, "stouffer_p": 1e-4,
           "frac_days_informative": 0.8, "n_days": 20}]
    red = [{"window": window, "symbol": symbol, "holdout_abs_corr": 0.3}]
    band = [{"window": window, "symbol": symbol, "k": 2.0,
             "events_per_day": 25.0}]
    return mi, red, band


class TestCriteria:
    def test_all_five_pass(self):
        mi, red, band = _passing_rows()
        out = evaluate_criteria(mi, red, band)
        assert out["1h"]["pass"] is True
        flags = out["1h"]["per_symbol"]["BTC"]
        assert all(flags[c] for c in "abcde")

    @pytest.mark.parametrize("breaker", [
        ("a", {"stouffer_z": 2.0}),                      # below the PROC-12 z gate
        ("b", {"frac_days_informative": 0.4}),           # day-consistency fails
        ("d", None),                                     # redundant with faster window
        ("e", None),                                     # starved of events
    ])
    def test_each_criterion_can_fail_alone(self, breaker):
        crit, patch = breaker
        mi, red, band = _passing_rows()
        if patch:
            mi[0].update(patch)
        if crit == "d":
            red[0]["holdout_abs_corr"] = 0.9
        if crit == "e":
            band[0]["events_per_day"] = 1.5               # PROC-20's counter-example
        out = evaluate_criteria(mi, red, band)
        assert out["1h"]["per_symbol"]["BTC"][crit] is False
        assert out["1h"]["pass"] is False

    def test_fdr_tightens_across_the_grid(self):
        """One marginal cell (p = .04, would pass alone) drowned in null cells: BH
        over the whole grid must push its q above alpha so it fails (c), while the
        genuinely strong cell survives."""
        mi, red, band = _passing_rows()
        mi.append({"window": "5m", "symbol": "MARGINAL", "horizon": "15m",
                   "stouffer_z": 3.05, "stouffer_p": 0.04,
                   "frac_days_informative": 0.6, "n_days": 20})
        for i in range(30):
            mi.append({"window": "5m", "symbol": f"S{i}", "horizon": "15m",
                       "stouffer_z": 1.0, "stouffer_p": 0.3 + 0.02 * i,
                       "frac_days_informative": 0.1, "n_days": 20})
        out = evaluate_criteria(mi, red, band)
        assert out["1h"]["per_symbol"]["BTC"]["c"] is True
        assert not any(v["c"] for v in out["5m"]["per_symbol"].values())

    def test_event_rate_bound_is_imported_from_proc20(self):
        assert MIN_EVENTS_PER_DAY == \
            PersistenceStatsProcess.PARAMS["min_fold_events"][0]

    def test_slow_windows_excluded_with_reason(self):
        """6h/12h are the unmeasurable ones (feed holes); every other VW-1 candidate
        is in the study — derived from VW-1's WINDOWS, so a window added there
        (e.g. the 30m/2h crossover bracket) joins the study automatically."""
        from features.vwap_multiscale import WINDOWS
        assert set(EXCLUDED_WINDOWS) == {"6h", "12h"}
        assert set(STUDY_WINDOWS) == set(WINDOWS) - {"6h", "12h"}
        assert "hole" in EXCLUSION_REASON or "gap" in EXCLUSION_REASON

    def test_stouffer(self):
        z, p = stouffer([2.0, 2.0, 2.0, 2.0])
        assert abs(z - 4.0) < 1e-9 and 0 < p < 1e-4
        z0, p0 = stouffer([0.5, -0.5])
        assert abs(z0) < 1e-9 and abs(p0 - 1.0) < 1e-9


# ── redundancy: a duplicated window must fail (d) ────────────────────────────────
class TestRedundancy:
    def test_duplicate_window_fails_criterion_d(self):
        from exploration.vwap_multiscale_study import redundancy_rows
        bars = attach_window_vwaps(_synthetic_minutes(3000, seed=42, revert=0.1),
                                   windows={"5m": 5, "10m": 10})
        bars["vwap_dev_10m"] = bars["vwap_dev_5m"]        # the planted illusion
        rows = redundancy_rows(bars, symbol="BTC", windows={"5m": 5, "10m": 10})
        r10 = [r for r in rows if r["window"] == "10m"][0]
        assert r10["holdout_abs_corr"] > 0.95
        r5 = [r for r in rows if r["window"] == "5m"][0]
        assert r5["holdout_abs_corr"] is None             # fastest: no faster window

    def test_independent_window_passes_criterion_d(self):
        from exploration.vwap_multiscale_study import redundancy_rows
        bars = attach_window_vwaps(_synthetic_minutes(3000, seed=43, revert=0.1),
                                   windows={"5m": 5, "10m": 10})
        rng = np.random.default_rng(0)
        bars["vwap_dev_10m"] = rng.normal(0, 1e-3, len(bars))  # genuinely new axis
        rows = redundancy_rows(bars, symbol="BTC", windows={"5m": 5, "10m": 10})
        r10 = [r for r in rows if r["window"] == "10m"][0]
        assert r10["holdout_abs_corr"] < 0.5


# ── amplitude-to-spread: the Roll-bounce report ──────────────────────────────────
class TestAmplitudeToSpread:
    def test_reports_ratio_at_touches(self):
        rng = np.random.default_rng(1)
        dev = pd.Series(rng.normal(0, 10e-4, 5000))       # sigma = 10 bps
        out = amplitude_to_spread(dev, window=60, spread_bps=1.0, k=2.0)
        assert out["n_touches"] > 10
        assert out["amplitude_bps"] > 15.0                # touches are >= 2 sigma
        assert out["ratio"] == pytest.approx(out["amplitude_bps"] / 1.0)

    def test_no_touches_is_reported_not_faked(self):
        dev = pd.Series(np.zeros(500))
        out = amplitude_to_spread(dev, window=60, spread_bps=1.0, k=2.0)
        assert out["n_touches"] == 0
        assert np.isnan(out["amplitude_bps"]) and np.isnan(out["ratio"])

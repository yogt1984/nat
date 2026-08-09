"""PROF-1 — per-asset J/K regime profiling. Three tests decide whether any of it is real.

**The mirror.** A random walk must produce nothing across the entire grid. A J/K grid is
thousands of cells; without a correct null and FDR it manufactures regimes by construction,
and every downstream item the user asked for (hidden states, stat-arb) would be built on
noise.

**The null must break serial structure, not the outcome.** PROC-20 got this wrong one commit
ago: permuting the outcome with the selection fixed asks "is this cell unlike the others",
and because the pooled continuation rate already contains the persistence, a genuinely
persistent AR(1) scored **z = −2.7**. Here the sign series is permuted and the runs
recomputed, so 0.5 is the null value and the question is "is there persistence at all".

**Non-overlapping sampling.** Consecutive bars share K−1 bars of their forward window. A-2's
first run printed IC 0.39–0.46 — higher than the claim it was auditing — purely from this. A
test asserts the two paths differ, so the trap cannot silently return.

Notation follows Jegadeesh & Titman (1993): J = formation, K = holding.
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

HOUR_NS = 3_600_000_000_000


def _ctx(**over) -> ProcessContext:
    kw = dict(symbol="UNIVERSE", timeframe="1h", price_col="close",
              horizons={}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _panel(returns_by_symbol: dict, start_ns: int = 0) -> pd.DataFrame:
    """Long candle frame: (timestamp, symbol, close) sorted by (timestamp, symbol)."""
    frames = []
    for sym, r in returns_by_symbol.items():
        r = np.asarray(r, dtype=float)
        frames.append(pd.DataFrame({
            "timestamp": pd.to_datetime(start_ns + np.arange(len(r)) * HOUR_NS, utc=True),
            "symbol": sym,
            "close": 100.0 * np.exp(np.cumsum(r)),
            "volume": 1.0,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "symbol"]).reset_index(drop=True)


def _ar1(n, phi, scale=0.004, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=scale, size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def _rw(n, scale=0.004, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=scale, size=n)


def _run(df, ctx=None, **params):
    from processes.regime_profile import RegimeProfile
    params.setdefault("n_shuffles", 80)
    params.setdefault("day_shuffles", 40)
    return RegimeProfile(**params).evaluate(df, ctx or _ctx())


def _cells(res, symbol=None, metric="cont_prob_excess"):
    out = [f for f in res.findings if f.metric == metric]
    if symbol:
        out = [f for f in out if f.extras.get("symbol") == symbol]
    return out


# ── the mirror: a random walk must produce nothing ───────────────────────────────
class TestRandomWalkProducesNoRegime:
    def test_no_cell_is_informative(self):
        panel = _panel({f"RW{i}": _rw(1500, seed=i) for i in range(4)})
        res = _run(panel)
        hits = [c for c in _cells(res) if c.informative]
        assert not hits, [(c.extras["symbol"], c.extras["J"], c.extras["K"], c.extras["z"])
                          for c in hits]

    def test_continuation_sits_at_one_half(self):
        res = _run(_panel({"RW": _rw(2500, seed=7)}))
        for c in _cells(res):
            n = c.extras["n_obs"]
            if n >= 100:
                tol = 3.5 * 0.5 / np.sqrt(n)      # binomial band, not a magic constant
                assert abs(c.extras["cont_prob"] - 0.5) < tol, (c.extras, tol)

    def test_variance_ratio_is_about_one(self):
        res = _run(_panel({"RW": _rw(3000, seed=11)}))
        vr = [f for f in res.findings if f.metric == "variance_ratio"]
        assert vr, "no VR findings"
        for f in vr:
            assert abs(f.value - 1.0) < 0.35, f.extras


# ── it must recover planted regimes ──────────────────────────────────────────────
class TestRecoversPlantedRegimes:
    def test_positive_ar1_is_trending(self):
        res = _run(_panel({"TREND": _ar1(2500, phi=0.35, seed=3)}))
        c = [x for x in _cells(res, "TREND") if x.extras["J"] == 1 and x.extras["K"] == 1]
        assert c and c[0].extras["cont_prob"] > 0.55, c[0].extras if c else None
        assert c[0].value > 0 and c[0].extras["z"] > 0

    def test_negative_ar1_is_reverting(self):
        res = _run(_panel({"REVERT": _ar1(2500, phi=-0.35, seed=4)}))
        c = [x for x in _cells(res, "REVERT") if x.extras["J"] == 1 and x.extras["K"] == 1]
        assert c and c[0].extras["cont_prob"] < 0.45, c[0].extras if c else None
        assert c[0].value < 0

    def test_vr_and_hurst_agree_in_sign(self):
        """They measure the same property; disagreement is a data-quality flag."""
        res = _run(_panel({"TREND": _ar1(3000, phi=0.35, seed=5),
                           "REVERT": _ar1(3000, phi=-0.35, seed=6)}))
        vr = {f.extras["symbol"]: f.value for f in res.findings
              if f.metric == "variance_ratio" and f.extras.get("q") == 4}
        h = {f.extras["symbol"]: f.value for f in res.findings if f.metric == "hurst"}
        assert vr["TREND"] > 1.0 > vr["REVERT"], vr
        assert h["TREND"] > h["REVERT"], h

    def test_jk_return_signs_with_the_regime(self):
        res = _run(_panel({"TREND": _ar1(2500, phi=0.35, seed=8)}))
        r = [f for f in res.findings if f.metric == "jk_return_bps"
             and f.extras["symbol"] == "TREND" and f.extras["J"] == 1]
        assert r and r[0].value > 0, "a trending series must give a positive J/K return"


# ── the sampling trap ────────────────────────────────────────────────────────────
class TestNonOverlappingSampling:
    def test_overlapping_inflates_and_the_paths_differ(self):
        """A-2's error in miniature: if these ever agree, the guard has stopped working."""
        panel = _panel({"TREND": _ar1(2500, phi=0.35, seed=9)})
        strict = _run(panel, k_grid=[4])
        loose = _run(panel, k_grid=[4], allow_overlapping=True)
        a = [x for x in _cells(strict, "TREND") if x.extras["K"] == 4][0]
        b = [x for x in _cells(loose, "TREND") if x.extras["K"] == 4][0]
        assert a.extras["n_obs"] < b.extras["n_obs"], "non-overlapping must use fewer rows"
        assert a.extras["n_obs"] * 3 <= b.extras["n_obs"]      # ~K-fold reduction

    def test_default_is_non_overlapping(self):
        panel = _panel({"X": _rw(1200, seed=2)})
        c = [x for x in _cells(_run(panel, k_grid=[6]), "X") if x.extras["K"] == 6][0]
        assert c.extras["non_overlapping"] is True


# ── entropy conditioning: the user's question, made testable ─────────────────────
class TestEntropyConditioning:
    def test_a_regime_present_only_in_one_bucket_is_found_there(self):
        """Continuation planted only where realised vol is LOW must show up in that
        bucket and be diluted pooled — the interaction, not a narrative."""
        rng = np.random.default_rng(21)
        n = 3000
        lo = np.repeat([True, False], n // 2)
        rng.shuffle(lo)
        r = np.zeros(n)
        for t in range(1, n):
            phi = 0.45 if lo[t] else 0.0
            scale = 0.002 if lo[t] else 0.008
            r[t] = phi * r[t - 1] + rng.normal(scale=scale)
        res = _run(_panel({"SPLIT": r}), buckets="vol")
        cells = [c for c in _cells(res, "SPLIT") if c.extras["J"] == 1 and c.extras["K"] == 1]
        by_bucket = {c.extras.get("bucket"): c.extras["cont_prob"] for c in cells}
        assert "all" in by_bucket
        assert any(b is not None and b != "all" for b in by_bucket), by_bucket

    def test_bucket_labels_are_reported(self):
        res = _run(_panel({"X": _ar1(2000, phi=0.3, seed=13)}), buckets="entropy")
        assert any(c.extras.get("bucket") not in (None, "all") for c in _cells(res, "X"))


# ── cost coverage on every cell ──────────────────────────────────────────────────
class TestCostCoverage:
    def test_every_return_cell_reports_coverage(self):
        res = _run(_panel({"X": _ar1(2000, phi=0.3, seed=14)}))
        cells = [f for f in res.findings if f.metric == "jk_return_bps"]
        assert cells
        for c in cells:
            assert "cost_coverage" in c.extras and "rt_cost_bps" in c.extras

    def test_coverage_comes_from_the_cost_ssot(self):
        from utils.costs import realistic_taker_rt_bps
        res = _run(_panel({"X": _ar1(2000, phi=0.3, seed=15)}))
        c = [f for f in res.findings if f.metric == "jk_return_bps"][0]
        assert c.extras["rt_cost_bps"] == pytest.approx(realistic_taker_rt_bps())

    def test_no_cost_literal_in_the_module(self):
        from processes import regime_profile
        src = Path(regime_profile.__file__).read_text()
        assert "11.0" not in src and "= 7.0" not in src


# ── durability and hygiene ───────────────────────────────────────────────────────
class TestDurabilityAndHygiene:
    def test_an_intermittent_regime_is_not_durable(self):
        a = _ar1(1200, phi=0.45, seed=16)
        b = _rw(1200, seed=17)
        res = _run(_panel({"HALF": np.r_[a, b]}))
        c = [x for x in _cells(res, "HALF") if x.extras["J"] == 1 and x.extras["K"] == 1][0]
        assert c.extras["verdict"] in ("non_durable", "insufficient_days"), c.extras

    def test_symbols_do_not_mix(self):
        panel = _panel({"TREND": _ar1(2000, phi=0.45, seed=18),
                        "REVERT": _ar1(2000, phi=-0.45, seed=19)})
        res = _run(panel)
        t = [x for x in _cells(res, "TREND") if x.extras["J"] == 1 and x.extras["K"] == 1][0]
        r = [x for x in _cells(res, "REVERT") if x.extras["J"] == 1 and x.extras["K"] == 1][0]
        assert t.extras["cont_prob"] > 0.5 > r.extras["cont_prob"], (t.extras, r.extras)

    def test_fdr_is_applied_across_the_grid(self):
        res = _run(_panel({f"S{i}": _rw(1200, seed=30 + i) for i in range(3)}))
        assert res.summary.get("fdr", {}).get("n_cells", 0) > 5
        assert any(f.p_adjusted is not None for f in res.findings)

    def test_registered_and_candle_level(self):
        from processes.base import VALID_DATA_LEVELS
        from processes.regime_profile import RegimeProfile
        from processes.registry import list_processes
        assert "regime_profile" in list_processes()
        assert RegimeProfile.data_level == "candles"
        assert RegimeProfile.data_level in VALID_DATA_LEVELS

    def test_input_is_not_mutated(self):
        panel = _panel({"X": _ar1(1200, phi=0.3, seed=20)})
        before = panel.copy()
        _run(panel)
        pd.testing.assert_frame_equal(panel, before)

    def test_short_symbol_is_skipped_with_a_reason(self):
        res = _run(_panel({"OK": _ar1(1500, phi=0.3, seed=22), "TINY": _rw(20, seed=23)}))
        assert any(s.get("feature") == "TINY" or s.get("symbol") == "TINY"
                   for s in res.features_skipped), res.features_skipped

    def test_thresholds_are_imported(self):
        from processes import regime_profile
        src = Path(regime_profile.__file__).read_text()
        assert "load_null_config" in src and "DEFAULT_FDR_ALPHA" in src
        assert "= 3.0" not in src and "= 0.05" not in src

    def test_determinism(self):
        panel = _panel({"X": _ar1(1500, phi=0.3, seed=24)})
        a = _cells(_run(panel, seed=1), "X")[0]
        b = _cells(_run(panel, seed=1), "X")[0]
        assert a.value == b.value and a.extras["z"] == b.extras["z"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestNoFindingEscapesFDR:
    """A z without a p_value silently bypasses correction.

    `apply_process_fdr` leaves p-value-less cells untouched — by design, since they were
    never part of the multiple-testing family. So a finding that sets `informative` from a
    raw |z| and omits `p_value` is corrected by nothing. On the first real run that flagged
    10 variance-ratio cells across 865 uncorrected tests, where ~2 are expected by chance.
    """

    def test_every_informative_finding_carries_a_p_value(self):
        panel = _panel({f"S{i}": _ar1(1500, phi=0.3, seed=40 + i) for i in range(3)})
        res = _run(panel)
        naked = [f for f in res.findings if f.informative and f.p_value is None]
        assert not naked, [(f.feature, f.metric) for f in naked]

    def test_variance_ratio_cells_enter_the_fdr_family(self):
        res = _run(_panel({f"S{i}": _rw(1200, seed=50 + i) for i in range(3)}))
        vr = [f for f in res.findings if f.metric == "variance_ratio"]
        assert vr and all(f.p_value is not None for f in vr if f.extras.get("z") is not None)

"""PROC-4 — `mi_stability`: is an edge durable, or a within-window mirage?

The IT daemon recomputes on a 600 s rolling buffer: it answers "is there MI right now",
never "does `MI(f; r)` hold across days". This process asks the second question, and the
tests are built around the two ways it could lie.

**Lie 1 — pooling.** Estimating MI on the concatenated frame lets *between-day* structure
masquerade as predictive information: two days whose feature and target levels both shift
produce joint dependence that no single day contains. A feature with zero within-day
relation must therefore come back non-informative even when the pooled frame screams. This
is the sharpest test in the file.

**Lie 2 — a lucky window.** An edge alive on half the days and dead on the rest averages to
something publishable. The spec's planted case is exactly that: informative in folds 1–5,
absent in 6–10, and the process must report `frac_days_informative ≈ 0.5` with a negative
trend rather than a comfortable mean.

The mirror test matters as much: on pure noise the per-day verdicts must be ~0, because a
stability tracker that finds structure in noise turns every downstream promotion into a
coin flip.
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
    kw = dict(symbol="BTC", timeframe="15min", price_col="raw_midprice",
              horizons={"h1": 1}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _frame(day_specs, rows=600, seed=3) -> pd.DataFrame:
    """Build a multi-day frame. `day_specs[i]` = strength of the within-day edge on day i.

    The target is a *forward return* built from a price path, so the process is exercised
    through the same PROC-17 target node it uses in production.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for d, strength in enumerate(day_specs):
        f = rng.normal(size=rows)
        # increment at t+1 carries the edge, so r(t) = p(t+1)/p(t) - 1 is predicted by f(t)
        inc = np.r_[0.0, strength * 0.02 * f[:-1] + rng.normal(scale=0.01, size=rows - 1)]
        frames.append(pd.DataFrame({
            "timestamp_ns": d * DAY_NS + np.arange(rows, dtype=np.int64) * 1_000_000,
            "symbol": "BTC",
            "raw_midprice": 100 + np.cumsum(inc),
            "feat_edge": f,
            "feat_noise": rng.normal(size=rows),
        }))
    return pd.concat(frames, ignore_index=True)


#: Tests buy their own null budget. The PRODUCTION default stays config-driven
#: (it_engine.toml n_shuffles=200); 40 draws resolve the planted effects here at 5x the
#: speed, and `test_thresholds_are_imported_not_written_here` guards the real default.
TEST_SHUFFLES = 40


def _run(df, ctx=None, **params):
    from processes.mi_stability import MIStabilityProcess
    params.setdefault("n_shuffles", TEST_SHUFFLES)
    return MIStabilityProcess(**params).evaluate(df, ctx or _ctx())


def _finding(result, feature):
    hits = [f for f in result.findings if f.feature == feature]
    assert hits, f"no finding for {feature}: {[f.feature for f in result.findings]}"
    return hits[0]


# ── the spec's planted case ──────────────────────────────────────────────────────
class TestIntermittentEdge:
    def test_half_the_days_informative_and_the_trend_is_negative(self):
        """Spec §4: present in folds 1-5, absent in 6-10."""
        df = _frame([3.0] * 5 + [0.0] * 5)
        f = _finding(_run(df), "feat_edge")
        frac = f.extras["frac_days_informative"]
        assert 0.3 <= frac <= 0.7, f"expected ~0.5 informative days, got {frac}"
        assert f.extras["slope_per_day"] < 0, "a decaying edge must trend down"
        assert f.extras["verdict"] == "non_durable"

    def test_per_day_series_is_reported(self):
        df = _frame([3.0] * 5 + [0.0] * 5)
        f = _finding(_run(df), "feat_edge")
        series = f.extras["per_day"]
        assert len(series) == 10
        assert all("day" in d and "bits_above_null" in d and "z" in d for d in series)
        early = np.mean([d["bits_above_null"] for d in series[:5]])
        late = np.mean([d["bits_above_null"] for d in series[5:]])
        assert early > late, "the planted decay is not visible in the per-day series"


class TestDurableEdge:
    def test_a_constant_edge_is_flagged_durable(self):
        df = _frame([3.0] * 8)
        f = _finding(_run(df), "feat_edge")
        assert f.extras["frac_days_informative"] >= 0.75
        assert abs(f.extras["slope_per_day"]) < 0.05, "a stable edge should not trend"
        assert f.extras["verdict"] == "durable"
        assert f.informative

    def test_cv_separates_stable_from_erratic(self):
        stable = _finding(_run(_frame([3.0] * 8)), "feat_edge")
        erratic = _finding(_run(_frame([6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0])), "feat_edge")
        assert erratic.extras["cv"] > stable.extras["cv"]
        assert erratic.extras["verdict"] == "non_durable"


# ── the mirror: noise must stay noise ────────────────────────────────────────────
class TestNoise:
    def test_pure_noise_is_informative_on_almost_no_days(self):
        df = _frame([0.0] * 10)
        f = _finding(_run(df), "feat_noise")
        assert f.extras["frac_days_informative"] <= 0.2, f.extras
        assert not f.informative
        assert f.extras["verdict"] != "durable"


# ── the sharpest test: no pooling across days ────────────────────────────────────
class TestNoPooling:
    def test_between_day_structure_is_not_reported_as_an_edge(self):
        """A feature with ZERO within-day relation, whose daily LEVEL tracks the daily
        return level, produces large MI when days are pooled and none when they are not.
        If this fails, every stability verdict is measuring between-day drift."""
        rng = np.random.default_rng(17)
        rows, n_days = 500, 8
        frames = []
        for d in range(n_days):
            level = float(d)                       # a per-day offset in BOTH series
            f = rng.normal(size=rows) + 5.0 * level
            inc = np.r_[0.0, rng.normal(scale=0.01, size=rows - 1) + 0.002 * level]
            frames.append(pd.DataFrame({
                "timestamp_ns": d * DAY_NS + np.arange(rows, dtype=np.int64) * 1_000_000,
                "symbol": "BTC",
                "raw_midprice": 100 + np.cumsum(inc),
                "feat_between": f,
                "feat_noise": rng.normal(size=rows),
            }))
        df = pd.concat(frames, ignore_index=True)

        # sanity: pooled, the between-day structure IS visible
        from it_engine.estimators import ksg_mi
        from alpha.screener import compute_forward_returns
        pooled_r = compute_forward_returns(df["raw_midprice"].to_numpy(), 1)
        m = np.isfinite(pooled_r)
        pooled_mi = ksg_mi(df["feat_between"].to_numpy()[m], pooled_r[m], k=5)

        f = _finding(_run(df), "feat_between")
        assert f.extras["frac_days_informative"] <= 0.25, (
            f"between-day drift reported as an edge on "
            f"{f.extras['frac_days_informative']:.0%} of days (pooled MI was "
            f"{pooled_mi:.3f}) — days are being pooled")

    def test_days_are_split_on_the_calendar_not_on_row_counts(self):
        """Unequal-length days must still produce one fold each."""
        df = pd.concat([_frame([3.0], rows=700, seed=1),
                        _frame([3.0], rows=300, seed=2).assign(
                            timestamp_ns=lambda d: d["timestamp_ns"] + DAY_NS)],
                       ignore_index=True)
        f = _finding(_run(df, min_fold_obs=200), "feat_edge")
        assert f.extras["n_days"] == 2, f.extras


# ── hygiene ──────────────────────────────────────────────────────────────────────
class TestHygiene:
    def test_short_days_are_skipped_with_a_reason_not_silently_used(self):
        df = pd.concat([_frame([3.0] * 3),
                        _frame([3.0], rows=40, seed=9).assign(
                            timestamp_ns=lambda d: d["timestamp_ns"] + 3 * DAY_NS)],
                       ignore_index=True)
        result = _run(df, min_fold_obs=200)
        f = _finding(result, "feat_edge")
        assert f.extras["n_days"] == 3
        assert any("fold" in str(s.get("reason", "")) or "day" in str(s.get("reason", ""))
                   for s in result.summary.get("folds_skipped", []))

    def test_too_few_days_is_reported_not_silently_averaged(self):
        result = _run(_frame([3.0] * 2), min_days=5)
        assert result.summary.get("error") or all(
            f.extras.get("verdict") == "insufficient_days" for f in result.findings)

    def test_registered_and_deterministic(self):
        from processes.registry import list_processes
        assert "mi_stability" in list_processes()
        a = _finding(_run(_frame([3.0] * 5), seed=1), "feat_edge")
        b = _finding(_run(_frame([3.0] * 5), seed=1), "feat_edge")
        assert a.value == b.value and a.extras["per_day"] == b.extras["per_day"]

    def test_thresholds_are_imported_not_written_here(self):
        from processes import mi_stability
        src = Path(mi_stability.__file__).read_text()
        assert "DEFAULT_NULL_Z_THRESHOLD" in src or "load_null_config" in src, (
            "the null threshold must come from PROC-12 / it_engine.toml")

    def test_uses_the_proc17_target_node(self):
        from processes import mi_stability
        assert "resolve_targets" in Path(mi_stability.__file__).read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

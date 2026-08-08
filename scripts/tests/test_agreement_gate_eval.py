"""A-1 — `agreement_gate_eval`: does conditioning on agreement actually add anything?

§5 makes two claims about the hierarchical combiner. A-2 killed the first (the composite
scores 0.06–0.10 against `trend_ema_short` alone at 0.20). This process tests the second,
which is the more interesting one: *"L2 conditional-on-agreement IC exceeds unconditional —
the first architecture structurally addressing §2."* It is the only structure in the record
claiming to attack the adverse-selection collapse head-on.

**The trap is selection, not estimation.** Splitting any sample on a condition and reporting
the better half is guaranteed to produce a lift; with a fast signal and a slow one, the
agreement subset is smaller, differently distributed, and its IC has a wider sampling
distribution, so the *maximum* of (agree, disagree) beats the pooled figure by construction.
A test that only checks "agreement IC > unconditional IC" would pass on pure noise.

So the null here permutes the **gate**, not the outcome: it reshuffles which observations
count as agreeing while holding the fast signal, the target and the subset SIZE fixed. That
asks the only question worth asking — *is this particular partition better than an arbitrary
partition of the same shape?* The decisive test plants two independent random signals and
asserts the process reports no lift.
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
              horizons={"h6": 6}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _frame(n=3000, n_days=6, lift=0.0, base=0.35, seed=1) -> pd.DataFrame:
    """Fast + slow signals and a forward return.

    `base` is the fast signal's edge when the two DISagree; `lift` is what agreement adds.
    lift=0 means agreement carries no information — the null case that must report nothing.
    """
    rng = np.random.default_rng(seed)
    fast = rng.normal(size=n)
    slow = rng.normal(size=n)
    agree = np.sign(fast) == np.sign(slow)
    strength = np.where(agree, base + lift, base)
    inc = np.r_[0.0, strength[:-1] * 0.02 * fast[:-1] + rng.normal(scale=0.01, size=n - 1)]
    per = n // n_days
    return pd.DataFrame({
        "timestamp_ns": (np.arange(n) // per) * DAY_NS
                        + (np.arange(n) % per) * 300_000_000_000,
        "symbol": "BTC",
        "raw_midprice": 100 * np.exp(np.cumsum(inc)),
        "fast_sig": fast,
        "slow_sig": slow,
    })


def _run(df, ctx=None, **params):
    from processes.agreement_gate_eval import AgreementGateEval
    params.setdefault("fast", "fast_sig")
    params.setdefault("slow", "slow_sig")
    params.setdefault("n_shuffles", 120)
    return AgreementGateEval(**params).evaluate(df, ctx or _ctx())


def _cell(res, metric="agreement_ic_lift"):
    hits = [f for f in res.findings if f.metric == metric]
    assert hits, f"no {metric} finding: {[f.metric for f in res.findings]}"
    return hits[0]


# ── the decisive test: an arbitrary partition must not look like a gate ──────────
class TestSelectionIsNotAnEdge:
    def test_two_independent_signals_report_no_lift(self):
        """Agreement between independent signals partitions the sample but carries no
        information. If this reports a lift, every §5-style claim is unfalsifiable."""
        res = _run(_frame(lift=0.0))
        f = _cell(res)
        assert not f.informative, f.extras
        assert abs(f.extras["z"]) < 3.0, f.extras

    def test_the_raw_lift_can_be_positive_while_the_calibrated_one_is_not(self):
        """The point of the null: a naive difference-of-ICs is noisy and often positive.
        The process must not promote that."""
        res = _run(_frame(lift=0.0, seed=7))
        f = _cell(res)
        assert "raw_lift" in f.extras and "z" in f.extras
        assert not f.informative

    def test_shuffling_preserves_the_agreement_subset_size(self):
        """The null must hold the partition SHAPE fixed — otherwise it compares against a
        differently-sized subset and measures sample size, not structure."""
        res = _run(_frame(lift=0.0))
        e = _cell(res).extras
        assert e["n_agree"] > 0 and e["n_disagree"] > 0
        assert e["null_preserves_subset_size"] is True


# ── it must still find a real gate ───────────────────────────────────────────────
class TestRecoversAPlantedGate:
    def test_a_genuine_lift_is_reported(self):
        res = _run(_frame(lift=0.9, base=0.1, seed=3))
        f = _cell(res)
        assert f.extras["raw_lift"] > 0
        assert f.extras["z"] >= 3.0, f.extras
        assert f.informative

    def test_a_negative_gate_is_reported_as_negative(self):
        """Agreement that HURTS must not be reported as an edge in either direction."""
        res = _run(_frame(lift=-0.25, base=0.35, seed=5))
        f = _cell(res)
        assert f.extras["raw_lift"] < 0
        assert not f.informative

    def test_conditional_and_unconditional_ic_are_both_reported(self):
        """§5's claim is a COMPARISON; hiding either half makes it uncheckable."""
        e = _cell(_run(_frame(lift=0.6))).extras
        for k in ("ic_agree", "ic_disagree", "ic_unconditional"):
            assert k in e and e[k] is not None


# ── day-durability, per §4.9 ─────────────────────────────────────────────────────
class TestDurability:
    def test_an_intermittent_gate_is_not_durable(self):
        a = _frame(n=1500, n_days=3, lift=0.9, base=0.1, seed=11)
        b = _frame(n=1500, n_days=3, lift=0.0, base=0.1, seed=12)
        b["timestamp_ns"] = b["timestamp_ns"] + 3 * DAY_NS
        f = _cell(_run(pd.concat([a, b], ignore_index=True)))
        assert 0.2 <= f.extras["frac_days_informative"] <= 0.8, f.extras
        assert f.extras["verdict"] == "non_durable"

    def test_a_constant_gate_is_durable(self):
        f = _cell(_run(_frame(n=3000, n_days=6, lift=0.9, base=0.1, seed=13)))
        assert f.extras["frac_days_informative"] >= 0.6
        assert f.extras["verdict"] == "durable"


# ── contract ─────────────────────────────────────────────────────────────────────
class TestContract:
    def test_registered(self):
        from processes.registry import list_processes
        assert "agreement_gate_eval" in list_processes()

    def test_input_is_not_mutated(self):
        df = _frame()
        before = df.copy()
        _run(df)
        pd.testing.assert_frame_equal(df, before)

    def test_missing_signal_column_is_reported(self):
        res = _run(_frame(), fast="nope")
        assert res.summary.get("error") and "nope" in res.summary["error"]

    def test_a_degenerate_partition_is_refused(self):
        """If one side is (nearly) empty there is no comparison to make."""
        df = _frame()
        df["slow_sig"] = df["fast_sig"].abs() + 1.0      # always agrees
        res = _run(df)
        assert res.summary.get("error") or not _cell(res).informative

    def test_thresholds_are_imported(self):
        from processes import agreement_gate_eval as m
        src = Path(m.__file__).read_text()
        assert "load_null_config" in src and "= 3.0" not in src

    def test_it_is_a_standing_eval(self):
        """The row asks for a standing eval, not a one-off script."""
        from processes.standing import STANDING_EVALS
        assert any("agreement" in k for k in STANDING_EVALS)

    def test_determinism(self):
        df = _frame()
        assert _cell(_run(df, seed=4)).value == _cell(_run(df, seed=4)).value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""XS-10 — the standing rotation tracker: making an 8-month wait self-executing.

§7.8's only actionable conclusion is *"this needs ~325 rebalances and we have 83"*. A
conclusion like that decays into nothing unless something re-runs it as the `XS-7` archive
grows, so this module turns the wait into a measured trajectory.

Three properties carry it:

  1. **The power arithmetic is exact.** `n_required = (t*/SR_period)^2` is the number the
     whole plan rests on — it is what says "8 months" rather than "2.5 years", and it moved
     by 4x when XS-9 doubled the Sharpe. An error here misprices the entire research
     schedule, so it is pinned against hand-computed values.

  2. **The trajectory is append-only.** Its value is the *sequence* — whether t is climbing
     as data accrues, or whether the Sharpe is decaying as the sample grows (the honest
     early-warning that §7.8's in-sample design choice was optimistic). Overwriting would
     leave a single number that always looks like the present.

  3. **Criteria are evaluated, never redefined.** The six pre-registered checks are imported
     from the XS-6 driver; a tracker that quietly relaxed them as data accrued would be
     the exact failure the pre-registration exists to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from xs.trajectory import (  # noqa: E402
    CRITERIA,
    append_trajectory,
    evaluate_criteria,
    power_status,
    read_trajectory,
)


# ── 1. the arithmetic the schedule rests on ──────────────────────────────

def test_power_status_is_hand_computable():
    """SR_ann 2.12 over 365 periods -> SR_period 0.111 -> n(t=2) = (2/0.111)^2 ~= 325."""
    p = power_status(sharpe_annual=2.12, n_periods=83, periods_per_year=365)
    assert p["sharpe_period"] == pytest.approx(2.12 / 365 ** 0.5, rel=1e-9)
    assert p["t_stat"] == pytest.approx(2.12 / 365 ** 0.5 * 83 ** 0.5, rel=1e-9)
    assert p["n_required_t2"] == pytest.approx((2.0 / (2.12 / 365 ** 0.5)) ** 2, rel=1e-9)
    assert 320 < p["n_required_t2"] < 330
    assert p["n_remaining"] == pytest.approx(p["n_required_t2"] - 83, rel=1e-9)


def test_doubling_the_sharpe_quarters_the_requirement():
    """n ∝ 1/SR² — the relation that turned 2.55 years into 0.89."""
    lo = power_status(sharpe_annual=1.06, n_periods=83)
    hi = power_status(sharpe_annual=2.12, n_periods=83)
    assert lo["n_required_t2"] / hi["n_required_t2"] == pytest.approx(4.0, rel=1e-6)


def test_a_resolved_series_reports_no_remaining_need():
    p = power_status(sharpe_annual=2.12, n_periods=5000)
    assert p["n_remaining"] == 0
    assert p["resolved"] is True


def test_zero_or_negative_sharpe_is_never_resolvable():
    """A dead strategy must not report a finite path to significance."""
    for sr in (0.0, -1.5):
        p = power_status(sharpe_annual=sr, n_periods=83)
        assert p["resolved"] is False
        assert p["n_required_t2"] == float("inf")


# ── 2. the sequence is the product ───────────────────────────────────────

def test_trajectory_appends_and_preserves_order(tmp_path):
    f = tmp_path / "traj.jsonl"
    for n, sr in ((83, 2.12), (120, 1.90), (200, 1.75)):
        append_trajectory(f, {"n_periods": n, "sharpe_annual": sr})
    rows = read_trajectory(f)
    assert [r["n_periods"] for r in rows] == [83, 120, 200]
    assert [r["sharpe_annual"] for r in rows] == [2.12, 1.90, 1.75], (
        "the point of the sequence is seeing the Sharpe decay as the sample grows"
    )


def test_every_row_is_stamped(tmp_path):
    f = tmp_path / "traj.jsonl"
    append_trajectory(f, {"n_periods": 83, "sharpe_annual": 2.12})
    row = read_trajectory(f)[0]
    assert "ts" in row and "git_sha" in row


def test_reading_a_missing_trajectory_is_empty_not_an_error(tmp_path):
    assert read_trajectory(tmp_path / "nope.jsonl") == []


# ── 3. criteria are imported, not redefined ──────────────────────────────

def test_criteria_match_the_preregistered_set():
    assert set(CRITERIA) == {"a", "b", "c", "d", "e", "f"}
    assert "0.5" in CRITERIA["a"] and "0.55" in CRITERIA["c"] and "0.30" in CRITERIA["d"]


def test_evaluate_criteria_reproduces_the_xs9_verdict():
    """§7.8's measured numbers must yield 4 of 6 — a,c,d,f pass; b,e fail."""
    passed, failed = evaluate_criteria({
        "sharpe_net": 2.12, "dsr_p": 0.31, "positive_share": 0.55,
        "max_day_share": 0.30, "oos_is_ratio": 0.447, "sign_stable_2x": True,
    })
    assert sorted(passed) == ["a", "c", "d", "f"]
    assert sorted(failed) == ["b", "e"]


def test_a_full_pass_is_reported_as_such():
    passed, failed = evaluate_criteria({
        "sharpe_net": 2.5, "dsr_p": 0.01, "positive_share": 0.6,
        "max_day_share": 0.2, "oos_is_ratio": 0.9, "sign_stable_2x": True,
    })
    assert failed == [] and len(passed) == 6


def test_a_missing_metric_fails_its_criterion_rather_than_passing_silently():
    """An absent number is not a satisfied criterion."""
    passed, failed = evaluate_criteria({"sharpe_net": 2.5})
    assert "a" in passed
    assert {"b", "c", "d", "e", "f"} <= set(failed)

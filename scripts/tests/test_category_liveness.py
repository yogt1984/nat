"""Planted tests for the category-liveness check (BUG-6 remediation).

Why this check exists, given `NaN Ratio` already flags dead columns:

  On 2026-08-08 `nat data validate` reported "90 columns exceed 1% NaN threshold",
  49 of them at exactly 100%. On 2026-07-15 — a day when the `regime_` category was
  *alive* — it reported the same shape. The line is identical whether or not a
  23-feature category has just died, so the regression that BUG-6 records went
  unnoticed for 13 days while the validator was, technically, failing about it.

  The unit that carries the signal is the CATEGORY, not the column. A category
  going wholly dead means a subsystem stopped computing; scattered NaN means the
  data is noisy. Collapsing 90 columns to "regime is wholly dead" is the whole
  point — it turns an unreadable standing FAIL into one named item.

  Chronically-unwired categories must be DECLARED, not tolerated by threshold:
  `hm_` (heatmap) has never produced a finite value in 76 days of files, so it is
  expected-dead and stays quiet. Anything else wholly dead is a hard failure.
  A declaration is auditable; a threshold that happens to swallow it is not.

Red-first: `check_category_liveness`, `ValidationConfig.expected_dead_categories`
and the "Category Liveness" hard check do not exist yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import validate_data as vd


START_NS = 1_781_532_000_000_000_000
STEP_NS = 100_000_000


def _frame(n: int = 1200, *, dead: tuple[str, ...] = (), partial: dict | None = None
           ) -> pd.DataFrame:
    """A feature frame with two live categories plus whatever is asked for.

    Categories here mirror the real prefixes: `raw_`, `imbalance_`, `regime_`, `hm_`.
    """
    ts = START_NS + np.arange(n, dtype=np.int64) * STEP_NS
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp_ns": ts,
        "symbol": "BTC",
        "sequence_id": np.arange(n, dtype=np.int64),
        # always-live categories
        "raw_midprice": 50_000 + rng.normal(0, 5, n),
        "raw_spread_bps": np.abs(rng.normal(1, 0.1, n)),
        "imbalance_qty_l1": rng.uniform(-1, 1, n),
        "imbalance_qty_l5": rng.uniform(-1, 1, n),
        # categories under test — live unless named in `dead`
        "regime_divergence_1h": rng.normal(0, 1, n),
        "regime_absorption_1h": rng.normal(0, 1, n),
        "hm_depth_p50": rng.normal(0, 1, n),
    })
    for prefix in dead:
        for c in [c for c in df.columns if c.startswith(prefix)]:
            df[c] = np.nan
    for col, frac in (partial or {}).items():
        idx = df.index[: int(len(df) * frac)]
        df.loc[idx, col] = np.nan
    return df


# --------------------------------------------------------------------------- #
# The check exists and is wired as a hard check
# --------------------------------------------------------------------------- #

def test_check_is_registered_as_hard():
    """A dead subsystem makes the data unusable for anything conditioned on it."""
    assert "Category Liveness" in vd.HARD_CHECKS


def test_expected_dead_is_declared_not_inferred():
    """`hm_` is known-unwired (0 finite values in 76 days) and must be declared."""
    cfg = vd.ValidationConfig()
    assert "hm" in cfg.expected_dead_categories


# --------------------------------------------------------------------------- #
# The decisive pair: the same frame, one category flipped
# --------------------------------------------------------------------------- #

def test_all_live_passes():
    res = vd.check_category_liveness(_frame(), vd.ValidationConfig())
    assert res.passed, res.message


def test_declared_dead_category_alone_still_passes():
    """`hm_` wholly dead is the standing state on every real file — it must be quiet."""
    res = vd.check_category_liveness(_frame(dead=("hm_",)), vd.ValidationConfig())
    assert res.passed, res.message


def test_undeclared_dead_category_fails_and_is_named():
    """The BUG-6 case: regime goes wholly NaN and the check must say so by name."""
    res = vd.check_category_liveness(_frame(dead=("hm_", "regime_")), vd.ValidationConfig())
    assert not res.passed
    assert "regime" in res.message
    # The declared-dead category must NOT be reported as the problem.
    assert "hm" not in res.message.replace("hm_depth", "")
    assert res.details["dead_categories"] == ["regime"]


# --------------------------------------------------------------------------- #
# The distinction the check exists to draw
# --------------------------------------------------------------------------- #

def test_partially_dead_category_does_not_fire():
    """Scattered NaN is noise, not a dead subsystem — that is `NaN Ratio`'s job.

    Without this the check would duplicate NaN Ratio and inherit its unreadability.
    """
    df = _frame(dead=("hm_",))
    df["regime_divergence_1h"] = np.nan          # one of two regime columns dead
    res = vd.check_category_liveness(df, vd.ValidationConfig())
    assert res.passed, res.message


def test_reports_per_category_counts_even_when_passing():
    """The counts are the point: 8/31 dead vs 31/31 is the difference BUG-6 turned on."""
    res = vd.check_category_liveness(_frame(dead=("hm_",)), vd.ValidationConfig())
    assert res.details["categories"]["regime"] == {"dead": 0, "total": 2}
    assert res.details["categories"]["hm"] == {"dead": 1, "total": 1}


# --------------------------------------------------------------------------- #
# Degenerate inputs are refused, not scored
# --------------------------------------------------------------------------- #

def test_empty_frame_is_refused():
    res = vd.check_category_liveness(pd.DataFrame(), vd.ValidationConfig())
    assert not res.passed
    assert "no data" in res.message.lower()


def test_metadata_columns_are_not_categories():
    """`symbol`/`sequence_id` must never be scored as feature categories."""
    res = vd.check_category_liveness(_frame(dead=("hm_",)), vd.ValidationConfig())
    for meta in ("timestamp", "symbol", "sequence"):
        assert meta not in res.details["categories"]


def test_a_category_of_all_nan_from_the_first_row_is_still_dead():
    """No warmup grace: a category NaN for the whole file never started."""
    df = _frame(n=100, dead=("hm_", "regime_"))
    res = vd.check_category_liveness(df, vd.ValidationConfig())
    assert not res.passed


# --------------------------------------------------------------------------- #
# Integration: the verdict actually changes
# --------------------------------------------------------------------------- #

def test_verdict_is_fail_when_a_category_dies():
    """The check must move the verdict, not merely print — FAIL exits nonzero."""
    from datetime import datetime

    report = vd.ValidationReport(
        data_dir="/tmp", timestamp=datetime(2026, 8, 9), total_files=1,
        total_rows=1200, symbols=["BTC"], date_range=("2026-08-09", "2026-08-09"),
        checks=[
            vd.CheckResult(name="Category Liveness", passed=False,
                           message="1 feature category(ies) wholly NaN: regime (23 features)"),
        ],
    )
    assert report.verdict == "FAIL"

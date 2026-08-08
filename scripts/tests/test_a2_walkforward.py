"""A-2 — combiner revalidation harness. The test that matters is the leakage test.

§5's composite IC (BTC .178 / ETH .248 / SOL .359) is the last unrefuted capital-relevant
claim in the record, and its own source flags the tell: **monotonically rising fold ICs**.
The provenance says why that is expected — `models/hierarchical_combiner/weights_BTC.json`
carries `training_date 2026-06-11`, while the OOS window was 2026-06-08→10. The weights were
fitted *after* the period they were scored on, so the "out-of-sample" evaluation was scored
with parameters that had seen it.

So this harness has one job: fit weights **strictly before** each evaluation fold, and prove
it cannot do otherwise. The decisive test plants a feature that predicts only inside the
training window and asserts the walk-forward path reports ~zero out-of-sample IC while the
in-sample path reports a large one. If that ever fails, every number the harness produces is
worthless in the same way the original was.
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

from alpha.walkforward_ic import (  # noqa: E402
    fit_ic_weights, walk_forward_ic, fold_bounds,
)


def _frame(n=2400, seed=0, signal_cols=("f_good",), noise_cols=("f_noise",),
           strength=0.5, switch_at=None) -> pd.DataFrame:
    """Bars with a forward return; `switch_at` kills the signal after that row."""
    rng = np.random.default_rng(seed)
    cols = {}
    for c in list(signal_cols) + list(noise_cols):
        cols[c] = rng.normal(size=n)
    fwd = rng.normal(scale=1.0, size=n)
    for c in signal_cols:
        eff = np.full(n, strength)
        if switch_at is not None:
            eff[switch_at:] = 0.0
        fwd = fwd + eff * cols[c]
    return pd.DataFrame({**cols, "fwd_ret": fwd})


# ── the decisive test ────────────────────────────────────────────────────────────
class TestNoLeakage:
    def test_an_edge_that_dies_before_the_holdout_is_not_reported(self):
        """A feature predictive only in the training window must produce ~0 OOS IC.

        This is the §5 failure mode in miniature: weights fitted on a period that
        includes (or postdates) the scoring window will always look good.
        """
        df = _frame(n=2400, switch_at=1200, strength=1.2)
        res = walk_forward_ic(df, ["f_good", "f_noise"], "fwd_ret", n_folds=6)
        late = [f for f in res["folds"] if f["start"] >= 1200]
        assert late, "test needs folds after the switch"
        assert abs(np.mean([f["ic"] for f in late])) < 0.06, (
            f"IC {np.mean([f['ic'] for f in late]):.3f} on folds where the edge is gone — "
            "weights are seeing their own evaluation window")

    def test_in_sample_fitting_does_report_the_dead_edge(self):
        """The control: the same data scored the WRONG way must look good, or the test
        above proves nothing about the harness."""
        df = _frame(n=2400, switch_at=1200, strength=1.2)
        w = fit_ic_weights(df, ["f_good", "f_noise"], "fwd_ret")      # whole sample
        composite = sum(w[c] * df[c] for c in w)
        from scipy.stats import spearmanr
        ic_all = spearmanr(composite, df["fwd_ret"]).statistic
        assert abs(ic_all) > 0.15, f"in-sample IC {ic_all:.3f} — the plant is too weak"

    def test_fold_weights_differ_across_folds(self):
        """If every fold used the same weights, nothing is being re-fitted."""
        res = walk_forward_ic(_frame(n=2400), ["f_good", "f_noise"], "fwd_ret", n_folds=6)
        seen = [tuple(round(v, 6) for v in f["weights"].values()) for f in res["folds"]]
        assert len(set(seen)) > 1, "weights identical across folds — no walk-forward"

    def test_a_fold_never_sees_its_own_rows(self):
        df = _frame(n=1200)
        res = walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=4, embargo=0)
        for f in res["folds"]:
            assert f["train_end"] <= f["start"], (
                f"train ends at {f['train_end']} but the fold starts at {f['start']}")

    def test_embargo_widens_the_gap(self):
        df = _frame(n=1200)
        a = walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=4, embargo=0)
        b = walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=4, embargo=50)
        assert all(f["start"] - f["train_end"] >= 50 for f in b["folds"])
        assert b["folds"][0]["train_end"] < a["folds"][0]["train_end"]


# ── it must still find a real edge ───────────────────────────────────────────────
class TestRecoversARealEdge:
    def test_a_persistent_edge_is_reported(self):
        res = walk_forward_ic(_frame(n=2400, strength=0.8), ["f_good", "f_noise"],
                              "fwd_ret", n_folds=6)
        assert res["pooled_ic"] > 0.15, res["pooled_ic"]
        assert res["positive_fold_share"] >= 0.8

    def test_pure_noise_gives_nothing(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"f_a": rng.normal(size=2000), "f_b": rng.normal(size=2000),
                           "fwd_ret": rng.normal(size=2000)})
        res = walk_forward_ic(df, ["f_a", "f_b"], "fwd_ret", n_folds=6)
        assert abs(res["pooled_ic"]) < 0.06, res["pooled_ic"]

    def test_sign_is_recovered(self):
        df = _frame(n=2400, strength=-0.8)
        res = walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=6)
        assert res["pooled_ic"] > 0.1, "an IC-weighted composite must absorb the sign"


# ── reporting contract ───────────────────────────────────────────────────────────
class TestReporting:
    def test_per_fold_series_is_returned_not_just_a_mean(self):
        """§4.9's lesson: the pooled mean is the wrong statistic; consistency is the test."""
        res = walk_forward_ic(_frame(), ["f_good"], "fwd_ret", n_folds=5)
        assert len(res["folds"]) == 5
        assert all(k in res for k in
                   ("pooled_ic", "positive_fold_share", "max_fold_share", "ic_std"))

    def test_monotonicity_of_fold_ics_is_flagged(self):
        """The §5 tell. A harness that cannot see it would repeat the mistake."""
        res = walk_forward_ic(_frame(n=2400), ["f_good"], "fwd_ret", n_folds=6)
        assert "fold_ic_trend" in res and "fold_ic_monotone" in res

    def test_fold_bounds_are_contiguous_and_cover_the_frame(self):
        b = fold_bounds(1000, n_folds=5, min_train=100)
        assert b[0][0] >= 100
        for (s0, e0), (s1, _) in zip(b, b[1:]):
            assert e0 == s1, "folds must tile without gaps or overlap"
        assert b[-1][1] == 1000

    def test_too_little_data_is_reported_not_averaged(self):
        res = walk_forward_ic(_frame(n=120), ["f_good"], "fwd_ret", n_folds=6,
                              min_train=200)
        assert res["error"] or res["n_folds_used"] == 0


class TestHygiene:
    def test_input_is_not_mutated(self):
        df = _frame()
        before = df.copy()
        walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=4)
        pd.testing.assert_frame_equal(df, before)

    def test_nan_rows_are_dropped_not_filled(self):
        df = _frame(n=1200)
        df.loc[df.index[100:150], "f_good"] = np.nan
        res = walk_forward_ic(df, ["f_good"], "fwd_ret", n_folds=4)
        assert res["n_obs"] < len(df)

    def test_missing_feature_is_reported(self):
        res = walk_forward_ic(_frame(), ["not_a_column"], "fwd_ret", n_folds=4)
        assert res["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

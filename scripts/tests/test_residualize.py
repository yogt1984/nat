"""PROC-15 — `residualize`: orthogonalization as a first-class transform.

`res_f(t) = f(t) − beta' Z(t)`, with **beta fit on the training prefix only**. The
process answers "what does f know that Z doesn't?" and, unlike the set-level CMI ranking
PROC-3 produces, it emits a *tradeable series* rather than a verdict.

Two properties carry the whole unit, and both are attacked here:

  1. **Orthogonality is achieved out of sample.** Fitting beta guarantees zero correlation
     on the fit segment by construction — that is arithmetic, not evidence. The test that
     matters measures `corr(res_f, Z)` on the HOLDOUT, which the fit never saw.
  2. **No lookahead.** If beta were fit on the full sample the residual would be
     contaminated by the future. The decisive test perturbs the holdout segment violently
     and asserts the prefix residuals do not move by even a float — the same pattern
     `pca_combo` uses for its own fit.

Everything else (self-residualization, degenerate/constant conditioners, NaN propagation,
the transform contract, determinism) is here to stop the unit from silently returning
something that merely looks like a residual.
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
    kw = dict(symbol="BTC", timeframe="15min", price_col="raw_midprice",
              horizons={"h": 1}, costs={})
    kw.update(over)
    return ProcessContext(**kw)


def _planted(n=2000, seed=5) -> pd.DataFrame:
    """f = unique_signal + market_mode. Residualizing vs the mode must leave the unique part."""
    rng = np.random.default_rng(seed)
    mode = rng.normal(size=n)                      # the common factor
    unique = rng.normal(size=n)                    # what f knows that the mode doesn't
    return pd.DataFrame({
        "raw_midprice": 100 + np.cumsum(rng.normal(scale=0.01, size=n)),
        "market_mode": mode,
        "feat_f": unique + 2.0 * mode,             # loaded on the mode
        "feat_g": rng.normal(size=n),              # unrelated
        "_unique_truth": unique,                   # bookkeeping for the assertions
    })


def _run(df, features=None, conditioning=("market_mode",), ctx=None, **params):
    from processes.residualize import ResidualizeProcess
    p = ResidualizeProcess(features=features, conditioning=list(conditioning), **params)
    return p.transform(df, ctx or _ctx())


# ── 1. orthogonality, measured where it counts ───────────────────────────────────
class TestOrthogonality:
    def test_residual_is_orthogonal_to_the_conditioner_on_the_holdout(self):
        df = _planted()
        derived, result = _run(df, features=["feat_f"])
        res = derived["res_feat_f"].to_numpy()
        z = df["market_mode"].to_numpy()
        cut = int(len(df) * 0.7)
        holdout = slice(cut, None)
        rho = abs(np.corrcoef(res[holdout], z[holdout])[0, 1])
        assert rho < 0.08, f"holdout corr(res, Z) = {rho:.3f} — not orthogonalized"

    def test_the_unique_component_survives(self):
        """Removing the mode must not remove what made the feature interesting."""
        df = _planted()
        derived, _ = _run(df, features=["feat_f"])
        res = derived["res_feat_f"].to_numpy()
        truth = df["_unique_truth"].to_numpy()
        cut = int(len(df) * 0.7)
        rho = abs(np.corrcoef(res[cut:], truth[cut:])[0, 1])
        assert rho > 0.9, f"unique signal destroyed (corr with truth {rho:.3f})"

    def test_multivariate_conditioning_orthogonalizes_against_all_of_it(self):
        df = _planted()
        rng = np.random.default_rng(2)
        df["mode2"] = rng.normal(size=len(df))
        df["feat_f"] = df["feat_f"] + 1.5 * df["mode2"]
        derived, _ = _run(df, features=["feat_f"], conditioning=("market_mode", "mode2"))
        res = derived["res_feat_f"].to_numpy()
        cut = int(len(df) * 0.7)
        for z_col in ("market_mode", "mode2"):
            rho = abs(np.corrcoef(res[cut:], df[z_col].to_numpy()[cut:])[0, 1])
            assert rho < 0.1, f"residual still loaded on {z_col} ({rho:.3f})"

    def test_an_unrelated_feature_is_left_essentially_alone(self):
        df = _planted()
        derived, _ = _run(df, features=["feat_g"])
        rho = abs(np.corrcoef(derived["res_feat_g"].to_numpy(), df["feat_g"].to_numpy())[0, 1])
        assert rho > 0.95, "residualizing against an unrelated Z should be near-identity"


# ── 2. no lookahead — the decisive property ──────────────────────────────────────
class TestNoLookahead:
    def test_holdout_perturbation_cannot_move_the_prefix_residuals(self):
        """If beta saw the future, corrupting the holdout would change prefix residuals."""
        df = _planted()
        base, _ = _run(df, features=["feat_f"])

        poisoned = df.copy()
        cut = int(len(df) * 0.7)
        rng = np.random.default_rng(99)
        poisoned.loc[poisoned.index[cut:], "feat_f"] = rng.normal(scale=50, size=len(df) - cut)
        poisoned.loc[poisoned.index[cut:], "market_mode"] = rng.normal(scale=50, size=len(df) - cut)
        after, _ = _run(poisoned, features=["feat_f"])

        np.testing.assert_allclose(
            base["res_feat_f"].to_numpy()[:cut], after["res_feat_f"].to_numpy()[:cut],
            rtol=0, atol=0,
            err_msg="prefix residuals moved when only the HOLDOUT changed — beta is "
                    "being fit on data the residual is not allowed to see")

    def test_reported_beta_is_the_prefix_beta(self):
        df = _planted()
        _, result = _run(df, features=["feat_f"])
        cut = int(len(df) * 0.7)
        f = df["feat_f"].to_numpy()[:cut]
        z = df["market_mode"].to_numpy()[:cut]
        expected = np.polyfit(z, f, 1)[0]
        got = result.summary["betas"]["feat_f"]["market_mode"]
        assert abs(got - expected) < 1e-6, f"beta {got} != prefix OLS {expected}"

    def test_fit_fraction_is_respected(self):
        df = _planted()
        a, ra = _run(df, features=["feat_f"], fit_frac=0.5)
        b, rb = _run(df, features=["feat_f"], fit_frac=0.9)
        assert ra.summary["n_fit_rows"] < rb.summary["n_fit_rows"]
        assert not np.allclose(a["res_feat_f"].to_numpy(), b["res_feat_f"].to_numpy())


# ── 3. degenerate inputs must not produce plausible-looking garbage ──────────────
class TestDegenerate:
    def test_self_residualization_is_dropped_but_recorded(self):
        """res(X | X) is identically zero — a degenerate series that would score as
        nonsense downstream. Dropping it is right; dropping it silently is not."""
        df = _planted()
        _, result = _run(df, features=["market_mode"], conditioning=("market_mode",))
        reasons = {s["feature"]: s["reason"] for s in result.features_skipped}
        assert "market_mode" in reasons and "conditioning" in reasons["market_mode"]
        assert "market_mode" not in result.features_tested

    def test_a_conditioner_is_never_residualized_against_itself_silently(self):
        """With features=None the conditioner must be excluded from the target set."""
        df = _planted()
        derived, result = _run(df, features=None)
        assert "res_market_mode" not in derived.columns
        assert "market_mode" not in result.features_tested

    def test_constant_conditioner_is_skipped_not_inverted(self):
        df = _planted()
        df["dead_mode"] = 1.0
        derived, result = _run(df, features=["feat_f"], conditioning=("dead_mode",))
        err = (result.summary.get("error") or "")
        assert err or result.summary.get("conditioning_skipped"), (
            "a constant conditioner has no invertible covariance — say so")

    def test_all_nan_feature_is_skipped_with_a_reason(self):
        df = _planted()
        df["feat_dead"] = np.nan
        _, result = _run(df, features=["feat_f", "feat_dead"])
        assert any(s["feature"] == "feat_dead" for s in result.features_skipped)

    def test_nan_rows_propagate_rather_than_impute(self):
        df = _planted()
        df.loc[df.index[10:20], "feat_f"] = np.nan
        df.loc[df.index[30:40], "market_mode"] = np.nan
        derived, _ = _run(df, features=["feat_f"])
        res = derived["res_feat_f"].to_numpy()
        assert np.isnan(res[10:20]).all() and np.isnan(res[30:40]).all()
        assert np.isfinite(res[100:200]).all()

    def test_missing_conditioning_column_is_an_error(self):
        df = _planted()
        _, result = _run(df, features=["feat_f"], conditioning=("not_a_column",))
        assert "not_a_column" in (result.summary.get("error") or "")


# ── 4. the transform contract ────────────────────────────────────────────────────
class TestContract:
    def test_registered(self):
        from processes.registry import list_processes
        assert "residualize" in list_processes()

    def test_index_preserved_and_input_untouched(self):
        df = _planted()
        before = df.copy()
        derived, _ = _run(df, features=["feat_f"])
        pd.testing.assert_index_equal(derived.index, df.index)
        pd.testing.assert_frame_equal(df, before)

    def test_output_naming_and_chainability(self):
        df = _planted()
        derived, result = _run(df, features=["feat_f", "feat_g"])
        assert set(derived.columns) == {"res_feat_f", "res_feat_g"}
        assert result.summary["derived_columns"] == ["res_feat_f", "res_feat_g"]

    def test_findings_report_the_orthogonality_achieved(self):
        df = _planted()
        _, result = _run(df, features=["feat_f"])
        f = [x for x in result.findings if x.feature == "res_feat_f"][0]
        assert f.metric == "holdout_abs_corr_z"
        assert 0.0 <= f.value < 0.1
        assert "r2_fit" in f.extras and "betas" in f.extras

    def test_price_column_is_never_a_target(self):
        df = _planted()
        _, result = _run(df, features=None)
        assert not any(c.startswith("raw_midprice") for c in result.features_tested)


class TestDeterminism:
    def test_same_inputs_same_residuals(self):
        df = _planted()
        a, _ = _run(df, features=["feat_f"])
        b, _ = _run(df, features=["feat_f"])
        np.testing.assert_array_equal(a["res_feat_f"].to_numpy(),
                                      b["res_feat_f"].to_numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

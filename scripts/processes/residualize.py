"""PROC-15: `residualize` — orthogonalization as a first-class transform.

    res_f(t) = f(t) − beta' Z(t),   beta fit on the TRAINING PREFIX only

PROC-3 answers "are these two features telling me the same thing?" with a number.
This answers the follow-up — "then give me only the part that is new" — and answers it with
a *series*, so the pure-innovation component is tradeable, chainable
(`--score-with ic_horizon`) and compilable by PROC-1 rather than merely reportable.

Two design points carry the unit:

**The fit is prefix-only.** Beta is estimated on the first `fit_frac` of the rows and applied
to the whole series, exactly as `pca_combo` fits its scaler and loadings. A full-sample fit
would make the residual a function of its own future — the residual would look beautifully
orthogonal and be unusable. `tests/test_residualize.py` perturbs the holdout violently and
asserts the prefix residuals do not move by a single float.

**Orthogonality is reported where it was not fitted.** On the fit segment `corr(res, Z) = 0`
holds by construction of OLS; that is arithmetic, not evidence. The finding this process
emits is therefore the **holdout** `|corr(res, Z)|` — the number that can actually fail.

Degenerate inputs are refused rather than papered over: a constant or collinear conditioning
set has no invertible covariance, and inverting it anyway would produce enormous betas and a
residual that is mostly numerical noise. NaNs propagate (NaN in → NaN out); nothing is
imputed.

Spec: `docs/archive/in_progress/tasks_assigned_12_6_26/process_signal_design.md` §S7
(tracked as PROC-15). Sibling: `pca_combo` (linear composition; this is linear *decomposition*).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    Finding, ProcessContext, ProcessResult, TransformProcess, make_run_id,
    partition_usable_columns,
)
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")

#: Prefix of every emitted column.
RES_PREFIX = "res_"

#: Below this fit-segment standard deviation a conditioner carries no variation to remove.
_MIN_Z_STD = 1e-12

#: Condition number above which the conditioning set is treated as collinear.
_MAX_COND = 1e10


@register
class ResidualizeProcess(TransformProcess):
    """Prefix-fitted OLS residuals of features against a conditioning set."""

    PARAMS = {
        "features": (None, "columns to residualize; None = all numeric non-meta, "
                           "excluding the conditioning set and price columns"),
        "conditioning": (None, "conditioning columns Z (required)"),
        "fit_frac": (0.7, "leading fraction of rows used to fit beta (no lookahead)"),
        "min_obs": (100, "minimum jointly-valid fit rows"),
    }

    def name(self) -> str:
        return "residualize"

    def transform(self, bars: pd.DataFrame, ctx: ProcessContext
                  ) -> tuple[pd.DataFrame, ProcessResult]:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        empty = pd.DataFrame(index=bars.index)

        # ── conditioning set ─────────────────────────────────────────────────────
        z_cols = list(self.params.get("conditioning") or [])
        if not z_cols:
            return empty, result.finalize(
                time.time() - t0, error="conditioning set is required (nothing to remove)")
        missing = [c for c in z_cols if c not in bars.columns]
        if missing:
            return empty, result.finalize(
                time.time() - t0, error=f"conditioning column(s) not in data: {missing}")

        n = len(bars)
        cut = max(1, int(n * float(self.params["fit_frac"])))
        min_obs = int(self.params["min_obs"])

        Z_full = bars[z_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        z_std = np.nanstd(Z_full[:cut], axis=0)
        dead = [c for c, s in zip(z_cols, z_std) if not np.isfinite(s) or s < _MIN_Z_STD]
        if dead:
            result.summary_extra = {"conditioning_skipped": dead}
            res = result.finalize(
                time.time() - t0,
                error=f"conditioning column(s) constant on the fit segment: {dead} — "
                      "no variation to remove, and their covariance is not invertible")
            res.summary["conditioning_skipped"] = dead
            return empty, res

        # ── target features ──────────────────────────────────────────────────────
        feats = self.params.get("features")
        self_cond: list[str] = []
        if feats is None:
            feats = [c for c in bars.columns
                     if pd.api.types.is_numeric_dtype(bars[c])
                     and c not in z_cols
                     and not c.startswith(_PRICE_PREFIXES)
                     and c != ctx.price_col
                     and not c.startswith("_")]
        else:
            self_cond = [c for c in feats if c in z_cols]
            feats = [c for c in feats if c not in z_cols]
        usable, skipped = partition_usable_columns(bars, list(feats), min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped
        # a feature residualized against itself is identically zero — dropped, but on
        # the record, because a silent omission is indistinguishable from a bug
        for c in (self_cond if feats is not None else []):
            result.features_skipped.append(
                {"feature": c, "reason": "is_conditioning_column (res would be identically 0)"})
        if not usable:
            return empty, result.finalize(time.time() - t0, error="no usable features")

        # ── fit beta on the prefix, apply everywhere ─────────────────────────────
        out: dict[str, np.ndarray] = {}
        betas: dict[str, dict] = {}
        for col in usable:
            f_full = bars[col].to_numpy(dtype=np.float64, na_value=np.nan)
            fit_rows = np.zeros(n, dtype=bool)
            fit_rows[:cut] = True
            ok = fit_rows & np.isfinite(f_full) & np.isfinite(Z_full).all(axis=1)
            if int(ok.sum()) < min_obs:
                result.features_skipped.append(
                    {"feature": col, "reason": f"fit_rows={int(ok.sum())}<{min_obs}"})
                continue

            A = np.column_stack([Z_full[ok], np.ones(int(ok.sum()))])   # + intercept
            if np.linalg.cond(A) > _MAX_COND:
                result.features_skipped.append(
                    {"feature": col, "reason": "collinear_conditioning_set"})
                continue
            coef, *_ = np.linalg.lstsq(A, f_full[ok], rcond=None)

            A_full = np.column_stack([Z_full, np.ones(n)])
            res_series = f_full - A_full @ coef          # NaN in Z or f propagates
            out[RES_PREFIX + col] = res_series

            fitted = A @ coef
            var = float(np.var(f_full[ok]))
            r2 = float(1.0 - np.var(f_full[ok] - fitted) / var) if var > 0 else 0.0
            betas[col] = {z: round(float(b), 8) for z, b in zip(z_cols, coef[:-1])}
            betas[col]["_intercept"] = round(float(coef[-1]), 8)

            # orthogonality is only evidence where it was NOT fitted
            hold = np.zeros(n, dtype=bool)
            hold[cut:] = True
            hm = hold & np.isfinite(res_series) & np.isfinite(Z_full).all(axis=1)
            max_abs_corr = 0.0
            if int(hm.sum()) > 10:
                for j in range(Z_full.shape[1]):
                    zz = Z_full[hm, j]
                    if np.std(zz) > _MIN_Z_STD and np.std(res_series[hm]) > _MIN_Z_STD:
                        c = abs(float(np.corrcoef(res_series[hm], zz)[0, 1]))
                        max_abs_corr = max(max_abs_corr, c)
            result.findings.append(Finding(
                feature=RES_PREFIX + col, horizon="n/a", metric="holdout_abs_corr_z",
                value=round(max_abs_corr, 6), informative=bool(max_abs_corr < 0.1),
                extras={"source_feature": col, "conditioning": z_cols,
                        "r2_fit": round(r2, 6), "betas": betas[col],
                        "n_fit_rows": int(ok.sum()), "n_holdout_rows": int(hm.sum())},
            ))

        if not out:
            return empty, result.finalize(time.time() - t0,
                                          error="no feature could be residualized")

        derived = pd.DataFrame(out, index=bars.index)
        result.finalize(time.time() - t0)
        result.summary.update({
            "derived_columns": list(derived.columns),
            "conditioning": z_cols,
            "fit_frac": float(self.params["fit_frac"]),
            "n_fit_rows": cut,
            "betas": betas,
            "max_holdout_abs_corr": (round(max(f.value for f in result.findings), 6)
                                     if result.findings else None),
        })
        return derived, result

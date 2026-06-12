"""
PCA combination — orthogonal composite features scored by held-out IC.

Fits a PCA on the TRAINING PREFIX of the data only (`fit_frac`, default the
first 70%) — scaler statistics and component loadings never see the holdout,
so the derived series carry no lookahead. The full series is then projected
onto the fitted components, yielding `pc_1..pc_k` ordered by explained
variance.

Each component is scored by full-sample Spearman IC against forward returns
ON THE HOLDOUT SEGMENT ONLY (out-of-sample by construction). Orthogonality
is a first-class property: `summary.orthogonality` reports the maximum
off-diagonal |correlation| between components measured on the holdout —
near-zero is the point of using PCA for signal combination (decorrelated
composites size independently in a portfolio).

For FDR-rigorous scoring, chain the derived output into the ic_horizon
process (`--score-with ic_horizon`); the per-component IC here is a ranking
device, not a significance claim.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from alpha.screener import compute_forward_returns

from .base import Finding, ProcessContext, ProcessResult, TransformProcess, make_run_id, partition_usable_columns
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


@register
class PCAComboProcess(TransformProcess):
    """Train-prefix PCA -> orthogonal pc_1..pc_k scored by holdout IC."""

    PARAMS = {
        "features": (None, "list of name prefixes to combine; None = all non-meta numeric"),
        "n_components": (5, "components to keep (capped at usable feature count)"),
        "fit_frac": (0.7, "fraction of rows used to fit scaler + PCA (no lookahead)"),
        "standardize": (True, "z-score features with train-segment stats before PCA"),
        "min_ic": (0.015, "holdout |IC| for the informative flag"),
        "min_obs": (100, "minimum valid observations per input column"),
    }

    def name(self) -> str:
        return "pca_combo"

    def transform(
        self, bars: pd.DataFrame, ctx: ProcessContext,
    ) -> tuple[pd.DataFrame, ProcessResult]:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        from sklearn.decomposition import PCA

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=int(self.params["min_obs"]))
        result.features_tested = usable
        result.features_skipped = skipped
        if len(usable) < 2:
            return pd.DataFrame(index=bars.index), result.finalize(
                time.time() - t0, error=f"need >= 2 usable features, got {len(usable)}")

        n = len(bars)
        split = int(n * float(self.params["fit_frac"]))
        if split < 20 or n - split < 20:
            return pd.DataFrame(index=bars.index), result.finalize(
                time.time() - t0, error=f"split {split}/{n - split} too small")

        X = bars[usable].to_numpy(dtype=np.float64, na_value=np.nan)

        # Train-segment statistics only — impute and scale with no lookahead
        mu = np.nanmean(X[:split], axis=0)
        X_filled = np.where(np.isfinite(X), X, mu)
        if self.params["standardize"]:
            sd = np.nanstd(X[:split], axis=0)
            sd[sd < 1e-15] = 1.0
            X_filled = (X_filled - mu) / sd

        k = min(int(self.params["n_components"]), len(usable), split - 1)
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X_filled[:split])
        comps = pca.transform(X_filled)

        pc_names = [f"pc_{i + 1}" for i in range(k)]
        derived = pd.DataFrame(comps, columns=pc_names, index=bars.index)
        for meta in ("bar_start", "symbol"):
            if meta in bars.columns:
                derived[meta] = bars[meta]

        # Score each component on the holdout segment only
        prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        fwd = {
            h_name: compute_forward_returns(prices, h_bars)[split:]
            for h_name, h_bars in ctx.horizons.items()
        }
        for i, pc in enumerate(pc_names):
            x_hold = comps[split:, i]
            ic_by_h = {}
            for h_name, fr in fwd.items():
                m = np.isfinite(x_hold) & np.isfinite(fr)
                if m.sum() < 30:
                    continue
                rho, _ = stats.spearmanr(x_hold[m], fr[m])
                ic_by_h[h_name] = round(float(rho), 4) if np.isfinite(rho) else 0.0
            best_h = max(ic_by_h, key=lambda h: abs(ic_by_h[h]), default=None)
            best_ic = ic_by_h.get(best_h, 0.0) if best_h else 0.0

            loadings = pca.components_[i]
            top_idx = np.argsort(-np.abs(loadings))[:5]
            result.findings.append(Finding(
                feature=pc, horizon=best_h, metric="ic_holdout",
                value=best_ic,
                threshold=float(self.params["min_ic"]),
                informative=bool(abs(best_ic) >= float(self.params["min_ic"])),
                extras={
                    "explained_var": round(float(pca.explained_variance_ratio_[i]), 4),
                    "ic_by_horizon": ic_by_h,
                    "loadings_top5": {
                        usable[j]: round(float(loadings[j]), 4) for j in top_idx
                    },
                },
            ))

        result.finalize(time.time() - t0)
        # Orthogonality measured OUT of sample — the property the portfolio
        # sizing relies on, not the in-sample tautology
        hold = comps[split:]
        if hold.shape[0] > 10 and k > 1:
            corr = np.corrcoef(hold, rowvar=False)
            off_diag = np.abs(corr[~np.eye(k, dtype=bool)])
            result.summary["orthogonality"] = round(float(off_diag.max()), 4)
        else:
            result.summary["orthogonality"] = None
        result.summary["explained_var_total"] = round(
            float(pca.explained_variance_ratio_.sum()), 4)
        return derived, result

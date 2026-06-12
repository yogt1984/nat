"""
IC / horizon-sweep process — the workhorse statistical description.

For each feature x horizon, computes the expanding-window Spearman rank IC
(anchored at t=0 — no lookahead) against forward returns, with explicit time
dependence:

  - the full expanding-IC curve (how IC evolved as data accumulated)
  - the IC decay curve across the horizon sweep, with a fitted half-life
    |IC(h)| ~ |IC_0| * exp(-h / tau)

plus turnover, breakeven cost, and BH-FDR correction across all tests in the
run. A feature is informative iff p_adj < fdr_alpha AND |IC| >= min_ic AND
breakeven_bps >= min_breakeven_bps — the same gate as the G1 alpha screen
(`scripts/alpha/screener.py`), whose primitives this process reuses.

Note on semantics inherited from the screener: `ic_mean` is the mean of the
last <= recent_k expansion points (the most data-rich estimates), while
`ic_std` is over the whole expanding series — early low-sample points make it
conservative.

Deliberate deviation from the screener: the screener derives its p-value
from a t-test over the expansion points, but those windows overlap almost
completely (each contains all previous data), so the test treats one noisy
correlation as n_windows independent replications and flags pure noise as
significant. Here the p-value comes from the full-sample Spearman rho with
an effective sample size n_eff = n_valid / horizon_bars, the standard
conservative correction for overlapping forward returns.

With `target_col` set (e.g. `tb_label` from the triple_barrier process), the
column replaces forward returns as the prediction target under the single
horizon name "label".
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from alpha.screener import (
    benjamini_hochberg,
    compute_breakeven_bps,
    compute_expanding_ic,
    compute_forward_returns,
    compute_turnover,
)

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id, partition_usable_columns
from .registry import register

# Price columns are trivially correlated with returns — never score them
_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


@register
class ICHorizonProcess(EvaluationProcess):
    """Expanding-window Spearman IC across a horizon sweep with FDR control."""

    PARAMS = {
        "features": (None, "list of name prefixes to score; None = all non-meta numeric"),
        "fdr_alpha": (0.05, "BH-FDR significance threshold"),
        "min_ic": (0.015, "minimum |IC| for the informative gate"),
        "min_breakeven_bps": (2.0, "minimum breakeven cost for the informative gate"),
        "min_obs": (50, "minimum valid observations per column / expansion window"),
        "recent_k": (10, "number of latest expansion points averaged into ic_mean"),
        "target_col": (None, "alternative target column replacing forward returns"),
    }

    def name(self) -> str:
        return "ic_horizon"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe,
            params=dict(self.params),
        )

        min_obs = int(self.params["min_obs"])
        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped

        # Targets: forward returns per horizon, or an explicit label column
        target_col = self.params["target_col"] or ctx.target_col
        if target_col:
            if target_col not in bars.columns:
                return result.finalize(time.time() - t0,
                                       error=f"target_col '{target_col}' not in data")
            tgt = bars[target_col].to_numpy(dtype=np.float64, na_value=np.nan)
            targets = {"label": (0, tgt)}
        else:
            prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
            targets = {
                h_name: (h_bars, compute_forward_returns(prices, h_bars))
                for h_name, h_bars in ctx.horizons.items()
            }
        vol_target = {
            h: float(np.nanstd(fr)) if np.isfinite(fr).sum() > 10 else 0.0
            for h, (_, fr) in targets.items()
        }

        findings: list[Finding] = []
        for feat in usable:
            vals = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
            turnover = compute_turnover(vals)
            per_horizon: list[tuple[str, int, Finding]] = []

            for h_name, (h_bars, fr) in targets.items():
                ic_series = compute_expanding_ic(vals, fr, min_obs=min_obs)
                valid_ics = ic_series[~np.isnan(ic_series)]
                if len(valid_ics) < 2:
                    continue

                recent = valid_ics[-min(len(valid_ics), int(self.params["recent_k"])):]
                ic_mean = float(np.mean(recent))
                ic_std = float(np.std(valid_ics))
                ic_ir = ic_mean / ic_std if ic_std > 1e-15 else 0.0
                n_win = len(valid_ics)

                # Full-sample Spearman with overlap-corrected effective n
                # (see module docstring — expansion points are not replications)
                joint = np.isfinite(vals) & np.isfinite(fr)
                rho_full = 0.0
                if joint.sum() >= min_obs:
                    rho, _ = stats.spearmanr(vals[joint], fr[joint])
                    rho_full = float(rho) if np.isfinite(rho) else 0.0
                n_eff = max(joint.sum() / max(h_bars, 1), 8.0)
                denom = max(1.0 - rho_full * rho_full, 1e-12)
                t_stat = rho_full * np.sqrt((n_eff - 2) / denom)
                p_value = float(2 * stats.t.sf(abs(t_stat), df=max(n_eff - 2, 1)))
                breakeven = compute_breakeven_bps(ic_mean, vol_target[h_name], turnover)

                per_horizon.append((h_name, h_bars, Finding(
                    feature=feat, horizon=h_name, metric="ic_mean",
                    value=round(ic_mean, 6),
                    threshold=float(self.params["min_ic"]),
                    p_value=p_value,
                    extras={
                        "ic_ir": round(ic_ir, 4),
                        "ic_std": round(ic_std, 6),
                        "ic_full_sample": round(rho_full, 6),
                        "t_stat": round(float(t_stat), 4),
                        "n_eff": round(float(n_eff), 1),
                        "n_windows": n_win,
                        "turnover": round(turnover, 4) if np.isfinite(turnover) else None,
                        "breakeven_bps": round(breakeven, 2) if np.isfinite(breakeven) else None,
                        "ic_expanding": [round(float(v), 4) for v in valid_ics],
                    },
                )))

            # IC decay across the horizon sweep + fitted half-life
            decay = {h: f.value for h, _, f in per_horizon}
            halflife = _fit_ic_halflife(
                [(hb, abs(f.value)) for _, hb, f in per_horizon]
            )
            for _, _, f in per_horizon:
                f.extras["ic_decay"] = decay
                f.extras["ic_decay_halflife_bars"] = halflife
                findings.append(f)

        # BH-FDR across every test in this run
        if findings:
            p = np.array([f.p_value for f in findings], dtype=np.float64)
            adj = benjamini_hochberg(p, alpha=float(self.params["fdr_alpha"]))
            for f, pa in zip(findings, adj):
                f.p_adjusted = round(float(pa), 6) if not np.isnan(pa) else 1.0
                be = f.extras.get("breakeven_bps")
                f.informative = bool(
                    f.p_adjusted < float(self.params["fdr_alpha"])
                    and abs(f.value) >= float(self.params["min_ic"])
                    and (be is None or be >= float(self.params["min_breakeven_bps"]))
                )

        result.findings = findings
        return result.finalize(time.time() - t0)


def _fit_ic_halflife(points: list[tuple[int, float]]) -> float | None:
    """Half-life (in bars) of |IC(h)| ~ |IC_0| * exp(-h/tau) via log-linear fit."""
    pts = [(h, ic) for h, ic in points if ic > 1e-6 and h > 0]
    if len(pts) < 2:
        return None
    h = np.array([p[0] for p in pts], dtype=np.float64)
    log_ic = np.log(np.array([p[1] for p in pts], dtype=np.float64))
    slope = np.polyfit(h, log_ic, 1)[0]
    if slope >= 0:
        return None  # no decay over this sweep
    return round(float(np.log(2.0) / -slope), 2)

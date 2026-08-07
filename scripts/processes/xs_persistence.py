"""XS-4 — `xs_persistence`: does the ranking outlive the rebalance?

XS-3 established that two scores rank relative forward returns (FINDINGS §7.4). That is
necessary, not sufficient. If a score's *ordering* is reshuffled by the time you next
rebalance, every rebalance is a fresh draw: turnover is maximal, the full spread is paid
each time, and the strategy chases a signal that has already gone. `TASKS.md` sets the
bar — "must exceed the rebalance cadence or the rotation is churn by construction".

**What is measured.** Rank autocorrelation as a function of lag: at each lag *k*, the
Spearman correlation between the cross-sectional ranking at *t* and at *t+k*, averaged
over *t*. The decay is summarised as a half-life, and the verdict compares that half-life
to the intended cadence — persistence is meaningless in absolute terms, only relative to
how often you trade.

Two details that decide correctness:

* **Only pairs present at BOTH ends of a lag are compared.** The panel has holes by design
  (PROC-19): a pair listed on day 40 had no rank on day 10 and therefore cannot have
  "changed rank" between them. Treating absence as a rank change would understate
  persistence exactly for the recently-listed tail.
* **The half-life is fitted on the positive part of the decay**, and cross-checked against
  the first empirical crossing of 0.5. An exponential fit through noise-level
  autocorrelations would otherwise extrapolate a confident half-life out of nothing.

This process ranks a score column already present in the frame, so it composes with
whatever produced it (XS-2's estimators, or a combiner) rather than re-deriving scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


from processes.base import EvaluationProcess, Finding, ProcessResult, make_run_id
from processes.registry import register


@register
class XsPersistence(EvaluationProcess):
    """Rank autocorrelation half-life of a cross-sectional score."""

    kind = "evaluation"
    data_level = "candles"

    def name(self) -> str:
        return "xs_persistence"

    def __init__(self, score_col: str = "score", max_lag: int = 168,
                 cadence: int = 24, min_pairs: int = 20, min_overlap: int = 10,
                 stride: int = 1, **kw):
        self.score_col = str(score_col)
        self.max_lag = int(max_lag)
        self.cadence = int(cadence)
        self.min_pairs = int(min_pairs)
        self.min_overlap = int(min_overlap)
        #: Subsample start times when estimating each lag's autocorrelation. The estimate
        #: is a MEAN over start times, so striding widens its standard error but does not
        #: bias it — and consecutive start times are near-duplicates anyway.
        self.stride = max(1, int(stride))
        self.params = dict(stride=self.stride, score_col=self.score_col, max_lag=self.max_lag,
                           cadence=self.cadence, min_pairs=self.min_pairs,
                           min_overlap=self.min_overlap)

    def required_columns(self) -> list[str]:
        return [self.score_col]

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman via argsort ranks — the hot path, called once per (lag, t).

        `scipy.stats.spearmanr` is ~40x slower here and its tie correction is irrelevant:
        these are continuous scores, where exact ties are measure-zero. A tie would be
        broken arbitrarily rather than averaged, which cannot move a mean over thousands
        of cross-sections.
        """
        n = x.size
        rx = np.empty(n, dtype=float); rx[np.argsort(x, kind="stable")] = np.arange(n)
        ry = np.empty(n, dtype=float); ry[np.argsort(y, kind="stable")] = np.arange(n)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
        return float((rx * ry).sum() / d) if d > 0 else float("nan")

    def _rank_autocorr(self, arr: np.ndarray, lag: int) -> float:
        """Mean Spearman(rank_t, rank_{t+lag}) over usable t (strided)."""
        if lag == 0:
            return 1.0
        vals = []
        n = arr.shape[0]
        for t in range(0, n - lag, self.stride):
            a, b = arr[t], arr[t + lag]
            # Pairs present at BOTH ends only: absence is not a rank change.
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < self.min_overlap:
                continue
            x, y = a[both], b[both]
            if np.all(x == x[0]) or np.all(y == y[0]):
                continue
            c = self._spearman(x, y)
            if np.isfinite(c):
                vals.append(c)
        return float(np.mean(vals)) if vals else float("nan")

    def _half_life(self, lags: np.ndarray, ac: np.ndarray) -> tuple[float, str]:
        """Half-life from an exponential fit, cross-checked against the 0.5 crossing."""
        finite = np.isfinite(ac)
        lags, ac = lags[finite], ac[finite]
        if len(ac) < 3:
            return float("nan"), "insufficient lags"

        # Empirical: first lag whose autocorrelation is at or below 0.5.
        below = np.where(ac <= 0.5)[0]
        crossing = float(lags[below[0]]) if len(below) else float("inf")

        # Fit log(ac) ~ -lambda * lag over the positive, above-noise part only. Fitting
        # through noise-level values would extrapolate a confident half-life from nothing.
        usable = (ac > 0.05) & (lags > 0)
        if usable.sum() >= 3:
            slope = np.polyfit(lags[usable], np.log(ac[usable]), 1)[0]
            fitted = float(np.log(0.5) / slope) if slope < 0 else float("inf")
        else:
            fitted = crossing

        if not np.isfinite(fitted) and not np.isfinite(crossing):
            return float("inf"), "no decay observed within max_lag"
        # Prefer the fit; fall back to the crossing when the fit is degenerate.
        hl = fitted if np.isfinite(fitted) else crossing
        return hl, f"fit={fitted:.1f} crossing={crossing}"

    # ── contract ─────────────────────────────────────────────────────────

    def evaluate(self, frame: pd.DataFrame, ctx) -> ProcessResult:
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind, symbol=ctx.symbol,
            timeframe=ctx.timeframe, params=dict(self.params))

        if self.score_col not in frame.columns:
            result.features_skipped = [{"feature": self.score_col,
                                        "reason": "score column not in frame"}]
            result.summary = {"n_lags": 0, "skipped_reason": "missing score column"}
            return result

        wide = frame.pivot_table(index="timestamp", columns="symbol",
                                 values=self.score_col, aggfunc="last").sort_index()

        widest = int(wide.notna().sum(axis=1).max() or 0)
        if widest < self.min_pairs:
            result.features_skipped = [{"feature": self.score_col,
                                        "reason": f"widest cross-section {widest} < min_pairs"}]
            result.summary = {"n_lags": 0,
                              "skipped_reason": f"widest cross-section {widest} < "
                                                f"min_pairs={self.min_pairs}"}
            return result

        lags = np.arange(0, min(self.max_lag, len(wide) - 1) + 1)
        arr = wide.to_numpy(dtype=float)
        ac = np.array([self._rank_autocorr(arr, int(k)) for k in lags])
        hl, how = self._half_life(lags, ac)

        # The verdict is relative: persistence only matters against the trading cadence.
        passes = np.isfinite(hl) and hl > self.cadence or hl == float("inf")
        verdict = (f"half-life {hl:.1f} bars > cadence {self.cadence} — tradeable at this cadence"
                   if passes else
                   f"half-life {hl:.1f} bars <= cadence {self.cadence} — churn by "
                   f"construction at this cadence")

        result.findings = [Finding(
            feature=f"xs_rank_persistence[{self.score_col}]",
            horizon=f"{self.cadence}bar", metric="rank_half_life_bars",
            value=(float(hl) if np.isfinite(hl) else float("inf")),
            threshold=float(self.cadence), informative=bool(passes),
            extras={"autocorr": [None if not np.isfinite(v) else round(float(v), 4)
                                 for v in ac],
                    "lags": [int(k) for k in lags],
                    "how": how, "verdict": verdict,
                    "mean_pairs": round(float(wide.notna().sum(axis=1).mean()), 1)},
        )]
        result.features_tested = [self.score_col]
        result.summary = {"n_lags": int(len(lags)), "half_life_bars": (
            float(hl) if np.isfinite(hl) else None), "verdict": verdict}
        return result

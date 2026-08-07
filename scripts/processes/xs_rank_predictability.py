"""XS-3 — `xs_rank_predictability`: does any score rank the universe?

Track C's kill test. `THREE_CLASS_RESEARCH_PROPOSAL.md` §9 makes the verdict terminal —
"Track C stops if XS-3 finds no score family significant after FDR on >= 90 d" — so this
process must be able to return nothing, and be trusted when it does.

**What is measured.** At each rebalance the universe is ranked by a score computed from
each pair's own trailing history (XS-2), and that ranking is compared to the pairs'
*relative* forward returns via Spearman rank-IC. The statistic is the mean rank-IC across
rebalances; the verdict is that mean against a permutation null.

Three decisions define whether the number means anything:

1. **Rank within the cross-section, never pool.** Pooling scores and returns across dates
   manufactures correlation out of common time variation — if scores run high on days the
   whole market rallied, a pooled correlation is large while the cross-sectional ordering
   carries nothing. Every IC here is computed inside one timestamp.

2. **Returns are cross-sectionally demeaned.** Class 3 allocates *between* pairs, so the
   only question is relative performance. Without demeaning, a score correlated with beta
   would score well for tracking the market rather than for selecting within it.

3. **The null shuffles PAIR LABELS within each cross-section**, independently per date.
   Shuffling returns through time instead would destroy each pair's own return
   distribution and the date structure, answering a different question. This is the
   panel-aware form of PROC-12's discipline, and it reuses `NullResult` so the reported
   vocabulary (bits_above_null / z / p) is identical to every other calibrated process —
   only here the "estimator" is a mean rank-IC rather than an MI.

Multiple score families in one run is a multiple-comparison problem by construction, so
findings carry BH-FDR q-values (PROC-13).

Honest limits, stated because the verdict is terminal: rank-IC is a *signal-level*
measure and says nothing about cost, capacity or fill — `XS-5` and `XS-6` own those, and
FINDINGS §7.2 already shows the widest-spread pairs are nearly empty at the touch. A
positive verdict here is a licence to continue, never a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from it_engine.null_calibration import NullResult
from processes.base import EvaluationProcess, Finding, ProcessResult, make_run_id
from processes.registry import register
from xs.features import hurst_rs, momentum_strength, realized_vol

#: score name -> callable over a pair's trailing CLOSE series.
#: `permutation_entropy` is deliberately absent: FINDINGS §7.3 measured it as
#: non-discriminating across this universe (IQR 0.0005), so including it would spend an
#: FDR slot on a score already known to carry no cross-sectional information.
SCORERS = {
    "momentum": lambda c: momentum_strength(c),
    "hurst": lambda c: hurst_rs(c),
    "vol": lambda c: realized_vol(np.diff(np.log(c))) if len(c) > 2 else np.nan,
}


@register
class XsRankPredictability(EvaluationProcess):
    """Rank-IC of per-pair scores against relative forward returns, null-calibrated."""

    kind = "evaluation"
    data_level = "candles"

    def name(self) -> str:
        return "xs_rank_predictability"

    def __init__(self, scores=("momentum", "hurst", "vol"), lookback: int = 168,
                 horizon: int = 24, rebalance_every: int = 24, min_pairs: int = 20,
                 n_shuffles: int = 200, min_ic: float = 0.02, z_threshold: float = 3.0,
                 fdr_q: float = 0.05, seed: int = 42, return_mode: str = "simple", **kw):
        self.scores = tuple(scores)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.rebalance_every = int(rebalance_every)
        self.min_pairs = int(min_pairs)
        self.n_shuffles = int(n_shuffles)
        self.min_ic = float(min_ic)
        self.z_threshold = float(z_threshold)
        self.fdr_q = float(fdr_q)
        self.seed = int(seed)
        #: "simple" (P1/P0-1) or "log". Ranks are invariant to monotone transforms, so
        #: these agree unless something non-monotone is happening — which makes the pair
        #: a cheap check that a vol-ranked result is not an artifact of return skew.
        self.return_mode = str(return_mode)
        self.params = dict(return_mode=self.return_mode, scores=list(self.scores), lookback=self.lookback,
                           horizon=self.horizon, rebalance_every=self.rebalance_every,
                           min_pairs=self.min_pairs, n_shuffles=self.n_shuffles,
                           min_ic=self.min_ic, z_threshold=self.z_threshold,
                           fdr_q=self.fdr_q, seed=self.seed)

    def required_columns(self) -> list[str]:
        return ["close"]

    # ── internals ────────────────────────────────────────────────────────

    def _wide(self, frame: pd.DataFrame) -> pd.DataFrame:
        """(timestamp x symbol) close matrix. NaN where a pair had not listed — which is
        information, not a gap to fill: PROC-19 keeps the panel's holes deliberately."""
        return frame.pivot_table(index="timestamp", columns="symbol",
                                 values="close", aggfunc="last").sort_index()

    def _cross_sections(self, wide: pd.DataFrame):
        """Yield (t, scores_by_symbol, relative_forward_return_by_symbol) per rebalance."""
        idx = wide.index
        # Stop `horizon` bars early: beyond that there is no forward return to compare
        # against, and scoring those dates would be scoring nothing.
        last = len(idx) - self.horizon
        for i in range(self.lookback, last, self.rebalance_every):
            hist = wide.iloc[i - self.lookback:i + 1]
            ratio = wide.iloc[i + self.horizon] / wide.iloc[i]
            fwd = np.log(ratio) if self.return_mode == "log" else ratio - 1.0

            # NB explicit parens: `&` binds tighter than `>=`, so writing this inline
            # silently yields an all-False mask and hence zero cross-sections.
            has_now = hist.iloc[-1].notna()
            has_fwd = fwd.notna()
            enough_history = hist.notna().sum() >= self.lookback * 0.8
            live = hist.columns[has_now & has_fwd & enough_history]
            if len(live) < self.min_pairs:
                continue
            r = fwd[live].astype(float)
            # Relative performance: the only thing a between-pair allocator can act on.
            yield idx[i], hist[live], r - r.mean()

    def _mean_rank_ic(self, per_date: list[tuple[np.ndarray, np.ndarray]]) -> float:
        ics = []
        for s, r in per_date:
            if len(s) < 3 or np.all(s == s[0]) or np.all(r == r[0]):
                continue
            ic = stats.spearmanr(s, r).statistic
            if np.isfinite(ic):
                ics.append(ic)
        return float(np.mean(ics)) if ics else float("nan")

    def _null(self, per_date, rng) -> tuple[float, float, float]:
        """Permutation null: shuffle pair labels INSIDE each cross-section, independently.

        This preserves every pair's own return, the date structure, and the number of
        pairs live on each date — it destroys only the score-to-pair assignment, which is
        precisely the hypothesis under test.
        """
        draws = []
        for _ in range(self.n_shuffles):
            shuffled = [(s, rng.permutation(r)) for s, r in per_date]
            v = self._mean_rank_ic(shuffled)
            if np.isfinite(v):
                draws.append(v)
        if len(draws) < 2:
            return float("nan"), float("nan"), 0.0
        a = np.asarray(draws)
        return float(a.mean()), float(a.std(ddof=1)), a

    # ── contract ─────────────────────────────────────────────────────────

    def evaluate(self, frame: pd.DataFrame, ctx) -> ProcessResult:
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params))
        wide = self._wide(frame)
        sections = list(self._cross_sections(wide))

        if not sections:
            result.summary = {
                "n_rebalances": 0,
                "skipped_reason": (
                    f"no cross-section reached min_pairs={self.min_pairs} with "
                    f"lookback={self.lookback} and horizon={self.horizon}"
                ),
            }
            result.features_skipped = [{"feature": s, "reason": "no valid cross-section"}
                                       for s in self.scores]
            return result

        rng = np.random.default_rng(self.seed)
        findings: list[Finding] = []

        for score_name in self.scores:
            fn = SCORERS.get(score_name)
            if fn is None:
                result.features_skipped.append(
                    {"feature": score_name, "reason": "unknown score"})
                continue

            per_date = []
            for _t, hist, rel in sections:
                vals = np.array([fn(hist[c].dropna().to_numpy(float)) for c in hist.columns])
                ok = np.isfinite(vals) & np.isfinite(rel.to_numpy(float))
                if ok.sum() >= self.min_pairs:
                    per_date.append((vals[ok], rel.to_numpy(float)[ok]))

            if not per_date:
                result.features_skipped.append(
                    {"feature": score_name, "reason": "score produced no usable cross-section"})
                continue

            raw = self._mean_rank_ic(per_date)
            null_mean, null_std, draws = self._null(per_date, rng)
            z = ((raw - null_mean) / null_std) if null_std and np.isfinite(null_std) else float("nan")
            p = (float((np.abs(draws) >= abs(raw)).sum()) + 1.0) / (len(draws) + 1.0) \
                if isinstance(draws, np.ndarray) else float("nan")

            nr = NullResult(raw_bits=raw, null_mean=null_mean, null_std=null_std,
                            bits_above_null=raw - null_mean, z=z, p=p,
                            n_shuffles=self.n_shuffles)
            findings.append(Finding(
                feature=f"xs_{score_name}", horizon=f"{self.horizon}bar",
                metric="rank_ic_mean", value=round(raw, 6),
                threshold=self.min_ic, p_value=p,
                # TWO-sided in z. Rank-IC is signed, unlike the MI the other calibrated
                # processes report: a reliably NEGATIVE IC is a real signal that trades
                # with the sign inverted, and a one-sided `z >= threshold` would discard
                # it silently. The direction is reported as `polarity` instead — which is
                # also what PROC-1's compiler requires before it will emit an algorithm.
                informative=bool(abs(raw) >= self.min_ic and np.isfinite(z)
                                 and abs(z) >= self.z_threshold),
                extras={**nr.to_dict(), "n_rebalances": len(per_date),
                        "polarity": ("positive" if raw > 0 else "negative"),
                        "mean_pairs": round(float(np.mean([len(s) for s, _ in per_date])), 1)},
            ))

        # PROC-13: several score families in one run is multiple testing by construction.
        self._apply_fdr(findings)

        result.findings = findings
        result.features_tested = [f.feature for f in findings]
        result.summary = {
            "n_rebalances": len(sections),
            "n_informative": sum(1 for f in findings if f.informative),
            "last_rebalance": str(sections[-1][0]),
            "universe_mean": round(float(np.mean([s.shape[1] for _, s, _ in sections])), 1),
        }
        return result

    def _apply_fdr(self, findings: list[Finding]) -> None:
        """Benjamini-Hochberg over the score families tested in this run."""
        scored = [f for f in findings if f.p_value is not None and np.isfinite(f.p_value)]
        if not scored:
            return
        order = sorted(range(len(scored)), key=lambda i: scored[i].p_value)
        m = len(scored)
        prev = 1.0
        for rank, i in enumerate(reversed(order), start=1):
            k = m - rank + 1
            q = min(prev, scored[i].p_value * m / k)
            scored[i].p_adjusted = round(float(q), 6)
            prev = q
        for f in scored:
            if f.p_adjusted is not None and f.p_adjusted > self.fdr_q:
                f.informative = False

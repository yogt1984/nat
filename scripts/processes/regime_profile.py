"""PROF-1: per-asset J/K regime profiling — does this asset trend, revert, or neither?

The question, in the literature's own notation. Jegadeesh & Titman (1993) call it a **J/K
strategy**: `J` is the formation (lookback) period, `K` the holding period. "If it ran up over
the last J bars, does it continue over the next K?" is `P(sign r_{t,t+K} = sign r_{t-J,t})`,
against a null of 0.5. The companions are **VR(q)** — Lo & MacKinlay's (1988) variance ratio
`Var(r_q)/(q·Var(r_1))`, >1 trending, <1 reverting, =1 a random walk — and the **Hurst
exponent** (>0.5 persistent). J/K is the primary grid here because it is simultaneously a
statistic and a strategy, so it can be falsified on its own terms.

**Why the candle universe.** Every refuted result in the record shares one property: the
predicted move is comparable to the cost. 177 assets at 15m/1h/1d is where that ratio can
exceed 1; the three majors at 1–5 s is where 13 of 14 refutations happened. So every return
cell carries **`cost_coverage` = |jk_return| / rt_cost**, reported alongside significance and
never derived later — a cell with coverage < 1 is not tradeable however significant it is.

**Three ways this unit could manufacture a regime, all refused:**

- **The wrong null.** Permuting the *outcome* with the selection fixed asks "is this cell
  unlike the other cells", and since the pooled continuation rate already contains the
  persistence, that null scored a genuinely persistent AR(1) at z = −2.7 (PROC-20, corrected
  one commit ago). Here the **sign series is permuted and the runs recomputed**, so 0.5 is the
  null and the question is "is there persistence at all".
- **Overlapping windows.** Consecutive bars share K−1 bars of their forward window, which
  inflates every statistic; A-2's first run printed IC 0.39–0.46, *higher* than the claim it
  was auditing, from exactly this. Sampling is non-overlapping by default and the override is
  explicit.
- **An uncorrected grid.** symbols × J × K × buckets is thousands of cells, so BH-FDR
  (PROC-13) runs across all of them and per-day durability (PROC-4 folds) decides
  `durable | non_durable` — §4.9's binding failures were day-consistency and concentration,
  never the pooled mean.

**Entropy conditioning** answers "how do the entropy features behave under trend vs reversion"
as a measured interaction: each J/K cell is recomputed within terciles of a causal per-asset
percentile (`xs.features.rolling_self_percentile`) of permutation entropy or realised vol.
§5's `ent_book_shape` gate (+22 % IC lift) is the prior; XS-2's cross-sectional null result
(entropy IQR 0.0005) is the caution — entropy may separate *within* an asset while failing to
rank *across* them.

Spec: the approved PROF-1 plan. Contract: `docs/contracts/process.md`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from it_engine.null_calibration import load_null_config

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id
from .fdr import DEFAULT_FDR_ALPHA, apply_process_fdr
from .registry import register

BPS = 1e4


@register
class RegimeProfile(EvaluationProcess):
    """J/K continuation grid + variance ratio + Hurst, per asset, null-calibrated."""

    data_level = "candles"

    PARAMS = {
        "j_grid": ([1, 2, 4, 8, 16], "formation periods, in bars (Jegadeesh-Titman J)"),
        "k_grid": ([1, 2, 4, 8, 16], "holding periods, in bars (K)"),
        "vr_q": ([2, 4, 8, 16, 32], "aggregation horizons for the variance ratio"),
        "buckets": (None, "conditioning: None | 'entropy' | 'vol'"),
        "n_buckets": (3, "terciles by default"),
        "pct_window": (250, "lookback for the causal per-asset percentile"),
        "n_shuffles": (None, "sign permutations (default: it_engine.toml)"),
        "day_shuffles": (40, "sign permutations per day fold"),
        "min_obs": (60, "minimum non-overlapping observations for a cell"),
        "min_bars": (200, "minimum bars for a symbol to be profiled at all"),
        "min_days": (3, "minimum day folds before a durability verdict"),
        "min_frac_informative": (0.6, "day fraction required for 'durable'"),
        "allow_overlapping": (False, "USE ONLY to demonstrate the inflation; never in a study"),
        "seed": (0, "RNG seed"),
    }

    def name(self) -> str:
        return "regime_profile"

    # ── entry point ──────────────────────────────────────────────────────────────
    def evaluate(self, panel: pd.DataFrame, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params
        price_col = ctx.price_col if ctx.price_col in panel.columns else "close"
        for need in ("symbol", price_col):
            if need not in panel.columns:
                return result.finalize(time.time() - t0, error=f"missing column: {need}")

        cfg = load_null_config()
        z_thr = float(cfg["null_z_threshold"])
        n_shuffles = int(p["n_shuffles"] or cfg["n_shuffles"])
        from utils.costs import realistic_taker_rt_bps
        rt_cost = float(realistic_taker_rt_bps())

        tcol = "timestamp" if "timestamp" in panel.columns else None
        profiled = []
        for sym, grp in panel.groupby("symbol", sort=True):
            g = grp.sort_values(tcol) if tcol else grp
            px = g[price_col].to_numpy(dtype=np.float64)
            px = px[np.isfinite(px) & (px > 0)]
            if len(px) < int(p["min_bars"]):
                result.features_skipped.append(
                    {"feature": str(sym), "symbol": str(sym),
                     "reason": f"bars {len(px)} < {int(p['min_bars'])}"})
                continue
            profiled.append(str(sym))
            r = np.diff(np.log(px))                       # bar log returns
            days = self._day_index(g, len(r))
            self._symbol_stats(result, str(sym), r, z_thr)
            self._jk_grid(result, str(sym), r, days, z_thr, n_shuffles, rt_cost,
                          self._bucket_labels(r))

        result.features_tested = profiled
        report = apply_process_fdr(result, alpha=DEFAULT_FDR_ALPHA)
        result.finalize(time.time() - t0)
        result.summary.update({
            "fdr": {"alpha": report.alpha, "n_cells": report.n_cells,
                    "n_discoveries": report.n_discoveries},
            "n_symbols": len(profiled),
            "n_skipped": len(result.features_skipped),
            "rt_cost_bps": rt_cost,
            "non_overlapping": not bool(p["allow_overlapping"]),
            "notation": "Jegadeesh-Titman J/K: J=formation bars, K=holding bars",
        })
        return result

    # ── per-symbol: variance ratio + Hurst ───────────────────────────────────────
    def _symbol_stats(self, result, sym: str, r: np.ndarray, z_thr: float) -> None:
        for q in self.params["vr_q"]:
            vr, z = _variance_ratio(r, int(q))
            if vr is None:
                continue
            # A z WITHOUT a p_value silently escapes FDR: apply_process_fdr leaves
            # p-value-less cells untouched, so 865 VR tests would be corrected by nothing
            # and ~2 would clear |z|>=3 by chance. Lo-MacKinlay's z is asymptotically
            # normal, so convert it and let the cell join the family.
            pv = None
            if z is not None and np.isfinite(z):
                from scipy.stats import norm
                pv = float(2.0 * norm.sf(abs(float(z))))
            result.findings.append(Finding(
                feature=f"{sym}_vr{q}", horizon=f"q{q}", metric="variance_ratio",
                value=round(float(vr), 6), threshold=z_thr,
                p_value=None if pv is None else round(pv, 8),
                informative=bool(z is not None and abs(z) >= z_thr),
                extras={"symbol": sym, "q": int(q),
                        "z": None if z is None else round(float(z), 3),
                        "reading": "trending" if vr > 1 else "reverting"}))
        try:
            from xs.features import hurst_rs
            h = float(hurst_rs(r))
        except Exception:                                  # pragma: no cover - defensive
            h = float("nan")
        if np.isfinite(h):
            result.findings.append(Finding(
                feature=f"{sym}_hurst", horizon="n/a", metric="hurst",
                value=round(h, 6), informative=False,
                extras={"symbol": sym,
                        "reading": "persistent" if h > 0.5 else "anti_persistent"}))

    # ── the J/K grid ─────────────────────────────────────────────────────────────
    def _jk_grid(self, result, sym: str, r: np.ndarray, days: np.ndarray, z_thr: float,
                 n_shuffles: int, rt_cost: float, bucket_of: Optional[np.ndarray]) -> None:
        p = self.params
        buckets = [None] if bucket_of is None else \
            [None] + sorted({int(b) for b in bucket_of[np.isfinite(bucket_of)]})

        for J in p["j_grid"]:
            for K in p["k_grid"]:
                form, fwd = _jk_returns(r, int(J), int(K))
                stride = 1 if p["allow_overlapping"] else int(K)
                for b in buckets:
                    sel = np.isfinite(form) & np.isfinite(fwd)
                    if b is not None:
                        sel &= (bucket_of == b)
                    idx = np.flatnonzero(sel)[::stride]
                    if len(idx) < int(p["min_obs"]):
                        continue
                    self._emit_cell(result, sym, int(J), int(K),
                                    "all" if b is None else f"b{b}",
                                    form, fwd, idx, days, z_thr, n_shuffles, rt_cost)

    def _emit_cell(self, result, sym, J, K, bucket, form, fwd, idx, days,
                   z_thr, n_shuffles, rt_cost) -> None:
        p = self.params
        sgn = np.sign(form[idx])
        cont = (sgn == np.sign(fwd[idx])).astype(np.float64)
        cont_prob = float(cont.mean())
        jk_ret = float(np.mean(sgn * fwd[idx]) * BPS)

        rng = np.random.default_rng(int(p["seed"]))
        z, pval, null_mean = _score(cont_prob,
                                    _sign_permutation_null(form, fwd, idx, n_shuffles, rng))

        per_day = self._per_day(form, fwd, idx, days, int(p["day_shuffles"]),
                                int(p["seed"]), z_thr)
        frac = float(np.mean([d["informative"] for d in per_day])) if per_day else 0.0
        verdict = ("insufficient_days" if len(per_day) < int(p["min_days"])
                   else "durable" if frac >= float(p["min_frac_informative"])
                   else "non_durable")
        base = {"symbol": sym, "J": J, "K": K, "bucket": bucket,
                "n_obs": int(len(idx)), "cont_prob": round(cont_prob, 6),
                "non_overlapping": not bool(p["allow_overlapping"]),
                "z": None if z is None else round(z, 3),
                "null_mean": None if null_mean is None else round(null_mean, 6),
                "frac_days_informative": round(frac, 4), "n_days": len(per_day),
                "verdict": verdict, "per_day": per_day}

        result.findings.append(Finding(
            feature=f"{sym}_J{J}K{K}_{bucket}", horizon=f"K{K}",
            metric="cont_prob_excess", value=round(cont_prob - 0.5, 6),
            p_value=None if pval is None else round(pval, 6), threshold=z_thr,
            informative=bool(z is not None and abs(z) >= z_thr and verdict == "durable"),
            extras=base))
        result.findings.append(Finding(
            feature=f"{sym}_J{J}K{K}_{bucket}", horizon=f"K{K}",
            metric="jk_return_bps", value=round(jk_ret, 4), informative=False,
            # Cost coverage travels WITH the number, never derived later: it is the
            # statistic that explains 13 of 14 refutations in the record.
            extras={**base, "rt_cost_bps": round(rt_cost, 4),
                    "cost_coverage": round(abs(jk_ret) / rt_cost, 4) if rt_cost > 0 else None}))

    # ── helpers on the instance ──────────────────────────────────────────────────
    def _per_day(self, form, fwd, idx, days, day_shuffles, seed, z_thr) -> list[dict]:
        if days is None:
            return []
        out = []
        for d in np.unique(days[idx]):
            sub = idx[days[idx] == d]
            if len(sub) < 10:
                continue
            cp = float((np.sign(form[sub]) == np.sign(fwd[sub])).mean())
            rng = np.random.default_rng(seed + int(d))
            z, _, _ = _score(cp, _sign_permutation_null(form, fwd, sub, day_shuffles, rng))
            out.append({"day": int(d), "n": int(len(sub)), "cont_prob": round(cp, 6),
                        "z": None if z is None else round(z, 3),
                        "informative": bool(z is not None and abs(z) >= z_thr)})
        return out

    def _day_index(self, g: pd.DataFrame, n: int) -> Optional[np.ndarray]:
        if "timestamp" not in g.columns:
            return None
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        d = (ts.astype("int64") // 86_400_000_000_000).to_numpy()
        return d[1:1 + n] if len(d) > n else None

    def _bucket_labels(self, r: np.ndarray) -> Optional[np.ndarray]:
        """Causal per-asset terciles of entropy or vol; None when unconditioned."""
        which = self.params["buckets"]
        if which not in ("entropy", "vol"):
            return None
        w = int(self.params["pct_window"])
        s = pd.Series(r)
        if which == "vol":
            stat = s.rolling(w, min_periods=max(20, w // 5)).std()
        else:
            from xs.features import permutation_entropy
            stat = s.rolling(w, min_periods=max(20, w // 5)).apply(
                lambda x: permutation_entropy(x, order=3), raw=True)
        try:
            from xs.features import rolling_self_percentile
            pct = rolling_self_percentile(stat, window=w)
        except Exception:                                  # pragma: no cover - defensive
            pct = stat.rank(pct=True)
        v = pd.to_numeric(pct, errors="coerce").to_numpy(dtype=np.float64)
        nb = int(self.params["n_buckets"])
        out = np.full(len(r), np.nan)
        m = np.isfinite(v)
        out[m] = np.clip((v[m] * nb).astype(int), 0, nb - 1)
        return out


# ── module-level statistics ──────────────────────────────────────────────────────
def _jk_returns(r: np.ndarray, J: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Formation return over the past J bars and forward return over the next K.

    Both are aligned to index t: `form[t]` ends at t (known), `fwd[t]` starts at t+1.
    """
    n = len(r)
    c = np.concatenate([[0.0], np.cumsum(r)])
    form = np.full(n, np.nan)
    fwd = np.full(n, np.nan)
    if n > J:
        form[J:] = c[J + 1:n + 1] - c[1:n - J + 1]
    if n > K:
        fwd[:n - K] = c[1 + K:n + 1] - c[1:n - K + 1]
    return form, fwd


def _variance_ratio(r: np.ndarray, q: int):
    """Lo-MacKinlay VR(q) with the heteroskedasticity-robust z-statistic."""
    x = r[np.isfinite(r)]
    n = len(x)
    if q < 2 or n < 4 * q:
        return None, None
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    if var1 <= 0:
        return None, None
    agg = np.convolve(x, np.ones(q), mode="valid")         # overlapping q-sums
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((agg - q * mu) ** 2) / m if m > 0 else np.nan
    if not np.isfinite(varq):
        return None, None
    # NOT varq / (q * var1): the Lo-MacKinlay normaliser m = q(nq-q+1)(1-q/nq) already
    # carries the q, so dividing again returns VR/q and reads every series as reverting.
    vr = varq / var1
    # heteroskedasticity-robust variance of VR (Lo-MacKinlay 1988, eq. 18)
    d = (x - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(d[j:] * d[:-j])
        delta = num / (np.sum(d) ** 2 / n) / n if np.sum(d) > 0 else 0.0
        theta += ((2.0 * (q - j) / q) ** 2) * delta
    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else None
    return float(vr), (float(z) if z is not None and np.isfinite(z) else None)


def _sign_permutation_null(form, fwd, idx, n_draws: int, rng) -> np.ndarray:
    """Continuation rates under a permuted FORMATION sign.

    Permuting the sign — not the outcome — is what makes 0.5 the null value. The
    alternative (shuffling the outcome with the selection fixed) asks whether this cell
    differs from the others, and since the pooled rate already contains the persistence it
    scores a genuinely persistent series NEGATIVE. That is PROC-20's corrected bug.
    """
    s = np.sign(form[idx])
    f = np.sign(fwd[idx])
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        out[i] = float((rng.permutation(s) == f).mean())
    return out


def _score(stat: float, draws: np.ndarray):
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None, None, None
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    z = float((stat - mu) / sd) if sd > 0 else 0.0
    pv = float((np.abs(d - mu) >= abs(stat - mu)).sum() + 1) / (d.size + 1)
    return z, pv, mu

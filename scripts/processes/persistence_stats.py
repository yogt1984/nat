"""PROC-20: `persistence_stats` — does momentum persist, and do band excursions revert?

Two classical questions the platform has never measured in the time domain:

  **A. Momentum persistence.** Given `k` consecutive same-sign bar returns, what is
     `P(next return has the same sign)`, and what does the next `h` bars mark out in the
     run's direction? `spectral` answers the frequency-domain version (PSD, spectral-slope
     Hurst, OU half-life) and `trend_hurst_300/600` exists as a *feature*, but nothing here
     has ever produced a run-length distribution or a conditional continuation rate.

  **B. Band excursion.** When price reaches `k·sigma` from a rolling VWAP midline, does it
     revert, how far, and how fast? `research/new/vwap_sd_channel.txt` (LF7) already
     contains this table — from **one day, n = 4–31 per cell**, labelled "PRIORS ONLY". This
     process is how that becomes a study.

**Why this shape.** Both families are *conditional means over selected events*, which is the
most reliable way to manufacture a result: a run-length-5 bucket holds few events, a 3-sigma
band fewer, and somewhere in a k × horizon grid something always looks significant. Three
defences, all imported rather than invented:

  * **per-cell permutation null (PROC-12)** — the outcome is shuffled while the *selection*
    stays fixed, so the reported number is always an excess over what the same buckets yield
    on scrambled outcomes. A random walk must come back at 0.5 with nothing informative.
  * **BH-FDR across the whole grid (PROC-13)** — this is one sweep, corrected as one.
  * **per-day verdicts (PROC-4)** — every cell carries `frac_days_informative` and a
    `durable | non_durable` call, because §4.9's lesson is that day-consistency, not the
    pooled mean, is what fails.

Everything is causal: runs and the rolling midline/sigma use only past bars, markouts use
only future ones, and the event bar itself is never part of its own markout.

**Not a strategy.** The band statistics use the touch price as the fill proxy, which
*overstates* fills — price must trade THROUGH a resting order to fill it (LF7 §3). Any profit
claim goes through the A4 queue simulation (`execution/queue_value.py`); this unit produces
the statistics those parameters are supposed to be read off, nothing more.

Spec: the approved plan for PROC-20. Contract: `docs/contracts/process.md`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from it_engine.null_calibration import load_null_config

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id
from .fdr import DEFAULT_FDR_ALPHA, apply_process_fdr
from .mi_stability import MIStabilityProcess
from .registry import register

BPS = 1e4


@register
class PersistenceStatsProcess(EvaluationProcess):
    """Run-length continuation + band-excursion reversion, null-calibrated per day."""

    data_level = "bars"

    PARAMS = {
        "max_run_length": (5, "report P(continue | run length = 1..K)"),
        "vwap_window": (60, "bars in the rolling midline / sigma window"),
        "k_grid": ([1.0, 1.5, 2.0, 2.5, 3.0], "band multiples to test (LF7's grid)"),
        "embargo_bars": (30, "minimum gap between counted touches"),
        "revert_cap_bars": (None, "bars to wait for a midline revert (default: vwap_window)"),
        "n_shuffles": (None, "pooled permutation draws (default: it_engine.toml)"),
        "day_shuffles": (50, "permutation draws per day fold (cheap: means, not KSG)"),
        "min_events": (30, "minimum events before a cell is estimated"),
        "min_fold_events": (10, "minimum events for a day fold to count"),
        "min_days": (3, "minimum usable days before a durability verdict"),
        "min_frac_informative": (0.6, "day fraction required for 'durable'"),
        "families": (["momentum", "band"], "which families to run"),
        "volume_col": ("flow_volume_1s", "volume for the VWAP midline; falls back to a mean"),
        "seed": (0, "RNG seed"),
    }

    def name(self) -> str:
        return "persistence_stats"

    # ── entry point ──────────────────────────────────────────────────────────────
    def evaluate(self, bars: pd.DataFrame, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params

        days = MIStabilityProcess._day_key(bars)          # PROC-4's calendar folds, reused
        if days is None:
            return result.finalize(
                time.time() - t0,
                error="no time column to split days on — a durability verdict needs to "
                      "know which day a row came from")
        if ctx.price_col not in bars.columns:
            return result.finalize(time.time() - t0,
                                   error=f"price column '{ctx.price_col}' not in data")

        price = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isfinite(price).sum() < 3:
            return result.finalize(time.time() - t0, error="no usable price series")

        cfg = load_null_config()
        gates = {"z": float(cfg["null_z_threshold"]), "i_min": float(cfg["i_min"]),
                 "n_shuffles": int(p["n_shuffles"] or cfg["n_shuffles"])}

        # backward (known-at-t) returns drive the runs; forward returns are the markout
        r_back = np.full(len(price), np.nan)
        r_back[1:] = np.diff(np.log(price))
        fwd = {h: _forward_log_return(price, n) for h, n in ctx.horizons.items()}

        if "momentum" in p["families"]:
            self._momentum_family(result, r_back, fwd, days, gates)
        if "band" in p["families"]:
            self._band_family(result, bars, ctx, price, fwd, days, gates)

        report = apply_process_fdr(result, alpha=DEFAULT_FDR_ALPHA)
        result.finalize(time.time() - t0)
        result.summary.update({
            "fdr": {"alpha": report.alpha, "n_cells": report.n_cells,
                    "n_discoveries": report.n_discoveries},
            "run_length_distribution": _run_length_histogram(
                r_back, int(p["max_run_length"])),
            "n_days": int(len(np.unique(days))),
            "vwap_window": int(p["vwap_window"]),
            "embargo_bars": int(p["embargo_bars"]),
            "fill_proxy_caveat": "band touches use the touch price; real fills require price "
                                 "to trade THROUGH a resting order (worse entry, more "
                                 "adverse). A4 queue sim gates any profit claim.",
        })
        return result

    # ── family A: momentum persistence ───────────────────────────────────────────
    def _momentum_family(self, result, r_back, fwd, days, gates) -> None:
        p = self.params
        sign = np.sign(r_back)
        runs = _run_lengths(sign)
        continued = _continuation(sign)

        for k in range(1, int(p["max_run_length"]) + 1):
            sel = (runs == k) & np.isfinite(continued)
            # NULL: permute the SIGN SERIES and recompute runs. Shuffling the outcome with
            # the selection held fixed would ask "is bucket k unlike the other buckets?" —
            # and since the global continuation rate already contains the persistence, a
            # strongly persistent series scores NEGATIVE against it. The question here is
            # "is there persistence at all", so the serial structure is what must break.
            self._emit(result, sel, continued, days, gates,
                       metric="p_continue_excess", family="momentum",
                       horizon="next_bar", extras={"run_length": k}, centre=0.5,
                       null_fn=_sign_permutation_null(sign, k))
            for h_name, fv in fwd.items():
                signed = sign * fv * BPS            # markout in the run's own direction
                sel_h = (runs == k) & np.isfinite(signed)
                # NULL: permute the forward returns, keep the run state. Breaks the
                # association without changing the return distribution.
                self._emit(result, sel_h, signed, days, gates,
                           metric="markout_bps", family="momentum",
                           horizon=h_name, extras={"run_length": k}, centre=0.0,
                           null_fn=_weighted_permutation_null(sign * BPS, fv))

    # ── family B: band excursion ─────────────────────────────────────────────────
    def _band_family(self, result, bars, ctx, price, fwd, days, gates) -> None:
        p = self.params
        w = int(p["vwap_window"])
        vol_col = p["volume_col"]
        if vol_col in bars.columns:
            vol = bars[vol_col].to_numpy(dtype=np.float64, na_value=np.nan)
            vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
            midline = _rolling_vwap(price, vol, w)
            midline_kind = f"vwap({vol_col})"
        else:
            midline = _rolling_mean(price, w)
            midline_kind = "mean"

        dev = (price - midline) / midline                  # causal by construction
        sigma = _rolling_std(dev, w)
        cap = int(p["revert_cap_bars"] or w)

        for k in p["k_grid"]:
            k = float(k)
            touch = np.isfinite(dev) & np.isfinite(sigma) & (sigma > 0) & \
                (np.abs(dev) >= k * sigma)
            events = _embargo(touch, int(p["embargo_bars"]))
            direction = -np.sign(dev)                      # reverting direction

            for h_name, fv in fwd.items():
                signed = direction * fv * BPS
                sel = events & np.isfinite(signed)
                self._emit(result, sel, signed, days, gates,
                           metric="markout_bps", family="band", horizon=h_name,
                           extras={"k": k, "midline": midline_kind}, centre=0.0,
                           null_fn=_weighted_permutation_null(direction * BPS, fv))

            ttr = _time_to_revert(dev, events, cap)
            sel = events & np.isfinite(ttr)
            self._emit(result, sel, ttr, days, gates,
                       metric="time_to_revert", family="band", horizon="n/a",
                       extras={"k": k, "cap_bars": cap, "midline": midline_kind},
                       centre=None, null_fn=None)

    # ── one cell: pooled stat + null + per-day verdict ───────────────────────────
    def _emit(self, result, sel, values, days, gates, *, metric, family, horizon,
              extras, centre: Optional[float], null_fn) -> None:
        """One grid cell. `null_fn(mask, rng, n_draws)` returns the permutation draws for
        that selection; each family supplies the exchangeability its question requires."""
        p = self.params
        n = int(sel.sum())
        base = {"family": family, "horizon": horizon, "n_events": n,
                "embargo_bars": int(p["embargo_bars"]), **extras}

        if n < int(p["min_events"]):
            result.findings.append(Finding(
                feature=_cell_name(family, metric, extras), horizon=horizon, metric=metric,
                value=0.0, informative=False,
                extras={**base, "verdict": "insufficient_events", "per_day": []}))
            return

        stat = float(np.mean(values[sel]))
        rng = np.random.default_rng(int(p["seed"]))
        z = p_val = null_mean = None
        if null_fn is not None:
            draws = null_fn(sel, rng, gates["n_shuffles"])
            z, p_val, null_mean = _score(stat, draws)

        per_day = []
        for d in np.unique(days[sel]):
            dsel = sel & (days == d)
            if int(dsel.sum()) < int(p["min_fold_events"]):
                continue
            drng = np.random.default_rng(int(p["seed"]) + int(d))
            dval = float(np.mean(values[dsel]))
            dz, dinfo = None, False
            if null_fn is not None:
                ddraws = null_fn(dsel, drng, int(p["day_shuffles"]))
                dz, _, _ = _score(dval, ddraws)
                dinfo = bool(dz is not None and abs(dz) >= gates["z"])
            per_day.append({"day": int(d), "n_events": int(dsel.sum()),
                            "value": round(dval, 6), "z": dz, "informative": dinfo})

        frac = float(np.mean([d["informative"] for d in per_day])) if per_day else 0.0
        slope = 0.0
        if len(per_day) > 1:
            slope = float(np.polyfit(np.arange(len(per_day), dtype=float),
                                     [d["value"] for d in per_day], 1)[0])
        if len(per_day) < int(p["min_days"]):
            verdict = "insufficient_days"
        elif frac >= float(p["min_frac_informative"]):
            verdict = "durable"
        else:
            verdict = "non_durable"

        excess = stat - centre if centre is not None else stat
        result.findings.append(Finding(
            feature=_cell_name(family, metric, extras), horizon=horizon, metric=metric,
            value=round(float(excess), 6),
            p_value=None if p_val is None else round(p_val, 6),
            threshold=gates["z"],
            informative=bool(z is not None and abs(z) >= gates["z"] and verdict == "durable"),
            extras={**base,
                    "p_continue": round(stat, 6) if metric == "p_continue_excess" else None,
                    "pooled_value": round(stat, 6),
                    "z": None if z is None else round(z, 3),
                    "null_mean": None if null_mean is None else round(null_mean, 6),
                    "frac_days_informative": round(frac, 4),
                    "n_days": len(per_day), "slope_per_day": round(slope, 6),
                    "verdict": verdict, "per_day": per_day}))


# ── helpers ──────────────────────────────────────────────────────────────────────
def _score(stat: float, draws: np.ndarray):
    """(z, one-sided empirical p, null mean) for a statistic against permutation draws."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None, None, None
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    z = float((stat - mu) / sd) if sd > 0 else 0.0
    p = float((np.abs(d - mu) >= abs(stat - mu)).sum() + 1) / (d.size + 1)   # two-sided
    return z, p, mu


def _continuation(sign: np.ndarray) -> np.ndarray:
    """1 where the next bar keeps the sign, 0 where it flips, NaN where undefined."""
    out = np.full(len(sign), np.nan)
    if len(sign) > 1:
        out[:-1] = (sign[1:] == sign[:-1]).astype(float)
    out[~np.isfinite(sign)] = np.nan
    return out


def _sign_permutation_null(sign: np.ndarray, k: int):
    """Null for P(continue | run = k): permute the SIGN SERIES and recompute the runs.

    Holding the selection fixed and shuffling the outcome would test a different
    hypothesis ("is bucket k unlike the average bucket"), and because the pooled
    continuation rate already contains the persistence, that null makes a genuinely
    persistent series score negative. Breaking the serial structure is what makes 0.5 the
    null value, which is the question actually being asked.
    """
    def draws(mask: np.ndarray, rng: np.random.Generator, n_draws: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.array([])
        lo, hi = int(idx.min()), int(idx.max()) + 1      # permute within the cell's span
        base = sign[lo:hi]
        finite = base[np.isfinite(base)]
        out = np.full(n_draws, np.nan)
        for i in range(n_draws):
            s = np.full(len(base), np.nan)
            s[np.isfinite(base)] = rng.permutation(finite)
            r = _run_lengths(s)
            c = _continuation(s)
            m = (r == k) & np.isfinite(c)
            if m.any():
                out[i] = float(c[m].mean())
        return out

    return draws


def _weighted_permutation_null(weight: np.ndarray, outcome: np.ndarray):
    """Null for E[weight x outcome | selection]: permute the OUTCOME, keep the weight.

    Shuffling the already-multiplied product would preserve its global mean, which under
    genuine persistence is non-zero — so the null would absorb the very effect being
    tested. Permuting the raw forward return and re-applying the (unshuffled) direction
    breaks the association and leaves the null centred on zero.
    """
    def draws(mask: np.ndarray, rng: np.random.Generator, n_draws: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.array([])
        w = weight[idx]
        pool = outcome[np.isfinite(outcome)]
        if pool.size == 0:
            return np.array([])
        out = np.full(n_draws, np.nan)
        for i in range(n_draws):
            sampled = rng.choice(pool, size=idx.size, replace=False if pool.size >= idx.size else True)
            v = w * sampled
            v = v[np.isfinite(v)]
            if v.size:
                out[i] = float(v.mean())
        return out

    return draws


def _forward_log_return(price: np.ndarray, n: int) -> np.ndarray:
    """r(t) = log p(t+n) - log p(t). The event bar is never part of its own markout."""
    out = np.full(len(price), np.nan)
    if n < 1 or n >= len(price):
        return out
    lp = np.log(price)
    out[:-n] = lp[n:] - lp[:-n]
    return out


def _run_lengths(sign: np.ndarray) -> np.ndarray:
    """Causal count of consecutive identical signs ending at t (0 where undefined)."""
    out = np.zeros(len(sign), dtype=np.int64)
    run = 0
    prev = np.nan
    for t, s in enumerate(sign):
        if not np.isfinite(s) or s == 0:
            run, prev = 0, np.nan
            out[t] = 0
            continue
        run = run + 1 if s == prev else 1
        prev = s
        out[t] = run
    return out


def _run_length_histogram(r_back: np.ndarray, max_k: int) -> dict:
    runs = _run_lengths(np.sign(r_back))
    return {str(k): int((runs == k).sum()) for k in range(1, max_k + 1)}


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(w, min_periods=w).mean().to_numpy()


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(w, min_periods=w).std(ddof=1).to_numpy()


def _rolling_vwap(price: np.ndarray, vol: np.ndarray, w: int) -> np.ndarray:
    pv = pd.Series(price * vol).rolling(w, min_periods=w).sum().to_numpy()
    vv = pd.Series(vol).rolling(w, min_periods=w).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pv / vv
    return np.where(np.isfinite(out), out, _rolling_mean(price, w))


def _embargo(mask: np.ndarray, gap: int) -> np.ndarray:
    """Keep the first touch, then suppress everything within `gap` bars of it."""
    out = np.zeros(len(mask), dtype=bool)
    last = -(gap + 1)
    for t in np.flatnonzero(mask):
        if t - last > gap:
            out[t] = True
            last = t
    return out


def _time_to_revert(dev: np.ndarray, events: np.ndarray, cap: int) -> np.ndarray:
    """Bars from each event until the deviation first crosses the midline (NaN if never)."""
    out = np.full(len(dev), np.nan)
    n = len(dev)
    for t in np.flatnonzero(events):
        s0 = np.sign(dev[t])
        stop = min(n, t + cap + 1)
        for u in range(t + 1, stop):
            if np.isfinite(dev[u]) and np.sign(dev[u]) != s0:
                out[t] = u - t
                break
    return out


def _cell_name(family: str, metric: str, extras: dict) -> str:
    if family == "momentum":
        return f"momentum_run{extras['run_length']}_{metric}"
    return f"band_k{extras['k']:g}_{metric}"

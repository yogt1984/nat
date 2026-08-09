"""VW-2: which multi-scale VWAP windows earn a feature column?

The ingestor ships one VWAP (5 s). `specs/multiscale_vwap.md` proposes six slow windows —
a 12-column schema migration — and requires the answer to be measured offline first. This
driver is that measurement: VW-1's bucketed anchor over the `data/trades/` archive, judged
by the spec's five pre-registered criteria, every statistic imported from an existing
process rather than written here.

**Anchor path.** Trades are aggregated to per-minute (notional, volume, close) and fed to
the same `VwapRing` VW-1 ships — one pseudo-trade per active minute with price
`notional/volume` and size `volume`, which contributes exactly the same sums as the raw
stream. Warm-up and gap refusal are therefore VW-1's, not a re-implementation, and
`tests/test_vwap_multiscale_study.py` pins the minute path to the per-trade path.

**Criteria (spec §A2), and where each number comes from:**

- (a) informative vs permutation null, z >= 3 at >= 1 horizon — PROC-4 (`mi_stability`)
  per-day nulls (PROC-12), aggregated across days by Stouffer's method (the per-day z's
  are the unit; the combination is standard meta-analysis, not a new statistic);
- (b) `frac_days_informative` >= 0.55 — PROC-4's own day-consistency series;
- (c) survives BH-FDR across the (window x horizon x symbol) grid — PROC-13's
  `benjamini_hochberg` over the Stouffer p's, one sweep corrected as one;
- (d) not redundant with the next-faster window. The spec's literal quantity —
  holdout |corr(residual, faster)| — is degenerate for the case it exists to catch: an
  exact duplicate residualizes to zero, whose correlation with anything is ~0, a false
  PASS. The planted duplicate test demands the criterion fail there, so the study gates
  on holdout |corr(slow_dev, fast_dev)| < 0.5 (the fit/holdout split is PROC-15's), and
  records PROC-15's residual finding alongside for the record;
- (e) band-touch event rate high enough that PROC-20's per-day verdict is reachable —
  >= `min_fold_events` events/day, imported from `PersistenceStatsProcess.PARAMS`
  (PROC-20's ~1.5/day was the counter-example: a cell that can never earn a day fold).

**Windows.** 6h/12h are excluded by default: VW-1's smoke found the trade feed's holes
(e.g. 2026-08-07 BTC: 49.7% active minutes, a 586-minute hole) leave the 12h window 0%
available and 6h at 6.4% — a verdict on them would be a verdict on the gaps. They rejoin
via --windows once the streak holds.

Roll (1984) is the null for any "oscillation": bounce alone produces reversion with
amplitude ~ the spread, so the amplitude-to-spread ratio is reported per (symbol, window)
— below ~1 there is no edge by construction, whatever the MI says.

Artifact: `reports/vwap_multiscale_study.json`. Read-only; no capital path.

Usage:  python -m exploration.vwap_multiscale_study [--symbols BTC ETH SOL] [--days N]
            [--n-shuffles N] [--workers 3] [--windows 5m 10m 15m 1h]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

warnings.filterwarnings("ignore")

from features.vwap_multiscale import (  # noqa: E402
    DEFAULT_MAX_GAP_MINUTES, NS_PER_MIN, WINDOWS, VwapRing,
)
from processes.base import ProcessContext, ProcessResult, get_provenance  # noqa: E402
from processes.mi_stability import MIStabilityProcess  # noqa: E402
from processes.persistence_stats import PersistenceStatsProcess  # noqa: E402
from processes.residualize import RES_PREFIX, ResidualizeProcess  # noqa: E402
from alpha.screener import benjamini_hochberg  # noqa: E402

TRADES_DIR = Path(__file__).resolve().parents[2] / "data" / "trades"
FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "vwap_multiscale_study.json"

#: The windows under study: every VW-1 candidate except the unmeasurable slow pair,
#: so a window added to VW-1 (the 30m/2h crossover bracket) joins automatically.
EXCLUDED_WINDOWS: dict[str, int] = {k: v for k, v in WINDOWS.items()
                                    if k in ("6h", "12h")}
STUDY_WINDOWS: dict[str, int] = {k: v for k, v in WINDOWS.items()
                                 if k not in EXCLUDED_WINDOWS}
EXCLUSION_REASON = (
    "trade-feed holes make the slow windows unmeasurable: VW-1 smoke on 2026-08-07 BTC "
    "found 49.7% active minutes and a 586-minute hole -> 12h window 0% available, 6h "
    "6.4%. A verdict on 6h/12h would be a verdict on the gaps; they wait for the streak."
)

#: Criterion (e): events/day needed for PROC-20's per-day verdict to be reachable.
#: Imported from the process, not chosen here (gates imported, not invented).
MIN_EVENTS_PER_DAY: int = int(PersistenceStatsProcess.PARAMS["min_fold_events"][0])

#: Markout horizons, in 1-minute bars.
HORIZONS = {"5m": 5, "15m": 15, "1h": 60}

#: Spec §A2 gates.
Z_GATE = 3.0
FRAC_GATE = 0.55
CORR_GATE = 0.5
FDR_ALPHA = 0.05
K_REF = 2.0                     # LF7's band region is k ~ 2.0-2.5; events judged at 2.0


# ── anchor path: trades -> minutes -> VW-1 ring ──────────────────────────────────
def aggregate_minutes(trades: pd.DataFrame) -> pd.DataFrame:
    """Single-symbol trades -> one row per ACTIVE minute: notional, volume, close.

    Sums decompose, so feeding these to `VwapRing` as one pseudo-trade per minute
    (price = notional/volume, size = volume) reproduces the per-trade sums exactly —
    the property the equivalence test pins.
    """
    t = trades[trades["size"] > 0]
    if len(t) == 0:
        return pd.DataFrame(columns=["minute", "timestamp_ns", "notional", "volume",
                                     "close", "last_ts"])
    t = t.sort_values("timestamp_ns", kind="stable")
    minute = t["timestamp_ns"].to_numpy(dtype=np.int64) // NS_PER_MIN
    g = pd.DataFrame({
        "minute": minute,
        "notional": t["price"].to_numpy(dtype=np.float64)
        * t["size"].to_numpy(dtype=np.float64),
        "volume": t["size"].to_numpy(dtype=np.float64),
        "close": t["price"].to_numpy(dtype=np.float64),
        "last_ts": t["timestamp_ns"].to_numpy(dtype=np.int64),
    }).groupby("minute", sort=True).agg(
        notional=("notional", "sum"), volume=("volume", "sum"),
        close=("close", "last"), last_ts=("last_ts", "max"),
    ).reset_index()
    g["timestamp_ns"] = g["minute"] * NS_PER_MIN
    return g[["minute", "timestamp_ns", "notional", "volume", "close", "last_ts"]]


def combine_minutes(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-file minute aggregates (a minute may straddle files: sums add,
    close follows the latest trade)."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["minute", "timestamp_ns", "notional", "volume",
                                     "close", "last_ts"])
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["minute", "last_ts"], kind="stable")
    g = df.groupby("minute", sort=True).agg(
        notional=("notional", "sum"), volume=("volume", "sum"),
        close=("close", "last"), last_ts=("last_ts", "max"),
    ).reset_index()
    g["timestamp_ns"] = g["minute"] * NS_PER_MIN
    return g[["minute", "timestamp_ns", "notional", "volume", "close", "last_ts"]]


def attach_window_vwaps(minutes: pd.DataFrame,
                        windows: Optional[dict[str, int]] = None,
                        max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES) -> pd.DataFrame:
    """Add `vwap_{w}` / `vwap_dev_{w}` per window, via VW-1's own ring.

    Deviation is against the minute close — the same reference the per-trade path
    uses at the minute's last trade, which is what makes the two paths comparable.
    """
    windows = windows or STUDY_WINDOWS
    out = minutes.copy()
    n = len(out)
    vw = {w: np.full(n, np.nan) for w in windows}
    if n:
        ring = VwapRing(max_minutes=max(windows.values()),
                        max_gap_minutes=max_gap_minutes)
        mins = out["minute"].to_numpy(dtype=np.int64)
        notional = out["notional"].to_numpy(dtype=np.float64)
        volume = out["volume"].to_numpy(dtype=np.float64)
        for i in range(n):
            ring.add(int(mins[i]) * NS_PER_MIN,
                     notional[i] / volume[i], volume[i])
            for name, w in windows.items():
                vw[name][i] = ring.vwap(w)
    close = out["close"].to_numpy(dtype=np.float64)
    for name in windows:
        v = vw[name]
        out[f"vwap_{name}"] = v
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"vwap_dev_{name}"] = np.where(
                np.isfinite(v) & (v > 0), (close - v) / v, np.nan)
    return out


# ── process wrappers (all verdict logic stays inside the processes) ──────────────
def mi_cells(bars: pd.DataFrame, symbol: str, horizons: dict[str, int],
             n_shuffles: Optional[int] = None, min_fold_obs: int = 64,
             max_samples: int = 1440, seed: int = 0,
             min_days: int = 1) -> ProcessResult:
    """PROC-4 over the `vwap_dev_*` columns — with NON-OVERLAPPING targets.

    Each horizon h evaluates a frame strided by h rows, so consecutive rows carry
    disjoint forward windows. Dense minute rows with a 1h target overlap 60x, and
    PROC-12's permutation null assumes exchangeable rows — the first smoke of this
    study produced z ~ 50-70 rising with window length, the same mechanism as §5's
    phantom-IC incident (overlapping windows, IC 0.39-0.46 out of nothing). Striding
    is the record's own defense, applied as data prep; the estimator is untouched.

    The cost is honest sample size: 1440/h rows per day, so a day fold supports a 5m
    horizon (288) and 15m (96, with min_fold_obs = 64 ~ 13x the KSG k), but a 1h
    horizon leaves 24 rows — below any defensible KSG fold, so those cells are
    refused by PROC-4 and the refusal is recorded, not averaged over. Horizons count
    ACTIVE minutes: across a feed hole the wall-clock span is longer, and rows whose
    window refused (NaN dev) fall out of the valid mask.
    """
    merged: Optional[ProcessResult] = None
    for hname, h in horizons.items():
        strided = bars.iloc[::max(1, int(h))].reset_index(drop=True)
        ctx = ProcessContext(symbol=symbol, timeframe="1m", price_col="close",
                             horizons={hname: 1}, costs={})
        proc = MIStabilityProcess(features=["vwap_dev_"], n_shuffles=n_shuffles,
                                  min_fold_obs=min_fold_obs, max_samples=max_samples,
                                  seed=seed, min_days=min_days)
        res = proc.evaluate(strided, ctx)
        if merged is None:
            merged = res
            merged.summary = {"targets": "non-overlapping (stride = horizon)",
                              "horizons": {}}
        else:
            merged.findings.extend(res.findings)
        merged.summary["horizons"][hname] = {
            "stride": int(h), "rows": len(strided),
            "n_days_used": res.summary.get("n_days_used"),
            "folds_skipped": res.summary.get("folds_skipped"),
            "error": res.summary.get("error"),
        }
    return merged


def band_result(bars: pd.DataFrame, symbol: str, window_minutes: int,
                horizons: dict[str, int], n_shuffles: Optional[int] = None,
                seed: int = 0) -> ProcessResult:
    """PROC-20 band family with the midline window set to the candidate window."""
    ctx = ProcessContext(symbol=symbol, timeframe="1m", price_col="close",
                         horizons=horizons, costs={})
    proc = PersistenceStatsProcess(families=["band"], vwap_window=window_minutes,
                                   volume_col="volume", n_shuffles=n_shuffles,
                                   seed=seed)
    return proc.evaluate(bars, ctx)


def redundancy_rows(bars: pd.DataFrame, symbol: str,
                    windows: Optional[dict[str, int]] = None,
                    fit_frac: float = 0.7) -> list[dict]:
    """Criterion (d) per window vs its next-faster sibling.

    Gate quantity: holdout |corr(slow_dev, fast_dev)| — see the module docstring for
    why the spec's literal residual-correlation is degenerate on an exact duplicate.
    PROC-15's residual finding is run and recorded alongside.
    """
    windows = windows or STUDY_WINDOWS
    ordered = sorted(windows.items(), key=lambda kv: kv[1])
    rows: list[dict] = [{"window": ordered[0][0], "symbol": symbol,
                         "holdout_abs_corr": None, "conditioning": None,
                         "res_holdout_abs_corr": None, "r2_fit": None}]
    n = len(bars)
    cut = max(1, int(n * fit_frac))
    ctx = ProcessContext(symbol=symbol, timeframe="1m", price_col="close",
                         horizons={}, costs={})
    for (fast, _), (slow, _) in zip(ordered, ordered[1:]):
        f_col, s_col = f"vwap_dev_{fast}", f"vwap_dev_{slow}"
        row = {"window": slow, "symbol": symbol, "conditioning": f_col,
               "holdout_abs_corr": None, "res_holdout_abs_corr": None, "r2_fit": None}
        s = bars[s_col].to_numpy(dtype=np.float64, na_value=np.nan)
        f = bars[f_col].to_numpy(dtype=np.float64, na_value=np.nan)
        hold = np.zeros(n, dtype=bool)
        hold[cut:] = True
        m = hold & np.isfinite(s) & np.isfinite(f)
        if int(m.sum()) > 10 and np.std(s[m]) > 1e-15 and np.std(f[m]) > 1e-15:
            row["holdout_abs_corr"] = abs(float(np.corrcoef(s[m], f[m])[0, 1]))
        # recorded, not gated: redundancy vs the FASTEST window, for the case where
        # the chain criterion kills every nested sibling but a slow window still
        # carries an axis the 5m column does not
        fastest = bars[f"vwap_dev_{ordered[0][0]}"].to_numpy(
            dtype=np.float64, na_value=np.nan)
        mf = hold & np.isfinite(s) & np.isfinite(fastest)
        if int(mf.sum()) > 10 and np.std(s[mf]) > 1e-15 \
                and np.std(fastest[mf]) > 1e-15:
            row["holdout_abs_corr_vs_fastest"] = abs(
                float(np.corrcoef(s[mf], fastest[mf])[0, 1]))
        proc = ResidualizeProcess(features=[s_col], conditioning=[f_col],
                                  fit_frac=fit_frac)
        _, res = proc.transform(bars, ctx)
        for fd in res.findings:
            if fd.feature == RES_PREFIX + s_col:
                row["res_holdout_abs_corr"] = fd.value
                row["r2_fit"] = fd.extras.get("r2_fit")
        rows.append(row)
    return rows


# ── aggregation & criteria ───────────────────────────────────────────────────────
def stouffer(zs) -> tuple[float, float]:
    """Combine per-day z's: Z = sum(z)/sqrt(n); two-sided normal p."""
    z = np.asarray([v for v in zs if v is not None and np.isfinite(v)],
                   dtype=np.float64)
    if z.size == 0:
        return float("nan"), float("nan")
    combined = float(z.sum() / math.sqrt(z.size))
    p = float(math.erfc(abs(combined) / math.sqrt(2.0)))
    return combined, p


def mi_rows_from_result(res: ProcessResult, symbol: str) -> list[dict]:
    """One row per (window, horizon) cell: Stouffer over PROC-4's per-day z's."""
    rows = []
    for f in res.findings:
        if not f.feature.startswith("vwap_dev_"):
            continue
        per_day = f.extras.get("per_day", [])
        z, p = stouffer([d.get("z") for d in per_day])
        rows.append({
            "window": f.feature.removeprefix("vwap_dev_"),
            "symbol": symbol, "horizon": f.horizon,
            "stouffer_z": None if np.isnan(z) else round(z, 3),
            "stouffer_p": None if np.isnan(p) else p,
            "frac_days_informative": f.extras.get("frac_days_informative"),
            "n_days": f.extras.get("n_days"),
            "mean_bits_above_null": f.extras.get("mean_bits_above_null"),
            "verdict": f.extras.get("verdict"),
        })
    return rows


def band_rows_from_result(res: ProcessResult, window: str, symbol: str) -> list[dict]:
    """Events/day per k, from PROC-20's band cells."""
    n_days = max(1, int(res.summary.get("n_days", 1)))
    seen: dict[float, int] = {}
    for f in res.findings:
        k = f.extras.get("k")
        if k is None or f.extras.get("family") != "band":
            continue
        seen[float(k)] = max(seen.get(float(k), 0), int(f.extras.get("n_events", 0)))
    return [{"window": window, "symbol": symbol, "k": k,
             "n_events": n, "events_per_day": round(n / n_days, 2)}
            for k, n in sorted(seen.items())]


def evaluate_criteria(mi_rows: list[dict], red_rows: list[dict],
                      band_rows: list[dict], alpha: float = FDR_ALPHA,
                      z_gate: float = Z_GATE, frac_gate: float = FRAC_GATE,
                      corr_gate: float = CORR_GATE, k_ref: float = K_REF,
                      min_events_per_day: float = MIN_EVENTS_PER_DAY) -> dict:
    """Apply the spec's five gates; a window passes iff some (symbol, horizon) cell
    passes (a) AND (b) AND (c) simultaneously and the (window, symbol) passes (d), (e).

    BH-FDR runs over the WHOLE grid handed in — one sweep, corrected as one.
    """
    pvals = np.array([r["stouffer_p"] if r.get("stouffer_p") is not None else np.nan
                      for r in mi_rows], dtype=np.float64)
    qvals = benjamini_hochberg(pvals, alpha=alpha) if len(pvals) else np.array([])
    for r, q in zip(mi_rows, qvals):
        r["q_value"] = None if np.isnan(q) else float(q)

    red = {(r["window"], r["symbol"]): r for r in red_rows}
    events = {(r["window"], r["symbol"]): r for r in band_rows
              if float(r.get("k", -1)) == float(k_ref)}

    pairs: dict[tuple, list[dict]] = {}
    for r in mi_rows:
        pairs.setdefault((r["window"], r["symbol"]), []).append(r)

    out: dict[str, dict] = {}
    for (window, symbol), cells in sorted(pairs.items()):
        def _z(c):
            return c["stouffer_z"] if c.get("stouffer_z") is not None else -np.inf
        a = any(_z(c) >= z_gate for c in cells)
        b = any((c.get("frac_days_informative") or 0.0) >= frac_gate for c in cells)
        c_flag = any(c.get("q_value") is not None and c["q_value"] <= alpha
                     and _z(c) >= z_gate for c in cells)
        joint = [c for c in cells
                 if _z(c) >= z_gate
                 and (c.get("frac_days_informative") or 0.0) >= frac_gate
                 and c.get("q_value") is not None and c["q_value"] <= alpha]
        rrow = red.get((window, symbol))
        corr = rrow.get("holdout_abs_corr") if rrow else None
        d = corr is None or corr < corr_gate
        erow = events.get((window, symbol))
        e = bool(erow and erow.get("events_per_day", 0.0) >= min_events_per_day)
        best = max(cells, key=_z)
        sym_pass = bool(joint) and d and e
        out.setdefault(window, {"pass": False, "per_symbol": {}})
        out[window]["per_symbol"][symbol] = {
            "a": bool(a), "b": bool(b), "c": bool(c_flag), "d": bool(d), "e": bool(e),
            "pass": sym_pass,
            "best_cell": {"horizon": best["horizon"], "z": best.get("stouffer_z"),
                          "q": best.get("q_value"),
                          "frac_days": best.get("frac_days_informative")},
            "holdout_abs_corr_vs_faster": corr,
            "events_per_day_at_k2": erow.get("events_per_day") if erow else None,
        }
        out[window]["pass"] = out[window]["pass"] or sym_pass
    return out


# ── amplitude-to-spread (the Roll-bounce report) ─────────────────────────────────
def amplitude_to_spread(dev: pd.Series, window: int, spread_bps: float,
                        k: float = K_REF) -> dict:
    """Median |deviation| in bps at k-sigma touches, over the median spread.

    Below ~1 the 'oscillation' is bid-ask bounce and there is no edge by
    construction (Roll 1984), whatever the information criteria say.
    """
    d = pd.Series(np.asarray(dev, dtype=np.float64))
    sigma = d.rolling(window, min_periods=window).std(ddof=1)
    touch = d.abs().ge(k * sigma) & sigma.gt(0) & d.notna()
    n = int(touch.sum())
    amp = float(d[touch].abs().median() * 1e4) if n else float("nan")
    ratio = amp / spread_bps if (n and np.isfinite(spread_bps) and spread_bps > 0) \
        else float("nan")
    return {"n_touches": n, "amplitude_bps": amp,
            "spread_bps": spread_bps, "ratio": ratio, "k": k}


def sample_spread_bps(symbol: str, days: list[str], files_per_day: int = 2,
                      max_days: int = 10) -> float:
    """Median `raw_spread_bps` from a sample of the features archive (NaN if absent)."""
    vals: list[float] = []
    for day in days[-max_days:]:
        d = FEATURES_DIR / day
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.parquet"))[:files_per_day]:
            try:
                df = pd.read_parquet(f, columns=["symbol", "raw_spread_bps"])
            except Exception:
                continue
            v = df.loc[df["symbol"] == symbol, "raw_spread_bps"].to_numpy(
                dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(float(np.median(v)))
    return float(np.median(vals)) if vals else float("nan")


# ── driver ───────────────────────────────────────────────────────────────────────
def _symbol_episode(task) -> dict:
    """Everything for one symbol: load -> anchor -> PROC-4 / PROC-20 / (d) / ratio."""
    symbol, days, windows, n_shuffles = task
    parts = []
    n_files = 0
    for day in days:
        ddir = TRADES_DIR / day
        if not ddir.is_dir():
            continue
        for f in sorted(ddir.glob("*.parquet")):
            try:
                df = pd.read_parquet(
                    f, columns=["timestamp_ns", "symbol", "price", "size"])
            except Exception:
                continue
            n_files += 1
            sub = df[df["symbol"] == symbol]
            if len(sub):
                parts.append(aggregate_minutes(sub))
    minutes = combine_minutes(parts)
    if len(minutes) < 2 * max(windows.values()):
        return {"symbol": symbol, "error": f"only {len(minutes)} active minutes"}

    bars = attach_window_vwaps(minutes, windows=windows)
    mi = mi_cells(bars, symbol, HORIZONS, n_shuffles=n_shuffles, min_days=3)
    bands = {}
    for name, w in windows.items():
        bands[name] = band_result(bars, symbol, w, HORIZONS, n_shuffles=n_shuffles)
    red = redundancy_rows(bars, symbol, windows=windows)

    spread = sample_spread_bps(symbol, days)
    ratios = {name: amplitude_to_spread(bars[f"vwap_dev_{name}"], w, spread)
              for name, w in windows.items()}

    active_frac = round(
        len(minutes) / max(1, int(minutes["minute"].iloc[-1]
                                  - minutes["minute"].iloc[0] + 1)), 4)
    return {
        "symbol": symbol,
        "n_files": n_files,
        "n_active_minutes": len(minutes),
        "active_minute_frac": active_frac,
        "mi_rows": mi_rows_from_result(mi, symbol),
        "mi_summary": mi.summary,
        "band_rows": [r for name, b in bands.items()
                      for r in band_rows_from_result(b, name, symbol)],
        "band_fdr": {name: b.summary.get("fdr") for name, b in bands.items()},
        "red_rows": red,
        "amplitude_to_spread": ratios,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--days", type=int, default=0, help="last N days (0 = all)")
    ap.add_argument("--n-shuffles", type=int, default=None,
                    help="permutation draws (default: config/it_engine.toml)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--windows", nargs="+", default=list(STUDY_WINDOWS),
                    choices=list(WINDOWS))
    args = ap.parse_args(argv)

    windows = {k: WINDOWS[k] for k in args.windows}
    days = sorted(d.name for d in TRADES_DIR.iterdir() if d.is_dir())
    if args.days:
        days = days[-args.days:]
    print(f"VW-2 study · {len(days)} days × {args.symbols} · windows={list(windows)}\n"
          f"excluded={list(EXCLUDED_WINDOWS)} ({EXCLUSION_REASON[:60]}…)\n"
          f"gates: z>={Z_GATE} frac>={FRAC_GATE} BH q<={FDR_ALPHA} "
          f"|corr|<{CORR_GATE} events/day>={MIN_EVENTS_PER_DAY} (from PROC-20)",
          flush=True)

    tasks = [(s, days, windows, args.n_shuffles) for s in args.symbols]
    from concurrent.futures import ProcessPoolExecutor
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ep in pool.map(_symbol_episode, tasks):
            episodes.append(ep)
            print(f"  {ep['symbol']}: "
                  + (ep.get("error") or f"{ep['n_active_minutes']} active minutes "
                     f"({ep['active_minute_frac']:.0%}), "
                     f"{len(ep['mi_rows'])} MI cells"), flush=True)

    ok = [e for e in episodes if "error" not in e]
    mi_rows = [r for e in ok for r in e["mi_rows"]]
    red_rows = [r for e in ok for r in e["red_rows"]]
    band_rows = [r for e in ok for r in e["band_rows"]]
    verdicts = evaluate_criteria(mi_rows, red_rows, band_rows)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "spec": "docs/specs/multiscale_vwap.md §A2",
        "provenance": get_provenance(),
        "windows": list(windows),
        "excluded_windows": {"names": list(EXCLUDED_WINDOWS),
                             "reason": EXCLUSION_REASON},
        "gates": {"z": Z_GATE, "frac_days": FRAC_GATE, "fdr_alpha": FDR_ALPHA,
                  "corr": CORR_GATE, "events_per_day": MIN_EVENTS_PER_DAY,
                  "events_source": "PersistenceStatsProcess.PARAMS.min_fold_events",
                  "k_ref": K_REF},
        "horizons": HORIZONS, "days": days, "symbols": args.symbols,
        "n_shuffles": args.n_shuffles,
        "criterion_d_note": "gated on holdout |corr(slow,fast)|; spec's literal "
                            "residual-corr recorded per row as res_holdout_abs_corr "
                            "(degenerate on exact duplicates — see module docstring)",
        "verdicts": verdicts,
        "episodes": episodes,
    }, indent=1, default=float))
    print(f"\nartifact: {REPORT}\n")

    print(f"{'window':<8}{'symbol':<8}{'a':>3}{'b':>3}{'c':>3}{'d':>3}{'e':>3}"
          f"{'z_best':>8}{'q':>10}{'frac':>6}{'ev/day':>8}{'amp/spread':>12}  pass")
    for window, v in sorted(verdicts.items(), key=lambda kv: WINDOWS[kv[0]]):
        for symbol, s in sorted(v["per_symbol"].items()):
            ep = next((e for e in ok if e["symbol"] == symbol), {})
            ratio = (ep.get("amplitude_to_spread", {}).get(window) or {}).get("ratio")
            bc = s["best_cell"]
            print(f"{window:<8}{symbol:<8}"
                  + "".join(f"{'Y' if s[c] else '.':>3}" for c in "abcde")
                  + f"{(bc['z'] if bc['z'] is not None else float('nan')):>8.2f}"
                  f"{(bc['q'] if bc['q'] is not None else float('nan')):>10.2g}"
                  f"{(bc['frac_days'] if bc['frac_days'] is not None else float('nan')):>6.2f}"
                  f"{(s['events_per_day_at_k2'] if s['events_per_day_at_k2'] is not None else float('nan')):>8.1f}"
                  f"{(ratio if ratio is not None else float('nan')):>12.2f}"
                  f"  {'PASS' if s['pass'] else '—'}")
        print(f"{window:<8}{'=> ' + ('EARNS A COLUMN' if v['pass'] else 'no'):<40}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

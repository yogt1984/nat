"""TC-1: does trend continue at 15m-1h, where the record is silent?

The record refutes continuation where it has looked: 1m/5m momentum is anti-persistent
(§5, 34/36 cells negative) and the daily cross-section REVERTS (XS-3, rank-IC −0.039,
z −4.56). Between those sits an untested band — 15m and 1h — and the XS-1 fetcher can
pull the venue's full history for it (1h reaches ~90 days, 15m ~60).

**Engine: PROC-20's momentum family per (pair, interval), nothing new.** Run-length
continuation with the sign-permutation null (the serial structure is what breaks, so a
random walk scores 0), per-day durability verdicts, within-run FDR — all imported.

**The gate cell is next-bar continuation (`p_continue_excess`).** It is non-overlapping
by construction. Multi-bar markouts overlap across rows and §7.12 showed exactly what
that does to a permutation null on this platform (z = 50-70 out of nothing), so markouts
are recorded as descriptive context and never gate.

**One sweep, corrected as one.** BH-FDR runs across the whole (pair × interval × k)
grid of gate cells (PROC-13), and the sweep is recorded to the program-level FDR ledger
— 177 pairs is 177 chances for a great-looking argmax.

Artifact: `reports/trend_continuation_study.json`. Read-only; no capital path.

Usage:  python -m exploration.trend_continuation_study [--intervals 15m 1h]
            [--max-symbols N] [--n-shuffles N] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
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

from processes.base import ProcessContext, ProcessResult, get_provenance  # noqa: E402
from processes.candles import DEFAULT_CANDLE_DIR, available_candle_symbols  # noqa: E402
from processes.fdr import (  # noqa: E402
    DEFAULT_FDR_ALPHA, FdrReport, apply_process_fdr, default_ledger_path,
    record_sweep,
)
from processes.persistence_stats import PersistenceStatsProcess  # noqa: E402

REPORT = Path(__file__).resolve().parents[2] / "reports" / "trend_continuation_study.json"

#: The one metric allowed to gate: next-bar continuation, non-overlapping by
#: construction. Markouts overlap and are descriptive only (§7.12's lesson).
GATE_METRIC = "p_continue_excess"

#: Descriptive markout horizons per interval, in bars.
INTERVALS: dict[str, dict] = {
    "15m": {"horizons": {"4bar": 4, "16bar": 16}},     # 1 h and 4 h ahead
    "1h": {"horizons": {"4bar": 4, "24bar": 24}},      # 4 h and 1 d ahead
}

MIN_BARS = 500          # below this a per-day durability verdict is unreachable


def continuation_result(candles: pd.DataFrame, symbol: str, interval: str,
                        n_shuffles: Optional[int] = None, max_run_length: int = 5,
                        day_shuffles: int = 50, seed: int = 0,
                        horizons: Optional[dict[str, int]] = None) -> ProcessResult:
    """PROC-20 momentum family on one pair's candles.

    The archive's `timestamp` is tz-aware; PROC-4's day-splitter needs a numeric
    column, so `timestamp_ns` is derived here — the frame refusal ("cannot say which
    day a row is from") is for frames without time, not for this schema.
    """
    df = candles.copy()
    if "timestamp_ns" not in df.columns:
        # .as_unit("ns"): a us-resolution timestamp cast straight to int64 would put
        # every row on epoch-day ~20 and collapse the calendar folds to one.
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_ns"] = ts.dt.as_unit("ns").astype("int64")
    horizons = horizons or INTERVALS[interval]["horizons"]
    ctx = ProcessContext(symbol=symbol, timeframe=interval, price_col="close",
                         horizons=horizons, costs={})
    proc = PersistenceStatsProcess(families=["momentum"], max_run_length=max_run_length,
                                   n_shuffles=n_shuffles, day_shuffles=day_shuffles,
                                   seed=seed)
    res = proc.evaluate(df, ctx)
    for f in res.findings:
        f.extras["symbol"] = symbol
        f.extras["interval"] = interval
    return res


def grid_fdr(results: list[ProcessResult],
             alpha: float = DEFAULT_FDR_ALPHA) -> FdrReport:
    """BH over every gate cell of the whole sweep — one family, one correction.

    Only `p_continue_excess` cells enter: markouts are descriptive and letting them
    into the family would both dilute the correction and launder them into
    discoveries."""
    gate = [f for r in results for f in r.findings if f.metric == GATE_METRIC]
    return apply_process_fdr(gate, alpha=alpha)


def sign_summary(results: list[ProcessResult],
                 metric: str = GATE_METRIC) -> list[dict]:
    """Per (interval, run_length): how many pairs point which way, and how many
    survive. The 1m/5m record was 34/36 negative — the sign distribution IS the
    finding, not the argmax."""
    cells: dict[tuple, list] = {}
    for r in results:
        for f in r.findings:
            if f.metric != metric:
                continue
            cells.setdefault((r.timeframe, int(f.extras["run_length"])), []).append(f)
    rows = []
    for (interval, k), fs in sorted(cells.items()):
        vals = np.array([f.value for f in fs], dtype=np.float64)
        rows.append({
            "interval": interval, "run_length": k, "metric": metric,
            "n_pairs": len(fs),
            "n_pos": int((vals > 0).sum()), "n_neg": int((vals < 0).sum()),
            "n_informative_pos": sum(1 for f in fs if f.informative and f.value > 0),
            "n_informative_neg": sum(1 for f in fs if f.informative and f.value < 0),
            "median_excess": round(float(np.median(vals)), 5) if len(vals) else None,
            "n_events_total": int(sum(f.extras.get("n_events", 0) for f in fs)),
        })
    return rows


def _slim(f) -> dict:
    """Artifact row for one cell — everything but the per-day list (size)."""
    e = f.extras
    return {"symbol": e.get("symbol"), "interval": e.get("interval"),
            "feature": f.feature, "horizon": f.horizon, "metric": f.metric,
            "value": f.value, "p_value": f.p_value, "q_value": f.p_adjusted,
            "z": e.get("z"), "n_events": e.get("n_events"),
            "verdict": e.get("verdict"), "n_days": e.get("n_days"),
            "frac_days_informative": e.get("frac_days_informative"),
            "informative": f.informative}


def _episode(task):
    symbol, interval, n_shuffles = task
    path = DEFAULT_CANDLE_DIR / f"{symbol}_{interval}.parquet"
    try:
        candles = pd.read_parquet(path)
    except Exception as exc:
        return {"symbol": symbol, "interval": interval, "error": str(exc)[:100]}
    if len(candles) < MIN_BARS:
        return {"symbol": symbol, "interval": interval,
                "skipped": f"{len(candles)} bars < {MIN_BARS}"}
    res = continuation_result(candles, symbol, interval, n_shuffles=n_shuffles)
    return {"symbol": symbol, "interval": interval, "n_bars": len(candles),
            "result": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", nargs="+", default=list(INTERVALS),
                    choices=list(INTERVALS))
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--n-shuffles", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    tasks = []
    universe: dict[str, int] = {}
    for interval in args.intervals:
        syms = available_candle_symbols(interval=interval)
        if args.max_symbols:
            syms = syms[:args.max_symbols]
        universe[interval] = len(syms)
        tasks += [(s, interval, args.n_shuffles) for s in syms]
    print(f"TC-1 · {[f'{i}:{n}' for i, n in universe.items()]} pairs · "
          f"gate={GATE_METRIC} (non-overlapping) · one BH family", flush=True)
    if not tasks:
        print("no candle files — run scripts/data/fetch_candles.py --universe first")
        return 1

    from concurrent.futures import ProcessPoolExecutor
    episodes, results = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, ep in enumerate(pool.map(_episode, tasks, chunksize=2), 1):
            episodes.append(ep)
            if "result" in ep:
                results.append(ep["result"])
            if i % 25 == 0:
                print(f"  [{i}/{len(tasks)}]", flush=True)

    report = grid_fdr(results)
    rows = sign_summary(results)
    prov = get_provenance()
    record_sweep(default_ledger_path(), process="persistence_stats",
                 target="next_bar_continuation_15m_1h",
                 n_tested=report.n_pvalued, git_sha=prov.get("git_sha"),
                 alpha=report.alpha, n_discoveries=report.n_discoveries)

    skipped = [{k: e[k] for k in ("symbol", "interval", "skipped")}
               for e in episodes if "skipped" in e]
    errors = [{k: e.get(k) for k in ("symbol", "interval", "error")}
              for e in episodes if "error" in e]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "provenance": prov,
        "gate_metric": GATE_METRIC,
        "gate_note": "markout cells are descriptive only — overlapping horizons "
                     "must not gate (FINDINGS §7.12)",
        "intervals": {i: {"n_pairs": universe.get(i)} for i in args.intervals},
        "n_shuffles": args.n_shuffles,
        "fdr": {"alpha": report.alpha, "n_cells": report.n_cells,
                "n_pvalued": report.n_pvalued,
                "n_discoveries": report.n_discoveries,
                "discoveries": report.discoveries[:50]},
        "sign_summary": rows,
        "cells": [_slim(f) for r in results for f in r.findings],
        "skipped": skipped, "errors": errors,
        "coverage": {"episodes": len(episodes), "with_result": len(results),
                     "skipped": len(skipped), "errors": len(errors)},
    }, indent=1, default=float))
    print(f"\nartifact: {REPORT}")
    print(f"FDR: {report.n_discoveries} of {report.n_pvalued} gate cells survive "
          f"(alpha={report.alpha})\n")

    print(f"{'interval':<9}{'k':>3}{'pairs':>7}{'pos':>6}{'neg':>6}"
          f"{'inf+':>6}{'inf-':>6}{'median':>10}{'events':>9}")
    for r in rows:
        print(f"{r['interval']:<9}{r['run_length']:>3}{r['n_pairs']:>7}"
              f"{r['n_pos']:>6}{r['n_neg']:>6}{r['n_informative_pos']:>6}"
              f"{r['n_informative_neg']:>6}{r['median_excess']:>10.4f}"
              f"{r['n_events_total']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

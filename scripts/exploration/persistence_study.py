"""PROC-20 full study: does momentum persist, and do VWAP band excursions revert?

Runs `persistence_stats` at the config budget over every available day × BTC/ETH/SOL, at
**1 m and 5 m bars** — the horizons where a move can exceed 11 bps round-trip, which is the
property every refuted result in the record lacks.

Two questions, one run:

  **A. Momentum persistence.** `P(continue | run length k)` against a sign-permutation null,
     plus markout in the run's own direction. A random walk gives 0.5 and geometric run
     lengths; anything above that is the effect.
  **B. Band excursion.** `research/new/vwap_sd_channel.txt` (LF7) predicts, from ONE day with
     n = 4–31 per cell: shallow touches (k ≤ 1.5) are **continuation** (adverse), capture
     appears at **k ≈ 2.0–2.5**, and **SOL > ETH > BTC** by thin-book ordering. This study
     either reproduces that ordering with N in the thousands or contradicts it — both are
     results, and LF7's parameter choices are supposed to be read off whichever it is.

Structure: day × symbol loads and bar-aggregates in parallel (cheap I/O), then the process
runs ONCE per (symbol, timeframe) over the concatenated series so the pooled permutation
null sees the whole record rather than a day at a time. Per-day verdicts come from the
process itself (PROC-4 folds); this driver only does I/O, fan-out and reporting.

Artifact: `reports/persistence_study.json`. Read-only, no capital path — band touches use
the touch price, which overstates fills; A4's queue sim gates any profit claim.

Usage:  python -m exploration.persistence_study [--days N] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "persistence_study.json"

TICK_COLUMNS = ["timestamp_ns", "symbol", "raw_midprice", "flow_volume_1s"]
MIN_TICKS = 50_000

#: bar timeframe -> (seconds per bar, horizons in bars)
TIMEFRAMES = {
    "1min": (60, {"1m": 1, "5m": 5, "30m": 30, "2h": 120}),
    "5min": (300, {"5m": 1, "25m": 5, "2h": 24, "8h": 96}),
}

#: LF7's grid, plus the shallow band its priors call adverse.
K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
VWAP_WINDOW = 60          # bars
EMBARGO_BARS = 30         # bars between counted touches
MAX_RUN_LENGTH = 6


def _bars_for(task):
    """One (day, symbol) → bar frames per timeframe. Parallel, I/O-bound."""
    day, sym = task
    from cluster_pipeline.loader import load_parquet
    try:
        df = load_parquet(str(DATA_DIR), symbols=[sym], start_date=day, end_date=day,
                          columns=TICK_COLUMNS, max_memory_mb=2500)
    except Exception as exc:
        return day, sym, {"error": str(exc)[:110]}, {}
    if len(df) < MIN_TICKS:
        return day, sym, {"skipped": f"{len(df)} ticks"}, {}

    out = {}
    for tf, (secs, _) in TIMEFRAMES.items():
        g = df["timestamp_ns"] // (secs * 10**9)
        bars = df.groupby(g).agg(
            timestamp_ns=("timestamp_ns", "last"),
            raw_midprice=("raw_midprice", "last"),
            flow_volume_1s=("flow_volume_1s", "sum"),
        ).reset_index(drop=True)
        bars["symbol"] = sym
        out[tf] = bars
    return day, sym, {"n_ticks": len(df)}, out


def _run_one(args_tuple) -> dict:
    """One (symbol, timeframe) over the whole concatenated bar series."""
    sym, tf, bars_records, horizons = args_tuple
    from processes.base import ProcessContext
    from processes.persistence_stats import PersistenceStatsProcess

    bars = pd.DataFrame(bars_records)
    ctx = ProcessContext(symbol=sym, timeframe=tf, price_col="raw_midprice",
                         horizons=horizons, costs={})
    res = PersistenceStatsProcess(
        max_run_length=MAX_RUN_LENGTH, vwap_window=VWAP_WINDOW, k_grid=K_GRID,
        embargo_bars=EMBARGO_BARS,
    ).evaluate(bars, ctx)
    cells = []
    for f in res.findings:
        e = f.extras
        cells.append({
            "symbol": sym, "timeframe": tf, "cell": f.feature, "metric": f.metric,
            "family": e["family"], "horizon": str(f.horizon), "n_events": e["n_events"],
            "value": f.value, "pooled_value": e.get("pooled_value"),
            "z": e.get("z"), "p_value": f.p_value, "bh_q": f.p_adjusted,
            "frac_days_informative": e.get("frac_days_informative"),
            "n_days": e.get("n_days"), "verdict": e.get("verdict"),
            "informative": bool(f.informative),
            "k": e.get("k"), "run_length": e.get("run_length"),
        })
    return {"symbol": sym, "timeframe": tf, "n_bars": len(bars),
            "summary": {k: v for k, v in res.summary.items() if k != "top"},
            "cells": cells}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    days = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    if args.days:
        days = days[-args.days:]
    tasks = [(d, s) for d in days for s in args.symbols]
    print(f"PROC-20 study · {len(days)} days × {args.symbols} · timeframes "
          f"{list(TIMEFRAMES)} · k_grid={K_GRID}", flush=True)

    from concurrent.futures import ProcessPoolExecutor

    # phase 1 — parallel load + bar aggregation
    collected = {(s, tf): [] for s in args.symbols for tf in TIMEFRAMES}
    coverage = {"episodes": len(tasks), "used": 0, "skipped": 0, "errors": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (day, sym, status, frames) in enumerate(
                pool.map(_bars_for, tasks, chunksize=1), 1):
            if status.get("error"):
                coverage["errors"] += 1
            elif status.get("skipped"):
                coverage["skipped"] += 1
            else:
                coverage["used"] += 1
                for tf, b in frames.items():
                    collected[(sym, tf)].append(b)
            if i % 20 == 0:
                print(f"  [{i}/{len(tasks)}] bars built", flush=True)
    print(f"coverage: {coverage}", flush=True)

    # phase 2 — one full-budget process run per (symbol, timeframe)
    jobs = []
    for (sym, tf), blocks in collected.items():
        if not blocks:
            continue
        bars = pd.concat(blocks, ignore_index=True).sort_values("timestamp_ns")
        jobs.append((sym, tf, bars.to_dict("list"), TIMEFRAMES[tf][1]))
    print(f"running {len(jobs)} (symbol, timeframe) combinations", flush=True)

    runs = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs) or 1)) as pool:
        for r in pool.map(_run_one, jobs):
            runs.append(r)
            print(f"  done {r['symbol']} {r['timeframe']}: {r['n_bars']:,} bars, "
                  f"{len(r['cells'])} cells", flush=True)

    all_cells = [c for r in runs for c in r["cells"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"days": days, "symbols": args.symbols, "timeframes": {k: v[1] for k, v in
                                                               TIMEFRAMES.items()},
         "k_grid": K_GRID, "vwap_window": VWAP_WINDOW, "embargo_bars": EMBARGO_BARS,
         "coverage": coverage,
         "runs": [{k: v for k, v in r.items() if k != "cells"} for r in runs],
         "cells": all_cells}, indent=1, default=float))
    print(f"\nartifact: {REPORT}")

    # ── momentum: P(continue | k) ────────────────────────────────────────────────
    print("\nA. MOMENTUM — P(continue | run length k), excess over sign-permutation null")
    print(f"{'tf':>6}{'sym':>5}" + "".join(f"{'k=' + str(k):>12}" for k in
                                           range(1, MAX_RUN_LENGTH + 1)))
    for tf in TIMEFRAMES:
        for sym in args.symbols:
            row = {c["run_length"]: c for c in all_cells
                   if c["timeframe"] == tf and c["symbol"] == sym
                   and c["metric"] == "p_continue_excess"}
            cells = []
            for k in range(1, MAX_RUN_LENGTH + 1):
                c = row.get(k)
                if not c or c["n_events"] < 30:
                    cells.append(f"{'-':>12}")
                else:
                    star = "*" if c["informative"] else " "
                    cells.append(f"{c['value']:>+9.4f}{star}{'':>2}")
            print(f"{tf:>6}{sym:>5}" + "".join(cells))

    # ── band: LF7's table, properly powered ──────────────────────────────────────
    print("\nB. BAND — markout in the reverting direction (bps) by k; LF7 predicts "
          "adverse at k<=1.5, capture at k~2.0-2.5, SOL>ETH>BTC")
    for tf in TIMEFRAMES:
        hz = list(TIMEFRAMES[tf][1])[-2]        # a mid-length horizon
        print(f"  [{tf}, horizon {hz}]")
        print(f"{'sym':>7}" + "".join(f"{'k=' + str(k):>13}" for k in K_GRID))
        for sym in args.symbols:
            row = {c["k"]: c for c in all_cells
                   if c["timeframe"] == tf and c["symbol"] == sym
                   and c["metric"] == "markout_bps" and c["family"] == "band"
                   and c["horizon"] == hz}
            cells = []
            for k in K_GRID:
                c = row.get(k)
                if not c or c["n_events"] < 30:
                    cells.append(f"{'-':>13}")
                else:
                    star = "*" if c["informative"] else " "
                    cells.append(f"{c['value']:>+8.2f}({c['n_events']:>3}){star}"[:13].rjust(13))
            print(f"{sym:>7}" + "".join(cells))

    n_info = sum(1 for c in all_cells if c["informative"])
    print(f"\ncells: {len(all_cells)} · informative after FDR + day-durability: {n_info}")
    print("* = null-significant AND durable across days. Band touches use the touch price "
          "(overstates fills) — A4 queue sim gates any profit claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Walk-forward audit sweep across symbols and timeframes.

Runs `ScalpingProfilerProcess.run(forward_test=True)` for every
(symbol, timeframe) cell in a 3x3 grid (BTC/ETH/SOL by 1min/2min/5min by
default) and writes one `walk_forward_{SYM}_{TF}.json` per cell into a
dated audit directory.

The aggregator (`audit_aggregate.py`) then consumes those JSONs to
produce the final keep/monitor/drop decision per feature.

Usage
-----
    python scripts/audit_sweep.py
    python scripts/audit_sweep.py --symbols BTC ETH --timeframes 1min 5min
    python scripts/audit_sweep.py --out-dir reports/profiler/audit_2026-05-11
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from scalping_profiler import (
    ScalpingProfilerProcess,
    load_profiler_config,
)

log = logging.getLogger("audit_sweep")

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_TIMEFRAMES = ["1min", "2min", "5min"]


def run_sweep(
    symbols: List[str],
    timeframes: List[str],
    data_dir: str,
    out_dir: Path,
    config_path: str,
) -> dict:
    """Execute every (symbol, timeframe) cell and collect summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_profiler_config(config_path)

    summary = {
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_dir": data_dir,
        "symbols": symbols,
        "timeframes": timeframes,
        "cells": [],
    }

    total = len(symbols) * len(timeframes)
    idx = 0
    for sym in symbols:
        for tf in timeframes:
            idx += 1
            t0 = time.time()
            cell_cfg = dict(base_config)
            cell_cfg["timeframe"] = tf
            log.info("[%d/%d] %s @ %s — starting", idx, total, sym, tf)
            cell = {"symbol": sym, "timeframe": tf}
            try:
                proc = ScalpingProfilerProcess(
                    config=cell_cfg,
                    data_dir=data_dir,
                    report_dir=str(out_dir),
                    state_file=str(out_dir / f".profiler_state_{sym}_{tf}.json"),
                )
                _, wf = proc.run(symbol=sym, forward_test=True)
                cell.update({
                    "keep": wf.keep_count,
                    "monitor": wf.monitor_count,
                    "drop": wf.drop_count,
                    "total_bars": wf.total_bars,
                    "n_folds": wf.n_folds,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                log.info(
                    "[%d/%d] %s @ %s — keep=%d monitor=%d drop=%d (%.1fs)",
                    idx, total, sym, tf,
                    wf.keep_count, wf.monitor_count, wf.drop_count,
                    cell["elapsed_s"],
                )
            except Exception as e:
                cell["error"] = repr(e)
                cell["elapsed_s"] = round(time.time() - t0, 1)
                log.error("[%d/%d] %s @ %s — FAILED: %r", idx, total, sym, tf, e)
            summary["cells"].append(cell)

    summary["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Sweep complete. Summary: %s", summary_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    ap.add_argument("--data-dir", default="data/features")
    ap.add_argument("--config", default="config/pipeline.toml")
    ap.add_argument("--out-dir", default=None,
                    help="default: reports/profiler/audit_YYYY-MM-DD/")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    today = datetime.date.today().isoformat()
    out_dir = Path(args.out_dir or f"reports/profiler/audit_{today}")

    summary = run_sweep(
        symbols=args.symbols,
        timeframes=args.timeframes,
        data_dir=args.data_dir,
        out_dir=out_dir,
        config_path=args.config,
    )

    # Friendly final table
    print(f"\n{'='*72}")
    print(f"  AUDIT SWEEP COMPLETE — outputs in {out_dir}/")
    print(f"{'='*72}")
    print(f"  {'symbol':<6} {'tf':<6} {'keep':>5} {'monitor':>8} {'drop':>5} {'bars':>7} {'elapsed':>9}")
    for c in summary["cells"]:
        if "error" in c:
            print(f"  {c['symbol']:<6} {c['timeframe']:<6} {'FAILED':>32}  ({c['elapsed_s']}s)")
        else:
            print(f"  {c['symbol']:<6} {c['timeframe']:<6} "
                  f"{c['keep']:>5} {c['monitor']:>8} {c['drop']:>5} "
                  f"{c['total_bars']:>7} {c['elapsed_s']:>8.1f}s")
    print()
    print(f"Next step:  python scripts/audit_aggregate.py {out_dir}")


if __name__ == "__main__":
    main()

"""
Process runner — load data, execute a process, chain transforms, persist.

Flow:
  1. Resolve params: PARAMS defaults < config/processes.toml [name] < CLI overrides
  2. Peek the parquet schema (pyarrow, no data read) for available columns
  3. Load ONLY required columns via cluster_pipeline.loader.load_parquet
     (columns= pruning + max_memory_mb guard + date-dir filtering)
  4. bars-level processes: aggregate_bars() then resolve the bar price column
     (the screener's candidate loop); tick-level processes get raw ticks
  5. Dispatch by kind; transform output is saved as parquet and optionally
     chained into an evaluation process (--score-with ic_horizon)
  6. Stamp provenance + a cheap data fingerprint, persist JSON + index row

Standalone: python -m processes.runner ic_horizon --symbol BTC
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow `python scripts/processes/runner.py` without the editable install
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from processes.base import (  # noqa: E402
    ProcessContext, ProcessResult, get_provenance,
)
from processes.registry import get_process, list_processes  # noqa: E402
from processes import persistence  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "processes.toml"
DEFAULT_DATA_DIR = ROOT / "data" / "features"

# Same defaults as the alpha screener
DEFAULT_HORIZONS = {
    "5min": {"15min": 3, "1h": 12, "4h": 48},
    "15min": {"1h": 4, "4h": 16, "1d": 96},
    "1h": {"4h": 4, "1d": 24, "3d": 72},
    "4h": {"1d": 6, "3d": 18, "1w": 42},
}

# Tick-level horizon labels (10 Hz)
TICK_HORIZONS = {"1s": 10, "5s": 50}

_META_LOAD = ["timestamp_ns", "symbol"]


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _peek_schema_columns(data_dir: Path) -> list[str]:
    """Column names from the newest parquet file's footer — no data read."""
    import pyarrow.parquet as pq
    files = sorted(data_dir.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return list(pq.read_schema(files[-1]).names)


def _data_fingerprint(data_dir: Path, start_date, end_date) -> str:
    """Cheap deterministic fingerprint: sha256 over (relative path, size).

    Replaced by scripts/provenance.data_fingerprint() (plan T2) when it lands.
    """
    try:
        from provenance import data_fingerprint as _df  # type: ignore
        return _df(data_dir, start_date=start_date, end_date=end_date)
    except Exception:
        pass
    h = hashlib.sha256()
    for f in sorted(data_dir.glob("**/*.parquet")):
        h.update(f"{f.relative_to(data_dir)}:{f.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def _resolve_bar_price_col(columns, price_col: str) -> str:
    for cand in (f"{price_col}_close", f"{price_col}_mean", f"{price_col}_last", price_col):
        if cand in columns:
            return cand
    raise ValueError(
        f"No price column found (tried {price_col} variants). "
        f"Price-like columns: {[c for c in columns if 'price' in c.lower()]}"
    )


def run_process(
    name: str,
    symbol: str = "BTC",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    params: Optional[dict] = None,
    score_with: Optional[str] = None,
    save: bool = True,
    out_dir: Path | str = persistence.DEFAULT_OUT_DIR,
    db_path: Path | str | None = persistence.DEFAULT_DB_PATH,
) -> ProcessResult:
    """Execute one process run end-to-end. Returns the (persisted) result."""
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars

    t0 = time.time()
    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    timeframe = timeframe or defaults.get("timeframe", "15min")
    price_col = defaults.get("price_col", "raw_midprice")
    max_memory_mb = defaults.get("max_memory_mb", 4000)
    data_dir = Path(data_dir)

    merged = {**cfg.get(name, {}), **(params or {})}
    proc = get_process(name, **merged)

    available = _peek_schema_columns(data_dir)
    required = proc.required_columns(available)
    load_set = set(required) | {c for c in _META_LOAD if c in available}
    if price_col in available:
        load_set.add(price_col)
    load_cols = sorted(load_set)
    log.info("Loading %d/%d columns from %s", len(load_cols), len(available), data_dir)

    df = load_parquet(
        str(data_dir),
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        columns=load_cols,
        max_memory_mb=max_memory_mb,
    )
    n_rows = len(df)

    if proc.data_level == "ticks":
        frame = df.reset_index(drop=True)
        resolved_price = price_col
        horizons = TICK_HORIZONS
        n_bars = None
    else:
        frame = aggregate_bars(df, timeframe=timeframe)
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] == symbol].reset_index(drop=True)
        resolved_price = _resolve_bar_price_col(frame.columns, price_col)
        horizons = DEFAULT_HORIZONS.get(timeframe, {"4h": 4, "1d": 24})
        n_bars = len(frame)
    del df

    from utils.costs import load_costs
    ctx = ProcessContext(
        symbol=symbol,
        timeframe=timeframe,
        price_col=resolved_price,
        horizons=horizons,
        costs=load_costs(),
        data_dir=str(data_dir),
        start_date=start_date,
        end_date=end_date,
    )

    derived_df = None
    if proc.kind == "transform":
        derived_df, result = proc.transform(frame, ctx)
    else:
        result = proc.evaluate(frame, ctx)

    result.provenance = get_provenance()
    result.data = {
        "dir": str(data_dir),
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": n_rows,
        "n_bars": n_bars,
        "fingerprint": _data_fingerprint(data_dir, start_date, end_date),
    }

    # An errored or empty transform produces nothing worth saving or scoring
    if derived_df is not None and (result.summary.get("error") or derived_df.empty
                                   or not len(derived_df.columns)):
        derived_df = None

    if derived_df is not None and save:
        parquet_path = persistence.save_derived(result, derived_df, out_dir=out_dir)
        result.derived = {
            "columns": [c for c in derived_df.columns if c not in ("bar_start", "symbol")],
            "parquet": str(parquet_path),
            "scored_by": None,
        }

    # Chain: score derived series with an evaluation process
    if derived_df is not None and score_with:
        scorer = get_process(score_with, **cfg.get(score_with, {}))
        score_frame = derived_df.copy()
        if resolved_price not in score_frame.columns and resolved_price in frame.columns:
            score_frame[resolved_price] = frame[resolved_price].to_numpy()
        score_result = scorer.evaluate(score_frame, ctx)
        score_result.provenance = result.provenance
        score_result.data = result.data
        if save:
            persistence.save_result(score_result, out_dir=out_dir, db_path=db_path)
        if result.derived is not None:
            result.derived["scored_by"] = score_result.run_id

    result.summary["runtime_s"] = round(time.time() - t0, 1)
    if save:
        persistence.save_result(result, out_dir=out_dir, db_path=db_path)
    return result


def _parse_param(kv: str):
    key, _, raw = kv.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"--param expects k=v, got '{kv}'")
    low = raw.lower()
    if low in ("true", "false"):
        val = low == "true"
    elif low in ("none", "null"):
        val = None
    elif "," in raw:
        val = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
    return key, val


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="NAT analytical process runner",
        epilog=f"Registered processes: {', '.join(list_processes())}",
    )
    parser.add_argument("name", help="Process name (see `nat process list`)")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--timeframe", default=None,
                        help="Bar timeframe (default from config/processes.toml)")
    parser.add_argument("--start-date", default=None, help="e.g. 2026-06-05")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--features", default=None,
                        help="Comma-separated name prefixes to score")
    parser.add_argument("--param", action="append", default=[], metavar="K=V",
                        help="Process param override (repeatable)")
    parser.add_argument("--score-with", default=None,
                        help="Evaluation process to chain onto transform output")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full result JSON")
    parser.add_argument("--top", type=int, default=15, help="Findings rows to print")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    params = dict(_parse_param(kv) for kv in args.param)
    if args.features:
        params["features"] = [p.strip() for p in args.features.split(",")]

    result = run_process(
        args.name,
        symbol=args.symbol,
        data_dir=args.data_dir,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        params=params,
        score_with=args.score_with,
        save=not args.no_save,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    s = result.summary
    print(f"\n  {result.run_id}")
    print(f"  process={result.process} kind={result.kind} symbol={result.symbol} "
          f"timeframe={result.timeframe}")
    print(f"  tested={s['n_tested']} informative={s['n_informative']} "
          f"skipped={len(result.features_skipped)} runtime={s['runtime_s']}s")
    if s.get("error"):
        print(f"  ERROR: {s['error']}")
    ranked = sorted(result.findings, key=lambda f: abs(f.value), reverse=True)
    if ranked:
        print(f"\n  {'feature':<40} {'horizon':>8} {'metric':>14} "
              f"{'value':>10} {'p_adj':>8} {'info':>5}")
        for f in ranked[: args.top]:
            p_adj = f"{f.p_adjusted:.4f}" if f.p_adjusted is not None else "-"
            mark = "*" if f.informative else ""
            print(f"  {f.feature:<40} {str(f.horizon):>8} {f.metric:>14} "
                  f"{f.value:>10.5f} {p_adj:>8} {mark:>5}")
    if result.derived:
        print(f"\n  derived: {result.derived['columns']} -> {result.derived['parquet']}")
        if result.derived.get("scored_by"):
            print(f"  scored by: {result.derived['scored_by']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import dataclasses
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
from processes.fdr import (  # noqa: E402
    DEFAULT_FDR_ALPHA, apply_process_fdr, default_ledger_path, record_sweep,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
try:
    import nat_paths
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nat_paths
CONFIG_PATH = nat_paths.config_dir() / "processes.toml"
DEFAULT_DATA_DIR = nat_paths.features_dir()

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


def _chain_load_columns(proc, scorer, available: list[str]) -> set[str]:
    """Columns the loader must fetch for a (transform -> scorer) chain.

    Pruning to the TRANSFORM's required_columns alone loads zero features for
    triple_barrier (it needs only price) — the chained scorer then has nothing to
    score. Load the union of both processes' needs instead.
    """
    cols = set(proc.required_columns(available))
    if scorer is not None:
        cols |= set(scorer.required_columns(available))
    return cols


def _build_score_frame(frame, derived_df, resolved_price: str, tgt: Optional[str]):
    """The frame a chained scorer sees.

    Label mode (tgt set): the ORIGINAL features plus the derived columns — scoring a
    label against only its own tb_* siblings is vacuous (the scorer excludes them as
    leakage, leaving nothing). Derived columns win on a name clash: they are freshly
    computed. Feature mode (no tgt): the derived series alone, plus the price column
    so the scorer can compute forward returns (pca_combo -> ic_horizon unchanged).
    """
    if tgt:
        score_frame = frame.copy()
        for c in derived_df.columns:
            score_frame[c] = derived_df[c].to_numpy()
        return score_frame
    score_frame = derived_df.copy()
    if resolved_price not in score_frame.columns and resolved_price in frame.columns:
        score_frame[resolved_price] = frame[resolved_price].to_numpy()
    return score_frame


def _resolve_score_target(transform, explicit: Optional[str] = None) -> Optional[str]:
    """The target a chained scorer should use: an explicit override, else the transform's
    declared ``target_column()`` (e.g. triple_barrier -> tb_label)."""
    if explicit:
        return explicit
    fn = getattr(transform, "target_column", None)
    return fn() if callable(fn) else None


def _fdr_and_ledger(res: ProcessResult, ctx: ProcessContext, cfg: dict, save: bool) -> None:
    """PROC-13: BH-correct a sweep's cells in place, then (when persisting) ledger the run.

    Composes onto every evaluation result so no argmax is ever surfaced without its BH
    q-value, and the sweep is recorded in the program-level ledger for cross-run accounting.
    A result with no p-valued findings (e.g. a transform) is a no-op.
    """
    if not res.findings:
        return
    alpha = float((cfg.get("fdr", {}) or {}).get("alpha", DEFAULT_FDR_ALPHA))
    rep = apply_process_fdr(res, alpha=alpha)
    if rep.n_pvalued == 0:
        return
    res.summary["fdr"] = {
        "alpha": rep.alpha, "n_cells": rep.n_cells, "n_pvalued": rep.n_pvalued,
        "n_discoveries": rep.n_discoveries, "argmax": rep.argmax,
    }
    res.summary["n_informative"] = sum(1 for f in res.findings if f.informative)
    if save:
        record_sweep(
            default_ledger_path(),
            process=res.process,
            target=ctx.target_col or "forward_return",
            n_tested=rep.n_pvalued,
            git_sha=(res.provenance or {}).get("git_sha"),
            alpha=rep.alpha,
            n_discoveries=rep.n_discoveries,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
        )


def _resolve_bar_price_col(columns, price_col: str) -> str:
    for cand in (f"{price_col}_close", f"{price_col}_mean", f"{price_col}_last", price_col):
        if cand in columns:
            return cand
    raise ValueError(
        f"No price column found (tried {price_col} variants). "
        f"Price-like columns: {[c for c in columns if 'price' in c.lower()]}"
    )



def _run_candles_process(*, proc, name, symbols, interval, data_dir, start_date,
                         end_date, cfg, save, out_dir, db_path, t0):
    """Execute a cross-sectional process over the candle archive (PROC-19).

    Differs from the tick/bar path in three ways that all matter: the source is
    `data/candles/` rather than `data/features/`, the frame is long over MANY symbols
    rather than one, and the price column is `close` (bars have no mid).
    """
    from processes.candles import (CANDLE_PRICE_COL, DEFAULT_CANDLE_DIR,
                                   available_candle_symbols, load_candles)
    from utils.costs import load_costs

    candle_dir = Path(data_dir)
    if not candle_dir.exists() or candle_dir == Path(DEFAULT_DATA_DIR):
        candle_dir = DEFAULT_CANDLE_DIR

    universe = list(symbols) if symbols else available_candle_symbols(candle_dir, interval)
    if not universe:
        raise FileNotFoundError(f"no candle symbols at {interval} under {candle_dir}")

    frame, load_report = load_candles(
        universe, interval=interval, data_dir=candle_dir,
        start_date=start_date, end_date=end_date, return_report=True,
    )
    if load_report["missing"] or load_report["empty"]:
        # Named, never silent: a rank over 140 pairs believing it covered 177 is biased
        # by whatever the missing 37 had in common.
        log.warning("candles: %d requested, %d loaded (missing=%s empty=%s)",
                    len(universe), len(load_report["loaded"]),
                    load_report["missing"][:8], load_report["empty"][:8])

    ctx = ProcessContext(
        symbol=(universe[0] if len(universe) == 1 else "UNIVERSE"),
        timeframe=interval,
        price_col=CANDLE_PRICE_COL,
        horizons=DEFAULT_HORIZONS.get(interval, {"1d": 24, "7d": 168}),
        costs=load_costs(),
        data_dir=str(candle_dir),
        start_date=start_date,
        end_date=end_date,
        symbols=load_report["loaded"],
    )

    result = proc.evaluate(frame, ctx)
    result.provenance = get_provenance()
    result.data = {
        "dir": str(candle_dir),
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "n_rows": len(frame),
        "n_symbols_requested": len(universe),
        "n_symbols_loaded": len(load_report["loaded"]),
        "symbols_missing": load_report["missing"],
        "symbols_empty": load_report["empty"],
    }
    _fdr_and_ledger(result, ctx, cfg, save)
    result.runtime_s = round(time.time() - t0, 2)
    if save:
        persistence.save_result(result, out_dir=out_dir, db_path=db_path)
    return result


def run_process(
    name: str,
    symbol: str = "BTC",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    symbols: Optional[list] = None,
    interval: str = "1h",
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    params: Optional[dict] = None,
    score_with: Optional[str] = None,
    score_target: Optional[str] = None,
    score_params: Optional[dict] = None,
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
    scorer = (
        get_process(score_with, **{**cfg.get(score_with, {}), **(score_params or {})})
        if score_with else None
    )

    # PROC-19 — the `candles` data level: a MULTI-symbol long frame from the XS-1
    # archive, for cross-sectional (Class-3) processes. Routed before the tick/bar path
    # because the source, the shape and the price column all differ.
    if proc.data_level == "candles":
        return _run_candles_process(
            proc=proc, name=name, symbols=symbols, interval=interval,
            data_dir=data_dir, start_date=start_date, end_date=end_date,
            cfg=cfg, save=save, out_dir=out_dir, db_path=db_path, t0=t0,
        )

    available = _peek_schema_columns(data_dir)
    required = _chain_load_columns(proc, scorer, available)
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
    # PROC-13: FDR-correct the sweep + ledger the run (no-op for transforms).
    _fdr_and_ledger(result, ctx, cfg, save)

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
    if derived_df is not None and scorer is not None:
        # PROC-5: point the scorer at the transform's declared target (e.g. tb_label),
        # so it evaluates the label rather than silently falling back to forward returns.
        # In label mode the scorer sees the ORIGINAL features + the derived label.
        tgt = _resolve_score_target(proc, score_target)
        score_frame = _build_score_frame(frame, derived_df, resolved_price, tgt)
        score_ctx = dataclasses.replace(ctx, target_col=tgt) if tgt else ctx
        score_result = scorer.evaluate(score_frame, score_ctx)
        score_result.provenance = result.provenance
        score_result.data = result.data
        _fdr_and_ledger(score_result, score_ctx, cfg, save)
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
    parser.add_argument("--score-target", default=None,
                        help="Target column for the chained scorer (default: the "
                             "transform's declared target, e.g. triple_barrier -> tb_label)")
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
        score_target=args.score_target,
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

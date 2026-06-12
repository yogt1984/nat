#!/usr/bin/env python3
"""
Nightly report — one-shot overnight pass producing a self-contained HTML report.

Sections:
  health    Per-day/per-symbol coverage hours, row counts, gaps
  wiring    NaN-wiring diagnostics: whale/liquidation/concentration coverage,
            range checks, position-tracker status (feeds the 48h concentration
            viability assessment in docs/tasks_assigned_12_6_26/)
  features  Per-category NaN coverage, key-feature distributions, drift flags,
            top IC features from the alpha screener
  gauntlet  Multi-day OOS sweep across all algorithms (subprocess to
            scripts/alpha/overnight_sweep.py)
  viz       Embedded matplotlib figures (price overview, equity curves,
            NaN heatmap, wiring histograms)

State is saved to reports/nightly/<date>.json and the HTML re-rendered after
every section, so a crash at any point still leaves a valid morning report.

Subcommands:
  run       Run the full pass (default)
  report    Print the latest nightly summary + file paths
  open      Open the latest nightly HTML in a browser

Usage:
  python scripts/nightly_report.py run                          # full (~2h)
  python scripts/nightly_report.py run --last 2 --quick --skip-gauntlet
  python scripts/nightly_report.py report
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data.features import available_dates, load_bars, load_features  # noqa: E402
from data.schema import BASE_FEATURES, OPTIONAL_FEATURES  # noqa: E402
from viz.features import COLORS, apply_style  # noqa: E402

REPORTS_DIR = ROOT / "reports"
NIGHTLY_DIR = REPORTS_DIR / "nightly"
PID_FILE = REPORTS_DIR / ".nightly.pid"
GAUNTLET_PID = REPORTS_DIR / ".gauntlet.pid"
GAUNTLET_JSON = REPORTS_DIR / "gauntlet_latest.json"
SCREEN_JSON = REPORTS_DIR / "alpha_screen.json"

SYMBOLS_DEFAULT = ["BTC", "ETH", "SOL"]
SECTION_ORDER = ["health", "wiring", "features", "gauntlet", "viz"]
GAP_SECONDS = 60.0
COMPLETE_DAY_HOURS = 20.0  # days with fewer hours are excluded from drift baseline
MIN_ROWS_PER_HOUR = 30_000  # matches validate_data.ValidationConfig.min_records_per_hour
SCREEN_MAX_AGE_H = 24.0

NAN_WIRING_CATEGORIES = ["whale", "liquidation", "concentration"]

# Expected ranges from docs/tasks_assigned_12_6_26/01_concentration_viability_assessment.md
RANGE_SPECS = {
    "top5_concentration": (0.05, 0.50),
    "top10_concentration": (0.10, 0.60),
    "herfindahl_index": (0.01, 0.25),
    "gini_coefficient": (0.30, 0.80),
}

# Representative columns for distribution summaries (one or two per category)
KEY_FEATURES = [
    "raw_midprice", "raw_spread_bps",
    "imbalance_qty_l1", "imbalance_depth_weighted",
    "flow_intensity", "flow_aggressor_ratio_30s", "flow_volume_30s",
    "vol_returns_1m", "vol_zscore",
    "ent_permutation_returns_16", "ent_tick_30s",
    "ctx_funding_rate", "ctx_oi_change_pct_5m", "ctx_premium_bps",
    "trend_momentum_300", "trend_hurst_600",
    "mf_rsi_5m", "mf_bb_width_5m",
    "illiq_kyle_100", "illiq_composite",
    "toxic_vpin_50", "toxic_index",
    "derived_regime_indicator",
    "micro_obi_velocity",
    "hawkes_branching_ratio",
]

# col -> category lookup over the full known schema
CATEGORY_OF = {
    col: cat
    for cat, cols in {**BASE_FEATURES, **OPTIONAL_FEATURES}.items()
    for col in cols
}

# Module-global state reference so the signal handler can flush a partial report
_CURRENT: dict | None = None
_ARGS = None


def _log(msg: str):
    print(msg)
    sys.stdout.flush()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _sharpe(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from daily PnL array (same formula as the gauntlet)."""
    if len(daily_pnl) < 2:
        return 0.0
    mu = np.mean(daily_pnl)
    sigma = np.std(daily_pnl, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


def _save_state(state: dict):
    NIGHTLY_DIR.mkdir(parents=True, exist_ok=True)
    path = NIGHTLY_DIR / f"{state['date']}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, default=_json_default)
    tmp.rename(path)
    _write_html(state)


def _write_html(state: dict):
    NIGHTLY_DIR.mkdir(parents=True, exist_ok=True)
    path = NIGHTLY_DIR / f"{state['date']}.html"
    tmp = path.with_suffix(".htmltmp")
    tmp.write_text(render_html(state))
    tmp.rename(path)


def _fig_to_b64(fig, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_data_dir(args) -> Path:
    p = Path(args.data_dir)
    return p if p.is_absolute() else ROOT / p


def _window_dates(args) -> list[str]:
    """Last N available dates (inclusive of today's partial day)."""
    dates = available_dates(data_dir=_resolve_data_dir(args))
    return dates[-args.last:] if args.last else dates


def _header_snapshot(data_dir: Path) -> dict:
    """Metadata-only data snapshot (like data_health() but without the
    expensive available_symbols() scan that reads every file)."""
    dates = available_dates(data_dir=data_dir)
    total_files = 0
    total_rows = 0
    for d in dates:
        for f in (data_dir / d).glob("*.parquet"):
            try:
                total_rows += pq.read_metadata(str(f)).num_rows
                total_files += 1
            except (pa.ArrowInvalid, OSError):
                continue
    latest_str = None
    freshness = None
    if dates:
        for f in sorted((data_dir / dates[-1]).glob("*.parquet"), reverse=True):
            try:
                table = pq.read_table(str(f), columns=["timestamp_ns"])
                if table.num_rows:
                    ts = table.column("timestamp_ns")[table.num_rows - 1].as_py()
                    latest = pd.Timestamp(ts, unit="ns")
                    latest_str = latest.isoformat()
                    freshness = float((pd.Timestamp.now() - latest).total_seconds())
                    break
            except (pa.ArrowInvalid, OSError, KeyError):
                continue
    return {"total_rows": total_rows, "total_files": total_files,
            "latest_timestamp": latest_str, "freshness_seconds": freshness}


# ── Section: data health ──────────────────────────────────────────────────

def section_health(state: dict, args) -> dict:
    data_dir = _resolve_data_dir(args)
    window = _window_dates(args)
    if not window:
        return {"note": "no data found", "days": {}, "missing_days": []}

    # Calendar days spanned by the window that have no data dir at all
    span = pd.date_range(window[0], window[-1], freq="D").strftime("%Y-%m-%d")
    missing = [d for d in span if d not in window]

    days: dict[str, dict] = {}
    for d in window:
        per_sym = {}
        for sym in args.symbols:
            df = load_features(symbols=[sym], date_range=(d, d),
                               columns=["timestamp_ns"], data_dir=data_dir,
                               validate=False)
            if df.empty:
                per_sym[sym] = {"rows": 0, "hours": 0.0, "rows_per_hour": 0,
                                "gaps": [], "n_gaps": 0}
                continue
            ts = df["timestamp_ns"].values
            hours = float((ts[-1] - ts[0]) / 1e9 / 3600)
            diffs = np.diff(ts) / 1e9
            gap_idx = np.where(diffs > GAP_SECONDS)[0]
            gaps = sorted(
                ({"start": pd.Timestamp(int(ts[i]), unit="ns").isoformat(),
                  "seconds": float(diffs[i])} for i in gap_idx),
                key=lambda g: -g["seconds"],
            )[:5]
            per_sym[sym] = {
                "rows": int(len(ts)),
                "hours": round(hours, 2),
                "rows_per_hour": int(len(ts) / hours) if hours > 0 else 0,
                "n_gaps": int(len(gap_idx)),
                "gaps": gaps,
            }
        days[d] = per_sym
        _log(f"  [health] {d}: " + ", ".join(
            f"{s}={v['hours']:.1f}h/{v['rows']:,}r" for s, v in per_sym.items()))

    return {"days": days, "missing_days": missing, "window": window,
            "gap_threshold_s": GAP_SECONDS, "min_rows_per_hour": MIN_ROWS_PER_HOUR}


# ── Section: NaN-wiring diagnostics ───────────────────────────────────────

def _position_tracker_status() -> dict:
    """Report whether [position_tracker] is enabled in config/ing.toml."""
    cfg_path = ROOT / "config" / "ing.toml"
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        return {"configured": False, "enabled": False, "note": f"config unreadable: {e}"}
    pt = cfg.get("position_tracker")
    if pt is None:
        return {"configured": False, "enabled": False,
                "note": "[position_tracker] absent/commented out in config/ing.toml"}
    return {"configured": True, "enabled": bool(pt.get("enabled", False)),
            "config": {k: v for k, v in pt.items() if k != "initial_wallets"},
            "n_initial_wallets": len(pt.get("initial_wallets", []))}


def _scan_ingestor_log() -> dict:
    """Best-effort scan of the newest ingestor log for wallet-discovery markers."""
    log_dir = ROOT / "logs"
    out = {"log_file": None, "users_populated": None, "wallet_lines": 0}
    try:
        logs = sorted(log_dir.glob("ingestor*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return out
        log_file = logs[-1]
        out["log_file"] = log_file.name
        with open(log_file, errors="replace") as f:
            tail = f.readlines()[-20000:]
        for line in tail:
            if "WsTrade.users field IS populated" in line:
                out["users_populated"] = True
            elif "WsTrade.users field not populated" in line:
                if out["users_populated"] is None:
                    out["users_populated"] = False
            if "wallet" in line.lower():
                out["wallet_lines"] += 1
    except OSError:
        pass
    return out


def section_wiring(state: dict, args) -> dict:
    data_dir = _resolve_data_dir(args)
    window = _window_dates(args)
    wiring_cols = [c for cat in NAN_WIRING_CATEGORIES for c in OPTIONAL_FEATURES[cat]]

    coverage: dict[str, dict] = {c: {"non_nan": 0, "total": 0} for c in wiring_cols}
    per_day: dict[str, dict[str, float]] = {}  # date -> category -> coverage pct
    samples: dict[str, list[np.ndarray]] = {c: [] for c in wiring_cols}

    for d in window:
        day_counts = {c: {"non_nan": 0, "total": 0} for c in wiring_cols}
        for sym in args.symbols:
            df = load_features(symbols=[sym], date_range=(d, d),
                               columns=wiring_cols, data_dir=data_dir,
                               validate=False)
            if df.empty:
                continue
            for c in wiring_cols:
                if c not in df.columns:
                    continue
                vals = df[c].values
                ok = vals[~np.isnan(vals)]
                coverage[c]["non_nan"] += len(ok)
                coverage[c]["total"] += len(vals)
                day_counts[c]["non_nan"] += len(ok)
                day_counts[c]["total"] += len(vals)
                if len(ok):
                    samples[c].append(ok[:: max(1, len(ok) // 5000)])
            del df
        per_day[d] = {}
        for cat in NAN_WIRING_CATEGORIES:
            nn = sum(day_counts[c]["non_nan"] for c in OPTIONAL_FEATURES[cat])
            tot = sum(day_counts[c]["total"] for c in OPTIONAL_FEATURES[cat])
            per_day[d][cat] = round(100.0 * nn / tot, 2) if tot else 0.0
        _log(f"  [wiring] {d}: " + ", ".join(f"{k}={v}%" for k, v in per_day[d].items()))

    # Per-column coverage + range checks + histograms (bins stored, not raw values)
    columns = {}
    histograms = {}
    for c in wiring_cols:
        tot = coverage[c]["total"]
        nn = coverage[c]["non_nan"]
        entry = {"coverage_pct": round(100.0 * nn / tot, 3) if tot else 0.0,
                 "category": CATEGORY_OF.get(c, "?")}
        if samples[c]:
            vals = np.concatenate(samples[c])
            entry.update({
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            })
            if c in RANGE_SPECS:
                lo, hi = RANGE_SPECS[c]
                in_range = float(np.mean((vals >= lo) & (vals <= hi)) * 100)
                entry["expected_range"] = [lo, hi]
                entry["pct_in_range"] = round(in_range, 1)
            counts, edges = np.histogram(vals, bins=40)
            histograms[c] = {"counts": counts.tolist(),
                             "edges": [float(e) for e in edges]}
        columns[c] = entry

    # Verdict per the viability decision matrix. position_count (a concentration
    # feature) is the number of tracked positions — the closest observable proxy
    # for "wallets tracked"; OI coverage needs the tracker's own log line.
    tracker = _position_tracker_status()
    log_scan = _scan_ingestor_log()
    pos_med = columns.get("position_count", {}).get("median")
    any_data = any(v["coverage_pct"] > 0.1 for v in columns.values())
    if not any_data:
        verdict = "unavailable"
        reason = ("no non-NaN values in whale/liquidation/concentration columns"
                  + ("" if tracker["enabled"] else " (position tracker disabled)"))
    elif pos_med is not None and pos_med >= 50:
        verdict, reason = "viable", f"median position_count {pos_med:.0f} >= 50"
    elif pos_med is not None and pos_med >= 20:
        verdict, reason = "noisy", f"median position_count {pos_med:.0f} in [20, 50)"
    else:
        verdict = "noisy"
        reason = "data present but position_count low/unavailable — inspect manually"

    return {"tracker": tracker, "log_scan": log_scan, "verdict": verdict,
            "verdict_reason": reason, "per_day": per_day, "columns": columns,
            "histograms": histograms}


# ── Section: feature statistics ───────────────────────────────────────────

def _nan_counts_file(fpath: Path) -> tuple[int, dict[str, int]]:
    """Count NaN+null per float column in one parquet file (streamed, ~200MB peak)."""
    table = pq.read_table(str(fpath))
    counts = {}
    for field in table.schema:
        if pa.types.is_floating(field.type):
            col = table.column(field.name)
            nan_ct = pc.sum(pc.is_nan(col)).as_py() or 0
            counts[field.name] = int(nan_ct) + col.null_count
    return table.num_rows, counts


def _ic_screen(args) -> dict:
    """Top-IC features from the alpha screener (cached if <24h old)."""
    age_h = None
    if SCREEN_JSON.exists():
        age_h = (time.time() - SCREEN_JSON.stat().st_mtime) / 3600
    if age_h is not None and age_h < SCREEN_MAX_AGE_H:
        source = f"cached ({age_h:.1f}h old)"
    elif args.quick:
        if age_h is None:
            return {"source": "skipped (--quick, no cached results)", "top": []}
        source = f"cached-stale ({age_h:.0f}h old, --quick)"
    else:
        _log("  [features] alpha_screen.json stale — running screener (this can take a while)")
        r = subprocess.run([sys.executable, "-m", "scripts.alpha.screener"],
                           cwd=str(ROOT), timeout=3600)
        source = "fresh run" if r.returncode == 0 else f"screener exited {r.returncode}, using stale cache"
    try:
        with open(SCREEN_JSON) as f:
            screen = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"source": f"unreadable: {e}", "top": []}

    results = screen.get("results", [])
    sig = [r for r in results if r.get("significant", r.get("ic_p_adjusted", 1) < 0.05)]
    pool = sig if sig else results
    pool = sorted(pool, key=lambda r: -abs(r.get("ic_ir", 0)))[:20]
    top = [{k: r.get(k) for k in ("feature", "symbol", "horizon", "ic_mean",
                                  "ic_ir", "ic_p_adjusted", "turnover",
                                  "breakeven_bps")} for r in pool]
    return {"source": source, "generated": screen.get("timestamp"),
            "n_tested": screen.get("n_features_tested"),
            "n_significant": screen.get("n_significant"), "top": top}


def section_features(state: dict, args) -> dict:
    data_dir = _resolve_data_dir(args)
    window = _window_dates(args)
    stride = 100 if args.quick else 20

    # 1. NaN coverage per category x day, streamed one hourly file at a time
    cat_day: dict[str, dict[str, float]] = {}
    uncategorized: set[str] = set()
    for d in window:
        day_dir = data_dir / d
        files = [f for f in sorted(day_dir.glob("*.parquet"))
                 if not f.name.endswith(".tmp")]
        col_nan: dict[str, int] = {}
        rows = 0
        for f in files:
            try:
                n, counts = _nan_counts_file(f)
            except (pa.ArrowInvalid, OSError) as e:
                _log(f"  [features] skipping {f.name}: {e}")
                continue
            rows += n
            for c, ct in counts.items():
                col_nan[c] = col_nan.get(c, 0) + ct
        cat_cov: dict[str, list[float]] = {}
        for c, nan_ct in col_nan.items():
            cat = CATEGORY_OF.get(c)
            if cat is None:
                uncategorized.add(c)
                cat = "uncategorized"
            cat_cov.setdefault(cat, []).append(1.0 - nan_ct / rows if rows else 0.0)
        cat_day[d] = {cat: round(100.0 * float(np.mean(v)), 2)
                      for cat, v in sorted(cat_cov.items())}
        _log(f"  [features] {d}: NaN coverage over {rows:,} rows, "
             f"{len(col_nan)} float cols")

    # 2. Distributions + drift on KEY_FEATURES (per symbol, subsampled)
    daily_means: dict[str, dict[str, dict[str, float]]] = {}  # sym -> date -> col -> mean
    sample_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in args.symbols}
    complete_days: dict[str, list[str]] = {s: [] for s in args.symbols}
    health_days = state["sections"].get("health", {}).get("days", {})
    for d in window:
        for sym in args.symbols:
            df = load_features(symbols=[sym], date_range=(d, d),
                               columns=KEY_FEATURES, data_dir=data_dir,
                               validate=False)
            if df.empty:
                continue
            cols = [c for c in KEY_FEATURES if c in df.columns]
            daily_means.setdefault(sym, {})[d] = {
                c: float(df[c].mean()) for c in cols}
            sample_frames[sym].append(df[cols].iloc[::stride])
            hours = health_days.get(d, {}).get(sym, {}).get("hours")
            if hours is None or hours >= COMPLETE_DAY_HOURS:
                complete_days[sym].append(d)
            del df

    distributions: dict[str, dict[str, dict]] = {}
    drift_flags: list[dict] = []
    for sym in args.symbols:
        if not sample_frames[sym]:
            continue
        all_samples = pd.concat(sample_frames[sym], ignore_index=True)
        distributions[sym] = {}
        for c in all_samples.columns:
            v = all_samples[c].dropna().values
            if len(v) == 0:
                continue
            p = np.percentile(v, [1, 25, 50, 75, 99])
            distributions[sym][c] = {
                "mean": float(np.mean(v)), "std": float(np.std(v)),
                "p1": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
                "p75": float(p[3]), "p99": float(p[4]), "n": int(len(v)),
            }
        # Drift: z-score of last day's mean vs prior complete days
        sym_days = sorted(daily_means.get(sym, {}).keys())
        if len(sym_days) >= 4:
            last = sym_days[-1]
            baseline_days = [d for d in sym_days[:-1] if d in complete_days[sym]]
            if len(baseline_days) >= 3:
                for c in distributions[sym]:
                    base = np.array([daily_means[sym][d].get(c, np.nan)
                                     for d in baseline_days])
                    base = base[~np.isnan(base)]
                    cur = daily_means[sym][last].get(c)
                    if len(base) < 3 or cur is None or np.isnan(cur):
                        continue
                    sd = np.std(base, ddof=1)
                    if sd < 1e-15:
                        continue
                    z = (cur - np.mean(base)) / sd
                    if abs(z) > 3:
                        drift_flags.append({"symbol": sym, "feature": c,
                                            "z": round(float(z), 2),
                                            "last_mean": cur,
                                            "baseline_mean": float(np.mean(base))})
    drift_flags.sort(key=lambda f: -abs(f["z"]))

    # 3. IC screening
    ic = _ic_screen(args)

    return {"nan_coverage": cat_day, "uncategorized_cols": sorted(uncategorized),
            "distributions": distributions, "drift_flags": drift_flags,
            "daily_means": daily_means, "ic_screen": ic, "stride": stride}


# ── Section: gauntlet (algorithm performance) ─────────────────────────────

def _gauntlet_running() -> int | None:
    if GAUNTLET_PID.exists():
        try:
            pid = int(GAUNTLET_PID.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            return None
    return None


def _parse_gauntlet(symbols: list[str]) -> dict | None:
    try:
        with open(GAUNTLET_JSON) as f:
            gj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    daily = gj.get("daily", [])
    if not daily:
        return None

    dates = [r["date"] for r in daily]
    algos = sorted({a for r in daily for a in r["algorithms"]})
    algo_daily: dict[str, list[float]] = {}
    summary = []
    for algo in algos:
        per_symbol = {}
        all_daily = np.zeros(len(daily))
        trades = 0
        for s in symbols:
            arr = np.array([r["algorithms"].get(algo, {}).get(s, {})
                            .get("total_net_bps", 0.0) for r in daily])
            all_daily += arr
            t = sum(r["algorithms"].get(algo, {}).get(s, {}).get("trades", 0)
                    for r in daily)
            trades += t
            per_symbol[s] = {"total_bps": round(float(np.sum(arr)), 1),
                             "sharpe": round(_sharpe(arr), 2), "trades": int(t)}
        algo_daily[algo] = [round(float(x), 2) for x in all_daily]
        summary.append({
            "algo": algo,
            "total_bps": round(float(np.sum(all_daily)), 1),
            "sharpe": round(_sharpe(all_daily), 2),
            "win_pct": round(float(np.mean(all_daily > 0) * 100), 0),
            "trades": int(trades),
            "per_symbol": per_symbol,
        })
    summary.sort(key=lambda r: -r["sharpe"])

    return {"generated": gj.get("generated"), "gauntlet_status": gj.get("status"),
            "dates_tested": gj.get("dates_tested"), "dates_total": gj.get("dates_total"),
            "cost_bps_rt": gj.get("cost_bps_rt"), "elapsed_min": gj.get("elapsed_min"),
            "dates": dates, "summary": summary, "algo_daily": algo_daily,
            "per_date": [{"date": r["date"], "hours": r.get("hours"),
                          "n_algos": len(r["algorithms"])} for r in daily]}


def section_gauntlet(state: dict, args) -> dict:
    pid = _gauntlet_running()
    if args.skip_gauntlet or pid is not None:
        why = "--skip-gauntlet" if args.skip_gauntlet else f"gauntlet already running (PID {pid})"
        parsed = _parse_gauntlet(args.symbols)
        if parsed is None:
            return {"source": f"skipped ({why}); no cached gauntlet_latest.json"}
        parsed["source"] = f"cached gauntlet_latest.json ({why})"
        return parsed

    cmd = [sys.executable, str(ROOT / "scripts" / "alpha" / "overnight_sweep.py"),
           "run", "--data-dir", args.data_dir, "--symbols", *args.symbols]
    if args.last:
        cmd += ["--last", str(args.last)]
    _log(f"  [gauntlet] launching: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    parsed = _parse_gauntlet(args.symbols)
    if parsed is None:
        return {"source": f"run exited {r.returncode}, no parsable results"}
    parsed["source"] = "fresh run" if r.returncode == 0 else \
        f"run exited {r.returncode} — partial results"
    parsed["partial"] = r.returncode != 0 or parsed.get("gauntlet_status") != "complete"
    return parsed


# ── Section: visualizations ───────────────────────────────────────────────

def _fig_price_overview(args, window) -> str:
    apply_style()
    data_dir = _resolve_data_dir(args)
    # load_bars always validates; a column-subset load trips a spurious
    # "schema drift" warning, so mark this data_dir as already validated
    from data import features as _feat
    _feat._session_validated.add(str(data_dir))
    fig, axes = plt.subplots(len(args.symbols), 1, figsize=(13, 3 * len(args.symbols)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    spec = {"timestamp_ns": ("timestamp_ns", "first"),
            "price": ("raw_midprice", "last"),
            "volume": ("flow_volume_30s", "mean")}
    for ax, sym in zip(axes, args.symbols):
        bars = load_bars(symbols=[sym], date_range=(window[0], window[-1]),
                         bar_seconds=300, agg_spec=spec, data_dir=data_dir)
        if bars.empty:
            ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                    transform=ax.transAxes, color="#8b949e")
            ax.set_title(sym)
            continue
        t = pd.to_datetime(bars["timestamp_ns"], unit="ns")
        ax.plot(t, bars["price"], color=COLORS[0], lw=0.9)
        ax.set_ylabel(f"{sym} mid")
        ax2 = ax.twinx()
        ax2.fill_between(t, bars["volume"], color=COLORS[3], alpha=0.25, lw=0)
        ax2.set_yticks([])
        ax.set_title(f"{sym} — 5min bars (volume shaded)", fontsize=10)
        ax.grid(True, alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_equity(gsec: dict) -> str | None:
    if not gsec.get("summary"):
        return None
    apply_style()
    top = [r["algo"] for r in gsec["summary"][:8]]
    dates = gsec["dates"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, algo in enumerate(top):
        eq = np.cumsum(gsec["algo_daily"][algo])
        ax.plot(range(len(dates)), eq, label=algo, color=COLORS[i % len(COLORS)],
                lw=1.5, marker="o", ms=3,
                ls="-" if i < len(COLORS) else "--")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cumulative net bps (all symbols)")
    ax.set_title("Equity curves — top 8 algorithms by Sharpe", fontsize=11)
    ax.axhline(0, color="#8b949e", lw=0.7)
    ax.legend(fontsize=8, ncol=2, framealpha=0.2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_daily_pnl(gsec: dict) -> str | None:
    if not gsec.get("summary"):
        return None
    apply_style()
    top = [r["algo"] for r in gsec["summary"][:4]]
    dates = gsec["dates"]
    x = np.arange(len(dates))
    width = 0.8 / len(top)
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, algo in enumerate(top):
        ax.bar(x + i * width - 0.4 + width / 2, gsec["algo_daily"][algo],
               width=width, label=algo, color=COLORS[i % len(COLORS)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("daily net bps")
    ax.set_title("Daily PnL — top 4 algorithms", fontsize=11)
    ax.axhline(0, color="#8b949e", lw=0.7)
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(True, alpha=0.4, axis="y")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_nan_heatmap(fsec: dict) -> str | None:
    cov = fsec.get("nan_coverage", {})
    if not cov:
        return None
    apply_style()
    dates = sorted(cov.keys())
    base_cats = list(BASE_FEATURES.keys())
    opt_cats = list(OPTIONAL_FEATURES.keys())
    seen = {c for d in dates for c in cov[d]}
    cats = ([c for c in base_cats if c in seen] + [c for c in opt_cats if c in seen]
            + (["uncategorized"] if "uncategorized" in seen else []))
    mat = np.full((len(cats), len(dates)), np.nan)
    for j, d in enumerate(dates):
        for i, c in enumerate(cats):
            if c in cov[d]:
                mat[i, j] = cov[d][c]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(dates) + 3), 0.38 * len(cats) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    for i in range(len(cats)):
        for j in range(len(dates)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="#0d1117")
    ax.set_title("Non-NaN coverage %% by feature category x day", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_wiring_hists(wsec: dict, category: str) -> str:
    apply_style()
    cols = OPTIONAL_FEATURES[category]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.2 * nrows))
    for ax, c in zip(np.ravel(axes), cols):
        h = wsec.get("histograms", {}).get(c)
        if h:
            edges = np.array(h["edges"])
            ax.bar(edges[:-1], h["counts"], width=np.diff(edges),
                   align="edge", color=COLORS[0], alpha=0.85)
        else:
            ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                    transform=ax.transAxes, color="#8b949e", fontsize=9)
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in np.ravel(axes)[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"{category} feature distributions (non-NaN values)", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


def section_viz(state: dict, args) -> dict:
    window = _window_dates(args)
    figures = state.setdefault("figures", {})
    produced = []
    if window:
        figures["price_overview"] = _fig_price_overview(args, window)
        produced.append("price_overview")
    gsec = state["sections"].get("gauntlet", {})
    for name, fn in (("equity_curves", _fig_equity), ("daily_pnl", _fig_daily_pnl)):
        img = fn(gsec)
        if img:
            figures[name] = img
            produced.append(name)
    fsec = state["sections"].get("features", {})
    img = _fig_nan_heatmap(fsec)
    if img:
        figures["nan_heatmap"] = img
        produced.append("nan_heatmap")
    wsec = state["sections"].get("wiring", {})
    if wsec.get("columns"):
        for cat in NAN_WIRING_CATEGORIES:
            figures[f"wiring_hist_{cat}"] = _fig_wiring_hists(wsec, cat)
            produced.append(f"wiring_hist_{cat}")
    return {"produced": produced}


# ── HTML rendering ────────────────────────────────────────────────────────

_CSS = """
body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; margin:0; padding:24px; }
h1 { font-size:22px; } h2 { font-size:17px; border-bottom:1px solid #30363d; padding-bottom:6px; margin-top:36px; }
.panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 18px; margin:12px 0; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; margin-left:8px; }
.ok { background:#1f6feb33; color:#58a6ff; } .good { background:#23863633; color:#3fb950; }
.warn { background:#9e6a0333; color:#d29922; } .bad { background:#da363333; color:#f85149; }
.banner { padding:10px 16px; border-radius:8px; margin:12px 0; font-weight:600; }
.banner.bad { background:#da363322; border:1px solid #f85149; }
.banner.warn { background:#9e6a0322; border:1px solid #d29922; }
table { border-collapse:collapse; font-size:12.5px; margin:10px 0; }
th, td { border:1px solid #30363d; padding:4px 10px; text-align:right; }
th { background:#21262d; color:#8b949e; } td:first-child, th:first-child { text-align:left; }
td.good { color:#3fb950; } td.bad { color:#f85149; } td.warn { color:#d29922; }
img { max-width:100%; border:1px solid #30363d; border-radius:6px; margin:8px 0; }
.meta { color:#8b949e; font-size:12px; } code { background:#21262d; padding:1px 5px; border-radius:4px; }
"""


def _esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _tbl(headers: list[str], rows: list[list]) -> str:
    """rows cells are either values or (value, css_class) tuples."""
    out = ["<table><tr>" + "".join(f"<th>{_esc(h)}</th>" for h in headers) + "</tr>"]
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, tuple):
                cells.append(f'<td class="{cell[1]}">{_esc(cell[0])}</td>')
            else:
                cells.append(f"<td>{_esc(cell)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _img(state: dict, name: str) -> str:
    b64 = state.get("figures", {}).get(name)
    if not b64:
        return '<div class="meta">(figure not generated)</div>'
    return f'<img src="data:image/png;base64,{b64}" alt="{_esc(name)}">'


def _section_badge(sec: dict | None) -> str:
    if sec is None:
        return '<span class="badge warn">pending</span>'
    st = sec.get("status", "?")
    cls = {"ok": "good", "error": "bad", "skipped": "warn"}.get(st, "warn")
    el = sec.get("elapsed_s")
    el_txt = f" · {el:.0f}s" if isinstance(el, (int, float)) else ""
    return f'<span class="badge {cls}">{_esc(st)}{el_txt}</span>'


def _render_health(state: dict) -> str:
    sec = state["sections"].get("health")
    if not sec or sec.get("status") != "ok":
        return ""
    rows = []
    for d, syms in sec.get("days", {}).items():
        for s, v in syms.items():
            hours_cls = "good" if v["hours"] >= COMPLETE_DAY_HOURS else \
                ("warn" if v["hours"] >= 4 else "bad")
            rate_cls = "good" if v["rows_per_hour"] >= MIN_ROWS_PER_HOUR else "warn"
            gap_txt = "; ".join(f"{g['seconds']:.0f}s @ {g['start'][11:19]}"
                                for g in v["gaps"][:3]) or "—"
            rows.append([d, s, f"{v['rows']:,}", (f"{v['hours']:.1f}", hours_cls),
                         (f"{v['rows_per_hour']:,}", rate_cls), v["n_gaps"], gap_txt])
    html = _tbl(["Date", "Sym", "Rows", "Hours", "Rows/h", f"Gaps >{GAP_SECONDS:.0f}s",
                 "Top gaps"], rows)
    if sec.get("missing_days"):
        html += (f'<div class="banner warn">No data directories for: '
                 f'{_esc(", ".join(sec["missing_days"]))}</div>')
    html += _img(state, "price_overview")
    return html


def _render_wiring(state: dict) -> str:
    sec = state["sections"].get("wiring")
    if not sec or sec.get("status") != "ok":
        return ""
    t = sec["tracker"]
    v = sec["verdict"]
    v_cls = {"viable": "good", "noisy": "warn", "unavailable": "bad"}.get(v, "warn")
    ls = sec["log_scan"]
    users = {True: "YES — wallet discovery active", False: "NO — discovery unavailable",
             None: "no marker found in logs"}[ls.get("users_populated")]
    html = (
        f'<div class="panel">Verdict: <span class="badge {v_cls}">{_esc(v)}</span> '
        f'<span class="meta">{_esc(sec["verdict_reason"])}</span><br>'
        f'<span class="meta">position_tracker: '
        f'{"ENABLED" if t["enabled"] else "disabled"} '
        f'({_esc(t.get("note", "config/ing.toml"))}) · WsTrade.users populated: '
        f'{_esc(users)} · log: {_esc(ls.get("log_file") or "n/a")}</span></div>'
    )
    day_rows = [[d] + [f"{v.get(c, 0):.1f}%" for c in NAN_WIRING_CATEGORIES]
                for d, v in sec["per_day"].items()]
    html += _tbl(["Date"] + [f"{c} cov" for c in NAN_WIRING_CATEGORIES], day_rows)
    col_rows = []
    for c, e in sec["columns"].items():
        cov = e["coverage_pct"]
        cov_cls = "good" if cov > 50 else ("warn" if cov > 0.1 else "bad")
        rng = "—"
        rng_cls = ""
        if "expected_range" in e:
            lo, hi = e["expected_range"]
            rng = f"[{lo}, {hi}] → {e['pct_in_range']:.0f}% in"
            rng_cls = "good" if e["pct_in_range"] >= 80 else "warn"
        col_rows.append([
            c, e["category"], (f"{cov:.2f}%", cov_cls),
            f"{e.get('min', float('nan')):.4g}" if "min" in e else "—",
            f"{e.get('median', float('nan')):.4g}" if "median" in e else "—",
            f"{e.get('max', float('nan')):.4g}" if "max" in e else "—",
            (rng, rng_cls) if rng_cls else rng,
        ])
    html += _tbl(["Column", "Category", "Coverage", "Min", "Median", "Max",
                  "Expected range"], col_rows)
    for cat in NAN_WIRING_CATEGORIES:
        html += _img(state, f"wiring_hist_{cat}")
    return html


def _render_features(state: dict) -> str:
    sec = state["sections"].get("features")
    if not sec or sec.get("status") != "ok":
        return ""
    html = _img(state, "nan_heatmap")
    if sec.get("uncategorized_cols"):
        html += (f'<div class="meta">Columns not in scripts/data/schema.py: '
                 f'{_esc(", ".join(sec["uncategorized_cols"][:20]))}</div>')
    flags = sec.get("drift_flags", [])
    if flags:
        html += "<h3>Drift flags (|z| &gt; 3 vs prior complete days)</h3>"
        html += _tbl(["Sym", "Feature", "z", "Last mean", "Baseline mean"],
                     [[f["symbol"], f["feature"], (f"{f['z']:+.1f}", "bad"),
                       f"{f['last_mean']:.4g}", f"{f['baseline_mean']:.4g}"]
                      for f in flags[:25]])
    else:
        html += '<div class="panel good">No drift flags (|z| &gt; 3).</div>'
    ic = sec.get("ic_screen", {})
    html += (f'<h3>Top IC features</h3><div class="meta">source: '
             f'{_esc(ic.get("source", "?"))} · screened: {_esc(ic.get("n_tested", "?"))} '
             f'· significant: {_esc(ic.get("n_significant", "?"))}</div>')
    if ic.get("top"):
        html += _tbl(["Feature", "Sym", "Horizon", "IC", "IC IR", "p(adj)",
                      "Turnover", "Breakeven bps"],
                     [[r["feature"], r["symbol"], r["horizon"],
                       f"{r['ic_mean']:+.3f}", f"{r['ic_ir']:+.2f}",
                       f"{r['ic_p_adjusted']:.3f}", f"{r['turnover']:.2f}",
                       f"{r['breakeven_bps']:.1f}"] for r in ic["top"]])
    # Distribution summary for the first symbol (full stats in the JSON sidecar)
    dists = sec.get("distributions", {})
    if dists:
        sym = sorted(dists.keys())[0]
        html += (f"<h3>Key feature distributions — {sym} "
                 f'<span class="meta">(other symbols in JSON sidecar; '
                 f'subsample stride {sec.get("stride")})</span></h3>')
        html += _tbl(["Feature", "Mean", "Std", "p1", "p50", "p99"],
                     [[c, f"{d['mean']:.4g}", f"{d['std']:.4g}", f"{d['p1']:.4g}",
                       f"{d['p50']:.4g}", f"{d['p99']:.4g}"]
                      for c, d in dists[sym].items()])
    return html


def _render_gauntlet(state: dict) -> str:
    sec = state["sections"].get("gauntlet")
    if not sec or sec.get("status") != "ok":
        return ""
    html = (f'<div class="meta">source: {_esc(sec.get("source", "?"))} · '
            f'dates: {_esc(sec.get("dates_tested", "?"))}/{_esc(sec.get("dates_total", "?"))} '
            f'· cost: {_esc(sec.get("cost_bps_rt", "?"))} bps RT · '
            f'gauntlet elapsed: {_esc(sec.get("elapsed_min", "?"))} min</div>')
    if sec.get("partial"):
        html += '<div class="banner warn">Partial gauntlet results.</div>'
    if not sec.get("summary"):
        return html + '<div class="panel warn">No gauntlet results available.</div>'
    syms = sorted(sec["summary"][0]["per_symbol"].keys())
    headers = ["Algorithm", "Σ bps", "Sharpe", "Win%", "Trades"] + \
        [f"{s} bps / SR" for s in syms]
    rows = []
    for r in sec["summary"]:
        cls = "good" if r["sharpe"] > 1 else ("bad" if r["sharpe"] < 0 else "")
        row = [r["algo"], (f"{r['total_bps']:+.1f}", cls),
               (f"{r['sharpe']:+.2f}", cls), f"{r['win_pct']:.0f}%",
               f"{r['trades']:,}"]
        for s in syms:
            ps = r["per_symbol"][s]
            row.append(f"{ps['total_bps']:+.0f} / {ps['sharpe']:+.1f}")
        rows.append(row)
    html += _tbl(headers, rows)
    html += _img(state, "equity_curves")
    html += _img(state, "daily_pnl")
    return html


def render_html(state: dict) -> str:
    status = state.get("status", "?")
    status_cls = {"complete": "good", "running": "ok",
                  "interrupted": "warn"}.get(status, "bad")
    hdr = state.get("header", {})
    fresh = hdr.get("freshness_seconds")
    banner = ""
    if fresh is None:
        banner = '<div class="banner bad">No data freshness info — is there any data?</div>'
    elif fresh > 600:
        banner = (f'<div class="banner bad">Ingestor data is stale: last tick '
                  f'{fresh / 3600:.1f}h ago.</div>')
    errors = [n for n, s in state.get("sections", {}).items()
              if s.get("status") == "error"]
    if errors:
        banner += (f'<div class="banner warn">Sections with errors: '
                   f'{_esc(", ".join(errors))} — details in their panels.</div>')

    sections_html = []
    renderers = {"health": ("Data Health", _render_health),
                 "wiring": ("NaN Wiring / Position Tracker", _render_wiring),
                 "features": ("Feature Statistics", _render_features),
                 "gauntlet": ("Algorithm Performance (Gauntlet OOS)", _render_gauntlet)}
    for key, (title, fn) in renderers.items():
        sec = state.get("sections", {}).get(key)
        body = fn(state)
        if sec and sec.get("status") == "error":
            body = (f'<div class="panel bad"><b>Section failed:</b> '
                    f'{_esc(sec.get("error", "?"))}<br>'
                    f'<pre class="meta">{_esc(sec.get("traceback", ""))[:2000]}</pre></div>')
        elif sec is None:
            body = '<div class="meta">Not run yet.</div>'
        sections_html.append(f"<h2>{title}{_section_badge(sec)}</h2>{body}")

    win = state.get("header", {}).get("window", [])
    win_txt = f"{win[0]} → {win[-1]}" if win else "?"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NAT nightly {_esc(state['date'])}</title>
<style>{_CSS}</style></head><body>
<h1>NAT Nightly Report — {_esc(state['date'])}
<span class="badge {status_cls}">{_esc(status)}</span></h1>
<div class="meta">generated {_esc(state.get('generated', '?'))} ·
window {_esc(win_txt)} · elapsed {state.get('elapsed_min', 0):.1f} min ·
symbols {_esc(', '.join(state.get('args', {}).get('symbols', [])))} ·
rows on disk {hdr.get('total_rows', 0):,} · latest tick {_esc(hdr.get('latest_timestamp', '?'))}</div>
{banner}
{''.join(sections_html)}
<div class="meta" style="margin-top:30px">JSON sidecar:
<code>reports/nightly/{_esc(state['date'])}.json</code> ·
generated by <code>nat nightly</code></div>
</body></html>"""


# ── run / report / open ───────────────────────────────────────────────────

SECTIONS = {
    "health": section_health,
    "wiring": section_wiring,
    "features": section_features,
    "gauntlet": section_gauntlet,
    "viz": section_viz,
}


def _handle_signal(signum, frame):
    _log(f"\n  Caught signal {signum} — flushing partial report")
    if _CURRENT is not None:
        _CURRENT["status"] = "interrupted"
        _CURRENT["elapsed_min"] = round((time.time() - _CURRENT["_t0"]) / 60, 1)
        _CURRENT.pop("_t0", None)
        _save_state(_CURRENT)
    PID_FILE.unlink(missing_ok=True)
    sys.exit(130)


def cmd_run(args) -> int:
    global _CURRENT

    # Refuse double-start (stale PID self-clears)
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Nightly report already running (PID {pid}). Aborting.")
            return 1
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    wanted = SECTION_ORDER if not args.sections else \
        [s for s in SECTION_ORDER if s in args.sections.split(",")]

    report_date = args.date or datetime.now().strftime("%Y-%m-%d")
    t0 = time.time()
    state = {
        "date": report_date,
        "generated": _now_iso(),
        "status": "running",
        "args": {"last": args.last, "quick": args.quick,
                 "skip_gauntlet": args.skip_gauntlet, "symbols": args.symbols,
                 "sections": wanted},
        "header": {**_header_snapshot(_resolve_data_dir(args)),
                   "window": _window_dates(args)},
        "elapsed_min": 0.0,
        "sections": {},
        "figures": {},
    }
    state["_t0"] = t0
    _CURRENT = state

    _log(f"Nightly Report — {report_date}")
    _log(f"  PID: {os.getpid()}")
    _log(f"  Window: {state['header']['window'] or 'EMPTY'}")
    _log(f"  Sections: {wanted}")
    _save_state(state)  # skeleton report exists from minute 0

    for name in wanted:
        _log(f"\n[{name}] starting")
        ts = time.time()
        try:
            result = SECTIONS[name](state, args)
            result["status"] = result.get("status", "ok")
            if name == "gauntlet" and result.get("source", "").startswith("skipped"):
                result["status"] = "skipped"
        except Exception as e:  # noqa: BLE001 — every section must be non-fatal
            result = {"status": "error", "error": str(e),
                      "traceback": traceback.format_exc()}
            _log(f"[{name}] ERROR: {e}")
        result["elapsed_s"] = round(time.time() - ts, 1)
        state["sections"][name] = result
        state["elapsed_min"] = round((time.time() - t0) / 60, 1)
        _save_state(state)
        _log(f"[{name}] done in {result['elapsed_s']:.0f}s ({result['status']})")

    state["status"] = "complete"
    state["generated"] = _now_iso()
    state["elapsed_min"] = round((time.time() - t0) / 60, 1)
    state.pop("_t0", None)
    _save_state(state)
    PID_FILE.unlink(missing_ok=True)

    html_path = NIGHTLY_DIR / f"{report_date}.html"
    _log(f"\nDone in {state['elapsed_min']:.1f} min")
    _log(f"  HTML: {html_path}")
    _log(f"  JSON: {NIGHTLY_DIR / (report_date + '.json')}")
    n_err = sum(1 for s in state["sections"].values() if s["status"] == "error")
    if n_err:
        _log(f"  WARNING: {n_err} section(s) errored — see report panels")
    return 0


def _latest(suffix: str) -> Path | None:
    if not NIGHTLY_DIR.exists():
        return None
    files = sorted(NIGHTLY_DIR.glob(f"*.{suffix}"))
    return files[-1] if files else None


def cmd_report(args) -> int:
    path = _latest("json")
    if path is None:
        print("  No nightly reports found. Run `nat nightly` first.")
        return 1
    with open(path) as f:
        state = json.load(f)
    print(f"\n  Nightly report {state['date']} — {state['status']} "
          f"({state.get('elapsed_min', 0):.1f} min)")
    print(f"  Generated: {state.get('generated', '?')}")
    for name in SECTION_ORDER:
        sec = state.get("sections", {}).get(name)
        if sec is None:
            print(f"    {name:<10s} —")
            continue
        extra = ""
        if name == "wiring" and "verdict" in sec:
            extra = f" verdict={sec['verdict']}"
        if name == "features":
            extra = f" drift_flags={len(sec.get('drift_flags', []))}"
        if name == "gauntlet" and sec.get("summary"):
            best = sec["summary"][0]
            extra = f" best={best['algo']} (SR {best['sharpe']:+.2f})"
        print(f"    {name:<10s} {sec.get('status', '?'):<8s}"
              f" {sec.get('elapsed_s', 0):>6.0f}s{extra}")
    print(f"\n  HTML: {path.with_suffix('.html')}")
    print(f"  JSON: {path}\n")
    return 0


def cmd_open(args) -> int:
    path = _latest("html")
    if path is None:
        print("  No nightly HTML found. Run `nat nightly` first.")
        return 1
    print(f"  Opening {path}")
    subprocess.Popen(["xdg-open", str(path)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Nightly report — overnight stats + performance + viz")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the full nightly pass")
    run_p.add_argument("--last", type=int, default=7,
                       help="Lookback: last N available dates (default 7)")
    run_p.add_argument("--data-dir", default="data/features")
    run_p.add_argument("--symbols", nargs="+", default=SYMBOLS_DEFAULT)
    run_p.add_argument("--quick", action="store_true",
                       help="Coarser subsampling, never launch the screener")
    run_p.add_argument("--skip-gauntlet", action="store_true",
                       help="Use cached gauntlet results instead of running the sweep")
    run_p.add_argument("--sections", type=str, default=None,
                       help=f"Comma list of sections to run ({','.join(SECTION_ORDER)})")
    run_p.add_argument("--date", type=str, default=None,
                       help="Override report date (default: today)")
    run_p.set_defaults(func=cmd_run)

    report_p = sub.add_parser("report", help="Print latest nightly summary")
    report_p.set_defaults(func=cmd_report)

    open_p = sub.add_parser("open", help="Open latest nightly HTML")
    open_p.set_defaults(func=cmd_open)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["run"] + sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

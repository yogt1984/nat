#!/usr/bin/env python3
"""nat oos --window <N>d — longitudinal validation of the deployable algorithms.

An analysis layer over the gauntlet's per-date daily P&L
(``reports/gauntlet_*.json``). For each (algorithm, symbol) over a trailing
window of trading days it computes:

  - annualized Sharpe (canonical ``utils.metrics.sharpe_daily``)
  - a daily-PnL walk-forward holdout: train on the first ``train_frac`` of the
    days, test on the rest -> IS Sharpe, OOS Sharpe, OOS/IS ratio (overfit signal)
  - max drawdown (bps), win rate, total bps, day count
  - deflated Sharpe probability (Bailey & Lopez de Prado, 2014) adjusted for the
    number of (algorithm, symbol) strategies searched -- HIGHER = more robust to
    multiple testing
  - cross-algorithm complementarity (pairwise daily-PnL correlation)

This does NOT re-run backtests; it reads the daily P&L the gauntlet already
produced. Run ``nat gauntlet run --last N`` first to (re)generate / extend it.

Metrics only -- pass/fail gates (G4) live in the alpha pipeline, not here.
Coverage is reported honestly: if fewer than the requested days are on disk,
that is surfaced, never silently truncated.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent  # .../scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from utils.metrics import sharpe_daily  # noqa: E402

# The deployable winners (see docs/STATE). Order is display order.
DEPLOYABLE = ["3f_liquidity", "jump_detector", "funding_reversion",
              "optimal_entry", "surprise_signal"]
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]
REPORTS_DIR = _SCRIPTS.parent / "reports"


# ── pure stats (planted-testable; no I/O) ────────────────────────────────────

def max_drawdown_bps(daily_pnl: np.ndarray) -> float:
    """Max peak-to-trough drawdown of cumulative daily P&L, in bps (>= 0)."""
    arr = np.asarray(daily_pnl, dtype=float)
    if arr.size == 0:
        return 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def walk_forward_holdout(daily_pnl, train_frac: float = 0.67) -> dict:
    """Single chronological holdout over daily P&L.

    Train on the first ``train_frac`` of days, test on the remainder. Returns
    annualized IS/OOS Sharpe and the OOS/IS ratio -- the standard overfit signal
    (a robust edge keeps OOS/IS >~ 0.7; an overfit one collapses out of sample).
    """
    arr = np.asarray(daily_pnl, dtype=float)
    n = arr.size
    out = {"n_train": 0, "n_test": 0, "is_sharpe": 0.0,
           "oos_sharpe": 0.0, "oos_is_ratio": float("nan")}
    if n < 4:
        return out
    k = int(round(n * train_frac))
    k = max(2, min(k, n - 2))  # keep >= 2 observations on each side
    train, test = arr[:k], arr[k:]
    is_s = float(sharpe_daily(train))
    oos_s = float(sharpe_daily(test))
    ratio = (oos_s / is_s) if abs(is_s) > 1e-9 else float("nan")
    out.update(n_train=int(train.size), n_test=int(test.size),
               is_sharpe=is_s, oos_sharpe=oos_s, oos_is_ratio=ratio)
    return out


def strategy_stats(daily_pnl, n_trials: int = 1, train_frac: float = 0.67) -> dict:
    """Full per-(algo, symbol) metric bundle over a window of daily P&L."""
    arr = np.asarray(daily_pnl, dtype=float)
    wf = walk_forward_holdout(arr, train_frac)
    stats = {
        "days": int(arr.size),
        "total_bps": float(np.sum(arr)) if arr.size else 0.0,
        "sharpe": float(sharpe_daily(arr)) if arr.size else 0.0,
        "win_pct": float(np.mean(arr > 0) * 100.0) if arr.size else 0.0,
        "max_dd_bps": max_drawdown_bps(arr),
        "walk_forward": wf,
    }
    # Deflated Sharpe (lazy import keeps the pure-stat functions dependency-light).
    # Bailey & Lopez de Prado (2014): probability the OOS Sharpe survives having
    # searched n_trials strategies. HIGHER = more robust. Reported, not gated.
    try:
        from backtest.walk_forward import compute_deflated_sharpe
        stats["deflated_sharpe_dsr"] = float(compute_deflated_sharpe(
            observed_sharpe=wf["oos_sharpe"], n_trials=max(3, int(n_trials))))
    except Exception as exc:  # pragma: no cover - defensive
        stats["deflated_sharpe_dsr"] = None
        stats["deflated_error"] = str(exc)
    return stats


def pairwise_correlations(series_by_key: dict) -> list:
    """Pairwise Pearson correlation of daily-P&L series (complementarity check)."""
    keys = [k for k, v in series_by_key.items()
            if np.asarray(v).size >= 3 and np.std(np.asarray(v, float)) > 1e-12]
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = np.asarray(series_by_key[keys[i]], float)
            b = np.asarray(series_by_key[keys[j]], float)
            n = min(a.size, b.size)
            if n >= 3:
                out.append({"pair": [keys[i], keys[j]],
                            "corr": float(np.corrcoef(a[:n], b[:n])[0, 1])})
    return out


# ── I/O: merge gauntlet daily P&L over the window ────────────────────────────

def load_daily_matrix(reports_dir: Path, window_days, symbols, algos):
    """Merge all gauntlet/overnight daily entries (dedup by date, latest run wins),
    take the last ``window_days`` dates, return (window_dates, all_dates, matrix)
    where matrix[algo][symbol] is a chronological np.array of daily net bps."""
    files = sorted(reports_dir.glob("gauntlet_*.json")) + \
        sorted(reports_dir.glob("overnight_*.json"))
    date_to_daily = {}
    for f in files:
        try:
            rep = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for d in rep.get("daily", []):
            if d.get("date"):
                date_to_daily[d["date"]] = d
    all_dates = sorted(date_to_daily)
    window_dates = all_dates[-window_days:] if window_days else all_dates
    matrix = {}
    for a in algos:
        matrix[a] = {}
        for s in symbols:
            series = []
            for date in window_dates:
                algorithms = date_to_daily[date].get("algorithms", {})
                cell = algorithms.get(a, {}).get(s, {})
                series.append(float(cell.get("total_net_bps", 0.0)))
            matrix[a][s] = np.array(series, dtype=float)
    return window_dates, all_dates, matrix


# ── orchestration ────────────────────────────────────────────────────────────

def build_report(window_days, symbols, algos, reports_dir, train_frac) -> dict:
    window_dates, all_dates, matrix = load_daily_matrix(
        reports_dir, window_days, symbols, algos)
    # n_trials = breadth of the search (multiple-testing correction).
    n_trials = max(3, len(algos) * len(symbols))
    results, portfolio_series = {}, {}
    for a in algos:
        results[a] = {}
        algo_daily = None
        for s in symbols:
            series = matrix[a][s]
            results[a][s] = strategy_stats(series, n_trials=n_trials,
                                           train_frac=train_frac)
            algo_daily = series.copy() if algo_daily is None else algo_daily + series
        portfolio_series[a] = algo_daily if algo_daily is not None else np.array([])
    return {
        "window_days_requested": window_days,
        "days_available": len(window_dates),
        "total_dates_on_disk": len(all_dates),
        "date_range": [window_dates[0], window_dates[-1]] if window_dates else [],
        "symbols": symbols,
        "algos": algos,
        "n_trials": n_trials,
        "train_frac": train_frac,
        "results": results,
        "complementarity": pairwise_correlations(portfolio_series),
        "note": ("Metrics only; G4 pass/fail lives in the alpha pipeline. "
                 "deflated_sharpe_dsr is Bailey-LdP: higher = more robust to "
                 "the n_trials strategies searched."),
    }


def _json_safe(obj):
    """Recursively replace non-finite floats with None so output is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _fmt_ratio(x):
    return "  n/a" if (x is None or (isinstance(x, float) and not math.isfinite(x))) else f"{x:+.2f}"


def print_human(report: dict) -> None:
    req = report["window_days_requested"]
    avail = report["days_available"]
    dr = report["date_range"]
    print(f"\n  Longitudinal OOS — last {req}d window "
          f"({avail} trading days available"
          + (f", {dr[0]}→{dr[1]}" if dr else "") + ")")
    if avail < (req or 0):
        print(f"  ⚠ only {avail}/{req} days on disk — run `nat gauntlet run --last {req}` "
              f"to extend (no data was fabricated)")
    print(f"  n_trials (multiple-testing breadth) = {report['n_trials']}; "
          f"walk-forward train_frac = {report['train_frac']}\n")
    header = (f"  {'algo':<18}{'sym':<5}{'days':>5}{'totBps':>9}"
              f"{'Sharpe':>8}{'OOS/IS':>8}{'maxDD':>8}{'win%':>7}{'DSR':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in report["algos"]:
        for s in report["symbols"]:
            st = report["results"][a][s]
            wf = st["walk_forward"]
            dsr = st.get("deflated_sharpe_dsr")
            dsr_s = "  n/a" if dsr is None else f"{dsr:.2f}"
            print(f"  {a:<18}{s:<5}{st['days']:>5}{st['total_bps']:>9.1f}"
                  f"{st['sharpe']:>8.2f}{_fmt_ratio(wf['oos_is_ratio']):>8}"
                  f"{st['max_dd_bps']:>8.1f}{st['win_pct']:>7.0f}{dsr_s:>7}")
    comp = report["complementarity"]
    if comp:
        print(f"\n  Complementarity (per-algo daily-P&L correlation; "
              f"<0.35 = diversifying):")
        for c in comp:
            flag = "" if abs(c["corr"]) < 0.35 else "  (correlated)"
            print(f"    {c['pair'][0]:<18} × {c['pair'][1]:<18} {c['corr']:+.2f}{flag}")
    print()


def _parse_window(w) -> int:
    s = str(w).strip().lower().rstrip("d")
    return int(s) if s else 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Longitudinal OOS validation over accumulated gauntlet daily P&L")
    ap.add_argument("--window", default="30d",
                    help="Trailing window in trading days, e.g. 30d or 30")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--algos", nargs="+", default=DEPLOYABLE,
                    help="Algorithms to analyze (default: the 5 deployables)")
    ap.add_argument("--reports-dir", default=str(REPORTS_DIR))
    ap.add_argument("--train-frac", type=float, default=0.67,
                    help="Walk-forward train fraction (default 0.67)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = ap.parse_args(argv)

    report = build_report(_parse_window(args.window), args.symbols, args.algos,
                          Path(args.reports_dir), args.train_frac)
    if args.json:
        print(json.dumps(_json_safe(report), indent=2))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

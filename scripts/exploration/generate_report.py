#!/usr/bin/env python3
"""
NAT Experiment Report Generator.

Loads all collected data, validates quality, runs profiling per symbol,
backtests strategies, and outputs a comprehensive markdown report with
clear next-step recommendations.

Usage:
    python3 scripts/generate_report.py
    python3 scripts/generate_report.py --symbols BTC ETH SOL
    python3 scripts/generate_report.py --timeframe 1h
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

# Setup path
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

DATA_DIR = PROJECT_ROOT / "data" / "features"
REPORT_DIR = PROJECT_ROOT / "reports"

from cluster_pipeline.loader import load_parquet
from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.hierarchy import profile
from cluster_pipeline.validate import validate


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _data_summary(df: pd.DataFrame) -> dict:
    """Compute basic data statistics."""
    info = {}
    info["total_rows"] = len(df)

    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"]
        start = pd.to_datetime(ts.min(), unit="ns")
        end = pd.to_datetime(ts.max(), unit="ns")
        info["start"] = str(start)
        info["end"] = str(end)
        info["duration_hours"] = round((ts.max() - ts.min()) / 1e9 / 3600, 1)
    else:
        info["start"] = "unknown"
        info["end"] = "unknown"
        info["duration_hours"] = 0

    if "symbol" in df.columns:
        info["symbols"] = sorted(df["symbol"].unique().tolist())
        info["rows_per_symbol"] = {
            sym: int((df["symbol"] == sym).sum()) for sym in info["symbols"]
        }
    else:
        info["symbols"] = ["unknown"]
        info["rows_per_symbol"] = {}

    # NaN stats
    nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    info["nan_ratio"] = round(nan_ratio, 4)

    # File count
    parquets = list(DATA_DIR.rglob("*.parquet"))
    info["n_files"] = len(parquets)
    info["disk_mb"] = round(sum(f.stat().st_size for f in parquets) / 1e6, 1)

    return info


def _quality_check(df: pd.DataFrame) -> dict:
    """Run basic data quality checks."""
    checks = {}

    # Gap detection
    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"].sort_values().values
        diffs = np.diff(ts) / 1e9  # seconds
        gaps = diffs[diffs > 5.0]
        checks["n_gaps_over_5s"] = len(gaps)
        checks["longest_gap_s"] = round(float(np.max(diffs)), 1) if len(diffs) > 0 else 0
        checks["median_interval_s"] = round(float(np.median(diffs)), 3) if len(diffs) > 0 else 0
    else:
        checks["n_gaps_over_5s"] = -1
        checks["longest_gap_s"] = -1

    # NaN per column (top offenders)
    nan_per_col = df.isnull().sum()
    nan_pct = (nan_per_col / len(df) * 100).sort_values(ascending=False)
    top_nan = nan_pct[nan_pct > 1.0].head(10)
    checks["columns_with_nan"] = {col: round(pct, 2) for col, pct in top_nan.items()}

    # Feature count
    numeric = df.select_dtypes(include=[np.number]).columns
    checks["n_numeric_features"] = len(numeric)

    return checks


def _profile_symbol(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """Run profiling pipeline for a single symbol."""
    result = {"symbol": symbol}

    sym_df = df[df["symbol"] == symbol].copy() if "symbol" in df.columns else df.copy()
    result["n_rows"] = len(sym_df)

    if len(sym_df) < 1000:
        result["status"] = "insufficient_data"
        result["verdict"] = "COLLECT"
        result["reason"] = f"Only {len(sym_df)} rows — need at least 1000"
        return result

    try:
        bars = aggregate_bars(sym_df, timeframe=timeframe)
        result["n_bars"] = len(bars)
    except Exception as e:
        result["status"] = "aggregation_failed"
        result["verdict"] = "ERROR"
        result["reason"] = str(e)
        return result

    if len(bars) < 50:
        result["status"] = "insufficient_bars"
        result["verdict"] = "COLLECT"
        result["reason"] = f"Only {len(bars)} bars — need at least 50"
        return result

    # Run profiling
    try:
        prof = profile(sym_df, vector="entropy", timeframe=timeframe, include_spectral=True)
    except Exception as e:
        result["status"] = "profiling_failed"
        result["verdict"] = "ERROR"
        result["reason"] = str(e)
        return result

    macro = prof.macro
    result["status"] = "complete"

    # Structure test
    if hasattr(macro, "structure_test") and macro.structure_test:
        result["hopkins"] = round(macro.structure_test.hopkins_statistic, 3)
        result["dip_p"] = round(macro.structure_test.dip_p_value, 4) if hasattr(macro.structure_test, "dip_p_value") else None
    else:
        result["hopkins"] = None

    if macro.early_exit:
        result["verdict"] = "DROP"
        result["reason"] = "No structure found (Hopkins/dip test failed)"
        result["k"] = 0
        return result

    result["k"] = macro.k
    result["silhouette"] = round(macro.quality.silhouette, 3)
    result["bootstrap_ari"] = round(macro.stability.mean_ari, 3)

    # Transitions
    if hasattr(macro, "transitions") and macro.transitions:
        tm = macro.transitions
        if hasattr(tm, "matrix"):
            diag = [tm.matrix[i][i] for i in range(len(tm.matrix))]
            result["self_transition_rate"] = round(float(np.mean(diag)), 3)
        if hasattr(tm, "durations") and tm.durations:
            all_durs = []
            for durs in tm.durations.values():
                all_durs.extend(durs)
            result["mean_duration"] = round(float(np.mean(all_durs)), 1) if all_durs else 0

    # State distribution
    hierarchy = prof.hierarchy
    states = []
    for i in range(hierarchy.n_micro_total):
        mask = hierarchy.micro_labels == i
        n_state = int(mask.sum())
        if n_state > 0:
            states.append({"id": i, "n_bars": n_state, "pct": round(n_state / len(bars), 2)})
    result["states"] = states

    # Validation
    try:
        price_col = "raw_midprice_mean" if "raw_midprice_mean" in prof.bars.columns else None
        prices = prof.bars[price_col].values if price_col else np.ones(len(prof.bars))
        verdict = validate(prof, prices)
        result["verdict"] = verdict.overall
        result["q1_pass"] = verdict.q1_structural["pass"]
        result["q2_pass"] = verdict.q2_predictive["pass"]
        result["q3_pass"] = verdict.q3_operational["pass"]
        result["summary"] = verdict.summary

        # Q2 detail
        kruskal = verdict.q2_predictive.get("kruskal_results", {})
        p_values = [v.get("p_value", 1.0) for v in kruskal.values() if isinstance(v, dict)]
        if p_values:
            result["q2_best_p"] = round(min(p_values), 4)

        result["reason"] = {
            "GO": "All gates passed. Clusters are real, predictive, and tradeable.",
            "PIVOT": "Clusters predict returns but don't persist long enough. Try longer bars.",
            "COLLECT": "Clusters exist but insufficient evidence of predictive power. More data needed.",
            "DROP": "No meaningful cluster structure found.",
        }.get(verdict.overall, verdict.summary)

    except Exception as e:
        result["verdict"] = "ERROR"
        result["reason"] = f"Validation failed: {e}"

    return result


def _backtest_funding(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """Run funding reversion backtest for a symbol."""
    from strategies.funding_reversion import FundingReversion

    sym_df = df[df["symbol"] == symbol].copy() if "symbol" in df.columns else df.copy()

    try:
        bars = aggregate_bars(sym_df, timeframe=timeframe)
    except Exception:
        return {"symbol": symbol, "status": "failed", "reason": "aggregation error"}

    price_col = None
    for col in ["raw_midprice_mean", "raw_midprice"]:
        if col in bars.columns:
            price_col = col
            break

    if price_col is None or len(bars) < 20:
        return {"symbol": symbol, "status": "insufficient"}

    prices = bars[price_col].values.astype(np.float64)
    strat = FundingReversion(zscore_entry=1.5, zscore_exit=0.3, max_position=1.0)
    features = strat.compute_features(bars)
    signal = strat.generate_signal(features)

    sig = signal.values[:-1]
    returns = np.diff(prices) / prices[:-1]
    strat_returns = sig * returns

    cost_per_trade = 0.00035
    position_changes = np.abs(np.diff(np.concatenate([[0], sig])))
    costs = position_changes * cost_per_trade
    net_returns = strat_returns - costs

    valid = ~np.isnan(net_returns)
    nr = net_returns[valid]

    if len(nr) == 0:
        return {"symbol": symbol, "status": "no_valid_returns"}

    total = np.prod(1 + nr) - 1
    gross = np.prod(1 + strat_returns[valid]) - 1
    std = np.std(nr)
    sharpe = (np.mean(nr) / std * np.sqrt(96 * 365)) if std > 0 else 0
    n_trades = int(np.sum(position_changes[valid] > 0.01))
    time_in = np.mean(np.abs(sig[valid]) > 0.01)

    return {
        "symbol": symbol,
        "status": "complete",
        "n_bars": int(valid.sum()),
        "gross_return_pct": round(gross * 100, 4),
        "net_return_pct": round(total * 100, 4),
        "sharpe": round(sharpe, 3),
        "n_trades": n_trades,
        "time_in_market": round(time_in, 3),
    }


def generate_report(timeframe: str = "15min", symbols: list = None) -> str:
    """Generate the full experiment report."""
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# NAT Experiment Report")
    lines.append(f"")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Timeframe:** {timeframe}")

    # --- Load data ---
    print("[1/5] Loading data...")
    try:
        df = load_parquet(str(DATA_DIR))
    except Exception as e:
        lines.append(f"\n**FATAL:** Could not load data: {e}")
        return "\n".join(lines)

    data_info = _data_summary(df)
    if symbols is None:
        symbols = data_info["symbols"]

    lines.append(f"**Symbols:** {', '.join(symbols)}")
    lines.append("")

    # --- Data Summary ---
    lines.append(_section("1. Data Summary"))
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total rows | {data_info['total_rows']:,} |")
    lines.append(f"| Duration | {data_info['duration_hours']} hours |")
    lines.append(f"| Date range | {data_info['start']} to {data_info['end']} |")
    lines.append(f"| Files | {data_info['n_files']} |")
    lines.append(f"| Disk | {data_info['disk_mb']} MB |")
    lines.append(f"| NaN ratio | {data_info['nan_ratio']:.2%} |")

    for sym, count in data_info.get("rows_per_symbol", {}).items():
        lines.append(f"| {sym} rows | {count:,} |")

    # --- Quality ---
    print("[2/5] Running quality checks...")
    quality = _quality_check(df)
    lines.append(_section("2. Data Quality"))
    lines.append(f"| Check | Result |")
    lines.append(f"|-------|--------|")
    lines.append(f"| Gaps > 5s | {quality['n_gaps_over_5s']} |")
    lines.append(f"| Longest gap | {quality['longest_gap_s']}s |")
    lines.append(f"| Median interval | {quality['median_interval_s']}s |")
    lines.append(f"| Numeric features | {quality['n_numeric_features']} |")

    if quality["columns_with_nan"]:
        lines.append(f"")
        lines.append(f"**Columns with >1% NaN:**")
        for col, pct in quality["columns_with_nan"].items():
            lines.append(f"- `{col}`: {pct}%")

    # --- Profiling per symbol ---
    print("[3/5] Running profiling per symbol...")
    lines.append(_section("3. Regime Profiling"))

    profiling_results = {}
    for sym in symbols:
        print(f"  Profiling {sym}...")
        result = _profile_symbol(df, sym, timeframe)
        profiling_results[sym] = result

        lines.append(f"")
        lines.append(f"### {sym}")
        lines.append(f"")

        if result.get("status") != "complete":
            lines.append(f"**Status:** {result.get('status', 'unknown')}")
            lines.append(f"**Verdict:** {result.get('verdict', 'N/A')}")
            lines.append(f"**Reason:** {result.get('reason', 'N/A')}")
            continue

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Bars | {result.get('n_bars', '?')} |")
        lines.append(f"| Hopkins | {result.get('hopkins', '?')} |")
        lines.append(f"| k (optimal clusters) | {result.get('k', '?')} |")
        lines.append(f"| Silhouette | {result.get('silhouette', '?')} |")
        lines.append(f"| Bootstrap ARI | {result.get('bootstrap_ari', '?')} |")
        lines.append(f"| Self-transition rate | {result.get('self_transition_rate', '?')} |")
        lines.append(f"| Mean duration (bars) | {result.get('mean_duration', '?')} |")
        lines.append(f"| Q1 (structural) | {'PASS' if result.get('q1_pass') else 'FAIL'} |")
        lines.append(f"| Q2 (predictive) | {'PASS' if result.get('q2_pass') else 'FAIL'} |")
        lines.append(f"| Q3 (operational) | {'PASS' if result.get('q3_pass') else 'FAIL'} |")
        lines.append(f"| **Verdict** | **{result.get('verdict', '?')}** |")

        if result.get("states"):
            lines.append(f"")
            lines.append(f"**States:**")
            lines.append(f"| State | Bars | % |")
            lines.append(f"|-------|------|---|")
            for s in result["states"]:
                lines.append(f"| {s['id']} | {s['n_bars']} | {s['pct']:.0%} |")

    # --- Backtest ---
    print("[4/5] Running strategy backtests...")
    lines.append(_section("4. Strategy Backtests"))
    lines.append(f"")
    lines.append(f"### Funding Rate Mean-Reversion (z > 1.5)")
    lines.append(f"")
    lines.append(f"| Symbol | Gross | Net | Sharpe | Trades | Time in mkt |")
    lines.append(f"|--------|-------|-----|--------|--------|-------------|")

    for sym in symbols:
        print(f"  Backtesting {sym}...")
        bt = _backtest_funding(df, sym, timeframe)
        if bt.get("status") == "complete":
            lines.append(
                f"| {sym} | {bt['gross_return_pct']:+.4f}% | {bt['net_return_pct']:+.4f}% "
                f"| {bt['sharpe']:.3f} | {bt['n_trades']} | {bt['time_in_market']:.1%} |"
            )
        else:
            lines.append(f"| {sym} | — | — | — | — | {bt.get('status', 'failed')} |")

    # --- Recommendations ---
    print("[5/5] Generating recommendations...")
    lines.append(_section("5. Recommendations"))

    verdicts = [r.get("verdict", "ERROR") for r in profiling_results.values()]

    if all(v == "GO" for v in verdicts):
        lines.append(dedent("""
        **ALL SYMBOLS: GO**

        Clusters are real, predict returns, and persist long enough to trade.

        **Next steps:**
        1. Train regime-conditioned models (separate predictor per regime)
        2. Backtest with maker-only execution (0% fee on Hyperliquid)
        3. Run walk-forward validation across full dataset
        4. If net Sharpe > 1.0 after costs, proceed to paper trading
        """))

    elif any(v == "GO" for v in verdicts):
        go_syms = [s for s, r in profiling_results.items() if r.get("verdict") == "GO"]
        other_syms = [s for s, r in profiling_results.items() if r.get("verdict") != "GO"]
        lines.append(dedent(f"""
        **PARTIAL GO: {', '.join(go_syms)}**

        Some symbols show tradeable regime structure.

        **Next steps:**
        1. Focus strategy development on {', '.join(go_syms)}
        2. For {', '.join(other_syms)}: collect more data or try different timeframe
        3. Run regime-conditioned backtest on GO symbols only
        """))

    elif any(v == "PIVOT" for v in verdicts):
        lines.append(dedent("""
        **PIVOT NEEDED**

        Clusters are real and predict returns but don't persist long enough to trade at this timeframe.

        **Next steps:**
        1. Re-run with `--timeframe 1h` (4x longer bars = longer regime duration)
        2. If still PIVOT at 1h, try `--timeframe 4h`
        3. Consider using regime as a filter (not a signal) — only trade when in favorable regime
        """))

    elif any(v == "COLLECT" for v in verdicts):
        lines.append(dedent("""
        **COLLECT MORE DATA**

        Clusters exist structurally but insufficient evidence of predictive power.

        **Next steps:**
        1. Continue data collection for 7-14 more days
        2. Re-run this report after accumulating more data
        3. Check if data gaps or NaN issues are degrading quality
        """))

    elif all(v == "DROP" for v in verdicts):
        lines.append(dedent("""
        **DROP — No Structure Found**

        The profiling pipeline found no meaningful regime structure in any symbol.

        **Next steps:**
        1. Try different feature vectors: `--vector volatility` or `--vector trend`
        2. Try different timeframes: `--timeframe 5min` or `--timeframe 1h`
        3. Check data quality — are all feature categories populated?
        4. If nothing works after 3 attempts, the entropy-based regime hypothesis may not hold
        """))

    else:
        lines.append(dedent("""
        **ERRORS ENCOUNTERED**

        Some profiling runs failed. Check the per-symbol results above.

        **Next steps:**
        1. Fix any data loading or aggregation errors
        2. Ensure sufficient data per symbol (>1000 rows minimum)
        3. Re-run this report
        """))

    # --- Strategy-specific recommendations ---
    lines.append(f"### Strategy Notes")
    lines.append(f"")

    has_bt = any(
        _backtest_funding(df, s, timeframe).get("status") == "complete"
        for s in symbols
    )
    lines.append(dedent("""
    **Funding reversion:** Gross alpha is typically positive (funding extremes do revert),
    but transaction costs dominate at 15-min bars due to frequent signal flips.

    **If gross > 0 but net < 0:**
    - Try 1h or 4h bars to reduce churn
    - Add minimum hold period (e.g., 4 bars = 1 hour)
    - Use maker orders (0% fee) instead of taker (3.5 bps)
    - Align entries to 8h funding settlement boundaries

    **If gross < 0:**
    - Funding reversion may not work for this symbol/period
    - Try different z-score thresholds
    - Consider that the signal is already arbitraged away
    """))

    # --- End ---
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by `scripts/generate_report.py` at {now}*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NAT Experiment Report Generator")
    parser.add_argument("--timeframe", type=str, default="15min", help="Bar timeframe (default: 15min)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to analyze (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: reports/experiment_report.md)")
    args = parser.parse_args()

    report = generate_report(timeframe=args.timeframe, symbols=args.symbols)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else REPORT_DIR / "experiment_report.md"
    output.write_text(report)
    print(f"\nReport written to: {output}")
    print(f"  {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()

"""
Alpha Screener — systematic measurement of feature predictive power.

For each feature, computes:
  - Information Coefficient (rank IC) vs forward returns at multiple horizons
  - IC mean, std, information ratio, autocorrelation
  - Turnover (cost of trading the signal)
  - Breakeven cost (minimum fee structure for profitability)
  - Benjamini-Hochberg FDR-corrected p-values

Usage:
    python -m scripts.alpha.screener [--data-dir DATA_DIR] [--timeframe 15min]
    python -m scripts.alpha.screener --help

Output:
    reports/alpha_screen.json   — full per-feature metrics
    printed summary table       — top features ranked by IC
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "features"
REPORT_DIR = ROOT / "reports"

# Hyperliquid fee structure (loaded from config/costs.toml)
from utils.costs import taker_bps as _taker_bps, maker_bps as _maker_bps
TAKER_FEE_BPS = _taker_bps()
MAKER_FEE_BPS = _maker_bps()

# Minimum bars for meaningful IC computation
MIN_BARS_FOR_IC = 50

# Rolling window for IC computation (in bars)
IC_WINDOW_BARS = {
    "15min": 7 * 96,    # 7 days at 96 bars/day
    "1h": 7 * 24,       # 7 days at 24 bars/day
    "4h": 7 * 6,        # 7 days at 6 bars/day
}

# Forward return horizons mapped to bar counts
FORWARD_HORIZONS = {
    "15min": {"1h": 4, "4h": 16, "1d": 96},
    "1h": {"4h": 4, "1d": 24, "3d": 72},
    "4h": {"1d": 6, "3d": 18, "1w": 42},
}


@dataclass
class FeatureAlpha:
    """Alpha metrics for a single feature at a single horizon."""
    feature: str
    symbol: str
    timeframe: str
    horizon: str
    horizon_bars: int
    ic_mean: float
    ic_std: float
    ic_ir: float           # information ratio = ic_mean / ic_std
    ic_t_stat: float
    ic_p_value: float       # raw p-value from t-test
    ic_p_adjusted: float    # BH-FDR corrected p-value
    ic_autocorr: float      # lag-1 autocorrelation of IC series
    ic_hit_rate: float      # fraction of windows with IC same sign as mean
    n_windows: int
    turnover: float         # mean |delta(feature)| / std(feature)
    breakeven_bps: float    # minimum fee to remain profitable
    ann_ic_ir: float        # annualized IC IR
    significant: bool       # passes FDR-corrected threshold
    it_cost_viable: bool = False   # IT engine cost viability flag
    it_mi_bits: float = 0.0        # MI from IT engine (max across horizons)


@dataclass
class ScreenResult:
    """Complete screening result."""
    timestamp: str
    data_dir: str
    symbols: List[str]
    timeframe: str
    n_features_tested: int
    n_significant: int
    n_bars_per_symbol: Dict[str, int]
    total_tests: int
    fdr_threshold: float
    results: List[FeatureAlpha]
    it_features_loaded: int = 0       # IT engine features available
    it_cost_viable_count: int = 0     # IT engine cost-viable features


def compute_forward_returns(
    prices: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Compute forward returns: r(t) = price(t+h) / price(t) - 1."""
    n = len(prices)
    fwd = np.full(n, np.nan)
    if horizon >= n:
        return fwd
    fwd[:n - horizon] = prices[horizon:] / prices[:n - horizon] - 1
    return fwd


def compute_rolling_ic(
    feature: np.ndarray,
    forward_returns: np.ndarray,
    window: int,
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute rolling Spearman rank IC between feature and forward returns.

    Returns array of IC values, one per window. NaN where insufficient data.
    """
    n = len(feature)
    valid = ~(np.isnan(feature) | np.isnan(forward_returns))

    # Use non-overlapping windows for independence
    ic_values = []
    start = 0
    while start + window <= n:
        end = start + window
        mask = valid[start:end]
        n_valid = mask.sum()

        if n_valid >= min_obs:
            f_win = feature[start:end][mask]
            r_win = forward_returns[start:end][mask]

            # Check for constant arrays
            if np.std(f_win) < 1e-15 or np.std(r_win) < 1e-15:
                ic_values.append(0.0)
            else:
                rho, _ = stats.spearmanr(f_win, r_win)
                if np.isnan(rho):
                    ic_values.append(0.0)
                else:
                    ic_values.append(rho)
        else:
            ic_values.append(np.nan)

        start = end

    return np.array(ic_values)


def compute_expanding_ic(
    feature: np.ndarray,
    forward_returns: np.ndarray,
    min_obs: int = 50,
    step: int | None = None,
) -> np.ndarray:
    """
    Compute expanding-window Spearman IC (anchored at t=0, expanding forward).

    Avoids lookahead bias: IC at each expansion point uses only data
    available up to that point. Returns array of IC values, one per
    expansion step.

    Parameters
    ----------
    feature : array of feature values
    forward_returns : array of forward returns (same length)
    min_obs : minimum observations before first IC (default 50)
    step : expansion step size (default n // 20)
    """
    n = len(feature)
    if step is None:
        step = max(n // 20, 1)

    valid = ~(np.isnan(feature) | np.isnan(forward_returns))
    ic_values = []

    boundary = min_obs
    while boundary <= n:
        mask = valid[:boundary]
        n_valid = mask.sum()

        if n_valid >= min_obs:
            f_win = feature[:boundary][mask]
            r_win = forward_returns[:boundary][mask]

            if np.std(f_win) < 1e-15 or np.std(r_win) < 1e-15:
                ic_values.append(0.0)
            else:
                rho, _ = stats.spearmanr(f_win, r_win)
                ic_values.append(0.0 if np.isnan(rho) else rho)
        else:
            ic_values.append(np.nan)

        boundary += step

    return np.array(ic_values)


def compute_turnover(feature: np.ndarray) -> float:
    """
    Compute signal turnover: mean|f(t) - f(t-1)| / std(f).

    High turnover = expensive to trade. Low turnover = cheap.
    """
    valid = feature[~np.isnan(feature)]
    if len(valid) < 2:
        return np.nan
    sigma = np.std(valid)
    if sigma < 1e-15:
        return 0.0
    diffs = np.abs(np.diff(valid))
    return float(np.mean(diffs) / sigma)


def compute_breakeven_bps(
    ic_mean: float,
    vol_returns: float,
    turnover: float,
) -> float:
    """
    Compute breakeven trading cost in basis points.

    breakeven = |IC| * vol(returns) / turnover * 10000

    If IC is 0.03 with vol=0.01 and turnover=0.5:
    breakeven = 0.03 * 0.01 / 0.5 * 10000 = 6 bps → profitable at taker fees (3.5 bps)
    """
    if turnover < 1e-10 or np.isnan(turnover):
        return np.inf  # zero turnover = infinite breakeven (never trades)
    return abs(ic_mean) * vol_returns / turnover * 10_000


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Returns adjusted p-values. Features with adjusted p < alpha are significant.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    # Handle NaN p-values
    valid_mask = ~np.isnan(p_values)
    adjusted = np.full(n, np.nan)

    valid_p = p_values[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return adjusted

    # Sort p-values
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # BH adjustment: p_adj(i) = p(i) * n / rank(i)
    ranks = np.arange(1, n_valid + 1)
    adjusted_sorted = sorted_p * n_valid / ranks

    # Enforce monotonicity (step-down)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    # Map back to original order
    valid_adjusted = np.empty(n_valid)
    valid_adjusted[sorted_idx] = adjusted_sorted

    adjusted[valid_mask] = valid_adjusted
    return adjusted


def screen_features(
    data_dir: str | Path = DATA_DIR,
    timeframe: str = "15min",
    symbols: Optional[List[str]] = None,
    fdr_alpha: float = 0.05,
    min_ic: float = 0.015,
    min_breakeven_bps: float = 2.0,
    price_col: str = "raw_midprice",
    it_state_dir: Optional[str] = None,
    it_boost_factor: float = 1.0,
    it_prefilter: bool = False,
) -> ScreenResult:
    """
    Screen all features for predictive power against forward returns.

    This is the core function — it answers: "Which of the 191 features
    predict future returns after correcting for multiple testing?"

    IT engine integration (optional):
      - it_state_dir: load IT engine state for cost viability and MI scores
      - it_boost_factor: multiply IC ranking for IT cost-viable features
      - it_prefilter: restrict screening to IT-selected features only
    """
    # Lazy imports to avoid circular deps
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars
    from cluster_pipeline.config import META_COLUMNS

    data_dir = Path(data_dir)
    logger.info(f"Loading data from {data_dir}")
    t0 = time.time()

    df = load_parquet(str(data_dir))
    logger.info(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # Aggregate to bars
    logger.info(f"Aggregating to {timeframe} bars...")
    bars = aggregate_bars(df, timeframe=timeframe)
    logger.info(f"Produced {len(bars)} bars")

    # Determine symbols
    if "symbol" in bars.columns:
        available_symbols = sorted(bars["symbol"].unique())
    else:
        available_symbols = ["ALL"]
    if symbols:
        available_symbols = [s for s in available_symbols if s in symbols]
    logger.info(f"Symbols: {available_symbols}")

    # --- Load IT engine state (optional) ---
    it_states: Dict[str, "ITState"] = {}
    it_features_loaded = 0
    it_cost_viable_count = 0
    if it_state_dir:
        try:
            from it_engine.state import ITState
        except ImportError:
            from it_engine.state import ITState
        for sym in available_symbols:
            state = ITState.load(sym, data_dir=it_state_dir)
            if state.mi_matrix:
                it_states[sym] = state
                it_features_loaded += len(state.mi_matrix)
                it_cost_viable_count += sum(
                    1 for v in state.cost_viable.values() if v
                )
        if it_states:
            logger.info(
                "IT engine: loaded %d symbols, %d features, %d cost-viable",
                len(it_states), it_features_loaded, it_cost_viable_count,
            )

    # Identify feature columns (aggregated names like feat_mean, feat_std, feat_last)
    meta_cols = {"bar_start", "bar_end", "symbol", "tick_count"}
    feature_cols = [
        c for c in bars.columns
        if c not in meta_cols
        and bars[c].dtype in (np.float64, np.float32, np.int64, float, int)
    ]

    # IT prefilter: restrict to IT cost-viable features only
    if it_prefilter and it_states:
        it_viable_bases = set()
        for state in it_states.values():
            it_viable_bases.update(state.selected_features)
            it_viable_bases.update(
                f for f, v in state.cost_viable.items() if v
            )
        # Match bar column names (e.g. "ofi_imbalance_mean") to IT feature bases
        it_bar_cols = [
            c for c in feature_cols
            if any(c.startswith(base) for base in it_viable_bases)
        ]
        if it_bar_cols:
            logger.info(
                "IT prefilter: %d -> %d features",
                len(feature_cols), len(it_bar_cols),
            )
            feature_cols = it_bar_cols

    logger.info(f"Feature columns: {len(feature_cols)}")

    # Find the price column in aggregated bars
    price_candidates = [
        f"{price_col}_mean", f"{price_col}_close",
        f"{price_col}_last", price_col,
    ]
    bar_price_col = None
    for pc in price_candidates:
        if pc in bars.columns:
            bar_price_col = pc
            break
    if bar_price_col is None:
        raise ValueError(
            f"No price column found. Tried: {price_candidates}. "
            f"Available: {[c for c in bars.columns if 'price' in c.lower()]}"
        )
    logger.info(f"Using price column: {bar_price_col}")

    # Get forward horizons for this timeframe
    horizons = FORWARD_HORIZONS.get(timeframe, {"4h": 4, "1d": 24})

    all_results: List[FeatureAlpha] = []
    n_bars_per_symbol: Dict[str, int] = {}

    for sym in available_symbols:
        if "symbol" in bars.columns:
            sym_bars = bars[bars["symbol"] == sym].reset_index(drop=True)
        else:
            sym_bars = bars.reset_index(drop=True)

        n_bars = len(sym_bars)
        n_bars_per_symbol[sym] = n_bars
        logger.info(f"{sym}: {n_bars} bars")

        if n_bars < MIN_BARS_FOR_IC:
            logger.warning(f"{sym}: only {n_bars} bars, need {MIN_BARS_FOR_IC}. Skipping.")
            continue

        prices = sym_bars[bar_price_col].values.astype(np.float64)

        # Compute forward returns for each horizon
        fwd_returns: Dict[str, np.ndarray] = {}
        vol_returns: Dict[str, float] = {}
        for h_name, h_bars in horizons.items():
            fr = compute_forward_returns(prices, h_bars)
            fwd_returns[h_name] = fr
            valid_fr = fr[~np.isnan(fr)]
            vol_returns[h_name] = float(np.std(valid_fr)) if len(valid_fr) > 10 else 0.0

        # Screen each feature against each horizon
        for feat_col in feature_cols:
            # Skip price columns themselves (trivially correlated)
            if feat_col.startswith("raw_midprice") or feat_col.startswith("raw_microprice"):
                continue

            feat_vals = sym_bars[feat_col].values.astype(np.float64)

            # Skip if constant or all NaN
            valid_feat = feat_vals[~np.isnan(feat_vals)]
            if len(valid_feat) < MIN_BARS_FOR_IC or np.std(valid_feat) < 1e-15:
                continue

            turnover = compute_turnover(feat_vals)

            for h_name, h_bars in horizons.items():
                fr = fwd_returns[h_name]
                vol_r = vol_returns[h_name]

                # Expanding-window IC: anchor at t=0, expand forward
                # Avoids lookahead bias present in fixed rolling windows
                ic_series = compute_expanding_ic(feat_vals, fr, min_obs=50)
                valid_ics = ic_series[~np.isnan(ic_series)]

                if len(valid_ics) < 2:
                    continue

                # Use recent expansion points for stability metrics
                # (early points have few samples, late points are most reliable)
                recent_ics = valid_ics[-min(len(valid_ics), 10):]
                ic_mean = float(np.mean(recent_ics))
                ic_std = float(np.std(valid_ics))
                ic_ir = ic_mean / ic_std if ic_std > 1e-15 else 0.0
                n_windows = len(valid_ics)

                # t-statistic and p-value
                t_stat = ic_mean / (ic_std / np.sqrt(n_windows)) if ic_std > 1e-15 else 0.0
                p_value = float(2 * stats.t.sf(abs(t_stat), df=n_windows - 1))

                # IC autocorrelation (persistence)
                if len(valid_ics) >= 3:
                    ic_autocorr = float(np.corrcoef(valid_ics[:-1], valid_ics[1:])[0, 1])
                    if np.isnan(ic_autocorr):
                        ic_autocorr = 0.0
                else:
                    ic_autocorr = 0.0

                # Hit rate: fraction of windows with IC same sign as mean
                if abs(ic_mean) > 1e-15:
                    ic_hit_rate = float(np.mean(np.sign(valid_ics) == np.sign(ic_mean)))
                else:
                    ic_hit_rate = 0.5

                # Breakeven cost
                breakeven = compute_breakeven_bps(ic_mean, vol_r, turnover)

                # Annualized IC IR (for comparability across timeframes)
                # Scale by sqrt(observations per year / expansion steps)
                bars_per_year = {"15min": 96 * 365, "1h": 24 * 365, "4h": 6 * 365}
                bpy = bars_per_year.get(timeframe, 24 * 365)
                # Each expansion step covers n_bars/n_windows bars
                bars_per_step = n_bars / n_windows if n_windows > 0 else n_bars
                ann_ic_ir = ic_ir * np.sqrt(bpy / bars_per_step) if bars_per_step > 0 else 0.0

                all_results.append(FeatureAlpha(
                    feature=feat_col,
                    symbol=sym,
                    timeframe=timeframe,
                    horizon=h_name,
                    horizon_bars=h_bars,
                    ic_mean=round(ic_mean, 6),
                    ic_std=round(ic_std, 6),
                    ic_ir=round(ic_ir, 4),
                    ic_t_stat=round(t_stat, 4),
                    ic_p_value=p_value,
                    ic_p_adjusted=1.0,  # filled below
                    ic_autocorr=round(ic_autocorr, 4),
                    ic_hit_rate=round(ic_hit_rate, 4),
                    n_windows=n_windows,
                    turnover=round(turnover, 4),
                    breakeven_bps=round(breakeven, 2),
                    ann_ic_ir=round(ann_ic_ir, 4),
                    significant=False,  # filled below
                ))

    # Apply BH-FDR correction across ALL tests
    if all_results:
        p_values = np.array([r.ic_p_value for r in all_results])
        adjusted = benjamini_hochberg(p_values, alpha=fdr_alpha)
        for i, r in enumerate(all_results):
            r.ic_p_adjusted = round(float(adjusted[i]), 6) if not np.isnan(adjusted[i]) else 1.0
            r.significant = (
                r.ic_p_adjusted < fdr_alpha
                and abs(r.ic_mean) >= min_ic
                and r.breakeven_bps >= min_breakeven_bps
            )

    # --- IT engine annotation and boost ---
    if it_states:
        for r in all_results:
            it_state = it_states.get(r.symbol)
            if it_state is None:
                continue
            # Match bar feature name to IT base feature name
            # Bar names are like "ofi_imbalance_mean", IT names are "ofi_imbalance"
            base_feat = None
            for it_feat in it_state.mi_matrix:
                if r.feature.startswith(it_feat):
                    base_feat = it_feat
                    break
            if base_feat is None:
                continue

            # Annotate with IT metrics
            r.it_cost_viable = it_state.cost_viable.get(base_feat, False)
            mi_vals = it_state.mi_matrix.get(base_feat, {})
            r.it_mi_bits = max(mi_vals.values()) if mi_vals else 0.0

            # Boost ranking for IT cost-viable features
            if r.it_cost_viable and it_boost_factor > 1.0:
                r.ann_ic_ir *= it_boost_factor

        n_it_boosted = sum(1 for r in all_results if r.it_cost_viable)
        if n_it_boosted:
            logger.info("IT boost: %d features boosted by %.1fx", n_it_boosted, it_boost_factor)

    # Sort by annualized IC IR descending (incorporates IT boost)
    all_results.sort(key=lambda r: abs(r.ann_ic_ir), reverse=True)

    n_significant = sum(1 for r in all_results if r.significant)

    return ScreenResult(
        timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
        data_dir=str(data_dir),
        symbols=available_symbols,
        timeframe=timeframe,
        n_features_tested=len(feature_cols),
        n_significant=n_significant,
        n_bars_per_symbol=n_bars_per_symbol,
        total_tests=len(all_results),
        fdr_threshold=fdr_alpha,
        results=all_results,
        it_features_loaded=it_features_loaded,
        it_cost_viable_count=it_cost_viable_count,
    )


def print_screen_results(result: ScreenResult, top_n: int = 40) -> None:
    """Print a summary table of screening results."""
    print()
    print("=" * 90)
    print("  ALPHA SCREENING RESULTS")
    print("=" * 90)
    print(f"  Data:       {result.data_dir}")
    print(f"  Timeframe:  {result.timeframe}")
    print(f"  Symbols:    {', '.join(result.symbols)}")
    for sym, n in result.n_bars_per_symbol.items():
        print(f"    {sym}: {n} bars")
    print(f"  Features:   {result.n_features_tested}")
    print(f"  Tests run:  {result.total_tests}")
    print(f"  Significant: {result.n_significant} (FDR < {result.fdr_threshold})")
    print()

    # Gate check
    n_sig = result.n_significant
    if n_sig >= 5:
        gate = "PASS"
        gate_msg = f"{n_sig} features pass all criteria"
    elif n_sig >= 1:
        gate = "WEAK"
        gate_msg = f"Only {n_sig} features pass — marginal"
    else:
        gate = "FAIL"
        gate_msg = "No features pass significance + IC + cost thresholds"

    print(f"  Gate G1: {gate} — {gate_msg}")
    print()

    # Significant features table
    sig_results = [r for r in result.results if r.significant]
    if sig_results:
        print(f"  --- Significant Features (FDR < {result.fdr_threshold}, "
              f"|IC| > 0.015, breakeven > 2 bps) ---")
        print()
        _print_table(sig_results)

    # Top features by |IC| (regardless of significance)
    print()
    print(f"  --- Top {top_n} Features by |IC| ---")
    print()
    _print_table(result.results[:top_n])

    # Summary by feature category
    print()
    print("  --- IC Summary by Feature Category ---")
    print()
    categories: Dict[str, List[float]] = {}
    for r in result.results:
        # Extract category from feature name: e.g. "ent_tick_1m_mean" -> "ent"
        cat = r.feature.split("_")[0]
        categories.setdefault(cat, []).append(abs(r.ic_mean))
    cat_summary = sorted(
        [(cat, np.mean(ics), np.max(ics), len(ics))
         for cat, ics in categories.items()],
        key=lambda x: x[2], reverse=True,
    )
    print(f"  {'Category':<12} {'Mean|IC|':>10} {'Max|IC|':>10} {'Count':>6}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*6}")
    for cat, mean_ic, max_ic, count in cat_summary:
        print(f"  {cat:<12} {mean_ic:>10.4f} {max_ic:>10.4f} {count:>6}")

    print()
    print("=" * 90)
    print()


def _print_table(results: List[FeatureAlpha]) -> None:
    """Print a formatted table of feature alpha metrics."""
    header = (
        f"  {'Feature':<35} {'Sym':>4} {'Hor':>4} "
        f"{'IC':>8} {'IC_IR':>7} {'t-stat':>7} "
        f"{'p_adj':>8} {'Hit%':>5} {'Turn':>6} "
        f"{'BE_bps':>7} {'Sig':>4}"
    )
    print(header)
    print(f"  {'-'*35} {'-'*4} {'-'*4} "
          f"{'-'*8} {'-'*7} {'-'*7} "
          f"{'-'*8} {'-'*5} {'-'*6} "
          f"{'-'*7} {'-'*4}")

    for r in results:
        sig_mark = " *" if r.significant else "  "
        be_str = f"{r.breakeven_bps:>7.1f}" if r.breakeven_bps < 1e6 else "    inf"
        print(
            f"  {r.feature:<35} {r.symbol:>4} {r.horizon:>4} "
            f"{r.ic_mean:>+8.4f} {r.ic_ir:>7.3f} {r.ic_t_stat:>7.2f} "
            f"{r.ic_p_adjusted:>8.4f} {r.ic_hit_rate:>5.0%} {r.turnover:>6.2f} "
            f"{be_str} {sig_mark}"
        )


def save_results(result: ScreenResult, output_dir: Path = REPORT_DIR) -> Path:
    """Save screening results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
    data = {
        "timestamp": result.timestamp,
        "data_dir": result.data_dir,
        "symbols": result.symbols,
        "timeframe": result.timeframe,
        "n_features_tested": result.n_features_tested,
        "n_significant": result.n_significant,
        "n_bars_per_symbol": result.n_bars_per_symbol,
        "total_tests": result.total_tests,
        "fdr_threshold": result.fdr_threshold,
        "it_features_loaded": result.it_features_loaded,
        "it_cost_viable_count": result.it_cost_viable_count,
        "results": [asdict(r) for r in result.results],
    }

    outfile = output_dir / "alpha_screen.json"
    with open(outfile, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return outfile


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NAT Alpha Screener")
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--timeframe", type=str, default="15min",
        choices=["5min", "15min", "1h", "4h"],
        help="Bar aggregation timeframe",
    )
    parser.add_argument(
        "--symbols", type=str, nargs="*", default=None,
        help="Symbols to screen (default: all)",
    )
    parser.add_argument(
        "--fdr-alpha", type=float, default=0.05,
        help="FDR significance threshold",
    )
    parser.add_argument(
        "--top-n", type=int, default=40,
        help="Number of top features to display",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(REPORT_DIR),
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--it-state-dir", type=str, default=None,
        help="IT engine state directory (enables MI-based boosting)",
    )
    parser.add_argument(
        "--it-boost", type=float, default=1.5,
        help="Boost factor for IT cost-viable features",
    )
    parser.add_argument(
        "--it-prefilter", action="store_true",
        help="Only screen IT-selected features",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    result = screen_features(
        data_dir=args.data_dir,
        timeframe=args.timeframe,
        symbols=args.symbols,
        fdr_alpha=args.fdr_alpha,
        it_state_dir=args.it_state_dir,
        it_boost_factor=args.it_boost,
        it_prefilter=args.it_prefilter,
    )

    print_screen_results(result, top_n=args.top_n)

    outfile = save_results(result, output_dir=Path(args.output_dir))
    print(f"  Results saved to: {outfile}")
    print()


if __name__ == "__main__":
    main()

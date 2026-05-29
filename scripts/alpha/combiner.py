"""
Feature Combination Engine (Alpha Roadmap Step 2).

Takes top-N features from Step 1 (alpha screener), deduplicates by
correlation, and combines into a single signal z(t) in [-1, +1].

Quality Gate G2:
  - Combined IC > 0.8 * max(individual ICs)
  - Combined turnover < 2x avg individual turnover
  - Combined signal not > 0.9 correlated with any single feature
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
from scipy import stats

from alpha.screener import compute_rolling_ic, compute_turnover, FORWARD_HORIZONS
from cluster_pipeline.loader import load_parquet
from cluster_pipeline.preprocess import aggregate_bars

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """A selected feature from the screener."""
    name: str
    symbol: str
    horizon: str
    horizon_bars: int
    ic_mean: float
    turnover: float


@dataclass
class CombineResult:
    """Output of the combination step."""
    features_selected: List[str]
    features_after_dedup: List[str]
    n_bars: int
    method: str  # "equal" or "ic_weighted"
    combined_ic: float
    max_individual_ic: float
    combined_turnover: float
    avg_individual_turnover: float
    max_single_corr: float
    gate_ic_pass: bool
    gate_turnover_pass: bool
    gate_corr_pass: bool
    gate_pass: bool


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_screen_results(
    path: str | Path = "reports/alpha_screen.json",
) -> list[dict]:
    """Load screener output and return result list."""
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def select_top_features(
    results: list[dict],
    symbol: str = "BTC",
    horizon: Optional[str] = None,
    top_n: int = 20,
    require_significant: bool = False,
) -> List[FeatureSpec]:
    """Pick top-N features by |IC_mean| for a given symbol/horizon."""
    filtered = [r for r in results if r["symbol"] == symbol]
    if horizon:
        filtered = [r for r in filtered if r["horizon"] == horizon]
    if require_significant:
        filtered = [r for r in filtered if r["significant"]]

    # Sort by absolute IC descending
    filtered.sort(key=lambda r: abs(r["ic_mean"]), reverse=True)

    # Deduplicate feature names (keep best horizon per feature)
    seen = set()
    specs = []
    for r in filtered:
        if r["feature"] in seen:
            continue
        seen.add(r["feature"])
        specs.append(FeatureSpec(
            name=r["feature"],
            symbol=r["symbol"],
            horizon=r["horizon"],
            horizon_bars=r["horizon_bars"],
            ic_mean=r["ic_mean"],
            turnover=r["turnover"],
        ))
        if len(specs) >= top_n:
            break
    return specs


def _compute_corr_matrix(
    features_df: pl.DataFrame,
    names: List[str],
) -> np.ndarray:
    """Compute correlation matrix with NaN-safe median imputation."""
    mat = features_df.select(names).to_numpy()
    for i in range(mat.shape[1]):
        col = mat[:, i]
        mask = np.isfinite(col)
        if mask.sum() > 0:
            mat[~mask, i] = np.nanmedian(col)
        else:
            mat[:, i] = 0.0
    return np.corrcoef(mat.T)


def deduplicate_by_correlation(
    features_df: pl.DataFrame,
    specs: List[FeatureSpec],
    max_corr: float = 0.7,
    method: str = "cluster",
) -> List[FeatureSpec]:
    """Remove redundant features, keeping the highest-|IC| representative.

    Methods:
        "cluster"  — Hierarchical clustering on correlation distance.
                     Groups features into clusters where within-cluster
                     correlation exceeds max_corr, then picks the best-IC
                     feature per cluster. Handles transitive correlations
                     (A~B=0.75, B~C=0.75 form one cluster even if A~C=0.56).
        "pairwise" — Legacy greedy pairwise removal.
    """
    names = [s.name for s in specs if s.name in features_df.columns]
    if len(names) <= 1:
        return [s for s in specs if s.name in names]

    corr = _compute_corr_matrix(features_df, names)
    ic_lookup = {s.name: abs(s.ic_mean) for s in specs}

    if method == "cluster":
        survivors = _dedup_cluster(names, corr, ic_lookup, max_corr)
    else:
        survivors = _dedup_pairwise(names, corr, ic_lookup, max_corr)

    drop = set(names) - set(survivors)
    if drop:
        log.info("Dropped %d correlated features: %s", len(drop), drop)
    return [s for s in specs if s.name in survivors]


def _dedup_cluster(
    names: List[str],
    corr: np.ndarray,
    ic_lookup: dict,
    max_corr: float,
) -> List[str]:
    """Hierarchical clustering dedup — one representative per cluster."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(names)
    # Convert correlation to distance: d = 1 - |corr|
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negative (numerical noise can cause tiny negatives)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="complete")
    # Cut at distance = 1 - max_corr → features with |corr| > max_corr
    # are in the same cluster
    labels = fcluster(Z, t=1.0 - max_corr, criterion="distance")

    # Pick best-IC feature per cluster
    survivors = []
    for cluster_id in np.unique(labels):
        cluster_names = [names[i] for i in range(n) if labels[i] == cluster_id]
        best = max(cluster_names, key=lambda nm: ic_lookup.get(nm, 0))
        survivors.append(best)
    return survivors


def _dedup_pairwise(
    names: List[str],
    corr: np.ndarray,
    ic_lookup: dict,
    max_corr: float,
) -> List[str]:
    """Legacy greedy pairwise removal."""
    drop = set()
    for i in range(len(names)):
        if names[i] in drop:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in drop:
                continue
            if abs(corr[i, j]) > max_corr:
                if ic_lookup.get(names[i], 0) < ic_lookup.get(names[j], 0):
                    drop.add(names[i])
                    break
                else:
                    drop.add(names[j])
    return [n for n in names if n not in drop]


def standardize_features(
    df: pl.DataFrame,
    feature_names: List[str],
    window: int = 2880,  # ~30 days at 96 bars/day for 15min
) -> pl.DataFrame:
    """Rolling z-score standardization for each feature."""
    result = df.clone()
    for name in feature_names:
        if name not in result.columns:
            continue
        col = result[name].to_numpy().astype(np.float64)
        z = np.full_like(col, np.nan)
        for t in range(window, len(col)):
            win = col[t - window : t]
            valid = win[np.isfinite(win)]
            if len(valid) < window // 2:
                continue
            mu = np.mean(valid)
            sd = np.std(valid)
            if sd > 1e-12:
                z[t] = (col[t] - mu) / sd
        result = result.with_columns(pl.Series(name=name, values=z))
    return result


def combine_signals(
    df: pl.DataFrame,
    specs: List[FeatureSpec],
    method: str = "equal",
) -> np.ndarray:
    """Combine standardized features into a single signal in [-1, +1].

    method:
        "equal"      — simple mean of z-scores
        "ic_weighted" — weighted by IC magnitude
    """
    names = [s.name for s in specs if s.name in df.columns]
    if not names:
        return np.zeros(len(df))

    mat = df.select(names).to_numpy()

    if method == "ic_weighted":
        weights = np.array([abs(s.ic_mean) for s in specs if s.name in df.columns])
        weights = weights / weights.sum()
        raw = np.nansum(mat * weights[None, :], axis=1)
    else:
        raw = np.nanmean(mat, axis=1)

    # Rank-normalize to [-1, +1]
    signal = np.full_like(raw, np.nan)
    valid = np.isfinite(raw)
    if valid.sum() > 1:
        ranks = stats.rankdata(raw[valid])
        signal[valid] = 2.0 * ranks / len(ranks) - 1.0

    return signal


def evaluate_quality_gate(
    df: pl.DataFrame,
    signal: np.ndarray,
    specs: List[FeatureSpec],
    forward_returns: np.ndarray,
    ic_window: int = 672,  # 7 days at 96 bars/day
) -> CombineResult:
    """Run quality gate G2 on the combined signal."""
    names = [s.name for s in specs if s.name in df.columns]

    # Combined IC
    combined_ic_vals = compute_rolling_ic(signal, forward_returns, ic_window)
    combined_ic = float(np.nanmean(combined_ic_vals)) if len(combined_ic_vals) > 0 else 0.0

    # Individual ICs
    individual_ics = []
    for name in names:
        feat = df[name].to_numpy()
        ic_vals = compute_rolling_ic(feat, forward_returns, ic_window)
        individual_ics.append(abs(float(np.nanmean(ic_vals))) if len(ic_vals) > 0 else 0.0)
    max_individual_ic = max(individual_ics) if individual_ics else 0.0

    # Combined turnover
    combined_turnover = compute_turnover(signal)
    avg_individual_turnover = float(np.mean([s.turnover for s in specs])) if specs else 0.0

    # Max correlation with any single feature
    max_single_corr = 0.0
    valid = np.isfinite(signal)
    for name in names:
        feat = df[name].to_numpy()
        both_valid = valid & np.isfinite(feat)
        if both_valid.sum() > 10:
            c = abs(np.corrcoef(signal[both_valid], feat[both_valid])[0, 1])
            max_single_corr = max(max_single_corr, c)

    # Gates
    gate_ic = abs(combined_ic) > 0.8 * max_individual_ic
    gate_turnover = combined_turnover < 2.0 * avg_individual_turnover if avg_individual_turnover > 0 else True
    gate_corr = max_single_corr < 0.9

    return CombineResult(
        features_selected=[s.name for s in specs],
        features_after_dedup=names,
        n_bars=len(df),
        method="equal",
        combined_ic=combined_ic,
        max_individual_ic=max_individual_ic,
        combined_turnover=combined_turnover,
        avg_individual_turnover=avg_individual_turnover,
        max_single_corr=max_single_corr,
        gate_ic_pass=gate_ic,
        gate_turnover_pass=gate_turnover,
        gate_corr_pass=gate_corr,
        gate_pass=gate_ic and gate_turnover and gate_corr,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_combine(
    screen_path: str | Path = "reports/alpha_screen.json",
    data_dir: str | Path = "data/features",
    symbol: str = "BTC",
    horizon: Optional[str] = None,
    top_n: int = 20,
    max_corr: float = 0.7,
    dedup_method: str = "cluster",
    timeframe: str = "15min",
    method: str = "equal",
    output: Optional[str | Path] = None,
) -> tuple[np.ndarray, CombineResult]:
    """Full Step 2 pipeline: screen → select → dedup → combine → gate."""
    # 1. Load screen results
    results = load_screen_results(screen_path)
    specs = select_top_features(results, symbol=symbol, horizon=horizon, top_n=top_n)
    if not specs:
        print("  No features found in screen results.")
        return np.array([]), CombineResult(
            features_selected=[], features_after_dedup=[], n_bars=0,
            method=method, combined_ic=0, max_individual_ic=0,
            combined_turnover=0, avg_individual_turnover=0, max_single_corr=0,
            gate_ic_pass=False, gate_turnover_pass=False, gate_corr_pass=False,
            gate_pass=False,
        )

    # Use the best horizon from selected features if not specified
    if not horizon:
        horizon = specs[0].horizon
    horizon_bars = specs[0].horizon_bars

    print(f"  Selected {len(specs)} features (top by |IC|)")
    for s in specs[:5]:
        print(f"    {s.name:40s}  IC={s.ic_mean:+.4f}  horizon={s.horizon}")
    if len(specs) > 5:
        print(f"    ... and {len(specs) - 5} more")

    # 2. Load and aggregate data
    print(f"\n  Loading data from {data_dir}...")
    df = load_parquet(str(data_dir))
    df = df.filter(pl.col("symbol") == symbol)
    df = aggregate_bars(df, timeframe=timeframe)
    print(f"  {len(df)} bars after {timeframe} aggregation")

    # 3. Deduplicate by correlation (hierarchical clustering or pairwise)
    specs = deduplicate_by_correlation(df, specs, max_corr=max_corr, method=dedup_method)
    print(f"  {len(specs)} features after {dedup_method} dedup (threshold={max_corr})")

    # 4. Standardize
    feature_names = [s.name for s in specs if s.name in df.columns]
    print(f"  Standardizing {len(feature_names)} features (rolling z-score)...")
    df = standardize_features(df, feature_names)

    # 5. Combine
    signal = combine_signals(df, specs, method=method)
    valid_pct = np.isfinite(signal).mean() * 100
    print(f"  Combined signal: {valid_pct:.0f}% valid, method={method}")

    # 6. Forward returns for quality gate
    price_col = None
    for candidate in ["raw_midprice_mean", "raw_midprice_close", "raw_midprice_last"]:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        print("  WARNING: No price column found, skipping quality gate")
        fwd = np.zeros(len(df))
    else:
        prices = df[price_col].to_numpy()
        fwd = np.full_like(prices, np.nan)
        fwd[:-horizon_bars] = prices[horizon_bars:] / prices[:-horizon_bars] - 1.0

    # 7. Quality gate
    ic_window = {"15min": 672, "1h": 168, "4h": 42}.get(timeframe, 672)
    result = evaluate_quality_gate(df, signal, specs, fwd, ic_window=ic_window)
    result.method = method

    # Print gate results
    def _g(passed):
        return "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"

    print(f"\n  Gate G2 Results:")
    print(f"    IC: combined={result.combined_ic:.4f} vs 0.8*max={0.8*result.max_individual_ic:.4f}  [{_g(result.gate_ic_pass)}]")
    print(f"    Turnover: {result.combined_turnover:.4f} vs 2x avg={2*result.avg_individual_turnover:.4f}  [{_g(result.gate_turnover_pass)}]")
    print(f"    Max corr with single: {result.max_single_corr:.4f} vs 0.9  [{_g(result.gate_corr_pass)}]")
    print(f"    Overall: [{_g(result.gate_pass)}]")

    # 8. Save if requested
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n  Saved result to {out_path}")

    return signal, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feature Combination Engine (Alpha Roadmap Step 2)",
    )
    parser.add_argument("--screen", default="reports/alpha_screen.json", help="Screen results JSON")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--horizon", default=None, help="Filter to specific horizon")
    parser.add_argument("--top", type=int, default=20, help="Top N features")
    parser.add_argument("--max-corr", type=float, default=0.8, help="Correlation dedup threshold")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--method", default="equal", choices=["equal", "ic_weighted"])
    parser.add_argument("--output", default="reports/alpha_combine.json")
    args = parser.parse_args()

    run_combine(
        screen_path=args.screen,
        data_dir=args.data_dir,
        symbol=args.symbol,
        horizon=args.horizon,
        top_n=args.top,
        max_corr=args.max_corr,
        timeframe=args.timeframe,
        method=args.method,
        output=args.output,
    )


if __name__ == "__main__":
    main()

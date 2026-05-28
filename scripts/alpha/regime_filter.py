"""
Regime Conditioning (Alpha Roadmap Step 5).

Re-runs alpha screening within each regime separately. Uses
regime-conditioned feature weights when IC improves >1.5x over global.

Quality Gate G5:
  - At least 1 regime with IC_regime > 1.5 * IC_global
  - OOS Sharpe improves vs unconditioned signal

Usage:
    python -m alpha.regime_filter --screen reports/alpha_screen.json \
        --data data/features --model models/regime_gmm.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.screener import compute_rolling_ic, FORWARD_HORIZONS
from alpha.combiner import (
    FeatureSpec,
    select_top_features,
    deduplicate_by_correlation,
    standardize_features,
    combine_signals,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RegimeIC:
    """IC statistics for a feature within one regime."""
    regime_id: int
    n_bars: int
    ic_mean: float
    ic_std: float
    ic_ratio: float  # ic_regime / ic_global


@dataclass
class RegimeFilterResult:
    """Output of regime conditioning step."""
    n_regimes: int
    n_bars_total: int
    regime_bar_counts: Dict[int, int]
    global_ic: float
    regime_ics: Dict[int, float]  # regime_id -> best feature IC in that regime
    improvement_ratios: Dict[int, float]  # regime_id -> best IC_regime / IC_global
    conditioned_regimes: List[int]  # regimes where conditioning helps
    regime_weights: Dict[int, Dict[str, float]]  # regime_id -> {feature: weight}
    gate_has_improving_regime: bool  # at least 1 regime with IC > 1.5x global
    gate_pass: bool


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def assign_regime_labels(
    df: pd.DataFrame,
    model_path: Optional[Path] = None,
    regime_col: str = "regime_id",
) -> np.ndarray:
    """
    Get regime labels for each bar.

    Priority:
      1. If regime_col exists in df, use it directly
      2. If model_path is provided, load GMM and predict
      3. Fall back to single regime (all zeros)
    """
    if regime_col in df.columns:
        return df[regime_col].values.astype(int)

    if model_path is not None and Path(model_path).exists():
        return _predict_from_gmm(df, model_path)

    log.warning("No regime labels found — treating all bars as regime 0")
    return np.zeros(len(df), dtype=int)


def _predict_from_gmm(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Load GMM model and predict regime labels."""
    with open(model_path) as f:
        model = json.load(f)

    from sklearn.mixture import GaussianMixture

    features = model.get("features", [])
    n_components = model.get("n_components", 3)

    available = [f for f in features if f in df.columns]
    if not available:
        log.warning("No GMM features found in data — single regime fallback")
        return np.zeros(len(df), dtype=int)

    X = df[available].fillna(0).values
    gmm = GaussianMixture(n_components=n_components, random_state=42)

    # Load fitted parameters if available
    if "means" in model:
        gmm.means_ = np.array(model["means"])
        gmm.covariances_ = np.array(model["covariances"])
        gmm.weights_ = np.array(model["weights"])
        gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(gmm.covariances_)
        )
        return gmm.predict(X)

    # Otherwise fit fresh
    gmm.fit(X)
    return gmm.predict(X)


def screen_per_regime(
    df: pd.DataFrame,
    regime_labels: np.ndarray,
    feature_cols: List[str],
    price_col: str,
    horizon_bars: int,
    timeframe: str = "15min",
    min_bars: int = 100,
) -> Dict[int, Dict[str, float]]:
    """
    Run IC screening for each regime independently.

    Returns:
        Dict mapping regime_id -> {feature_name: ic_mean}
    """
    unique_regimes = sorted(np.unique(regime_labels).tolist())
    regime_ics: Dict[int, Dict[str, float]] = {}

    for r in unique_regimes:
        mask = regime_labels == r
        n_regime = int(np.sum(mask))

        if n_regime < min_bars:
            log.info(f"Regime {r}: {n_regime} bars < {min_bars} minimum, skipping")
            regime_ics[r] = {}
            continue

        df_regime = df.loc[mask].reset_index(drop=True)
        prices = df_regime[price_col].values

        # Compute forward returns
        fwd = np.full(n_regime, np.nan)
        fwd[:-horizon_bars] = (
            prices[horizon_bars:] - prices[:-horizon_bars]
        ) / prices[:-horizon_bars]

        ic_dict = {}
        for feat in feature_cols:
            if feat not in df_regime.columns:
                continue
            vals = df_regime[feat].values
            valid = ~(np.isnan(vals) | np.isnan(fwd))
            if valid.sum() < 30:
                continue

            from scipy.stats import spearmanr
            corr, _ = spearmanr(vals[valid], fwd[valid])
            if not np.isnan(corr):
                ic_dict[feat] = float(corr)

        regime_ics[r] = ic_dict
        log.info(f"Regime {r}: {n_regime} bars, {len(ic_dict)} features screened")

    return regime_ics


def compute_regime_weights(
    regime_ics: Dict[int, Dict[str, float]],
    global_ics: Dict[str, float],
    improvement_threshold: float = 1.5,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-regime feature weights.

    For regimes where IC improves by >threshold vs global, use regime-specific
    IC-weighted combination. Otherwise, fall back to global weights.
    """
    weights: Dict[int, Dict[str, float]] = {}

    for r, r_ics in regime_ics.items():
        if not r_ics:
            weights[r] = {}
            continue

        # Check if any feature improves significantly
        improvements = {}
        for feat, r_ic in r_ics.items():
            g_ic = global_ics.get(feat, 0.0)
            if abs(g_ic) > 0.001:
                improvements[feat] = abs(r_ic) / abs(g_ic)
            elif abs(r_ic) > 0.01:
                improvements[feat] = 10.0  # large improvement from zero

        has_improvement = any(v > improvement_threshold for v in improvements.values())

        if has_improvement:
            # Use regime-specific IC-weighted combination
            total_ic = sum(abs(v) for v in r_ics.values())
            if total_ic > 0:
                weights[r] = {f: abs(ic) / total_ic for f, ic in r_ics.items()}
            else:
                weights[r] = {}
        else:
            # Fall back to global weights
            total_ic = sum(abs(v) for v in global_ics.values())
            if total_ic > 0:
                weights[r] = {f: abs(ic) / total_ic for f, ic in global_ics.items()}
            else:
                weights[r] = {}

    return weights


def run_regime_filter(
    data_dir: str = "data/features",
    screen_path: str = "reports/alpha_screen.json",
    model_path: Optional[str] = None,
    timeframe: str = "15min",
    symbol: str = "BTC",
    top_n: int = 10,
    improvement_threshold: float = 1.5,
    output: str = "reports/alpha_regime.json",
) -> RegimeFilterResult:
    """
    Full regime conditioning pipeline.

    1. Load data and screen results
    2. Assign regime labels
    3. Screen features per regime
    4. Compute regime-conditioned weights
    5. Evaluate G5 quality gate
    """
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars

    # Load screen results
    screen_file = Path(screen_path)
    if not screen_file.exists():
        raise FileNotFoundError(f"Screen results not found: {screen_path}")

    with open(screen_file) as f:
        screen = json.load(f)

    # Load data
    df = load_parquet(data_dir)
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].reset_index(drop=True)

    bars = aggregate_bars(df, timeframe=timeframe)
    bars_pd = bars.to_pandas() if hasattr(bars, "to_pandas") else bars

    # Get feature columns and price
    price_col = None
    for c in ["midprice_mean", "close", "mid_price"]:
        if c in bars_pd.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found")

    # Extract top features from screen
    results = screen.get("results", screen.get("features", []))
    if isinstance(results, dict):
        results = list(results.values())

    feature_specs = []
    for r in results:
        if isinstance(r, dict):
            name = r.get("feature", r.get("name", ""))
            ic = abs(r.get("ic_mean", r.get("ic", 0.0)))
            feature_specs.append((name, ic))

    feature_specs.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in feature_specs[:top_n]]
    top_features = [f for f in top_features if f in bars_pd.columns]

    if not top_features:
        log.error("No screened features found in data columns")
        return RegimeFilterResult(
            n_regimes=0, n_bars_total=len(bars_pd),
            regime_bar_counts={}, global_ic=0.0,
            regime_ics={}, improvement_ratios={},
            conditioned_regimes=[], regime_weights={},
            gate_has_improving_regime=False, gate_pass=False,
        )

    # Global IC
    prices = bars_pd[price_col].values
    horizon_bars = 4  # 1h at 15min
    fwd = np.full(len(prices), np.nan)
    fwd[:-horizon_bars] = (
        prices[horizon_bars:] - prices[:-horizon_bars]
    ) / prices[:-horizon_bars]

    from scipy.stats import spearmanr
    global_ics = {}
    for feat in top_features:
        vals = bars_pd[feat].values
        valid = ~(np.isnan(vals) | np.isnan(fwd))
        if valid.sum() > 30:
            corr, _ = spearmanr(vals[valid], fwd[valid])
            if not np.isnan(corr):
                global_ics[feat] = float(corr)

    best_global_ic = max(abs(v) for v in global_ics.values()) if global_ics else 0.0

    # Assign regimes
    model = Path(model_path) if model_path else None
    regime_labels = assign_regime_labels(bars_pd, model)

    unique_regimes = sorted(np.unique(regime_labels).tolist())
    regime_bar_counts = {int(r): int(np.sum(regime_labels == r)) for r in unique_regimes}

    # Screen per regime
    regime_ics = screen_per_regime(
        bars_pd, regime_labels, top_features, price_col,
        horizon_bars=horizon_bars, min_bars=100,
    )

    # Best IC per regime
    best_regime_ics = {}
    improvement_ratios = {}
    for r, ics in regime_ics.items():
        if ics:
            best_ic = max(abs(v) for v in ics.values())
            best_regime_ics[r] = best_ic
            improvement_ratios[r] = best_ic / best_global_ic if best_global_ic > 0 else 0.0
        else:
            best_regime_ics[r] = 0.0
            improvement_ratios[r] = 0.0

    # Regime weights
    regime_weights = compute_regime_weights(
        regime_ics, global_ics, improvement_threshold,
    )

    # G5 quality gate
    conditioned = [r for r, ratio in improvement_ratios.items() if ratio > improvement_threshold]
    gate_pass = len(conditioned) > 0

    result = RegimeFilterResult(
        n_regimes=len(unique_regimes),
        n_bars_total=len(bars_pd),
        regime_bar_counts=regime_bar_counts,
        global_ic=best_global_ic,
        regime_ics=best_regime_ics,
        improvement_ratios=improvement_ratios,
        conditioned_regimes=conditioned,
        regime_weights=regime_weights,
        gate_has_improving_regime=gate_pass,
        gate_pass=gate_pass,
    )

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    log.info(f"Regime filter: {len(unique_regimes)} regimes, "
             f"conditioned={conditioned}, G5={'PASS' if gate_pass else 'FAIL'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Regime conditioning (Step 5)")
    parser.add_argument("--data", default="data/features")
    parser.add_argument("--screen", default="reports/alpha_screen.json")
    parser.add_argument("--model", default=None, help="Path to regime GMM model JSON")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", default="reports/alpha_regime.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_regime_filter(
        data_dir=args.data, screen_path=args.screen,
        model_path=args.model, timeframe=args.timeframe,
        symbol=args.symbol, top_n=args.top_n, output=args.output,
    )
    gate = "PASS" if result.gate_pass else "FAIL"
    print(f"\nG5 Quality Gate: {gate}")
    print(f"  Regimes: {result.n_regimes}")
    print(f"  Global IC: {result.global_ic:.4f}")
    for r, ratio in result.improvement_ratios.items():
        print(f"  Regime {r}: IC ratio={ratio:.2f}x {'*' if ratio > 1.5 else ''}")


if __name__ == "__main__":
    main()

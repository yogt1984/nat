#!/usr/bin/env python3
"""
Cluster Quality Analysis Script

Analyzes collected feature data to assess GMM regime classification quality.
Outputs comprehensive report with HMM readiness assessment.

Usage:
    python scripts/analyze_clusters.py --data-dir ./data/features --symbol BTC
    python scripts/analyze_clusters.py --data-dir ./data/features --symbol BTC --model models/regime_gmm.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cluster_quality import (
    compute_all_metrics,
    compute_bootstrap_stability,
    compute_temporal_stability,
    compute_all_external_validation,
    compute_quality_score,
    StabilityMetrics,
)


def load_parquet_data(
    data_dir: Path,
    symbol: str,
    hours_back: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load Parquet feature data for analysis.

    Args:
        data_dir: Directory containing Parquet files
        symbol: Asset symbol (BTC, ETH, SOL)
        hours_back: Optional - only load last N hours

    Returns:
        Polars DataFrame with features
    """
    print(f"Loading data from {data_dir} for {symbol}...")

    # Find Parquet files for symbol
    pattern = f"{symbol}_*.parquet"
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise ValueError(f"No Parquet files found for {symbol} in {data_dir}")

    print(f"Found {len(files)} Parquet files")

    # Load and concatenate
    dfs = []
    for file in files:
        df = pl.read_parquet(file)
        dfs.append(df)

    df = pl.concat(dfs)

    # Filter by time if requested
    if hours_back is not None:
        cutoff = datetime.now() - timedelta(hours=hours_back)
        df = df.filter(pl.col("timestamp") >= cutoff)

    print(f"Loaded {len(df)} samples")
    return df


def extract_5d_features(df: pl.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Extract 5D feature space for GMM clustering.

    Features:
    1. Kyle's Lambda (liquidity)
    2. VPIN (informed trading)
    3. Absorption z-score (price response)
    4. Hurst exponent (persistence)
    5. Whale net flow (smart money)

    Args:
        df: Polars DataFrame with features

    Returns:
        (features array, feature names)
    """
    feature_names = [
        "kyle_lambda_100",      # Kyle's Lambda (closest available)
        "vpin_50",              # VPIN
        "absorption_zscore",    # From regime features
        "hurst_300",            # Hurst exponent
        "whale_net_flow_1h",    # Whale flow (or 0 if not available)
    ]

    # Extract features
    features = []
    for name in feature_names:
        if name in df.columns:
            features.append(df[name].to_numpy())
        else:
            print(f"Warning: {name} not found, using zeros")
            features.append(np.zeros(len(df)))

    X = np.column_stack(features)

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]

    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
    return X, feature_names


def cluster_features(
    X: np.ndarray,
    model_path: Optional[Path] = None,
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Cluster features using GMM.

    Args:
        X: Feature matrix (n_samples, n_features)
        model_path: Optional path to trained GMM model
        n_clusters: Number of clusters if training from scratch

    Returns:
        Cluster labels (n_samples,)
    """
    from sklearn.mixture import GaussianMixture

    if model_path and model_path.exists():
        print(f"Loading GMM model from {model_path}")
        # Note: This would require loading from JSON format
        # For now, train from scratch
        print("Warning: GMM loading not implemented, training from scratch")

    print(f"Training GMM with {n_clusters} clusters...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=42,
        n_init=10,
    )

    labels = gmm.fit_predict(X)
    print(f"Clustering complete: {len(np.unique(labels))} clusters found")

    return labels


def compute_forward_returns(
    df: pl.DataFrame,
    horizons: list = [60, 300, 3600],
) -> dict:
    """
    Compute forward returns at multiple horizons.

    Args:
        df: DataFrame with price data
        horizons: List of forward horizons in seconds

    Returns:
        Dict of horizon -> returns array
    """
    print("Computing forward returns...")

    # Get midprice
    if "midprice" in df.columns:
        prices = df["midprice"].to_numpy()
    else:
        print("Warning: midprice not found, using placeholder")
        return {}

    returns = {}
    for horizon in horizons:
        # Simple forward return calculation
        # Note: This assumes ~100ms sampling, adjust as needed
        samples_forward = horizon // 100  # Assuming 100ms per sample

        if samples_forward < len(prices):
            forward_prices = np.roll(prices, -samples_forward)
            ret = (forward_prices - prices) / prices
            ret[-samples_forward:] = 0  # Invalid at end
            returns[horizon] = ret

    return returns


def compute_forward_volatility(df: pl.DataFrame) -> np.ndarray:
    """
    Compute forward-looking volatility measure.

    Args:
        df: DataFrame with price data

    Returns:
        Volatility array
    """
    if "midprice" in df.columns:
        prices = df["midprice"].to_numpy()
        # Rolling std of returns (30-second window)
        window = 300
        returns = np.diff(prices) / prices[:-1]

        vol = np.zeros(len(prices))
        for i in range(window, len(returns)):
            vol[i] = np.std(returns[i-window:i])

        return vol
    else:
        return np.zeros(len(df))


def run_analysis(
    data_dir: Path,
    symbol: str,
    model_path: Optional[Path] = None,
    hours_back: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Run complete cluster quality analysis.

    Args:
        data_dir: Directory with Parquet files
        symbol: Asset symbol
        model_path: Optional GMM model path
        hours_back: Optional time window
        output_path: Optional output file path
    """
    print("=" * 70)
    print("CLUSTER QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Data directory: {data_dir}")
    print(f"Time window: {hours_back if hours_back else 'All'} hours")
    print()

    # Load data
    df = load_parquet_data(data_dir, symbol, hours_back)

    # Extract features
    X, feature_names = extract_5d_features(df)

    if len(X) < 100:
        print("Error: Not enough data for analysis (need at least 100 samples)")
        return

    # Cluster
    labels = cluster_features(X, model_path)

    # Compute internal metrics
    print("\n" + "=" * 70)
    print("COMPUTING INTERNAL QUALITY METRICS")
    print("=" * 70)
    metrics = compute_all_metrics(X, labels, compute_gap=True)
    print(metrics.summary())

    # Compute stability metrics
    print("\n" + "=" * 70)
    print("COMPUTING STABILITY METRICS")
    print("=" * 70)

    from sklearn.cluster import KMeans
    def cluster_func(X):
        return KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X)

    bootstrap = compute_bootstrap_stability(
        X, cluster_func, n_bootstraps=50, random_state=42
    )
    print(f"Bootstrap Stability: mean_ari={bootstrap.mean_ari:.3f}, pct_stable={bootstrap.pct_stable:.1%}")

    timestamps = np.arange(len(X))
    temporal = compute_temporal_stability(X, timestamps, cluster_func)
    print(f"Temporal Stability: temporal_ari={temporal.temporal_ari:.3f}, drift={temporal.proportion_drift:.3f}")

    stability = StabilityMetrics(bootstrap=bootstrap, temporal=temporal)

    # Compute external validation
    print("\n" + "=" * 70)
    print("COMPUTING EXTERNAL VALIDATION")
    print("=" * 70)

    forward_returns = compute_forward_returns(df)
    forward_vol = compute_forward_volatility(df)

    # Match lengths
    min_len = min(len(labels), len(forward_vol))
    labels_subset = labels[:min_len]
    forward_vol_subset = forward_vol[:min_len]

    # Adjust returns dict
    forward_returns_subset = {}
    for horizon, returns in forward_returns.items():
        forward_returns_subset[horizon] = returns[:min_len]

    validation = compute_all_external_validation(
        labels_subset,
        forward_returns_subset,
        forward_vol_subset,
        timestamps[:min_len],
    )

    print(f"Return Differentiation: {len(validation.return_differentiation)} horizons tested")
    for horizon, result in validation.return_differentiation.items():
        print(f"  {horizon}s: significant={result.significant}, eta²={result.eta_squared:.3f}")

    if validation.volatility_differentiation:
        print(f"Volatility Differentiation: significant={validation.volatility_differentiation.significant}")

    if validation.transitions:
        print(f"Transition Matrix: self_transition={validation.transitions.self_transition_rate:.3f}")

    # Compute composite score
    print("\n" + "=" * 70)
    print("COMPOSITE QUALITY ASSESSMENT")
    print("=" * 70)

    score = compute_quality_score(metrics, stability, validation)
    print(score.summary())

    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"Cluster Quality Analysis Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Samples: {len(X)}\n")
            f.write(f"\n{metrics.summary()}\n")
            f.write(f"\n{score.summary()}\n")
        print(f"\nReport saved to {output_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cluster quality for GMM regime classification"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/features"),
        help="Directory containing Parquet files (default: ./data/features)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Asset symbol to analyze (default: BTC)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained GMM model (optional)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        help="Only analyze last N hours of data (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output report file path (optional)",
    )

    args = parser.parse_args()

    try:
        run_analysis(
            data_dir=args.data_dir,
            symbol=args.symbol,
            model_path=args.model,
            hours_back=args.hours,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

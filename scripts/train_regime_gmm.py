#!/usr/bin/env python3
"""
Train GMM Regime Classifier on Collected Feature Data

This script trains a Gaussian Mixture Model on the 5D feature space
to identify market regimes (Accumulation, Markup, Distribution, Markdown, Ranging).

Output:
- models/regime_gmm.json: Model parameters for Rust inference
- models/cluster_stats.json: Statistics for each cluster (for label assignment)

Usage:
    python scripts/train_regime_gmm.py --data-dir data/features --output-dir models
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import polars as pl
except ImportError:
    print("Error: polars not installed. Run: pip install polars")
    sys.exit(1)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


# Feature columns for regime space.
# Must match the Rust inference side (ing/src/state/mod.rs emit_features).
# Whale flow is excluded: all-NaN in current data (feature not yet computed).
# This makes the GMM a 4D model until whale flow is wired.
FEATURE_COLUMNS = [
    "illiq_kyle_100",              # Kyle's Lambda (liquidity)
    "toxic_vpin_50",               # VPIN (informed trading)
    "regime_absorption_zscore",    # Absorption ratio z-score
    "trend_hurst_300",             # Hurst exponent (persistence)
]

# Rust inference dimension order — must match exactly.
# See rust/ing/src/state/mod.rs ~line 205: gmm_input array.
RUST_INFERENCE_ORDER = [
    "illiq_kyle_100",              # features.illiquidity.kyle_lambda_100
    "toxic_vpin_50",               # features.toxicity.vpin_50
    "regime_absorption_zscore",    # regime_features.absorption_zscore
    "trend_hurst_300",             # features.trend.hurst_300
]


def load_features(data_dir: Path, sample_frac: float = 1.0) -> Tuple[np.ndarray, pl.DataFrame]:
    """
    Load all Parquet files and extract feature space for GMM training.

    Args:
        data_dir: Directory containing Parquet files (searched recursively)
        sample_frac: Fraction of data to sample (for faster iteration)

    Returns:
        (X, df) where X is the feature matrix and df is the full dataframe
    """
    files = list(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    print(f"Found {len(files)} Parquet files")

    # Load and concatenate
    dfs = []
    for f in sorted(files):
        try:
            df = pl.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        raise ValueError("No data loaded from Parquet files")

    df = pl.concat(dfs, how="diagonal_relaxed")
    print(f"Loaded {len(df)} total rows")

    # Sample if requested
    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        print(f"Sampled to {len(df)} rows")

    # Verify all required columns exist in schema
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {sorted(df.columns)}"
        )

    # Check for all-NaN columns (e.g. whale features not yet computed)
    for col in FEATURE_COLUMNS:
        nan_rate = df[col].is_null().mean() + df[col].is_nan().mean()
        if nan_rate > 0.95:
            raise ValueError(
                f"Column '{col}' is {nan_rate:.0%} NaN — cannot train on missing data. "
                f"Ensure the ingestor is computing this feature."
            )
        elif nan_rate > 0.1:
            print(f"Warning: '{col}' has {nan_rate:.1%} NaN (will drop those rows)")

    # Select feature columns and drop rows with any null/NaN
    df_features = df.select(FEATURE_COLUMNS)
    # Replace NaN with null for uniform drop_nulls handling
    df_features = df_features.with_columns(
        [pl.col(c).fill_nan(None) for c in FEATURE_COLUMNS]
    ).drop_nulls()

    print(f"After dropping nulls: {len(df_features)} rows ({len(df) - len(df_features)} dropped)")

    if len(df_features) < 100:
        raise ValueError(
            f"Only {len(df_features)} valid rows after dropping NaN — need at least 100"
        )

    X = df_features.to_numpy()
    return X, df


def select_n_components(X: np.ndarray, max_components: int = 10) -> int:
    """
    Select optimal number of GMM components using BIC.

    Args:
        X: Scaled feature matrix
        max_components: Maximum components to try

    Returns:
        Optimal number of components
    """
    print("\nSelecting number of components via BIC...")

    bics = []
    for n in range(2, min(max_components + 1, len(X) // 100)):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            n_init=3,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(X)
        bics.append((n, gmm.bic(X)))
        print(f"  n={n}: BIC={gmm.bic(X):.2f}")

    # Find minimum BIC
    best_n, best_bic = min(bics, key=lambda x: x[1])
    print(f"\nOptimal components: {best_n} (BIC={best_bic:.2f})")

    return best_n


def train_gmm(
    X: np.ndarray,
    n_components: int = 5,
    auto_select: bool = False,
) -> Tuple[GaussianMixture, StandardScaler]:
    """
    Train GMM on feature data.

    Args:
        X: Raw feature matrix
        n_components: Number of mixture components
        auto_select: Whether to auto-select n_components via BIC

    Returns:
        (gmm, scaler) trained model and scaler
    """
    print(f"\nTraining GMM with {n_components} components...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature statistics after scaling:")
    for i, col in enumerate(FEATURE_COLUMNS):
        print(f"  {col}: mean={X_scaled[:, i].mean():.4f}, std={X_scaled[:, i].std():.4f}")

    # Auto-select components if requested
    if auto_select:
        n_components = select_n_components(X_scaled, max_components=10)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        n_init=10,
        random_state=42,
        max_iter=300,
        tol=1e-4,
    )
    gmm.fit(X_scaled)

    # Compute silhouette score
    labels = gmm.predict(X_scaled)
    if len(set(labels)) > 1:
        sil_score = silhouette_score(X_scaled, labels)
        print(f"\nSilhouette score: {sil_score:.4f}")

    print(f"Converged: {gmm.converged_}")
    print(f"Log-likelihood: {gmm.score(X_scaled):.4f}")

    return gmm, scaler


def analyze_clusters(
    gmm: GaussianMixture,
    scaler: StandardScaler,
    X: np.ndarray,
) -> Dict[int, Dict]:
    """
    Analyze cluster characteristics for semantic labeling.

    Args:
        gmm: Trained GMM
        scaler: Feature scaler
        X: Original (unscaled) feature matrix

    Returns:
        Dictionary of cluster statistics
    """
    print("\nAnalyzing clusters...")

    X_scaled = scaler.transform(X)
    labels = gmm.predict(X_scaled)

    stats = {}
    for c in range(gmm.n_components):
        mask = labels == c
        cluster_X = X[mask]

        if len(cluster_X) == 0:
            continue

        stats[c] = {
            "count": int(mask.sum()),
            "proportion": float(mask.mean()),
            "feature_means": {
                col: float(cluster_X[:, i].mean())
                for i, col in enumerate(FEATURE_COLUMNS)
            },
            "feature_stds": {
                col: float(cluster_X[:, i].std())
                for i, col in enumerate(FEATURE_COLUMNS)
            },
            "scaled_center": gmm.means_[c].tolist(),
        }

        print(f"\nCluster {c} ({stats[c]['count']} samples, {stats[c]['proportion']*100:.1f}%):")
        for col, mean in stats[c]["feature_means"].items():
            std = stats[c]["feature_stds"][col]
            print(f"  {col}: {mean:.4f} (±{std:.4f})")

    return stats


def suggest_labels(cluster_stats: Dict[int, Dict]) -> Dict[int, str]:
    """
    Suggest semantic labels based on 4D cluster characteristics.

    Based on expected regime signatures (without whale flow):
    - Accumulation: low λ, low VPIN, high absorption
    - Markup: high λ, low VPIN, high Hurst
    - Distribution: low λ, high VPIN, high absorption
    - Markdown: high λ, high VPIN, high Hurst
    - Ranging: moderate all, low Hurst
    """
    suggestions = {}

    for c, stats in cluster_stats.items():
        scaled = stats["scaled_center"]

        scores = {
            "accumulation": 0.0,
            "markup": 0.0,
            "distribution": 0.0,
            "markdown": 0.0,
            "ranging": 0.0,
        }

        # 4D: [kyle_lambda, vpin, absorption_zscore, hurst]
        kyle = scaled[0]
        vpin = scaled[1]
        absorption = scaled[2]
        hurst = scaled[3]

        # Accumulation: low λ, low VPIN, high absorption
        scores["accumulation"] = -kyle - vpin + absorption

        # Markup: high λ, low VPIN, high Hurst
        scores["markup"] = kyle - vpin + hurst

        # Distribution: low λ, high VPIN, high absorption
        scores["distribution"] = -kyle + vpin + absorption

        # Markdown: high λ, high VPIN, high Hurst
        scores["markdown"] = kyle + vpin + hurst

        # Ranging: low Hurst, neutral everything else
        scores["ranging"] = -abs(kyle) - abs(vpin) - hurst

        best_label = max(scores, key=scores.get)
        suggestions[c] = best_label

        print(f"\nCluster {c} suggested label: {best_label}")
        print(f"  Scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))}")

    return suggestions


def export_model(
    gmm: GaussianMixture,
    scaler: StandardScaler,
    output_path: Path,
) -> None:
    """
    Export model parameters as JSON for Rust consumption.
    """
    model_params = {
        "n_components": gmm.n_components,
        "means": gmm.means_.tolist(),
        "covariances": gmm.covariances_.tolist(),
        "weights": gmm.weights_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "feature_names": FEATURE_COLUMNS,
    }

    with open(output_path, 'w') as f:
        json.dump(model_params, f, indent=2)

    print(f"\nExported model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GMM regime classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing Parquet feature files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of GMM components (regimes)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Auto-select number of components via BIC",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to sample (for faster iteration)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GMM Regime Classifier Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading features from {args.data_dir}...")
    try:
        X, df = load_features(args.data_dir, sample_frac=args.sample_frac)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate feature data, run the ing binary to collect data:")
        print("  cargo run --release -- --symbol BTC")
        return 1
    except ValueError as e:
        print(f"\nError: {e}")
        return 1

    # Train GMM
    gmm, scaler = train_gmm(X, args.n_components, args.auto_select)

    # Analyze clusters
    cluster_stats = analyze_clusters(gmm, scaler, X)

    # Suggest labels
    label_suggestions = suggest_labels(cluster_stats)

    # Export model
    export_model(gmm, scaler, args.output_dir / "regime_gmm.json")

    # Export cluster statistics
    output_stats = {
        "clusters": cluster_stats,
        "suggested_labels": label_suggestions,
        "feature_names": FEATURE_COLUMNS,
    }
    with open(args.output_dir / "cluster_stats.json", 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"Exported cluster stats to {args.output_dir / 'cluster_stats.json'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review cluster_stats.json and assign semantic labels")
    print(f"2. Copy regime_gmm.json to your deployment location")
    print(f"3. Load the model in Rust using RegimeClassifier::load()")

    return 0


if __name__ == "__main__":
    sys.exit(main())

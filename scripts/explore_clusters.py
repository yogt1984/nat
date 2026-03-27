#!/usr/bin/env python3
"""
Exploratory Cluster Visualization for NAT Features

This script explores the natural structure in collected feature data
WITHOUT imposing predefined assumptions about regimes.

Approach:
1. Load feature data from Parquet files
2. Explore different feature subsets
3. Reduce dimensionality (PCA, UMAP, t-SNE)
4. Visualize and look for natural clusters
5. If clusters exist, characterize them

Usage:
    python scripts/explore_clusters.py --data-dir rust/data/features
    python scripts/explore_clusters.py --data-dir rust/data/features --subset entropy
    python scripts/explore_clusters.py --data-dir rust/data/features --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

try:
    import polars as pl
except ImportError:
    print("Error: polars not installed. Run: pip install polars")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# Optional imports with graceful fallback
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP visualization disabled.")
    print("  Install with: pip install umap-learn")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not installed. HDBSCAN clustering disabled.")
    print("  Install with: pip install hdbscan")


# =============================================================================
# Feature Subsets - Different "views" of the market
# =============================================================================

FEATURE_SUBSETS = {
    "entropy": {
        "description": "Information-theoretic features measuring market uncertainty",
        "patterns": ["ent_", "entropy"],
    },
    "orderbook": {
        "description": "Order book structure and imbalance features",
        "patterns": ["imb_", "spread", "depth", "book"],
    },
    "flow": {
        "description": "Trade flow and volume features",
        "patterns": ["flow_", "volume", "vwap", "aggressor"],
    },
    "illiquidity": {
        "description": "Market impact and liquidity features",
        "patterns": ["ill_", "kyle", "amihud", "lambda"],
    },
    "toxicity": {
        "description": "Adverse selection and informed trading",
        "patterns": ["tox_", "vpin", "adverse"],
    },
    "trend": {
        "description": "Trend strength and momentum features",
        "patterns": ["trend_", "hurst", "momentum", "rsi"],
    },
    "volatility": {
        "description": "Volatility and variance features",
        "patterns": ["vol_", "volatility", "variance", "std"],
    },
    "whale": {
        "description": "Whale activity and concentration",
        "patterns": ["whale_", "concentration", "gini"],
    },
    "regime": {
        "description": "Regime detection features (absorption, churn, etc.)",
        "patterns": ["regime_", "absorption", "churn", "divergence", "range_"],
    },
    "all": {
        "description": "All numeric features",
        "patterns": None,  # Special case: use all
    },
}


def load_data(data_dir: Path, sample_frac: float = 1.0, symbol: Optional[str] = None) -> pl.DataFrame:
    """Load Parquet files from data directory."""
    files = list(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    print(f"Found {len(files)} Parquet files")

    dfs = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        raise ValueError("No data loaded")

    df = pl.concat(dfs)
    print(f"Loaded {len(df):,} total rows")

    # Filter by symbol if specified
    if symbol and "symbol" in df.columns:
        df = df.filter(pl.col("symbol") == symbol)
        print(f"Filtered to {len(df):,} rows for symbol {symbol}")

    # Sample if requested
    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        print(f"Sampled to {len(df):,} rows")

    return df


def get_feature_columns(df: pl.DataFrame, subset: str) -> List[str]:
    """Get feature columns matching a subset pattern."""
    if subset not in FEATURE_SUBSETS:
        raise ValueError(f"Unknown subset: {subset}. Available: {list(FEATURE_SUBSETS.keys())}")

    patterns = FEATURE_SUBSETS[subset]["patterns"]

    # Get all numeric columns
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        and c not in ["timestamp_ns", "sequence_id", "timestamp"]
    ]

    if patterns is None:  # "all" subset
        return numeric_cols

    # Filter by patterns
    matching = []
    for col in numeric_cols:
        col_lower = col.lower()
        if any(p.lower() in col_lower for p in patterns):
            matching.append(col)

    return matching


def prepare_features(df: pl.DataFrame, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Prepare feature matrix: select, drop nulls, scale."""
    # Select columns that exist
    available = [c for c in columns if c in df.columns]
    if not available:
        raise ValueError(f"None of the requested columns found in data")

    missing = set(columns) - set(available)
    if missing:
        print(f"Warning: {len(missing)} columns not found: {list(missing)[:5]}...")

    # Extract and clean
    X_df = df.select(available).drop_nulls()
    print(f"After dropping nulls: {len(X_df):,} rows, {len(available)} features")

    if len(X_df) < 100:
        raise ValueError("Too few samples after dropping nulls")

    X = X_df.to_numpy()

    # Handle infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available


def compute_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA and return embedding + explained variance."""
    pca = PCA(n_components=min(n_components, X.shape[1]))
    embedding = pca.fit_transform(X)
    return embedding, pca.explained_variance_ratio_


def compute_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Compute UMAP embedding."""
    if not HAS_UMAP:
        raise ImportError("UMAP not available")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        n_jobs=-1,
    )
    return reducer.fit_transform(X)


def compute_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Compute t-SNE embedding."""
    # t-SNE is slow, use PCA first if high-dimensional
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X) // 4),
        random_state=42,
        n_jobs=-1,
    )
    return tsne.fit_transform(X)


def find_clusters(X: np.ndarray, method: str = "gmm", max_clusters: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Find clusters in the data using various methods.
    Returns labels and metadata about the clustering.
    """
    results = {"method": method}

    if method == "gmm":
        # Try different numbers of components, select by BIC
        bics = []
        for n in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
            gmm.fit(X)
            bics.append((n, gmm.bic(X)))

        best_n = min(bics, key=lambda x: x[1])[0]
        gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=5)
        labels = gmm.fit_predict(X)

        results["n_clusters"] = best_n
        results["bic_scores"] = bics
        results["probabilities"] = gmm.predict_proba(X)

    elif method == "kmeans":
        # Use silhouette score to select k
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_k = kmeans.fit_predict(X)
            score = silhouette_score(X, labels_k)
            scores.append((k, score))

        best_k = max(scores, key=lambda x: x[1])[0]
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        results["n_clusters"] = best_k
        results["silhouette_scores"] = scores

    elif method == "hdbscan":
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN not available")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(50, len(X) // 100))
        labels = clusterer.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        results["n_clusters"] = n_clusters
        results["n_noise"] = n_noise
        results["noise_ratio"] = n_noise / len(labels)

    elif method == "dbscan":
        # DBSCAN with automatic eps estimation
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        distances = np.sort(distances[:, -1])

        # Use knee point as eps
        eps = np.percentile(distances, 90)

        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        results["n_clusters"] = n_clusters
        results["eps"] = eps

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Compute silhouette score if we have valid clusters
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)
    if n_unique >= 2:
        mask = labels != -1
        if mask.sum() > n_unique:
            results["silhouette"] = silhouette_score(X[mask], labels[mask])

    return labels, results


def analyze_clusters(X: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> Dict:
    """Analyze what makes each cluster different."""
    analysis = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        mask = labels == label
        cluster_X = X[mask]

        # Mean and std of each feature in this cluster
        means = cluster_X.mean(axis=0)
        stds = cluster_X.std(axis=0)

        # Which features are most distinctive (far from global mean)?
        global_mean = X.mean(axis=0)
        global_std = X.std(axis=0)

        # Z-score of cluster mean vs global
        distinctiveness = (means - global_mean) / (global_std + 1e-10)

        # Top distinctive features
        top_idx = np.argsort(np.abs(distinctiveness))[::-1][:10]

        analysis[label] = {
            "size": mask.sum(),
            "proportion": mask.mean(),
            "top_features": [
                {
                    "name": feature_names[i],
                    "cluster_mean": means[i],
                    "cluster_std": stds[i],
                    "distinctiveness": distinctiveness[i],
                }
                for i in top_idx
            ],
        }

    return analysis


def plot_embedding(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Feature Embedding",
    color_by: Optional[np.ndarray] = None,
    color_label: str = "Value",
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.3,
    s: float = 1,
) -> plt.Axes:
    """Plot 2D embedding with optional coloring."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Noise"
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=f"{label_name} ({mask.sum():,})",
                alpha=alpha,
                s=s,
            )
        ax.legend(loc="upper right", fontsize=8)

    elif color_by is not None:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color_by,
            cmap="viridis",
            alpha=alpha,
            s=s,
        )
        plt.colorbar(scatter, ax=ax, label=color_label)

    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=s, c="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    return ax


def plot_cluster_characteristics(analysis: Dict, title: str = "Cluster Characteristics"):
    """Plot what makes each cluster distinctive."""
    n_clusters = len(analysis)
    if n_clusters == 0:
        print("No clusters to plot")
        return

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 6))
    if n_clusters == 1:
        axes = [axes]

    for ax, (label, info) in zip(axes, analysis.items()):
        features = info["top_features"][:8]  # Top 8
        names = [f["name"][:20] for f in features]
        values = [f["distinctiveness"] for f in features]
        colors = ["green" if v > 0 else "red" for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Distinctiveness (z-score)")
        ax.set_title(f"Cluster {label}\n({info['size']:,} samples, {info['proportion']*100:.1f}%)")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_pca_variance(variance_ratio: np.ndarray, ax: Optional[plt.Axes] = None):
    """Plot PCA explained variance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    cumulative = np.cumsum(variance_ratio)
    x = np.arange(1, len(variance_ratio) + 1)

    ax.bar(x, variance_ratio, alpha=0.7, label="Individual")
    ax.plot(x, cumulative, "r-o", label="Cumulative")
    ax.axhline(y=0.9, color="g", linestyle="--", label="90% threshold")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()

    # Find number of components for 90% variance
    n_90 = np.argmax(cumulative >= 0.9) + 1
    ax.annotate(
        f"{n_90} components\nfor 90%",
        xy=(n_90, 0.9),
        xytext=(n_90 + 2, 0.8),
        arrowprops=dict(arrowstyle="->"),
    )

    return ax


def run_exploration(
    df: pl.DataFrame,
    subset: str,
    output_dir: Path,
    methods: List[str] = ["pca", "umap"],
    cluster_method: str = "gmm",
    max_samples: int = 50000,
):
    """Run full exploration pipeline for a feature subset."""
    print(f"\n{'='*60}")
    print(f"Exploring: {subset}")
    print(f"Description: {FEATURE_SUBSETS[subset]['description']}")
    print(f"{'='*60}")

    # Get feature columns
    columns = get_feature_columns(df, subset)
    print(f"Found {len(columns)} features matching '{subset}'")

    if len(columns) < 2:
        print(f"Skipping {subset}: too few features")
        return

    print(f"Features: {columns[:10]}{'...' if len(columns) > 10 else ''}")

    # Prepare data
    try:
        X, feature_names = prepare_features(df, columns)
    except ValueError as e:
        print(f"Skipping {subset}: {e}")
        return

    # Subsample if too large
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        print(f"Subsampled to {max_samples:,} points for visualization")

    # Create figure
    n_methods = len(methods) + 1  # +1 for PCA variance plot
    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))

    # PCA (always compute for variance analysis)
    print("Computing PCA...")
    pca_embedding, variance_ratio = compute_pca(X, n_components=min(20, X.shape[1]))

    # Plot PCA variance
    plot_pca_variance(variance_ratio[:20], ax=axes[0, 0])

    # Find clusters
    print(f"Finding clusters using {cluster_method}...")
    labels, cluster_info = find_clusters(X, method=cluster_method)
    print(f"Found {cluster_info.get('n_clusters', 'N/A')} clusters")
    if "silhouette" in cluster_info:
        print(f"Silhouette score: {cluster_info['silhouette']:.3f}")

    # Plot PCA embedding
    plot_embedding(
        pca_embedding[:, :2],
        labels=labels,
        title=f"PCA - {subset}",
        ax=axes[1, 0],
    )

    # Other methods
    method_idx = 1
    for method in methods:
        if method == "pca":
            continue  # Already done

        print(f"Computing {method.upper()}...")
        try:
            if method == "umap":
                embedding = compute_umap(X)
            elif method == "tsne":
                embedding = compute_tsne(X)
            else:
                continue

            # Plot without clusters
            plot_embedding(
                embedding,
                title=f"{method.upper()} - {subset} (raw)",
                ax=axes[0, method_idx],
            )

            # Plot with clusters
            plot_embedding(
                embedding,
                labels=labels,
                title=f"{method.upper()} - {subset} (clustered)",
                ax=axes[1, method_idx],
            )

            method_idx += 1

        except Exception as e:
            print(f"  {method} failed: {e}")

    plt.suptitle(f"Feature Exploration: {subset}", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"explore_{subset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Analyze clusters
    if cluster_info.get("n_clusters", 0) >= 2:
        print("\nCluster Analysis:")
        analysis = analyze_clusters(X, labels, feature_names)

        for label, info in analysis.items():
            print(f"\n  Cluster {label}: {info['size']:,} samples ({info['proportion']*100:.1f}%)")
            print("  Top distinctive features:")
            for feat in info["top_features"][:5]:
                direction = "HIGH" if feat["distinctiveness"] > 0 else "LOW"
                print(f"    - {feat['name']}: {direction} (z={feat['distinctiveness']:.2f})")

        # Plot cluster characteristics
        fig = plot_cluster_characteristics(analysis, title=f"Cluster Characteristics: {subset}")
        if fig:
            char_path = output_dir / f"clusters_{subset}.png"
            plt.savefig(char_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {char_path}")
            plt.close()

    return {
        "subset": subset,
        "n_features": len(feature_names),
        "n_samples": len(X),
        "cluster_info": cluster_info,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explore natural clusters in NAT feature data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("rust/data/features"),
        help="Directory containing Parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/exploration"),
        help="Directory for output plots",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=f"Feature subset to explore. Options: {list(FEATURE_SUBSETS.keys())}",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Filter to specific symbol (e.g., BTC)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to sample",
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="gmm",
        choices=["gmm", "kmeans", "hdbscan", "dbscan"],
        help="Clustering method to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum samples for visualization",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List available columns and exit",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NAT Feature Exploration")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    try:
        df = load_data(args.data_dir, sample_frac=args.sample_frac, symbol=args.symbol)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate data, run: make run")
        print("Let it collect data for a few minutes, then try again.")
        return 1

    # List columns if requested
    if args.list_columns:
        print(f"\nAvailable columns ({len(df.columns)}):")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            print(f"  {col}: {dtype}")
        return 0

    # Determine methods to use
    methods = ["pca"]
    if HAS_UMAP:
        methods.append("umap")
    methods.append("tsne")  # Always available via sklearn

    # Run exploration
    if args.subset:
        # Single subset
        run_exploration(
            df, args.subset, args.output_dir,
            methods=methods,
            cluster_method=args.cluster_method,
            max_samples=args.max_samples,
        )
    else:
        # All subsets
        results = []
        for subset in FEATURE_SUBSETS.keys():
            if subset == "all":
                continue  # Skip "all" in batch mode
            result = run_exploration(
                df, subset, args.output_dir,
                methods=methods,
                cluster_method=args.cluster_method,
                max_samples=args.max_samples,
            )
            if result:
                results.append(result)

        # Summary
        print("\n" + "=" * 60)
        print("EXPLORATION SUMMARY")
        print("=" * 60)
        for r in results:
            n_clusters = r["cluster_info"].get("n_clusters", "?")
            silhouette = r["cluster_info"].get("silhouette", None)
            sil_str = f", silhouette={silhouette:.3f}" if silhouette else ""
            print(f"  {r['subset']:15} {r['n_features']:3} features, {n_clusters} clusters{sil_str}")

    print(f"\nPlots saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the plots - do natural clusters appear?")
    print("2. Which feature subsets show clearest structure?")
    print("3. What do the cluster characteristics suggest?")
    print("4. Try different --cluster-method options")

    return 0


if __name__ == "__main__":
    sys.exit(main())

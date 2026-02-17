"""
Validate NAT Core Hypothesis

Computes mutual information between entropy features and regime labels.
Makes go/no-go decision based on whether signal exists.

Key Question: Does entropy predict regime?
- If MI(entropy, regime) > threshold → Signal exists, proceed
- If MI(entropy, regime) < threshold → No signal, pivot

Usage:
    python validate_hypothesis.py
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from config import RESULTS_DIR, MI_SIGNIFICANCE_THRESHOLD


def compute_mutual_information(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    n_neighbors: int = 5,
) -> Dict[str, float]:
    """
    Compute mutual information between features and regime labels.

    Uses sklearn's mutual_info_classif which handles continuous features.
    """
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Compute MI for each feature
    mi_scores = mutual_info_classif(features, y, n_neighbors=n_neighbors, random_state=42)

    return dict(zip(feature_names, mi_scores))


def compute_conditional_entropy(
    entropy_values: np.ndarray,
    regime_labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute H(regime | entropy) and compare to H(regime).

    If H(regime | entropy) << H(regime), entropy is informative.
    """
    # Discretize entropy
    entropy_bins = pd.qcut(entropy_values, q=n_bins, labels=False, duplicates='drop')

    # H(regime)
    regime_counts = pd.Series(regime_labels).value_counts(normalize=True)
    h_regime = -np.sum(regime_counts * np.log2(regime_counts + 1e-10))

    # H(regime | entropy)
    h_regime_given_entropy = 0.0
    for bin_val in np.unique(entropy_bins):
        mask = entropy_bins == bin_val
        p_bin = mask.mean()

        if p_bin > 0:
            regime_in_bin = pd.Series(regime_labels[mask]).value_counts(normalize=True)
            h_in_bin = -np.sum(regime_in_bin * np.log2(regime_in_bin + 1e-10))
            h_regime_given_entropy += p_bin * h_in_bin

    # Information gain
    info_gain = h_regime - h_regime_given_entropy
    info_gain_ratio = info_gain / h_regime if h_regime > 0 else 0

    return {
        'h_regime': h_regime,
        'h_regime_given_entropy': h_regime_given_entropy,
        'information_gain': info_gain,
        'information_gain_ratio': info_gain_ratio,
    }


def statistical_significance_test(
    mi_score: float,
    n_samples: int,
    n_permutations: int = 1000,
    features: np.ndarray = None,
    labels: np.ndarray = None,
) -> Tuple[float, bool]:
    """
    Test if MI score is statistically significant via permutation test.

    Returns p-value and whether to reject null hypothesis.
    """
    if features is None or labels is None:
        # Use approximation based on sample size
        # MI approximately follows chi-squared under null
        # Degrees of freedom ~ (n_bins - 1) * (n_classes - 1)
        df = 9 * 2  # Assuming 10 bins and 3 classes
        chi2_stat = 2 * n_samples * mi_score
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    else:
        # Permutation test
        null_mi = []
        for _ in range(n_permutations):
            shuffled_labels = np.random.permutation(labels)
            le = LabelEncoder()
            y = le.fit_transform(shuffled_labels)
            mi = mutual_info_classif(features.reshape(-1, 1), y, n_neighbors=5, random_state=None)[0]
            null_mi.append(mi)

        p_value = np.mean(np.array(null_mi) >= mi_score)

    significant = p_value < 0.05
    return p_value, significant


def create_visualization(labels_df: pd.DataFrame, mi_scores: Dict[str, float]):
    """Create visualization of validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Entropy distribution by regime
    ax1 = axes[0, 0]
    for regime in ['MR', 'TF', 'NA']:
        data = labels_df[labels_df['regime'] == regime]['avg_entropy']
        if len(data) > 0:
            ax1.hist(data, bins=30, alpha=0.5, label=f'{regime} (n={len(data)})', density=True)
    ax1.set_xlabel('Permutation Entropy (16)')
    ax1.set_ylabel('Density')
    ax1.set_title('Entropy Distribution by Regime')
    ax1.legend()
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Theoretical midpoint')

    # 2. Mutual Information scores
    ax2 = axes[0, 1]
    feature_names = list(mi_scores.keys())
    scores = list(mi_scores.values())
    colors = ['green' if s > MI_SIGNIFICANCE_THRESHOLD else 'red' for s in scores]

    bars = ax2.barh(feature_names, scores, color=colors)
    ax2.axvline(x=MI_SIGNIFICANCE_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold ({MI_SIGNIFICANCE_THRESHOLD})')
    ax2.set_xlabel('Mutual Information (bits)')
    ax2.set_title('MI(Feature, Regime)')
    ax2.legend()

    # 3. Entropy vs Strategy PnL scatter
    ax3 = axes[1, 0]
    ax3.scatter(labels_df['avg_entropy'], labels_df['asmm_pnl'],
                alpha=0.3, label='ASMM', color='blue')
    ax3.scatter(labels_df['avg_entropy'], labels_df['trend_pnl'],
                alpha=0.3, label='TrendFollow', color='orange')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Average Entropy')
    ax3.set_ylabel('Strategy PnL')
    ax3.set_title('Entropy vs Strategy Performance')
    ax3.legend()

    # Add trend lines
    z_asmm = np.polyfit(labels_df['avg_entropy'], labels_df['asmm_pnl'], 1)
    z_trend = np.polyfit(labels_df['avg_entropy'], labels_df['trend_pnl'], 1)
    x_line = np.linspace(labels_df['avg_entropy'].min(), labels_df['avg_entropy'].max(), 100)
    ax3.plot(x_line, np.polyval(z_asmm, x_line), 'b--', alpha=0.7)
    ax3.plot(x_line, np.polyval(z_trend, x_line), 'r--', alpha=0.7)

    # 4. Regime distribution pie chart
    ax4 = axes[1, 1]
    regime_counts = labels_df['regime'].value_counts()
    colors_pie = {'MR': 'blue', 'TF': 'orange', 'NA': 'gray'}
    ax4.pie(regime_counts.values, labels=regime_counts.index,
            colors=[colors_pie.get(r, 'gray') for r in regime_counts.index],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Regime Distribution')

    plt.tight_layout()

    # Save figure
    fig_path = RESULTS_DIR / 'validation_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")

    plt.show()


def make_decision(mi_scores: Dict[str, float], conditional_entropy: Dict[str, float]) -> str:
    """
    Make go/no-go decision based on validation results.
    """
    # Key metrics
    entropy_mi = mi_scores.get('avg_entropy', 0)
    imbalance_mi = mi_scores.get('avg_imbalance_persist', 0)
    aggressor_mi = mi_scores.get('avg_aggressor_mom', 0)

    # Composite score
    composite_mi = entropy_mi + 0.5 * imbalance_mi + 0.5 * aggressor_mi

    info_gain_ratio = conditional_entropy.get('information_gain_ratio', 0)

    print("\n" + "=" * 60)
    print("GO / NO-GO DECISION")
    print("=" * 60)

    print(f"\nKey Metrics:")
    print(f"  Entropy MI:           {entropy_mi:.4f} (threshold: {MI_SIGNIFICANCE_THRESHOLD})")
    print(f"  Imbalance Persist MI: {imbalance_mi:.4f}")
    print(f"  Aggressor Mom MI:     {aggressor_mi:.4f}")
    print(f"  Composite MI:         {composite_mi:.4f}")
    print(f"  Info Gain Ratio:      {info_gain_ratio:.4f}")

    # Decision logic
    if entropy_mi > MI_SIGNIFICANCE_THRESHOLD:
        decision = "GO"
        reason = "Entropy shows significant predictive power for regime"
        confidence = "HIGH" if entropy_mi > 2 * MI_SIGNIFICANCE_THRESHOLD else "MEDIUM"
    elif composite_mi > MI_SIGNIFICANCE_THRESHOLD * 1.5:
        decision = "GO (COMPOSITE)"
        reason = "Composite signal shows predictive power even though entropy alone is weak"
        confidence = "MEDIUM"
    elif info_gain_ratio > 0.1:
        decision = "INVESTIGATE"
        reason = "Information gain exists but MI test inconclusive - needs more data"
        confidence = "LOW"
    else:
        decision = "NO-GO / PIVOT"
        reason = "No significant signal detected - consider alternative approaches"
        confidence = "HIGH"

    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"CONFIDENCE: {confidence}")
    print(f"REASON: {reason}")
    print(f"{'='*60}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if decision == "GO":
        print("  1. Proceed to Phase 3: Train XGBoost regime classifier")
        print("  2. Use entropy as primary feature, add composite signals")
        print("  3. Implement walk-forward validation")
    elif decision == "GO (COMPOSITE)":
        print("  1. Proceed with caution - composite signal, not pure entropy")
        print("  2. Focus on imbalance_persistence + aggressor_momentum")
        print("  3. Consider ensemble of weak signals")
    elif decision == "INVESTIGATE":
        print("  1. Collect more data (target: 24+ hours)")
        print("  2. Try different entropy measures (sample entropy, ApEn)")
        print("  3. Investigate cross-asset entropy")
    else:
        print("  1. Consider pivoting to funding rate prediction (proven alpha)")
        print("  2. Try transfer entropy networks instead of permutation entropy")
        print("  3. Investigate information geometry approach")

    return decision


def main():
    parser = argparse.ArgumentParser(description='Validate NAT hypothesis')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to regime labels parquet')
    args = parser.parse_args()

    print("=" * 60)
    print("NAT Week 1-2 Validation - Hypothesis Testing")
    print("=" * 60)

    # Load labels
    if args.labels:
        labels_path = Path(args.labels)
    else:
        labels_path = RESULTS_DIR / 'regime_labels.parquet'

    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        print("Please run 'python regime_labeler.py' first")
        return

    labels_df = pd.read_parquet(labels_path)
    print(f"Loaded {len(labels_df)} labeled windows")

    # Feature columns
    feature_cols = ['avg_entropy', 'avg_imbalance_persist', 'avg_aggressor_mom']
    features = labels_df[feature_cols].values
    regime_labels = labels_df['regime'].values

    # 1. Compute Mutual Information
    print("\n" + "-" * 40)
    print("Computing Mutual Information...")
    print("-" * 40)

    mi_scores = compute_mutual_information(
        features, regime_labels, feature_cols
    )

    print("\nMutual Information Scores:")
    for name, score in sorted(mi_scores.items(), key=lambda x: -x[1]):
        status = "SIGNIFICANT" if score > MI_SIGNIFICANCE_THRESHOLD else "not significant"
        print(f"  {name}: {score:.4f} ({status})")

    # 2. Compute Conditional Entropy
    print("\n" + "-" * 40)
    print("Computing Conditional Entropy...")
    print("-" * 40)

    cond_entropy = compute_conditional_entropy(
        labels_df['avg_entropy'].values,
        regime_labels
    )

    print(f"  H(regime):                 {cond_entropy['h_regime']:.4f} bits")
    print(f"  H(regime | entropy):       {cond_entropy['h_regime_given_entropy']:.4f} bits")
    print(f"  Information Gain:          {cond_entropy['information_gain']:.4f} bits")
    print(f"  Information Gain Ratio:    {cond_entropy['information_gain_ratio']:.4f}")

    # 3. Statistical Significance
    print("\n" + "-" * 40)
    print("Statistical Significance Tests...")
    print("-" * 40)

    for col in feature_cols:
        mi_score = mi_scores[col]
        p_value, significant = statistical_significance_test(
            mi_score, len(labels_df),
            features=labels_df[col].values,
            labels=regime_labels
        )
        status = "SIGNIFICANT" if significant else "NOT significant"
        print(f"  {col}: p-value = {p_value:.4f} ({status})")

    # 4. Correlation Analysis
    print("\n" + "-" * 40)
    print("Correlation Analysis...")
    print("-" * 40)

    # Entropy vs ASMM performance
    corr_asmm, p_asmm = stats.pearsonr(labels_df['avg_entropy'], labels_df['asmm_pnl'])
    corr_trend, p_trend = stats.pearsonr(labels_df['avg_entropy'], labels_df['trend_pnl'])

    print(f"  Entropy vs ASMM PnL:   r = {corr_asmm:.4f}, p = {p_asmm:.4f}")
    print(f"  Entropy vs Trend PnL:  r = {corr_trend:.4f}, p = {p_trend:.4f}")

    # Expected: negative correlation with ASMM (high entropy = bad for MM)
    #           positive correlation with Trend (high entropy = ???)
    if corr_asmm < 0 and p_asmm < 0.05:
        print("  ✓ Expected: High entropy hurts market making")
    if corr_trend > 0 and p_trend < 0.05:
        print("  ? High entropy helps trend following (unexpected)")

    # 5. Create Visualization
    print("\n" + "-" * 40)
    print("Creating Visualization...")
    print("-" * 40)

    try:
        create_visualization(labels_df, mi_scores)
    except Exception as e:
        print(f"  Warning: Could not create visualization: {e}")

    # 6. Make Decision
    decision = make_decision(mi_scores, cond_entropy)

    # 7. Save results
    results = {
        'n_samples': len(labels_df),
        'mi_scores': mi_scores,
        'conditional_entropy': cond_entropy,
        'correlation_asmm': corr_asmm,
        'correlation_trend': corr_trend,
        'decision': decision,
    }

    results_path = RESULTS_DIR / 'validation_results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()

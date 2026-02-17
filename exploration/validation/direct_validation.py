#!/usr/bin/env python3
"""
Direct Hypothesis Validation for NAT

Instead of simulating strategies to label regimes, this directly tests:
1. Does low entropy predict momentum (autocorrelation > 0)?
2. Does high entropy predict mean-reversion (autocorrelation < 0)?

This is a cleaner test of the core hypothesis.

Usage:
    python direct_validation.py
    python direct_validation.py --synthetic
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from config import FEATURES_DIR, RESULTS_DIR, MI_SIGNIFICANCE_THRESHOLD


def compute_future_characteristics(df: pd.DataFrame, horizon: int = 50) -> pd.DataFrame:
    """
    Compute future price behavior characteristics for each point.

    For each timestamp t, compute:
    - future_autocorr: Autocorrelation of returns in [t, t+horizon]
    - future_momentum: Cumulative return in [t, t+horizon]
    - future_volatility: Std of returns in [t, t+horizon]
    """
    prices = df['raw_mid_price'].values
    n = len(prices)

    future_autocorr = []
    future_momentum = []
    future_volatility = []
    future_regime = []  # Derived from autocorr

    for i in range(n - horizon):
        future_prices = prices[i:i+horizon]
        returns = np.diff(future_prices) / future_prices[:-1]

        if len(returns) < 10:
            future_autocorr.append(np.nan)
            future_momentum.append(np.nan)
            future_volatility.append(np.nan)
            future_regime.append('NA')
            continue

        # Autocorrelation at lag 1
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Total momentum
        momentum = (future_prices[-1] - future_prices[0]) / future_prices[0]

        # Volatility
        vol = np.std(returns)

        future_autocorr.append(autocorr)
        future_momentum.append(momentum)
        future_volatility.append(vol)

        # Label regime based on autocorrelation
        # Positive autocorr = trending (TF regime)
        # Negative autocorr = mean-reverting (MR regime)
        if autocorr > 0.1:
            future_regime.append('TF')
        elif autocorr < -0.1:
            future_regime.append('MR')
        else:
            future_regime.append('NA')

    # Pad with NaN for last horizon points
    future_autocorr.extend([np.nan] * horizon)
    future_momentum.extend([np.nan] * horizon)
    future_volatility.extend([np.nan] * horizon)
    future_regime.extend(['NA'] * horizon)

    df = df.copy()
    df['future_autocorr'] = future_autocorr
    df['future_momentum'] = future_momentum
    df['future_volatility'] = future_volatility
    df['future_regime'] = future_regime

    return df


def direct_hypothesis_test(df: pd.DataFrame) -> dict:
    """
    Directly test: Does entropy predict future price behavior?

    Core hypothesis:
    - Low entropy → Trending (positive autocorr) → TrendFollow works
    - High entropy → Mean-reverting (negative autocorr) → ASMM works
    """
    # Remove NaN
    df_clean = df.dropna(subset=['future_autocorr', 'ent_permutation_16'])

    if len(df_clean) < 100:
        print("WARNING: Not enough clean data points")
        return {}

    entropy = df_clean['ent_permutation_16'].values
    autocorr = df_clean['future_autocorr'].values
    momentum = df_clean['future_momentum'].values
    regimes = df_clean['future_regime'].values

    print(f"\nAnalyzing {len(df_clean)} data points...")

    results = {}

    # Test 1: Correlation between entropy and autocorrelation
    corr_autocorr, p_autocorr = stats.pearsonr(entropy, autocorr)
    results['corr_entropy_autocorr'] = corr_autocorr
    results['p_entropy_autocorr'] = p_autocorr

    print(f"\n1. ENTROPY vs FUTURE AUTOCORRELATION")
    print(f"   Correlation: {corr_autocorr:.4f}")
    print(f"   P-value:     {p_autocorr:.4f}")

    # HYPOTHESIS: Low entropy should predict POSITIVE autocorr (trends)
    # So correlation should be NEGATIVE (low entropy → high autocorr)
    if corr_autocorr < 0 and p_autocorr < 0.05:
        print(f"   ✓ SUPPORTS HYPOTHESIS: Low entropy predicts trending behavior")
        results['hypothesis_1_supported'] = True
    elif corr_autocorr > 0 and p_autocorr < 0.05:
        print(f"   ✗ CONTRADICTS HYPOTHESIS: High entropy predicts trending behavior")
        results['hypothesis_1_supported'] = False
    else:
        print(f"   ? INCONCLUSIVE: No significant relationship")
        results['hypothesis_1_supported'] = None

    # Test 2: Entropy by regime (if regimes have variance)
    regime_counts = pd.Series(regimes).value_counts()
    print(f"\n2. REGIME DISTRIBUTION (based on future autocorr)")
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"   {regime}: {count} ({pct:.1f}%)")

    results['regime_distribution'] = regime_counts.to_dict()

    # Test 3: Entropy mean by regime
    if len(regime_counts) > 1 and 'TF' in regime_counts and 'MR' in regime_counts:
        tf_entropy = entropy[regimes == 'TF']
        mr_entropy = entropy[regimes == 'MR']

        print(f"\n3. ENTROPY BY REGIME")
        print(f"   TF (trending):      mean = {tf_entropy.mean():.4f}, std = {tf_entropy.std():.4f}")
        print(f"   MR (mean-reverting): mean = {mr_entropy.mean():.4f}, std = {mr_entropy.std():.4f}")

        # T-test
        t_stat, t_pval = stats.ttest_ind(tf_entropy, mr_entropy)
        print(f"   T-test: t = {t_stat:.4f}, p = {t_pval:.4f}")

        results['tf_entropy_mean'] = tf_entropy.mean()
        results['mr_entropy_mean'] = mr_entropy.mean()
        results['t_stat'] = t_stat
        results['t_pval'] = t_pval

        # HYPOTHESIS: TF should have LOWER entropy than MR
        if tf_entropy.mean() < mr_entropy.mean() and t_pval < 0.05:
            print(f"   ✓ SUPPORTS HYPOTHESIS: Trending periods have lower entropy")
            results['hypothesis_2_supported'] = True
        elif tf_entropy.mean() > mr_entropy.mean() and t_pval < 0.05:
            print(f"   ✗ CONTRADICTS HYPOTHESIS: Trending periods have higher entropy")
            results['hypothesis_2_supported'] = False
        else:
            print(f"   ? INCONCLUSIVE")
            results['hypothesis_2_supported'] = None

    # Test 4: Mutual Information
    # Only compute if we have regime variance
    valid_regimes = regimes[regimes != 'NA']
    valid_entropy = entropy[regimes != 'NA']

    if len(np.unique(valid_regimes)) > 1:
        le = LabelEncoder()
        y = le.fit_transform(valid_regimes)
        mi = mutual_info_classif(valid_entropy.reshape(-1, 1), y, n_neighbors=5, random_state=42)[0]

        print(f"\n4. MUTUAL INFORMATION")
        print(f"   MI(entropy, regime) = {mi:.4f}")
        print(f"   Threshold:            {MI_SIGNIFICANCE_THRESHOLD}")

        results['mi_entropy_regime'] = mi

        if mi > MI_SIGNIFICANCE_THRESHOLD:
            print(f"   ✓ SIGNIFICANT: Entropy predicts regime")
            results['mi_significant'] = True
        else:
            print(f"   ✗ NOT SIGNIFICANT")
            results['mi_significant'] = False
    else:
        print(f"\n4. MUTUAL INFORMATION")
        print(f"   Cannot compute - only one regime present")
        results['mi_entropy_regime'] = 0
        results['mi_significant'] = False

    # Test 5: Composite signal test
    print(f"\n5. COMPOSITE SIGNAL TEST")

    if 'imbalance_persistence' in df_clean.columns:
        imb_persist = df_clean['imbalance_persistence'].values

        # Imbalance persistence should be higher during trends
        corr_imb, p_imb = stats.pearsonr(imb_persist, autocorr)
        print(f"   Imbalance persistence vs autocorr: r = {corr_imb:.4f}, p = {p_imb:.4f}")
        results['corr_imb_autocorr'] = corr_imb

        if corr_imb > 0 and p_imb < 0.05:
            print(f"   ✓ Imbalance persistence predicts trending")

    if 'flow_aggressor_momentum' in df_clean.columns:
        agg_mom = df_clean['flow_aggressor_momentum'].values

        # Aggressor momentum should correlate with future momentum
        corr_agg, p_agg = stats.pearsonr(np.abs(agg_mom), np.abs(autocorr))
        print(f"   |Aggressor momentum| vs |autocorr|: r = {corr_agg:.4f}, p = {p_agg:.4f}")
        results['corr_agg_autocorr'] = corr_agg

    return results


def make_decision(results: dict) -> str:
    """Make go/no-go decision based on direct validation results."""
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)

    # Count supporting evidence
    support_count = 0
    total_tests = 0

    if 'hypothesis_1_supported' in results:
        total_tests += 1
        if results['hypothesis_1_supported'] is True:
            support_count += 1

    if 'hypothesis_2_supported' in results:
        total_tests += 1
        if results['hypothesis_2_supported'] is True:
            support_count += 1

    if 'mi_significant' in results:
        total_tests += 1
        if results['mi_significant'] is True:
            support_count += 1

    print(f"\nSupporting evidence: {support_count}/{total_tests} tests")

    # Decision logic
    if support_count >= 2:
        decision = "GO"
        confidence = "HIGH" if support_count == total_tests else "MEDIUM"
        reason = "Multiple tests support entropy-regime relationship"
    elif support_count == 1:
        decision = "INVESTIGATE"
        confidence = "LOW"
        reason = "Partial support - collect more data"
    else:
        decision = "NO-GO / PIVOT"
        confidence = "HIGH" if total_tests >= 2 else "MEDIUM"
        reason = "No significant support for entropy hypothesis"

    print(f"\nDECISION:   {decision}")
    print(f"CONFIDENCE: {confidence}")
    print(f"REASON:     {reason}")

    if decision == "GO":
        print("\nNEXT STEPS:")
        print("  1. Proceed to XGBoost classifier training")
        print("  2. Use walk-forward validation")
        print("  3. Build regime-conditioned backtester")
    elif decision == "INVESTIGATE":
        print("\nNEXT STEPS:")
        print("  1. Collect more live data (24+ hours)")
        print("  2. Try different entropy measures (sample entropy, ApEn)")
        print("  3. Investigate at different timescales")
    else:
        print("\nALTERNATIVE APPROACHES:")
        print("  1. Funding rate prediction (proven, simpler)")
        print("  2. Transfer entropy networks")
        print("  3. Information geometry approach")
        print("  4. Cross-exchange lead-lag")

    return decision


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic data with KNOWN regime structure for testing."""
    print(f"Generating {n_samples} synthetic samples with known regime structure...")
    np.random.seed(42)

    # Generate regime sequence (alternating every ~500 samples)
    regimes = []
    current_regime = 'TF'
    while len(regimes) < n_samples:
        length = int(np.clip(np.random.exponential(500), 200, 1000))
        regimes.extend([current_regime] * length)
        current_regime = 'MR' if current_regime == 'TF' else 'TF'
    regimes = regimes[:n_samples]

    # Generate price series with regime-dependent behavior
    base_price = 50000
    returns = np.zeros(n_samples)

    for i in range(1, n_samples):
        if regimes[i] == 'TF':
            # Trending: POSITIVE autocorrelation
            momentum = 0.4 * returns[i-1]  # Strong momentum
            returns[i] = momentum + np.random.normal(0, 0.0003)
        else:
            # Mean-reverting: NEGATIVE autocorrelation
            mean_rev = -0.3 * returns[i-1]  # Mean reversion
            returns[i] = mean_rev + np.random.normal(0, 0.0005)

    prices = base_price * np.cumprod(1 + returns)

    # Generate features
    data = []
    for i in range(n_samples):
        regime = regimes[i]

        # CRITICAL: Entropy should be LOWER in TF regime (more predictable)
        # and HIGHER in MR regime (more random)
        if regime == 'TF':
            entropy = np.clip(np.random.normal(0.25, 0.08), 0.05, 0.5)
        else:
            entropy = np.clip(np.random.normal(0.75, 0.08), 0.5, 0.95)

        data.append({
            'raw_mid_price': prices[i],
            'ent_permutation_16': entropy,
            'imbalance_persistence': np.random.normal(0.3 if regime == 'TF' else 0.1, 0.1),
            'flow_aggressor_momentum': np.random.normal(0.2 if regime == 'TF' else 0, 0.15),
            '_true_regime': regime,
        })

    df = pd.DataFrame(data)

    # Print regime distribution
    regime_counts = df['_true_regime'].value_counts()
    print(f"True regime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} ({count/len(df)*100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description='Direct NAT hypothesis validation')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--input', type=str, default=None,
                       help='Input parquet file')
    args = parser.parse_args()

    print("=" * 60)
    print("NAT DIRECT HYPOTHESIS VALIDATION")
    print("=" * 60)
    print("\nCore Question: Does entropy predict future price behavior?")
    print("  - Low entropy → Trending (positive autocorr)?")
    print("  - High entropy → Mean-reverting (negative autocorr)?")

    # Load or generate data
    if args.synthetic:
        print("\n[SYNTHETIC MODE]")
        df = generate_synthetic_data(10000)
    else:
        if args.input:
            input_path = Path(args.input)
        else:
            parquet_files = sorted(FEATURES_DIR.glob('*.parquet'))
            if not parquet_files:
                print("\nNo data files found. Use --synthetic for testing.")
                return
            input_path = parquet_files[-1]  # Most recent

        print(f"\nLoading {input_path}...")
        df = pd.read_parquet(input_path)

    print(f"Loaded {len(df)} samples")

    # Compute future characteristics
    print("\nComputing future price behavior...")
    df = compute_future_characteristics(df, horizon=50)

    # Run direct hypothesis test
    results = direct_hypothesis_test(df)

    # Make decision
    decision = make_decision(results)

    # Save results
    results['decision'] = decision
    results['n_samples'] = len(df)
    results['timestamp'] = datetime.utcnow().isoformat()

    import json
    output_path = RESULTS_DIR / 'direct_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Create visualization
    try:
        create_visualization(df, results)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")


def create_visualization(df: pd.DataFrame, results: dict):
    """Create visualization of direct validation results."""
    df_clean = df.dropna(subset=['future_autocorr', 'ent_permutation_16'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Entropy vs Future Autocorrelation scatter
    ax1 = axes[0, 0]
    ax1.scatter(df_clean['ent_permutation_16'], df_clean['future_autocorr'],
                alpha=0.3, s=10)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5)

    # Add trend line
    z = np.polyfit(df_clean['ent_permutation_16'], df_clean['future_autocorr'], 1)
    x_line = np.linspace(df_clean['ent_permutation_16'].min(),
                         df_clean['ent_permutation_16'].max(), 100)
    ax1.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2, label='Trend')

    ax1.set_xlabel('Permutation Entropy')
    ax1.set_ylabel('Future Autocorrelation')
    ax1.set_title(f"Entropy vs Future Autocorr (r={results.get('corr_entropy_autocorr', 0):.3f})")
    ax1.legend()

    # 2. Entropy distribution by future regime
    ax2 = axes[0, 1]
    for regime in ['TF', 'MR', 'NA']:
        data = df_clean[df_clean['future_regime'] == regime]['ent_permutation_16']
        if len(data) > 0:
            ax2.hist(data, bins=30, alpha=0.5, label=f'{regime} (n={len(data)})', density=True)
    ax2.set_xlabel('Permutation Entropy')
    ax2.set_ylabel('Density')
    ax2.set_title('Entropy Distribution by Future Regime')
    ax2.legend()

    # 3. Future autocorrelation distribution
    ax3 = axes[1, 0]
    ax3.hist(df_clean['future_autocorr'], bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='TF threshold')
    ax3.axvline(x=-0.1, color='blue', linestyle='--', alpha=0.5, label='MR threshold')
    ax3.set_xlabel('Future Autocorrelation')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Future Return Autocorrelation')
    ax3.legend()

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
DIRECT VALIDATION SUMMARY
{'='*40}

Samples analyzed: {len(df_clean)}

Entropy vs Autocorrelation:
  Correlation: {results.get('corr_entropy_autocorr', 'N/A'):.4f}
  P-value: {results.get('p_entropy_autocorr', 'N/A'):.4f}

Entropy by Regime:
  TF mean: {results.get('tf_entropy_mean', 'N/A')}
  MR mean: {results.get('mr_entropy_mean', 'N/A')}
  T-test p: {results.get('t_pval', 'N/A')}

Mutual Information:
  MI(entropy, regime): {results.get('mi_entropy_regime', 'N/A'):.4f}
  Significant: {results.get('mi_significant', 'N/A')}

DECISION: {results.get('decision', 'N/A')}
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    fig_path = RESULTS_DIR / 'direct_validation_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")

    plt.close()


if __name__ == '__main__':
    main()

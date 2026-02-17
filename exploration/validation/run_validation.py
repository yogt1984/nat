#!/usr/bin/env python3
"""
NAT Week 1-2 Validation Runner

Complete validation pipeline:
1. Collect data (or use synthetic for quick test)
2. Label regimes
3. Compute mutual information
4. Make go/no-go decision

Usage:
    # Quick test with synthetic data (5 minutes)
    python run_validation.py --synthetic

    # Short real data collection (10 minutes)
    python run_validation.py --duration 600

    # Full validation (1 hour minimum recommended)
    python run_validation.py --duration 3600

    # Multi-day collection (7 days = 604800 seconds)
    python run_validation.py --duration 604800 --background
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import FEATURES_DIR, RESULTS_DIR, ASSETS


def generate_synthetic_data(
    n_samples: int = 10000,
    symbols: list = None,
) -> pd.DataFrame:
    """
    Generate synthetic market data with known regime structure.

    This allows testing the validation pipeline without waiting for live data.
    The synthetic data has:
    - Clear regime structure (so we can verify the pipeline works)
    - Realistic feature distributions
    - Controlled entropy levels
    """
    if symbols is None:
        symbols = ASSETS[:1]  # Just BTC for testing

    print(f"Generating {n_samples} synthetic samples...")
    np.random.seed(42)

    all_data = []

    for symbol in symbols:
        # Generate regime sequence
        # Regime changes every ~500 samples on average
        regimes = []
        current_regime = np.random.choice(['MR', 'TF'])

        while len(regimes) < n_samples:
            # Random regime length between 100 and 2000
            length = int(np.clip(np.random.exponential(500), 100, 2000))
            regimes.extend([current_regime] * length)
            current_regime = 'TF' if current_regime == 'MR' else 'MR'

        regimes = regimes[:n_samples]

        # Base price random walk
        base_price = 50000  # BTC-like
        returns = np.zeros(n_samples)

        for i in range(n_samples):
            if regimes[i] == 'TF':
                # Trending: momentum + small noise
                if i > 0:
                    momentum = 0.3 * returns[i-1] if i > 0 else 0
                else:
                    momentum = 0
                returns[i] = momentum + np.random.normal(0, 0.0005)
            else:
                # Mean-reverting: negative autocorrelation + noise
                if i > 0:
                    mean_reversion = -0.2 * returns[i-1]
                else:
                    mean_reversion = 0
                returns[i] = mean_reversion + np.random.normal(0, 0.0008)

        prices = base_price * np.cumprod(1 + returns)

        # Generate features based on regime
        data = []
        for i in range(n_samples):
            regime = regimes[i]

            # Entropy: lower in TF regime (more predictable), higher in MR (more random)
            if regime == 'TF':
                entropy_base = 0.3 + np.random.normal(0, 0.1)
            else:
                entropy_base = 0.7 + np.random.normal(0, 0.1)

            entropy = np.clip(entropy_base, 0.05, 0.95)

            # Imbalance persistence: higher in TF regime
            if regime == 'TF':
                imb_persist = 0.5 + np.random.normal(0, 0.2)
            else:
                imb_persist = 0.1 + np.random.normal(0, 0.2)

            imb_persist = np.clip(imb_persist, -1, 1)

            # Aggressor momentum: more extreme in TF regime
            if regime == 'TF':
                agg_mom = np.sign(returns[i]) * (0.3 + abs(np.random.normal(0, 0.2)))
            else:
                agg_mom = np.random.normal(0, 0.2)

            agg_mom = np.clip(agg_mom, -1, 1)

            # Other features
            spread_bps = 1 + np.random.exponential(2)
            imbalance = np.random.normal(0, 0.3)
            volume = np.random.exponential(100)

            data.append({
                'timestamp': datetime.utcnow().isoformat(),
                'timestamp_ns': i * 100_000_000,  # 100ms intervals
                'symbol': symbol,
                'raw_mid_price': prices[i],
                'raw_spread_bps': spread_bps,
                'raw_bid_depth_l5': np.random.exponential(1000),
                'raw_ask_depth_l5': np.random.exponential(1000),
                'imbalance_l5': imbalance,
                'imbalance_l10': imbalance + np.random.normal(0, 0.1),
                'imbalance_persistence': imb_persist,
                'flow_aggressor_ratio': 0.5 + agg_mom / 2,
                'flow_volume_5s': volume,
                'flow_trade_count_5s': int(np.random.exponential(20)),
                'flow_aggressor_momentum': agg_mom,
                'vol_realized_100': abs(np.random.normal(0.2, 0.1)),
                'vol_realized_20': abs(np.random.normal(0.25, 0.15)),
                'vol_ratio': np.random.uniform(0.5, 2.0),
                'ent_permutation_8': entropy + np.random.normal(0, 0.05),
                'ent_permutation_16': entropy,
                'ent_permutation_32': entropy + np.random.normal(0, 0.03),
                'ent_book_shape': np.random.uniform(0.3, 0.9),
                'ent_trade_size': np.random.uniform(0.2, 0.8),
                'ent_rate_of_change': np.random.normal(0, 0.02),
                'ent_zscore': np.random.normal(0, 1),
                'composite_regime_signal': (
                    0.4 * (1 - entropy) +
                    0.3 * abs(imb_persist) +
                    0.3 * abs(agg_mom)
                ),
                '_true_regime': regime,  # Hidden ground truth
            })

        all_data.extend(data)

    df = pd.DataFrame(all_data)

    # Print regime distribution
    regime_counts = df['_true_regime'].value_counts()
    print(f"Generated regime distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} ({count/len(df)*100:.1f}%)")

    # Remove ground truth before saving (to simulate real data)
    df_save = df.drop(columns=['_true_regime'])

    return df_save, df['_true_regime']


def run_validation_pipeline(
    use_synthetic: bool = False,
    duration: int = 600,
    symbols: list = None,
):
    """Run the complete validation pipeline."""
    if symbols is None:
        symbols = ASSETS

    print("=" * 70)
    print("NAT WEEK 1-2 VALIDATION PIPELINE")
    print("=" * 70)
    print(f"Mode: {'SYNTHETIC (test)' if use_synthetic else 'LIVE DATA'}")
    print(f"Symbols: {symbols}")
    if not use_synthetic:
        print(f"Duration: {duration} seconds ({duration/3600:.1f} hours)")
    print("=" * 70)

    # Step 1: Get data
    print("\n" + "=" * 50)
    print("STEP 1: DATA COLLECTION")
    print("=" * 50)

    if use_synthetic:
        # Generate synthetic data
        df, true_regimes = generate_synthetic_data(n_samples=10000, symbols=symbols[:1])

        # Save synthetic data
        output_path = FEATURES_DIR / f"synthetic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved synthetic data to {output_path}")

    else:
        # Run live data collection
        print(f"\nCollecting live data for {duration} seconds...")
        print("This may take a while. Press Ctrl+C to stop early.\n")

        try:
            subprocess.run([
                sys.executable, 'data_collector.py',
                '--duration', str(duration),
                '--symbols', *symbols
            ], check=True)
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        except subprocess.CalledProcessError as e:
            print(f"Data collection failed: {e}")
            return

    # Check if we have data
    parquet_files = list(FEATURES_DIR.glob('*.parquet'))
    if not parquet_files:
        print("ERROR: No data collected!")
        return

    print(f"\nFound {len(parquet_files)} data files")

    # Step 2: Label regimes
    print("\n" + "=" * 50)
    print("STEP 2: REGIME LABELING")
    print("=" * 50)

    try:
        subprocess.run([
            sys.executable, 'regime_labeler.py'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Regime labeling failed: {e}")
        return

    # Step 3: Validate hypothesis
    print("\n" + "=" * 50)
    print("STEP 3: HYPOTHESIS VALIDATION")
    print("=" * 50)

    try:
        subprocess.run([
            sys.executable, 'validate_hypothesis.py'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Hypothesis validation failed: {e}")
        return

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - regime_labels.parquet: Labeled regime data")
    print(f"  - validation_results.json: MI scores and decision")
    print(f"  - validation_results.png: Visualization")

    if use_synthetic:
        print("\n⚠️  NOTE: This was a synthetic test run!")
        print("   Synthetic data has KNOWN regime structure, so MI should be high.")
        print("   For real validation, run with live data:")
        print("   python run_validation.py --duration 3600")


def main():
    parser = argparse.ArgumentParser(
        description='NAT Week 1-2 Validation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick synthetic test (5 min):
    python run_validation.py --synthetic

  Short real data (10 min):
    python run_validation.py --duration 600

  Full validation (1 hour):
    python run_validation.py --duration 3600

  Multi-day collection:
    python run_validation.py --duration 604800
        """
    )
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for quick testing')
    parser.add_argument('--duration', type=int, default=600,
                       help='Data collection duration in seconds (default: 600)')
    parser.add_argument('--symbols', nargs='+', default=ASSETS,
                       help=f'Symbols to collect (default: {ASSETS})')

    args = parser.parse_args()

    run_validation_pipeline(
        use_synthetic=args.synthetic,
        duration=args.duration,
        symbols=args.symbols,
    )


if __name__ == '__main__':
    main()

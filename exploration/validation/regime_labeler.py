"""
Regime Labeler for NAT Validation

Simulates ASMM (market making) and TrendFollow strategies on historical data
to label each time period as MR (mean-reversion), TF (trend-following), or NA.

Usage:
    python regime_labeler.py --input data/features/*.parquet
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FEATURES_DIR, RESULTS_DIR, INITIAL_CAPITAL,
    MAKER_FEE, TAKER_FEE, POSITION_SIZE_PCT
)


@dataclass
class Position:
    side: str  # 'long' or 'short' or 'flat'
    size: float
    entry_price: float
    entry_time: int


@dataclass
class StrategyResult:
    pnl: float
    n_trades: int
    win_rate: float
    sharpe: float


class ASMMSimulator:
    """
    Simplified Avellaneda-Stoikov Market Making Simulator.

    ASMM profits when price oscillates around a mean (mean-reversion regime).
    It loses when price trends strongly in one direction.
    """

    def __init__(self, spread_bps: float = 5.0, inventory_limit: float = 1.0):
        self.spread_bps = spread_bps
        self.inventory_limit = inventory_limit
        self.inventory = 0.0
        self.pnl = 0.0
        self.trades = 0
        self.wins = 0

    def reset(self):
        self.inventory = 0.0
        self.pnl = 0.0
        self.trades = 0
        self.wins = 0

    def simulate_window(self, df: pd.DataFrame) -> StrategyResult:
        """
        Simulate ASMM on a data window.

        Logic:
        - Place bid/ask around mid price with spread
        - Get filled when price moves through our levels
        - Profit = spread captured - adverse selection loss
        """
        self.reset()

        returns_list = []
        last_mid = df['raw_mid_price'].iloc[0]

        for i in range(1, len(df)):
            mid = df['raw_mid_price'].iloc[i]
            spread = df['raw_spread_bps'].iloc[i]
            imbalance = df['imbalance_l5'].iloc[i]

            if mid <= 0 or last_mid <= 0:
                last_mid = mid
                continue

            price_change = (mid - last_mid) / last_mid

            # Simulate market making
            # We place bids below mid and asks above mid
            our_spread = self.spread_bps / 10000

            # Probability of getting filled based on imbalance
            # Positive imbalance → more likely our ask gets hit
            fill_prob_ask = 0.3 * (1 + imbalance)
            fill_prob_bid = 0.3 * (1 - imbalance)

            trade_pnl = 0.0

            # Ask fill (we go short)
            if np.random.random() < fill_prob_ask and self.inventory > -self.inventory_limit:
                # Capture half spread
                trade_pnl += our_spread / 2
                self.inventory -= 0.1
                self.trades += 1

            # Bid fill (we go long)
            if np.random.random() < fill_prob_bid and self.inventory < self.inventory_limit:
                trade_pnl += our_spread / 2
                self.inventory += 0.1
                self.trades += 1

            # Mark-to-market on inventory
            inventory_pnl = self.inventory * price_change

            # Net PnL for this period
            period_pnl = trade_pnl + inventory_pnl - abs(self.trades) * MAKER_FEE
            self.pnl += period_pnl
            returns_list.append(period_pnl)

            if period_pnl > 0:
                self.wins += 1

            last_mid = mid

        # Compute Sharpe
        returns = np.array(returns_list)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(len(returns))

        return StrategyResult(
            pnl=self.pnl,
            n_trades=self.trades,
            win_rate=self.wins / max(1, len(returns_list)),
            sharpe=sharpe
        )


class TrendFollowSimulator:
    """
    Simple Trend Following Simulator.

    TrendFollow profits when price moves consistently in one direction.
    It loses in choppy, mean-reverting markets.
    """

    def __init__(self, lookback: int = 20, threshold: float = 0.001):
        self.lookback = lookback
        self.threshold = threshold
        self.position: Optional[Position] = None
        self.pnl = 0.0
        self.trades = 0
        self.wins = 0

    def reset(self):
        self.position = None
        self.pnl = 0.0
        self.trades = 0
        self.wins = 0

    def simulate_window(self, df: pd.DataFrame) -> StrategyResult:
        """
        Simulate trend following on a data window.

        Logic:
        - Compute momentum signal (price change over lookback)
        - Go long if momentum > threshold, short if < -threshold
        - Hold position until signal reverses
        """
        self.reset()

        if len(df) < self.lookback + 10:
            return StrategyResult(pnl=0, n_trades=0, win_rate=0, sharpe=0)

        returns_list = []
        prices = df['raw_mid_price'].values

        for i in range(self.lookback, len(df)):
            current_price = prices[i]
            lookback_price = prices[i - self.lookback]

            if current_price <= 0 or lookback_price <= 0:
                continue

            # Momentum signal
            momentum = (current_price - lookback_price) / lookback_price

            # Current position P&L
            period_pnl = 0.0
            if self.position:
                price_change = (current_price - prices[i-1]) / prices[i-1]
                if self.position.side == 'long':
                    period_pnl = price_change * self.position.size
                else:
                    period_pnl = -price_change * self.position.size

            # Signal-based trading
            if momentum > self.threshold:
                # Want to be long
                if self.position is None or self.position.side != 'long':
                    # Close short if exists
                    if self.position and self.position.side == 'short':
                        exit_pnl = (self.position.entry_price - current_price) / self.position.entry_price
                        exit_pnl *= self.position.size
                        period_pnl += exit_pnl - TAKER_FEE
                        if exit_pnl > TAKER_FEE:
                            self.wins += 1
                        self.trades += 1

                    # Open long
                    self.position = Position('long', 1.0, current_price, i)
                    period_pnl -= TAKER_FEE
                    self.trades += 1

            elif momentum < -self.threshold:
                # Want to be short
                if self.position is None or self.position.side != 'short':
                    # Close long if exists
                    if self.position and self.position.side == 'long':
                        exit_pnl = (current_price - self.position.entry_price) / self.position.entry_price
                        exit_pnl *= self.position.size
                        period_pnl += exit_pnl - TAKER_FEE
                        if exit_pnl > TAKER_FEE:
                            self.wins += 1
                        self.trades += 1

                    # Open short
                    self.position = Position('short', 1.0, current_price, i)
                    period_pnl -= TAKER_FEE
                    self.trades += 1

            self.pnl += period_pnl
            returns_list.append(period_pnl)

        # Close final position
        if self.position and len(prices) > 0:
            final_price = prices[-1]
            if self.position.side == 'long':
                exit_pnl = (final_price - self.position.entry_price) / self.position.entry_price
            else:
                exit_pnl = (self.position.entry_price - final_price) / self.position.entry_price
            self.pnl += exit_pnl - TAKER_FEE
            self.trades += 1

        # Compute Sharpe
        returns = np.array(returns_list) if returns_list else np.array([0])
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(len(returns))

        return StrategyResult(
            pnl=self.pnl,
            n_trades=self.trades,
            win_rate=self.wins / max(1, self.trades),
            sharpe=sharpe
        )


def label_regimes(
    df: pd.DataFrame,
    window_size: int = 500,  # ~50 seconds at 100ms
    stride: int = 100,       # ~10 seconds
) -> pd.DataFrame:
    """
    Label each window as MR, TF, or NA based on strategy performance.

    Returns DataFrame with regime labels aligned to original data.
    """
    asmm = ASMMSimulator()
    trend = TrendFollowSimulator()

    labels = []
    indices = []
    details = []

    print(f"Labeling regimes for {len(df)} samples...")

    for start_idx in tqdm(range(0, len(df) - window_size, stride)):
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:end_idx]

        # Simulate both strategies
        asmm_result = asmm.simulate_window(window)
        trend_result = trend.simulate_window(window)

        # Determine regime
        asmm_better = asmm_result.pnl > trend_result.pnl
        trend_better = trend_result.pnl > asmm_result.pnl

        # Thresholds for "meaningful" outperformance
        min_edge = 0.001  # 0.1% minimum edge

        if asmm_better and asmm_result.pnl > min_edge:
            regime = 'MR'  # Mean-reversion (ASMM works)
        elif trend_better and trend_result.pnl > min_edge:
            regime = 'TF'  # Trend-following works
        else:
            regime = 'NA'  # Neither works well

        # Get average entropy for this window
        avg_entropy = window['ent_permutation_16'].mean()
        avg_imbalance_persist = window['imbalance_persistence'].mean()
        avg_aggressor_mom = window['flow_aggressor_momentum'].mean()

        labels.append(regime)
        indices.append(start_idx + window_size // 2)  # Center of window
        details.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'asmm_pnl': asmm_result.pnl,
            'trend_pnl': trend_result.pnl,
            'asmm_sharpe': asmm_result.sharpe,
            'trend_sharpe': trend_result.sharpe,
            'avg_entropy': avg_entropy,
            'avg_imbalance_persist': avg_imbalance_persist,
            'avg_aggressor_mom': avg_aggressor_mom,
        })

    # Create labels DataFrame
    labels_df = pd.DataFrame(details)

    # Print summary
    regime_counts = labels_df['regime'].value_counts()
    print("\n" + "=" * 50)
    print("REGIME LABELING SUMMARY")
    print("=" * 50)
    print(f"Total windows: {len(labels_df)}")
    for regime, count in regime_counts.items():
        pct = count / len(labels_df) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    # Entropy statistics by regime
    print("\nEntropy by Regime:")
    for regime in ['MR', 'TF', 'NA']:
        regime_data = labels_df[labels_df['regime'] == regime]
        if len(regime_data) > 0:
            avg_ent = regime_data['avg_entropy'].mean()
            std_ent = regime_data['avg_entropy'].std()
            print(f"  {regime}: entropy = {avg_ent:.3f} +/- {std_ent:.3f}")

    return labels_df


def main():
    parser = argparse.ArgumentParser(description='Label regimes for NAT validation')
    parser.add_argument('--input', type=str, default=None,
                       help='Input parquet file(s). If not specified, uses latest in data/features/')
    parser.add_argument('--window', type=int, default=500,
                       help='Window size for regime labeling (default: 500)')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride between windows (default: 100)')
    args = parser.parse_args()

    print("=" * 60)
    print("NAT Week 1-2 Validation - Regime Labeling")
    print("=" * 60)

    # Find input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_files = sorted(FEATURES_DIR.glob('*.parquet'))

    if not input_files:
        print("ERROR: No parquet files found!")
        print(f"Please run 'python data_collector.py' first or specify --input")
        return

    print(f"Input files: {[f.name for f in input_files]}")

    # Process each file
    all_labels = []

    for input_file in input_files:
        print(f"\nProcessing {input_file.name}...")
        df = pd.read_parquet(input_file)
        print(f"  Loaded {len(df)} rows")

        labels_df = label_regimes(df, window_size=args.window, stride=args.stride)
        labels_df['source_file'] = input_file.name

        all_labels.append(labels_df)

    # Combine all labels
    combined_labels = pd.concat(all_labels, ignore_index=True)

    # Save labels
    output_path = RESULTS_DIR / 'regime_labels.parquet'
    combined_labels.to_parquet(output_path, index=False)
    print(f"\nSaved labels to {output_path}")

    # Also save CSV for inspection
    csv_path = RESULTS_DIR / 'regime_labels.csv'
    combined_labels.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    print("\nNext step: Run 'python validate_hypothesis.py' to compute mutual information")


if __name__ == '__main__':
    main()

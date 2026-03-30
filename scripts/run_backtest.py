#!/usr/bin/env python3
"""
NAT Backtester - Main CLI Runner

Usage:
    python scripts/run_backtest.py --symbol BTC --strategy accumulation_long
    python scripts/run_backtest.py --symbol BTC --strategy accumulation_long --walk-forward
    python scripts/run_backtest.py --list-strategies
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import load_features, validate_features_for_strategy
from backtest.strategy import get_strategy, get_all_strategies
from backtest.costs import CostModel, hyperliquid_taker, conservative
from backtest.engine import run_backtest
from backtest.walk_forward import walk_forward_validation
from backtest.ml_strategy import (
    create_ml_strategy,
    create_ml_quantile_strategy,
    join_predictions_with_features,
)


def main():
    parser = argparse.ArgumentParser(
        description="NAT Backtester - Validate trading strategies on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simple backtest
  python scripts/run_backtest.py --symbol BTC --strategy whale_flow_simple

  # Run walk-forward validation (recommended)
  python scripts/run_backtest.py --symbol BTC --strategy accumulation_long --walk-forward

  # Use conservative costs
  python scripts/run_backtest.py --symbol BTC --strategy accumulation_long --cost-model conservative

  # Backtest ML model predictions
  python scripts/run_backtest.py --symbol BTC --ml-predictions predictions.parquet --walk-forward

  # ML strategy with custom thresholds
  python scripts/run_backtest.py --symbol BTC --ml-predictions predictions.parquet \\
      --ml-entry-threshold 0.002 --ml-exit-threshold 0.0

  # ML strategy using quantiles (top 25%)
  python scripts/run_backtest.py --symbol BTC --ml-predictions predictions.parquet \\
      --ml-quantile --ml-entry-threshold 0.75 --ml-exit-threshold 0.50

  # List available strategies
  python scripts/run_backtest.py --list-strategies
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing Parquet feature files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Symbol to backtest (e.g., BTC, ETH)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="whale_flow_simple",
        help="Strategy name to test",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation instead of simple backtest",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of folds for walk-forward validation",
    )
    parser.add_argument(
        "--cost-model",
        type=str,
        choices=["taker", "conservative", "zero"],
        default="taker",
        help="Cost model to use",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    # ML strategy arguments
    parser.add_argument(
        "--ml-predictions",
        type=Path,
        help="Path to ML predictions Parquet file (enables ML strategy mode)",
    )
    parser.add_argument(
        "--ml-entry-threshold",
        type=float,
        default=0.001,
        help="Entry threshold for ML predictions (default: 0.001 = 0.1%% return)",
    )
    parser.add_argument(
        "--ml-exit-threshold",
        type=float,
        default=0.0,
        help="Exit threshold for ML predictions (default: 0.0)",
    )
    parser.add_argument(
        "--ml-quantile",
        action="store_true",
        help="Use quantile-based thresholds instead of absolute values",
    )
    parser.add_argument(
        "--ml-direction",
        type=str,
        choices=["long", "short"],
        default="long",
        help="Trade direction for ML strategy (default: long)",
    )

    args = parser.parse_args()

    # List strategies
    if args.list_strategies:
        print("\nAvailable Strategies:")
        print("=" * 60)
        for name, strategy in get_all_strategies().items():
            print(f"\n{name}:")
            print(f"  Direction: {strategy.direction}")
            print(f"  Stop Loss: {strategy.stop_loss_pct}%")
            print(f"  Take Profit: {strategy.take_profit_pct}%")
            print(f"  Max Hold: {strategy.max_holding_bars} bars")
            print(f"  Required Features: {', '.join(strategy.required_features)}")
            if strategy.description:
                print(f"  Description: {strategy.description}")
        return

    # Select cost model
    if args.cost_model == "taker":
        cost_model = hyperliquid_taker()
    elif args.cost_model == "conservative":
        cost_model = conservative()
    else:
        cost_model = CostModel(fee_bps=0, slippage_bps=0)

    print(f"Cost Model: {cost_model}")

    # Load data
    print(f"\nLoading data for {args.symbol} from {args.data_dir}...")
    try:
        dataset = load_features(args.data_dir, args.symbol)
        print(f"Loaded {dataset.n_rows:,} rows")
        print(f"Time range: {dataset.start_time} to {dataset.end_time}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("\nMake sure you have collected feature data by running the ingestor:")
        print("  make run")
        return 1

    # Determine if using ML strategy or rule-based strategy
    if args.ml_predictions:
        # ML Strategy Mode
        print(f"\n{'='*70}")
        print("ML STRATEGY MODE")
        print(f"{'='*70}")

        if not args.ml_predictions.exists():
            print(f"Error: Predictions file not found: {args.ml_predictions}")
            return 1

        # Create ML strategy
        try:
            if args.ml_quantile:
                print(f"\nCreating quantile-based ML strategy...")
                strategy, predictions = create_ml_quantile_strategy(
                    predictions_path=args.ml_predictions,
                    entry_quantile=args.ml_entry_threshold,
                    exit_quantile=args.ml_exit_threshold,
                    direction=args.ml_direction,
                )
            else:
                print(f"\nCreating threshold-based ML strategy...")
                strategy, predictions = create_ml_strategy(
                    predictions_path=args.ml_predictions,
                    entry_threshold=args.ml_entry_threshold,
                    exit_threshold=args.ml_exit_threshold,
                    direction=args.ml_direction,
                )

            print(f"\nStrategy: {strategy.name}")
            print(f"Description: {strategy.description}")

        except Exception as e:
            print(f"Error creating ML strategy: {e}")
            return 1

        # Join predictions with features
        print(f"\nJoining predictions with feature data...")
        dataset.df = join_predictions_with_features(
            dataset.df,
            predictions,
            timestamp_col="timestamp",
        )

        # Check if we have predictions
        n_with_predictions = dataset.df.filter(
            dataset.df["prediction"].is_not_nan()
        ).height
        if n_with_predictions == 0:
            print("\nError: No matching timestamps between predictions and features")
            print("  Predictions time range and features time range must overlap")
            return 1

        print(f"  Ready for backtest with {n_with_predictions} predictions")

    else:
        # Rule-based Strategy Mode
        print(f"\n{'='*70}")
        print("RULE-BASED STRATEGY MODE")
        print(f"{'='*70}")

        try:
            strategy = get_strategy(args.strategy)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        # Validate features exist
        missing = validate_features_for_strategy(dataset, strategy.required_features)
        if missing:
            print(f"\nWarning: Missing features for strategy {strategy.name}:")
            for f in missing:
                print(f"  - {f}")
            print("\nAvailable features:")
            for f in sorted(dataset.feature_columns)[:20]:
                print(f"  - {f}")
            if len(dataset.feature_columns) > 20:
                print(f"  ... and {len(dataset.feature_columns) - 20} more")

            print("\nCannot run backtest without required features.")
            return 1

    # Run backtest
    if args.walk_forward:
        print(f"\nRunning walk-forward validation with {args.n_folds} folds...")
        result = walk_forward_validation(
            dataset.df,
            strategy,
            cost_model,
            n_splits=args.n_folds,
        )
        print(result.summary())

        # Final verdict
        if result.is_valid:
            print("\n[PASS] Strategy passes walk-forward validation")
            print("       Consider paper trading before live deployment")
        else:
            print("\n[FAIL] Strategy fails walk-forward validation")
            print("       DO NOT deploy - likely overfit or no edge")

    else:
        print(f"\nRunning full backtest on {dataset.n_rows:,} bars...")
        result = run_backtest(dataset.df, strategy, cost_model)
        print(result.summary())

        # Exit reason breakdown
        if result.total_trades > 0:
            print("\nExit Reasons:")
            for reason, count in result.exit_reason_breakdown().items():
                pct = count / result.total_trades * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")

        # Warnings
        if result.sharpe_ratio > 3.0:
            print("\n[WARNING] Sharpe > 3.0 is suspicious - possible overfitting")
        if result.total_trades < 30:
            print("\n[WARNING] Too few trades for statistical significance")
        if result.max_drawdown_pct > 20:
            print("\n[WARNING] Max drawdown > 20% - high risk")

        print("\n[NOTE] Run with --walk-forward for proper validation")


if __name__ == "__main__":
    sys.exit(main() or 0)

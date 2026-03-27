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

    # Get strategy
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

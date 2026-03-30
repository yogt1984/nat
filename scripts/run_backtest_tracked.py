#!/usr/bin/env python3
"""
Backtest Runner with Automatic Experiment Tracking

Runs backtest and automatically registers results in experiment tracking.

Usage:
    python scripts/run_backtest_tracked.py \\
        --ml-predictions ./predictions.parquet \\
        --ml-entry-threshold 0.001 \\
        --ml-exit-threshold 0.0 \\
        --walk-forward
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest.data_loader import load_features
from backtest.costs import CostModel, hyperliquid_taker, conservative
from backtest.engine import run_backtest
from backtest.walk_forward import walk_forward_validation
from backtest.ml_strategy import create_ml_strategy, join_predictions_with_features
from experiment_tracking import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Run backtest with tracking")

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
        "--ml-predictions",
        type=Path,
        required=True,
        help="Path to ML predictions Parquet file",
    )
    parser.add_argument(
        "--ml-entry-threshold",
        type=float,
        default=0.001,
        help="Entry threshold for ML predictions",
    )
    parser.add_argument(
        "--ml-exit-threshold",
        type=float,
        default=0.0,
        help="Exit threshold for ML predictions",
    )
    parser.add_argument(
        "--ml-direction",
        type=str,
        choices=["long", "short"],
        default="long",
        help="Trade direction",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation",
    )
    parser.add_argument(
        "--cost-model",
        type=str,
        choices=["taker", "conservative", "zero"],
        default="taker",
        help="Cost model to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for backtest results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ML BACKTEST WITH TRACKING")
    print("=" * 70)

    # Select cost model
    if args.cost_model == "taker":
        cost_model = hyperliquid_taker()
    elif args.cost_model == "conservative":
        cost_model = conservative()
    else:
        cost_model = CostModel(fee_bps=0, slippage_bps=0)

    # Load data
    print(f"\nLoading data for {args.symbol} from {args.data_dir}...")
    try:
        dataset = load_features(args.data_dir, args.symbol)
        print(f"Loaded {dataset.n_rows:,} rows")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return 1

    # Create ML strategy
    print(f"\nCreating ML strategy...")
    strategy, predictions = create_ml_strategy(
        predictions_path=args.ml_predictions,
        entry_threshold=args.ml_entry_threshold,
        exit_threshold=args.ml_exit_threshold,
        direction=args.ml_direction,
    )

    print(f"Strategy: {strategy.name}")

    # Join predictions with features
    print(f"\nJoining predictions with features...")
    dataset.df = join_predictions_with_features(
        dataset.df,
        predictions,
        timestamp_col="timestamp",
    )

    # Run backtest
    if args.walk_forward:
        print(f"\nRunning walk-forward validation...")
        result = walk_forward_validation(
            dataset.df,
            strategy,
            cost_model,
            n_splits=4,
        )
        print(result.summary())

        # Convert to dict for JSON serialization
        backtest_results = {
            "validation_type": "walk_forward",
            "n_folds": 4,
            "average_oos_sharpe": result.average_oos_sharpe,
            "average_oos_is_ratio": result.average_oos_is_ratio,
            "is_valid": result.is_valid,
            "validation_message": result.validation_message,
            "fold_results": [
                {
                    "fold": i + 1,
                    "in_sample_sharpe": fold.in_sample_sharpe,
                    "out_of_sample_sharpe": fold.out_of_sample_sharpe,
                    "oos_is_ratio": fold.oos_is_ratio,
                }
                for i, fold in enumerate(result.fold_results)
            ],
        }

        # Add aggregated metrics
        if result.is_valid and result.fold_results:
            # Use last fold as representative
            last_fold = result.fold_results[-1]
            backtest_results.update({
                "sharpe_ratio": result.average_oos_sharpe,
                "total_return_pct": getattr(last_fold, "total_return_pct", None),
                "max_drawdown_pct": getattr(last_fold, "max_drawdown_pct", None),
                "win_rate": getattr(last_fold, "win_rate", None),
                "total_trades": getattr(last_fold, "total_trades", None),
            })

    else:
        print(f"\nRunning full backtest...")
        result = run_backtest(dataset.df, strategy, cost_model)
        print(result.summary())

        # Convert to dict
        backtest_results = {
            "validation_type": "simple",
            "sharpe_ratio": result.sharpe_ratio,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "avg_trade_pnl_pct": result.avg_trade_pnl_pct,
            "total_costs_pct": result.total_costs_pct,
        }

    # Add metadata
    backtest_results["metadata"] = {
        "symbol": args.symbol,
        "predictions_path": str(args.ml_predictions),
        "strategy_name": strategy.name,
        "entry_threshold": args.ml_entry_threshold,
        "exit_threshold": args.ml_exit_threshold,
        "direction": args.ml_direction,
        "cost_model": args.cost_model,
        "backtested_at": datetime.now().isoformat(),
    }

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(backtest_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Register in experiment tracking
    try:
        tracker = ExperimentTracker()
        experiment_id = tracker.register_backtest(
            predictions_path=args.ml_predictions,
            backtest_results=backtest_results,
            strategy_params={
                "entry_threshold": args.ml_entry_threshold,
                "exit_threshold": args.ml_exit_threshold,
                "direction": args.ml_direction,
            },
        )
        print(f"📊 Backtest tracked: {experiment_id}")
    except Exception as e:
        print(f"Warning: Failed to track backtest: {e}")

    print()

    # Final verdict for walk-forward
    if args.walk_forward:
        if result.is_valid:
            print("[PASS] Strategy passes walk-forward validation")
        else:
            print("[FAIL] Strategy fails walk-forward validation")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

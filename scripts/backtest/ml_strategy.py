"""
ML Strategy Module

Integrates trained ML models with the backtesting framework.
Loads model predictions and generates trading signals based on forecast values.

Usage:
    from backtest.ml_strategy import create_ml_strategy

    strategy = create_ml_strategy(
        predictions_path="./predictions.parquet",
        entry_threshold=0.001,  # Enter when prediction > 0.1% return
        exit_threshold=0.0,     # Exit when prediction < 0%
        confidence_threshold=None,  # Optional confidence filtering
    )
"""

import polars as pl
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

from .strategy import Strategy


@dataclass
class MLPredictions:
    """Container for ML model predictions."""
    df: pl.DataFrame  # Must have: timestamp, prediction, model_name
    model_name: str
    n_predictions: int
    prediction_stats: dict

    def __repr__(self) -> str:
        return (
            f"MLPredictions(model={self.model_name}, "
            f"n={self.n_predictions}, "
            f"mean={self.prediction_stats['mean']:.6f}, "
            f"std={self.prediction_stats['std']:.6f})"
        )


def load_predictions(
    predictions_path: Path,
    timestamp_col: str = "timestamp",
) -> MLPredictions:
    """
    Load ML predictions from Parquet file.

    Args:
        predictions_path: Path to predictions Parquet file
        timestamp_col: Name of timestamp column

    Returns:
        MLPredictions object

    Raises:
        ValueError: If required columns missing
        FileNotFoundError: If file doesn't exist
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    print(f"Loading predictions from {predictions_path}...")
    df = pl.read_parquet(predictions_path)

    # Validate required columns
    required_cols = {timestamp_col, "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {df.columns}"
        )

    # Get model name if available
    model_name = "unknown"
    if "model_name" in df.columns:
        model_name = df["model_name"][0]

    # Filter out NaN predictions
    df_valid = df.filter(pl.col("prediction").is_not_nan())
    n_filtered = len(df) - len(df_valid)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} NaN predictions")

    # Compute statistics
    prediction_stats = {
        "mean": float(df_valid["prediction"].mean()),
        "std": float(df_valid["prediction"].std()),
        "min": float(df_valid["prediction"].min()),
        "max": float(df_valid["prediction"].max()),
        "q25": float(df_valid["prediction"].quantile(0.25)),
        "q50": float(df_valid["prediction"].quantile(0.50)),
        "q75": float(df_valid["prediction"].quantile(0.75)),
    }

    print(f"  Model: {model_name}")
    print(f"  Predictions: {len(df_valid)}")
    print(f"  Range: [{prediction_stats['min']:.6f}, {prediction_stats['max']:.6f}]")
    print(f"  Mean: {prediction_stats['mean']:.6f} ± {prediction_stats['std']:.6f}")

    return MLPredictions(
        df=df_valid,
        model_name=model_name,
        n_predictions=len(df_valid),
        prediction_stats=prediction_stats,
    )


def join_predictions_with_features(
    features_df: pl.DataFrame,
    predictions: MLPredictions,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """
    Join predictions with feature DataFrame.

    Args:
        features_df: DataFrame with features and prices
        predictions: MLPredictions object
        timestamp_col: Name of timestamp column for joining

    Returns:
        DataFrame with predictions joined
    """
    # Ensure timestamp columns have same name
    pred_df = predictions.df.rename({timestamp_col: "timestamp"})

    # Join on timestamp
    joined = features_df.join(
        pred_df.select(["timestamp", "prediction"]),
        on="timestamp",
        how="left"
    )

    # Count successful joins
    n_matched = joined.filter(pl.col("prediction").is_not_nan()).height
    print(f"  Matched {n_matched}/{len(features_df)} timestamps ({n_matched/len(features_df)*100:.1f}%)")

    return joined


def create_ml_strategy(
    predictions_path: Path,
    entry_threshold: float = 0.001,
    exit_threshold: float = 0.0,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0,
    max_holding_bars: int = 600,
    direction: Literal["long", "short"] = "long",
    confidence_threshold: Optional[float] = None,
    name_suffix: str = "",
) -> tuple[Strategy, MLPredictions]:
    """
    Create an ML-based trading strategy.

    Args:
        predictions_path: Path to predictions Parquet file
        entry_threshold: Enter when prediction > this value (long) or < this (short)
        exit_threshold: Exit when prediction crosses this threshold
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        max_holding_bars: Maximum holding time in bars
        direction: "long" or "short"
        confidence_threshold: Optional minimum prediction magnitude for entry
        name_suffix: Optional suffix for strategy name

    Returns:
        (Strategy, MLPredictions) tuple

    Example:
        strategy, preds = create_ml_strategy(
            predictions_path=Path("./predictions.parquet"),
            entry_threshold=0.002,  # Enter on 0.2% predicted return
            exit_threshold=0.0,     # Exit when prediction turns negative
            direction="long"
        )
    """
    # Load predictions
    predictions = load_predictions(predictions_path)

    # Adjust thresholds based on direction
    if direction == "short":
        # For shorts, invert thresholds
        entry_threshold = -abs(entry_threshold)
        exit_threshold = -abs(exit_threshold) if exit_threshold != 0 else 0

    # Define entry condition
    def entry_condition(df: pl.DataFrame) -> pl.Series:
        """Enter when prediction exceeds threshold."""
        if "prediction" not in df.columns:
            # No predictions available, return all False
            return pl.Series("entry", [False] * len(df))

        # Filter out NaN predictions explicitly
        valid_mask = df["prediction"].is_not_nan()

        if direction == "long":
            # Long: Enter when prediction is bullish
            condition = (df["prediction"] > entry_threshold) & valid_mask

            # Optional: Require minimum confidence (absolute prediction value)
            if confidence_threshold is not None:
                condition = condition & (df["prediction"].abs() > confidence_threshold)
        else:
            # Short: Enter when prediction is bearish
            condition = (df["prediction"] < entry_threshold) & valid_mask

            if confidence_threshold is not None:
                condition = condition & (df["prediction"].abs() > confidence_threshold)

        return condition.fill_null(False)

    # Define exit condition
    def exit_condition(df: pl.DataFrame) -> pl.Series:
        """Exit when prediction crosses exit threshold or reverses."""
        if "prediction" not in df.columns:
            return pl.Series("exit", [False] * len(df))

        # Filter out NaN predictions explicitly
        valid_mask = df["prediction"].is_not_nan()

        if direction == "long":
            # Long exit: prediction drops below threshold
            condition = (df["prediction"] < exit_threshold) & valid_mask
        else:
            # Short exit: prediction rises above threshold
            condition = (df["prediction"] > exit_threshold) & valid_mask

        return condition.fill_null(False)

    # Create strategy name
    strategy_name = f"ml_{direction}"
    if name_suffix:
        strategy_name += f"_{name_suffix}"

    # Build description
    description = (
        f"ML-based {direction} strategy using {predictions.model_name}. "
        f"Entry: prediction {'>' if direction == 'long' else '<'} {entry_threshold:.4f}, "
        f"Exit: prediction {'<' if direction == 'long' else '>'} {exit_threshold:.4f}"
    )

    if confidence_threshold:
        description += f", Confidence: |pred| > {confidence_threshold:.4f}"

    strategy = Strategy(
        name=strategy_name,
        entry_condition=entry_condition,
        exit_condition=exit_condition,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_bars=max_holding_bars,
        direction=direction,
        required_features=["prediction"],  # Only needs prediction column
        description=description,
    )

    return strategy, predictions


def create_ml_quantile_strategy(
    predictions_path: Path,
    entry_quantile: float = 0.75,
    exit_quantile: float = 0.50,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0,
    max_holding_bars: int = 600,
    direction: Literal["long", "short"] = "long",
    name_suffix: str = "",
) -> tuple[Strategy, MLPredictions]:
    """
    Create ML strategy using quantile thresholds instead of absolute values.

    This is useful when you don't know appropriate absolute thresholds,
    or when predictions have varying scales across models.

    Args:
        predictions_path: Path to predictions Parquet file
        entry_quantile: Enter when prediction > this quantile (long) or < this (short)
        exit_quantile: Exit when prediction crosses this quantile
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        max_holding_bars: Maximum holding time in bars
        direction: "long" or "short"
        name_suffix: Optional suffix for strategy name

    Returns:
        (Strategy, MLPredictions) tuple

    Example:
        # Enter on top 25% predictions, exit when drops to median
        strategy, preds = create_ml_quantile_strategy(
            predictions_path=Path("./predictions.parquet"),
            entry_quantile=0.75,
            exit_quantile=0.50,
            direction="long"
        )
    """
    # Load predictions
    predictions = load_predictions(predictions_path)

    # Compute quantile thresholds
    if direction == "long":
        entry_threshold = predictions.prediction_stats[f"q{int(entry_quantile*100)}"]
        exit_threshold = predictions.prediction_stats[f"q{int(exit_quantile*100)}"]
    else:
        # For shorts, use inverted quantiles
        entry_threshold = predictions.prediction_stats[f"q{int((1-entry_quantile)*100)}"]
        exit_threshold = predictions.prediction_stats[f"q{int((1-exit_quantile)*100)}"]

    print(f"\n  Quantile Strategy Thresholds:")
    print(f"    Entry: {entry_quantile:.2f} quantile = {entry_threshold:.6f}")
    print(f"    Exit:  {exit_quantile:.2f} quantile = {exit_threshold:.6f}")

    # Use the absolute threshold strategy with computed thresholds
    strategy, _ = create_ml_strategy(
        predictions_path=predictions_path,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_bars=max_holding_bars,
        direction=direction,
        confidence_threshold=None,
        name_suffix=f"q{int(entry_quantile*100)}" + (f"_{name_suffix}" if name_suffix else ""),
    )

    # Update description
    strategy.description = (
        f"ML-based {direction} strategy using {predictions.model_name} (quantile). "
        f"Entry: prediction > {entry_quantile:.0%} quantile ({entry_threshold:.6f}), "
        f"Exit: prediction < {exit_quantile:.0%} quantile ({exit_threshold:.6f})"
    )

    return strategy, predictions


def get_ml_strategies(predictions_path: Path) -> dict:
    """
    Get a collection of ML strategies with different parameters.

    Args:
        predictions_path: Path to predictions Parquet file

    Returns:
        Dictionary of strategy_name -> (Strategy, MLPredictions)
    """
    strategies = {}

    # Conservative: High threshold, tighter stops
    strategy, preds = create_ml_strategy(
        predictions_path=predictions_path,
        entry_threshold=0.002,  # 0.2% predicted return
        exit_threshold=0.0,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_bars=600,
        direction="long",
        name_suffix="conservative"
    )
    strategies["ml_long_conservative"] = (strategy, preds)

    # Aggressive: Lower threshold, wider stops
    strategy, preds = create_ml_strategy(
        predictions_path=predictions_path,
        entry_threshold=0.0005,  # 0.05% predicted return
        exit_threshold=-0.0002,  # Allow small negative
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
        max_holding_bars=1200,
        direction="long",
        name_suffix="aggressive"
    )
    strategies["ml_long_aggressive"] = (strategy, preds)

    # Quantile-based (top 25%)
    strategy, preds = create_ml_quantile_strategy(
        predictions_path=predictions_path,
        entry_quantile=0.75,
        exit_quantile=0.50,
        stop_loss_pct=2.5,
        take_profit_pct=5.0,
        max_holding_bars=800,
        direction="long",
        name_suffix="top25"
    )
    strategies["ml_long_quantile"] = (strategy, preds)

    return strategies

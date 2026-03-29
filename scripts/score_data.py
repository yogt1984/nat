#!/usr/bin/env python3
"""
Model Scoring Script

Generates predictions on new data using trained models.
Supports both sklearn (Elastic Net) and LightGBM models.

Usage:
    python scripts/score_data.py --model models/elasticnet_*.pkl --data ./data/features
    python scripts/score_data.py --model models/lightgbm_*.txt --data ./data/features --output predictions.parquet
    python scripts/score_data.py --model models/latest.pkl --data ./data/features --evaluate
"""

import argparse
import sys
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import (
    load_sklearn_model,
    load_lightgbm_model,
    get_latest_model,
)


def load_parquet_data(data_dir: Path, hours_back: Optional[int] = None) -> pl.DataFrame:
    """
    Load Parquet feature data.

    Args:
        data_dir: Directory containing Parquet files
        hours_back: Optional - only load last N hours

    Returns:
        Polars DataFrame with features
    """
    print(f"Loading data from {data_dir}...")

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found in {data_dir}")

    print(f"Found {len(files)} Parquet files")

    # Load and concatenate
    dfs = []
    for file in files:
        df = pl.read_parquet(file)
        dfs.append(df)

    df = pl.concat(dfs)

    # Filter by time if requested
    if hours_back is not None:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours_back)
        if "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") >= cutoff)

    print(f"Loaded {len(df)} samples")
    return df


def extract_features(df: pl.DataFrame, feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from DataFrame.

    Args:
        df: Polars DataFrame
        feature_names: List of feature column names

    Returns:
        (features array, valid mask)
    """
    print(f"Extracting {len(feature_names)} features...")

    # Check if all features exist
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print(f"Available features: {df.columns[:10]}... (showing first 10)")
        raise ValueError(f"Missing required features: {missing_features}")

    # Extract features
    X = df.select(feature_names).to_numpy()

    # Create valid mask (rows without NaN)
    valid_mask = ~np.isnan(X).any(axis=1)

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"Warning: {n_invalid} samples contain NaN values and will be filtered")

    X_valid = X[valid_mask]
    print(f"Valid samples: {len(X_valid)}/{len(X)}")

    return X_valid, valid_mask


def compute_forward_returns(
    df: pl.DataFrame,
    horizon: int = 600,
) -> np.ndarray:
    """
    Compute forward returns for evaluation.

    Args:
        df: DataFrame with price data
        horizon: Forward horizon in ticks

    Returns:
        Forward returns array
    """
    if "midprice" not in df.columns:
        return None

    prices = df["midprice"].to_numpy()
    returns = np.zeros(len(prices))

    for i in range(len(prices) - horizon):
        returns[i] = (prices[i + horizon] - prices[i]) / prices[i]

    return returns


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Evaluate predictions against true values.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Remove any NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {}

    metrics = {
        "r2_score": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n_samples": len(y_true),
    }

    return metrics


def score_sklearn_model(
    model_path: Path,
    df: pl.DataFrame,
    evaluate: bool = False,
    horizon: int = 600,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Score data using sklearn model.

    Args:
        model_path: Path to saved model
        df: DataFrame with features
        evaluate: Whether to compute evaluation metrics
        horizon: Forward horizon for evaluation

    Returns:
        (predictions, metadata dict, evaluation metrics dict)
    """
    print(f"\nLoading sklearn model from {model_path}...")
    model, scaler, metadata = load_sklearn_model(model_path)

    print(f"Model: {metadata.model_name}")
    print(f"Trained: {metadata.training_date}")
    print(f"Features: {len(metadata.feature_names)}")

    # Extract features
    X, valid_mask = extract_features(df, metadata.feature_names)

    # Apply scaler if present
    if scaler is not None:
        print("Applying feature scaling...")
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_scaled)

    # Evaluation
    eval_metrics = {}
    if evaluate:
        print("\nEvaluating predictions...")
        y_true = compute_forward_returns(df, horizon)
        if y_true is not None:
            y_true_valid = y_true[valid_mask]
            # Match lengths
            min_len = min(len(y_true_valid), len(predictions))
            eval_metrics = evaluate_predictions(
                y_true_valid[:min_len],
                predictions[:min_len]
            )

            print(f"  R² Score: {eval_metrics.get('r2_score', 0):.4f}")
            print(f"  RMSE: {eval_metrics.get('rmse', 0):.6f}")
            print(f"  MAE: {eval_metrics.get('mae', 0):.6f}")
            print(f"  Correlation: {eval_metrics.get('correlation', 0):.4f}")

    return predictions, metadata.to_dict(), eval_metrics


def score_lightgbm_model(
    model_path: Path,
    df: pl.DataFrame,
    evaluate: bool = False,
    horizon: int = 600,
) -> Tuple[np.ndarray, dict, dict]:
    """
    Score data using LightGBM model.

    Args:
        model_path: Path to saved model
        df: DataFrame with features
        evaluate: Whether to compute evaluation metrics
        horizon: Forward horizon for evaluation

    Returns:
        (predictions, metadata dict, evaluation metrics dict)
    """
    print(f"\nLoading LightGBM model from {model_path}...")
    model, metadata = load_lightgbm_model(model_path)

    print(f"Model: {metadata.model_name}")
    print(f"Trained: {metadata.training_date}")
    print(f"Features: {len(metadata.feature_names)}")

    # Extract features
    X, valid_mask = extract_features(df, metadata.feature_names)

    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X)

    # Evaluation
    eval_metrics = {}
    if evaluate:
        print("\nEvaluating predictions...")
        y_true = compute_forward_returns(df, horizon)
        if y_true is not None:
            y_true_valid = y_true[valid_mask]
            # Match lengths
            min_len = min(len(y_true_valid), len(predictions))
            eval_metrics = evaluate_predictions(
                y_true_valid[:min_len],
                predictions[:min_len]
            )

            print(f"  R² Score: {eval_metrics.get('r2_score', 0):.4f}")
            print(f"  RMSE: {eval_metrics.get('rmse', 0):.6f}")
            print(f"  MAE: {eval_metrics.get('mae', 0):.6f}")
            print(f"  Correlation: {eval_metrics.get('correlation', 0):.4f}")

    return predictions, metadata.to_dict(), eval_metrics


def save_predictions(
    predictions: np.ndarray,
    df: pl.DataFrame,
    valid_mask: np.ndarray,
    output_path: Path,
    metadata: dict,
    eval_metrics: dict,
):
    """
    Save predictions to Parquet file.

    Args:
        predictions: Prediction array
        df: Original DataFrame
        valid_mask: Mask of valid samples
        output_path: Output file path
        metadata: Model metadata
        eval_metrics: Evaluation metrics
    """
    print(f"\nSaving predictions to {output_path}...")

    # Create full predictions array with NaN for invalid samples
    full_predictions = np.full(len(df), np.nan)
    full_predictions[valid_mask] = predictions

    # Create output DataFrame
    output_df = df.select(["timestamp"] if "timestamp" in df.columns else [])
    output_df = output_df.with_columns([
        pl.Series("prediction", full_predictions),
        pl.Series("model_name", [metadata["model_name"]] * len(df)),
    ])

    # Save to Parquet
    output_df.write_parquet(output_path)

    print(f"Saved {len(predictions)} predictions")
    print(f"Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score data using trained model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model file (.pkl or .txt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data/features"),
        help="Directory containing Parquet feature files (default: ./data/features)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Parquet file for predictions (optional)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute evaluation metrics (requires forward returns)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
        help="Forward horizon in ticks for evaluation (default: 600)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        help="Only score last N hours of data (optional)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL SCORING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    if args.evaluate:
        print(f"Evaluation: Enabled (horizon={args.horizon} ticks)")
    print()

    # Check model file exists
    if not args.model.exists():
        # Try to find it as a pattern
        model_dir = args.model.parent
        model_pattern = args.model.name

        if model_pattern == "latest.pkl" or model_pattern == "latest.txt":
            print("Looking for latest model...")
            model_type = "elasticnet" if model_pattern.endswith(".pkl") else "lightgbm"
            args.model = get_latest_model(model_dir, model_type)
            if args.model is None:
                print(f"Error: No models found in {model_dir}")
                return
            print(f"Using latest model: {args.model}")
        else:
            print(f"Error: Model file not found: {args.model}")
            print()
            print("Available models:")
            from utils.model_io import list_models
            models = list_models(model_dir)
            for m in models:
                print(f"  - {m['model_file']}")
            return

    # Load data
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        return

    df = load_parquet_data(args.data, args.hours)

    # Determine model type and score
    if args.model.suffix == ".pkl":
        predictions, metadata, eval_metrics = score_sklearn_model(
            args.model, df, args.evaluate, args.horizon
        )
        # Get valid mask for saving
        _, valid_mask = extract_features(df, metadata["feature_names"])
    elif args.model.suffix == ".txt":
        predictions, metadata, eval_metrics = score_lightgbm_model(
            args.model, df, args.evaluate, args.horizon
        )
        # Get valid mask for saving
        _, valid_mask = extract_features(df, metadata["feature_names"])
    else:
        print(f"Error: Unknown model format: {args.model.suffix}")
        print("Supported formats: .pkl (sklearn), .txt (LightGBM)")
        return

    # Save predictions if output specified
    if args.output:
        save_predictions(predictions, df, valid_mask, args.output, metadata, eval_metrics)

    print()
    print("=" * 70)
    print("SCORING COMPLETE")
    print("=" * 70)
    print(f"Generated {len(predictions)} predictions")
    if eval_metrics:
        print(f"R² Score: {eval_metrics.get('r2_score', 0):.4f}")
        print(f"Correlation: {eval_metrics.get('correlation', 0):.4f}")
    print()


if __name__ == "__main__":
    main()

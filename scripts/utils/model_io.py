"""
Model I/O Utilities

Handles saving and loading of trained models with metadata.
Supports scikit-learn models (Elastic Net) and LightGBM models.
"""

import json
import joblib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ModelMetadata:
    """Metadata for a trained model."""

    def __init__(
        self,
        model_type: str,
        model_name: str,
        feature_names: list,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_date: str,
        snapshot_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.feature_names = feature_names
        self.hyperparameters = hyperparameters
        self.performance_metrics = performance_metrics
        self.training_date = training_date
        self.snapshot_name = snapshot_name
        self.experiment_id = experiment_id
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "hyperparameters": self.hyperparameters,
            "performance_metrics": self.performance_metrics,
            "training_date": self.training_date,
            "snapshot_name": self.snapshot_name,
            "experiment_id": self.experiment_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Load from dictionary."""
        return cls(
            model_type=data["model_type"],
            model_name=data["model_name"],
            feature_names=data["feature_names"],
            hyperparameters=data["hyperparameters"],
            performance_metrics=data["performance_metrics"],
            training_date=data["training_date"],
            snapshot_name=data.get("snapshot_name"),
            experiment_id=data.get("experiment_id"),
            notes=data.get("notes"),
        )


def save_sklearn_model(
    model: Any,
    scaler: Optional[Any],
    metadata: ModelMetadata,
    output_dir: Path,
    model_filename: Optional[str] = None,
) -> Path:
    """
    Save scikit-learn model with metadata.

    Args:
        model: Trained sklearn model
        scaler: Optional fitted scaler
        metadata: Model metadata
        output_dir: Directory to save model
        model_filename: Optional custom filename (otherwise auto-generated)

    Returns:
        Path to saved model file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if model_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{metadata.model_name}_{timestamp}.pkl"

    model_path = output_dir / model_filename
    metadata_path = output_dir / model_filename.replace(".pkl", "_metadata.json")

    # Save model and scaler
    model_data = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata.to_dict(),
    }

    joblib.dump(model_data, model_path, compress=3)

    # Save metadata separately for easy inspection
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    return model_path


def load_sklearn_model(model_path: Path) -> Tuple[Any, Optional[Any], ModelMetadata]:
    """
    Load scikit-learn model with metadata.

    Args:
        model_path: Path to saved model file

    Returns:
        (model, scaler, metadata)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model data
    model_data = joblib.load(model_path)

    model = model_data["model"]
    scaler = model_data.get("scaler")
    metadata = ModelMetadata.from_dict(model_data["metadata"])

    print(f"Loaded model: {metadata.model_name}")
    print(f"Trained: {metadata.training_date}")
    print(f"Features: {len(metadata.feature_names)}")

    return model, scaler, metadata


def save_lightgbm_model(
    model: Any,
    metadata: ModelMetadata,
    output_dir: Path,
    model_filename: Optional[str] = None,
) -> Path:
    """
    Save LightGBM model with metadata.

    Args:
        model: Trained LightGBM model
        metadata: Model metadata
        output_dir: Directory to save model
        model_filename: Optional custom filename (otherwise auto-generated)

    Returns:
        Path to saved model file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if model_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{metadata.model_name}_{timestamp}.txt"

    model_path = output_dir / model_filename
    metadata_path = output_dir / model_filename.replace(".txt", "_metadata.json")

    # Save LightGBM model (native format)
    model.save_model(str(model_path))

    # Save metadata separately
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    return model_path


def load_lightgbm_model(model_path: Path) -> Tuple[Any, ModelMetadata]:
    """
    Load LightGBM model with metadata.

    Args:
        model_path: Path to saved model file

    Returns:
        (model, metadata)
    """
    import lightgbm as lgb

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load LightGBM model
    model = lgb.Booster(model_file=str(model_path))

    # Load metadata
    metadata_path = model_path.parent / model_path.name.replace(".txt", "_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = ModelMetadata.from_dict(json.load(f))
    else:
        # Create minimal metadata if not found
        metadata = ModelMetadata(
            model_type="lightgbm",
            model_name="unknown",
            feature_names=[],
            hyperparameters={},
            performance_metrics={},
            training_date="unknown",
        )

    print(f"Loaded model: {metadata.model_name}")
    print(f"Trained: {metadata.training_date}")

    return model, metadata


def list_models(model_dir: Path) -> list:
    """
    List all saved models in directory.

    Args:
        model_dir: Directory containing models

    Returns:
        List of model info dictionaries
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return []

    models = []

    # Find all metadata files
    for metadata_path in model_dir.glob("*_metadata.json"):
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Get model file path
        model_file = metadata_path.name.replace("_metadata.json", ".pkl")
        if not (model_dir / model_file).exists():
            model_file = metadata_path.name.replace("_metadata.json", ".txt")

        models.append({
            "model_file": model_file,
            "metadata_file": metadata_path.name,
            "model_name": metadata.get("model_name", "unknown"),
            "model_type": metadata.get("model_type", "unknown"),
            "training_date": metadata.get("training_date", "unknown"),
            "test_r2": metadata.get("performance_metrics", {}).get("test_r2", None),
        })

    # Sort by training date (newest first)
    models.sort(key=lambda x: x["training_date"], reverse=True)

    return models


def get_latest_model(model_dir: Path, model_type: Optional[str] = None) -> Optional[Path]:
    """
    Get path to most recently trained model.

    Args:
        model_dir: Directory containing models
        model_type: Optional filter by model type (elasticnet, lightgbm)

    Returns:
        Path to latest model file, or None if not found
    """
    models = list_models(model_dir)

    if model_type:
        models = [m for m in models if m["model_type"] == model_type]

    if not models:
        return None

    return model_dir / models[0]["model_file"]

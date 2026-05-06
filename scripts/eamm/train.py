"""
EAMM Module 4: Model Training Pipeline

Trains a LightGBM model to predict optimal spread from the 19-dim context vector.

Two modes:
  A) Classification: predict which spread level (class) is optimal
  B) Regression: predict continuous optimal spread in bps

Reference: EAMM_SPEC.md §1.7
"""

import json
import pickle
import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime


@dataclass
class TrainResult:
    """Result of model training.

    Attributes
    ----------
    model : lgb.LGBMClassifier or lgb.LGBMRegressor
    mode : str ("classification" or "regression")
    feature_names : list of str
    feature_importances : dict of {name: importance}
    train_score : float (accuracy or R^2)
    n_train : int
    n_classes : int (only for classification)
    model_path : str (where saved)
    """
    model: object
    mode: str
    feature_names: list
    feature_importances: dict
    train_score: float
    n_train: int
    n_classes: int
    model_path: Optional[str] = None


def train_eamm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    mode: Literal["classification", "regression"] = "regression",
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    save_dir: Optional[str] = "models",
) -> TrainResult:
    """Train EAMM spread prediction model.

    Parameters
    ----------
    X : np.ndarray, shape (N, 19)
        Context feature matrix.
    y : np.ndarray, shape (N,)
        Labels — class indices for classification, bps values for regression.
    feature_names : list of str
        Feature names (length 19).
    mode : "classification" or "regression"
    n_estimators : int
    max_depth : int
    learning_rate : float
    save_dir : str or None
        Directory to save model. None = don't save.

    Returns
    -------
    TrainResult
    """
    # Clean input
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "classification":
        n_classes = int(np.max(y) + 1)
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=100,
            verbose=-1,
            n_jobs=-1,
            num_class=n_classes if n_classes > 2 else None,
        )
        model.fit(X, y)
        train_score = float(np.mean(model.predict(X) == y))
    else:
        n_classes = 0
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=100,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        train_score = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Feature importance
    importances = dict(
        sorted(
            zip(feature_names, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # Save
    model_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = str(save_dir / f"eamm_{mode}_{date_str}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_names, "mode": mode}, f)

        # Save importance as JSON
        imp_path = str(save_dir / f"eamm_importance_{date_str}.json")
        with open(imp_path, "w") as f:
            json.dump(importances, f, indent=2)

    return TrainResult(
        model=model,
        mode=mode,
        feature_names=feature_names,
        feature_importances=importances,
        train_score=train_score,
        n_train=len(y),
        n_classes=n_classes,
        model_path=model_path,
    )


def load_eamm_model(model_path: str) -> TrainResult:
    """Load a saved EAMM model."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    feature_names = data["feature_names"]
    mode = data["mode"]

    importances = dict(
        zip(feature_names, model.feature_importances_.tolist())
    )

    return TrainResult(
        model=model,
        mode=mode,
        feature_names=feature_names,
        feature_importances=importances,
        train_score=0.0,  # unknown after reload
        n_train=0,
        n_classes=model.n_classes_ if hasattr(model, "n_classes_") else 0,
        model_path=model_path,
    )


def predict_spread(train_result: TrainResult, X: np.ndarray) -> np.ndarray:
    """Predict optimal spread from context features.

    Returns
    -------
    np.ndarray:
        For regression: continuous spread in bps (N,)
        For classification: class probabilities (N, K)
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    model = train_result.model

    if train_result.mode == "regression":
        preds = model.predict(X)
        # Ensure non-negative
        preds = np.maximum(preds, 0.0)
        return preds
    else:
        return model.predict_proba(X)

#!/usr/bin/env python3
"""
Real-Time Model Serving API

Serves trained ML models via REST API with live prediction generation.
Integrates with experiment tracking to serve the best validated models.

Features:
- Load and serve trained models (sklearn, LightGBM)
- Real-time prediction generation
- Best model selection by metric
- Model hot-reloading
- Performance monitoring

Usage:
    # Start server
    python scripts/model_serving.py --port 8000

    # Serve best model
    python scripts/model_serving.py --serve-best --metric sharpe_ratio

    # Development mode with hot-reload
    uvicorn scripts.model_serving:app --reload --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /health              - Health check
    GET  /models              - List available models
    GET  /models/best         - Get best model by metric
    POST /predict             - Single prediction
    POST /predict/batch       - Batch predictions
    POST /reload              - Reload models

Example:
    # Get health status
    curl http://localhost:8000/health

    # List models
    curl http://localhost:8000/models

    # Predict
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}'

    # Get best model
    curl http://localhost:8000/models/best?metric=sharpe_ratio
"""

import argparse
import json
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import (
    load_sklearn_model,
    load_lightgbm_model,
    ModelMetadata,
)
from experiment_tracking import ExperimentTracker

# FastAPI imports
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    # Allow module import for testing even without FastAPI
    # Create dummy base class for models
    class BaseModel:
        pass
    Field = lambda *args, **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f
    FastAPI = HTTPException = Request = None
    JSONResponse = None
    uvicorn = None


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class PredictRequest(BaseModel):
    """Request schema for single prediction."""
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_id: Optional[str] = Field(None, description="Optional: specific model ID to use")

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numeric")
        return v


class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions."""
    features: List[List[float]] = Field(..., description="List of feature vectors")
    model_id: Optional[str] = Field(None, description="Optional: specific model ID to use")

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Each sample must be a list of features")
        # Check all rows have same length
        lengths = [len(row) for row in v]
        if len(set(lengths)) > 1:
            raise ValueError(f"All feature vectors must have same length, got: {set(lengths)}")
        return v


class PredictResponse(BaseModel):
    """Response schema for predictions."""
    prediction: float
    model_id: str
    model_type: str
    timestamp: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[float]
    model_id: str
    model_type: str
    n_predictions: int
    timestamp: str
    latency_ms: float
    avg_latency_per_sample_ms: float


class ModelInfo(BaseModel):
    """Model information schema."""
    model_id: str
    model_path: str
    model_type: str
    model_name: str
    n_features: int
    feature_names: List[str]
    experiment_id: Optional[str]
    snapshot_name: Optional[str]
    performance_metrics: Optional[Dict[str, float]]
    loaded: bool
    last_used: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    models_loaded: int
    total_predictions: int
    api_version: str


# =============================================================================
# Model Cache
# =============================================================================

class ModelCache:
    """Cache for loaded models with hot-reloading support."""

    def __init__(self, cache_size: int = 5):
        self.cache_size = cache_size
        self.models: Dict[str, Tuple[Any, Any, ModelMetadata]] = {}  # model_id -> (model, scaler, metadata)
        self.last_used: Dict[str, datetime] = {}
        self.load_times: Dict[str, float] = {}
        self.prediction_counts: Dict[str, int] = {}

    def get(self, model_path: Path) -> Optional[Tuple[Any, Any, ModelMetadata]]:
        """Get model from cache."""
        model_id = self._get_model_id(model_path)
        if model_id in self.models:
            self.last_used[model_id] = datetime.now()
            self.prediction_counts[model_id] = self.prediction_counts.get(model_id, 0) + 1
            return self.models[model_id]
        return None

    def put(self, model_path: Path, model: Any, scaler: Any, metadata: ModelMetadata, load_time: float):
        """Add model to cache."""
        model_id = self._get_model_id(model_path)

        # Evict oldest if cache full
        if len(self.models) >= self.cache_size and model_id not in self.models:
            oldest_id = min(self.last_used.items(), key=lambda x: x[1])[0]
            del self.models[oldest_id]
            del self.last_used[oldest_id]
            del self.load_times[oldest_id]
            if oldest_id in self.prediction_counts:
                del self.prediction_counts[oldest_id]

        self.models[model_id] = (model, scaler, metadata)
        self.last_used[model_id] = datetime.now()
        self.load_times[model_id] = load_time
        if model_id not in self.prediction_counts:
            self.prediction_counts[model_id] = 0

    def clear(self):
        """Clear all cached models."""
        self.models.clear()
        self.last_used.clear()
        self.load_times.clear()
        self.prediction_counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.models),
            "max_cache_size": self.cache_size,
            "models": list(self.models.keys()),
            "prediction_counts": self.prediction_counts,
            "load_times": self.load_times,
        }

    @staticmethod
    def _get_model_id(model_path: Path) -> str:
        """Generate unique model ID from path."""
        return hashlib.md5(str(model_path).encode()).hexdigest()[:12]


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages model loading and serving."""

    def __init__(self, models_dir: Path = Path("./models"), cache_size: int = 5):
        self.models_dir = Path(models_dir)
        self.cache = ModelCache(cache_size=cache_size)
        self.tracker = ExperimentTracker()
        self.start_time = time.time()
        self.total_predictions = 0

    def load_model(self, model_path: Path, use_cache: bool = True) -> Tuple[Any, Any, ModelMetadata]:
        """
        Load model from disk with caching.

        Args:
            model_path: Path to model file
            use_cache: Whether to use cache

        Returns:
            (model, scaler, metadata)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check cache
        if use_cache:
            cached = self.cache.get(model_path)
            if cached is not None:
                return cached

        # Load model
        start = time.time()
        if model_path.suffix == ".pkl":
            model, scaler, metadata = load_sklearn_model(model_path)
        elif model_path.suffix == ".txt":
            model, metadata = load_lightgbm_model(model_path)
            scaler = None
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        load_time = time.time() - start

        # Cache model
        if use_cache:
            self.cache.put(model_path, model, scaler, metadata, load_time)

        return model, scaler, metadata

    def get_best_model(self, metric: str = "sharpe_ratio", min_trades: int = 30) -> Optional[Path]:
        """
        Get best model from experiment tracking.

        Args:
            metric: Metric to optimize (sharpe_ratio, total_return_pct, win_rate)
            min_trades: Minimum trades for statistical significance

        Returns:
            Path to best model or None
        """
        best_exp = self.tracker.get_best_experiment(metric=metric, min_trades=min_trades)
        if best_exp is None:
            return None

        # Extract model path from experiment
        if "training" in best_exp and "model_path" in best_exp["training"]:
            model_path = Path(best_exp["training"]["model_path"])
            if model_path.exists():
                return model_path

        return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in models directory."""
        models = []

        # Get all model files
        model_files = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.txt"))

        for model_path in sorted(model_files):
            # Load metadata only
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata_dict = json.load(f)

                model_id = self.cache._get_model_id(model_path)
                models.append({
                    "model_id": model_id,
                    "model_path": str(model_path),
                    "model_type": metadata_dict.get("model_type"),
                    "model_name": metadata_dict.get("model_name"),
                    "n_features": metadata_dict.get("n_features"),
                    "feature_names": metadata_dict.get("feature_names"),
                    "experiment_id": metadata_dict.get("experiment_id"),
                    "snapshot_name": metadata_dict.get("snapshot_name"),
                    "performance_metrics": metadata_dict.get("performance_metrics"),
                    "loaded": model_id in self.cache.models,
                    "last_used": self.cache.last_used.get(model_id, None),
                })

        return models

    def predict(self, model_path: Path, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            model_path: Path to model
            features: Feature array (n_samples, n_features)

        Returns:
            Predictions array
        """
        model, scaler, metadata = self.load_model(model_path)

        # Validate feature count
        if features.shape[1] != len(metadata.feature_names):
            raise ValueError(
                f"Feature mismatch: model expects {len(metadata.feature_names)} features, "
                f"got {features.shape[1]}"
            )

        # Apply scaling if available
        if scaler is not None:
            features = scaler.transform(features)

        # Predict
        predictions = model.predict(features)

        # Track
        self.total_predictions += len(predictions)

        return predictions

    def get_health(self) -> Dict[str, Any]:
        """Get service health information."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "models_loaded": len(self.cache.models),
            "total_predictions": self.total_predictions,
            "api_version": "1.0.0",
            "cache_stats": self.cache.get_stats(),
        }


# =============================================================================
# FastAPI Application
# =============================================================================

# Create app if FastAPI is available
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="NAT Model Serving API",
        description="Real-time ML model serving for validated trading models",
        version="1.0.0",
    )
    # Global model manager
    manager = ModelManager()
else:
    app = None
    manager = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and statistics.
    """
    health = manager.get_health()
    return HealthResponse(**health)


@app.get("/models")
async def list_models():
    """
    List all available models.

    Returns:
        List of model information dictionaries
    """
    try:
        models = manager.list_available_models()
        return {
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/models/best")
async def get_best_model(
    metric: str = "sharpe_ratio",
    min_trades: int = 30,
):
    """
    Get best model by metric.

    Args:
        metric: Metric to optimize (sharpe_ratio, total_return_pct, win_rate)
        min_trades: Minimum trades for statistical significance

    Returns:
        Best model information
    """
    try:
        best_model_path = manager.get_best_model(metric=metric, min_trades=min_trades)

        if best_model_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found with metric={metric}, min_trades={min_trades}"
            )

        # Load metadata
        metadata_path = best_model_path.parent / f"{best_model_path.stem}_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Get experiment details
        experiment_id = metadata.get("experiment_id")
        experiment = None
        if experiment_id:
            experiment = manager.tracker.get_experiment(experiment_id)

        return {
            "model_path": str(best_model_path),
            "model_id": manager.cache._get_model_id(best_model_path),
            "metadata": metadata,
            "experiment": experiment,
            "selected_by": {
                "metric": metric,
                "min_trades": min_trades,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting best model: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_single(request: PredictRequest):
    """
    Generate single prediction.

    Args:
        request: Prediction request with features

    Returns:
        Prediction response
    """
    start_time = time.time()

    try:
        # Get model path
        if request.model_id:
            # Use specific model by ID
            models = manager.list_available_models()
            model_info = next((m for m in models if m["model_id"] == request.model_id), None)
            if model_info is None:
                raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")
            model_path = Path(model_info["model_path"])
        else:
            # Use best model
            model_path = manager.get_best_model()
            if model_path is None:
                raise HTTPException(
                    status_code=404,
                    detail="No best model available. Please specify model_id."
                )

        # Prepare features
        features = np.array(request.features).reshape(1, -1)

        # Generate prediction
        predictions = manager.predict(model_path, features)
        prediction = float(predictions[0])

        # Load metadata for response
        _, _, metadata = manager.load_model(model_path)

        latency_ms = (time.time() - start_time) * 1000

        return PredictResponse(
            prediction=prediction,
            model_id=manager.cache._get_model_id(model_path),
            model_type=metadata.model_type,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Generate batch predictions.

    Args:
        request: Batch prediction request

    Returns:
        Batch prediction response
    """
    start_time = time.time()

    try:
        # Get model path
        if request.model_id:
            models = manager.list_available_models()
            model_info = next((m for m in models if m["model_id"] == request.model_id), None)
            if model_info is None:
                raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")
            model_path = Path(model_info["model_path"])
        else:
            model_path = manager.get_best_model()
            if model_path is None:
                raise HTTPException(
                    status_code=404,
                    detail="No best model available. Please specify model_id."
                )

        # Prepare features
        features = np.array(request.features)

        # Generate predictions
        predictions = manager.predict(model_path, features)
        predictions_list = [float(p) for p in predictions]

        # Load metadata
        _, _, metadata = manager.load_model(model_path)

        latency_ms = (time.time() - start_time) * 1000
        avg_latency_per_sample = latency_ms / len(predictions)

        return BatchPredictResponse(
            predictions=predictions_list,
            model_id=manager.cache._get_model_id(model_path),
            model_type=metadata.model_type,
            n_predictions=len(predictions),
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            avg_latency_per_sample_ms=avg_latency_per_sample,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/reload")
async def reload_models():
    """
    Reload all models (clear cache).

    Useful for hot-reloading updated models.
    """
    try:
        old_count = len(manager.cache.models)
        manager.cache.clear()
        return {
            "status": "success",
            "message": f"Cleared {old_count} models from cache",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload error: {str(e)}")


# =============================================================================
# CLI
# =============================================================================

def main():
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed. Install with:")
        print("  pip install fastapi uvicorn")
        print("  or")
        print("  conda install -c conda-forge fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Real-Time Model Serving API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server
  python scripts/model_serving.py --port 8000

  # Serve best model by Sharpe ratio
  python scripts/model_serving.py --serve-best --metric sharpe_ratio

  # Development mode with hot-reload
  uvicorn scripts.model_serving:app --reload --host 0.0.0.0 --port 8000
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing models (default: ./models)"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=5,
        help="Model cache size (default: 5)"
    )
    parser.add_argument(
        "--serve-best",
        action="store_true",
        help="Only serve the best model"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return_pct", "win_rate"],
        help="Metric for best model selection (default: sharpe_ratio)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    args = parser.parse_args()

    # Initialize manager
    global manager
    manager = ModelManager(models_dir=args.models_dir, cache_size=args.cache_size)

    # Pre-load best model if requested
    if args.serve_best:
        print(f"Pre-loading best model by {args.metric}...")
        best_model = manager.get_best_model(metric=args.metric)
        if best_model:
            manager.load_model(best_model)
            print(f"✅ Loaded best model: {best_model}")
        else:
            print(f"⚠️  No best model found by {args.metric}")

    # Start server
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           NAT MODEL SERVING API                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Health: http://{args.host}:{args.port}/health")
    print(f"Docs:   http://{args.host}:{args.port}/docs")
    print(f"")
    print(f"Models directory: {args.models_dir}")
    print(f"Cache size: {args.cache_size}")
    print(f"")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

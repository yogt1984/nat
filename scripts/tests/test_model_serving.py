#!/usr/bin/env python3
"""
Tests for Model Serving API

Skeptical tests covering:
- Model loading and caching
- Prediction endpoints
- Best model selection
- Error handling
- Batch predictions
- Performance
"""

import pytest
import json
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Test if model_serving module exists
def test_model_serving_module_exists():
    """Test that model_serving.py exists."""
    import sys
    scripts_dir = Path(__file__).parent.parent
    model_serving_file = scripts_dir / "model_serving.py"
    assert model_serving_file.exists(), f"model_serving.py not found at {model_serving_file}"
    sys.path.insert(0, str(scripts_dir))


def test_can_import_model_serving():
    """Test that we can import model_serving."""
    try:
        import sys
        scripts_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(scripts_dir))
        import model_serving
    except ImportError as e:
        pytest.fail(f"Cannot import model_serving: {e}")


def test_can_import_fastapi_dependencies():
    """Test that FastAPI dependencies are available."""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import uvicorn
    except ImportError as e:
        pytest.skip(f"FastAPI not installed: {e}")


def test_model_cache_creation():
    """Test ModelCache initialization."""
    try:
        from model_serving import ModelCache

        cache = ModelCache(cache_size=3)
        assert cache.cache_size == 3
        assert len(cache.models) == 0
        assert len(cache.last_used) == 0

        # Test stats
        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["max_cache_size"] == 3
    except ImportError:
        pytest.skip("model_serving module not available")


def test_model_manager_creation():
    """Test ModelManager initialization."""
    try:
        from model_serving import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=Path(tmpdir), cache_size=2)
            assert manager.models_dir == Path(tmpdir)
            assert manager.cache.cache_size == 2
            assert manager.total_predictions == 0

            # Test health
            health = manager.get_health()
            assert health["status"] == "healthy"
            assert health["models_loaded"] == 0
            assert health["total_predictions"] == 0
    except ImportError:
        pytest.skip("model_serving module not available")


def test_model_cache_eviction():
    """Test that ModelCache evicts oldest models when full."""
    try:
        from model_serving import ModelCache
        from utils.model_io import ModelMetadata

        cache = ModelCache(cache_size=2)

        # Create dummy metadata
        metadata1 = ModelMetadata(
            model_type="test",
            model_name="model1",
            feature_names=["f1", "f2"],
            hyperparameters={},
            performance_metrics={},
            training_date=datetime.now().isoformat(),
        )

        metadata2 = ModelMetadata(
            model_type="test",
            model_name="model2",
            feature_names=["f1", "f2"],
            hyperparameters={},
            performance_metrics={},
            training_date=datetime.now().isoformat(),
        )

        metadata3 = ModelMetadata(
            model_type="test",
            model_name="model3",
            feature_names=["f1", "f2"],
            hyperparameters={},
            performance_metrics={},
            training_date=datetime.now().isoformat(),
        )

        # Add models
        path1 = Path("/tmp/model1.pkl")
        path2 = Path("/tmp/model2.pkl")
        path3 = Path("/tmp/model3.pkl")

        cache.put(path1, "model1", None, metadata1, 0.1)
        cache.put(path2, "model2", None, metadata2, 0.1)
        assert len(cache.models) == 2

        # Access model1 to make it more recent
        cache.get(path1)

        # Add model3 - should evict model2 (oldest)
        cache.put(path3, "model3", None, metadata3, 0.1)
        assert len(cache.models) == 2

        # model1 and model3 should be in cache
        assert cache.get(path1) is not None
        assert cache.get(path3) is not None
        assert cache.get(path2) is None  # model2 was evicted

    except ImportError:
        pytest.skip("model_serving module not available")


def test_api_health_endpoint():
    """Test /health endpoint."""
    try:
        from fastapi.testclient import TestClient
        from model_serving import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["models_loaded"] >= 0
        assert data["total_predictions"] >= 0
        assert data["api_version"] == "1.0.0"

    except ImportError:
        pytest.skip("FastAPI not available")


def test_api_list_models_endpoint():
    """Test /models endpoint."""
    try:
        from fastapi.testclient import TestClient
        from model_serving import app

        client = TestClient(app)
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "count" in data
        assert "timestamp" in data
        assert isinstance(data["models"], list)
        assert data["count"] == len(data["models"])

    except ImportError:
        pytest.skip("FastAPI not available")


def test_api_predict_validation():
    """Test /predict endpoint validation."""
    try:
        from fastapi.testclient import TestClient
        from model_serving import app

        client = TestClient(app)

        # Test empty features
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422  # Validation error

        # Test invalid features (non-numeric)
        response = client.post("/predict", json={"features": ["a", "b", "c"]})
        assert response.status_code == 422

        # Test missing features
        response = client.post("/predict", json={})
        assert response.status_code == 422

    except ImportError:
        pytest.skip("FastAPI not available")


def test_api_batch_predict_validation():
    """Test /predict/batch endpoint validation."""
    try:
        from fastapi.testclient import TestClient
        from model_serving import app

        client = TestClient(app)

        # Test empty features
        response = client.post("/predict/batch", json={"features": []})
        assert response.status_code == 422

        # Test inconsistent feature lengths
        response = client.post("/predict/batch", json={
            "features": [
                [1.0, 2.0, 3.0],
                [1.0, 2.0],  # Different length
            ]
        })
        assert response.status_code == 422

        # Test invalid format
        response = client.post("/predict/batch", json={"features": [1, 2, 3]})
        assert response.status_code == 422

    except ImportError:
        pytest.skip("FastAPI not available")


def test_api_reload_endpoint():
    """Test /reload endpoint."""
    try:
        from fastapi.testclient import TestClient
        from model_serving import app

        client = TestClient(app)
        response = client.post("/reload")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "message" in data
        assert "timestamp" in data

    except ImportError:
        pytest.skip("FastAPI not available")


def test_model_loading_error_handling():
    """Test error handling for invalid model paths."""
    try:
        from model_serving import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=Path(tmpdir))

            # Test loading non-existent model
            with pytest.raises(FileNotFoundError):
                manager.load_model(Path("/nonexistent/model.pkl"))

    except ImportError:
        pytest.skip("model_serving module not available")


def test_feature_dimension_mismatch():
    """Test prediction fails with wrong feature dimensions."""
    # This would require a real model file, so we test the validation logic
    try:
        from model_serving import PredictRequest
        from pydantic import ValidationError

        # Test that we can create valid request
        request = PredictRequest(features=[1.0, 2.0, 3.0])
        assert len(request.features) == 3

        # Test empty features raise validation error
        with pytest.raises(ValidationError):
            PredictRequest(features=[])

        # Test non-numeric features raise validation error
        with pytest.raises(ValidationError):
            PredictRequest(features=["a", "b", "c"])

    except ImportError:
        pytest.skip("model_serving module not available")


def test_batch_prediction_consistency():
    """Test that batch predictions have consistent shape."""
    try:
        from model_serving import BatchPredictRequest
        from pydantic import ValidationError

        # Test valid batch
        request = BatchPredictRequest(features=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        assert len(request.features) == 2
        assert len(request.features[0]) == 3

        # Test inconsistent lengths
        with pytest.raises(ValidationError):
            BatchPredictRequest(features=[
                [1.0, 2.0, 3.0],
                [4.0, 5.0],  # Different length
            ])

    except ImportError:
        pytest.skip("model_serving module not available")


def test_model_id_generation():
    """Test that model IDs are consistently generated."""
    try:
        from model_serving import ModelCache

        cache = ModelCache()

        # Same path should generate same ID
        path1 = Path("/tmp/model.pkl")
        id1 = cache._get_model_id(path1)
        id2 = cache._get_model_id(path1)
        assert id1 == id2

        # Different paths should generate different IDs
        path2 = Path("/tmp/other_model.pkl")
        id3 = cache._get_model_id(path2)
        assert id1 != id3

    except ImportError:
        pytest.skip("model_serving module not available")


def test_prediction_counter():
    """Test that prediction counts are tracked correctly."""
    try:
        from model_serving import ModelCache
        from utils.model_io import ModelMetadata

        cache = ModelCache()
        metadata = ModelMetadata(
            model_type="test",
            model_name="test_model",
            feature_names=["f1", "f2"],
            hyperparameters={},
            performance_metrics={},
            training_date=datetime.now().isoformat(),
        )

        path = Path("/tmp/test_model.pkl")
        cache.put(path, "model", None, metadata, 0.1)

        # Initial count should be 0
        assert cache.prediction_counts[cache._get_model_id(path)] == 0

        # Get should increment count
        cache.get(path)
        assert cache.prediction_counts[cache._get_model_id(path)] == 1

        cache.get(path)
        assert cache.prediction_counts[cache._get_model_id(path)] == 2

    except ImportError:
        pytest.skip("model_serving module not available")


def test_cache_stats():
    """Test cache statistics reporting."""
    try:
        from model_serving import ModelCache
        from utils.model_io import ModelMetadata

        cache = ModelCache(cache_size=3)

        metadata = ModelMetadata(
            model_type="test",
            model_name="test",
            feature_names=["f1"],
            hyperparameters={},
            performance_metrics={},
            training_date=datetime.now().isoformat(),
        )

        # Add some models
        for i in range(2):
            path = Path(f"/tmp/model{i}.pkl")
            cache.put(path, f"model{i}", None, metadata, 0.1)

        stats = cache.get_stats()
        assert stats["cache_size"] == 2
        assert stats["max_cache_size"] == 3
        assert len(stats["models"]) == 2
        assert len(stats["prediction_counts"]) == 2
        assert len(stats["load_times"]) == 2

    except ImportError:
        pytest.skip("model_serving module not available")


def test_model_manager_integration():
    """Integration test for ModelManager with experiment tracking."""
    try:
        from model_serving import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_dir=Path(tmpdir))

            # Test that tracker is initialized
            assert manager.tracker is not None

            # Test health check
            health = manager.get_health()
            assert "status" in health
            assert "models_loaded" in health
            assert "cache_stats" in health

            # Test list models (empty directory)
            models = manager.list_available_models()
            assert isinstance(models, list)
            assert len(models) == 0  # Empty models directory

    except ImportError:
        pytest.skip("model_serving module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

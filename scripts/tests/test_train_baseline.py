"""
Tests for baseline training script.

Verifies the training script structure is correct.
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_script_exists():
    """Script file should exist."""
    script_path = Path("scripts/train_baseline.py")
    assert script_path.exists(), "train_baseline.py should exist"


def test_can_import_dependencies():
    """Should be able to import required dependencies."""
    try:
        import sklearn
        import lightgbm
        assert sklearn is not None
        assert lightgbm is not None
    except ImportError as e:
        pytest.skip(f"Dependencies not installed: {e}")


def test_can_import_model_io():
    """Should be able to import model I/O utilities."""
    try:
        from utils.model_io import save_sklearn_model, load_sklearn_model
        assert save_sklearn_model is not None
        assert load_sklearn_model is not None
    except ImportError as e:
        pytest.fail(f"Failed to import model_io: {e}")

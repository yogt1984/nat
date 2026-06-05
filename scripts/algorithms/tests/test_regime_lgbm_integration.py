"""Integration tests for RegimeConditionedLGBM."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.regime_conditioned_lgbm import RegimeConditionedLGBM
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=400)


def test_run_batch_output_shape(bars):
    """run_batch returns correct columns and length."""
    rlgbm = RegimeConditionedLGBM()
    result = rlgbm.run_batch(bars)

    assert len(result) == len(bars)
    expected_cols = {
        "alg_rlgbm_signal", "alg_rlgbm_predicted_return",
        "alg_rlgbm_regime_used", "alg_rlgbm_regime_confidence",
    }
    assert set(result.columns) == expected_cols


def test_regime_routing_varies_output(bars):
    """With mock models, different regimes produce different predictions.

    Soft check: log result rather than hard-fail, since the test uses
    synthetic data where per-regime advantage may not materialize.
    """
    rlgbm = RegimeConditionedLGBM()

    # Inject mock models with distinct return values per group
    class MockBooster:
        def __init__(self, val):
            self._val = val
        def predict(self, X):
            return np.full(len(X), self._val)

    rlgbm._models = {
        "trending": MockBooster(0.005),
        "ranging": MockBooster(-0.003),
        "volatile": MockBooster(0.0),
        "global": MockBooster(0.001),
    }

    result = rlgbm.run_batch(bars)
    post_warmup = result.iloc[rlgbm.warmup:]
    pred = post_warmup["alg_rlgbm_predicted_return"].dropna()

    # With distinct mock models, we should see variation in predictions
    assert pred.std() > 0, "Expected variation in predictions across regimes"

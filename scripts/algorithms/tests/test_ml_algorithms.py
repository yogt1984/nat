"""Parametrized tests for all ML (bar_level=True) algorithms.

Auto-discovers ML algorithms from the registry and validates
common contract properties: bar_level flag, warmup bounds,
required column naming, no-model graceful degradation, and
run_batch output shape.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.registry import list_algorithms, get_algorithm, _REGISTRY


# Auto-discover ML algorithms (bar_level=True)
ML_ALGOS = [name for name in list_algorithms()
            if getattr(_REGISTRY[name], 'bar_level', False)]


def _make_bar_df_for_algo(algo, n_bars=200, seed=42):
    """Build a bar DataFrame containing all columns an algorithm needs."""
    rng = np.random.default_rng(seed)
    midprice = 50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars)))

    data = {
        "raw_midprice_mean": midprice,
        "bar_start": np.arange(n_bars) * 300_000_000_000,
    }

    for col in algo.required_columns():
        if col in data:
            continue
        if "midprice" in col:
            data[col] = midprice * (1 + rng.normal(0, 0.0003, n_bars))
        elif "spread" in col:
            data[col] = np.abs(rng.normal(0.5, 0.2, n_bars))
        elif "imbalance" in col or "pressure" in col or "directional" in col:
            data[col] = rng.uniform(-1, 1, n_bars)
        elif "hurst" in col or "momentum" in col:
            data[col] = rng.normal(0, 0.1, n_bars)
        elif "ent_" in col or "vpin" in col or "accumulation" in col or "clarity" in col:
            data[col] = rng.uniform(0, 1, n_bars)
        elif "vol_" in col:
            data[col] = np.abs(rng.normal(0.001, 0.0005, n_bars))
        elif "whale" in col:
            data[col] = rng.normal(0, 1000, n_bars)
        elif "regime" in col and "last" in col:
            data[col] = rng.choice([0, 1, 2, 3, 4, 5], n_bars).astype(float)
        elif "confidence" in col:
            data[col] = rng.uniform(0.3, 0.9, n_bars)
        elif "conc_" in col or "hhi" in col:
            data[col] = rng.uniform(0, 1, n_bars)
        elif "toxic" in col:
            data[col] = rng.uniform(0, 1, n_bars)
        elif "bb_pctb" in col:
            data[col] = rng.uniform(0, 1, n_bars)
        elif "ema" in col:
            data[col] = midprice * (1 + rng.normal(0, 0.0005, n_bars))
        elif "ratio" in col:
            data[col] = rng.uniform(0.5, 2.0, n_bars)
        else:
            data[col] = rng.normal(0, 1, n_bars)

    return pd.DataFrame(data)


@pytest.fixture(params=ML_ALGOS)
def ml_algo(request):
    return get_algorithm(request.param)


def test_bar_level_is_true(ml_algo):
    """All ML algorithms must have bar_level=True."""
    assert ml_algo.bar_level is True


def test_warmup_in_bars_not_ticks(ml_algo):
    """All warmup values < 1000 (bars, not 100ms ticks)."""
    for f in ml_algo.alg_features():
        assert f.warmup < 1000, (
            f"{ml_algo.name()}.{f.name} warmup={f.warmup} looks like ticks, not bars"
        )


def test_required_columns_have_suffixes(ml_algo):
    """All required columns end with valid bar-aggregation suffixes."""
    valid = ('_mean', '_std', '_last', '_sum', '_slope',
             '_close', '_open', '_high', '_low')
    for col in ml_algo.required_columns():
        assert any(col.endswith(s) for s in valid), (
            f"{ml_algo.name()} column '{col}' missing bar suffix"
        )


def test_no_model_graceful(ml_algo):
    """Algorithm with no trained model does not crash on step()."""
    tick = {c: 0.5 for c in ml_algo.required_columns()}
    # Also provide raw_midprice_mean if not in required (KNN needs it)
    tick.setdefault("raw_midprice_mean", 50000.0)
    result = ml_algo.step(tick)
    expected_keys = {f.name for f in ml_algo.alg_features()}
    assert set(result.keys()) == expected_keys


def test_run_batch_on_bar_df(ml_algo):
    """run_batch() returns DataFrame with correct columns."""
    df = _make_bar_df_for_algo(ml_algo, n_bars=200)
    result = ml_algo.run_batch(df)
    expected_cols = {f.name for f in ml_algo.alg_features()}
    assert set(result.columns) == expected_cols
    assert len(result) == len(df)

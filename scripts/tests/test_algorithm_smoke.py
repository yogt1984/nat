"""Conformance smoke test for the Algorithm contract.

Canonical test referenced by `docs/contracts/algorithm.md`, `CLAUDE.md`, and the
`planted-test-author` agent. Asserts every `@register`-ed `MicrostructureAlgorithm`
honors its contract (docs/contracts/algorithm.md):

  - `alg_features()` is non-empty and every output name starts with `alg_`
  - `step()` returns **exactly** the keys declared by `alg_features()` (the key-match
    contract — `run_batch()` masks this via `.get(name, NaN)`, so we call `step()` directly)
  - `run_batch()` yields the declared columns, input length, and NaN-blanked warmup rows
  - NaN required input -> NaN outputs, no crash/impute (tick-level algos)

Parametrized per algorithm so `pytest scripts/tests/test_algorithm_smoke.py -k <name>`
works, as the contract docs promise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.autodiscover import discover_all
from algorithms.registry import get_algorithm, list_algorithms

discover_all()
ALGOS = list_algorithms()
# step()/run_batch tick semantics apply to tick-level algos. Bar-level ML models are
# invoked through AlgorithmRunner's bar-aggregation (covered by test_bar_level_dispatch.py
# + each model's ML spec), so the strict step/run_batch/NaN checks below scope to
# tick-level algos; the alg_-prefix check still covers every registered algorithm.
TICK_ALGOS = [n for n in ALGOS if not get_algorithm(n).bar_level]


def _synthetic_df(required: list[str], n: int = 256, seed: int = 42) -> pd.DataFrame:
    """Finite synthetic frame covering an algorithm's required columns."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in required:
        if any(k in col for k in ("price", "mid", "vwap")):
            data[col] = 50_000 + rng.normal(0, 10, n).cumsum()
        else:
            data[col] = rng.normal(0, 1, n)
    data.setdefault("raw_midprice", 50_000 + rng.normal(0, 10, n).cumsum())
    return pd.DataFrame(data)


def test_registry_nonempty():
    assert ALGOS, "no algorithms registered (discover_all failed?)"


@pytest.mark.parametrize("name", ALGOS)
def test_alg_features_prefixed(name):
    """Every declared output feature starts with the mandated `alg_` prefix."""
    algo = get_algorithm(name)
    feats = algo.alg_features()
    assert feats, f"{name}: alg_features() is empty"
    bad = [f.name for f in feats if not f.name.startswith("alg_")]
    assert not bad, f"{name}: feature names missing 'alg_' prefix: {bad}"


@pytest.mark.parametrize("name", TICK_ALGOS)
def test_step_keys_match_alg_features(name):
    """step() returns exactly the keys declared by alg_features() — no more, no less."""
    algo = get_algorithm(name)
    algo.reset()
    df = _synthetic_df(algo.required_columns())
    tick = {c: float(df[c].iloc[0]) for c in algo.required_columns() if c in df.columns}
    out = algo.step(tick)
    assert isinstance(out, dict), f"{name}: step() must return a dict, got {type(out).__name__}"
    assert set(out.keys()) == set(algo.feature_names), (
        f"{name}: step() keys {sorted(out)} != alg_features() {sorted(algo.feature_names)}"
    )


@pytest.mark.parametrize("name", TICK_ALGOS)
def test_run_batch_shape_and_warmup(name):
    """run_batch yields exactly the declared columns, the input length, blanked warmup rows."""
    algo = get_algorithm(name)
    df = _synthetic_df(algo.required_columns())
    out = algo.run_batch(df)
    assert list(out.columns) == algo.feature_names, (
        f"{name}: run_batch columns {list(out.columns)} != {algo.feature_names}"
    )
    assert len(out) == len(df), f"{name}: run_batch length {len(out)} != {len(df)}"
    w = algo.warmup
    if 0 < w < len(df):
        assert out.iloc[:w].isna().all().all(), f"{name}: first {w} (warmup) rows not NaN-blanked"


@pytest.mark.parametrize("name", TICK_ALGOS)
def test_nan_in_nan_out(name):
    """A tick whose required inputs are all NaN must produce all-NaN outputs (no impute, no crash)."""
    algo = get_algorithm(name)
    algo.reset()
    tick = {c: float("nan") for c in algo.required_columns()}
    out = algo.step(tick)
    offenders = {k: v for k, v in out.items() if not (isinstance(v, float) and np.isnan(v))}
    assert not offenders, f"{name}: NaN inputs must yield NaN outputs; non-NaN: {offenders}"

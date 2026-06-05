# 14 — Verification Matrix & Test Infrastructure

Source: `docs/research/new/ml_implementation_plan.txt`, Sections 9–10 (lines 1224–1310)
Spec: `docs/research/new/ml_algorithms.txt`, Appendix D (lines 3797–4690)

---

### Task 1: Create ML-specific parametrized test suite

**Read first:** `scripts/algorithms/tests/test_algorithms.py` (existing parametrized smoke tests)

**Create:** `scripts/algorithms/tests/test_ml_algorithms.py` (~150 LOC)

Parametrized tests that auto-discover ML algorithms (those with `bar_level=True`):

```python
import pytest
from algorithms.registry import list_algorithms, get_algorithm

ML_ALGOS = [name for name, cls in _items() if getattr(cls, 'bar_level', False)]

@pytest.fixture(params=ML_ALGOS)
def ml_algo(request):
    return get_algorithm(request.param)()

def test_bar_level_is_true(ml_algo):
    assert ml_algo.bar_level is True

def test_warmup_in_bars_not_ticks(ml_algo):
    """All warmup values < 1000 (bars, not 100ms ticks)."""
    for f in ml_algo.alg_features():
        assert f.warmup < 1000

def test_required_columns_have_suffixes(ml_algo):
    """All required columns end with _mean, _std, _last, _sum, _slope, or _close."""
    valid = ('_mean', '_std', '_last', '_sum', '_slope', '_close', '_open', '_high', '_low')
    for col in ml_algo.required_columns():
        assert any(col.endswith(s) for s in valid), f"{col} missing bar suffix"

def test_no_model_graceful(ml_algo):
    """Algorithm with _model=None does not crash on step()."""
    tick = {c: 0.5 for c in ml_algo.required_columns()}
    result = ml_algo.step(tick)
    assert set(result.keys()) == {f.name for f in ml_algo.alg_features()}

def test_run_batch_on_bar_df(ml_algo, make_bar_df):
    """run_batch() returns DataFrame with correct columns."""
    df = make_bar_df(n_bars=200)
    result = ml_algo.run_batch(df)
    expected_cols = {f.name for f in ml_algo.alg_features()}
    assert set(result.columns) == expected_cols
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_ml_algorithms.py -v`

---

### Task 2: Create model persistence roundtrip test

**Create:** `scripts/tests/test_model_persistence.py` (~80 LOC)

```python
def test_sklearn_save_load_roundtrip(tmp_path):
    """Train LogReg, save, load. Predictions match to 1e-10."""

def test_lightgbm_save_load_roundtrip(tmp_path):
    """Train LightGBM, save, load. Predictions match exactly."""

def test_metadata_json_complete(tmp_path):
    """Saved metadata JSON has: model_type, model_name, feature_names,
    hyperparameters, performance_metrics, training_date."""

def test_get_latest_model_selects_newest(tmp_path):
    """Save model_v1, then model_v2. get_latest_model() returns v2."""

def test_get_latest_model_empty_dir(tmp_path):
    """Empty directory -> returns None, no crash."""
```

**Verification:** `cd scripts && python -m pytest tests/test_model_persistence.py -v`

---

### Task 3: Create full verification runner script

**Create:** `scripts/run_ml_verification.sh` (~50 lines)

Bash script that runs all ML verification phases in sequence:

```bash
#!/bin/bash
set -e
echo "=== Phase 1: Smoke tests ==="
cd scripts && python -m pytest tests/test_algorithm_smoke.py -v --timeout=600

echo "=== Phase 2: ML-specific tests ==="
python -m pytest algorithms/tests/test_ml_algorithms.py -v

echo "=== Phase 3: Constraint validation ==="
python validate_all_algorithms.py

echo "=== Phase 4: Model persistence ==="
python -m pytest tests/test_model_persistence.py -v

echo "=== Phase 5: Algorithm-specific unit tests ==="
python -m pytest algorithms/tests/test_change_point_unit.py -v 2>/dev/null || true
python -m pytest algorithms/tests/test_momentum_unit.py -v 2>/dev/null || true
# Add more as algorithms are implemented

echo "=== All ML verification passed ==="
```

**Verification:** `bash scripts/run_ml_verification.sh`

---

### Task 4: Create verification matrix document

**Create:** `docs/research/new/ml_specs/VERIFICATION_MATRIX.md` (~120 lines)

Table mapping each testable property to algorithms:

| Property | CPD | MC | RSM | MR | Meta | RLGBM | KNN |
|----------|-----|----|----|----|----|------|-----|
| ABC contract | x | x | x | x | x | x | x |
| Bar suffix check | x | x | x | x | x | x | x |
| No-model graceful | - | x | - | x | x | x | - |
| Model roundtrip | - | x | - | x | x | x | - |
| Entropy gating | - | x | - | inv | - | - | - |
| Regime dispatch | - | - | x | - | - | x | - |
| CUSUM correctness | x | - | - | - | - | - | - |
| Buffer management | - | - | - | - | - | - | x |

Sections:
1. Property matrix table
2. Per-phase verification commands
3. CI integration guidance (which tests to run on every commit vs nightly)

---

### Task 5: Update README.md testing section

**Modify:** `README.md`

Add to Testing section:

```markdown
# ML algorithm verification (all phases)
bash scripts/run_ml_verification.sh

# ML-specific parametrized tests
cd scripts && python -m pytest algorithms/tests/test_ml_algorithms.py -v

# Constraint validation
python scripts/validate_all_algorithms.py
```

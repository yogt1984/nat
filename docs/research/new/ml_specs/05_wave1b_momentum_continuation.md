# 05 — Wave 1b: Momentum Continuation Classifier (#1)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 3.2 (lines 579–740)
Spec: `docs/research/new/ml_algorithms.txt`, Section 1 (lines 60–604)

**Status: DONE.** Algorithm + training script implemented, 11/11 smoke tests pass. Awaiting training on real data.

---

### Task 1: Verify algorithm file and registration

**Read:** `scripts/algorithms/momentum_continuation.py`

Confirm:
- `bar_level = True`, `@register`, `name()` = `"momentum_continuation"`
- 7 `FEATURE_COLS` with bar-aggregated suffixes
- 3 output features: `alg_mc_signal`, `alg_mc_confidence`, `alg_mc_entropy_gate`
- No-model fallback: signal=0.0, confidence=0.5 (not NaN)
- Model loading via `model_io.get_latest_model()` + `load_sklearn_model()` in `__init__`
- Entropy gate: signal zeroed when `ent_tick_1m_mean >= entropy_ceiling`
- Dead zone: signal=0 when `p_short <= prob <= p_long`
- `run_batch()` has vectorized path

**Verification:** `cd scripts && python -c "from algorithms.registry import list_algorithms; assert 'momentum_continuation' in list_algorithms()"`

---

### Task 2: Verify training script

**Read:** `scripts/train_momentum.py`

Confirm:
- `load_bars()` loads parquet and aggregates to 5-min bars
- `build_dataset()` computes forward returns (horizon=20 bars), binary labels, checks balance
- `walk_forward_train()` does expanding-window CV with `StandardScaler` + `LogisticRegression(C, penalty='l2')`
- Decision gate: OOS AUC < 0.52 = FAIL
- Saves via `model_io.save_sklearn_model()` with `ModelMetadata`
- CLI: `--symbol`, `--data-dir`, `--C`, `--n-splits`, `--embargo`, `--output-dir`, `--dry-run`

---

### Task 3: Write algorithm-specific unit tests

**Create:** `scripts/algorithms/tests/test_momentum_unit.py` (~120 LOC)

```python
def test_entropy_gate_blocks_high_entropy():
    """tick with ent_tick_1m_mean=0.90 (> 0.85 ceiling) -> signal=0.0, gate=0.0"""

def test_entropy_gate_passes_low_entropy():
    """tick with ent_tick_1m_mean=0.50 -> gate=1.0"""

def test_dead_zone_zeroes_signal():
    """When 0.4 <= P(continuation) <= 0.6, signal must be 0.0"""

def test_no_model_returns_neutral():
    """With no model loaded: signal=0.0, confidence=0.5, gate computed normally."""

def test_signal_range_bounded():
    """Across 500 random ticks, alg_mc_signal stays in [-1, 1]."""

def test_confidence_range():
    """alg_mc_confidence always in [0, 1] when model loaded."""

def test_nan_input_propagation():
    """If any FEATURE_COL is NaN, all outputs are NaN."""
```

For tests needing a model, create a minimal `LogisticRegression` fit on 50 random samples and inject it via `algo._model = model`.

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_momentum_unit.py -v`

---

### Task 4: Write training pipeline unit tests

**Create:** `scripts/tests/test_train_momentum.py` (~100 LOC)

```python
def test_build_dataset_labels_binary():
    """Labels from build_dataset() are 0.0 or 1.0 only (no NaN in valid rows)."""

def test_build_dataset_drops_nan_features():
    """Rows with NaN in any FEATURE_COL are excluded from X, y."""

def test_forward_return_alignment():
    """fwd_ret[0] uses midprice[20], not midprice[19]. Off-by-one check."""

def test_walk_forward_produces_folds():
    """walk_forward_train() with n_splits=3 on 1000 synthetic samples produces 3 fold results."""

def test_walk_forward_embargo_respected():
    """For each fold, test_start >= train_end + embargo."""

def test_model_metadata_complete():
    """After training, ModelMetadata has all required fields."""
```

Use synthetic data: generate random X (1000x7) and binary y. No parquet I/O.

**Verification:** `cd scripts && python -m pytest tests/test_train_momentum.py -v`

---

### Task 5: Write integration test

**Create:** `scripts/algorithms/tests/test_momentum_integration.py` (~50 LOC)

```python
def test_run_batch_without_model(make_bar_df):
    """run_batch() with no model returns signal=0.0 for valid rows, NaN during warmup."""

def test_run_batch_with_injected_model(make_bar_df):
    """Inject a trained LogReg, run_batch() returns varied signals in [-1,1]."""

def test_save_load_roundtrip(tmp_path):
    """Train a model, save via model_io, load in new algorithm instance.
    Predictions match to 1e-10."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_momentum_integration.py -v`

---

### Task 6: Create algorithm specification document

**Create:** `docs/research/new/ml_specs/SPEC_MOMENTUM_CONTINUATION.md` (~120 lines)

Sections:
1. **Thesis:** low-entropy momentum persistence, entropy gate, whale confirmation
2. **Feature set:** table of 7 features with column names and descriptions
3. **Model:** LogisticRegression(C=1.0, L2), StandardScaler, signal mapping formula
4. **Training:** CLI command, walk-forward protocol, decision gate (AUC >= 0.52)
5. **Parameters:** table from `algorithms.toml` with defaults and ranges
6. **Output features:** table with name, range, warmup
7. **Differences from spec:** 7 features (not 9), C=1.0 (not 0.1), entropy_ceiling=0.85 (not 0.30)
8. **References:** Moskowitz et al. (2012), Bouchaud et al. (2004)

---

### Task 7: Update README.md

**Modify:** `README.md`

Add to the Algorithm Catalog table:

```markdown
| 29 | `momentum_continuation` | Logistic Regression momentum classifier | Moskowitz, Ooi & Pedersen (2012) |
```

**Verification:** `grep "momentum_continuation" README.md` returns at least 1 match.

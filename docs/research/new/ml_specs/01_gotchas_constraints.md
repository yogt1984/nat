# 01 — Gotchas & Constraints Guard Rails

Source: `docs/research/new/ml_implementation_plan.txt`, Section 0 (lines 53–224)

All tasks below must complete before any ML algorithm implementation begins.

---

### Task 1: Create constraint validation utility

**Read first:** `scripts/algorithms/registry.py`, `scripts/algorithms/base.py`

**Create:** `scripts/utils/ml_constraints.py` (~60 LOC)

Write a module with these functions:

```python
def validate_algorithm_class(cls) -> list[str]:
    """Return list of constraint violations for a MicrostructureAlgorithm subclass.
    Checks:
    1. cls() callable with no arguments (simulates @register)
    2. All alg_features() names start with 'alg_'
    3. required_columns() returns list of strings
    4. step() returns dict with exactly the keys from alg_features()
    5. If bar_level is True, required_columns use aggregated suffixes (_mean, _std, _last, _sum, _slope)
    """

def validate_model_path(model_path: str) -> bool:
    """Check model_path parent exists and is writable."""

def check_bar_suffix(col: str) -> bool:
    """Return True if column name ends with a valid bar-aggregation suffix."""
```

**Verification:** `python -c "from scripts.utils.ml_constraints import validate_algorithm_class; print('OK')"`

---

### Task 2: Write unit tests for constraint validation

**Create:** `scripts/tests/test_ml_constraints.py` (~80 LOC)

Tests:

```python
def test_no_arg_constructor_passes():
    """A class with all-default __init__ returns empty violations."""

def test_no_arg_constructor_fails():
    """A class requiring a positional arg returns 'constructor' violation."""

def test_feature_prefix_check():
    """Features not starting with 'alg_' are flagged."""

def test_bar_suffix_valid():
    """'ent_tick_1m_mean' passes, 'ent_tick_1m' fails."""

def test_step_key_mismatch():
    """step() returning extra or missing keys is flagged."""
```

Run each test with synthetic stub classes, no real algorithm imports needed.

**Verification:** `cd scripts && python -m pytest tests/test_ml_constraints.py -v`

---

### Task 3: Run constraint validation on all existing algorithms

**Create:** `scripts/validate_all_algorithms.py` (~40 LOC)

Script that:
1. Calls `discover_algorithms()` from `scripts/algorithms/autodiscover.py`
2. For each algorithm in the registry, calls `validate_algorithm_class(cls)`
3. Prints PASS/FAIL per algorithm
4. Exits with code 1 if any fail

**Verification:** `python scripts/validate_all_algorithms.py` — all existing algorithms must PASS.

---

### Task 4: Create developer reference document

**Create:** `docs/research/new/ml_specs/ML_DEVELOPER_GUIDE.md` (~120 lines)

Content derived from `ml_implementation_plan.txt` Section 0. Organize as:

1. **Registry contract** — `@register` calls `cls()` with no args. All `__init__` params need defaults.
2. **Bar-level algorithms** — set `bar_level = True`. `runner.py` handles aggregation. Use `_mean/_std/_last/_sum/_slope` suffixes in `required_columns()`.
3. **walk_forward_validation()** — takes Polars, not Pandas. Convert with `pl.from_pandas()`.
4. **Config loading** — `config/algorithms.toml` is not auto-injected. Algorithm loads explicitly or accepts kwargs.
5. **model_io conventions** — `save_sklearn_model()`, `load_sklearn_model()`, `get_latest_model()`. No model = return NaN.
6. **autodiscover.py** — silently swallows import errors. Always verify with `list_algorithms()`.
7. **AlgorithmFeature** — frozen dataclass. `name` starts with `alg_`. `warmup` is in bars for bar-level algorithms.
8. **Testing** — new algorithms auto-discovered by `test_algorithm_smoke.py`. Must pass all 11 parametrized tests.

Each section: 2–3 sentences + one code snippet showing the correct pattern.

---

### Task 5: Update README.md with ML algorithm section stub

**Modify:** `README.md`

Find the line `| 22-27 | Various | Additional signal algorithms | See scripts/algorithms/ |` in the Algorithm Catalog table. After the table, before `### Top Performer Algorithms`, insert:

```markdown
### ML Algorithms (Wave-Gated Pipeline)

NAT's ML algorithm pipeline is implemented in waves with hard decision gates.
Each wave must demonstrate positive OOS alpha before the next begins.
See `docs/research/new/ml_specs/` for per-algorithm specifications.

| Wave | Algorithm | Method | Status |
|------|-----------|--------|--------|
| 0 | Infrastructure | bar_level support, WelfordNormalizer | Done |
| 1 | `change_point_detector` | CUSUM + Bayesian OCD | Done |
| 1 | `momentum_continuation` | Logistic Regression (7 features) | Done (awaiting training) |
| 2 | `regime_state_machine` | Manual threshold 6-state classifier | Gated |
| 2 | `mean_reversion_detector` | LightGBM false-breakout detector | Gated |
| 2 | `meta_labeling` | De Prado triple-barrier precision filter | Gated |
| 3 | `regime_conditioned_lgbm` | Per-regime LightGBM ensemble | Gated |
| 3 | `knn_retrieval` | Mahalanobis nearest-neighbor | Gated |

Developer guide: `docs/research/new/ml_specs/ML_DEVELOPER_GUIDE.md`
```

Also update the header count: change `27 Algorithms` to `29 Algorithms` in both the ASCII banner line and the ToC entry.

**Verification:** `grep -c "29 Algorithms" README.md` should return 2.

---

### Task 6: Integration test — validate all registered algorithms

**Create:** `scripts/tests/test_ml_constraint_integration.py` (~40 LOC)

```python
def test_all_registered_algorithms_pass_constraints():
    """Import all algorithms, run validate_algorithm_class on each.
    Assert zero violations for every registered algorithm."""
```

This test imports `discover_algorithms`, iterates the registry, and calls
`validate_algorithm_class` on each class. It must pass on the current codebase
before any new ML algorithm is added.

**Verification:** `cd scripts && python -m pytest tests/test_ml_constraint_integration.py -v`

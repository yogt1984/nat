# 03 — Wave 0: Infrastructure Prerequisites

Source: `docs/research/new/ml_implementation_plan.txt`, Section 2 (lines 327–437)

**Status: DONE** on branch `feat/ml-wave0-wave1`. Tasks below document what was implemented and remaining verification/documentation work.

---

### Task 1: Verify bar_level support in base.py and runner.py

**Read:** `scripts/algorithms/base.py`, `scripts/algorithms/runner.py`

Confirm these exist:
- `base.py`: class attributes `bar_level: bool = False` and `bar_timeframe: str = "5min"` on `MicrostructureAlgorithm`
- `runner.py`: in `run_on_dataframe()`, if `algo.bar_level and "timestamp_ns" in df.columns`, calls `aggregate_bars()` before `run_batch()`, then forward-fills results to tick-level index

**Verification:**
```bash
cd scripts && python -c "
from algorithms.base import MicrostructureAlgorithm
assert hasattr(MicrostructureAlgorithm, 'bar_level')
assert MicrostructureAlgorithm.bar_level == False
print('bar_level support: OK')
"
```

---

### Task 2: Verify WelfordNormalizer extraction

**Read:** `scripts/utils/online.py`, `scripts/algorithms/cascade_probability.py`

Confirm:
- `scripts/utils/online.py` exists with `WelfordNormalizer` class (methods: `update`, `normalize`, `mean`, `std`, `reset`)
- `cascade_probability.py` imports from `utils.online` (not inline definition)

**Verification:** `cd scripts && python -m pytest tests/test_algorithm_smoke.py -k cascade --timeout=60`

---

### Task 3: Verify bar-level test fixtures

**Read:** `scripts/algorithms/tests/conftest.py`

Confirm these fixtures exist:
- `make_bar_df(n_bars=400)` — generates DataFrame with `_mean/_std/_last` columns
- `make_forward_returns(bars_df, horizon=20)` — forward returns from `raw_midprice_mean`
- `make_labeled_df(bars_df, fwd_returns)` — binary labels

**Verification:** `cd scripts && python -c "from algorithms.tests.conftest import make_bar_df; print(make_bar_df().columns.tolist()[:5])"`

---

### Task 4: Write regression test for bar_level dispatch

**Create:** `scripts/tests/test_bar_level_dispatch.py` (~80 LOC)

```python
def test_tick_algo_receives_raw_df():
    """A tick-level algorithm (bar_level=False) receives the original DataFrame unchanged."""

def test_bar_algo_receives_aggregated_bars():
    """A bar-level algorithm (bar_level=True) receives aggregated bars with _mean suffixes."""

def test_bar_algo_output_ffilled_to_tick_index():
    """Bar-level algorithm output is forward-filled to match tick-level index length."""

def test_existing_algos_unaffected():
    """Run 3 existing tick-level algorithms. Outputs identical before/after runner change."""
```

Use a minimal stub algorithm class with `bar_level=True` for the first three tests. For the fourth, pick `weighted_ofi`, `jump_detector`, `hawkes_intensity` and compare outputs.

**Verification:** `cd scripts && python -m pytest tests/test_bar_level_dispatch.py -v`

---

### Task 5: Document Wave 0 changes

**Create:** `docs/research/new/ml_specs/WAVE0_CHANGELOG.md` (~50 lines)

Content:
1. What changed: `base.py` (+2 attrs), `runner.py` (+bar dispatch), `utils/online.py` (new), `cascade_probability.py` (import change), `conftest.py` (+fixtures)
2. Why: all ML algorithms are bar-level; centralizing aggregation prevents copy-paste
3. How to use: set `bar_level = True` in algorithm class body
4. Backward compatibility: all existing tick-level algorithms unaffected (`bar_level=False` default)

---

### Task 6: Update README.md architecture section

**Modify:** `README.md`

Find the `### Algorithm Contract` equivalent text (or the algorithm section). Add after the algorithm catalog table:

```markdown
**Bar-level algorithms:** ML algorithms set `bar_level = True`. The runner
automatically calls `aggregate_bars()` before `run_batch()` and forward-fills
results to tick-level. No per-algorithm aggregation code needed.
```

**Verification:** `grep "bar_level" README.md` returns at least 1 match.

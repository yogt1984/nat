# 04 — Wave 1a: Change-Point Detector (#5)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 3.1 (lines 460–577)
Spec: `docs/research/new/ml_algorithms.txt`, Section 5 (lines 1858–2178)

**Status: DONE.** Algorithm implemented, 11/11 smoke tests pass.

---

### Task 1: Verify algorithm file and registration

**Read:** `scripts/algorithms/change_point_detector.py`

Confirm:
- Class `ChangePointDetector` inherits `MicrostructureAlgorithm`
- `bar_level = True`
- `@register` decorator present
- `name()` returns `"change_point_detector"`
- 4 output features: `alg_cpd_cusum_signal`, `alg_cpd_run_length`, `alg_cpd_change_prob`, `alg_cpd_regime_age`
- 3 required columns: `imbalance_qty_l1_mean`, `vol_returns_5m_last`, `ent_tick_1m_mean`
- CUSUM bilateral detector + Adams-MacKay Bayesian OCD implemented
- Run-length capped at `max_run_length`, renormalized

**Verification:**
```bash
cd scripts && python -c "
from algorithms.autodiscover import discover_algorithms; discover_algorithms()
from algorithms.registry import list_algorithms
assert 'change_point_detector' in list_algorithms()
print('Registration: OK')
"
```

---

### Task 2: Verify config, paper trader, and daily wiring

**Read:** `config/algorithms.toml`, `scripts/alpha/paper_trader_generic.py`, `scripts/alpha/paper_trader_daily.py`

Confirm:
- `[change_point_detector]` section in `algorithms.toml` with keys: `cusum_threshold`, `cusum_drift`, `hazard_rate`, `max_run_length`, `calibration_window`
- `"change_point_detector"` entry in `ALGO_CONFIG` dict with `primary="alg_cpd_cusum_signal"`, `polarity="high_long"`, `bar_agg="last"`
- `"change_point_detector"` in `DAILY_ALGOS` list

---

### Task 3: Write algorithm-specific unit tests

**Create:** `scripts/algorithms/tests/test_change_point_unit.py` (~120 LOC)

```python
def test_cusum_zero_on_constant_input():
    """Feed 200 identical ticks. CUSUM signal stays near 0. No alarms."""

def test_cusum_detects_mean_shift():
    """Feed 200 ticks at mean=0, then 200 at mean=2.0.
    CUSUM alarm fires within 30 bars of shift."""

def test_bayesian_run_length_grows_on_stable():
    """Feed 100 stable ticks. Expected run length > 50."""

def test_bayesian_detects_change():
    """Feed 100 stable + 100 shifted ticks.
    alg_cpd_change_prob > 0.3 within 20 bars of shift."""

def test_regime_age_resets_on_alarm():
    """After CUSUM fires, alg_cpd_regime_age resets to 0."""

def test_run_length_capped():
    """After max_run_length + 100 ticks, run_length_probs array
    never exceeds max_run_length entries."""

def test_nan_input_returns_nan():
    """If imbalance_qty_l1_mean is NaN, all outputs are NaN."""
```

Build ticks as dicts with the 3 required columns + dummy values for others.

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_change_point_unit.py -v`

---

### Task 4: Write integration test with real data shape

**Create:** `scripts/algorithms/tests/test_change_point_integration.py` (~50 LOC)

```python
def test_run_batch_on_bar_df(make_bar_df):
    """Run change_point_detector.run_batch() on synthetic bar DataFrame.
    Output has 4 columns. After warmup (100 bars), >50% of rows finite."""

def test_step_batch_consistency(make_bar_df):
    """step() iterated row-by-row produces same results as run_batch()
    for alg_cpd_cusum_signal (correlation > 0.95)."""
```

Use `make_bar_df` fixture from conftest.py.

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_change_point_integration.py -v`

---

### Task 5: Create algorithm specification document

**Create:** `docs/research/new/ml_specs/SPEC_CHANGE_POINT_DETECTOR.md` (~100 lines)

Sections:
1. **Purpose:** early-warning for regime transitions via CUSUM + Bayesian OCD
2. **Mathematical formulation:** CUSUM update equations (S+ and S-), Bayesian run-length (growth/changepoint/normalize steps), Student-t predictive distribution
3. **Parameters:** table of 5 config params with defaults and tuning ranges
4. **Output features:** table with name, range, warmup, description
5. **References:** Page (1954), Adams & MacKay (2007)
6. **File locations:** algorithm, config, paper trader entries

---

### Task 6: Update README.md algorithm table

**Modify:** `README.md`

In the Algorithm Catalog table, add before the `| 22-27 |` row:

```markdown
| 28 | `change_point_detector` | CUSUM + Bayesian OCD | Page (1954), Adams & MacKay (2007) |
```

**Verification:** `grep "change_point_detector" README.md` returns at least 1 match.

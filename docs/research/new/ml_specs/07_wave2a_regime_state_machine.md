# 07 — Wave 2a: Regime State Machine (#4)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 5.1 (lines 817–866)
Spec: `docs/research/new/ml_algorithms.txt`, Section 4 (lines 1468–1852)

**Status: NOT STARTED.** Gated on Wave 1 decision gate (Case A or B).

---

### Task 1: Implement algorithm class

**Read first:** `scripts/algorithms/base.py`, `scripts/algorithms/change_point_detector.py` (as pattern), `docs/research/new/ml_algorithms.txt` Section 4.2–4.3

**Create:** `scripts/algorithms/regime_state_machine.py` (~200 LOC)

```python
@register
class RegimeStateMachine(MicrostructureAlgorithm):
    bar_level = True

    # 6 states with manual thresholds
    ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN, HIGH_VOL, VOLATILE_NOISE = range(6)
```

Implementation requirements:
1. `__init__` with all-default params: `accum_thresh=0.70`, `distrib_thresh=0.70`, `entropy_low=0.30`, `entropy_high=0.60`, `hurst_trend=0.55`, `vol_noise_mult=2.0`, `min_duration=5`
2. `required_columns()`: `vol_returns_5m_last`, `trend_hurst_300_mean`, `ent_tick_1m_mean`, `whale_net_flow_4h_sum`, `toxic_vpin_50_mean`, `regime_accumulation_score_mean`, `regime_distribution_score_mean`, `trend_momentum_300_mean`
3. `step()`: compute score for each state (count conditions met), `state = argmax(scores)`, tie-break to VOLATILE_NOISE
4. Track `_regime_age` (bars since last state change), enforce `min_duration` hysteresis
5. `alg_features()`: 4 outputs — `alg_rsm_regime`, `alg_rsm_confidence`, `alg_rsm_transition_risk`, `alg_rsm_trade_allowed`
6. `run_batch()`: vectorized implementation using numpy where/select
7. `reset()`: clear internal state (`_current_regime`, `_regime_age`, rolling vol median)

No model loading. No training script.

---

### Task 2: Add config and paper trader wiring

**Modify:** `config/algorithms.toml` — add section:
```toml
[regime_state_machine]
accum_thresh = 0.70
distrib_thresh = 0.70
entropy_low = 0.30
entropy_high = 0.60
hurst_trend = 0.55
vol_noise_mult = 2.0
min_duration = 5
```

**Modify:** `scripts/alpha/paper_trader_generic.py` — add ALGO_CONFIG:
```python
"regime_state_machine": {
    "primary": "alg_rsm_regime",
    "polarity": "high_long",
    "bar_agg": "last",
},
```

**Modify:** `scripts/alpha/paper_trader_daily.py` — add to DAILY_ALGOS list.

---

### Task 3: Write unit tests

**Create:** `scripts/algorithms/tests/test_regime_sm_unit.py` (~150 LOC)

```python
def test_accumulation_detection():
    """Low entropy, positive whale, high absorption, low range_pos -> state 0."""

def test_volatile_noise_detection():
    """Very high vol, high entropy, high toxicity -> state 5."""

def test_regime_is_integer():
    """alg_rsm_regime always in {0,1,2,3,4,5} across 500 ticks."""

def test_confidence_range():
    """alg_rsm_confidence in [0, 1] always."""

def test_trade_allowed_zero_in_noise():
    """When state=5 (VOLATILE_NOISE), alg_rsm_trade_allowed=0."""

def test_min_duration_hysteresis():
    """State doesn't change for min_duration bars even if scores shift."""

def test_regime_age_increments():
    """alg_rsm_transition_risk derived from regime_age; age increments each bar."""

def test_tiebreak_favors_noise():
    """When two states tie, VOLATILE_NOISE wins (conservative)."""

def test_all_six_states_reachable():
    """Over varied synthetic data, all 6 states appear at least once."""

def test_nan_input_returns_nan():
    """NaN in any required column -> all outputs NaN."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_regime_sm_unit.py -v`

---

### Task 4: Write integration test

**Create:** `scripts/algorithms/tests/test_regime_sm_integration.py` (~50 LOC)

```python
def test_run_batch_state_distribution(make_bar_df):
    """No state occupancy < 5% or > 80% on 400-bar synthetic data."""

def test_step_batch_consistency(make_bar_df):
    """step() row-by-row and run_batch() produce same regime labels."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_regime_sm_integration.py -v`

---

### Task 5: Create specification document

**Create:** `docs/research/new/ml_specs/SPEC_REGIME_STATE_MACHINE.md` (~100 lines)

Sections: Purpose, 6-state table (conditions per state), scoring formula, hysteresis logic, parameters table, output features, references (Hamilton 1989, Wyckoff 1931).

---

### Task 6: Update README.md

**Modify:** `README.md` — add to Algorithm Catalog table:
```markdown
| 30 | `regime_state_machine` | 6-state manual threshold classifier | Hamilton (1989) |
```

# 15 — Risk Register, Rollback & Timeline

Source: `docs/research/new/ml_implementation_plan.txt`, Sections 12–14 (lines 1382–1561)

---

### Task 1: Create rollback utility script

**Create:** `scripts/ml_rollback.py` (~100 LOC)

CLI tool for quick removal or model rollback of ML algorithms:

```bash
# Remove algorithm from trading (keep code)
python scripts/ml_rollback.py disable momentum_continuation

# Re-enable
python scripts/ml_rollback.py enable momentum_continuation

# Roll back to previous model version
python scripts/ml_rollback.py rollback-model momentum_continuation

# List model versions
python scripts/ml_rollback.py list-models momentum_continuation
```

`disable` action:
1. Read `paper_trader_daily.py`, remove name from `DAILY_ALGOS` list
2. Read `paper_trader_generic.py`, comment out ALGO_CONFIG entry
3. Print confirmation

`enable` action: reverse of disable.

`rollback-model` action:
1. List files in `models/<name>/` sorted by timestamp
2. Move newest file to `models/<name>/archived/`
3. Print which model is now active via `get_latest_model()`

`list-models` action:
1. List all model files with timestamps and metadata summary

---

### Task 2: Write unit tests for rollback utility

**Create:** `scripts/tests/test_ml_rollback.py` (~80 LOC)

```python
def test_disable_removes_from_daily_algos(tmp_path):
    """Write a mock paper_trader_daily.py, disable algo, verify removed."""

def test_enable_adds_back(tmp_path):
    """After disable + enable, algo is back in DAILY_ALGOS."""

def test_rollback_model_archives_newest(tmp_path):
    """Create 2 model files. rollback moves newest to archived/."""

def test_list_models_empty_dir(tmp_path):
    """Empty model dir -> prints 'No models found'."""

def test_rollback_no_previous(tmp_path):
    """Only 1 model file -> rollback prints error, doesn't delete."""
```

**Verification:** `cd scripts && python -m pytest tests/test_ml_rollback.py -v`

---

### Task 3: Create ML health monitoring script

**Create:** `scripts/ml_health_check.py` (~100 LOC)

Nightly health check for deployed ML models:

```bash
python scripts/ml_health_check.py --data-dir data/features
```

Checks per deployed ML algorithm:
1. **Model age:** days since training_date in metadata. WARN if >14, CRITICAL if >30.
2. **IC rolling:** load last 7 days of `nat daily` output, compute Spearman(signal, fwd_return). WARN if IC < 0.005 for >2 days.
3. **Sharpe rolling:** 7-day and 30-day Sharpe from daily PnL. CRITICAL if 7d Sharpe < -0.5.
4. **NaN rate:** fraction of signal outputs that are NaN in last day. WARN if >20%.

Output summary:

```
ML HEALTH CHECK — 2026-06-05
=============================
momentum_continuation:  OK (age=3d, IC=0.04, Sharpe_7d=0.8)
change_point_detector:  OK (age=N/A, IC=0.02, Sharpe_7d=0.3)
```

Exit code 0 if all OK, 1 if any WARNING, 2 if any CRITICAL.

---

### Task 4: Write unit tests for health checks

**Create:** `scripts/tests/test_ml_health.py` (~80 LOC)

```python
def test_model_age_ok():
    """training_date 3 days ago -> OK"""

def test_model_age_warn():
    """training_date 20 days ago -> WARN"""

def test_model_age_critical():
    """training_date 35 days ago -> CRITICAL"""

def test_ic_warn():
    """IC=0.003 for 3 consecutive days -> WARN"""

def test_sharpe_critical():
    """7d Sharpe = -0.7 -> CRITICAL"""

def test_nan_rate_warn():
    """25% NaN signals -> WARN"""

def test_all_ok():
    """All metrics healthy -> exit code 0"""
```

**Verification:** `cd scripts && python -m pytest tests/test_ml_health.py -v`

---

### Task 5: Create risk register document

**Create:** `docs/research/new/ml_specs/RISK_REGISTER.md` (~100 lines)

| Risk | Severity | Mitigation | Monitoring |
|------|----------|------------|------------|
| Insufficient data | HIGH | Hard gate: bars >= 4000. Use num_leaves=15. | `check_data_sufficiency.py` |
| Overfitting | HIGH | Walk-forward, embargo=100, OOS/IS < 0.5 kills | Per-fold metrics in training |
| Data leakage | HIGH | Embargo, out-of-fold, forward-return alignment test | `test_forward_return_alignment` |
| Sunk cost fallacy | HIGH | Hard decision gates after each wave | `evaluate_wave*_gate.py` |
| Model staleness | MEDIUM | Weekly retrain, IC decay monitoring | `ml_health_check.py` |
| Complexity creep | MEDIUM | Start simplest (LogReg > LightGBM, manual > HMM) | Code review |
| Correlation clustering | MEDIUM | Pairwise Spearman < 0.5 gate | `evaluate_wave*_gate.py` |
| Regime mismatch | MEDIUM | RSM as gate, per-regime monitoring | `ml_health_check.py` |

Sections:
1. Risk table (above)
2. Rollback triggers (>50 bps loss, 5 consecutive losing days, rho > 0.6, OOS Sharpe < 0)
3. Rollback procedure: immediate (disable), full (delete), model-only (archive newest)
4. Tools: `ml_rollback.py`, `ml_health_check.py`

---

### Task 6: Create timeline document

**Create:** `docs/research/new/ml_specs/TIMELINE.md` (~50 lines)

| Phase | Duration | Deliverables | Gate |
|-------|----------|-------------|------|
| Wave 0 | 1 day | bar_level, WelfordNormalizer, fixtures | Smoke tests |
| Wave 1 | 5 days | CPD + Momentum + training pipeline | OOS Sharpe > 0.5 |
| Gate 1 | 1 day | Evaluate, decide proceed/stop | 4-case matrix |
| Wave 2 | 7 days | RSM + MR + Meta-labeling | 3+ algos positive |
| Gate 2 | 1 day | Evaluate, decide proceed/stop | 4-case matrix |
| Wave 3 | 5 days | Regime LGBM + KNN | Outperforms global |
| **Total committed** | **7 days** | Wave 0 + Wave 1 | |
| **Total if all pass** | **21 days** | Full ML portfolio | |

Include: first useful signal at day 6, most likely outcome 2-3 algorithms by day 15.

---

### Task 7: Update README.md

Add a new subsection under the ML Algorithms section:

```markdown
**Operations:**
```bash
python scripts/ml_health_check.py                    # nightly health check
python scripts/ml_rollback.py disable <algo>          # remove from trading
python scripts/ml_rollback.py rollback-model <algo>   # revert to previous model
```
```

Also add `ml_health_check.py` and `ml_rollback.py` to the Testing section commands.

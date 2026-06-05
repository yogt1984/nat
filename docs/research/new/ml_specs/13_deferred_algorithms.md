# 13 — Deferred Algorithms (#8, #9, #10)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 8 (lines 1156–1222)
Spec: `docs/research/new/ml_algorithms.txt`, Sections 8–10

NOT SCHEDULED. Tasks below define trigger-based re-evaluation and documentation only.

---

### Task 1: Create trigger monitoring script

**Create:** `scripts/check_deferred_triggers.py` (~80 LOC)

Script that checks whether conditions for re-evaluating deferred algorithms are met:

```bash
python scripts/check_deferred_triggers.py --data-dir data/features
```

Checks:
1. **Data volume:** `ls data/features/ | wc -l` > 60 days?
2. **Deployed ML count:** count algorithms in `DAILY_ALGOS` that start with known ML names. >= 4?
3. **Sharpe degradation:** for each deployed ML model, load last 14 days of `nat daily` results. Any model with Sharpe drop > 30% from peak?

Output for each deferred algorithm:

```
#8 HMM Emissions:      NOT TRIGGERED (data=35 days, need 60)
#9 Stacking:           NOT TRIGGERED (deployed ML=2, need 4)
#10 Online Learning:   NOT TRIGGERED (no Sharpe degradation detected)
```

---

### Task 2: Write unit tests

**Create:** `scripts/tests/test_deferred_triggers.py` (~60 LOC)

```python
def test_data_trigger_not_met():
    """35 days of data -> HMM trigger = False"""

def test_data_trigger_met():
    """65 days -> HMM trigger = True"""

def test_deployed_count_trigger():
    """4 ML algos in DAILY_ALGOS -> Stacking trigger = True"""

def test_sharpe_degradation_trigger():
    """Peak Sharpe 1.5, current 0.9 (40% drop) -> Online trigger = True"""

def test_sharpe_stable():
    """Peak 1.5, current 1.3 (13% drop) -> Online trigger = False"""
```

**Verification:** `cd scripts && python -m pytest tests/test_deferred_triggers.py -v`

---

### Task 3: Create deferred algorithms overview document

**Create:** `docs/research/new/ml_specs/DEFERRED_OVERVIEW.md` (~100 lines)

For each of #8, #9, #10:

**Section structure per algorithm:**
1. Name and method (1 line)
2. Why deferred (2–3 sentences)
3. Trigger conditions (bulleted list)
4. Estimated effort if triggered
5. Key dependencies

Also include:
- "What to do instead" section: accumulate data, retrain weekly, tune hyperparameters, SHAP analysis, different horizons
- Re-evaluation schedule: check triggers monthly or after each wave completion

---

### Task 4: Update README.md

In the ML Algorithms table, ensure the 3 deferred algorithms are listed with status "Deferred":

```markdown
| 4 | `hmm_emissions` | HMM with Gaussian emissions | Deferred |
| 4 | `stacking_ensemble` | Ridge/LightGBM stacking | Deferred |
| 4 | `online_learner` | Online SGD adaptation wrapper | Deferred |
```

Add a footnote: "Deferred algorithms are not scheduled. See `docs/research/new/ml_specs/DEFERRED_OVERVIEW.md` for trigger conditions."

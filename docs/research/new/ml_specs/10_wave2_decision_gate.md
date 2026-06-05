# 10 — Wave 2 Decision Gate

Source: `docs/research/new/ml_implementation_plan.txt`, Section 6 (lines 1015–1058)

Execute after all Wave 2 algorithms are deployed and evaluated.

---

### Task 1: Create Wave 2 gate evaluation script

**Read first:** `scripts/evaluate_wave1_gate.py` (as pattern)

**Create:** `scripts/evaluate_wave2_gate.py` (~130 LOC)

Collects metrics for all Wave 2 algorithms (#4, #2, #3) plus Wave 1 (#5, #1):

1. For each ML algorithm with a model: load OOS metrics from metadata JSON
2. Run `nat gauntlet run --last 3` and parse multi-day PnL per algorithm per symbol
3. Compute pairwise Spearman correlation matrix of all ML primary signals
4. Count how many of the committed ML algorithms have positive OOS Sharpe
5. Flag any pair with |rho| > 0.5 as "correlated"

Output decision:

```
WAVE 2 DECISION GATE
=====================
Algorithms with positive OOS: 3/4
Correlated pairs: none
Max pairwise |rho|: 0.28

DECISION: CASE A — PROCEED TO WAVE 3 (full)
```

| Case | Condition | Action |
|------|-----------|--------|
| A | 3+ algos positive OOS, uncorrelated | Wave 3 full (#7 + #6) |
| B | 2 algos positive, some correlation | Wave 3 partial (#7 only) |
| C | 1 algo positive | STOP, deploy only that one |
| D | 0 positive | STOP, ML adds no alpha |

CLI: `python scripts/evaluate_wave2_gate.py --data-dir data/features`

---

### Task 2: Write unit tests for gate logic

**Create:** `scripts/tests/test_wave2_gate.py` (~60 LOC)

```python
def test_case_a():
    """3 positive algos, max_rho=0.3 -> CASE_A"""

def test_case_b():
    """2 positive, rho=0.45 -> CASE_B"""

def test_case_c():
    """1 positive -> CASE_C"""

def test_case_d():
    """0 positive -> CASE_D"""

def test_correlation_flag():
    """Two algos with rho=0.55 flagged as correlated."""
```

Extract logic into `evaluate_wave2(algo_results: list[dict]) -> str`.

**Verification:** `cd scripts && python -m pytest tests/test_wave2_gate.py -v`

---

### Task 3: Create decision gate documentation

**Create:** `docs/research/new/ml_specs/GATE_WAVE2.md` (~50 lines)

Sections: Purpose, metrics collected (per-algo OOS + pairwise correlations), decision matrix (4 cases), how to run, record keeping location.

---

### Task 4: Update README.md

Add to Testing section:
```markdown
python scripts/evaluate_wave2_gate.py --data-dir data/features
```

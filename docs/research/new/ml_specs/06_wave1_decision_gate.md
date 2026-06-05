# 06 — Wave 1 Decision Gate

Source: `docs/research/new/ml_implementation_plan.txt`, Section 4 (lines 742–804)

Execute after Wave 1a + 1b are deployed and evaluated on real data.

---

### Task 1: Create decision gate evaluation script

**Read first:** `scripts/alpha/paper_trader_generic.py` (for how PnL is computed), `scripts/train_momentum.py` (for OOS metrics)

**Create:** `scripts/evaluate_wave1_gate.py` (~120 LOC)

CLI script that collects and evaluates all Wave 1 metrics:

```bash
python scripts/evaluate_wave1_gate.py --data-dir data/features --symbols BTC,ETH,SOL
```

Steps:
1. Load momentum training results (from `models/momentum_continuation/` metadata JSON)
2. Extract OOS AUC, OOS Sharpe proxy, OOS/IS ratio
3. Run `nat daily` output parsing for both algorithms across 3 symbols
4. Compute Spearman correlation of `alg_mc_signal` vs each existing winner signal
5. Check change_point_detector signal variance (std > 0.01)
6. Print decision matrix result:

```
WAVE 1 DECISION GATE
=====================
#1 Momentum OOS AUC:       0.54
#1 Momentum OOS Sharpe:    0.72
#1 Symbols positive:       2/3 (BTC, ETH)
#5 CPD signal variance:    0.034
#5 CPD vol correlation:    0.18

DECISION: CASE A — PROCEED TO WAVE 2 (full)
```

Decision logic:
- Case A: OOS Sharpe > 0.5 AND 2/3 symbols positive → proceed full
- Case B: OOS Sharpe > 0.5 BUT 1/3 symbols positive → proceed cautious (skip meta-labeling)
- Case C: OOS Sharpe 0.0–0.5 → investigate (print feature importance advice)
- Case D: OOS Sharpe < 0.0 → stop

Output `--json` for machine-readable results.

---

### Task 2: Write unit tests for gate logic

**Create:** `scripts/tests/test_wave1_gate.py` (~80 LOC)

```python
def test_case_a_full_proceed():
    """OOS Sharpe=0.7, symbols_positive=2 -> CASE_A"""

def test_case_b_cautious():
    """OOS Sharpe=0.6, symbols_positive=1 -> CASE_B"""

def test_case_c_investigate():
    """OOS Sharpe=0.3, symbols_positive=2 -> CASE_C"""

def test_case_d_stop():
    """OOS Sharpe=-0.2, symbols_positive=0 -> CASE_D"""

def test_cpd_independent_evaluation():
    """CPD with vol_corr=0.20 and variance=0.05 -> 'KEEP as gating signal'"""

def test_cpd_retire():
    """CPD with vol_corr=0.05 and variance=0.002 -> 'RETIRE'"""
```

Extract gate logic into a pure function `evaluate_gate(oos_sharpe, symbols_positive, cpd_variance, cpd_vol_corr)` for testability.

**Verification:** `cd scripts && python -m pytest tests/test_wave1_gate.py -v`

---

### Task 3: Create decision gate documentation

**Create:** `docs/research/new/ml_specs/GATE_WAVE1.md` (~60 lines)

Sections:
1. **Purpose:** hard gate between Wave 1 and Wave 2
2. **Metrics collected:** OOS Sharpe, OOS/IS ratio, per-symbol PnL, Spearman with winners, CPD variance/correlation
3. **Decision matrix:** table of 4 cases with conditions and actions
4. **CPD evaluation:** independent of ML gate (model-free algorithm)
5. **How to run:** `python scripts/evaluate_wave1_gate.py --data-dir data/features`
6. **Record keeping:** gate results saved to `data/research/wave1_gate_results.json`

---

### Task 4: Update README.md

**Modify:** `README.md`

In the ML Algorithms table (added in Task 5 of `01_gotchas_constraints.md`), ensure the Status column reflects actual results once the gate is evaluated. No change needed now — this task runs post-evaluation to update status from "Gated" to "Pass"/"Fail".

Add to the Testing section:

```markdown
# Wave 1 decision gate evaluation
python scripts/evaluate_wave1_gate.py --data-dir data/features --symbols BTC,ETH,SOL
```

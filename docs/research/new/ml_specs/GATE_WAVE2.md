# Wave 2 Decision Gate

Evaluate after all Wave 2 algorithms (#4 RSM, #2 MR, #3 Meta) are deployed and have OOS results. Determines whether to proceed to Wave 3.

---

## Metrics Collected

**Per-algorithm:**
- OOS Sharpe ratio (from walk-forward validation)
- OOS AUC (from model metadata)
- Whether the algorithm has positive OOS performance

**Cross-algorithm:**
- Pairwise Spearman correlation of primary signals
- Number of pairs with |rho| > 0.5 (flagged as correlated)
- Maximum absolute pairwise correlation

---

## Decision Matrix

| Case | Condition | Action |
|------|-----------|--------|
| A | 3+ algos positive OOS, no correlated pairs | Wave 3 full (#7 regime LightGBM + #6 KNN) |
| B | 2 algos positive, OR 3+ with correlated pairs | Wave 3 partial (#7 only) |
| C | 1 algo positive | STOP, deploy only that algorithm |
| D | 0 positive | STOP, ML adds no alpha |

---

## How to Run

```bash
# With real data (reads model metadata)
python scripts/evaluate_wave2_gate.py --data-dir data/features

# JSON output
python scripts/evaluate_wave2_gate.py --json

# Manual overrides (for testing)
python scripts/evaluate_wave2_gate.py --n-positive 3 --max-rho 0.28 --n-correlated 0
```

---

## Record Keeping

Results saved to `data/research/wave2_gate_results.json` with full per-algorithm breakdown. Exit code: 0 for CASE_A/B (proceed), 1 for CASE_C/D (stop).

---

## File Locations

| Purpose | Path |
|---------|------|
| Gate script | `scripts/evaluate_wave2_gate.py` |
| Unit tests | `scripts/tests/test_wave2_gate.py` |
| Results output | `data/research/wave2_gate_results.json` |

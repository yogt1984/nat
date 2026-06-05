# Wave 1 Decision Gate

Hard gate between Wave 1 (CPD + Momentum) and Wave 2 (RSM + Mean-Reversion + Meta-Labeling). Must be evaluated on real data before proceeding.

---

## Metrics Collected

| Metric | Source | Purpose |
|--------|--------|---------|
| OOS Sharpe | `models/momentum_continuation/*_metadata.json` | Primary quality measure |
| OOS AUC | Same metadata | Classification performance |
| OOS/IS ratio | Same metadata | Overfitting detection (want > 0.5) |
| Per-symbol PnL | `nat daily` output | Generalization across markets |
| Spearman with winners | Correlation of `alg_mc_signal` vs existing winner signals | Complementarity check |
| CPD signal variance | `std(alg_cpd_cusum_signal)` | Signal activity (dead signal = useless) |
| CPD vol correlation | Spearman(CPD signal, forward volatility) | Signal predictive power |

---

## Decision Matrix

| Case | OOS Sharpe | Symbols Positive | Action |
|------|-----------|-----------------|--------|
| **A** | >= 0.5 | >= 2/3 | Proceed to Wave 2 (full) |
| **B** | >= 0.5 | 1/3 | Proceed cautious (skip meta-labeling) |
| **C** | 0.0 – 0.5 | any | Investigate: tune features, check data quality |
| **D** | < 0.0 | any | Stop ML work — dataset does not support ML alpha |

## CPD Evaluation (Independent)

CPD is a model-free algorithm evaluated separately:

| CPD Variance | Vol Correlation | Decision |
|-------------|----------------|----------|
| > 0.01 | > 0.10 | KEEP as gating signal |
| otherwise | — | RETIRE |

---

## How to Run

```bash
# With real data
python scripts/evaluate_wave1_gate.py --data-dir data/features --symbols BTC,ETH,SOL

# JSON output
python scripts/evaluate_wave1_gate.py --data-dir data/features --json

# Manual override (for testing)
python scripts/evaluate_wave1_gate.py --oos-sharpe 0.7 --symbols-positive 2 --cpd-variance 0.03 --cpd-vol-corr 0.18
```

Exit code: 0 = proceed (Case A or B), 1 = hold/stop (Case C or D).

---

## Record Keeping

Gate results are saved to `data/research/wave1_gate_results.json` on every run. This file is the authoritative record of the Wave 1 gate decision and should be checked before starting Wave 2 work.

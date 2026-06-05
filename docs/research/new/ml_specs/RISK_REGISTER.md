# ML Risk Register

## Risk Table

| Risk | Severity | Mitigation | Monitoring |
|------|----------|------------|------------|
| Insufficient data | HIGH | Hard gate: bars >= 4000. Use num_leaves=15. | `check_data_sufficiency.py` |
| Overfitting | HIGH | Walk-forward, embargo=100, OOS/IS < 0.5 kills | Per-fold metrics in training |
| Data leakage | HIGH | Embargo, out-of-fold, forward-return alignment test | Purged K-fold tests |
| Sunk cost fallacy | HIGH | Hard decision gates after each wave | `evaluate_wave*_gate.py` |
| Model staleness | MEDIUM | Weekly retrain, IC decay monitoring | `ml_health_check.py` |
| Complexity creep | MEDIUM | Start simplest (LogReg > LightGBM, manual > HMM) | Code review |
| Correlation clustering | MEDIUM | Pairwise Spearman < 0.5 gate | `evaluate_wave2_gate.py` |
| Regime mismatch | MEDIUM | RSM as gate, per-regime monitoring | `ml_health_check.py` |

---

## Rollback Triggers

An ML algorithm should be rolled back or disabled if any of the following occur:

1. **Cumulative loss > 50 bps** over any 7-day window
2. **5 consecutive losing days** (daily PnL < 0 after costs)
3. **Pairwise correlation > 0.6** with another deployed algorithm (redundancy)
4. **OOS Sharpe < 0** over rolling 30-day window
5. **NaN rate > 20%** for more than 2 consecutive days (data issue)
6. **Model age > 30 days** without retraining (staleness)

---

## Rollback Procedures

### Immediate: Disable Algorithm
Remove from trading, keep code and models intact.
```bash
python scripts/ml_rollback.py disable <algo_name>
```
Re-enable after investigation:
```bash
python scripts/ml_rollback.py enable <algo_name>
```

### Model-Only: Revert to Previous Version
Archive the newest model, fall back to prior version.
```bash
python scripts/ml_rollback.py rollback-model <algo_name>
```
Verify active model:
```bash
python scripts/ml_rollback.py list-models <algo_name>
```

### Full: Remove Algorithm
Delete from `DAILY_ALGOS`, remove ALGO_CONFIG entry, archive model directory.
Only after confirming the algorithm cannot be fixed or retrained.

---

## Monitoring Tools

| Tool | Purpose | Frequency |
|------|---------|-----------|
| `ml_health_check.py` | Model age, NaN rate, Sharpe | Nightly |
| `check_deferred_triggers.py` | Deferred algo triggers | Monthly |
| `evaluate_wave1_gate.py` | Wave 1 decision gate | After Wave 1 |
| `evaluate_wave2_gate.py` | Wave 2 decision gate | After Wave 2 |
| `ml_rollback.py` | Disable/enable/rollback | On demand |
| `run_ml_verification.sh` | Full test suite | Pre-merge |

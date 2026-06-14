# Q2.1 — Revalidate Hierarchical Combiner + Ablation Study

**Phase**: Q2 — Validation & Signal Strengthening
**Priority**: 1 (first validation task)
**Status**: NOT STARTED
**Effort**: ~6h (revalidation + ablation)
**Depends on**: Q1.2 (7+ days data)

---

## Objective

Revalidate the 3-layer hierarchical signal combiner on 7+ days of data with proper walk-forward folds, and run ablation to quantify L2/L3 marginal contribution over L1 alone.

## Context

Current results (2 days, ~576 bars):
- BTC: IC +0.178, Sharpe +1.25
- ETH: IC +0.248, Sharpe +1.71
- SOL: IC +0.359, Sharpe +2.40

Concerns from the 10_6 report:
1. Monotonically increasing IC across folds (fold 0 < fold 1 < fold 2 < fold 3) — suspicious
2. L1 may dominate composite (active only ~43% of time)
3. SOL results likely inflated (shorter horizon, less data)
4. No live fill data to validate 11 bps RT cost assumption

With 7+ days we get 2000+ bars, allowing 4-fold walk-forward with 500+ bars per fold.

## Prerequisites

- Q1.2 complete (7+ consecutive days of clean data)
- Trained weights exist or training script runs cleanly

## Scope

**In scope**:
- Retrain hierarchical combiner per symbol on 7+ day dataset
- Walk-forward evaluation with 4-fold expanding window, 100-bar embargo
- Ablation: L1-only, L1+L2, L1+L3, full L1+L2+L3
- Cross-validate horizons: 30min, 1h, 2h, 5h, 12h per symbol
- Compare against existing deployable algorithms at matched horizons

**Out of scope**:
- Modifying the algorithm architecture
- Adding new features to any layer
- Live/paper trading

## Steps

1. Retrain on all 3 symbols:
   ```bash
   nat model train-hier --symbol BTC --dry-run
   nat model train-hier --symbol ETH --dry-run
   nat model train-hier --symbol SOL --dry-run
   ```
2. If dry-run results are reasonable, save weights:
   ```bash
   nat model train-hier --symbol BTC
   nat model train-hier --symbol ETH
   nat model train-hier --symbol SOL
   ```
3. Run ablation study — modify `run_batch()` to disable layers:
   - L1-only: set `l2_gated = 1.0`, `l3_scale = 1.0`
   - L1+L2: set `l3_scale = 1.0`
   - L1+L3: set `l2_gated = 1.0`
   - Full: no changes
4. Record IC, Sharpe, directional accuracy, L1 active rate for each variant
5. Cross-validate horizons (40, 60, 120, 360 bars) per symbol
6. Write results to `reports/hierarchical_revalidation_7day.md`

## Acceptance Criteria

- [ ] Walk-forward uses 4+ folds with 500+ bars each (not the previous 100-200)
- [ ] IC across folds does NOT monotonically increase (if it still does, flag as regime bias)
- [ ] Ablation quantifies L2/L3 delta: `|full_IC - L1_only_IC| > 0.02` (L2/L3 add value) OR ablation proves L1 dominates (simplify architecture)
- [ ] Cost-adjusted Sharpe > 0 on all 3 symbols at the IC-maximizing horizon
- [ ] SOL evaluated at matched horizon to BTC/ETH (60 bars, not 40)
- [ ] Results written to `reports/hierarchical_revalidation_7day.md`

## Testing / Verification

```bash
# 1. Training runs without error
nat model train-hier --symbol BTC --dry-run 2>&1 | tail -20

# 2. Evaluate on real data
nat algorithm evaluate --algorithm hierarchical_combiner --symbol BTC

# 3. Verify weights file updated
ls -la models/hierarchical_combiner/weights_*.json

# 4. Ablation comparison
python3 -c "
# After ablation results written to report:
# L1_IC, L1L2_IC, L1L3_IC, FULL_IC for each symbol
# Print delta table
print('Ablation deltas should show L2/L3 contribution')
"

# 5. Smoke test on real data
python3 scripts/alpha/paper_trader_generic.py \
  --algorithms hierarchical_combiner --symbols BTC --dry-run
```

## Key Files

- `scripts/algorithms/hierarchical_combiner.py` — algorithm implementation
- `scripts/train_hierarchical.py` — walk-forward training script
- `models/hierarchical_combiner/weights.json` — trained weights
- `models/hierarchical_combiner/weights_BTC.json` — per-symbol weights

## References

- Original 2-day report: `docs/research/new/10_6/hierarchical_combiner_report.md`
- Algorithm implementation: `scripts/algorithms/hierarchical_combiner.py`

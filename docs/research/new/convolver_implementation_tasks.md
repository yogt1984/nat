# Convolver Implementation Tasks

Step-by-step checklist for implementing and improving the convolver.
Execute in order. Each step includes the command, expected result, and go/no-go check.

---

## Phase 1: Fix the 0-Trade Bug

### 1. Change bar_agg from "mean" to "max"

**File:** `scripts/alpha/paper_trader_generic.py`, line 148

```python
# FROM:
"convolver": {
    "primary": "alg_conv_best_score",
    "polarity": "high_long",
    "bar_agg": "mean",
}

# TO:
"convolver": {
    "primary": "alg_conv_best_score",
    "polarity": "high_long",
    "bar_agg": "max",
}
```

**Why:** The convolver scores are 0.0 for ~95% of candles and spike briefly. Averaging over 5-min bars dilutes the spike below the z-score entry threshold.

### 2. Verify trades appear

```bash
nat daily --date 2026-06-01
```

- [ ] Convolver trades > 0
- [ ] If still 0 trades, check z-score distribution of max-aggregated scores

---

## Phase 2: Re-Run BTC Discovery

### 3. Back up existing kernels

```bash
cp models/convolver_kernels.npz models/convolver_kernels_20260524.npz
cp models/convolver_kernels.json models/convolver_kernels_20260524.json
```

### 4. Run BTC discovery with full data (~30 days)

```bash
cd /home/onat/nat
python scripts/analysis/convolver_discovery.py \
    --data-dir data/features \
    --symbol BTC \
    --candle-freq 60s \
    --window 20 \
    --save
```

**Expected:** ~43K candles, ~550 trap events (vs 308 in May-24), SVD threshold met for all event types.

### 5. Compare new kernels to old

```bash
python3 -c "
import json
old = json.load(open('models/convolver_kernels_20260524.json'))
new = json.load(open('models/convolver_kernels.json'))
print(f'Old: {old[\"n_kernels\"]} kernels, discovered {old[\"discovery_date\"]}')
print(f'New: {new[\"n_kernels\"]} kernels, discovered {new[\"discovery_date\"]}')
for k in new['kernels']:
    print(f'  {k[\"event_type\"]}:{k[\"channel\"]}:k{k[\"component_idx\"]} IC={k[\"ic\"]:.3f}')
"
```

- [ ] New kernel count >= 6
- [ ] ICs comparable or improved vs May-24
- [ ] Walk-forward decay ratio >= 0.50 for at least 1 kernel (stretch goal)

### 6. Re-run gauntlet with new kernels

```bash
nat daily --date 2026-06-02
```

- [ ] Convolver trades > 0 (confirms bar_agg fix + new kernels work)
- [ ] Record PnL per symbol

---

## Phase 3: Cross-Symbol Discovery

### 7. Run ETH discovery

```bash
python scripts/analysis/convolver_discovery.py \
    --data-dir data/features \
    --symbol ETH \
    --candle-freq 60s \
    --window 20 \
    --save

# Rename outputs to avoid overwriting BTC
mv models/convolver_kernels.npz models/convolver_kernels_ETH.npz
mv models/convolver_kernels.json models/convolver_kernels_ETH.json
```

### 8. Run SOL discovery

```bash
python scripts/analysis/convolver_discovery.py \
    --data-dir data/features \
    --symbol SOL \
    --candle-freq 60s \
    --window 20 \
    --save

mv models/convolver_kernels.npz models/convolver_kernels_SOL.npz
mv models/convolver_kernels.json models/convolver_kernels_SOL.json
```

### 9. Restore BTC kernels as primary

```bash
cp models/convolver_kernels_20260524.npz models/convolver_kernels.npz
cp models/convolver_kernels.json models/convolver_kernels.json
```

(Or copy the newly discovered BTC kernels back from Step 4's output.)

### 10. Test cross-symbol kernel universality

```bash
python3 -c "
import numpy as np

btc = np.load('models/convolver_kernels.npz')
eth = np.load('models/convolver_kernels_ETH.npz')
sol = np.load('models/convolver_kernels_SOL.npz')

for key in btc.files:
    for name, other in [('ETH', eth), ('SOL', sol)]:
        if key in other.files:
            cos = np.dot(btc[key], other[key]) / (
                np.linalg.norm(btc[key]) * np.linalg.norm(other[key]))
            print(f'{key}: BTC-{name} cos_sim = {cos:.3f}')
"
```

**Decision gate:**
- [ ] cos_sim > 0.80 across symbols → shapes are universal, proceed to pooled discovery
- [ ] cos_sim < 0.50 → shapes are symbol-specific, use per-symbol kernel libraries

---

## Phase 4: Integrate Convolver Features into ML Algorithms

### 11. Add convolver outputs to ML algorithm required_columns

For each ML algorithm (#1 Momentum, #5 Regime-Conditioned Entry):

```python
def required_columns(self) -> list[str]:
    return [
        # ... existing columns ...
        "alg_conv_best_score",
        "alg_conv_turtle_bull",
        "alg_conv_trap_bull",
    ]
```

No changes to `convolver.py` needed.

### 12. Add run_chain() to runner.py

The ML algorithms need convolver output as input. Add chained execution to `scripts/algorithms/runner.py`:

```python
def run_chain(self, df: pd.DataFrame, algorithms: list[MicrostructureAlgorithm]):
    """Run algorithms in dependency order, chaining outputs."""
    current_df = df.copy()
    results = []
    for algo in algorithms:
        result = AlgorithmRunner(algo).run_on_dataframe(current_df)
        current_df = pd.concat([current_df, result.features_df], axis=1)
        results.append(result)
    return results
```

**Depends on:** ML implementation plan Wave 0 (runner.py bar_level fix).

### 13. Verify ML algorithm uses convolver features

```bash
pytest scripts/tests/test_algorithm_smoke.py -k momentum
nat algorithm evaluate --algorithm momentum_continuation --symbol BTC
```

- [ ] No missing column errors
- [ ] Feature importance of alg_conv_* > 0.01 in LightGBM (if applicable)

---

## Phase 5: Ongoing Maintenance

### 14. Schedule quarterly re-discovery

Every 90 days (~130K new candles per symbol):

1. Back up current kernels
2. Re-run discovery for BTC, ETH, SOL
3. Compare new vs old kernels (cosine similarity)
4. Track metrics: n_survivors, mean IC, decay ratios, rolling rho
5. Update `models/convolver_kernels.*` if improved

### 15. Do NOT attempt multi-timeframe discovery

- 5min candles: feasible in ~5 months (not yet)
- 15min candles: requires 1.5+ years of data
- 1hr+ candles: infeasible (5+ years needed, market stationarity breaks first)

Stay at 60s resolution until 60s discovery is proven robust.

---

## Summary

| Phase | Steps | Time | Depends On |
|-------|-------|------|------------|
| 1. Fix 0-trade bug | 1-2 | 10 min | nothing |
| 2. BTC re-discovery | 3-6 | 1 hr compute | nothing (parallel with Phase 1) |
| 3. Cross-symbol | 7-10 | 2 hr compute | Phase 2 |
| 4. ML integration | 11-13 | 1-2 days code | ML plan Wave 0 |
| 5. Maintenance | 14-15 | ongoing | Phase 3 |

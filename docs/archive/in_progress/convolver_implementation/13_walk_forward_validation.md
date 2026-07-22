# Step 13: Walk-Forward Validation and Rolling Stability

## Context

Third overfitting safeguard. BH-FDR survivors must generalize to unseen data.
Walk-forward splits data temporally (not randomly) — training on older data,
testing on newer data — matching how the system operates in production.

## Walk-Forward IC

Split candle data at train_ratio (default 0.60):
- Train set: first 60% of candles (older data)
- Test set: last 40% of candles (newer data)

Run full discovery pipeline on train set. Evaluate surviving kernels on test set.

```
IC_decay_ratio = IC_OOS / IC_IS
```

File: `scripts/analysis/convolver_discovery.py`
Function: `walk_forward_validate(O, H, L, C, V, params, train_ratio, horizons)`

Returns: per-kernel IS IC, OOS IC, and decay ratio.

## Robustness Threshold

Decay ratio >= 0.50: kernel generalizes. The OOS IC is at least half the IS IC.

BTC discovery (May 2026): 0/6 survivors passed. Attributed to insufficient data
volume (12K candles = single market regime) rather than methodology failure.
High rolling stability (rho > 0.90) supports this interpretation.

## Rolling SVD Stability

Independent diagnostic. Measures whether the discovered shape is temporally
stable, regardless of IC.

```
For each rolling window of length roll_length, stride by stride candles:
  1. Run SVD on events within this window
  2. Extract first right singular vector v_1
  3. Compute cosine similarity between consecutive v_1 vectors

rho = mean(cosine_similarities)
```

File: `scripts/analysis/convolver_discovery.py`
Function: `rolling_stability(O, H, L, C, V, params, roll_length, stride)`

Returns: per-event-type mean and min stability.

## Stability Threshold

rho >= 0.70: shape is temporally stable (same pattern recurs across windows).

BTC results: turtle_bull rho=0.920, trap_bull rho=0.941.
Both far above threshold — shapes are real and consistent.

## Interpretation Matrix

| IC Decay | Stability | Interpretation |
|----------|-----------|----------------|
| >= 0.50 | >= 0.70 | Robust kernel, deploy |
| < 0.50 | >= 0.70 | Shape is real but needs more data for IC stability |
| >= 0.50 | < 0.70 | Accidental IC, shape is unstable |
| < 0.50 | < 0.70 | No signal |

BTC May 2026 is in row 2: shapes are real (high stability) but IC estimation
needs more data (only 12K candles, single regime).

## Dependencies

Prior steps: all discovery stages (steps 2-10)
Library: `numpy`

## Verification

```bash
# Full validation run (included in discovery pipeline with --save)
python scripts/analysis/convolver_discovery.py \
    --data-dir data/features --symbol BTC --save

# Check walk-forward results in output JSON
python3 -c "
import json
meta = json.load(open('models/convolver_kernels.json'))
for k in meta.get('kernels', []):
    print(f'{k[\"event_type\"]}:{k[\"channel\"]}:k{k[\"component_idx\"]} IC={k[\"ic\"]:.3f}')
"
```

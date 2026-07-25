# Q2.6 — Spannung Kalman Filter on Zero-Fee Pairs

**Phase**: Q2 — Validation & Signal Strengthening
**Priority**: 4 (exploratory, independent of main pipeline)
**Status**: NOT STARTED
**Effort**: ~8h
**Depends on**: Q1.2 (sufficient data)

---

## Objective

Implement a Kalman filter to extract the ultra-low frequency band (0.005–0.1 Hz) of L1 book imbalance and test profitability on zero-fee or low-fee trading pairs where the 0.2–0.4 bps per-trade edge becomes viable.

## Context

Spannung Phase D discovered that IC=0.45 is entirely concentrated in the ultra-low frequency band (periods 10–200s). The signal has OU dynamics with half-life 5–7s and dominant coherence with returns at 68s cycles. At taker fees (7 bps RT), the 0.2–0.4 bps per-trade edge is 20x too small. But on zero-fee pairs or with maker rebates, the edge may be extractable.

This task bridges both paths:
- **Quant**: If profitable, it's a new deployable strategy
- **PhD**: Kalman filter design for microstructure signals is a proposed research direction

## Prerequisites

- Q1.2 (7+ days data with 100ms imbalance data)
- Spectral analysis results from `docs/ideas/spannung.md` Phase D

## Scope

**In scope**:
- Implement Kalman filter tuned to ultra-low band (state: OU process, observation: L1 imbalance)
- OU parameter estimation: half-life 5–7s, diffusion from data
- Test on raw tick data (not bar-aggregated — aggregation destroys the signal per Phase C)
- Evaluate with `ent_book_shape` regime gating (Phase E)
- Cost model: zero-fee (Hyperliquid maker), 0.5 bps (maker + slippage), 1.5 bps (conservative)

**Out of scope**:
- Market-making strategy implementation (separate future task)
- Cross-exchange arbitrage
- Modifying the ingestor tick pipeline

## Steps

1. Estimate OU parameters from data:
   - Fit AR(1) to imbalance time series: `x(t+dt) = mu + phi*x(t) + sigma*eps`
   - Extract: `theta = -ln(phi)/dt`, `mu`, `sigma`
   - Validate: half-life should be 5–7s (consistent with Phase D)
2. Implement Kalman filter:
   - State: `x(t)` = OU process (true ultra-low-frequency imbalance)
   - Observation: `y(t)` = raw L1 imbalance (noisy)
   - Process noise: from OU diffusion parameter
   - Measurement noise: from high-frequency residual
3. Generate filtered signal: `x_filtered(t)` — the extracted ultra-low component
4. Backtest mean-reversion strategy:
   - Entry: `x_filtered` deviates > 1 sigma from mean
   - Exit: `x_filtered` returns to mean
   - Position size: proportional to deviation magnitude
5. Apply `ent_book_shape` gating:
   - Only trade when book shape entropy is in bottom quintile (Phase E finding)
6. Evaluate at 3 cost levels: 0, 0.5, 1.5 bps

## Acceptance Criteria

- [ ] OU half-life estimate falls in 4–8s range (consistent with Phase D)
- [ ] Kalman filtered signal has IC > 0.3 with 10s forward returns (vs 0.45 raw at tick level)
- [ ] Spectral analysis of filtered signal confirms energy concentrated below 0.1 Hz
- [ ] With `ent_book_shape` gating: IC > 0.45 (matching Phase E lift)
- [ ] At 0 bps cost: Sharpe > 2.0 (strong signal in ideal conditions)
- [ ] At 0.5 bps cost: Sharpe > 0 (marginally profitable with maker)
- [ ] At 1.5 bps cost: document exactly how much edge is lost
- [ ] Results written to `reports/spannung_kalman_results.md`

## Testing / Verification

```bash
# 1. OU parameter estimation
python3 -c "
# Fit AR(1), extract OU params
import pandas as pd, numpy as np
df = pd.read_parquet('data/features/2026-06-17/*.parquet')
imb = df['imbalance_qty_l1_last'].dropna()
phi = imb.autocorr(lag=1)
dt = 0.1  # 100ms
theta = -np.log(abs(phi)) / dt
half_life = np.log(2) / theta
print(f'OU theta={theta:.3f}, half-life={half_life:.2f}s')
assert 4 < half_life < 8, f'Half-life {half_life}s outside expected range'
"

# 2. Filtered signal spectral check
python3 scripts/analysis/spectral_analysis.py \
  --signal kalman_filtered --band 0.005,0.1

# 3. Backtest at multiple cost levels
for cost in 0.0 0.5 1.5; do
  python3 scripts/backtest/spannung_kalman.py \
    --symbol BTC --cost-bps $cost --output reports/kalman_cost_${cost}.json
done

# 4. Regime gating
python3 scripts/backtest/spannung_kalman.py \
  --symbol BTC --cost-bps 0.5 --regime-gate ent_book_shape
```

## Key Files

- `scripts/backtest/spannung_kalman.py` — Kalman filter + backtest (new)
- `scripts/analysis/spectral_analysis.py` — spectral validation
- `reports/spannung_kalman_results.md` — output report

## References

- Spannung Phase D–E: `docs/ideas/spannung.md`
- PhD brief: `docs/ideas/spannung_phd_brief.md`
- OU dynamics: half-life 5–7s, coherence at 68s (0.015 Hz)

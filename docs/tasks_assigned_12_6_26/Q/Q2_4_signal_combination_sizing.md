# Q2.4/Q2.5 — Feature Combination, Cost-Aware Sizing & Walk-Forward Validation

**Phase**: Q2 — Validation & Signal Strengthening
**Priority**: 3 (follows alpha screen)
**Status**: NOT STARTED
**Effort**: ~12h
**Depends on**: Q2.3 (alpha screen passes G1)

---

## Objective

Combine top features from the alpha screen into a composite signal, apply cost-aware position sizing, and validate out-of-sample with walk-forward + deflated Sharpe. Covers ROADMAP Steps 2, 3, and 4.

## Context

Step 1 (Q2.3) identifies which features predict returns. Steps 2-4 convert those features into a tradeable signal:
- Step 2: Combine survivors into a single composite signal
- Step 3: Only trade when expected gain > cost (the fix for Phase 1's cost problem)
- Step 4: Walk-forward validation with deflated Sharpe (corrects for multiple testing)

The previous Phase 1 had +4.18% edge but lost money because every signal change triggered a trade. Cost-aware sizing is the critical missing piece.

## Prerequisites

- Q2.3 passes Gate G1 (features with significant IC exist)
- `scripts/backtest/walk_forward.py` exists
- Deflated Sharpe implementation exists: `compute_deflated_sharpe()`

## Scope

**In scope** (3 ROADMAP steps):
1. Feature combination: IC-weighted z-score composite (Step 2)
2. Cost-aware position sizing: trade filter based on expected gain vs cost (Step 3)
3. Walk-forward validation with deflated Sharpe (Step 4)

**Out of scope**:
- Regime conditioning (Step 5 — task Q3.2)
- Multi-frequency integration (Step 6 — task Q3.2)
- Paper trading (Step 8 — task Q3.4)

## Steps

### Step 2 — Feature Combination
1. Take top-N features from screener (N ≤ 20)
2. Standardize each: `z_i(t) = (f_i(t) - rolling_mean) / rolling_std`
3. Compute correlation matrix, iteratively drop features with |corr| > 0.8
4. Start with equal-weight combination: `z(t) = mean(z_i(t))`
5. Normalize to [-1, +1] via cross-sectional rank
6. Evaluate Gate G2

### Step 3 — Cost-Aware Position Sizing
7. Estimate expected gain: `E[gain] = |z(t)-z(t-1)| * IC * vol(r) * sqrt(horizon)`
8. Only trade when `E[gain] > cost * 1.5` (safety margin)
9. Position size: `p(t) = clip(z(t) * scale_factor, -1, +1)`
10. First 30 days: 50% of max position (ramp-up)
11. Evaluate Gate G3

### Step 4 — Walk-Forward Validation
12. Run walk-forward: 5 splits, 600-bar embargo, OOS/IS threshold 0.7
13. Run combinatorial purged CV: 5 splits, 2 test splits
14. Compute deflated Sharpe with N = number of features screened in Step 1
15. Evaluate Gate G4

## Acceptance Criteria

### Gate G2 (Feature Combination):
- [ ] Combined IC > 0.8 × max(individual ICs)
- [ ] Combined turnover < 2× average individual turnover
- [ ] Combined signal not > 0.9 correlated with any single feature

### Gate G3 (Cost-Aware Sizing):
- [ ] Trade count drops by 50%+ vs unfiltered signal
- [ ] Net return INCREASES vs unfiltered (filter removes bad trades)
- [ ] Mean holding time > 2 hours

### Gate G4 (Walk-Forward — ALL must pass):
- [ ] OOS Sharpe > 0.5
- [ ] OOS/IS ratio > 0.7
- [ ] Deflated Sharpe p-value < 0.05
- [ ] Max drawdown < 5%
- [ ] Minimum 30 trades in OOS
- [ ] Profit factor > 1.2

## Testing / Verification

```bash
# 1. Feature combination
python3 scripts/alpha/combiner.py --input reports/alpha_screen.json --symbol BTC

# 2. Cost-aware sizing
python3 scripts/alpha/position.py --signal reports/combined_signal_BTC.json

# 3. Walk-forward validation
python3 scripts/backtest/walk_forward.py \
  --strategy combined_alpha --symbol BTC \
  --n-splits 5 --embargo 600

# 4. Gate checks
python3 -c "
import json
with open('reports/walk_forward_BTC.json') as f:
    wf = json.load(f)
checks = {
    'OOS Sharpe > 0.5': wf['oos_sharpe'] > 0.5,
    'OOS/IS > 0.7': wf['oos_is_ratio'] > 0.7,
    'Deflated p < 0.05': wf['deflated_p'] < 0.05,
    'Max DD < 5%': wf['max_drawdown'] < 0.05,
    'Trades >= 30': wf['n_trades_oos'] >= 30,
    'PF > 1.2': wf['profit_factor'] > 1.2,
}
for name, passed in checks.items():
    status = 'PASS' if passed else 'FAIL'
    print(f'  G4 {name}: {status}')
print(f'G4 overall: {\"PASS\" if all(checks.values()) else \"FAIL\"}')"
```

## Key Files

- `scripts/alpha/combiner.py` — feature combination (new per ROADMAP)
- `scripts/alpha/position.py` — cost-aware sizing (new per ROADMAP)
- `scripts/backtest/walk_forward.py` — existing walk-forward engine
- `reports/alpha_screen.json` — input from Q2.3

## References

- ROADMAP Steps 2-4: `docs/research/ROADMAP.md` lines 99–228
- Deflated Sharpe: Bailey & Lopez de Prado (2014)

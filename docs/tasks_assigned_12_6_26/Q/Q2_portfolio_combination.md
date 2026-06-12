# Q2.8 — Portfolio Combination of Deployable Algorithms

**Phase**: Q2 — Validation & Signal Strengthening
**Priority**: 3 (can run in parallel with Q2.3)
**Status**: NOT STARTED
**Effort**: ~6h
**Depends on**: Q1.1 (clean features)

---

## Objective

Evaluate the portfolio-level Sharpe of the 4 existing deployable algorithms (jump_detector, optimal_entry, funding_reversion, 3f_liquidity) combined via risk-parity weighting, exploiting their documented cross-correlation <0.35.

## Context

The 4 deployable algorithms have individually strong results at 100min horizon:

| Algorithm | BTC Sharpe | ETH Sharpe | SOL Sharpe |
|-----------|-----------|-----------|-----------|
| jump_detector | 1.6 | 6.2 | 6.2 |
| optimal_entry | 1.1 | 5.2 | — |
| funding_reversion | 0.4 | 6.0 | 1.7 |
| 3f_liquidity | 12.1 | 4.3 | 1.8 |

Cross-algorithm signal correlations are <0.35 (from `reports/signal_correlation.json`). Per-symbol specialization exists: 3f dominates BTC, jump/optimal dominate ETH/SOL. A properly combined portfolio should achieve Sharpe above any individual strategy.

## Prerequisites

- Q1.1 (clean features — all algorithms need their input features)
- All 4 algorithms registered and passing smoke tests

## Scope

**In scope**:
- Run all 4 algorithms on latest 30-day OOS data per symbol
- Compute per-symbol equity curves
- Risk-parity weighting: `w_i = (1/vol_i) / sum(1/vol_j)`
- Correlation-adjusted weights (reduce combined weight if corr > 0.8)
- Portfolio Sharpe, max drawdown, profit factor
- Compare portfolio vs best individual

**Out of scope**:
- Adding hierarchical combiner (insufficient data — Q2.1 first)
- Live trading execution
- Dynamic weight rebalancing (use static weights initially)

## Steps

1. Run all 4 algorithms on 30-day OOS:
   ```bash
   for alg in jump_detector optimal_entry funding_reversion 3f_liquidity; do
     nat algorithm evaluate --algorithm $alg --symbol BTC --oos-days 30
     nat algorithm evaluate --algorithm $alg --symbol ETH --oos-days 30
     nat algorithm evaluate --algorithm $alg --symbol SOL --oos-days 30
   done
   ```
2. Extract per-algorithm, per-symbol equity curves
3. Compute daily return correlation matrix (4×4 per symbol)
4. Compute risk-parity weights per symbol
5. Combine: portfolio return = sum(w_i * r_i)
6. Compute portfolio metrics: Sharpe, max DD, profit factor, Calmar ratio
7. Compare: portfolio Sharpe vs max(individual Sharpes)
8. Write `reports/portfolio_combination.json`

## Acceptance Criteria

- [ ] Portfolio Sharpe > max(individual Sharpes) on at least 2 of 3 symbols
- [ ] Portfolio max drawdown < 80% of worst individual max drawdown
- [ ] Cross-algorithm correlation matrix confirms <0.35 on current data (replicates prior finding)
- [ ] Risk-parity weights sum to 1.0 per symbol, no single weight > 0.5
- [ ] All 4 algorithms × 3 symbols evaluated (12 runs)
- [ ] Results documented in `reports/portfolio_combination.json` with per-symbol breakdown

## Testing / Verification

```bash
# 1. Individual algorithm evaluation
nat algorithm evaluate --algorithm 3f_liquidity --symbol BTC

# 2. Correlation check
python3 -c "
import json
with open('reports/signal_correlation.json') as f:
    corr = json.load(f)
for pair, val in corr.items():
    status = 'OK' if abs(val) < 0.35 else 'HIGH'
    print(f'{pair}: {val:.3f} [{status}]')
"

# 3. Portfolio metrics
python3 -c "
import json
with open('reports/portfolio_combination.json') as f:
    pf = json.load(f)
for sym in ['BTC', 'ETH', 'SOL']:
    data = pf[sym]
    print(f'{sym}: Portfolio Sharpe={data[\"portfolio_sharpe\"]:.2f}, '
          f'Best Individual={data[\"best_individual_sharpe\"]:.2f}, '
          f'Max DD={data[\"max_drawdown\"]:.2%}')
    assert data['portfolio_sharpe'] >= data['best_individual_sharpe'] * 0.8
"
```

## Key Files

- `scripts/alpha/portfolio.py` — portfolio combination (new per ROADMAP Step 7)
- `scripts/algorithms/` — individual algorithm implementations
- `reports/signal_correlation.json` — existing correlation data
- `reports/portfolio_combination.json` — output

## References

- Signal correlation data: `reports/signal_correlation.json`
- Situation analysis §I: `docs/tasks_assigned_12_6_26/situation_analysis.md`
- ROADMAP Step 7: `docs/research/ROADMAP.md` lines 299–319

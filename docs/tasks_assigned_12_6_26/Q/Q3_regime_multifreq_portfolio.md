# Q3.2/Q3.3 — Regime Conditioning, Multi-Freq Integration & Portfolio Assembly

**Phase**: Q3 — Paper Trading
**Priority**: 2 (enhances paper trading signal)
**Status**: NOT STARTED
**Effort**: ~10h
**Depends on**: Q2.5 (G4 pass), cluster validation results

---

## Objective

Apply regime conditioning (ROADMAP Step 5), multi-frequency integration (Step 6), and portfolio assembly (Step 7) to the validated signal before paper trading. These steps are optional enhancements — skip if they don't improve OOS metrics.

## Context

Steps 5–7 are conditional improvements to the base pipeline:
- **Regime conditioning**: Only useful if profiling produced real clusters (Q1+Q2 pass from cluster analysis). Current status: only orderflow@5min passes Q3 (KW p=3e-6, weak effect eta²=0.057).
- **Multi-frequency**: Combines macro direction (daily SMA) with micro timing (intraday signal).
- **Portfolio**: Risk-parity across symbols with drawdown controls.

All three have quality gates. If a step doesn't improve OOS metrics, it's skipped entirely.

## Prerequisites

- Q2.5 passes G4 (walk-forward validated base signal)
- Cluster analysis results: only proceed with regime conditioning if clusters are valid
- Macro data available for multi-frequency integration

## Scope

**In scope**:
- Step 5: Re-run alpha screening within each regime, compare IC
- Step 6: Macro filter (SMA 50/200) gates micro signal direction
- Step 7: Risk-parity weights, correlation adjustment, drawdown control

**Out of scope**:
- New regime detection methods
- Alternative macro indicators beyond SMA
- Dynamic weight rebalancing (static weights for initial paper trading)

## Steps

### Step 5 — Regime Conditioning
1. Check prerequisite: do valid regime labels exist? (silhouette > 0.25, ARI > 0.6)
2. If yes: re-run alpha screening within each regime separately
3. Compare `IC_within_regime` vs `IC_global`
4. If any regime has IC > 1.5× global: use regime-specific weights
5. Evaluate Gate G5

### Step 6 — Multi-Frequency Integration
6. Compute macro filter: `SMA(50) > SMA(200) AND price > SMA(50)`
7. Gate micro signal: long only when macro bullish, short only when bearish
8. Add profit-sensitive exit: tighten threshold when unrealized PnL > 2× cost
9. Evaluate Gate G6

### Step 7 — Portfolio Assembly
10. Run pipeline independently for BTC, ETH, SOL
11. Compute daily return correlation matrix
12. Risk-parity weights: `w_i = (1/vol_i) / sum(1/vol_j)`
13. Drawdown control: reduce to 50% when portfolio DD > 2%, restore at 1%
14. Evaluate Gate G7

## Acceptance Criteria

### Gate G5 (Regime — optional):
- [ ] At least 1 regime with IC > 1.5× global IC
- [ ] Regime-conditioned OOS Sharpe > global OOS Sharpe
- [ ] If neither passes: regime conditioning skipped (documented)

### Gate G6 (Multi-Frequency):
- [ ] Composite Sharpe > max(macro_only, micro_only)
- [ ] Composite max DD < min(macro_only_DD, micro_only_DD)
- [ ] If composite doesn't beat both: use best standalone

### Gate G7 (Portfolio):
- [ ] Portfolio Sharpe > max(individual symbol Sharpes)
- [ ] Portfolio max DD < 80% of worst individual max DD
- [ ] If portfolio doesn't improve: trade best symbol only

### Overall:
- [ ] Each gate evaluated independently with pass/fail documented
- [ ] Skipped steps clearly marked with reason
- [ ] Final signal (with all passing enhancements) ready for paper trader

## Testing / Verification

```bash
# 1. Regime conditioning (if clusters valid)
python3 scripts/alpha/regime_filter.py --symbol BTC \
  --input reports/alpha_screen.json

# 2. Multi-frequency integration
python3 scripts/alpha/multi_freq.py --symbol BTC \
  --micro-signal reports/combined_signal_BTC.json

# 3. Portfolio assembly
python3 scripts/alpha/portfolio.py \
  --symbols BTC,ETH,SOL --method risk_parity

# 4. Gate evaluation
python3 -c "
# Check all gates
gates = {'G5': False, 'G6': False, 'G7': False}  # load from results
for gate, passed in gates.items():
    print(f'{gate}: {\"PASS\" if passed else \"SKIP\"}')"
```

## Key Files

- `scripts/alpha/regime_filter.py` — regime conditioning (new per ROADMAP)
- `scripts/alpha/multi_freq.py` — multi-frequency integration (new per ROADMAP)
- `scripts/alpha/portfolio.py` — portfolio assembly (new per ROADMAP)

## References

- ROADMAP Steps 5–7: `docs/research/ROADMAP.md` lines 232–319
- Cluster analysis: `docs/tasks_assigned_12_6_26/situation_analysis.md` §IV

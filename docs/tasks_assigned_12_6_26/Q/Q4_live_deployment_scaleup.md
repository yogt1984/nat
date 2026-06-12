# Q4 — Live Deployment & Scale-Up

**Phase**: Q4 — Live Deployment
**Priority**: Final phase
**Status**: NOT STARTED
**Effort**: Ongoing (4+ months)
**Depends on**: Q3.4 (G8 pass — 14-day paper trading validated)

---

## Objective

Deploy validated algorithms with live capital on Hyperliquid, scaling from 1% to 25% over 4+ months with continuous monitoring and kill switch protection.

## Context

This is the terminal phase of the quant path. Every preceding gate (G1–G8) must have passed. The scale-up schedule is deliberately slow — each tier requires the previous tier's results to hold before increasing capital.

## Prerequisites

- G8 pass (14-day paper trading)
- Kill switch infrastructure operational (Q3.6)
- Cost model validated against paper trading fills (Q3.5)
- Hyperliquid API credentials configured
- Sufficient capital in Hyperliquid account

## Scope

**In scope**:
- Phased capital deployment: 1% → 5% → 10% → 25%
- Maker-only orders (lower cost, better fill tracking)
- Continuous IC decay monitoring
- Monthly pipeline re-run
- Live execution quality metrics (fill rate, slippage, latency)

**Out of scope**:
- Taker orders (maker-only in first deployment)
- New strategy development during deployment
- Cross-exchange deployment

## Steps & Scale-Up Schedule

### Tier 1: 1% Capital, Maker Only (Weeks 1–2)
1. Deploy with 1% of total capital
2. Maker orders only — observe fill rates
3. Measure: actual slippage vs model, fill rate by algorithm, latency
4. Compare live P&L to paper trading P&L for same period
5. **Gate**: live/paper ratio > 0.5 AND no kill switch trigger

### Tier 2: 5% Capital (Weeks 3–4)
6. Scale to 5% if Tier 1 holds
7. Monitor for market impact at increased size
8. Track: bid-ask bounce rate, queue position degradation
9. **Gate**: Sharpe holds AND no single-day loss > 0.5%

### Tier 3: 10% Capital (Months 2–3)
10. Scale to 10%
11. First monthly pipeline re-run from Step 1 (detect feature decay)
12. Re-evaluate: is the alpha still present on fresh data?
13. **Gate**: sustained positive Sharpe AND monthly re-run confirms signal

### Tier 4: 25% Max Capital (Month 4+)
14. Scale to 25% maximum — never exceed
15. Continuous monitoring becomes steady-state
16. Monthly pipeline re-runs mandatory
17. Quarterly strategy review: add/remove algorithms based on IC trends

## Acceptance Criteria

### Per-Tier Gates:
- [ ] **Tier 1→2**: Live/paper Sharpe ratio > 0.5, no kill switch triggers in 2 weeks
- [ ] **Tier 2→3**: No single-day loss > 0.5%, Sharpe > 0 over 2 weeks
- [ ] **Tier 3→4**: Monthly pipeline re-run confirms signal, sustained positive Sharpe for 2 months
- [ ] **Ongoing**: No kill switch trigger at any tier halts scale-up for the halt period

### Monitoring requirements:
- [ ] Live P&L dashboard updated every 15 minutes
- [ ] Execution quality metrics logged: fill rate, slippage, latency per algorithm
- [ ] Rolling 7-day IC computed and compared to training IC daily
- [ ] Kill switch daemon running independently at all times
- [ ] Telegram alerts for: kill switch trigger, daily P&L summary, IC decay warning

### Kill switch responses:
- [ ] Daily loss > 1%: auto-halt 24h, Telegram alert, no manual intervention needed
- [ ] Weekly DD > 2%: halt, require manual review and `nat risk resume`
- [ ] Monthly DD > 5%: kill all positions, re-run full pipeline from Step 1
- [ ] IC < 0 for 5 days: halt, investigate decay cause

## Testing / Verification

```bash
# 1. Pre-deployment checklist
python3 -c "
checks = {
    'G8 passed': True,  # from paper trading report
    'Kill switch active': True,  # nat risk status
    'Cost model validated': True,  # from Q3.5
    'Telegram alerts working': True,  # nat risk test-alert
    'API credentials set': True,  # env vars
}
for check, passed in checks.items():
    print(f'  {check}: {\"PASS\" if passed else \"FAIL\"}')"

# 2. Kill switch is running
nat risk status

# 3. Live monitoring
nat status  # Shows live P&L, positions, IC

# 4. Monthly pipeline re-run
nat pipeline start --from-step 1

# 5. Execution quality
python3 -c "
import json
with open('data/live/execution_quality.json') as f:
    eq = json.load(f)
for alg in eq:
    print(f'{alg[\"algorithm\"]}: fill_rate={alg[\"fill_rate\"]:.1%}, '
          f'slippage={alg[\"slippage_bps\"]:.2f}bps, '
          f'latency={alg[\"latency_ms\"]:.0f}ms')
"
```

## Key Files

- `scripts/live/executor.py` — live order execution (new)
- `scripts/risk/kill_switch.py` — risk monitoring daemon
- `data/live/` — live trade logs, execution quality metrics
- `config/costs.toml` — cost model (validated by paper trading)

## References

- ROADMAP Step 9: `docs/research/ROADMAP.md` lines 362–376
- Kill switch spec: Q3.6 task
- Paper trading results: Q3.4 task

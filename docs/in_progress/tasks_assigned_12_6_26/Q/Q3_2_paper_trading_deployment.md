# Q3.1/Q3.4 — Paper Trading Deployment & 14-Day Validation

**Phase**: Q3 — Paper Trading
**Priority**: 1 (gate to live deployment)
**Status**: NOT STARTED
**Effort**: ~10h setup + 14 days runtime
**Depends on**: Q2.5 (G4 pass — walk-forward validated)

---

## Objective

Deploy paper trading for the top algorithms, run for 14 consecutive days with daily reconciliation, and measure actual signal decay and execution model accuracy. This is Gate G8 — the final gate before live capital.

## Context

Every backtest overestimates live performance. Paper trading reveals:
- Fill rate issues (limit orders not getting filled when expected)
- Latency and infrastructure failures
- Signal IC decay vs training data
- Divergence between assumed and actual execution costs

No live capital is deployed until paper trading runs for 14 clean days and passes G8.

## Prerequisites

- Q2.5 passes G4 (walk-forward validated, deflated Sharpe significant)
- Paper trader script exists: `scripts/alpha/paper_trader_generic.py`
- Kill switch infrastructure (Q3.6) built in parallel

## Scope

**In scope**:
- Deploy paper trader for validated algorithms
- 15-minute signal computation cycle
- Daily PnL reconciliation: paper vs backtest prediction
- Rolling 7-day IC decay monitoring
- Infrastructure stability logging (errors, restarts, data gaps)
- Fill assumption validation (conditional IC on hypothetical fills)

**Out of scope**:
- Actual order execution (this is paper only)
- Live capital risk
- Strategy modification during the 14-day window

## Steps

1. Configure paper trader:
   ```bash
   python scripts/alpha/paper_trader_generic.py \
     --algorithms jump_detector,optimal_entry,funding_reversion,3f_liquidity \
     --symbols BTC,ETH,SOL \
     --interval 15min \
     --output-dir data/paper_trades/
   ```
2. Set up daily reconciliation cron:
   - Compute paper PnL for the day
   - Compare vs what walk-forward backtest predicted for same period
   - Log divergence ratio: `paper_sharpe / backtest_sharpe`
3. Monitor IC decay:
   - Rolling 7-day IC of live signal vs realized forward returns
   - Alert if live IC < 50% of training IC for 3 consecutive days
4. Log all infrastructure events: data gaps, computation errors, missed cycles
5. After 14 days: compile results into `reports/paper_trading_14day.md`

## Acceptance Criteria

### Gate G8 (ALL must pass):
- [ ] Paper Sharpe within 2x of backtest Sharpe (expect some degradation, but not catastrophic)
- [ ] No single day loss > 2%
- [ ] Signal IC has not decayed > 50% from training IC
- [ ] Infrastructure ran without errors for 14 consecutive days (no missed computation cycles)
- [ ] Mean daily PnL > 0 (net positive over 14 days)

### Data quality:
- [ ] `data/paper_trades/` contains 14 daily JSON files
- [ ] Each file has entries for all algorithms × all symbols
- [ ] Reconciliation log shows backtest-to-paper ratio for each day
- [ ] IC decay timeseries plotted and saved

### If G8 fails:
- [ ] Paper much worse than backtest → document execution model errors
- [ ] Single day > 2% loss → identify which algorithm/symbol caused it
- [ ] IC decay → measure how many days until IC crosses zero
- [ ] Infrastructure failure → list all error types and frequencies

## Testing / Verification

```bash
# 1. Dry-run paper trader (1 cycle)
python scripts/alpha/paper_trader_generic.py \
  --algorithms jump_detector --symbols BTC --dry-run

# 2. Check output format
python3 -c "
import json
from pathlib import Path
files = sorted(Path('data/paper_trades').glob('*.json'))
print(f'Paper trade files: {len(files)}')
if files:
    with open(files[-1]) as f:
        data = json.load(f)
    print(f'Latest: {files[-1].name}, entries: {len(data)}')
"

# 3. Daily reconciliation check
python3 -c "
import json
with open('data/paper_trades/reconciliation.json') as f:
    rec = json.load(f)
for day in rec:
    ratio = day['paper_sharpe'] / max(day['backtest_sharpe'], 0.01)
    status = 'OK' if ratio > 0.5 else 'WARN'
    print(f'{day[\"date\"]}: paper/backtest={ratio:.2f} [{status}]')
"

# 4. IC decay monitoring
python3 -c "
import json
with open('data/paper_trades/ic_decay.json') as f:
    decay = json.load(f)
for sym in ['BTC', 'ETH', 'SOL']:
    latest = decay[sym][-1]
    ratio = latest['live_ic'] / max(latest['train_ic'], 0.01)
    print(f'{sym}: live_IC={latest[\"live_ic\"]:.3f}, train_IC={latest[\"train_ic\"]:.3f}, ratio={ratio:.2f}')
"
```

## Key Files

- `scripts/alpha/paper_trader_generic.py` — paper trading engine
- `data/paper_trades/YYYY-MM-DD.json` — daily trade logs
- `data/paper_trades/reconciliation.json` — daily backtest comparison
- `data/paper_trades/ic_decay.json` — IC monitoring timeseries
- `reports/paper_trading_14day.md` — final report

## References

- ROADMAP Step 8: `docs/research/ROADMAP.md` lines 326–358
- Gate G8 specification in ROADMAP

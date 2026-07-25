# Q3.6 — Kill Switch Infrastructure

**Phase**: Q3 — Paper Trading
**Priority**: 1 (must be ready before live deployment)
**Status**: NOT STARTED
**Effort**: ~6h
**Depends on**: None (can start early)

---

## Objective

Build automated risk controls that halt trading when predefined loss thresholds are breached. Required before any live capital is deployed (Q4).

## Context

Kill switches are non-negotiable for live deployment. They must:
- Operate independently of the trading logic (separate process/service)
- Be impossible to accidentally disable
- Alert via Telegram when triggered
- Log all triggers with full context for post-mortem

Thresholds from ROADMAP Step 9:
- Daily loss > 1%: halt for 24 hours
- Weekly drawdown > 2%: halt and review pipeline
- Monthly drawdown > 5%: kill strategy, re-run full pipeline
- IC drops below 0 for 5 consecutive days: halt

## Prerequisites

- Telegram alerting configured (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)
- Paper trading infrastructure (Q3.1) for testing kill switches

## Scope

**In scope**:
- Kill switch daemon: independent process monitoring PnL
- 4 threshold levels with different responses (halt/review/kill)
- Telegram alerts on trigger
- Manual override mechanism (for intentional re-enable)
- Integration with paper trading for testing
- Prometheus metrics for kill switch state

**Out of scope**:
- Position liquidation logic (exchange handles this)
- Margin monitoring (exchange-level)
- Strategy modification on trigger (just halt)

## Steps

1. Create `scripts/risk/kill_switch.py`:
   - Daemon that polls PnL at 1-minute intervals
   - Reads from paper trade logs or live position tracking
   - Computes rolling daily, weekly, monthly PnL
   - Computes rolling 5-day IC
2. Implement threshold checks:
   ```python
   THRESHOLDS = {
       'daily_loss': {'limit': -0.01, 'action': 'halt_24h'},
       'weekly_dd': {'limit': -0.02, 'action': 'halt_review'},
       'monthly_dd': {'limit': -0.05, 'action': 'kill_strategy'},
       'ic_decay': {'limit': 0.0, 'days': 5, 'action': 'halt'},
   }
   ```
3. Implement halt mechanism:
   - Write `data/risk/halt_state.json` with trigger reason and resume time
   - Trading scripts check this file before executing any trade
   - `halt_24h`: auto-resume after 24 hours
   - `halt_review`: requires manual `nat risk resume` command
   - `kill_strategy`: requires full pipeline re-run before resume
4. Telegram alerts on trigger:
   - Message: threshold name, current value, limit, action taken
5. Add Prometheus metrics:
   - `nat_kill_switch_active{level="daily"}` — 0 or 1
   - `nat_kill_switch_triggers_total{level="daily"}` — counter
6. Test on paper trading data before live deployment

## Acceptance Criteria

- [ ] Kill switch daemon runs as independent process (not embedded in trading logic)
- [ ] All 4 thresholds trigger correctly when breached (tested with synthetic data)
- [ ] Telegram alert received within 60s of trigger
- [ ] `data/risk/halt_state.json` written on trigger with: timestamp, reason, threshold, value, resume_at
- [ ] Trading scripts refuse to execute when halt is active
- [ ] `nat risk status` shows current halt state
- [ ] `nat risk resume` clears `halt_review` state (requires confirmation)
- [ ] `nat risk resume` refuses to clear `kill_strategy` without pipeline re-run
- [ ] Kill switch cannot be accidentally disabled (no simple flag to skip)
- [ ] Prometheus metrics exported

## Testing / Verification

```bash
# 1. Unit test with synthetic PnL data
python3 -c "
from scripts.risk.kill_switch import KillSwitch
ks = KillSwitch(test_mode=True)
# Simulate daily loss > 1%
ks.update_pnl(-0.015)
assert ks.is_halted(), 'Should be halted after 1.5% daily loss'
print('Daily loss trigger: PASS')

# Check halt state file
import json
with open('data/risk/halt_state.json') as f:
    state = json.load(f)
assert state['reason'] == 'daily_loss'
assert state['action'] == 'halt_24h'
print('Halt state file: PASS')
"

# 2. Telegram alert test
python3 scripts/risk/kill_switch.py --test-alert

# 3. Trading script respects halt
python3 -c "
from scripts.risk.kill_switch import check_halt
halted, reason = check_halt()
print(f'Halted: {halted}, Reason: {reason}')
"

# 4. Resume mechanism
nat risk status
nat risk resume --confirm

# 5. Integration with paper trader
python3 scripts/alpha/paper_trader_generic.py \
  --algorithms jump_detector --symbols BTC --dry-run \
  --with-kill-switch
```

## Key Files

- `scripts/risk/kill_switch.py` — kill switch daemon (new)
- `scripts/risk/__init__.py` — risk module init (new)
- `data/risk/halt_state.json` — halt state persistence
- `nat` CLI — add `risk status` and `risk resume` commands

## References

- ROADMAP Step 9 kill switches: `docs/research/ROADMAP.md` lines 365–376
- Telegram config: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` env vars

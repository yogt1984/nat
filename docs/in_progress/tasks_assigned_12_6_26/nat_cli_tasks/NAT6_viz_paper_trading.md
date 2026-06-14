# NAT6 — `nat viz paper` — Paper Trading Analysis

**Priority**: 6 (~4h)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz library foundation), Q3 paper trading data

---

## Objective

Implement `nat viz paper` — terminal and chart views of paper trading P&L, fill analysis, signal-to-execution divergence, and IC decay monitoring. Essential for Gate G8 evaluation.

## Context

Paper trading (Q3.4) produces daily JSON logs in `data/paper_trades/`. Currently, analyzing these requires manual Python scripting. The existing `nat trade-viz` generates static PNGs via `scripts/trade_visualize.py` but only for snapshot views. This command provides a richer, real-time-capable analysis.

## Scope

**In scope**:
- Cumulative P&L curve (terminal sparkline + PNG chart)
- Per-algorithm P&L breakdown
- Daily reconciliation: paper vs backtest Sharpe ratio
- IC decay monitoring: rolling 7-day IC vs training IC
- Fill analysis: trade count, avg holding time, win rate
- G8 gate status check (all 5 criteria)

**Out of scope**:
- Live order execution
- Modifying paper trading logic
- Portfolio-level view (that's `nat viz portfolio`)

## Implementation

### `scripts/viz/trades.py` (new)

```python
def render_terminal(symbol: str = None, date: str = None,
                    data_dir: str = None) -> Panel:
    """Rich panel showing paper trading summary."""
    from .common import load_paper_trades
    trades = load_paper_trades(symbol, date)

    # Sections:
    # 1. Summary: total P&L, Sharpe, max DD, win rate, trade count
    # 2. Per-algorithm breakdown table
    # 3. Daily P&L sparkline
    # 4. IC decay: current vs training
    # 5. G8 gate checklist with PASS/FAIL
    ...

def render_figure(symbol: str = None, date: str = None,
                  data_dir: str = None) -> plt.Figure:
    """4-panel matplotlib: cumulative P&L, daily P&L bars, IC decay, drawdown."""
    ...

def check_g8(trades: list[dict], training_ic: dict) -> dict:
    """Evaluate Gate G8 criteria. Returns dict of {criterion: bool}."""
    return {
        'paper_sharpe_within_2x': ...,
        'no_day_gt_2pct_loss': ...,
        'ic_not_decayed_50pct': ...,
        'infra_14_clean_days': ...,
        'mean_daily_pnl_positive': ...,
    }
```

### Handler in `nat`

```python
def cmd_viz_paper(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.trades import render_terminal, render_figure, export_data
    if _json_mode(args):
        _output(export_data(_sym(args), getattr(args, 'date', None)), args)
    elif getattr(args, 'output', None):
        fig = render_figure(_sym(args), getattr(args, 'date', None))
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        _p("+", G, f"Saved to {args.output}")
    else:
        from rich.console import Console
        Console().print(render_terminal(_sym(args), getattr(args, 'date', None)))
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/trades.py` | Create new |
| `nat` | Add `cmd_viz_paper()` handler + register subparser |

## Acceptance Criteria

- [ ] `nat viz paper` shows paper trading summary for all symbols
- [ ] `nat viz paper --symbol BTC` filters to BTC only
- [ ] Output shows: cumulative P&L, Sharpe, max drawdown, win rate, trade count
- [ ] Per-algorithm breakdown table with individual P&L
- [ ] IC decay section: rolling 7-day IC vs training IC with trend indicator
- [ ] G8 gate checklist: each criterion shows PASS/FAIL with current vs threshold value
- [ ] `--json` exports all metrics as structured JSON
- [ ] `--output paper.png` saves 4-panel matplotlib chart
- [ ] Graceful message if no paper trade data exists: "No paper trades found in data/paper_trades/"
- [ ] `--date 2026-06-20` shows specific date's results

## Testing / Verification

```bash
# 1. Basic paper trading view
nat viz paper

# 2. Symbol filter
nat viz paper --symbol BTC

# 3. JSON export
nat viz paper --json

# 4. PNG export
nat viz paper --output /tmp/paper.png

# 5. G8 gate check
nat viz paper --json | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'g8_gates' in data:
    for gate, passed in data['g8_gates'].items():
        print(f'  {gate}: {\"PASS\" if passed else \"FAIL\"}')"

# 6. No data graceful handling
# (if paper trades don't exist yet)
nat viz paper 2>&1 | grep -i "no paper"
```

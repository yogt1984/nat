# NAT7 — `nat viz portfolio` — Portfolio Dashboard

**Priority**: 7 (~8h)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz library foundation), Q3 paper trading

---

## Objective

Implement `nat viz portfolio` — a multi-tab terminal dashboard showing portfolio-level P&L, per-algorithm exposure, cross-symbol correlation, and risk status. This is the primary monitoring view during paper and live trading.

## Context

During Q3 (paper trading) and Q4 (live deployment), the operator needs a single view that answers: "how is the portfolio performing, what's my risk, and should I intervene?" Currently this requires checking multiple commands and dashboards. The existing `nat monitor` covers system health but not trading state.

## Scope

**In scope**:
- Tab 1 — P&L: per-algorithm cumulative P&L, Sharpe, win rate, max drawdown
- Tab 2 — Exposure: per-symbol position size, risk-parity weights, net exposure
- Tab 3 — Correlation: cross-algorithm signal correlation matrix
- Tab 4 — Risk: kill switch status, IC decay, drawdown gauges
- `--live` auto-refresh mode
- `--json` data export

**Out of scope**:
- Order execution controls
- Position modification
- Historical portfolio analytics (use backtest tools)

## Implementation

### `scripts/viz/portfolio.py` (new)

```python
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns

def render_pnl_tab(hours: float = 24) -> Table:
    """Tab 1: Per-algorithm P&L table."""
    # Load paper/live trade data
    # Compute: cumulative PnL, daily Sharpe, win rate, max DD, trade count
    # Color: green for positive, red for negative
    ...

def render_exposure_tab(hours: float = 24) -> Table:
    """Tab 2: Per-symbol exposure and weights."""
    # Current positions per symbol
    # Risk-parity weights
    # Net long/short/flat status
    ...

def render_correlation_tab(hours: float = 24) -> Table:
    """Tab 3: Cross-algorithm correlation matrix."""
    # Load recent signals from all algorithms
    # Compute pairwise Spearman correlation
    # Color: green < 0.3, yellow 0.3-0.6, red > 0.6
    ...

def render_risk_tab() -> Panel:
    """Tab 4: Risk status panel."""
    # Kill switch: active/inactive per level
    # IC decay: current vs training per algorithm
    # Drawdown: daily/weekly/monthly gauges
    # Overall status: OK / WARNING / HALT
    ...

def render_terminal(hours: float = 24, tab: int = 1) -> Layout:
    """Full dashboard layout with selected tab."""
    tabs = {1: render_pnl_tab, 2: render_exposure_tab,
            3: render_correlation_tab, 4: render_risk_tab}
    return tabs.get(tab, render_pnl_tab)(hours)
```

### Handler in `nat`

```python
def cmd_viz_portfolio(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.portfolio import render_terminal, export_data
    tab = getattr(args, 'tab', 1)
    if _json_mode(args):
        _output(export_data(args.hours), args)
    elif getattr(args, 'live', False):
        from viz.terminal import live_refresh
        live_refresh(lambda: render_terminal(args.hours, tab))
    else:
        from rich.console import Console
        Console().print(render_terminal(args.hours, tab))
```

### Parser registration

```python
p = vs.add_parser('portfolio', help='Portfolio P&L and risk dashboard')
_add_common(p)
p.add_argument('--tab', type=int, default=1, choices=[1,2,3,4],
               help='Tab: 1=P&L, 2=Exposure, 3=Correlation, 4=Risk')
p.set_defaults(func=cmd_viz_portfolio)
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/portfolio.py` | Create new |
| `nat` | Add `cmd_viz_portfolio()` handler + register subparser |

## Acceptance Criteria

- [ ] `nat viz portfolio` shows Tab 1 (P&L) by default
- [ ] `nat viz portfolio --tab 2` shows exposure tab
- [ ] `nat viz portfolio --tab 4` shows risk status with kill switch state
- [ ] Tab 1 shows per-algorithm: cumulative P&L, Sharpe, win rate, max DD
- [ ] Tab 2 shows per-symbol: position, weight, net exposure
- [ ] Tab 3 shows correlation matrix with color coding (<0.3 green, >0.6 red)
- [ ] Tab 4 shows: kill switch status, IC decay per algo, drawdown gauges
- [ ] `--live` auto-refreshes every 5s
- [ ] `--json` exports all 4 tabs as structured JSON
- [ ] Graceful degradation: if no trading data exists, shows "No active positions"
- [ ] Risk tab reads `data/risk/halt_state.json` for kill switch status

## Testing / Verification

```bash
# 1. Default tab (P&L)
nat viz portfolio

# 2. All tabs
for tab in 1 2 3 4; do
  echo "=== Tab $tab ==="
  nat viz portfolio --tab $tab 2>&1 | head -10
done

# 3. JSON export
nat viz portfolio --json | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Keys:', list(data.keys()))"

# 4. Live mode (manual — Ctrl+C to exit)
nat viz portfolio --live --tab 1

# 5. No data handling
nat viz portfolio 2>&1  # Should show graceful message if no trades
```

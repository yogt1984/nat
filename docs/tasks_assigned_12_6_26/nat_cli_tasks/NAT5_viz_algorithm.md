# NAT5 — `nat viz algorithm` — Decision Trace

**Priority**: 5 (~8h)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz library foundation)

---

## Objective

Implement `nat viz algorithm <name>` — a terminal view showing an algorithm's signal timeline, entry/exit decisions, input features at trigger points, and P&L per trade. Answers: "why did the algorithm trade (or not trade) here?"

## Context

Currently, understanding why an algorithm triggered requires manually loading Parquet data, running the algorithm's `run_batch()`, and cross-referencing with paper trade logs. There is no single command for this. During paper trading (Q3) and live deployment (Q4), this will be essential for debugging and monitoring.

## Scope

**In scope**:
- Signal timeline: algorithm composite output over time with entry/exit markers
- Input feature values at each trigger point
- P&L per trade (if paper/live trade data exists)
- Current state: active/inactive, direction, confidence
- `--last 2h` time scoping
- Terminal rendering via rich
- `--png` for matplotlib chart export

**Out of scope**:
- Modifying algorithm implementations
- Real-time order execution
- Multiple algorithm comparison (that's `nat viz compare`)

## Implementation

### `scripts/viz/algorithms.py` (new)

```python
def render_terminal(algorithm: str, symbol: str, hours: float = 4,
                    data_dir: str = None) -> Panel:
    """Rich panel showing algorithm decision trace."""
    from scripts.algorithms.registry import get_algorithm
    from .common import load_features
    from .terminal import sparkline

    # Load features and run algorithm
    df = load_features(symbol, hours, data_dir)
    alg = get_algorithm(algorithm)
    signals = alg.run_batch(df)

    # Extract entries/exits from signal crossings
    composite_col = [c for c in signals.columns if 'composite' in c or c.endswith('_signal')]
    ...

    # Build output sections:
    # 1. Status line: current state
    # 2. Signal sparkline: Unicode timeline
    # 3. Last N signals table: time, direction, confidence, P&L, hold time
    # 4. Feature values at last trigger
    ...

def render_figure(algorithm: str, symbol: str, hours: float = 4,
                  data_dir: str = None) -> plt.Figure:
    """Matplotlib figure: price + signal overlay + entry/exit markers."""
    ...

def export_data(algorithm: str, symbol: str, hours: float = 4,
                data_dir: str = None) -> dict:
    """Raw data for --json output."""
    ...
```

### Handler in `nat`

```python
def cmd_viz_algorithm(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.algorithms import render_terminal, render_figure, export_data
    if _json_mode(args):
        _output(export_data(args.name, _sym(args), args.hours), args)
    elif getattr(args, 'output', None):
        fig = render_figure(args.name, _sym(args), args.hours)
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        _p("+", G, f"Saved to {args.output}")
    else:
        from rich.console import Console
        Console().print(render_terminal(args.name, _sym(args), args.hours))
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/algorithms.py` | Create new |
| `nat` | Add `cmd_viz_algorithm()` handler + register subparser with positional `name` arg |

## Acceptance Criteria

- [ ] `nat viz algorithm jump_detector --symbol BTC` shows signal trace for last 4h
- [ ] Output includes: current status (ACTIVE/INACTIVE), last signal details
- [ ] Signal timeline shows entry/exit markers with Unicode visualization
- [ ] Feature values at last trigger point are listed with threshold comparison
- [ ] Last 5 signals displayed in a table: time, direction, confidence, P&L, hold time
- [ ] `--hours 2` limits the lookback window
- [ ] `nat viz algorithm hierarchical_combiner --symbol ETH` works for multi-output algorithms
- [ ] `--json` exports structured signal data
- [ ] `--output chart.png` saves matplotlib chart with price + signal overlay
- [ ] Error message if algorithm name is invalid: "Unknown algorithm: <name>. Use `nat algorithm list`."
- [ ] Handles case where algorithm produced no signals in the window: "No signals in last Xh"

## Testing / Verification

```bash
# 1. Basic algorithm trace
nat viz algorithm jump_detector --symbol BTC

# 2. Different algorithm
nat viz algorithm 3f_liquidity --symbol BTC --hours 2

# 3. Hierarchical (multi-output)
nat viz algorithm hierarchical_combiner --symbol ETH

# 4. JSON export
nat viz algorithm jump_detector --symbol BTC --json

# 5. PNG export
nat viz algorithm jump_detector --symbol BTC --output /tmp/jump_trace.png
ls -la /tmp/jump_trace.png

# 6. Invalid algorithm
nat viz algorithm nonexistent_algo --symbol BTC
# Should print error with suggestion to use nat algorithm list

# 7. All registered algorithms work
for alg in $(nat algorithm list --json 2>/dev/null | python3 -c "
import sys,json; [print(a['name']) for a in json.load(sys.stdin).get('algorithms',[])]
" 2>/dev/null); do
  nat viz algorithm "$alg" --symbol BTC --hours 1 --json > /dev/null 2>&1 && echo "$alg: OK" || echo "$alg: FAIL"
done
```

# NAT4 — `nat viz features` — Live Feature Dashboard

**Priority**: 4 (~6h)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz library foundation)

---

## Objective

Implement `nat viz features` — a rich terminal table showing all features with current value, z-score, NaN rate, IC, and sparkline. Answers: "what is the pipeline producing right now?"

## Context

Currently there is no single command to see what the feature pipeline is outputting. Users must write ad-hoc Python to load Parquet and inspect values. This is the most frequently needed inspection: "are features flowing? which ones are dead? which ones have signal?"

## Scope

**In scope**:
- Terminal table via `rich` showing per-feature stats
- Columns: name, current value, z-score, NaN%, IC@5min, sparkline
- Filters: `--category`, `--top N`, `--sort`, `--alive-only`
- `--live` auto-refresh mode (5s interval)
- `--json` data export
- `--output <path>` PNG export via matplotlib

**Out of scope**:
- Single feature deep-dive (separate command: `nat viz feature <name>`)
- Historical IC analysis (use `nat signal test` for that)

## Implementation

### `scripts/viz/features.py` — extend existing

```python
def render_terminal(symbol: str, hours: float = 4, data_dir: str = None,
                    category: str = None, top: int = None,
                    sort: str = 'ic', alive_only: bool = False) -> Table:
    """Rich table of all features with stats."""
    from .common import load_features
    from .terminal import sparkline, ic_color, zscore_color

    df = load_features(symbol, hours, data_dir)

    # Compute per-feature stats
    stats = []
    for col in df.columns:
        if col in ('timestamp', 'symbol'):
            continue
        series = df[col]
        nan_pct = series.isna().mean()
        if alive_only and nan_pct > 0.99:
            continue
        clean = series.dropna()
        current = clean.iloc[-1] if len(clean) > 0 else float('nan')
        z = (current - clean.mean()) / clean.std() if len(clean) > 20 else float('nan')
        # IC: rank correlation with 5min forward return
        ic = _compute_ic(df, col, horizon_bars=1) if nan_pct < 0.5 else float('nan')
        spark = sparkline(clean.tail(60).tolist())
        stats.append({
            'name': col, 'value': current, 'zscore': z,
            'nan_pct': nan_pct, 'ic': ic, 'spark': spark,
        })

    # Filter by category
    if category:
        stats = [s for s in stats if category.lower() in s['name'].lower()]

    # Sort
    sort_key = {'ic': lambda s: abs(s['ic']) if s['ic']==s['ic'] else -1,
                'zscore': lambda s: abs(s['zscore']) if s['zscore']==s['zscore'] else -1,
                'nan': lambda s: s['nan_pct'],
                'name': lambda s: s['name']}
    stats.sort(key=sort_key.get(sort, sort_key['ic']), reverse=(sort != 'name'))

    if top:
        stats = stats[:top]

    # Build rich table
    table = Table(title=f"NAT Features — {symbol} (last {hours}h, {len(df)} rows)")
    table.add_column("Feature", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("NaN%", justify="right")
    table.add_column("IC@5m", justify="right")
    table.add_column("Spark")

    for s in stats:
        # ... format and color each column, add row
        pass

    # Summary row
    alive = sum(1 for s in stats if s['nan_pct'] < 0.99)
    dead = sum(1 for s in stats if s['nan_pct'] >= 0.99)
    table.caption = f"Active: {alive} | Dead: {dead}"

    return table
```

### Handler in `nat`

```python
def cmd_viz_features(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.features import render_terminal, export_data
    if _json_mode(args):
        _output(export_data(symbol=_sym(args), hours=args.hours), args)
    elif getattr(args, 'live', False):
        from viz.terminal import live_refresh
        live_refresh(lambda: render_terminal(
            symbol=_sym(args), hours=args.hours,
            category=args.category, top=args.top,
            sort=args.sort, alive_only=args.alive_only))
    else:
        from rich.console import Console
        Console().print(render_terminal(
            symbol=_sym(args), hours=args.hours,
            category=args.category, top=args.top,
            sort=args.sort, alive_only=args.alive_only))
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/features.py` | Extend with `render_terminal()`, `export_data()` |
| `nat` | Add `cmd_viz_features()` handler + register subparser |

## Acceptance Criteria

- [ ] `nat viz features --symbol BTC` prints a rich table with all non-NaN features
- [ ] Table shows: name, current value, z-score, NaN%, IC@5min, sparkline
- [ ] `--top 20` limits output to 20 rows sorted by |IC|
- [ ] `--category entropy` filters to entropy-related features only
- [ ] `--sort zscore` sorts by |z-score| descending
- [ ] `--alive-only` hides features with 100% NaN
- [ ] `--live` auto-refreshes every 5s (Ctrl+C to exit)
- [ ] `nat viz features --json` outputs structured JSON with all stats
- [ ] Dead features shown in dim/red, high-IC features in green
- [ ] Summary line shows active/dead/constant counts
- [ ] Runs in < 3s for 1 hour of data (~720 rows × 191 columns)

## Testing / Verification

```bash
# 1. Basic feature table
nat viz features --symbol BTC --hours 1

# 2. Filtered view
nat viz features --symbol BTC --top 10 --sort ic

# 3. Category filter
nat viz features --symbol ETH --category imbalance

# 4. Alive only
nat viz features --symbol BTC --alive-only

# 5. JSON export
nat viz features --symbol BTC --json | python3 -m json.tool | head -20

# 6. Live mode (manual test — Ctrl+C to exit)
nat viz features --symbol BTC --live

# 7. Performance
time nat viz features --symbol BTC --hours 1
# Should complete in < 3 seconds
```

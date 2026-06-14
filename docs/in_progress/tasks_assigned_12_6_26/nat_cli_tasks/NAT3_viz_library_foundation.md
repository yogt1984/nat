# NAT3 — Visualization Library Foundation

**Priority**: 3 (~6h)
**Status**: NOT STARTED
**Depends on**: None

---

## Objective

Build the shared foundation for `nat viz` commands: data loading (`common.py`), terminal rendering primitives (`terminal.py`), and the `nat viz` command group registration. All subsequent NAT4–NAT8 tasks depend on this.

## Context

Existing viz code is scattered across `scripts/viz/features.py`, `scripts/viz/correlations.py`, `scripts/visualize_scanner.py`, `scripts/trade_visualize.py`, etc. Each loads data independently, uses different styling, and has no shared rendering primitives. The foundation task creates the shared layer so viz commands are consistent.

## Scope

**In scope**:
- `scripts/viz/common.py` — data loading, theme, shared utilities
- `scripts/viz/terminal.py` — rich-based rendering primitives (sparklines, IC bars, tables)
- `nat viz` command group registration in `nat` script with `cmd_viz_help()`
- Common argument pattern (`--symbol`, `--hours`, `--date`, `--output`, `--live`, `--json`)

**Out of scope**:
- Individual viz commands (NAT4–NAT8)
- Modifying existing `nat visualize *` commands
- New dependencies (use only `rich`, `matplotlib`, `pandas` — already installed)

## Implementation

### Step 1 — `scripts/viz/common.py` (~2h)

```python
"""Shared data loading and styling for nat viz commands."""
from pathlib import Path
import pandas as pd

DATA_DEFAULT = Path(__file__).resolve().parent.parent.parent / "data" / "features"

THEME = {
    'positive': 'green', 'negative': 'red', 'neutral': 'dim',
    'header': 'bold cyan', 'warning': 'bold yellow', 'muted': 'dim white',
}

def load_features(symbol: str, hours: float = 4,
                  data_dir: Path = DATA_DEFAULT,
                  date: str | None = None) -> pd.DataFrame:
    """Load latest N hours of feature data from Parquet files.

    Scans date directories, reads Parquet files, filters by symbol
    and time range. Returns DataFrame with timestamp index.
    """
    ...

def load_algorithm_signals(algorithm: str, symbol: str,
                           hours: float = 4) -> pd.DataFrame:
    """Load algorithm signal output from paper trade logs or evaluation."""
    ...

def load_paper_trades(symbol: str, date: str | None = None) -> list[dict]:
    """Load paper trade logs from data/paper_trades/."""
    ...

def latest_parquet_dir(data_dir: Path = DATA_DEFAULT) -> Path:
    """Return most recent date directory in data/features/."""
    ...

def feature_categories() -> dict[str, list[str]]:
    """Return feature names grouped by category (entropy, flow, etc.)."""
    ...
```

### Step 2 — `scripts/viz/terminal.py` (~2h)

```python
"""Rich-based terminal rendering primitives for nat viz."""
from rich.console import Console
from rich.table import Table
from rich.live import Live
import time

console = Console()

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 12) -> str:
    """Unicode sparkline from numeric values."""
    if not values or all(v != v for v in values):  # all NaN
        return "—" * width
    clean = [v for v in values if v == v]
    if not clean:
        return "—" * width
    mn, mx = min(clean), max(clean)
    rng = mx - mn if mx != mn else 1.0
    chars = []
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        v = values[i]
        if v != v:
            chars.append("·")
        else:
            idx = int((v - mn) / rng * 7)
            chars.append(SPARK_CHARS[min(idx, 7)])
    return "".join(chars[:width])

def ic_color(ic: float) -> str:
    """Return rich color tag for IC value."""
    if abs(ic) > 0.10: return "bold green" if ic > 0 else "bold red"
    if abs(ic) > 0.05: return "green" if ic > 0 else "red"
    return "dim"

def zscore_color(z: float) -> str:
    """Return rich color tag for z-score."""
    if abs(z) > 2.0: return "bold yellow"
    if abs(z) > 1.0: return "cyan"
    return "dim"

def live_refresh(render_fn, interval: float = 5.0):
    """Auto-refreshing terminal display. Ctrl+C to exit."""
    with Live(render_fn(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(render_fn())
        except KeyboardInterrupt:
            pass
```

### Step 3 — Register `nat viz` group in `nat` (~1h)

In `build_parser()`:

```python
# ── viz ──
viz_p = sub.add_parser('viz', help='Visualization & inspection')
viz_p.set_defaults(func=cmd_viz_help)
vs = viz_p.add_subparsers(dest='subcmd')

def _viz_common(p):
    p.add_argument('--symbol', default='BTC', help='Symbol (default: BTC)')
    p.add_argument('--hours', type=float, default=4, help='Lookback hours (default: 4)')
    p.add_argument('--date', default=None, help='Specific date (YYYY-MM-DD)')
    p.add_argument('--data', default=str(DATA_DEFAULT), help='Data directory')
    p.add_argument('--output', default=None, help='Save to PNG/HTML file')
    p.add_argument('--live', action='store_true', help='Auto-refresh every 5s')
```

Add `cmd_viz_help()`:

```python
def cmd_viz_help(args=None):
    print(f"""
  {BOLD}Visualization & Inspection{W}

    nat viz features           Live feature dashboard (terminal table)
    nat viz feature <name>     Single feature deep-dive (time series + IC)
    nat viz algorithm <name>   Algorithm decision trace + P&L
    nat viz portfolio          Portfolio P&L, risk, and correlation
    nat viz paper              Paper trading analysis
    nat viz signals            Active signals heatmap
    nat viz spectral <name>    PSD + coherence for a feature
    nat viz regime             Regime state and transitions
    nat viz correlation        Cross-algorithm correlation matrix
    nat viz data-quality       NaN rates, gaps, schema health
    nat viz compare <a> <b>    Side-by-side algorithm comparison

  Common flags:
    --symbol BTC     Symbol (default: BTC)
    --hours 4        Lookback window (default: 4h)
    --live           Auto-refresh every 5s
    --output <path>  Save PNG/HTML instead of terminal
    --json           Export raw data as JSON

  Start with: nat viz features --symbol BTC
    """)
```

### Step 4 — Update `scripts/viz/__init__.py` (~30min)

```python
"""NAT visualization library.

Usage from nat CLI:
    nat viz features --symbol BTC
    nat viz algorithm jump_detector --symbol ETH

Usage from Python:
    from scripts.viz import features
    features.render_terminal(symbol='BTC', hours=1)
"""
from . import common, terminal
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/common.py` | Extend (exists but incomplete) |
| `scripts/viz/terminal.py` | Create new |
| `scripts/viz/__init__.py` | Create/update |
| `nat` | Add `nat viz` group + `cmd_viz_help()` in `build_parser()` |

## Acceptance Criteria

- [ ] `nat viz` prints curated help listing all available viz subcommands
- [ ] `nat viz -h` shows argparse help with common flags
- [ ] `scripts/viz/common.py` has `load_features()` that returns a DataFrame from Parquet
- [ ] `scripts/viz/terminal.py` has `sparkline()`, `ic_color()`, `live_refresh()`
- [ ] `sparkline([1,2,3,4,5])` returns a Unicode bar string
- [ ] `load_features('BTC', hours=1)` loads data without error on current data
- [ ] No new dependencies introduced (only `rich`, `matplotlib`, `pandas`)
- [ ] Existing `nat visualize *` commands still work unchanged

## Testing / Verification

```bash
# 1. nat viz shows help
nat viz
# Should print curated subcommand list

# 2. Common data loading
python3 -c "
from scripts.viz.common import load_features, latest_parquet_dir
d = latest_parquet_dir()
print(f'Latest data: {d}')
df = load_features('BTC', hours=1)
print(f'Loaded: {len(df)} rows, {len(df.columns)} cols')
assert len(df) > 0
"

# 3. Terminal primitives
python3 -c "
from scripts.viz.terminal import sparkline, ic_color
print(sparkline([1, 2, 3, 5, 8, 5, 3, 2, 1]))
print(ic_color(0.15))
print(ic_color(-0.03))
"

# 4. Existing viz unchanged
nat visualize scan --symbol BTC 2>&1 | head -2
```

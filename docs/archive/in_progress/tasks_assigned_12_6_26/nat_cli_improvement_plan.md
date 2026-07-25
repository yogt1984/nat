# NAT CLI Improvement Plan

**Date**: 2026-06-12
**Scope**: `nat` CLI tool analysis, improvement proposals, and unified visualization design

---

## I. Current State Analysis

### Architecture

- **File**: `/home/onat/nat/nat` — single 5,113-line Python script
- **Framework**: `argparse` with nested subparsers
- **Commands**: 216 handler functions (`cmd_*`), ~251 distinct command invocations
- **Dispatch**: `build_parser()` at line 3602 → `sub.add_subparsers()` → `args.func(args)`
- **Helpers**: `_sh()` (capture), `_exec()` (foreground), `_py()` (Python script), `_cargo()` (Rust build)
- **Output**: `_json_mode(args)` + `_output()` for dual human/JSON rendering
- **Help**: `nat help` (250-line curated text), `nat commands [--json]` (flat list), `nat -h` (argparse default)

### Command Groups (53 groups)

| Category | Groups | Commands |
|----------|--------|----------|
| System | start, stop, status, health, log, monitor, dashboard | ~10 |
| Config | config (show/get/validate) | 3 |
| Data | data (validate/explore/schema), fetch | 5 |
| Signal | signal (test/test-all), screen, scan | 4 |
| Analysis | profile, spannung (5 variants), macro, report | 9 |
| Alpha pipeline | alpha (13 subcommands) | 13 |
| Algorithm | algorithm (evaluate/list/config), kalman (2) | 5 |
| Backtest | backtest (8 subcommands) | 8 |
| Model | model (train/train-gmm/train-hier/list/score/serve) | 6 |
| Cluster | cluster (6 subcommands) | 6 |
| Validation | validate (skeptical/regression) | 2 |
| EAMM | eamm (run/regime/backtest) | 3 |
| Agent | agent/mf-agent/macro-agent/meta-agent (8 each) | 32+ |
| Discovery | discovery (start/once/status/stop), it-engine | 7 |
| Audit | audit (aggregate/sweep) | 2 |
| Experiment | experiment (6 subcommands) | 6 |
| Pipeline | pipeline (6 subcommands) | 6 |
| OOS | daily, oos30, gauntlet (4), tournament (8) | 14 |
| Visualization | visualize (7), trade-viz, 15m viz | 9 |
| Build/Deploy | build (7), run (3), test (18), docker (7), deploy (2) | 37 |
| Optimization | swarm (5), evolve (5) | 10 |
| Special | alg1 (3), 15m (3), help, commands, math, reports (3) | 12 |

### Existing Visualization Stack

| Component | Type | Framework | Port | Via |
|-----------|------|-----------|------|-----|
| Ingestor dashboard | Web (WebSocket) | Axum (Rust) | 8080 | `nat run serve` |
| Pipeline dashboard | Web (HTTP) | Python stdlib | 8050 | `nat dashboard` |
| Agent dashboard | Web (HTTP) | Python stdlib | 8060 | `nat agent dashboard` |
| Research frontend | Web (React) | Next.js | 3001 | `docker-compose up web` |
| Grafana metrics | Web | Grafana | 3002 | `docker-compose up grafana` |
| Terminal monitor | TUI | rich | — | `nat monitor [--tab N]` |
| OOS terminal | TUI | rich | — | `python scripts/oos_terminal.py` |
| Scanner plots | Static PNG | Matplotlib | — | `nat visualize scan` |
| Cluster plots | Static PNG | Matplotlib | — | `nat visualize cluster` |
| Trade plots | Static PNG | Matplotlib | — | `nat trade-viz` |
| Feature viz | Static PNG | Matplotlib | — | `scripts/viz/features.py` |

---

## II. Strengths

1. **Consistent dispatch pattern** — every command follows `cmd_xxx(args)` → `_py()` or `_sh()`, making it predictable to add commands
2. **JSON output on most commands** — `--json` flag enables AI agent and automation integration
3. **`nat commands --json`** — machine-readable command catalog for discovery
4. **Comprehensive `nat help`** — 250 lines of curated, grouped help text with math formulas
5. **Math epilogs** — complex commands include mathematical definitions (IC, Kelly, Spearman) in `--help` output
6. **Rich terminal output** — `nat monitor` and OOS terminal use `rich` for polished TUI
7. **Deep command hierarchy** — `nat alpha pipeline-start` reads naturally and organizes well

---

## III. Weaknesses

### A. Discoverability & Help

| Problem | Example | Impact |
|---------|---------|--------|
| **Inconsistent per-command help** | `nat alpha combine -h` has rich math epilog; `nat agent start -h` is a bare one-liner | Users can't predict which commands have useful help |
| **Group-level help is argparse default** | `nat alpha` prints generic argparse usage instead of the curated text from `nat help` | User has to know to run `nat help` to get the good help |
| **No maturity indicators** | `nat oos30` (proven, daily use) looks the same as `nat eamm run` (unimplemented spec) | User can't tell which commands are production-grade |
| **`nat help` is static** | Adding a command requires updating both `build_parser()` AND `cmd_help()` | Help text drifts from actual commands over time |
| **No command search** | Can't do `nat help --grep IC` to find IC-related commands | Must read 250 lines of help text |

### B. Visualization Fragmentation

| Problem | Example | Impact |
|---------|---------|--------|
| **5+ entry points** | `nat visualize scan`, `nat trade-viz`, `nat 15m viz`, `nat agent dashboard`, `nat cluster explore` | User can't find the right viz command |
| **Static PNG only** | `nat visualize *` writes PNGs to disk, no terminal preview | Must open file manager to see results |
| **No feature inspection** | Can't do `nat feature show imbalance_qty_l1 --last 1h` | Must write ad-hoc Python to inspect features |
| **No algorithm decision trace** | Can't see why an algorithm triggered or didn't | Must dig through paper trade logs manually |
| **No live signal view** | Can't watch current signal values updating | Must repeatedly query Parquet files |
| **Web dashboards are disconnected** | 5 separate HTTP servers on different ports, no unified entry | Must remember port numbers |

### C. Code Structure

| Problem | Detail | Impact |
|---------|--------|--------|
| **5,113-line monolith** | All 216 handlers in one file | Hard to navigate, merge conflicts, no separation of concerns |
| **Tight coupling** | `sys.path.insert(0, ...)` + `from module import func` inline in handlers | Import errors at runtime, not at load time |
| **Duplicate patterns** | Agent commands (`start/stop/once/status/queue/registry/graveyard/report`) duplicated 4× for micro/mf/macro/meta | ~400 lines of near-identical code |
| **No plugin system** | Adding a command requires editing `nat` directly | External tools can't register commands |

---

## IV. Proposed Improvements

### Improvement 1 — Unified `nat viz` Command Group

Consolidate all visualization under `nat viz` with consistent interface. Keep existing `nat visualize *` as backward-compatible aliases.

```
nat viz                              # List available viz commands with descriptions
nat viz features                     # Live feature table (terminal, rich)
nat viz feature <name>               # Single feature: time series + distribution + IC
nat viz algorithm <name>             # Algorithm signals + decisions + P&L trace
nat viz portfolio                    # Portfolio: weights, P&L, correlation, drawdown
nat viz orderbook                    # L1/L2 book state snapshot
nat viz signals                      # All active signals heatmap (terminal)
nat viz paper                        # Paper trading P&L + fill analysis
nat viz ic-heatmap                   # Feature IC heatmap (port from agent dashboard JS)
nat viz spectral <feature>           # PSD + coherence for a feature
nat viz regime                       # Current regime state + transition history
nat viz compare <alg1> <alg2>        # Side-by-side algorithm comparison
nat viz correlation                  # Cross-algorithm/feature correlation matrix
nat viz data-quality                 # NaN rates, gap analysis, schema health
```

**Design decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Rendering target** | Terminal-first via `rich`, `--png` for file export | Most common use is quick inspection, not report generation |
| **Live refresh** | `--live` flag, polls every 5s | Reuses existing Parquet files and Redis streams |
| **Data scoping** | `--symbol BTC --hours 4 --date 2026-06-12` | Consistent across all viz commands |
| **Output control** | `--output <path>` for PNG/HTML, `--json` for data | Same pattern as existing commands |
| **Backend library** | `rich` for terminal, `matplotlib` for static, `plotly` for HTML | Already dependencies, no new installs |
| **Backward compat** | `nat visualize scan` still works, aliased internally | Zero migration cost |

#### `nat viz features` — Live Feature Table

```
nat viz features --symbol BTC --hours 1 --top 20
nat viz features --symbol BTC --category entropy --live
```

Rich terminal table:

```
  NAT Feature Dashboard — BTC (last 1h, 720 rows)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Feature                       Value    Z-Score  NaN%   IC@5m   Spark
  ──────────────────────────────────────────────────────────────────────
  raw_spread_bps_mean            2.34     +0.8    0.0%  +0.139  ▁▂▃▅▆▇
  imbalance_qty_l1_mean          0.12     +1.2    0.0%  +0.088  ▃▅▆▇▆▅
  ent_book_shape_mean            3.41     -0.3    0.0%  +0.067  ▇▆▅▅▆▇
  hawkes_intensity_mean          0.08     +2.1    0.0%  +0.035  ▁▁▂▅▇▇
  whale_flow_imbalance           NaN       —     100%     —       —
  ...
  ──────────────────────────────────────────────────────────────────────
  Active: 135/191 | Dead (100% NaN): 56 | Const: 1
```

Filters: `--category`, `--top N`, `--sort ic|zscore|nan`, `--alive-only`

#### `nat viz algorithm` — Decision Trace

```
nat viz algorithm jump_detector --symbol BTC --last 2h
nat viz algorithm hierarchical_combiner --symbol ETH --date 2026-06-12
```

Terminal output:

```
  jump_detector — BTC (last 2h, 720 ticks)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status: INACTIVE (no jump detected)
  Last signal: 14:23:05 — LONG, confidence 0.73, held 12min, P&L +0.8 bps

  Signal Timeline:
  14:00 ─────────────────── 14:30 ────────── 15:00 ──────────── 15:30
        ▔▔▔▔▔▔▔▔▁▁▁▁▁▆██▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                    ↑ ENTRY              ↑ EXIT

  Inputs at last trigger:
    lee_mykland_stat:  4.23 (threshold: 3.5)   ← triggered
    vol_returns_5m:    0.0034
    flow_count_30s:    127

  Last 5 signals:
    14:23:05  LONG   +0.73  +0.8 bps  12min
    13:51:22  SHORT  -0.61  -1.2 bps   8min
    13:15:44  LONG   +0.82  +2.1 bps  18min
    ...
```

With `--png`: renders price chart with signal overlay, entry/exit markers, P&L waterfall.

#### `nat viz portfolio` — Portfolio Dashboard

```
nat viz portfolio --live
```

Multi-tab terminal dashboard (like `nat monitor`):

- **Tab 1 — P&L**: Per-algorithm cumulative P&L, daily Sharpe, win rate, max drawdown
- **Tab 2 — Exposure**: Per-symbol current position, risk-parity weights, net exposure
- **Tab 3 — Correlation**: Cross-algorithm signal correlation matrix (updated from recent data)
- **Tab 4 — Risk**: Kill switch status, IC decay warnings, daily/weekly/monthly drawdown gauges

### Improvement 2 — Group-Level Help

When user types `nat alpha` (no subcommand), show curated help instead of argparse default:

```python
def cmd_alpha_help(args):
    print(f"""
  {BOLD}Alpha Pipeline{W} — 9-step systematic alpha discovery

  Steps:
    nat alpha combine       Step 2: IC-weighted feature combination
    nat alpha size          Step 3: Kelly position sizing with cost filter
    nat alpha validate      Step 4: Walk-forward + deflated Sharpe
    ...

  Pipeline orchestration:
    nat alpha pipeline-start    Run all 9 steps with quality gates
    nat alpha pipeline-status   Show pipeline state

  Quality gates: G1 (screen) → G2 (combine) → G3 (sizing) → G4 (validate)
                 → G5 (regime) → G6 (multi-freq) → G7 (portfolio) → G8 (paper)

  Start with: nat alpha pipeline-status
    """)
```

Apply this pattern to all command groups: `nat agent`, `nat cluster`, `nat backtest`, `nat docker`, etc.

### Improvement 3 — Command Search

```
nat help --grep IC
nat help --grep kalman
nat help --grep paper
```

Searches both command names and help text, returns matching commands with context. Simple implementation:

```python
def cmd_help(args):
    if hasattr(args, 'grep') and args.grep:
        parser = build_parser()
        commands = walk_commands(parser)
        matches = [c for c in commands
                   if args.grep.lower() in c['name'].lower()
                   or args.grep.lower() in c['help'].lower()]
        for m in matches:
            print(f"  nat {m['name']:30s} {m['help']}")
        return
    # ... existing help text
```

### Improvement 4 — Maturity Tags

Add status tags to command help text:

```
nat algorithm evaluate     [PROVEN]  IC/drift evaluation
nat eamm run               [SPEC]   Full pipeline (not yet implemented)
nat alg1 live              [LIVE]   LIVE orders on Hyperliquid
nat hier train             [PRELIM] Hierarchical combiner (2-day data only)
```

Tags: `[PROVEN]` (30+ day OOS), `[LIVE]` (production), `[PRELIM]` (limited validation), `[SPEC]` (spec only, not implemented), `[BETA]` (functional but not validated).

### Improvement 5 — Script Modularization (Long-term)

Split `nat` from 5,113-line monolith into domain modules:

```
nat                              # Main entry point (~300 lines: imports, helpers, main())
scripts/cli/
├── __init__.py                  # register() protocol
├── system.py                    # start/stop/status/health/log/monitor/dashboard
├── data.py                      # data/signal/screen/scan/fetch
├── alpha.py                     # alpha pipeline (13 subcommands)
├── research.py                  # spannung/kalman/profile/macro/cluster/validate
├── agent.py                     # agent/mf-agent/macro-agent/meta-agent/it-engine
├── backtest.py                  # backtest/model/experiment
├── oos.py                       # daily/oos30/gauntlet/tournament
├── viz.py                       # Unified visualization (new)
├── build.py                     # build/test/docker/deploy
├── optimize.py                  # swarm/evolve/discovery
└── helpers.py                   # _sh, _exec, _py, _p, _banner, _json_mode, _output
```

Each module exports:

```python
def register(subparsers):
    """Register all commands in this module."""
    p = subparsers.add_parser('alpha', help='Alpha pipeline')
    asub = p.add_subparsers(dest='subcmd')
    # ... register subcommands
```

Main `nat` script becomes:

```python
def build_parser():
    p = argparse.ArgumentParser(prog='nat')
    p.add_argument('--json', action='store_true')
    sub = p.add_subparsers(dest='command')

    from scripts.cli import system, data, alpha, research, agent
    from scripts.cli import backtest, oos, viz, build, optimize
    for mod in [system, data, alpha, research, agent,
                backtest, oos, viz, build, optimize]:
        mod.register(sub)

    return p
```

**Benefits**: each domain is independently navigable (~300-500 LOC each), merge conflicts eliminated, new commands don't touch other domains.

**Migration**: incremental — move one group at a time, keep backward compat by importing from new location.

---

## V. Visualization Library Design

### File Structure

```
scripts/viz/
├── __init__.py               # Public API: render(name, **kwargs)
├── common.py                 # Shared: themes, color schemes, data loading, layout
├── terminal.py               # Rich-based terminal rendering (tables, sparklines, panels)
├── features.py               # Feature time series + distribution (exists, extend)
├── algorithms.py             # Algorithm signal + decision trace (new)
├── portfolio.py              # Portfolio dashboard (new)
├── orderbook.py              # Order book state visualization (new)
├── spectral.py               # PSD/coherence plots (wrap existing analysis code)
├── correlations.py           # Correlation matrix heatmap (exists, extend)
├── ic_heatmap.py             # IC heatmap (port from agent_dashboard.py JS → terminal)
├── distributions.py          # Distribution analysis, QQ plots (exists, extend)
├── events.py                 # Event detection visualization (exists, extend)
├── regime.py                 # Regime state + transitions (new)
└── trades.py                 # Paper/live trade P&L visualization (wrap trade_visualize.py)
```

### Common Interface

Every viz module exposes three rendering modes:

```python
# scripts/viz/features.py

def render_terminal(symbol: str, hours: float = 1, **kwargs) -> None:
    """Rich terminal output — default for `nat viz features`."""
    ...

def render_figure(symbol: str, hours: float = 1, **kwargs) -> plt.Figure:
    """Matplotlib figure — used by `nat viz features --png output.png`."""
    ...

def export_data(symbol: str, hours: float = 1, **kwargs) -> dict:
    """Raw data dict — used by `nat viz features --json`."""
    ...
```

### Data Loading (common.py)

Centralized data access so viz modules don't each reinvent Parquet loading:

```python
# scripts/viz/common.py

def load_features(symbol: str, hours: float, data_dir: Path = DATA_DEFAULT) -> pd.DataFrame:
    """Load latest N hours of feature data from Parquet files."""
    ...

def load_algorithm_signals(algorithm: str, symbol: str, hours: float) -> pd.DataFrame:
    """Load algorithm signal output for visualization."""
    ...

def load_paper_trades(symbol: str, date: str = None) -> pd.DataFrame:
    """Load paper trade logs."""
    ...

def subscribe_live(symbol: str) -> Iterator[dict]:
    """Subscribe to Redis feature stream for live updates."""
    ...

# Shared styling
THEME = {
    'positive': 'green', 'negative': 'red', 'neutral': 'dim',
    'header': 'bold cyan', 'warning': 'bold yellow',
}
```

### Terminal Rendering (terminal.py)

Reusable rich components:

```python
# scripts/viz/terminal.py

def sparkline(values: list[float], width: int = 20) -> str:
    """Unicode sparkline: ▁▂▃▄▅▆▇█"""
    ...

def ic_bar(ic: float, width: int = 10) -> str:
    """Colored IC bar: [████████──] +0.139"""
    ...

def zscore_indicator(z: float) -> str:
    """Z-score with color: [green]+1.2[/green] or [red]-2.5[/red]"""
    ...

def live_table(data_fn: callable, interval: float = 5.0) -> None:
    """Auto-refreshing rich table. Calls data_fn() every interval seconds."""
    ...

def multi_tab_dashboard(tabs: dict[str, callable], interval: float = 5.0) -> None:
    """Tabbed terminal dashboard (like nat monitor). Keys switch tabs."""
    ...
```

---

## VI. `nat viz` Command Registration

```python
# scripts/cli/viz.py (or inline in nat for initial implementation)

def register(subparsers):
    viz_p = subparsers.add_parser('viz', help='Visualization & inspection')
    viz_p.set_defaults(func=cmd_viz_help)
    vs = viz_p.add_subparsers(dest='subcmd')

    # Common args added to all viz commands
    def _add_common(p):
        p.add_argument('--symbol', default='BTC')
        p.add_argument('--hours', type=float, default=4)
        p.add_argument('--date', default=None)
        p.add_argument('--data', default=str(DATA_DEFAULT))
        p.add_argument('--output', default=None, help='Save PNG/HTML to path')
        p.add_argument('--live', action='store_true', help='Auto-refresh every 5s')

    # nat viz features
    p = vs.add_parser('features', help='Live feature dashboard (terminal)')
    _add_common(p)
    p.add_argument('--category', default=None, help='Filter by category')
    p.add_argument('--top', type=int, default=None, help='Show top N by IC')
    p.add_argument('--sort', default='ic', choices=['ic', 'zscore', 'nan', 'name'])
    p.add_argument('--alive-only', action='store_true', help='Hide 100% NaN features')
    p.set_defaults(func=cmd_viz_features)

    # nat viz feature <name>
    p = vs.add_parser('feature', help='Single feature deep-dive')
    _add_common(p)
    p.add_argument('name', help='Feature name (e.g., imbalance_qty_l1_mean)')
    p.set_defaults(func=cmd_viz_feature)

    # nat viz algorithm <name>
    p = vs.add_parser('algorithm', help='Algorithm decision trace')
    _add_common(p)
    p.add_argument('name', help='Algorithm name (e.g., jump_detector)')
    p.set_defaults(func=cmd_viz_algorithm)

    # nat viz portfolio
    p = vs.add_parser('portfolio', help='Portfolio P&L and risk dashboard')
    _add_common(p)
    p.set_defaults(func=cmd_viz_portfolio)

    # nat viz signals
    p = vs.add_parser('signals', help='Active signals heatmap')
    _add_common(p)
    p.set_defaults(func=cmd_viz_signals)

    # nat viz paper
    p = vs.add_parser('paper', help='Paper trading P&L + fill analysis')
    _add_common(p)
    p.set_defaults(func=cmd_viz_paper)

    # nat viz ic-heatmap
    p = vs.add_parser('ic-heatmap', help='Feature × gate IC heatmap')
    _add_common(p)
    p.set_defaults(func=cmd_viz_ic_heatmap)

    # nat viz spectral <feature>
    p = vs.add_parser('spectral', help='PSD + coherence for a feature')
    _add_common(p)
    p.add_argument('name', help='Feature name')
    p.set_defaults(func=cmd_viz_spectral)

    # nat viz regime
    p = vs.add_parser('regime', help='Regime state and transition history')
    _add_common(p)
    p.set_defaults(func=cmd_viz_regime)

    # nat viz compare <alg1> <alg2>
    p = vs.add_parser('compare', help='Side-by-side algorithm comparison')
    _add_common(p)
    p.add_argument('alg1', help='First algorithm')
    p.add_argument('alg2', help='Second algorithm')
    p.set_defaults(func=cmd_viz_compare)

    # nat viz correlation
    p = vs.add_parser('correlation', help='Cross-algorithm correlation matrix')
    _add_common(p)
    p.set_defaults(func=cmd_viz_correlation)

    # nat viz data-quality
    p = vs.add_parser('data-quality', help='NaN rates, gaps, schema health')
    _add_common(p)
    p.set_defaults(func=cmd_viz_data_quality)
```

---

## VII. Implementation Priority

| # | Improvement | Effort | Impact | Blocks |
|---|-------------|--------|--------|--------|
| 1 | `nat viz features` — terminal feature table | ~6h | High — answers "what's the pipeline doing?" | None |
| 2 | `nat viz algorithm` — decision trace | ~8h | High — answers "why did it trade?" | None |
| 3 | Group-level help for all command groups | ~4h | Medium — discoverability | None |
| 4 | `nat help --grep <term>` command search | ~1h | Medium — discoverability | None |
| 5 | `nat viz portfolio` — portfolio dashboard | ~8h | High — needed for paper/live trading | Q3 tasks |
| 6 | `nat viz paper` — paper trade analysis | ~4h | High — needed for G8 validation | Q3 tasks |
| 7 | `nat viz spectral` — wrap existing code | ~2h | Low — already in analysis scripts | None |
| 8 | Maturity tags on commands | ~2h | Low — cosmetic but useful | None |
| 9 | Script modularization (split into cli/) | ~12h | High long-term — maintenance | None |
| 10 | `nat viz` backward-compat aliases | ~1h | None — prevents breakage | #1 |

### Recommended order: 4 → 3 → 1 → 2 → 10 → 6 → 5 → 7 → 8 → 9

Start with quick wins (search, group help), then build core viz commands, then modularize.

---

## VIII. Design Decisions Summary

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Terminal library | `rich` | textual, curses, blessed | Already a dependency, used by monitor/OOS terminal, no new installs |
| Static chart library | `matplotlib` | plotly, bokeh | Already used by all existing viz, no new dependency |
| Interactive HTML | `plotly` (optional) | dash, streamlit | Lighter than full server, exports self-contained HTML |
| Command prefix | `nat viz` | `nat view`, `nat show`, `nat plot` | `viz` is short, unambiguous, and unused |
| Data source | Parquet files + Redis | Dedicated viz database | No new infrastructure, same data the ingestor writes |
| Live updates | Poll Parquet / Redis subscribe | WebSocket server | Simpler, no new service to run |
| Modularization | `scripts/cli/*.py` with `register()` | Click framework, Typer | Keeps argparse (no new dependency), incremental migration |
| Backward compat | Aliases from old to new | Deprecation warnings | Zero migration cost for existing workflows and scripts |

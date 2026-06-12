# NAT10 — Script Modularization (Split `nat` into `scripts/cli/`)

**Priority**: 10 (long-term, ~12h)
**Status**: NOT STARTED
**Depends on**: None (but best done after NAT1-NAT9 to avoid merge conflicts)

---

## Objective

Split the 5,113-line monolithic `nat` script into domain-specific modules under `scripts/cli/`, each responsible for registering and handling its command group. The main `nat` script becomes a ~300-line dispatcher.

## Context

The monolithic `nat` script has grown to 216 handler functions in a single file. This causes:
- Difficult navigation (5,113 lines, no separation of concerns)
- Merge conflicts when two features touch different command groups
- `sys.path.insert()` hacks for imports scattered throughout
- Duplicate patterns (agent commands duplicated 4× for micro/mf/macro/meta)

Modularization keeps the single `nat` entry point (no breaking changes) but distributes code into maintainable domain files.

## Scope

**In scope**:
- Split into 11 domain modules under `scripts/cli/`
- Each module exports `register(subparsers)` function
- Shared helpers in `scripts/cli/helpers.py`
- Main `nat` script reduced to ~300 lines
- Incremental migration: one group at a time

**Out of scope**:
- Switching to Click/Typer/other framework (keep argparse)
- Changing command names or behavior
- Adding new commands

## Target Structure

```
nat                               # Entry point (~300 lines)
scripts/cli/
├── __init__.py                   # Module registration protocol
├── helpers.py                    # _sh, _exec, _py, _p, _banner, _json_mode, _output, constants
├── system.py                     # start, stop, status, health, log, monitor, dashboard (~200 LOC)
├── data.py                       # data, signal, screen, scan, fetch, macro (~300 LOC)
├── alpha.py                      # alpha pipeline (13 subcommands) (~400 LOC)
├── research.py                   # spannung, kalman, profile, report, validate (~350 LOC)
├── agent.py                      # agent, mf-agent, macro-agent, meta-agent, it-engine (~500 LOC)
├── backtest.py                   # backtest, model, experiment (~400 LOC)
├── oos.py                        # daily, oos30, gauntlet, tournament, alg1 (~350 LOC)
├── viz.py                        # Unified viz (from NAT3-NAT8) (~300 LOC)
├── build.py                      # build, test, docker, deploy, run (~300 LOC)
└── optimize.py                   # swarm, evolve, discovery, audit (~250 LOC)
```

## Implementation

### Step 1 — Extract helpers (~1h)

Move shared utilities from `nat` to `scripts/cli/helpers.py`:

```python
# scripts/cli/helpers.py
ROOT = Path(__file__).resolve().parent.parent.parent
RUST = ROOT / "rust"
DATA_DEFAULT = ROOT / "data" / "features"
# ... all constants, _sh, _exec, _py, _p, _banner, _json_mode, _output
```

### Step 2 — Define registration protocol (~30min)

```python
# scripts/cli/__init__.py
"""CLI module registration protocol.

Each module must export:
    register(subparsers) -> None
"""
```

### Step 3 — Migrate one group at a time (~8h total)

Migration order (lowest risk first):
1. `build.py` — build/test/docker are self-contained
2. `optimize.py` — swarm/evolve/discovery are self-contained
3. `system.py` — start/stop/status
4. `data.py` — data/signal/screen
5. `oos.py` — oos30/gauntlet/tournament
6. `research.py` — spannung/kalman/cluster
7. `backtest.py` — backtest/model/experiment
8. `alpha.py` — alpha pipeline
9. `agent.py` — 4 agent types (dedup ~400 lines)
10. `viz.py` — already built in NAT3-NAT8

Each migration step:
1. Copy handler functions to new module
2. Update imports to use `from scripts.cli.helpers import ...`
3. Add `register(subparsers)` function
4. Replace handlers in `nat` with module import
5. Run tests, verify all commands still work

### Step 4 — Simplify main `nat` (~1h)

```python
#!/usr/bin/env python3
"""nat — unified NAT research terminal."""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.cli import helpers
from scripts.cli import (system, data, alpha, research, agent,
                          backtest, oos, viz, build, optimize)

MODULES = [system, data, alpha, research, agent,
           backtest, oos, viz, build, optimize]

def build_parser():
    p = argparse.ArgumentParser(prog='nat', description='NAT research terminal',
                                add_help=False)
    p.add_argument('--json', action='store_true')
    sub = p.add_subparsers(dest='command')
    for mod in MODULES:
        mod.register(sub)
    # Special top-level commands
    sub.add_parser('help').set_defaults(func=helpers.cmd_help)
    sub.add_parser('commands').set_defaults(func=helpers.cmd_commands)
    sub.add_parser('math').set_defaults(func=helpers.cmd_math)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        helpers.cmd_help()
        return
    args.func(args)

if __name__ == '__main__':
    main()
```

### Step 5 — Agent deduplication (~2h)

The 4 agent types share 8 identical subcommands. Extract shared pattern:

```python
# scripts/cli/agent.py

def _register_agent(sub, name, prefix, agent_module, help_text):
    """Register standard agent subcommands (start/stop/once/status/queue/registry/graveyard/report)."""
    p = sub.add_parser(name, help=help_text)
    p.set_defaults(func=lambda a: _agent_help(name, a))
    asub = p.add_subparsers(dest='subcmd')

    for cmd_name, handler_factory in [
        ('start', lambda: _agent_start),
        ('stop', lambda: _agent_stop),
        ('once', lambda: _agent_once),
        ('status', lambda: _agent_status),
        ('queue', lambda: _agent_queue),
        ('registry', lambda: _agent_registry),
        ('graveyard', lambda: _agent_graveyard),
        ('report', lambda: _agent_report),
    ]:
        sp = asub.add_parser(cmd_name)
        sp.set_defaults(func=handler_factory(), agent_module=agent_module, agent_prefix=prefix)

def register(subparsers):
    _register_agent(subparsers, 'agent', 'micro', 'scripts/agent/daemon.py', '[PROVEN] Microstructure agent')
    _register_agent(subparsers, 'mf-agent', 'mf', 'scripts/agent/mf_daemon.py', '[PROVEN] Medium-freq agent')
    _register_agent(subparsers, 'macro-agent', 'macro', 'scripts/agent/macro_daemon.py', '[PROVEN] Macro agent')
    # meta-agent has different subcommands, register separately
    ...
```

This eliminates ~400 lines of near-identical code.

## Files to Create / Modify

| File | Action | LOC |
|------|--------|-----|
| `scripts/cli/__init__.py` | Create | ~10 |
| `scripts/cli/helpers.py` | Create (extract from `nat`) | ~200 |
| `scripts/cli/system.py` | Create | ~200 |
| `scripts/cli/data.py` | Create | ~300 |
| `scripts/cli/alpha.py` | Create | ~400 |
| `scripts/cli/research.py` | Create | ~350 |
| `scripts/cli/agent.py` | Create (with dedup) | ~300 |
| `scripts/cli/backtest.py` | Create | ~400 |
| `scripts/cli/oos.py` | Create | ~350 |
| `scripts/cli/viz.py` | Already exists from NAT3-NAT8 | ~300 |
| `scripts/cli/build.py` | Create | ~300 |
| `scripts/cli/optimize.py` | Create | ~250 |
| `nat` | Reduce from 5,113 to ~300 lines | -4,800 |

## Acceptance Criteria

- [ ] `nat` script is < 400 lines (down from 5,113)
- [ ] All 251 commands work identically to before
- [ ] `nat help` output unchanged
- [ ] `nat commands --json` output unchanged (same command count)
- [ ] `nat test` passes
- [ ] `nat test agent` passes (350+ tests)
- [ ] Each `scripts/cli/*.py` module is < 500 LOC
- [ ] Agent commands deduplicated: 4 agents × 8 commands in < 100 lines (vs ~400 before)
- [ ] No `sys.path.insert()` hacks in module files (use proper imports)
- [ ] `scripts/cli/helpers.py` contains all shared constants and utilities

## Testing / Verification

```bash
# 1. Command count unchanged
nat commands --json | python3 -c "import sys,json; print(json.load(sys.stdin)['count'])"
# Should print same number as before (251 or current count)

# 2. Full test suite
nat test
nat test agent
nat test pipeline

# 3. Spot-check commands across all modules
nat status
nat data
nat alpha pipeline-status
nat agent status
nat algorithm list
nat oos30 --help
nat docker ps
nat evolve status
nat viz features --help

# 4. Line count verification
wc -l nat
# Should be < 400

wc -l scripts/cli/*.py
# Each file < 500, total ~3,500

# 5. Help unchanged
nat help | md5sum
# Compare with pre-migration checksum
```

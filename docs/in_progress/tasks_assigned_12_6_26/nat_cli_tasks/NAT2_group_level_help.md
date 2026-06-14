# NAT2 — Group-Level Help for All Command Groups

**Priority**: 2 (~4h)
**Status**: NOT STARTED
**Depends on**: None

---

## Objective

Replace the generic argparse usage output when a user types `nat <group>` (e.g., `nat alpha`, `nat agent`) with curated help text that mirrors the quality of `nat help` but scoped to that group.

## Context

Currently `nat alpha` (no subcommand) prints:

```
usage: nat alpha [-h] {combine,size,validate,...} ...
positional arguments: {combine,size,...}
```

This is unhelpful. The good help is in `nat help` under the "Alpha Pipeline" section, but users don't know that. Every group should print its own curated help when invoked without a subcommand.

## Scope

**In scope**:
- Curated help for all major groups: `alpha`, `agent`, `mf-agent`, `macro-agent`, `meta-agent`, `backtest`, `model`, `cluster`, `validate`, `eamm`, `pipeline`, `experiment`, `docker`, `swarm`, `evolve`, `discovery`, `data`, `config`, `visualize`, `build`, `test`, `algorithm`, `spannung`, `gauntlet`, `tournament`
- Each group help: 10-20 lines, lists subcommands with descriptions, shows workflow

**Out of scope**:
- Changing the subcommand structure itself
- Adding new commands

## Implementation

### Pattern

Replace the current default handler:

```python
# Current (prints argparse usage):
alpha_p.set_defaults(func=lambda a: alpha_p.print_help())

# New (prints curated help):
alpha_p.set_defaults(func=cmd_alpha_help)
```

### Example: `cmd_alpha_help()`

```python
def cmd_alpha_help(args=None):
    print(f"""
  {BOLD}Alpha Pipeline{W} — 9-step systematic alpha discovery

  Steps:
    nat alpha combine          Step 2: IC-weighted feature combination
    nat alpha size             Step 3: Kelly position sizing with cost filter
    nat alpha validate         Step 4: Walk-forward + deflated Sharpe
    nat alpha regime           Step 5: Quintile regime conditioning
    nat alpha multi-freq       Step 6: Multi-frequency signal integration
    nat alpha portfolio        Step 7: Multi-symbol portfolio assembly
    nat alpha paper            Step 8: Paper trading simulation
    nat alpha deploy           Step 9: Deployment readiness check

  Pipeline (orchestrated with quality gates G1-G8):
    nat alpha pipeline-start   Run all 9 steps with auto gating
    nat alpha pipeline-resume  Resume from last phase (--force-gate to skip)
    nat alpha pipeline-status  Show pipeline state and gate verdicts
    nat alpha pipeline-gates   Detailed gate metrics report
    nat alpha pipeline-step N  Run single step (1-9)

  Start with: nat alpha pipeline-status
    """)
```

### Groups to Update

| Group | Subcommands | Key info for help |
|-------|-------------|-------------------|
| `alpha` | 13 | 9-step pipeline, quality gates G1-G8 |
| `agent` | 9 | 5-gate protocol, microstructure tick-level |
| `mf-agent` | 8 | 4-gate, medium-frequency 1min-1h |
| `macro-agent` | 8 | 4-gate, macro 1h-24h |
| `meta-agent` | 8 | Cross-agent coordination, budget |
| `backtest` | 8 | Event-driven, walk-forward, ML |
| `model` | 6 | ElasticNet, LightGBM, GMM, hierarchical |
| `cluster` | 6 | GMM/HMM, Q1-Q3 quality gates |
| `algorithm` | 3 | 18 algorithms, IC evaluation |
| `docker` | 7 | Build, deploy, stack operations |
| `swarm` | 5 | 35D parameter sweep |
| `evolve` | 5 | Optuna CMA-ES/TPE/NSGA-II |
| `experiment` | 6 | Tracking, comparison, snapshots |
| `pipeline` | 6 | State machine IDLE→DONE |
| `gauntlet` | 4 | Multi-day OOS sweep |
| `tournament` | 8 | Continuous algorithm ranking |
| `data` | 4 | Parquet inspection, validation |
| `config` | 3 | TOML inspection, validation |
| `spannung` | 5 | Signal grid search, spectral |
| `discovery` | 4 | Continuous alpha discovery |

## Files to Modify

- `/home/onat/nat/nat` — add `cmd_<group>_help()` functions, update `set_defaults()` calls in `build_parser()`

## Acceptance Criteria

- [ ] `nat alpha` prints curated help with all 13 subcommands grouped logically
- [ ] `nat agent` prints curated help with 9 subcommands and mentions 5-gate protocol
- [ ] All 20+ groups above print curated help instead of argparse default
- [ ] Each group help is 10-20 lines (not a wall of text)
- [ ] Each group help includes a "Start with:" suggestion
- [ ] `nat alpha combine -h` still works (subcommand help unchanged)
- [ ] `nat help` still works (full help unchanged)

## Testing / Verification

```bash
# 1. Group help shows curated text
nat alpha 2>&1 | head -3
# Should show: "Alpha Pipeline — 9-step systematic alpha discovery"

# 2. All groups tested
for group in alpha agent backtest model cluster algorithm docker swarm evolve; do
  echo "=== $group ==="
  nat $group 2>&1 | head -2
done
# Each should show a descriptive title, not "usage: nat <group> [-h]"

# 3. Subcommand help still works
nat alpha combine -h
# Should show the existing rich epilog with math

# 4. Full help unchanged
nat help | wc -l
# Should be same line count as before
```

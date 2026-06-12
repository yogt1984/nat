# NAT9 — Maturity Tags & Backward-Compatible Aliases

**Priority**: 9 (~3h)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz commands exist to alias)

---

## Objective

Add maturity status tags to command help text so users can distinguish proven commands from experimental/unimplemented ones. Add backward-compatible aliases from old `nat visualize *` to new `nat viz *`.

## Context

All 251 commands look identical in help output. A user can't tell that `nat oos30` is battle-tested (runs daily, 30-day OOS) while `nat eamm run` references an unimplemented spec. Maturity tags solve this by adding `[PROVEN]`, `[LIVE]`, `[BETA]`, `[PRELIM]`, or `[SPEC]` to help text.

Separately, old visualization commands (`nat visualize scan`, `nat trade-viz`) should alias to the new `nat viz` group for zero-migration-cost transition.

## Scope

**In scope**:
- Maturity tags on all 251 commands in `nat help` and `nat commands`
- Tag definitions with clear criteria
- Backward-compatible aliases: `nat visualize *` → `nat viz *`
- `nat trade-viz` → `nat viz paper`

**Out of scope**:
- Changing command behavior
- Removing old commands
- Deprecation warnings (aliases are permanent, not transitional)

## Implementation

### Part 1 — Maturity Tags (~2h)

#### Tag Definitions

| Tag | Criteria | Color |
|-----|----------|-------|
| `[LIVE]` | Running in production with real capital | bold green |
| `[PROVEN]` | 30+ day OOS validated, used regularly | green |
| `[BETA]` | Functional, tested, but < 30 day OOS | yellow |
| `[PRELIM]` | Works but limited data / validation | dim yellow |
| `[SPEC]` | Spec exists, not implemented or broken | dim red |

#### Tag Assignment

Update help strings in `build_parser()`:

```python
# Before:
sub.add_parser('oos30', help='30-day OOS validation')

# After:
sub.add_parser('oos30', help='[PROVEN] 30-day OOS validation')
```

Key assignments:

| Command | Tag | Reason |
|---------|-----|--------|
| `nat start/stop/status` | `[PROVEN]` | Core system commands, daily use |
| `nat algorithm evaluate` | `[PROVEN]` | 18 algorithms, 30-day OOS |
| `nat oos30` | `[PROVEN]` | 30-day batch validation |
| `nat gauntlet run` | `[PROVEN]` | Multi-day sweep |
| `nat agent start` | `[PROVEN]` | 350+ tests, daily use |
| `nat alg1 live` | `[LIVE]` | Production orders on Hyperliquid |
| `nat alg1 paper` | `[PROVEN]` | Paper trading validated |
| `nat model train-hier` | `[PRELIM]` | Only 2-day data |
| `nat eamm run` | `[SPEC]` | Spec only, not deployable |
| `nat viz features` | `[BETA]` | New, functional |
| `nat cluster hmm-fit` | `[BETA]` | Works but limited validation |

#### Update `cmd_help()`

Add tags to the curated help text in `cmd_help()`:

```python
    nat oos30                  [PROVEN] 30-day batch OOS for 5 winners
    nat eamm run               [SPEC]   Full pipeline (not yet implemented)
    nat alg1 live              [LIVE]   LIVE orders on Hyperliquid
```

#### Update `cmd_commands()`

Include tag in JSON output:

```python
commands.append({
    "name": full_name,
    "help": help_text,
    "maturity": _extract_tag(help_text),  # "PROVEN", "BETA", etc.
    "args": cmd_args,
})
```

### Part 2 — Backward-Compatible Aliases (~1h)

Add aliases that delegate to new viz commands:

```python
# In build_parser(), after nat viz registration:

# Alias: nat visualize scan → nat viz data-quality (closest equivalent)
# Keep original nat visualize * working — they still call the old scripts
# Add note in nat visualize help pointing to nat viz

def cmd_visualize_compat(args):
    print(f"  {Y}Tip:{W} Try 'nat viz' for the new unified visualization suite.")
    print(f"  The 'nat visualize' commands still work as before.\n")
    # ... existing visualize help
```

For trade-viz:
```python
# nat trade-viz already works, just add cross-reference in help
trade_p.epilog = "See also: nat viz paper (richer paper trading analysis)"
```

## Files to Modify

- `/home/onat/nat/nat` — `cmd_help()`, `cmd_commands()`, `build_parser()` (tag strings + aliases)

## Acceptance Criteria

### Maturity Tags:
- [ ] `nat help` shows tags on all commands: `[PROVEN]`, `[LIVE]`, `[BETA]`, `[PRELIM]`, `[SPEC]`
- [ ] `nat commands` includes tags in help text
- [ ] `nat commands --json` includes `maturity` field per command
- [ ] Every command has a tag (no untagged commands)
- [ ] Tags are accurate: `[LIVE]` only on actually-live commands, `[SPEC]` on unimplemented

### Aliases:
- [ ] `nat visualize scan` still works (unchanged behavior)
- [ ] `nat visualize` shows tip pointing to `nat viz`
- [ ] `nat trade-viz` still works, epilog mentions `nat viz paper`
- [ ] No existing command behavior is broken

## Testing / Verification

```bash
# 1. Tags in help
nat help | grep -c "\[PROVEN\]\|\[LIVE\]\|\[BETA\]\|\[PRELIM\]\|\[SPEC\]"
# Should be > 50

# 2. Tags in commands --json
nat commands --json | python3 -c "
import sys, json
data = json.load(sys.stdin)
tags = {}
for c in data['commands']:
    tag = c.get('maturity', 'NONE')
    tags[tag] = tags.get(tag, 0) + 1
for tag, count in sorted(tags.items()):
    print(f'  {tag}: {count}')
"

# 3. Old commands still work
nat visualize scan --symbol BTC 2>&1 | head -2
nat trade-viz --symbol BTC 2>&1 | head -2

# 4. Alias tip shown
nat visualize 2>&1 | grep -i "nat viz"
```

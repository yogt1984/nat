# NAT1 — Command Search (`nat help --grep`)

**Priority**: 1 (quick win, ~1h)
**Status**: NOT STARTED
**Depends on**: None

---

## Objective

Add `--grep <term>` flag to `nat help` so users can search across command names and help text instead of reading 250 lines of help output.

## Context

With 251 commands, finding the right one is hard. `nat help` prints a static 250-line wall of text. `nat commands` lists all commands flat. Neither supports search. A user wanting IC-related commands must visually scan the entire output.

## Scope

**In scope**:
- `nat help --grep <term>` — search command names and descriptions
- Case-insensitive matching
- Show matching commands with highlighted context

**Out of scope**:
- Fuzzy matching (exact substring is sufficient)
- Interactive search (not a TUI, just filtered output)

## Implementation

### Step 1 — Add `--grep` argument to help parser

In `build_parser()` (~line 3602 of `nat`):

```python
help_p = sub.add_parser('help', help='Show full help')
help_p.add_argument('--grep', default=None, help='Search commands by keyword')
help_p.set_defaults(func=cmd_help)
```

### Step 2 — Add search logic to `cmd_help()`

In `cmd_help()` (~line 3027 of `nat`):

```python
def cmd_help(args=None):
    grep = getattr(args, 'grep', None) if args else None
    if grep:
        parser = build_parser()
        commands = []
        _walk_commands(parser, commands)
        term = grep.lower()
        matches = [c for c in commands
                   if term in c['name'].lower()
                   or term in c['help'].lower()]
        if matches:
            print(f"\n  {BOLD}Commands matching '{grep}' ({len(matches)}){W}\n")
            for m in matches:
                name = m['name'].replace(grep, f"{BOLD}{grep}{W}")
                print(f"    nat {name:40s} {m['help']}")
        else:
            print(f"\n  No commands matching '{grep}'")
        print()
        return
    # ... existing help text unchanged
```

### Step 3 — Extract `_walk_commands()` helper

Reuse the walking logic from `cmd_commands()` (~line 2564):

```python
def _walk_commands(parser_obj, commands, prefix=""):
    for action in parser_obj._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        help_map = {ca.dest: ca.help or "" for ca in action._choices_actions}
        for name, subparser in action.choices.items():
            full_name = f"{prefix} {name}".strip() if prefix else name
            commands.append({"name": full_name, "help": help_map.get(name, "")})
            _walk_commands(subparser, commands, full_name)
```

## Files to Modify

- `/home/onat/nat/nat` — `cmd_help()` (~line 3027), `build_parser()` (~line 3602)

## Acceptance Criteria

- [ ] `nat help --grep IC` returns commands containing "IC" in name or description
- [ ] `nat help --grep kalman` finds `nat kalman analysis` and `nat kalman drift`
- [ ] `nat help --grep paper` finds `nat alpha paper`, `nat trade-viz`, etc.
- [ ] Search is case-insensitive: `nat help --grep ALPHA` matches `nat alpha combine`
- [ ] `nat help --grep nonexistent` prints "No commands matching" message
- [ ] `nat help` (without `--grep`) still prints the full help text unchanged
- [ ] `nat help -h` shows the `--grep` option in argparse help

## Testing / Verification

```bash
# 1. Search for IC-related commands
nat help --grep IC
# Should list: signal test, profile scalp, algorithm evaluate, etc.

# 2. Search for paper trading
nat help --grep paper
# Should list: alpha paper, trade-viz, etc.

# 3. Case insensitive
nat help --grep KALMAN
# Should match kalman analysis, kalman drift

# 4. No matches
nat help --grep xyznonexistent
# Should print: "No commands matching 'xyznonexistent'"

# 5. Existing help unchanged
nat help | head -5
# Should show the same banner as before
```

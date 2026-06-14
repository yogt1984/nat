# Q1.3 — Implement Arch-p.1: SQLite Research Store

**Phase**: Q1 — Foundation & Data Quality
**Priority**: 3
**Status**: NOT STARTED
**Effort**: ~25h (across 12 items in Arch-p.1)
**Depends on**: None (can start immediately)

---

## Objective

Replace directory-scanning research data access with SQLite (WAL mode), add atomic writes, schema versioning, centralized symbol config, and API contract tests as specified in `docs/Arch-p.1.md`.

## Context

The Rust API currently scans `data/research/hypotheses/` and `data/research/cycles/` on every cache refresh — O(n) per scan, degrades with thousands of hypotheses. No cleanup policy, no schema versioning, no contract tests between Python writers and Rust readers. This blocks:
- **Quant path**: Systematic alpha screening (Q2.3) needs reliable research data storage for thousands of hypothesis evaluations
- **PhD path**: Reproducibility requires provenance tracking and schema-versioned records

Arch-p.1 contains 12 items across 4 priority tiers. Total effort ~25h.

## Prerequisites

- None (self-contained infrastructure work)

## Scope

**In scope** (Arch-p.1 items):
1. SQLite for research data (replace dir scanning) — ~4h
2. Atomic file writes (write-then-rename) — ~1h
3. Schema versioning on research output — ~1.5h
4. Centralize symbol configuration — ~2h
5. API contract tests — ~2-4h
6. Remove JSON state fallback — ~1.5h
7. Redis Pub/Sub → Streams upgrade — ~3h
8. Cross-service correlation IDs — ~2h
9. Research data retention policy — ~1h
10. Dependency injection for agents — ~2h
11. Gate strategy pattern — ~3h
12. Meta-agent coordination — ~3h

**Out of scope**:
- Arch-p.2 (unified Postgres, event bus) — deferred until >10 agents
- Arch-p.3 (pyproject.toml, provenance) — separate task Q1.4

## Steps

### Priority 1 — Data Integrity (~6.5h)
1. Add `research_output` table to `scripts/agent/state_store.py` with schema versioning
2. Update `research_output.py` to INSERT into SQLite (keep JSON as optional backup)
3. Replace `read_json_dir()` in `rust/api/src/routes/research.rs` with `rusqlite` reads
4. Add `rusqlite` dependency to `rust/api/Cargo.toml`
5. Implement write-then-rename pattern in `research_output.py` and `output/writer.rs`
6. Add `schema_version` field to all hypothesis and cycle records

### Priority 2 — Config & Contracts (~5.5h)
7. Create `config/symbols.toml`, add `load_symbols()` helper
8. Replace hardcoded symbol lists in 7 config files
9. Add contract tests in `rust/api/tests/contract.rs`
10. Remove JSON fallback from `AgentState`

### Priority 3-4 — Messaging & Code Quality (~14h)
11. Upgrade Redis Pub/Sub to Streams for research events
12. Add correlation IDs to cross-service logging
13. Implement 90-day retention policy
14. Refactor agent constructors for dependency injection
15. Extract gate protocol as strategy pattern
16. Add meta-agent budget and directives tables

## Acceptance Criteria

- [ ] `read_json_dir()` removed from Rust API — all research reads go through SQLite
- [ ] No `*.json.tmp` files left in `data/research/` after any operation
- [ ] Every research record has `schema_version` field
- [ ] `config/symbols.toml` is the single source of truth — no hardcoded `["BTC","ETH","SOL"]` in other configs
- [ ] At least 1 API contract test per endpoint in `rust/api/tests/`
- [ ] No `if self._path` branches remain in `scripts/agent/state_store.py`
- [ ] Redis Streams used for research events (Pub/Sub retained only for ephemeral feature streaming)
- [ ] Agent constructors accept injected dependencies
- [ ] All existing tests pass: `nat test agent` (350+ tests)

## Testing / Verification

```bash
# 1. Rust API tests including new contract tests
cd rust && cargo test --package api

# 2. Agent tests (350+ tests)
nat test agent

# 3. Verify SQLite research store
python3 -c "
from scripts.agent.state_store import StateStore
store = StateStore('data/agent/test_state.db')
# Should have research_output table
import sqlite3
conn = sqlite3.connect('data/agent/test_state.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print('Tables:', [t[0] for t in tables])
assert ('research_output',) in tables
"

# 4. Verify symbols centralization
python3 -c "
from scripts.config_utils import load_symbols
syms = load_symbols()
assert syms == ['BTC', 'ETH', 'SOL'], f'Unexpected: {syms}'
print('Symbols loaded from central config:', syms)
"

# 5. Verify no JSON fallback code
grep -r 'self._path' scripts/agent/state_store.py | wc -l
# Should be 0

# 6. Full pipeline smoke test
nat test pipeline
```

## Key Files

- `scripts/agent/state_store.py` — SQLite state management
- `scripts/agent/research_output.py` — hypothesis/cycle record writing
- `rust/api/src/routes/research.rs` — research data API endpoints
- `rust/api/Cargo.toml` — add rusqlite dependency
- `rust/ing/src/output/writer.rs` — atomic Parquet writes
- `config/symbols.toml` — centralized symbol config (new)
- `scripts/config_utils.py` — symbol loading helper (new)
- `config/agent.toml` — retention policy config

## References

- Full specification: `docs/Arch-p.1.md`
- 12 items with detailed implementation notes and code snippets

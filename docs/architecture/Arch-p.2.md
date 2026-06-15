# Architecture Phase 2 — Unified Data Plane

Phase 2 is the long-term structural refactor. Not needed at current scale (3 agents, single researcher, ~1000 hypotheses). Execute only if scaling to 10+ agents, multi-researcher, or multi-machine deployment becomes necessary.

**Prerequisite:** Complete Arch-p.1.md first.

---

## Design Principle

One store is authoritative. Everything else is a cache or projection.

```
┌──────────────────────────────────────────────────────┐
│                   Postgres (or SQLite WAL)            │
│                                                      │
│  Tables:                                             │
│    features_meta   — parquet file registry + schema  │
│    hypotheses      — full hypothesis records         │
│    cycles          — cycle summaries                 │
│    agent_state     — phase, cycle_count, config      │
│    registry        — validated signals + IC history  │
│    generator_stats — Thompson sampling weights       │
│    directives      — meta → agent commands           │
│    it_engine       — MI/CMI matrices per symbol      │
│    audit_log       — all state transitions           │
│                                                      │
│  Constraints:                                        │
│    - Foreign keys enforced                           │
│    - schema_version on every table                   │
│    - created_at / updated_at on all rows             │
│    - agent_type + hypothesis_id unique               │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     Rust Ingestor   Python      Rust API
     (WRITE only)    Agents      (READ only)
          │         (READ/WRITE)     │
          │            │            │
          └────────────┼────────────┘
                       │
                 Redis Streams
            (ephemeral notifications,
             NOT source of truth)
```

---

## Components

### 2.1 Canonical Database

**Choice:** Postgres if multi-machine; SQLite WAL if single-machine (current).

Single `nat.db` (or Postgres instance) replaces:
- `data/research/hypotheses/*.json` → `hypotheses` table
- `data/research/cycles/*.json` → `cycles` table
- `data/agent/agent_state.json` → `agent_state` table (already migrated in P1)
- `data/agent/registry.json` → `registry` table (already migrated in P1)
- `data/it_engine/state_*.json` → `it_engine` table
- `data/agent_meta/portfolio.json` → `portfolio` table
- Generator stats → `generator_stats` table (already migrated in P1)

Parquet files remain for bulk feature storage — they're optimized for columnar scans. But a `features_meta` table tracks what's on disk:
```sql
CREATE TABLE features_meta (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  date TEXT NOT NULL,
  path TEXT NOT NULL,
  row_count INTEGER,
  size_bytes INTEGER,
  schema_hash TEXT,
  created_at TEXT NOT NULL
);
```

This enables "what data do I have?" queries without scanning directories.

---

### 2.2 Shared Schema Package

A single schema definition consumed by both Rust and Python. Two options:

**Option A — Protobuf/FlatBuffers:**
```
schemas/
  hypothesis.proto
  cycle.proto
  feature_snapshot.proto
  agent_state.proto
```
Generate Rust structs and Python dataclasses from the same source. Guarantees wire compatibility.

**Option B — JSON Schema + codegen:**
```
schemas/
  hypothesis.json    (JSON Schema)
  cycle.json
```
Generate TypeScript types (frontend), Rust structs (API), Python dataclasses (agents) from same schema. Lighter weight, no protobuf toolchain.

Either eliminates the "types defined independently in each language" problem.

---

### 2.3 Event Bus with Guaranteed Delivery

Replace Redis Pub/Sub entirely with Redis Streams for all inter-service communication:

```
Streams:
  nat:events:research    — hypothesis lifecycle events (maxlen 50000)
  nat:events:features    — feature computation events (maxlen 10000)
  nat:events:alerts      — alerting events (maxlen 5000)
  nat:events:directives  — meta-agent → agent commands (maxlen 1000)

Consumer Groups:
  api-server       — reads research + alerts, forwards to WebSocket clients
  alert-service    — reads alerts, sends Telegram
  meta-agent       — reads research events for cross-agent coordination
```

Properties:
- At-least-once delivery (consumer acknowledges after processing)
- Replay from any point (crash recovery)
- Bounded (maxlen prevents unbounded growth)
- DB remains source of truth — streams are notifications, not state

---

### 2.4 Service Registry & Health

Replace "make run kills stale processes" with structured service management:

```toml
# config/services.toml
[ingestor]
binary = "rust/target/release/ing"
config = "config/ing.toml"
host = "su-35"
health_endpoint = "http://su-35:8080/health"
restart_policy = "always"

[api]
binary = "rust/target/release/nat-api"
health_endpoint = "http://localhost:3000/health"
restart_policy = "always"

[agent.microstructure]
command = "python3 scripts/agent/daemon.py start"
health_check = "python3 scripts/agent/daemon.py status"
restart_policy = "on-failure"

[agent.medium_freq]
command = "python3 scripts/agent/mf_daemon.py start"
health_check = "python3 scripts/agent/mf_daemon.py status"
restart_policy = "on-failure"

[meta_agent]
command = "python3 scripts/agent/meta_daemon.py start"
depends_on = ["agent.microstructure", "agent.medium_freq", "agent.macro"]
restart_policy = "on-failure"
```

Supervisor options: systemd units, docker-compose (already partial), or a lightweight Python supervisor that reads this config.

Aggregated health endpoint:
```
GET /health → {
  "ingestor": {"status": "ok", "last_emission": "2s ago"},
  "agents": {"microstructure": "SLEEPING", "medium_freq": "EXECUTE", "macro": "IDLE"},
  "redis": "connected",
  "db": "ok",
  "features_lag": "1.2s"
}
```

---

### 2.5 Rust API as Thin Read Layer

After the unified data plane, the Rust API becomes a pure query server:

```rust
// No more file scanning, no more cache TTL logic
async fn get_hypotheses(State(state): State<AppState>, Query(q): Query<HypothesisQuery>) -> Json<Page<Hypothesis>> {
    let conn = state.db.get().await?;
    let (items, total) = conn.query_hypotheses(&q).await?;
    Json(Page { items, total, offset: q.offset, limit: q.limit })
}
```

Benefits:
- No ResearchCache, no TTL, no staleness window
- Typed queries with proper filtering/pagination (SQL)
- Schema enforced at DB level, not at parse time
- Trivial to add new query patterns (by agent, by date range, by status)

---

### 2.6 Python Agents Write Through DB

Agents become DB-first:

```python
class ResearchAgent(ABC):
    def __init__(self, db: Database, event_bus: EventBus):
        self._db = db
        self._bus = event_bus

    def _register_hypothesis(self, h: Hypothesis):
        self._db.insert_hypothesis(h.to_record())
        self._bus.publish("nat:events:research", {
            "event": "hypothesis_registered",
            "id": h.id,
            "agent": self.agent_type,
        })
```

No more JSON file writes. The DB insertion is the canonical write. The event bus notification is best-effort (for real-time UI). If the event is lost, the UI refreshes from DB on next poll.

---

### 2.7 Reproducibility Layer

For PhD-grade reproducibility, add provenance tracking:

```sql
CREATE TABLE experiment_runs (
  id TEXT PRIMARY KEY,
  git_commit TEXT NOT NULL,
  config_snapshot JSON NOT NULL,    -- full merged config at runtime
  started_at TEXT NOT NULL,
  completed_at TEXT,
  hypothesis_ids JSON,              -- list of hypotheses produced
  environment JSON                  -- python version, package versions, machine
);
```

Every agent cycle records its git commit + config snapshot. Any result can be traced back to exact code + config state.

---

## Migration Path

The transition from P1 → P2 is incremental:

1. **DB already exists** (from P1 item 1.1) — extend tables, don't replace
2. **Add schema package** — generate types, swap in gradually per endpoint
3. **Migrate Rust API to DB reads** — one endpoint at a time, feature-flag old path
4. **Replace Pub/Sub with Streams** — consumer groups are backwards-compatible (old subscribers keep working during transition)
5. **Add service registry** — additive, doesn't break existing startup
6. **Add provenance** — purely additive table, no changes to existing flow

No big bang. Each step is independently deployable and reversible.

---

## When to Execute

Triggers that indicate P2 is needed:
- Adding a 4th+ research agent with different data requirements
- Multiple researchers needing isolated experiment runs
- Deploying agents across more than 2 machines
- Hypothesis count exceeding 10,000 (directory scanning even with SQLite becomes a bottleneck for complex queries)
- Need for audit trail / reproducibility (PhD, paper publication)

Until then, P1 is sufficient.

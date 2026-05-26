# Architecture Optimization Roadmap

Prioritized fixes for structural weaknesses identified in the NAT platform. Ordered by impact on research velocity and system reliability.

---

## Priority 1 — Data Integrity & Scalability

### 1.1 Replace Directory Scanning with SQLite for Research Data ✅

**Problem:** The Rust API scans `data/research/hypotheses/` and `data/research/cycles/` on every cache refresh. This is O(n) per scan and degrades as hypotheses accumulate into thousands. No cleanup policy — files grow forever. The 30s TTL cache means newly written hypotheses can be invisible for up to 30s.

**Current flow:**
```
Python agent → write JSON file → disk → Rust API scans directory → cache → serve
```

**Target flow:**
```
Python agent → INSERT into SQLite → Rust API reads SQLite (WAL mode) → serve
```

**Implementation:**
1. Extend `scripts/agent/state_store.py` with a `research_output` table:
   ```sql
   CREATE TABLE research_output (
     id TEXT PRIMARY KEY,
     kind TEXT NOT NULL,           -- 'hypothesis' | 'cycle'
     agent TEXT NOT NULL,
     generator TEXT,
     status TEXT,
     payload JSON NOT NULL,
     created_at TEXT NOT NULL,
     schema_version INTEGER DEFAULT 1
   );
   CREATE INDEX idx_ro_agent ON research_output(agent);
   CREATE INDEX idx_ro_kind ON research_output(kind);
   CREATE INDEX idx_ro_created ON research_output(created_at);
   ```
2. Update `research_output.py` to write to SQLite (keep JSON file write as optional backup during transition).
3. Replace `read_json_dir()` in `rust/api/src/routes/research.rs` with SQLite reads via `rusqlite`. WAL mode allows concurrent Python writes + Rust reads without locking.
4. Remove the `ResearchCache` TTL mechanism — SQLite queries are fast enough to serve directly.
5. Add a retention policy: archive hypotheses older than 90 days to a separate table or delete.

**Files:** `scripts/agent/state_store.py`, `scripts/agent/research_output.py`, `rust/api/Cargo.toml` (+rusqlite), `rust/api/src/routes/research.rs`, `rust/api/src/main.rs` (remove cache from AppState)

**Effort:** ~4h

---

### 1.2 Atomic File Writes (Write-Then-Rename) ✅

**Problem:** Python writes hypothesis JSON and Parquet files that Rust reads concurrently. If a read hits a partially-written file, it gets corrupted data or a parse error. Currently masked by small file sizes but not guaranteed safe.

**Implementation:**
1. In `research_output.py`, write to `{path}.tmp` then `os.rename()` to final path:
   ```python
   tmp = path.with_suffix(".json.tmp")
   tmp.write_text(json.dumps(record))
   tmp.rename(path)  # atomic on POSIX
   ```
2. Apply same pattern in Parquet writer rotation (`rust/ing/src/output/writer.rs`) — write to `.parquet.tmp`, rename on flush completion.
3. In Rust API `read_json_dir()`, skip files matching `*.tmp` glob.

**Files:** `scripts/agent/research_output.py`, `rust/ing/src/output/writer.rs`, `rust/api/src/routes/research.rs`

**Effort:** ~1h

---

### 1.3 Schema Versioning on Research Output ✅

**Problem:** Hypothesis JSON has no version marker. Schema changes (new gate type, renamed field) break silently — the Rust API uses `serde_json::Value` (untyped), so missing fields become null without error.

**Implementation:**
1. Add `"schema_version": 1` to every record in `build_hypothesis_record()` and `build_cycle_summary()`.
2. In Rust API, deserialize into typed structs with `#[serde(default)]` for backwards compatibility:
   ```rust
   #[derive(Deserialize)]
   struct HypothesisRecord {
       schema_version: Option<u32>,
       id: String,
       agent: String,
       // ...
   }
   ```
3. Add a migration function that normalizes old records (version 0 / missing) to current schema on read.
4. Log warnings for unknown schema versions rather than failing.

**Files:** `scripts/agent/research_output.py`, `rust/api/src/routes/research.rs`

**Effort:** ~1.5h

---

## Priority 2 — Configuration & Contracts

### 2.1 Centralize Symbol Configuration ✅

**Problem:** `["BTC", "ETH", "SOL"]` is hardcoded independently in 7 config files. Changing the symbol set requires editing every file, and forgetting one causes silent divergence.

**Implementation:**
1. Create `config/symbols.toml`:
   ```toml
   symbols = ["BTC", "ETH", "SOL"]
   ```
2. Add a `load_symbols()` helper in Python (`scripts/config_utils.py`):
   ```python
   def load_symbols() -> list[str]:
       return toml.load(ROOT / "config" / "symbols.toml")["symbols"]
   ```
3. Update `ing.toml` to reference: `symbols_file = "../config/symbols.toml"` (Rust reads this at startup).
4. Replace hardcoded symbol lists in `algorithms.toml`, `alpha.toml`, `discovery.toml`, `hypothesis_testing.toml`, `it_engine.toml`, `kalman.toml` with `symbols_file` references or have each Python script call `load_symbols()`.
5. Validate at startup: if a config still has an inline `symbols` key, log a deprecation warning.

**Files:** `config/symbols.toml` (new), `scripts/config_utils.py` (new helper), all 7 config files, `rust/ing/src/config.rs`

**Effort:** ~2h

---

### 2.2 API Contract Testing ✅

**Problem:** Frontend TypeScript interfaces and Rust API response shapes are coupled by convention. A field rename in Rust silently breaks the frontend. No integration test verifies the full chain.

**Implementation:**
1. Add a contract test in `rust/api/tests/` that starts the API, hits every endpoint, and asserts response shape against a snapshot:
   ```rust
   #[tokio::test]
   async fn contract_hypotheses_shape() {
       let resp = client.get("/api/research/hypotheses").send().await;
       let body: serde_json::Value = resp.json().await;
       // Assert top-level keys exist
       assert!(body["items"].is_array());
       assert!(body["total"].is_number());
       // Assert item shape
       let item = &body["items"][0];
       assert!(item["id"].is_string());
       assert!(item["agent"].is_string());
       assert!(item["gates"].is_array());
   }
   ```
2. Add corresponding TypeScript type assertion tests in `web/src/__tests__/api-contract.test.ts` that validate mock data matches the interface shapes.
3. Long-term: generate an OpenAPI spec from Rust types using `utoipa` and generate TypeScript types from it.

**Files:** `rust/api/tests/contract.rs` (new), `web/src/__tests__/api-contract.test.ts` (new)

**Effort:** ~2h (manual contract tests), ~4h (utoipa + codegen)

---

### 2.3 Remove JSON State Fallback ✅

**Problem:** Every `load()`/`save()` in `AgentState` branches on `if self._store` (SQLite) vs JSON path. The JSON fallback is legacy — SQLite migration is done and tested (63 tests). The dual-mode code doubles the surface area for state bugs.

**Implementation:**
1. Grep for all `if self._store` / `if self._path` branches in `scripts/agent/`.
2. Remove the JSON code paths. Keep only SQLite.
3. Remove `_auto_migrate()` (migration is a one-time operation, already run).
4. Simplify `AgentState.__init__` to require a `StateStore` — no optional path parameter.
5. Update tests that exercise JSON mode to use SQLite.

**Files:** `scripts/agent/state_store.py`, `scripts/agent/base.py`, tests in `scripts/tests/`

**Effort:** ~1.5h

---

## Priority 3 — Messaging & Observability

### 3.1 Upgrade Redis Pub/Sub to Streams ✅

**Problem:** Redis Pub/Sub is fire-and-forget. Messages published with no subscribers are lost permanently. If the API restarts during an agent cycle, all events during downtime are gone. No replay capability.

**Implementation:**
1. Replace `PUBLISH nat:research:events` with `XADD nat:research:stream`:
   ```python
   # research_output.py
   redis.xadd("nat:research:stream", {"event": json.dumps(payload)}, maxlen=10000)
   ```
2. In Rust API WebSocket handler, use `XREAD` with consumer group instead of `SUBSCRIBE`:
   ```rust
   // On connect: read from last-seen ID
   // On reconnect: replay missed events from stored ID
   redis.xread_group("nat-api", "ws-handler", &["nat:research:stream"], &[">"])
   ```
3. Keep `maxlen=10000` (bounded, ~24h of events at current volume) to prevent unbounded growth.
4. For feature streaming (`nat:features:{symbol}`), keep Pub/Sub — these are ephemeral and loss-tolerant.

**Files:** `scripts/agent/research_output.py`, `rust/api/src/routes/ws.rs`, `docker-compose.yml` (ensure Redis 7+ for streams)

**Effort:** ~3h

---

### 3.2 Cross-Service Correlation IDs ✅

**Problem:** No tracing from "hypothesis generated in Python" through "JSON written" → "cache refreshed in Rust" → "WebSocket event delivered to browser". Python has `cycle_id` and `hypothesis_id` in structured logs, but Rust doesn't propagate them.

**Implementation:**
1. Ensure every Redis event includes `hypothesis_id` and `cycle_id` (already partially done in `research_output.py`).
2. In Rust API, extract `hypothesis_id` from incoming Redis messages and attach to response/log context:
   ```rust
   tracing::info!(hypothesis_id = %id, "forwarding event to WebSocket clients");
   ```
3. In frontend WebSocket handler, log correlation IDs to browser console in development mode.
4. Standardize log levels: Redis connection failures should be `warn` in both Python and Rust (currently `debug` in Python, `error` in Rust).

**Files:** `scripts/agent/research_output.py`, `scripts/logging_config.py`, `rust/api/src/routes/ws.rs`

**Effort:** ~2h

---

### 3.3 Research Data Retention Policy ✅

**Problem:** Hypothesis JSON files and cycle summaries accumulate indefinitely in `data/research/`. No cleanup, no archival. With agents running continuously, this becomes thousands of files within weeks.

**Implementation:**
1. Add a `[research.retention]` section to `config/agent.toml`:
   ```toml
   [research.retention]
   max_age_days = 90
   archive_dir = "data/research/archive"
   ```
2. At the end of each agent cycle, run a cleanup pass:
   ```python
   def cleanup_old_records(research_dir, max_age_days, archive_dir):
       cutoff = datetime.now() - timedelta(days=max_age_days)
       for f in research_dir.glob("*.json"):
           record = json.loads(f.read_text())
           if parse_datetime(record["completed_at"]) < cutoff:
               shutil.move(f, archive_dir / f.name)
   ```
3. If SQLite is adopted (item 1.1), this becomes a simple `DELETE WHERE created_at < ?` with optional dump to archive.

**Files:** `config/agent.toml`, `scripts/agent/research_output.py` or `scripts/agent/base.py`

**Effort:** ~1h

---

## Priority 4 — Code Quality & Testability

### 4.1 Dependency Injection for Python Agents

**Problem:** `StateStore`, `ReportCache`, Redis connections, and generator modules are created inside agent constructors. Unit testing requires `unittest.mock.patch` on import paths, making tests brittle and tightly coupled.

**Implementation:**
1. Refactor `ResearchAgent.__init__` to accept dependencies:
   ```python
   class ResearchAgent(ABC):
       def __init__(self, store: StateStore, redis: Optional[Redis] = None,
                    cache: Optional[ReportCache] = None):
           self._store = store
           self._redis = redis or _default_redis()
           self._cache = cache or ReportCache()
   ```
2. Update subclass constructors to pass through.
3. In tests, inject in-memory SQLite and stub Redis:
   ```python
   store = StateStore(":memory:")
   agent = MicrostructureAgent(store=store, redis=FakeRedis())
   ```
4. Keep the current no-args construction path for CLI entry points (daemon.py `main()`), where defaults are fine.

**Files:** `scripts/agent/base.py`, `scripts/agent/daemon.py`, `scripts/agent/mf_daemon.py`, `scripts/agent/macro_daemon.py`, test files

**Effort:** ~2h

---

### 4.2 Extract Gate Protocol as Strategy Pattern

**Problem:** Gate logic lives inside `BaseRunner` methods. Adding domain-specific gates requires subclassing the runner. Gate composition and reordering requires changing `steps()` return list.

**Implementation:**
1. Define a `Gate` protocol:
   ```python
   class Gate(Protocol):
       name: str
       def evaluate(self, hypothesis: Hypothesis, data: pd.DataFrame) -> GateResult: ...

   @dataclass
   class GateResult:
       passed: bool
       metric: float
       threshold: float
       message: str
   ```
2. Convert existing gate checks (`check_ic_gate`, `check_cost_gate`, etc.) into Gate classes.
3. Runner `steps()` returns `list[Gate]` instead of `list[Callable]`.
4. `run_full()` iterates gates uniformly — no special-casing per runner type.

**Files:** `scripts/agent/gates.py` (new), `scripts/agent/base.py`, `scripts/agent/runner.py`, `scripts/agent/mf_runner.py`, `scripts/agent/macro_runner.py`

**Effort:** ~3h

---

### 4.3 Strengthen Meta-Agent Coordination

**Problem:** `MetaAgent` reads other agents' registries via SQLite but cannot direct them (pause a generator, enforce budget allocation, force a re-test). Budget allocation is computed but agents don't consume it.

**Implementation:**
1. Add a `budget` table to `StateStore`:
   ```sql
   CREATE TABLE budget (
     agent TEXT PRIMARY KEY,
     max_hypotheses_per_cycle INTEGER,
     updated_at TEXT
   );
   ```
2. `MetaAgent.allocate_budget()` writes to this table.
3. In `ResearchAgent._run_cycle()`, read budget before generating:
   ```python
   budget = self._store.get_budget(self.agent_type)
   if budget and self._cycle_tested >= budget.max_hypotheses_per_cycle:
       break
   ```
4. Add a `directives` table for meta-agent → agent communication:
   ```sql
   CREATE TABLE directives (
     id INTEGER PRIMARY KEY,
     target_agent TEXT NOT NULL,
     action TEXT NOT NULL,   -- 'pause_generator', 'retest_hypothesis', 'adjust_threshold'
     payload JSON,
     consumed INTEGER DEFAULT 0,
     created_at TEXT
   );
   ```
5. Agents check for pending directives at cycle start.

**Files:** `scripts/agent/state_store.py`, `scripts/agent/meta_daemon.py`, `scripts/agent/base.py`

**Effort:** ~3h

---

## Summary

| # | Item | Priority | Effort | Impact |
|---|------|----------|--------|--------|
| 1.1 | SQLite for research data | P1 | ~4h | Eliminates O(n) dir scans, enables retention, typed queries |
| 1.2 | Atomic file writes | P1 | ~1h | Prevents race condition corruption |
| 1.3 | Schema versioning | P1 | ~1.5h | Safe schema evolution, detects drift |
| 2.1 | Centralize symbols | P2 | ~2h | Single source of truth for asset config |
| 2.2 | API contract tests | P2 | ~2-4h | Catches frontend/backend drift before deploy |
| 2.3 | Remove JSON fallback | P2 | ~1.5h | Halves state management code paths |
| 3.1 | Redis Streams ✅ | P3 | ~3h | Reliable event delivery with replay |
| 3.2 | Correlation IDs ✅ | P3 | ~2h | End-to-end tracing across services |
| 3.3 | Retention policy ✅ | P3 | ~1h | Prevents unbounded file/data growth |
| 4.1 | Dependency injection | P4 | ~2h | Testable agents without mock.patch |
| 4.2 | Gate strategy pattern | P4 | ~3h | Composable, reorderable gate protocol |
| 4.3 | Meta-agent coordination | P4 | ~3h | Agents respect cross-agent budget/directives |

**Total:** ~25-27h across 12 items.

P1 items (6.5h) should be done before the next major agent expansion. P2 items (5.5-7.5h) prevent config/contract rot. P3-P4 (14h) are structural improvements that pay off as the system scales.

# NAT Engineering Backlog

Backlog derived from architecture review (2026-05-25). Organized by phase, priority within each phase, and estimated effort.

---

## Phase 1: Foundation (Backend Refactoring)

No new features. Fix the plumbing so everything built on top is solid.

### P1-1. Unified Data Access Layer

**Priority**: Critical
**Effort**: Medium (2-3 days)
**Files to create**: `scripts/data/features.py`, `scripts/data/catalog.py`, `scripts/data/schema.py`
**Files to modify**: Every script that globs `data/features/` independently

**Problem**: Every script (paper_trader, backtest, agents, pipeline runner) reimplements parquet loading with ad-hoc column filtering. One schema change breaks N scripts silently.

**Tasks**:
- [ ] Create `scripts/data/features.py` with `load_features(symbol, date_range, columns=None) -> pd.DataFrame`
- [ ] Create `scripts/data/catalog.py` with `available_dates()`, `available_symbols()`, `data_health_check()`
- [ ] Create `scripts/data/schema.py` with `FeatureSchema` class: version tracking, column validation, NaN-padding audit
- [ ] Migrate `scripts/alpha/paper_trader.py` inline `load_date()` to use unified loader
- [ ] Migrate `scripts/alpha/paper_trader_generic.py` to use unified loader
- [ ] Migrate `scripts/analysis/mf_liquidity_backtest.py` to use unified loader
- [ ] Migrate `scripts/cluster_pipeline/loader.py` to delegate to unified loader
- [ ] Migrate agent data loading in `scripts/agent/runner.py`, `mf_runner.py`, `macro_runner.py`
- [ ] Add integration test: schema change detected and reported

**Acceptance**: A single `from data.features import load_features` replaces all ad-hoc parquet globbing. Schema validation catches column renames at load time.

---

### P1-2. Consolidate Agent Daemons

**Priority**: High
**Effort**: Medium (2-3 days)
**Files to modify**: `scripts/agent/base.py`, `daemon.py`, `mf_daemon.py`, `macro_daemon.py`

**Problem**: 1,425 LOC across 3 daemon files with ~70% identical code. Only differences are generator configs, state paths, cycle intervals, and timeframe strings.

**Tasks**:
- [ ] Extract daemon main loop into `ResearchAgent` base class (currently defined but not fully leveraged)
- [ ] Move `load_config()`, `load_gen_stats()`, `save_gen_stats()` into base class
- [ ] Parameterize generator list, cycle interval, and state directory via config section name
- [ ] Reduce `mf_daemon.py` and `macro_daemon.py` to thin subclasses (~50 LOC each, override only generator-specific logic)
- [ ] Update `macro_daemon.py` CLI entry point to use consolidated class
- [ ] Verify all three agents still produce identical outputs after consolidation
- [ ] Update Makefile targets if entry points change

**Acceptance**: Total daemon LOC drops from 1,425 to ~600. Adding a 4th agent requires only a new TOML section and ~30 LOC subclass.

---

### P1-3. Consolidate Agent Runners

**Priority**: High
**Effort**: Medium (1-2 days)
**Files to modify**: `scripts/agent/base.py`, `runner.py`, `mf_runner.py`, `macro_runner.py`

**Problem**: 1,163 LOC across 3 runner files. `mf_runner` and `macro_runner` import 6 functions from `runner.py` but still reimplement `run_discovery()`, `run_replication_temporal()`, `run_replication_symbol()`, `run_correlation_check()` with ~90% identical code.

**Tasks**:
- [ ] Move gate threshold checks into `BaseRunner` as configurable parameters (not hardcoded per subclass)
- [ ] Extract `run_discovery()`, `run_replication_temporal()`, `run_replication_symbol()`, `run_correlation_check()` into `BaseRunner` with timeframe as a parameter
- [ ] Reduce `mf_runner.py` and `macro_runner.py` to gate-threshold config + any genuinely different logic
- [ ] Add test: all three runners produce identical gate evaluation given identical inputs and thresholds

**Acceptance**: Total runner LOC drops from 1,163 to ~500. Gate thresholds live in config, not code.

---

### P1-4. Replace JSON State Files with SQLite

**Priority**: High
**Effort**: Medium (2-3 days)
**Files to create**: `scripts/data/state.py`
**Files to modify**: `scripts/agent/base.py`, `scripts/pipeline_runner.py`, `scripts/alpha/alpha_pipeline.py`

**Problem**: 5 independent JSON files with no coordination, no atomic writes, no queryability. Crash during write corrupts state. Cannot answer "what is the system doing?" without reading 5 files.

**Tasks**:
- [ ] Create `scripts/data/state.py` with SQLite-backed `StateStore` class
- [ ] Define tables: `pipeline_runs`, `hypotheses`, `signals`, `trades`, `agent_cycles`
- [ ] Implement atomic state transitions (BEGIN/COMMIT around state changes)
- [ ] Migrate `pipeline_runner.py` state persistence to SQLite
- [ ] Migrate `alpha_pipeline.py` state persistence to SQLite
- [ ] Migrate agent state (`agent_state.json`) to SQLite
- [ ] Migrate hypothesis queue (`hypotheses.json`) to SQLite table
- [ ] Migrate signal registry (`registry.json`) to SQLite table
- [ ] Add `status` CLI command that queries SQLite and prints system-wide status
- [ ] Keep JSON export for backward compatibility (read from SQLite, write JSON for dashboard)
- [ ] Write crash-recovery test: kill mid-cycle, restart, verify state consistency

**Acceptance**: Single `data/nat.db` file. `python -m scripts.data.state status` prints all pipeline states, active agents, and hypothesis counts.

---

### P1-5. Config Inheritance and Deduplication

**Priority**: Medium
**Effort**: Small (1 day)
**Files to modify**: `config/agent.toml`, loader code in `scripts/agent/base.py`

**Problem**: `symbols = ["BTC", "ETH", "SOL"]` appears 4 times. `fdr_q = 0.05`, `ic_decay_ratio = 0.5`, `data_dir = "data/features"` duplicated across all agent sections.

**Tasks**:
- [ ] Add `[agent_base]` section in `agent.toml` with shared defaults (symbols, data_dir, fdr_q, ic_decay_ratio)
- [ ] Implement config merging in agent loader: `agent_mf` inherits from `agent_base`, overrides only what differs
- [ ] Remove duplicated keys from `[agent]`, `[agent_mf]`, `[agent_macro]` sections
- [ ] Add config validation: unknown keys → warning, missing required keys → error
- [ ] Document config schema in a comment block at top of `agent.toml`

**Acceptance**: Adding a new agent section requires only the keys that differ from base. Typos in key names produce warnings.

---

### P1-6. Structured Logging

**Priority**: Medium
**Effort**: Medium (2-3 days)
**Files to create**: `scripts/logging_config.py`
**Files to modify**: All Python scripts with `print()` calls (incremental)

**Problem**: 2,057 `print()` calls vs structured logging. No correlation IDs, no log levels, no centralized sink. Debugging production on su-35 requires grepping stdout.

**Tasks**:
- [ ] Create `scripts/logging_config.py` with `setup_logging(name, level, log_dir)` using `structlog` or stdlib `logging` with JSON formatter
- [ ] Replace `logging.basicConfig()` calls in all daemons with centralized setup
- [ ] Convert `print()` to `log.info/debug/warning` in critical paths first:
  - [ ] `scripts/agent/daemon.py` (and mf/macro variants)
  - [ ] `scripts/agent/runner.py` (and mf/macro variants)
  - [ ] `scripts/alpha/alpha_pipeline.py`
  - [ ] `scripts/pipeline_runner.py`
  - [ ] `scripts/execution/signal_bridge.py`
- [ ] Add correlation ID (cycle_id or hypothesis_id) to log context in agent loops
- [ ] Log to `data/logs/YYYY-MM-DD.jsonl` with rotation

**Acceptance**: `grep hypothesis_id data/logs/2026-05-25.jsonl` traces a single hypothesis through all gates. No new `print()` calls in modified files.

---

### P1-7. Dashboard Caching

**Priority**: Low
**Effort**: Small (half day)
**Files to modify**: `scripts/agent_dashboard.py`

**Problem**: Dashboard reads and parses JSON from disk on every HTTP request. At 10s auto-refresh that's 6 full parses/sec for zero benefit.

**Tasks**:
- [ ] Add in-memory cache with 60s TTL for `read_state()`, `build_heatmap_data()`, and all `/api/*` endpoints
- [ ] Add `Last-Modified` / `If-Modified-Since` headers to avoid re-sending unchanged data
- [ ] Add basic request logging (method, path, status, latency)
- [ ] Replace bare `except` handlers with specific exception types + logging

**Acceptance**: Disk I/O drops ~99%. Dashboard still reflects updates within 60s.

---

### P1-8. Integration Tests for Daemon Cycles

**Priority**: Medium
**Effort**: Medium (2 days)
**Files to create**: `scripts/tests/test_daemon_integration.py`, `scripts/tests/test_multi_agent.py`

**Problem**: 62 test files exist but no tests for the full MANIFEST → GENERATE → EXECUTE → REGISTER cycle. No multi-agent coordination tests. No crash-recovery tests.

**Tasks**:
- [ ] Create fixture with synthetic parquet data (small, deterministic, 3 dates × 2 symbols)
- [ ] Test single-agent full cycle: manifest scan → hypothesis generation → 5-gate execution → registration
- [ ] Test state persistence: run half a cycle, kill, restart, verify resume from correct state
- [ ] Test FDR control: inject N hypotheses with known p-values, verify BH correction produces expected survivors
- [ ] Test multi-agent: run micro + MF agents on same data, verify Meta Agent deduplicates correlated signals
- [ ] Test config validation: missing keys, invalid thresholds, unknown sections

**Acceptance**: `pytest scripts/tests/test_daemon_integration.py` passes. CI catches state machine regressions.

---

## Phase 2: Research API

Expose structured hypothesis data through the existing Axum API server.

### P2-1. Structured Hypothesis Output

**Priority**: Critical
**Effort**: Medium (2 days)
**Files to modify**: `scripts/agent/runner.py`, `scripts/agent/base.py`

**Problem**: Hypothesis results are scattered across JSON files without a consistent schema. The website needs structured, queryable data.

**Tasks**:
- [ ] Define hypothesis JSON schema with fields: id, agent, generator, claim, math (LaTeX source), gates (per-gate results), status, timestamps
- [ ] Modify runners to emit structured JSON per hypothesis to `data/research/hypotheses/`
- [ ] Include math derivation field (LaTeX string) per generator type
- [ ] Include full gate details: metric value, threshold, p-value, result
- [ ] Emit cycle summary JSON to `data/research/cycles/` after each agent cycle
- [ ] Write schema validation test

**Acceptance**: Each hypothesis produces a self-contained JSON with enough detail to render a full research page.

---

### P2-2. Research REST Endpoints

**Priority**: High
**Effort**: Medium (2-3 days)
**Files to modify**: `rust/api/src/` (add research module)

**Problem**: No API to query hypothesis history, signal registry, or cycle reports. The existing Axum API only serves live features.

**Tasks**:
- [ ] Add `GET /api/research/hypotheses` — paginated list, filterable by agent/generator/status/date
- [ ] Add `GET /api/research/hypotheses/:id` — full detail including math and gate results
- [ ] Add `GET /api/research/signals` — registered signals with IC history
- [ ] Add `GET /api/research/cycles` — cycle summaries (hypotheses tested, passed, FDR budget remaining)
- [ ] Add `GET /api/research/stats` — aggregate metrics (total tested, pass rate per gate, FDR budget)
- [ ] Add `GET /api/research/heatmap` — feature × horizon IC matrix data
- [ ] Read from SQLite (P1-4) or structured JSON files (P2-1)
- [ ] Add OpenAPI/Swagger documentation for all endpoints

**Acceptance**: `curl localhost:3000/api/research/hypotheses?agent=microstructure&status=REGISTERED` returns structured JSON.

---

### P2-3. WebSocket Research Stream

**Priority**: Medium
**Effort**: Small (1 day)
**Files to modify**: `rust/api/src/` (WebSocket handler)

**Problem**: No real-time updates when hypotheses complete. The dashboard must poll.

**Tasks**:
- [ ] Add `WS /ws/research` endpoint
- [ ] Publish events: `hypothesis_started`, `gate_passed`, `gate_failed`, `hypothesis_registered`, `cycle_completed`
- [ ] Use existing Redis pub/sub infrastructure (publish from Python agents, subscribe in Rust API)
- [ ] Add Python helper: `publish_research_event(event_type, payload)` using Redis

**Acceptance**: WebSocket client receives live events as hypotheses progress through gates.

---

## Phase 3: Research Website (Frontend)

Interactive web application for visualizing the hypothesis engine.

### P3-1. Project Scaffolding

**Priority**: Critical
**Effort**: Small (1 day)
**Files to create**: `web/` directory (Next.js app)

**Tasks**:
- [ ] Initialize Next.js project in `web/` with TypeScript
- [ ] Configure API proxy to Axum backend (port 3000)
- [ ] Install dependencies: Plotly.js, D3.js, KaTeX, Tailwind CSS
- [ ] Create layout: sidebar navigation, header with system status indicator
- [ ] Add Docker service for frontend in `docker-compose.yml`
- [ ] Add Makefile targets: `make web_dev`, `make web_build`

**Acceptance**: `make web_dev` starts frontend on localhost:3001 with hot reload, proxying API calls to localhost:3000.

---

### P3-2. Dashboard Page

**Priority**: Critical
**Effort**: Medium (2-3 days)

**Content**: System-wide overview at a glance.

**Tasks**:
- [ ] Agent status cards (micro/MF/macro): current phase, cycle count, last cycle time, hypotheses in queue
- [ ] Live cycle progress ring: animated arc showing current phase within the cycle state machine
- [ ] Recent hypothesis feed: last 20 hypotheses with status badges (PASS/FAIL/TESTING), auto-updates via WebSocket
- [ ] Aggregate stats bar: total tested / total registered / FDR budget remaining / active signals
- [ ] System health indicators: ingestor uptime, data freshness, last parquet timestamp per symbol
- [ ] Connect to `GET /api/research/stats` and `WS /ws/research`

---

### P3-3. Hypothesis Explorer Page

**Priority**: High
**Effort**: Medium (2-3 days)

**Content**: Searchable, filterable table of all hypotheses with drill-down.

**Tasks**:
- [ ] Sortable table: hypothesis ID, agent, generator, claim (truncated), IC, status, date
- [ ] Filters: agent type, generator, status (REGISTERED/GRAVEYARD/TESTING), date range, IC range
- [ ] Gate funnel visualization: animated funnel showing N tested → N pass G1 → ... → N registered
- [ ] Sankey diagram: generator → gate outcome flow (which generators produce which failure modes)
- [ ] Click row → navigate to Signal Detail page
- [ ] Paginated with infinite scroll or explicit pages
- [ ] Connect to `GET /api/research/hypotheses`

---

### P3-4. Signal Detail Page

**Priority**: High
**Effort**: Medium (3-4 days)

**Content**: Deep-dive into a single hypothesis.

**Tasks**:
- [ ] Header: hypothesis ID, claim, agent, generator, status badge, timestamps
- [ ] Gate waterfall chart: horizontal bar chart showing each gate's metric value vs threshold, color-coded PASS/FAIL
- [ ] Math derivation panel: KaTeX-rendered equations from the hypothesis `math` field, expandable sections for full derivation
- [ ] IC time series: D3/Plotly line chart showing IC over time with confidence bands
- [ ] Walk-forward equity curve: overlaid IS/OOS curves, gap = overfitting indicator
- [ ] Per-symbol breakdown: IC and Sharpe per symbol (BTC/ETH/SOL) as small multiples
- [ ] Related hypotheses: list of hypotheses with correlation > 0.3 (from orthogonality gate data)
- [ ] Paper references: clickable links to cited papers (Cont2014, Kyle1985, etc.)

---

### P3-5. IC Heatmap Page

**Priority**: High
**Effort**: Medium (2 days)

**Content**: Feature x horizon information coefficient matrix.

**Tasks**:
- [ ] Interactive heatmap (Plotly): rows = features, columns = horizons, cells = IC values
- [ ] Color scale: diverging (red negative, blue positive), saturation = significance
- [ ] Click cell → filter Hypothesis Explorer to that feature/horizon combo
- [ ] Toggle: raw IC vs FDR-adjusted IC vs cost-adjusted IC
- [ ] Row clustering: group features by category (flow, entropy, trend, etc.)
- [ ] Export as PNG/SVG
- [ ] Connect to `GET /api/research/heatmap`

---

### P3-6. Signal Registry Page

**Priority**: Medium
**Effort**: Medium (2-3 days)

**Content**: Promoted signals with live performance monitoring.

**Tasks**:
- [ ] Registry table: signal ID, agent, features used, IC (current + history), Sharpe, status
- [ ] IC decay curves: per-signal line chart with retirement threshold as horizontal reference line, signals crossing threshold turn red
- [ ] Correlation matrix heatmap: pairwise correlation between all registered signals
- [ ] Portfolio weights treemap: proportional area per signal, colored by agent type
- [ ] Signal lineage: which generator produced it, which gate scores, which cycle
- [ ] Connect to `GET /api/research/signals`

---

### P3-7. Math Lab Page

**Priority**: Medium
**Effort**: Medium (2-3 days)

**Content**: Centralized mathematical documentation for all methods.

**Tasks**:
- [ ] Organized by category: Feature Definitions, Gate Protocols, Signal Combination, Position Sizing
- [ ] KaTeX-rendered equations with expandable derivations
- [ ] Feature definitions: closed-form formula for each of the 217 features (source from FEATURES.md)
- [ ] Gate protocol math: IC definition, FDR (Benjamini-Hochberg), temporal replication bootstrap, orthogonality (R^2)
- [ ] Convolver math: SVD decomposition, cosine similarity scoring, kernel selection
- [ ] Liquidity signal math: z-score normalization, composite construction, percentile thresholds
- [ ] Position sizing: Kelly criterion derivation, cost adjustment
- [ ] Paper references with DOI links
- [ ] Search within equations (full-text search on LaTeX source)

---

### P3-8. Graveyard Page

**Priority**: Low
**Effort**: Small (1-2 days)

**Content**: Failed hypotheses with failure analysis — learn from what didn't work.

**Tasks**:
- [ ] Failure mode distribution: pie/donut chart by gate (G1 failures, G2 failures, etc.)
- [ ] Most common failure reasons per generator: bar chart
- [ ] "Near misses" table: hypotheses that passed 4/5 gates (closest to promotion)
- [ ] Failure trends over time: are generators improving or producing more failures?
- [ ] Recyclable hypotheses: failed hypotheses with IC > threshold but failed on replication (candidates for parameter tuning)
- [ ] Connect to `GET /api/research/hypotheses?status=GRAVEYARD`

---

### P3-9. Feature Interaction Network

**Priority**: Low
**Effort**: Medium (2-3 days)

**Content**: Visual exploration of feature relationships discovered by agents.

**Tasks**:
- [ ] D3 force-directed graph: nodes = features, edges = conditional mutual information > threshold
- [ ] Node size = number of hypotheses using that feature
- [ ] Node color = feature category (flow, entropy, trend, etc.)
- [ ] Edge thickness = mutual information strength
- [ ] Highlight clusters of co-discovered features
- [ ] Click node → filter Hypothesis Explorer to hypotheses using that feature
- [ ] Toggle: show only registered signal features vs all tested features

---

## Phase 4: Polish and Production

### P4-1. Real-Time WebSocket Updates

**Priority**: Medium
**Effort**: Small (1-2 days)

**Tasks**:
- [ ] Connect Dashboard page to `WS /ws/research` for live hypothesis feed
- [ ] Animate new hypotheses appearing in the feed (slide-in transition)
- [ ] Flash agent status card when a gate is passed/failed
- [ ] Update aggregate stats in real-time without full page refresh
- [ ] Add connection status indicator (connected/reconnecting/disconnected)

---

### P4-2. PDF Export per Hypothesis

**Priority**: Low
**Effort**: Medium (2 days)

**Tasks**:
- [ ] "Export PDF" button on Signal Detail page
- [ ] Generate LaTeX from hypothesis JSON (template with KaTeX → LaTeX conversion)
- [ ] Compile via pdflatex (server-side, same toolchain as existing docs)
- [ ] Include: claim, math derivation, gate results table, IC chart (embedded PNG), equity curve
- [ ] Return PDF as download

---

### P4-3. CI/CD for Frontend

**Priority**: Medium
**Effort**: Small (1 day)

**Tasks**:
- [ ] Add `web/` to CI pipeline (lint, type-check, build)
- [ ] Add Dockerfile for production frontend build (nginx serving static)
- [ ] Update `docker-compose.yml` with frontend service
- [ ] Add Makefile target: `make web_deploy`

---

### P4-4. Alert Integration

**Priority**: Low
**Effort**: Small (1 day)

**Tasks**:
- [ ] Telegram notification when a hypothesis is promoted to REGISTERED
- [ ] Telegram notification when a registered signal's IC crosses decay threshold
- [ ] Telegram daily digest: hypotheses tested, registered, retired in last 24h
- [ ] Use existing `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` infrastructure

---

## Summary

| Phase | Tasks | Estimated Effort | Focus |
|-------|-------|-----------------|-------|
| Phase 1 | P1-1 through P1-8 | 2-3 weeks | Backend plumbing, tech debt |
| Phase 2 | P2-1 through P2-3 | 1 week | API layer for research data |
| Phase 3 | P3-1 through P3-9 | 3-4 weeks | Research website |
| Phase 4 | P4-1 through P4-4 | 1 week | Polish and integration |

**Critical path**: P1-1 (data layer) → P1-4 (SQLite state) → P2-1 (structured output) → P2-2 (API) → P3-1 (scaffold) → P3-2 (dashboard).

**Parallel work**: P1-2/P1-3 (daemon consolidation) can proceed independently from P1-1. P3-7 (Math Lab) can be built from existing FEATURES.md without waiting for the API.

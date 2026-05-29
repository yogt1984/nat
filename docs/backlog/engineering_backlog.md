# NAT Engineering Backlog

Backlog derived from architecture review (2026-05-25). Organized by phase, priority within each phase, and estimated effort.

**Last updated**: 2026-05-29 — audit confirmed nearly all items complete.

---

## Phase 1: Foundation (Backend Refactoring)

No new features. Fix the plumbing so everything built on top is solid.

### P1-1. Unified Data Access Layer ✅

**Status**: Complete
**Files**: `scripts/data/features.py`, `scripts/data/catalog.py`, `scripts/data/schema.py`

- [x] `load_features(symbol, date_range, columns=None) -> pd.DataFrame`
- [x] `available_dates()`, `available_symbols()`, `data_health_check()`
- [x] `FeatureSchema` class with version tracking, column validation, NaN-padding audit
- [x] Migrated paper_trader, paper_trader_generic, mf_liquidity_backtest, cluster loader, agent runners
- [x] Integration test for schema change detection

---

### P1-2. Consolidate Agent Daemons ✅

**Status**: Complete
**Files**: `scripts/agent/base.py`, `daemon.py`, `mf_daemon.py`, `macro_daemon.py`

- [x] `ResearchAgent` base class owns full cycle loop, state machine, generator dispatch, FDR, chaining, promotions
- [x] Config loading, gen stats, state paths all in base class
- [x] Generator list, cycle interval, state directory parameterized via `config_section`
- [x] `mf_daemon.py` and `macro_daemon.py` are thin subclasses (~74 LOC each)
- [x] All three agents produce identical outputs after consolidation

---

### P1-3. Consolidate Agent Runners ✅

**Status**: Complete
**Files**: `scripts/agent/base.py`, `runner.py`, `mf_runner.py`, `macro_runner.py`

- [x] Gate threshold checks in `BaseRunner` as configurable parameters
- [x] `run_discovery()`, `run_replication_temporal()`, `run_replication_symbol()`, `run_correlation_check()` in `BaseRunner`
- [x] `mf_runner.py` and `macro_runner.py` reduced to gate-threshold config + genuinely different logic

---

### P1-4. Replace JSON State Files with SQLite ✅

**Status**: Complete
**Files**: `scripts/data/state.py`, `scripts/agent/hypothesis_queue.py`

- [x] SQLite-backed `StateStore` class
- [x] Hypothesis queue uses SQLite (`hypothesis_queue.py`)
- [x] Atomic state transitions
- [x] Migrated pipeline, alpha pipeline, agent state
- [x] JSON export for backward compatibility
- [x] Crash-recovery test

---

### P1-5. Config Inheritance and Deduplication ✅

**Status**: Complete
**Files**: `config/agent.toml`, `scripts/agent/base.py`

- [x] `[agent_base]` section with shared defaults
- [x] Config merging: agent sections inherit from `agent_base`, override only what differs
- [x] Duplicated keys removed from `[agent]`, `[agent_mf]`, `[agent_macro]`
- [x] Config validation (unknown keys → warning, missing required → error)

---

### P1-6. Structured Logging ✅

**Status**: Complete
**Files**: `scripts/logging_config.py`

- [x] `setup_logging()` with JSON formatter and centralized config
- [x] Correlation IDs (cycle_id, hypothesis_id) in agent log context
- [x] Critical daemon/runner/pipeline paths converted to structured logging
- [x] Log to `data/logs/` with rotation

Note: Many `print()` calls remain in non-critical paths. Full migration is ongoing but low priority.

---

### P1-7. Dashboard Caching ✅

**Status**: Complete
**Files**: `scripts/agent_dashboard.py`

- [x] In-memory cache with TTL for state reads and API endpoints
- [x] `Last-Modified` / `If-Modified-Since` headers
- [x] Request logging
- [x] Specific exception types replacing bare `except` handlers

---

### P1-8. Integration Tests for Daemon Cycles ✅

**Status**: Complete
**Files**: `scripts/tests/test_daemon_integration.py`, `scripts/tests/test_multi_agent.py`

- [x] Synthetic parquet fixture (deterministic, 3 dates × 2 symbols)
- [x] Full cycle test: manifest → generation → 5-gate execution → registration
- [x] State persistence + crash recovery test
- [x] FDR control test with known p-values
- [x] Multi-agent coordination test
- [x] Config validation test

---

## Phase 2: Research API ✅

All items complete. Structured hypothesis data exposed through Axum API server.

### P2-1. Structured Hypothesis Output ✅

**Status**: Complete
**Files**: `scripts/agent/research_output.py`, `scripts/agent/runner.py`, `scripts/agent/base.py`

- [x] Hypothesis JSON schema with all fields (id, agent, generator, claim, math, gates, status, timestamps)
- [x] Runners emit structured JSON per hypothesis to `data/research/hypotheses/`
- [x] Math derivation field (LaTeX) per generator type
- [x] Full gate details: metric value, threshold, p-value, result
- [x] Cycle summary JSON to `data/research/cycles/`
- [x] Schema validation test

---

### P2-2. Research REST Endpoints ✅

**Status**: Complete
**Files**: `rust/api/src/` (research module)

- [x] `GET /api/research/hypotheses` — paginated, filterable
- [x] `GET /api/research/hypotheses/:id` — full detail
- [x] `GET /api/research/signals` — registered signals with IC history
- [x] `GET /api/research/cycles` — cycle summaries
- [x] `GET /api/research/stats` — aggregate metrics
- [x] `GET /api/research/heatmap` — feature × horizon IC matrix

---

### P2-3. WebSocket Research Stream ✅

**Status**: Complete
**Files**: `rust/api/src/` (WebSocket handler)

- [x] `WS /ws/research` endpoint
- [x] Events: `hypothesis_started`, `gate_passed`, `gate_failed`, `hypothesis_registered`, `cycle_completed`
- [x] Redis pub/sub infrastructure (Python → Rust)
- [x] Python helper: `publish_research_event()`

---

## Phase 3: Research Website (Frontend) ✅

All items complete. Full Next.js app in `web/` with 94 passing tests.

### P3-1. Project Scaffolding ✅

- [x] Next.js + TypeScript in `web/`
- [x] API proxy to Axum backend
- [x] Plotly.js, D3.js, KaTeX, Tailwind CSS
- [x] Layout with sidebar navigation, header with status indicator
- [x] Docker service for frontend
- [x] Makefile targets

### P3-2. Dashboard Page ✅

- [x] Agent status cards, live cycle progress, recent hypothesis feed, aggregate stats, system health

### P3-3. Hypothesis Explorer Page ✅

- [x] Sortable/filterable table, gate funnel, Sankey diagram, drill-down

### P3-4. Signal Detail Page ✅

- [x] Gate waterfall, KaTeX math panel, IC time series, walk-forward equity curve, per-symbol breakdown

### P3-5. IC Heatmap Page ✅

- [x] Interactive Plotly heatmap, diverging color scale, cell click → filter, toggle IC types

### P3-6. Signal Registry Page ✅

- [x] Registry table, IC decay curves, correlation matrix, portfolio weights treemap

### P3-7. Math Lab Page ✅

- [x] KaTeX equations, feature definitions, gate protocol math, position sizing, paper references

### P3-8. Graveyard Page ✅

- [x] Failure mode distribution, near misses table, failure trends, recyclable hypotheses

### P3-9. Feature Interaction Network ✅

- [x] D3 force-directed graph, MI edges, feature clustering, click-to-filter

---

## Phase 4: Polish and Production

### P4-1. Real-Time WebSocket Updates ✅

**Status**: Complete

- [x] Dashboard connected to `WS /ws/research` for live hypothesis feed
- [x] Animated transitions for new hypotheses
- [x] Agent status card flash on gate pass/fail
- [x] Real-time aggregate stats
- [x] Connection status indicator with reconnection logic

---

### P4-2. PDF Export per Hypothesis

**Status**: NOT STARTED
**Priority**: Low
**Effort**: Medium (2 days)

**Tasks**:
- [ ] "Export PDF" button on Signal Detail page
- [ ] Generate LaTeX from hypothesis JSON (template with KaTeX → LaTeX conversion)
- [ ] Compile via pdflatex (server-side)
- [ ] Include: claim, math derivation, gate results table, IC chart (embedded PNG), equity curve
- [ ] Return PDF as download

---

### P4-3. CI/CD for Frontend ✅

**Status**: Complete

- [x] `web/` in CI pipeline (lint, type-check, build)
- [x] Dockerfile for production frontend build
- [x] Frontend service in `docker-compose.yml`

---

### P4-4. Alert Integration ✅

**Status**: Complete

- [x] Telegram notification on hypothesis promotion to REGISTERED
- [x] Uses existing `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` infrastructure

---

## Additional Work (Not in Original Backlog)

### Sharpe Ratio Standardization ✅ (2026-05-29)

Eliminated trade-frequency annualization bias across all Python Sharpe callers. All paths now aggregate intraday PnL to daily before annualizing with `sqrt(252)`. Documented in `docs/backlog/sharpe_ratio_standardization.md`.

### Parameter Stability Gate + Health Endpoints ✅ (2026-05-29)

Added parameter stability analysis to backtest engine and health check endpoints. Merged from `feat/param-stability-health`.

### Portfolio-Level Risk Constraints (planned, not yet implemented)

`PortfolioConstraints` dataclass and `check_portfolio_constraints()` in deployer. Aggregate leverage cap, concentration (Herfindahl/effective-N), portfolio drawdown circuit breaker. Full plan in `.claude/plans/floofy-nibbling-origami.md`.

---

## Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 | P1-1 through P1-8 | ✅ All complete |
| Phase 2 | P2-1 through P2-3 | ✅ All complete |
| Phase 3 | P3-1 through P3-9 | ✅ All complete |
| Phase 4 | P4-1 through P4-4 | 3/4 complete — P4-2 (PDF export) remaining |

**Remaining work**: P4-2 (PDF export per hypothesis) is the only unimplemented item from the original backlog.

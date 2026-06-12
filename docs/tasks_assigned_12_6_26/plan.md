# NAT Autonomous Trading Network — Implementation Plan

**Date:** 2026-06-12
**Status:** PLAN (not yet implemented)
**Estimated effort:** ~16 days (65% reuse, 35% new)

---

## 1. Objective

Build a continuously-running, dockerized system that:
1. Ingests data across multiple timeframes (tick, 5min, 1h, daily)
2. Discovers promising signals via autonomous research agents
3. Validates them through OOS walk-forward testing
4. Paper trades winners for 7+ days
5. Surfaces approval-ready signals to a human (single gate)
6. Deploys approved signals to live trading with capital scaling
7. Monitors live signals for decay and auto-retires degraded ones
8. Runs entirely in Docker, horizontally scalable in the cloud

---

## 2. What Already Exists (~65% built)

### 2.1 Research Agents

| Component | File | Status |
|-----------|------|--------|
| Base agent ABC | `scripts/agent/base.py` (1279 lines) | Done |
| Base runner ABC | `scripts/agent/base.py:45-386` | Done |
| Microstructure agent | `scripts/agent/daemon.py` (76 lines) | Done |
| Medium-freq agent | `scripts/agent/mf_daemon.py` (52 lines) | Done |
| Macro agent | `scripts/agent/macro_daemon.py` (52 lines) | Done |
| Meta-agent orchestrator | `scripts/agent/meta_daemon.py` | Done |
| 8 tick generators | `scripts/agent/generators/*.py` | Done |
| 3 MF generators | `scripts/agent/generators/medium_freq/` | Done |
| 3 macro generators | `scripts/agent/generators/macro/` | Done |
| Hypothesis queue | `scripts/agent/hypothesis_queue.py` (SQLite) | Done |
| 5-gate protocol | `scripts/agent/gates.py` | Done |
| Research output | `scripts/agent/research_output.py` | Done |

**Agent subclass pattern** (key design to follow for daily agent):
- Thin ~50-80 line subclass of `ResearchAgent`
- Overrides: `agent_type`, `config_section`, `default_generators`, `_rolling_ic_bar_period`, `_rolling_ic_horizon_default`
- Factory: `create_runner()` returns timeframe-specific `BaseRunner` subclass
- CLI: calls `cli_main(AgentClass, description)` from `base.py:1224`

### 2.2 State Store

| Table | Purpose |
|-------|---------|
| `agent_state` | Key-value per agent |
| `state_history` | Phase transitions |
| `hypotheses` | All hypotheses (queued/passed/failed) |
| `registry` | Registered signals with IC, decay, paper metrics |
| `generator_stats` | Per-generator hit rate |
| `research_output` | Structured records |
| `budget` | Meta-agent → agent compute allocation |
| `directives` | Meta-agent → agent commands |
| `llm_calls` | LLM invocation audit |
| `arxiv_papers` | Paper cache |

**File:** `scripts/data/state.py` (850+ lines)
**DB:** `data/nat.db` (SQLite, WAL mode)
**Migration pattern:** `_run_migrations()` at line 182 — list of `(name, sql)` tuples, applied once and tracked in `_migrations` table.

### 2.3 Execution

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Hyperliquid REST + WebSocket client | `scripts/execution/hyperliquid_client.py` | 473 | Done |
| Signal-to-order bridge | `scripts/execution/signal_bridge.py` | 571 | One-shot mode only |
| Paper trader (walk-forward) | `scripts/alpha/paper_trader_generic.py` | 617 | Done |
| Alpha pipeline (9-step) | `scripts/alpha/alpha_pipeline.py` | — | Done |
| OOS validator | `scripts/alpha/oos_validator.py` | — | Done |
| Deployer (capital scaling) | `scripts/alpha/deployer.py` | — | Done |

**Signal bridge current mode:** Each cycle (300s default): load bars → train z-score → compute signal → check kill switches → reconcile position → place/cancel orders → log to JSONL. Operates in `dry-run`, `paper`, or `live` mode.

**HyperliquidClient methods available for candle fetching:**
- `get_meta()`, `get_midprices()`, `get_positions()`, `get_fills()`, `get_open_orders()`
- `place_order()`, `place_maker_order()`, `cancel_order()`, `cancel_all()`
- Note: No candle endpoint — need Hyperliquid info API (`/info` with `candleSnapshot` action)

### 2.4 Docker Stack

| Service | Image | Port |
|---------|-------|------|
| redis | redis:7-alpine | 6379 |
| ingestor | Dockerfile.ingestor (Rust) | 8080 |
| api | Dockerfile.api (Rust) | 3010 |
| alerts | Dockerfile.alerts (Rust) | — |
| web | web/Dockerfile (Next.js) | 3001 |
| prometheus | prom/prometheus:v2.53.0 | 9090 |
| grafana | grafana/grafana:11.1.0 | 3002 |
| postgres | postgres:16-alpine | 5432 |
| optuna-dashboard | optuna/optuna-dashboard | 8070 |
| caddy | caddy:2-alpine | 80/443 |

**File:** `docker-compose.yml` (215 lines)
**Dockerfiles:** `docker/Dockerfile.ingestor`, `docker/Dockerfile.api`, `docker/Dockerfile.alerts`
**Network:** `nat-network`

### 2.5 Config

**File:** `config/agent.toml`

| Section | Cycle | Timeframe | Key thresholds |
|---------|-------|-----------|----------------|
| `[agent]` | 3600s (1h) | tick | min_ic=0.10, paper_sharpe_min=1.5 |
| `[agent_mf]` | 7200s (2h) | 5min | Same gate structure |
| `[agent_macro]` | 14400s (4h) | 1h | Same gate structure |
| `[meta_agent]` | 21600s (6h) | — | correlation_threshold=0.70 |

**Promotion thresholds** (`[agent.promotion]`):
- `paper_sharpe_min = 1.5`
- `paper_days = 7`
- `realized_ic_ratio_min = 0.8`
- `max_drawdown_pct = 2.0`

---

## 3. What Needs to Be Built (~35%)

### 3.1 Signal Lifecycle State Machine

**Problem:** Signals currently exist only in the `registry` table with a simple `status` field (`validated`/`paper`/`live`/`retired`). There's no automated pipeline connecting discovery → OOS → paper → approval → live. The `_check_promotions()` method (base.py:919) only logs promotion eligibility — it doesn't act.

**Solution:** New `signal_lifecycle` table + `lifecycle_history` audit trail in existing `nat.db`. High-level API in `scripts/signal_lifecycle.py`.

```
DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING → LIVE → MONITORING → RETIRED
     ↓            ↓            ↓                                       ↓
  REJECTED    REJECTED     REJECTED                                 RETIRED
```

- **DISCOVERED:** Agent's `register_signal()` inserts here after passing all gates
- **VALIDATED:** Promotion daemon triggers OOS walk-forward (4-fold, deflated Sharpe > 0)
- **PAPER_TRADING:** Promotion daemon starts paper_trader_generic.py subprocess
- **APPROVAL_PENDING:** After 7 days paper trading, if Sharpe > 0.5, max DD < 5%
- **LIVE:** Human approves via `nat lifecycle approve <id>` — sole human gate
- **MONITORING:** Signal bridge picks up, continuous IC tracking
- **RETIRED:** Auto-retire on IC decay (existing logic in base.py:853-914)

**Files to create/modify:**

| File | Action | Details |
|------|--------|---------|
| `scripts/data/state.py` | Modify | Add migration: `signal_lifecycle` + `lifecycle_history` tables. Add 8 methods: `lifecycle_insert()`, `lifecycle_transition()`, `lifecycle_update_metrics()`, `lifecycle_get()`, `lifecycle_query()`, `lifecycle_summary()`, `lifecycle_history()`. ~120 new lines. |
| `scripts/signal_lifecycle.py` | Create (~160 lines) | High-level API: `SignalLifecycle` class with `discover()`, `validate()`, `start_paper()`, `request_approval()`, `approve()`, `reject()`, `retire()`, `status()`, `summary()`. Best-effort Redis pub/sub for dashboard. |

**Schema:**
```sql
CREATE TABLE signal_lifecycle (
    signal_id        TEXT PRIMARY KEY,
    algorithm        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    horizon          TEXT NOT NULL DEFAULT '',
    agent            TEXT NOT NULL DEFAULT '',
    hypothesis_id    TEXT,
    state            TEXT NOT NULL DEFAULT 'DISCOVERED',
    state_entered_at TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    metrics          TEXT NOT NULL DEFAULT '{}',   -- JSON blob
    promoted_by      TEXT NOT NULL DEFAULT '',
    notes            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE lifecycle_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id  TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state   TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT '',
    at         TEXT NOT NULL
);
```

**Transition rules** (enforced in `lifecycle_transition()`):
| From | Allowed targets |
|------|-----------------|
| DISCOVERED | VALIDATED, REJECTED |
| VALIDATED | PAPER_TRADING, REJECTED |
| PAPER_TRADING | APPROVAL_PENDING, REJECTED |
| APPROVAL_PENDING | LIVE, REJECTED |
| LIVE | MONITORING |
| MONITORING | RETIRED |

### 3.2 Agent Integration

**Problem:** `register_signal()` in `base.py:307` writes to the `registry` table but doesn't insert into the lifecycle.

**Solution:** After `_store.append_signal()` at line 324, add lifecycle insertion. Minimal change — 10 lines.

**File:** `scripts/agent/base.py`
**Location:** `register_signal()` method, after line 324
**Change:**
```python
# After existing append_signal:
try:
    from signal_lifecycle import SignalLifecycle
    lc = SignalLifecycle(db_path=self._store.db_path)
    lc.discover(
        signal_id=self.h.id,
        algorithm=signal.name[:80],
        symbol=",".join(signal.symbols),
        horizon=f"{horizon_s}s",
        agent=self._agent or "",
        hypothesis_id=self.h.id,
        metrics={"expected_ic": signal.expected_ic},
    )
except Exception:
    log.debug("lifecycle insert failed (non-critical)", exc_info=True)
```

### 3.3 Daily Agent (Lower-Frequency)

**Problem:** Current agents cover tick (micro), 5min (MF), 1h (macro). No agent for 1h-24h daily signals that use candle data, funding rates, or cross-day patterns.

**Solution:** New `DailyAgent` following the exact pattern of `macro_daemon.py` (52 lines).

**Files to create:**

| File | Lines | Template | Details |
|------|-------|----------|---------|
| `scripts/agent/daily_daemon.py` | ~80 | `macro_daemon.py` | `DailyAgent(ResearchAgent)`: `agent_type="daily"`, `config_section="agent_daily"`, `_rolling_ic_bar_period="1h"`, cycle 14400s (4h), horizon 14400s default |
| `scripts/agent/daily_runner.py` | ~40 | `macro_runner.py` | `DailyRunner(BaseRunner)`: `TIMEFRAME="1h"`, 4-gate protocol, `DEFAULT_HORIZON_S=14400` |
| `scripts/agent/generators/daily/momentum.py` | ~200 | `macro/funding_meanrev.py` | Momentum signals: price trend, volume breakout on 1h-4h bars |
| `scripts/agent/generators/daily/mean_reversion.py` | ~200 | `macro/funding_meanrev.py` | Daily mean-reversion: funding rate cycles, basis convergence |
| `scripts/agent/generators/daily/cross_asset.py` | ~200 | `generators/cross_asset.py` | Cross-symbol daily correlation, lead-lag on daily horizon |
| `scripts/agent/generators/daily/__init__.py` | ~5 | — | Module init |

**Config addition** (`config/agent.toml`):
```toml
[agent_daily]
cycle_interval_s = 14400
max_experiments_per_cycle = 5
timeframe = "1h"
generators_enabled = ["daily_momentum", "daily_mean_reversion", "daily_cross_asset"]

[agent_daily.gates]
min_ic = 0.08
min_dIC = 0.03

[agent_daily.promotion]
paper_sharpe_min = 1.0
paper_days = 14
```

**Meta-agent update** (`scripts/agent/meta_daemon.py`):
- Add `"daily"` to the list of managed agents in Thompson sampling budget allocation
- ~5 lines changed

### 3.4 Candle Fetcher Daemon

**Problem:** Daily/macro agents need candle data (1m, 5m, 15m, 1h) that the tick ingestor doesn't produce. Currently `data/candles/` has 6 files covering only BTC/ETH/SOL at 1m and 15m.

**Solution:** Lightweight daemon fetching candles from Hyperliquid REST API.

**File:** `scripts/candle_daemon.py` (~80 lines)

**Design:**
- Runs every 60s
- Fetches 1m, 5m, 15m, 1h candles for BTC, ETH, SOL
- Writes to `data/candles/{symbol}_{interval}.parquet` (append mode)
- Uses Hyperliquid info API: `POST /info` with `{"type": "candleSnapshot", "coin": "BTC", "interval": "1h", "startTime": ...}`
- Health endpoint on port 8065

**Note:** `HyperliquidClient` doesn't have candle methods yet. Add `get_candles(symbol, interval, start_time)` method (~20 lines) to `scripts/execution/hyperliquid_client.py`.

### 3.5 Promotion Daemon

**Problem:** No automated system moves signals through DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING. The existing `_check_promotions()` in `base.py:919` only logs.

**Solution:** New daemon that polls `signal_lifecycle` table and triggers transitions.

**File:** `scripts/promotion_daemon.py` (~350 lines)

**Design:**
```
Loop every 300s:
  1. Query DISCOVERED signals
     → For each: spawn `scripts/alpha/oos_validator.py` as subprocess
     → On pass: transition to VALIDATED + store OOS metrics

  2. Query VALIDATED signals
     → For each: spawn `scripts/alpha/paper_trader_generic.py` as subprocess
     → Transition to PAPER_TRADING

  3. Query PAPER_TRADING signals where state_entered_at > 7 days ago
     → Read paper trade results from data/paper_trades/
     → If Sharpe > 0.5 and DD < 5%: transition to APPROVAL_PENDING
     → If Sharpe < 0 after 7 days: transition to REJECTED

  4. Query LIVE signals
     → Check rolling IC from registry
     → If IC decayed beyond threshold: transition to MONITORING → RETIRED

  5. Publish summary to Redis for dashboard
```

**Subprocess pattern:** Follows existing agent convention — child processes called via `subprocess.run()` (not imported) to prevent OOM. See `scripts/agent/runner.py` for pattern.

**Config section** (`config/agent.toml`):
```toml
[promotion]
poll_interval_s = 300
oos_sharpe_min = 0.0
paper_sharpe_min = 0.5
paper_days = 7
paper_max_dd_pct = 5.0
max_concurrent_paper = 5
```

### 3.6 Signal Bridge Daemon Mode

**Problem:** `signal_bridge.py` runs one-shot cycles. For autonomous operation, it needs a persistent daemon mode that reads LIVE signals from the lifecycle and manages positions.

**Solution:** Add daemon mode to existing `signal_bridge.py`.

**File:** `scripts/execution/signal_bridge.py`
**Changes (~80 new lines):**

1. New entry point: `run_daemon(poll_interval=300)` — continuous loop
2. Read LIVE signals from `signal_lifecycle` table instead of hardcoded algorithm
3. Fill tracking: append to `data/execution/fills_{date}.jsonl`
4. Daily P&L rollup: compute from fills, store in `data/execution/daily_pnl.json`
5. Prometheus metrics: signal count, position size, P&L
6. Kill switch integration: existing `evaluate_kill_switches()` applies per-signal

### 3.7 Dockerize Python Services

**Problem:** Only Rust services are dockerized. Python agents, promotion daemon, and signal bridge run as bare processes.

**Solution:** Shared Python Docker image + docker-compose services.

**Files to create:**

| File | Lines | Details |
|------|-------|---------|
| `docker/Dockerfile.agent` | ~30 | Python 3.11-slim, pip install requirements, WORKDIR /app |

**docker-compose.yml additions** (~60 new lines, 6 services):

```yaml
agent-micro:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/agent/daemon.py start
  volumes: [./data:/app/data, ./config:/app/config, ./scripts:/app/scripts]
  depends_on: [redis, ingestor]
  restart: unless-stopped

agent-mf:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/agent/mf_daemon.py start
  # same pattern...

agent-macro:
  # same pattern...

agent-daily:
  # same pattern...

meta-agent:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/agent/meta_daemon.py start
  # same pattern...

promotion-daemon:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/promotion_daemon.py
  # same pattern...

candle-fetcher:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/candle_daemon.py
  # same pattern...

signal-bridge:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/execution/signal_bridge.py --mode daemon
  environment: [HL_PRIVATE_KEY]
  # same pattern...
```

### 3.8 CLI Commands

**File:** `nat` script
**New commands:**

| Command | Handler | Description |
|---------|---------|-------------|
| `nat lifecycle status` | Print `SignalLifecycle.status()` | Show signal counts per state |
| `nat lifecycle list [--state X]` | Query + tabulate | List signals with filters |
| `nat lifecycle approve <id>` | `SignalLifecycle.approve()` | Human approval gate |
| `nat lifecycle reject <id>` | `SignalLifecycle.reject()` | Reject signal |
| `nat lifecycle history <id>` | Print transition log | Audit trail |
| `nat daily-agent start/once/status` | `cli_main(DailyAgent)` | Daily agent CLI |

### 3.9 Monitoring & Dashboards

**Grafana dashboards** (3 new JSON files in `grafana/dashboards/`):

1. **Signal Lifecycle Funnel** — Counts per state, transition rates, time-in-state
2. **Paper Trading Performance** — Per-signal Sharpe, drawdown, trade count
3. **Live Trading P&L** — Fills, positions, daily returns, cumulative P&L

**Prometheus metrics** (added to Python services):
- `nat_lifecycle_signals_total{state}` — Gauge per state
- `nat_lifecycle_transitions_total{from,to}` — Counter
- `nat_agent_cycles_total{agent}` — Counter
- `nat_paper_sharpe{signal_id}` — Gauge
- `nat_live_pnl_bps{symbol}` — Gauge

**prometheus.yml update:** Add scrape targets for Python service health endpoints.

---

## 4. Implementation Phases

### Phase 1: Signal Lifecycle Core (2 days)

**Goal:** Lifecycle table + API + CLI, no behavior change to existing agents.

| Step | File | Action |
|------|------|--------|
| 1.1 | `scripts/data/state.py` | Add 4 migrations (lifecycle table, indexes, history table, history index). Add 8 methods to `StateStore`. |
| 1.2 | `scripts/signal_lifecycle.py` | Create high-level API (~160 lines) |
| 1.3 | `nat` | Add `lifecycle` command group (status/list/approve/reject/history) |
| 1.4 | Test | `python3 -c "from scripts.signal_lifecycle import SignalLifecycle; ..."` end-to-end |

**Verification:**
```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from signal_lifecycle import SignalLifecycle
lc = SignalLifecycle()
sid = lc.discover(None, 'test_algo', 'BTC', agent='micro')
assert lc.validate(sid)
assert lc.start_paper(sid)
print(lc.status())
# cleanup
lc._store._conn.execute('DELETE FROM signal_lifecycle WHERE signal_id=?', (sid,))
lc._store._conn.execute('DELETE FROM lifecycle_history WHERE signal_id=?', (sid,))
lc._store._conn.commit()
"
nat lifecycle status
```

### Phase 2: Agent Integration (1 day)

**Goal:** Existing agents automatically feed discovered signals into the lifecycle.

| Step | File | Action |
|------|------|--------|
| 2.1 | `scripts/agent/base.py` | Add lifecycle insertion in `register_signal()` after line 324 (~10 lines) |
| 2.2 | Test | Run `nat agent once`, verify signal appears in `nat lifecycle status` |

### Phase 3: Dockerize Existing Agents (2 days)

**Goal:** All 4 agents + meta-agent run in Docker.

| Step | File | Action |
|------|------|--------|
| 3.1 | `docker/Dockerfile.agent` | Create shared Python image |
| 3.2 | `docker-compose.yml` | Add agent-micro, agent-mf, agent-macro, meta-agent services |
| 3.3 | Test | `docker compose up agent-micro`, verify research cycle runs |
| 3.4 | Test | `docker compose up meta-agent`, verify budget allocation |

### Phase 4: Daily Agent + Candle Fetcher (3 days)

**Goal:** New lower-frequency agent discovering 1h-24h signals.

| Step | File | Action |
|------|------|--------|
| 4.1 | `scripts/execution/hyperliquid_client.py` | Add `get_candles()` method (~20 lines) |
| 4.2 | `scripts/candle_daemon.py` | Create candle fetcher (~80 lines) |
| 4.3 | `scripts/agent/daily_daemon.py` | Create DailyAgent thin subclass (~80 lines) |
| 4.4 | `scripts/agent/daily_runner.py` | Create DailyRunner (~40 lines) |
| 4.5 | `scripts/agent/generators/daily/` | Create 3 generators (~200 lines each) |
| 4.6 | `config/agent.toml` | Add `[agent_daily]` section |
| 4.7 | `scripts/agent/meta_daemon.py` | Register daily agent in budget allocation |
| 4.8 | `docker-compose.yml` | Add agent-daily, candle-fetcher services |
| 4.9 | `nat` | Add `daily-agent` CLI commands |
| 4.10 | Test | `nat daily-agent once` completes a research cycle |

### Phase 5: Promotion Daemon (3 days)

**Goal:** Automated signal progression through lifecycle states.

| Step | File | Action |
|------|------|--------|
| 5.1 | `scripts/promotion_daemon.py` | Create promotion daemon (~350 lines) |
| 5.2 | `config/agent.toml` | Add `[promotion]` section |
| 5.3 | `docker-compose.yml` | Add promotion-daemon service |
| 5.4 | `nat` | Add `promotion status` CLI command |
| 5.5 | Test | Insert DISCOVERED signal, verify it progresses to APPROVAL_PENDING |

### Phase 6: Execution Upgrade (2 days)

**Goal:** Signal bridge reads LIVE signals and manages positions autonomously.

| Step | File | Action |
|------|------|--------|
| 6.1 | `scripts/execution/signal_bridge.py` | Add `run_daemon()` mode (~80 lines) |
| 6.2 | `scripts/execution/signal_bridge.py` | Add fill tracking + daily P&L |
| 6.3 | `docker-compose.yml` | Add signal-bridge service |
| 6.4 | Test | Approve signal via `nat lifecycle approve`, verify bridge picks it up |

### Phase 7: Monitoring & Integration (3 days)

**Goal:** Full observability + end-to-end test.

| Step | File | Action |
|------|------|--------|
| 7.1 | Python services | Add Prometheus metrics endpoints |
| 7.2 | `grafana/dashboards/` | Create 3 dashboard JSONs |
| 7.3 | `docker/prometheus/prometheus.yml` | Add scrape targets |
| 7.4 | `scripts/agent_dashboard.py` | Add lifecycle view tab |
| 7.5 | Integration | Full `docker compose up` with all services |
| 7.6 | E2E test | Ingestor → agent → OOS → paper → approve → live |
| 7.7 | Docs | Cloud deployment guide (env vars, secrets, scaling) |

---

## 5. File Inventory

### New Files (12)

| File | Lines | Phase |
|------|-------|-------|
| `scripts/signal_lifecycle.py` | ~160 | 1 |
| `scripts/candle_daemon.py` | ~80 | 4 |
| `scripts/promotion_daemon.py` | ~350 | 5 |
| `scripts/agent/daily_daemon.py` | ~80 | 4 |
| `scripts/agent/daily_runner.py` | ~40 | 4 |
| `scripts/agent/generators/daily/__init__.py` | ~5 | 4 |
| `scripts/agent/generators/daily/momentum.py` | ~200 | 4 |
| `scripts/agent/generators/daily/mean_reversion.py` | ~200 | 4 |
| `scripts/agent/generators/daily/cross_asset.py` | ~200 | 4 |
| `docker/Dockerfile.agent` | ~30 | 3 |
| `grafana/dashboards/signal_lifecycle.json` | ~200 | 7 |
| `grafana/dashboards/paper_trading.json` | ~150 | 7 |

### Modified Files (8)

| File | Change | Phase |
|------|--------|-------|
| `scripts/data/state.py` | +120 lines (lifecycle schema + methods) | 1 |
| `scripts/agent/base.py` | +10 lines (lifecycle insert in register_signal) | 2 |
| `scripts/execution/signal_bridge.py` | +80 lines (daemon mode, fill tracking) | 6 |
| `scripts/execution/hyperliquid_client.py` | +20 lines (get_candles method) | 4 |
| `scripts/agent/meta_daemon.py` | +5 lines (daily agent registration) | 4 |
| `config/agent.toml` | +20 lines (agent_daily + promotion sections) | 4, 5 |
| `docker-compose.yml` | +60 lines (8 new services) | 3, 4, 5, 6 |
| `nat` | +80 lines (lifecycle + daily-agent CLI) | 1, 4 |

---

## 6. Key Design Decisions

### 6.1 Single Human Gate
Only APPROVAL_PENDING → LIVE requires human action. Everything else is automated. This minimizes friction while preventing catastrophic deployment of a bad signal.

### 6.2 SQLite for Lifecycle (not Postgres)
The lifecycle table lives in the existing `nat.db` alongside the registry. Rationale:
- Single-writer (promotion daemon)
- WAL mode already supports concurrent dashboard reads
- No new infrastructure dependency
- Consistent with existing agent state architecture

### 6.3 Subprocess Execution for OOS/Paper
The promotion daemon calls `oos_validator.py` and `paper_trader_generic.py` as subprocesses, not imports. This follows the existing agent pattern (see `runner.py`) and prevents memory leaks from long-running evaluation.

### 6.4 Shared Docker Image
One `Dockerfile.agent` for all Python services. Different entry points via `command:`. This means one build, one image push, and consistent dependencies.

### 6.5 Lifecycle Separate from Registry
The `signal_lifecycle` table is separate from the `registry` table. The registry tracks signal metadata (IC, features, decay). The lifecycle tracks state progression. They are linked by `hypothesis_id`. This avoids schema changes to the registry table which is heavily used by existing code.

### 6.6 Daily Agent as Thin Subclass
The daily agent follows the exact same pattern as macro_daemon.py (52 lines). No new framework, no new abstractions. Just another timeframe with its own generators.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| K2: 82 dead features (NaN) | Daily agent generators may reference unavailable features | Generators validate feature availability from manifest |
| Zombie process (K5) | Agent containers crash silently | Docker `restart: unless-stopped` + health checks |
| Paper trading takes 7+ days | Slow signal progression | Run paper on historical data for backtested fast-forward |
| OOS validator subprocess hangs | Promotion daemon blocks | Timeout (600s) on subprocess.run(), log and skip |
| Signal bridge places bad orders | Capital loss | Kill switches (existing), dry-run first, 1% initial allocation |
| Candle API rate limits | Candle fetcher fails | Backoff with jitter, cache aggressively |

---

## 8. Dependencies & Ordering

```
Phase 1 (Lifecycle)
  ↓
Phase 2 (Agent Integration) ←── depends on Phase 1
  ↓
Phase 3 (Dockerize) ←── independent, can parallel with Phase 2
  ↓
Phase 4 (Daily Agent) ←── depends on Phase 3 for docker
  ↓
Phase 5 (Promotion) ←── depends on Phase 1 (lifecycle) + Phase 4 (daily agent optional)
  ↓
Phase 6 (Execution) ←── depends on Phase 1 (lifecycle) + Phase 5 (promotion feeds LIVE)
  ↓
Phase 7 (Monitoring) ←── depends on all above
```

**Parallelizable:** Phases 2 and 3 can run concurrently. Phase 4 generators can be developed in parallel with Phase 3 dockerization.

---

## 9. Acceptance Criteria

- [ ] `nat lifecycle status` shows signal counts per state
- [ ] `nat lifecycle approve <id>` transitions a signal to LIVE
- [ ] Running `nat agent once` creates a DISCOVERED signal in lifecycle
- [ ] `docker compose up agent-micro agent-mf agent-macro meta-agent` runs all agents
- [ ] `nat daily-agent once` completes a research cycle with daily generators
- [ ] Promotion daemon auto-transitions DISCOVERED → VALIDATED → PAPER_TRADING
- [ ] After 7 days paper trading, signal reaches APPROVAL_PENDING
- [ ] Signal bridge daemon mode reads LIVE signals and places orders
- [ ] Grafana signal lifecycle dashboard shows funnel
- [ ] Full `docker compose up` starts 15+ services, all healthy
- [ ] End-to-end: ingestion → discovery → OOS → paper → approval → live trade

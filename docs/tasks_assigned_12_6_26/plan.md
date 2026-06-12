# NAT Autonomous Trading Network — Implementation Plan

**Date:** 2026-06-12 (rev. 2 — integrated with Q/P roadmaps and the NAT CLI improvement plan)
**Status:** PLAN (not yet implemented)
**Estimated effort:** ~18 days (65% reuse, 35% new)
**Companion docs:** `phd_vs_quant_roadmap.md`, `Q/*.md`, `P/*.md`, `nat_cli_tasks/*.md`, `data_inventory.md`, `situation_analysis.md`

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

### 1.1 Position in the Roadmap

This network is the **execution backbone of the Quant path** and an **evidence generator for the PhD path**:

- It turns Q2 (validation), Q3 (paper trading), and Q4 (live deployment) from one-off analyst sessions into a standing pipeline. Q2.3's alpha screen feeds DISCOVERED; Q2.5's walk-forward is the VALIDATED gate; Q3.4's 14-day paper run is the APPROVAL_PENDING gate; Q4.1's 1% capital deployment is what `nat lifecycle approve` triggers.
- The sole human gate **is** the roadmap's G8 + approval step — not a new gate invented here.
- Every lifecycle record carries provenance (git SHA + data fingerprint, P1.5/Q1.4), which makes the funnel a PhD-grade audit trail: the P4.1 research statement can cite a fully reproducible discovery → live pipeline, and fill-conditional IC from MONITORING (Q3.5) is the empirical material for the preprint's adverse-selection narrative (D3 in the roadmap).
- The network does not accelerate the roadmap calendar — paper trading still takes 14 real days and data still accumulates at 1 day/day. What changes is that nothing waits on a human except the single approval.

### 1.2 Non-Negotiable Constraints Inherited from the Roadmap

1. **Gates are imported, not invented.** VALIDATED criteria = G4 (walk-forward + deflated Sharpe, `config/alpha.toml`). APPROVAL_PENDING criteria = G8 (14-day paper, Q3.4 spec). Kill-switch thresholds = ROADMAP Step 9. Any threshold in this plan that disagrees with those sources is a bug in this plan.
2. **Costs come from `config/costs.toml` via `load_costs()`** (Q1.4). OOS validation, paper trading, and live execution must not disagree on costs — that inconsistency is the exact failure Q1.4 exists to fix, and an automated promotion pipeline would amplify it.
3. **No ingestor disturbance during the accumulation window** (Q1.1, Jun 11–17). Docker work must never recreate the ingestor container until the 7-day streak completes — the streak blocks both paths.
4. **No live capital before G8 passes and the kill-switch daemon (Q3.6) is operational.**

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

> ⚠ **Inconsistency to resolve in Phase 1:** `paper_days = 7` here predates Gate G8, which requires **14 days** (Q3.4 spec). The lifecycle uses the new `[promotion]` section (G8-aligned, §3.5); `[agent.promotion]` remains agent-side eligibility *hinting* only and should be annotated as such, or aligned to 14.

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
- **VALIDATED:** Promotion daemon triggers OOS walk-forward — **G4 criteria** (Q2.5): 4-fold expanding window, 100-bar embargo, deflated Sharpe > 0, costs resolved via `load_costs()`. Requires ≥7 consecutive clean days of data (Q1.1 sufficiency check) — the hierarchical combiner's suspicious monotone-IC folds on 2 days of data are the cautionary tale here.
- **PAPER_TRADING:** Promotion daemon starts paper_trader_generic.py subprocess
- **APPROVAL_PENDING:** After **14 days** paper trading (G8, not 7), if **all G8 criteria** hold: paper Sharpe within 2x of backtest, no single-day loss > 2%, IC decay < 50% of training IC, zero missed computation cycles, mean daily PnL > 0 (Q3.4 spec)
- **LIVE:** Human approves via `nat lifecycle approve <id>` — sole human gate (= Q4.1 trigger: 1% capital, maker-only)
- **MONITORING:** Signal bridge picks up, continuous IC tracking + **fill-conditional IC measurement (Q3.5)** — this data feeds back into the cost model and the preprint's adverse-selection section
- **RETIRED:** Auto-retire on IC decay (existing logic in base.py:853-914) or kill-switch `kill_strategy` trigger (§3.7)

**Seeding — the funnel starts with real traffic.** The lifecycle is not only for agent-discovered signals. At Phase 1 completion:

| Signal | Entry state | Rationale |
|--------|------------|-----------|
| jump_detector, optimal_entry, funding_reversion, 3f_liquidity | VALIDATED | Already OOS-positive at 100min / 1.61 bps RT (situation_analysis §I) — paper trading is their next roadmap step (Q3.1) |
| hierarchical_combiner | DISCOVERED | Pending Q2.1 revalidation + Q2.2 ablation on the 7-day dataset — must earn VALIDATED through the daemon like everything else |
| mean_reversion_detector (LightGBM) | DISCOVERED | OOS AUC 0.55–0.58 but needs revalidation on 30+ days (open question 5) |

This makes Q3.1 (paper trading deployment) a lifecycle operation instead of a manual run, and exercises every transition with known-good signals before agent-discovered ones arrive.

**Files to create/modify:**

| File | Action | Details |
|------|--------|---------|
| `scripts/data/state.py` | Modify | Add migration: `signal_lifecycle` + `lifecycle_history` tables. Add 8 methods: `lifecycle_insert()`, `lifecycle_transition()`, `lifecycle_update_metrics()`, `lifecycle_get()`, `lifecycle_query()`, `lifecycle_summary()`, `lifecycle_history()`. ~120 new lines. |
| `scripts/signal_lifecycle.py` | Create (~160 lines) | High-level API: `SignalLifecycle` class with `discover()`, `validate()`, `start_paper()`, `request_approval()`, `approve()`, `reject()`, `retire()`, `status()`, `summary()`. Best-effort Redis pub/sub for dashboard. |

**Schema:**
```sql
CREATE TABLE signal_lifecycle (
    signal_id        TEXT PRIMARY KEY,
    schema_version   INTEGER NOT NULL DEFAULT 1,    -- Arch-p.1 (Q1.3) discipline
    algorithm        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    horizon          TEXT NOT NULL DEFAULT '',
    agent            TEXT NOT NULL DEFAULT '',
    hypothesis_id    TEXT,
    state            TEXT NOT NULL DEFAULT 'DISCOVERED',
    state_entered_at TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    metrics          TEXT NOT NULL DEFAULT '{}',    -- JSON blob
    git_sha          TEXT NOT NULL DEFAULT '',      -- provenance (P1.5)
    data_fingerprint TEXT NOT NULL DEFAULT '',      -- provenance (P1.5)
    promoted_by      TEXT NOT NULL DEFAULT '',
    notes            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE lifecycle_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id  TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state   TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT '',
    git_sha    TEXT NOT NULL DEFAULT '',            -- code version at transition time
    at         TEXT NOT NULL
);
```

**Provenance contract (P1.5 / Q1.4):** Every insert and transition stamps `git_sha` (via `scripts/provenance.py:get_provenance()`) and, for data-driven transitions (VALIDATED, APPROVAL_PENDING), the SHA-256 `data_fingerprint` of the exact Parquet inputs used. A reviewer's "can you reproduce Table 3?" and an operator's "which code promoted this signal?" become the same query.

**Coordination with Q1.3 (Arch-p.1):** Lifecycle tables follow the same `_run_migrations()` discipline (named `(name, sql)` tuples, tracked in `_migrations`) and carry `schema_version`. If Q1.3 lands first, the lifecycle migration rides its framework; if this plan lands first, it must not introduce a second migration mechanism. Add a contract test for the lifecycle schema to Q1.3's `rust/api/tests/contract.rs` batch if the Rust API ever reads it.

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

**Problem:** Current agents cover tick (micro), 5min bars (MF), and 1h bars with 1h–24h horizons (macro). The macro agent **already owns 1h–24h** — a "daily agent" duplicating that range would compete with it for the same hypotheses. The genuinely uncovered space is **multi-day horizons (1–7 days)**: funding-cycle seasonality across 8h settlements (a full cycle needs 7+ days of coverage per data_inventory §3), weekend microstructure shifts, daily momentum/reversal, cross-day OI patterns, and basis convergence. These need candle data plus the daily macro indicators that already exist (`data/macro/` — 365 days per symbol, currently unused by any agent).

**Solution:** New `DailyAgent` following the exact pattern of `macro_daemon.py` (52 lines), with horizons strictly above the macro agent's so the four agents partition the horizon space without overlap: micro (s–min), MF (min–1h), macro (1h–24h), daily (1–7d).

**K2 guard (dead features):** 82/236 feature columns are NaN until Q1.2 lands (whale/liquidation/concentration and their dependents). Daily generators must validate feature availability against the live manifest **at generator init** — referencing only the 154 live base features plus candle/macro data — rather than discovering NaN inputs at gate time and burning a cycle.

**Files to create:**

| File | Lines | Template | Details |
|------|-------|----------|---------|
| `scripts/agent/daily_daemon.py` | ~80 | `macro_daemon.py` | `DailyAgent(ResearchAgent)`: `agent_type="daily"`, `config_section="agent_daily"`, `_rolling_ic_bar_period="1h"`, cycle 21600s (6h — slower than macro's 4h; multi-day hypotheses don't change faster), horizon 86400s (1d) default, horizon set {1d, 2d, 3d, 7d} |
| `scripts/agent/daily_runner.py` | ~40 | `macro_runner.py` | `DailyRunner(BaseRunner)`: `TIMEFRAME="1h"`, 4-gate protocol, `DEFAULT_HORIZON_S=86400` |
| `scripts/agent/generators/daily/momentum.py` | ~200 | `macro/funding_meanrev.py` | Multi-day momentum: daily trend persistence, volume breakout on 4h-1d bars |
| `scripts/agent/generators/daily/mean_reversion.py` | ~200 | `macro/funding_meanrev.py` | Funding-cycle seasonality (8h settlements), basis convergence, weekend-effect reversion |
| `scripts/agent/generators/daily/cross_asset.py` | ~200 | `generators/cross_asset.py` | Cross-symbol daily correlation, lead-lag at 1-3d horizons, uses `data/macro/` indicators |
| `scripts/agent/generators/daily/__init__.py` | ~5 | — | Module init |

**Config addition** (`config/agent.toml`):
```toml
[agent_daily]
cycle_interval_s = 21600
max_experiments_per_cycle = 5
timeframe = "1h"
default_horizon_s = 86400          # 1d; horizon set {1d, 2d, 3d, 7d} — strictly above macro's 1h-24h
generators_enabled = ["daily_momentum", "daily_mean_reversion", "daily_cross_asset"]

[agent_daily.gates]
min_ic = 0.08
min_dIC = 0.03

[agent_daily.promotion]
paper_sharpe_min = 1.0
paper_days = 14
```

**Data sufficiency caveat:** at 1–7d horizons, 22 good days (current inventory) gives very few independent observations. The daily agent ships in Phase 4 but its gates should be expected to pass rarely before ~Aug 1 (60-day threshold per data_inventory §7) — that's correct behavior, not a bug. Generators lean on `data/macro/` (365 days) where possible to compensate.

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
  0. Data sufficiency check (scripts/check_data_sufficiency.py, Q1.1)
     → If < min_clean_days_for_oos consecutive clean days: skip step 1 this cycle
       (never validate on insufficient data — the hierarchical combiner's
        2-day monotone-IC folds are why this guard exists)

  1. Query DISCOVERED signals
     → For each: spawn `scripts/alpha/oos_validator.py` as subprocess
       with costs from load_costs() (Q1.4) — G4 criteria
     → On pass: transition to VALIDATED + store OOS metrics + provenance
       (git_sha, data_fingerprint of the exact parquet set used)

  2. Query VALIDATED signals
     → For each: spawn `scripts/alpha/paper_trader_generic.py` as subprocess
       (same load_costs() source — paper and OOS must agree on costs)
     → Transition to PAPER_TRADING

  3. Query PAPER_TRADING signals where state_entered_at > paper_days ago (G8: 14 days)
     → Read paper logs + daily reconciliation from data/paper_trades/
     → Apply G8 (Q3.4 spec): paper Sharpe within 2x of backtest, no single-day
       loss > 2%, IC decay < 50% of training IC, zero missed cycles,
       mean daily PnL > 0
     → All pass: transition to APPROVAL_PENDING
     → Mean daily PnL < 0 after window: transition to REJECTED
       (failure data is kept — D3 in the roadmap: paper-vs-backtest divergence
        is publishable adverse-selection evidence, not just a dead signal)

  4. Query LIVE signals
     → Check rolling IC from registry + halt_state.json from kill-switch daemon
     → If IC decayed beyond threshold or kill_strategy triggered:
       transition to MONITORING → RETIRED

  5. Publish summary to Redis for dashboard
```

**Subprocess pattern:** Follows existing agent convention — child processes called via `subprocess.run()` (not imported) to prevent OOM. See `scripts/agent/runner.py` for pattern.

**Config section** (`config/agent.toml`):
```toml
[promotion]
poll_interval_s = 300
# Thresholds below are the canonical roadmap gates — G4/G8 — not new inventions.
# Divergence between this section, config/alpha.toml [gates], and ROADMAP is a bug.
min_clean_days_for_oos = 7          # Q1.1 sufficiency guard
oos_deflated_sharpe_min = 0.0       # G4 (deflated Sharpe, Bailey & Lopez de Prado)
paper_days = 14                     # G8 — 14 days, not 7
paper_backtest_ratio_min = 0.5      # G8: paper Sharpe within 2x of backtest
paper_max_daily_loss_pct = 2.0      # G8: no single-day loss > 2%
paper_ic_decay_max = 0.5            # G8: IC decay < 50% of training IC
max_concurrent_paper = 5
costs_exchange = "hyperliquid"      # resolved via scripts/costs.py:load_costs() (Q1.4)
```

### 3.6 Signal Bridge Daemon Mode

**Problem:** `signal_bridge.py` runs one-shot cycles. For autonomous operation, it needs a persistent daemon mode that reads LIVE signals from the lifecycle and manages positions.

**Solution:** Add daemon mode to existing `signal_bridge.py`.

**File:** `scripts/execution/signal_bridge.py`
**Changes (~80 new lines):**

1. New entry point: `run_daemon(poll_interval=300)` — continuous loop
2. Read LIVE signals from `signal_lifecycle` table instead of hardcoded algorithm
3. Fill tracking: append to `data/execution/fills_{date}.jsonl` — these fills feed **Q3.5 (fill-conditional IC)**: measured execution quality vs assumed costs, the single most valuable dataset for both cost-model calibration and the preprint's adverse-selection section
4. Daily P&L rollup: compute from fills, store in `data/execution/daily_pnl.json`
5. Prometheus metrics: signal count, position size, P&L
6. **Portfolio-level sizing (Q2.8 / Q3.3):** when multiple signals are LIVE, weights come from risk parity with correlation adjustment via the existing `scripts/agent/meta_portfolio.py` — signals are *not* sized independently. The documented <0.35 cross-correlation of the 4 deployable algorithms is the reason portfolio combination should beat any single strategy; sizing them independently throws that away
7. **Kill-switch integration:** before every cycle, check `data/risk/halt_state.json` written by the *independent* kill-switch daemon (§3.7). The bridge never trades while halted and has no flag to skip the check. In-process `evaluate_kill_switches()` remains as a second layer, not the primary control

### 3.7 Kill-Switch Daemon (Q3.6 — promoted to first-class service)

**Problem:** The original draft relied on `evaluate_kill_switches()` inside the signal bridge. Q3.6 requires risk controls that are **independent of the trading logic** — a separate process the bridge cannot bypass, impossible to accidentally disable, with Telegram alerts. An autonomous network raises the stakes: there is no human watching each cycle, so the risk layer must not share a failure domain with the thing it polices.

**Solution:** Implement Q3.6 as specified — `scripts/risk/kill_switch.py` (~250 lines) — and make it a Docker service that must be healthy before the signal bridge starts.

**Design (from Q3_1 spec):**
- Polls PnL every 60s from paper logs / live fills
- 4 thresholds (ROADMAP Step 9): daily loss > 1% → `halt_24h`; weekly DD > 2% → `halt_review` (manual `nat risk resume`); monthly DD > 5% → `kill_strategy` (requires pipeline re-run); IC < 0 for 5 consecutive days → halt
- Writes `data/risk/halt_state.json` with trigger reason and resume time; signal bridge checks it before every cycle (§3.6)
- Telegram alert within 60s of trigger; Prometheus metrics `nat_kill_switch_active{level}`, `nat_kill_switch_triggers_total{level}`
- A `kill_strategy` trigger also transitions the affected signal LIVE → MONITORING → RETIRED in the lifecycle, closing the loop
- Tested against synthetic PnL breaches **during paper trading**, before any live capital (acceptance criteria in Q3_1)

**docker-compose:** new `kill-switch` service; `signal-bridge` gets `depends_on: [kill-switch]` with `condition: service_healthy`.

**CLI:** `nat risk status`, `nat risk resume [--confirm]` (refuses to clear `kill_strategy` without a pipeline re-run).

### 3.8 Dockerize Python Services

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

kill-switch:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/risk/kill_switch.py
  environment: [TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]
  healthcheck: { test: ["CMD", "python3", "-c", "import json; json.load(open('/app/data/risk/heartbeat.json'))"] }
  # same volume pattern...

signal-bridge:
  build: { context: ., dockerfile: docker/Dockerfile.agent }
  command: python3 scripts/execution/signal_bridge.py --mode daemon
  environment: [HL_PRIVATE_KEY]
  depends_on:
    kill-switch: { condition: service_healthy }
  # same pattern...
```

> ⚠ **Accumulation-window constraint (Q1.1):** Until the Jun 17 streak completes, all `docker compose up` invocations target *only the new services by name* — never a bare `docker compose up` that could recreate the ingestor container. The 7-day streak blocks both the hierarchical revalidation (Q2.1) and the preprint's results section; it is the single most expensive thing to break.

### 3.9 CLI Commands

**File:** `nat` script
**New commands:**

| Command | Handler | Description |
|---------|---------|-------------|
| `nat lifecycle status` | Print `SignalLifecycle.status()` | Show signal counts per state |
| `nat lifecycle list [--state X]` | Query + tabulate | List signals with filters |
| `nat lifecycle approve <id>` | `SignalLifecycle.approve()` | Human approval gate |
| `nat lifecycle reject <id>` | `SignalLifecycle.reject()` | Reject signal |
| `nat lifecycle history <id>` | Print transition log | Audit trail (includes git_sha per transition) |
| `nat daily-agent start/once/status` | `cli_main(DailyAgent)` | Daily agent CLI |
| `nat risk status` | Read halt_state.json | Kill-switch state |
| `nat risk resume [--confirm]` | Clear halt (rules in §3.7) | Manual override |

**Approval workflow — the gate must present evidence, not just flip a state.** `nat lifecycle approve <id>` is the only human decision in the network, so it prints a decision package before asking for confirmation: the G8 scorecard (all 5 criteria with values), the OOS metrics from VALIDATED, the provenance record (git_sha + data_fingerprint), and pointers to `nat viz paper <signal>` (NAT6) and `nat viz portfolio` (NAT7) for visual inspection. Approving without NAT6 available is allowed but warned — the operator is approving live capital on numbers they haven't visualized.

**Conventions (from the NAT CLI improvement plan):**
- New groups (`lifecycle`, `risk`, `daily-agent`) ship with curated group-level help (NAT2 pattern) — `nat lifecycle` with no subcommand prints scoped help, not raw argparse usage. With 251 existing commands, adding 8 more without this makes discoverability worse.
- All new commands carry NAT9 maturity tags: `[BETA]` until they've survived one full paper cycle; `nat lifecycle approve` graduates to `[LIVE]` only after the first real approval.
- If NAT10 (script modularization) has landed, register under `scripts/cli/lifecycle.py` and `scripts/cli/risk.py`; otherwise write handlers in the dispatch-table style NAT10 expects so extraction is mechanical. Do not add 8 more ad-hoc handler functions to the 5,113-line monolith.

### 3.10 Monitoring & Dashboards

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

**prometheus.yml update:** Add scrape targets for Python service health endpoints (including kill-switch).

**Division of labor with NAT viz tasks:** Grafana answers "is the system healthy?" (web, passive, always-on). The NAT6/NAT7 terminal commands answer "should I approve this signal / intervene?" (terminal, interactive, evidence for decisions). The lifecycle funnel appears in both deliberately — `nat lifecycle status` for the operator at the keyboard, the Grafana funnel for glanceability. NAT6 (`nat viz paper`) and NAT7 (`nat viz portfolio`) should be scheduled **before or alongside Phase 5** — once the promotion daemon starts producing APPROVAL_PENDING signals, the operator needs them to make approval decisions.

---

## 4. Implementation Phases

### Phase 0: Shared Foundations from Q1.4 / P1.5 (1 day)

**Goal:** The two roadmap building blocks the lifecycle depends on, pulled forward. This is not scope creep — it is the first ~4h of Q1.4 and P1.5, and it prevents building the lifecycle on hardcoded costs that Q1.4 would immediately rip out.

| Step | File | Action |
|------|------|--------|
| 0.1 | `scripts/provenance.py` | `get_provenance()` + `data_fingerprint()` per the P1_2 spec (~50 lines) |
| 0.2 | `scripts/costs.py` | `load_costs(exchange)` reading `config/costs.toml` per the Q1_4 spec (~40 lines) |
| 0.3 | Test | Fingerprint determinism (same parquet set → same hash); `load_costs("hyperliquid")` matches values currently used by `nat oos30` (regression: backtest results identical before/after) |

**Bonus:** completing 0.1/0.2 advances Q1.4 and P1.5 directly — shared work, counted once (roadmap §III synergies).

### Phase 1: Signal Lifecycle Core (2 days)

**Goal:** Lifecycle table + API + CLI, no behavior change to existing agents.

| Step | File | Action |
|------|------|--------|
| 1.1 | `scripts/data/state.py` | Add 4 migrations (lifecycle table, indexes, history table, history index) **using the Arch-p.1 migration discipline** — coordinate with Q1.3 so nat.db never grows two migration frameworks. Add 8 methods to `StateStore`. |
| 1.2 | `scripts/signal_lifecycle.py` | Create high-level API (~160 lines), provenance stamped on every insert/transition |
| 1.3 | `nat` | Add `lifecycle` command group (status/list/approve/reject/history) per §3.9 conventions |
| 1.4 | Seed | Insert the 4 deployable algorithms at VALIDATED, hierarchical_combiner + mean_reversion_detector at DISCOVERED (§3.1 seeding table) |
| 1.5 | Test | `python3 -c "from scripts.signal_lifecycle import SignalLifecycle; ..."` end-to-end |

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
| 3.3 | Test | `docker compose up agent-micro` (service by name — **never bare `up` before Jun 17**, §3.8 constraint), verify research cycle runs |
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

**Goal:** Automated signal progression through lifecycle states, gated by G4/G8.

| Step | File | Action |
|------|------|--------|
| 5.1 | `scripts/promotion_daemon.py` | Create promotion daemon (~350 lines), gates per §3.5: G4 for VALIDATED, G8 (14 days) for APPROVAL_PENDING, data-sufficiency guard, `load_costs()` for both subprocesses |
| 5.2 | `config/agent.toml` | Add `[promotion]` section; annotate or align `[agent.promotion].paper_days` (currently 7, contradicts G8) |
| 5.3 | `docker-compose.yml` | Add promotion-daemon service |
| 5.4 | `nat` | Add `promotion status` CLI command |
| 5.5 | Test | Insert DISCOVERED signal, verify it progresses to APPROVAL_PENDING with provenance stamped at each transition |

**Scheduling note:** NAT6 (`nat viz paper`) should land before the first signal reaches APPROVAL_PENDING (§3.10) — otherwise the human gate operates blind.

### Phase 6: Risk Layer + Execution Upgrade (3 days)

**Goal:** Independent kill-switch daemon operational, then signal bridge reads LIVE signals and manages positions autonomously. Risk layer ships **first** — the bridge refuses to start without it.

| Step | File | Action |
|------|------|--------|
| 6.1 | `scripts/risk/kill_switch.py` | Create kill-switch daemon per Q3.6 spec (~250 lines): 4 thresholds, halt_state.json, Telegram, Prometheus |
| 6.2 | `nat` | Add `risk status` / `risk resume` commands |
| 6.3 | Test | Synthetic PnL breach triggers each of the 4 thresholds; Telegram alert within 60s; `kill_strategy` refuses resume without pipeline re-run |
| 6.4 | `scripts/execution/signal_bridge.py` | Add `run_daemon()` mode (~80 lines): halt-state check, portfolio sizing via `meta_portfolio.py` (§3.6) |
| 6.5 | `scripts/execution/signal_bridge.py` | Add fill tracking + daily P&L + fill-conditional IC logging (Q3.5) |
| 6.6 | `docker-compose.yml` | Add kill-switch + signal-bridge services (bridge `depends_on` kill-switch healthy) |
| 6.7 | Test | Approve signal via `nat lifecycle approve`, verify bridge picks it up; trigger synthetic halt, verify bridge skips the cycle |

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

### Phase Calendar vs Roadmap

The build effort (~18 working days) is paced against the data milestones from `data_inventory.md` §7 — there is no point finishing the promotion daemon before the data it validates on exists:

| When | Network work | Roadmap context |
|------|--------------|-----------------|
| Jun 13–17 | Phases 0–2 (pure Python, zero ingestor contact) | Q1.1 streak running — the riskiest window; touch nothing on su-35 |
| Jun 17–24 | Phases 3–4 | Streak complete → Q2.1 hierarchical revalidation runs; seeded signals get real 7-day OOS data |
| Jun 24 – Jul 8 | Phases 5–6 | ~Jun 20: 30 good dates → OOS30 feasible; promotion daemon has enough history to validate against |
| Jul 8–15 | Phase 7 + first full paper windows start | Q2 milestone (Jul); every paper day counts toward Q3 |
| Aug | Network runs unattended; G8 windows complete | Q3 milestone — first APPROVAL_PENDING signals; D1 decision ("is there a trading business here?") gets answered by the funnel itself |
| Sep–Oct | First `nat lifecycle approve` → LIVE at 1% | Q4.1 — matches the roadmap calendar exactly |

**PhD-path dividend:** during Jun–Jul, while the network is being built and data accumulates, P1 (preprint, ~40h writing) is the parallel track per the roadmap's critical-path verdict. The network demands no analyst time once Phase 7 completes — that *is* its contribution to the PhD path: it frees the human for writing while generating the provenance-complete funnel and fill data (Q3.5) that the preprint and research statement will cite.

---

## 5. File Inventory

### New Files (16)

| File | Lines | Phase | Roadmap task served |
|------|-------|-------|---------------------|
| `scripts/provenance.py` | ~50 | 0 | P1.5 / Q1.4 |
| `scripts/costs.py` | ~40 | 0 | Q1.4 |
| `scripts/signal_lifecycle.py` | ~160 | 1 | Q3.1 automation |
| `scripts/candle_daemon.py` | ~80 | 4 | — |
| `scripts/promotion_daemon.py` | ~350 | 5 | Q2.5/Q3.4 automation |
| `scripts/risk/__init__.py` | ~5 | 6 | Q3.6 |
| `scripts/risk/kill_switch.py` | ~250 | 6 | Q3.6 |
| `scripts/agent/daily_daemon.py` | ~80 | 4 | — |
| `scripts/agent/daily_runner.py` | ~40 | 4 | — |
| `scripts/agent/generators/daily/__init__.py` | ~5 | 4 | — |
| `scripts/agent/generators/daily/momentum.py` | ~200 | 4 | — |
| `scripts/agent/generators/daily/mean_reversion.py` | ~200 | 4 | — |
| `scripts/agent/generators/daily/cross_asset.py` | ~200 | 4 | — |
| `docker/Dockerfile.agent` | ~30 | 3 | — |
| `grafana/dashboards/signal_lifecycle.json` | ~200 | 7 | — |
| `grafana/dashboards/paper_trading.json` | ~150 | 7 | — |

### Modified Files (9)

| File | Change | Phase |
|------|--------|-------|
| `config/costs.toml` | Complete exchange entries (Hyperliquid + Binance tiers) per Q1.4 | 0 |
| `scripts/data/state.py` | +120 lines (lifecycle schema + methods, Arch-p.1 migration style) | 1 |
| `scripts/agent/base.py` | +10 lines (lifecycle insert in register_signal) | 2 |
| `scripts/execution/signal_bridge.py` | +100 lines (daemon mode, halt check, portfolio sizing, fill tracking) | 6 |
| `scripts/execution/hyperliquid_client.py` | +20 lines (get_candles method) | 4 |
| `scripts/agent/meta_daemon.py` | +5 lines (daily agent registration) | 4 |
| `config/agent.toml` | +25 lines (agent_daily + promotion sections, paper_days reconciliation) | 4, 5 |
| `docker-compose.yml` | +70 lines (9 new services) | 3, 4, 5, 6 |
| `nat` | +100 lines (lifecycle + daily-agent + risk CLI, NAT2/NAT9/NAT10 conventions) | 1, 4, 6 |

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
The daily agent follows the exact same pattern as macro_daemon.py (52 lines). No new framework, no new abstractions. Just another timeframe with its own generators — at 1–7 day horizons, strictly above macro's 1h–24h, so the four agents partition the horizon space without overlap.

### 6.7 Gates Are Imported, Not Invented
All promotion thresholds reference the canonical roadmap gates: G4 (walk-forward + deflated Sharpe) for VALIDATED, G8 (14-day paper, 5 criteria) for APPROVAL_PENDING, ROADMAP Step 9 for kill switches. The `[promotion]` config section holds *references with defaults*; divergence between it, `config/alpha.toml [gates]`, and `[agent.promotion]` is a bug. Rationale: the first draft of this plan used 7-day / Sharpe > 0.5 paper criteria that silently contradicted G8 — exactly the threshold drift that Q1.4 exists to kill on the cost side. An automated pipeline amplifies threshold drift; a human running scripts manually at least notices.

### 6.8 Provenance on Every Transition (PhD-Grade)
Each lifecycle row and history entry carries git_sha + data_fingerprint (`scripts/provenance.py`, P1.5). This serves both paths with the same mechanism: Q — "which code and data produced the OOS result that promoted this signal?" when debugging a live regression; P — a fully auditable discovery → live funnel for the P4.1 research statement, and fill-conditional IC (Q3.5) as adverse-selection evidence for the preprint. Per the roadmap's synergy table, this is shared work counted once.

### 6.9 Risk Layer Outside the Failure Domain
The kill switch is a separate daemon (Q3.6), not a function inside the bridge. An autonomous network removes the human from the loop on every cycle — so the component that halts trading must not share a process, codebase path, or failure mode with the component that trades. The bridge checks `halt_state.json` and cannot skip the check; docker `depends_on` makes the dependency structural.

### 6.10 The Funnel Is Seeded, Not Empty
The 4 deployable algorithms enter at VALIDATED on day one. A lifecycle that waits months for agent-discovered signals to mature would be untested infrastructure exactly when the first real promotion happens. Seeding means every transition — including the human gate — gets exercised with known-good signals first, and Q3.1 (paper trading deployment of existing winners) happens *through* the network instead of beside it.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| K2: 82 dead features (NaN) | Daily agent generators may reference unavailable features | Generators validate feature availability from manifest at init (§3.3); Q1.2 fix in progress |
| Zombie process (K5) | Agent containers crash silently | Docker `restart: unless-stopped` + health checks |
| Paper trading takes 14 days (G8) | Slow signal progression | Accepted — G8 is a hard roadmap gate, not negotiable for speed. Historical fast-forward allowed only for *infrastructure testing*, never for gate decisions |
| OOS validator subprocess hangs | Promotion daemon blocks | Timeout (600s) on subprocess.run(), log and skip |
| Signal bridge places bad orders | Capital loss | Independent kill-switch daemon (Q3.6) + in-process checks, dry-run first, 1% initial allocation (Q4.1) |
| Candle API rate limits | Candle fetcher fails | Backoff with jitter, cache aggressively |
| Docker work interrupts Jun 11–17 streak | Q1.1 reset — both paths delayed by a week+ | Compose invocations target new services by name only until Jun 17 (§3.8); ingestor runs on su-35, never rebuilt by this plan |
| Gate-threshold drift between configs | Promotion silently contradicts roadmap gates | Single `[promotion]` section as authority (§6.7); cross-check added to Q1.3's contract-test batch |
| OOS validation on insufficient data | Promoting noise — the 2-day hierarchical-combiner lesson | `min_clean_days_for_oos = 7` guard via `check_data_sufficiency.py` before any G4 run |
| Lifecycle migration conflicts with Q1.3 | Two migration frameworks in nat.db | Phase 1 coordinates with Arch-p.1 discipline explicitly (§3.1) |
| Daily-agent gates pass rarely until ~Aug | Looks broken; temptation to loosen gates | Documented as expected (§3.3 sufficiency caveat) — loosening gates to make an agent "productive" is how overfit signals reach paper |

---

## 8. Dependencies & Ordering

```
Phase 0 (Provenance + Costs) ←── pulled forward from Q1.4/P1.5
  ↓
Phase 1 (Lifecycle, seeded) ←── coordinates with Q1.3 migrations
  ↓
Phase 2 (Agent Integration) ←── depends on Phase 1
  ↓
Phase 3 (Dockerize) ←── independent, can parallel with Phase 2; constrained by Q1.1 until Jun 17
  ↓
Phase 4 (Daily Agent) ←── depends on Phase 3 for docker
  ↓
Phase 5 (Promotion) ←── depends on Phase 1 + Q1.1 data (≥7 clean days); daily agent optional
  ↓
Phase 6 (Risk + Execution) ←── kill switch first, then bridge; depends on Phase 5 (promotion feeds LIVE)
  ↓
Phase 7 (Monitoring) ←── depends on all above; NAT6/NAT7 viz should land by Phase 5
```

**Parallelizable:** Phases 2 and 3 can run concurrently. Phase 4 generators can be developed in parallel with Phase 3 dockerization. Phase 0 can start today — it touches nothing running.

**External dependencies (roadmap tasks this plan consumes but does not own):** Q1.1 data streak (passive, Jun 17), Q1.2 dead-feature fix (improves agent inputs, not blocking), Q1.3 Arch-p.1 (coordinate migrations), Q2.1 hierarchical revalidation (decides whether the combiner earns VALIDATED), NAT6/NAT7 viz (approval evidence).

---

## 9. Acceptance Criteria

- [ ] `nat lifecycle status` shows signal counts per state, seeded with the 4 deployable algorithms at VALIDATED
- [ ] Every lifecycle row and transition carries `git_sha` + `data_fingerprint` (spot-check via `nat lifecycle history <id>`)
- [ ] OOS validator, paper trader, and signal bridge all resolve costs via `load_costs()` — `grep` for hardcoded bps in the new code returns nothing (Q1.4 criterion)
- [ ] `nat lifecycle approve <id>` prints the G8 scorecard + provenance before confirming, then transitions to LIVE
- [ ] Running `nat agent once` creates a DISCOVERED signal in lifecycle
- [ ] `docker compose up agent-micro agent-mf agent-macro meta-agent` runs all agents (by name — ingestor untouched)
- [ ] Jun 11–17 accumulation streak intact after Phases 0–3 land (verify via `data_inventory` daily check)
- [ ] `nat daily-agent once` completes a research cycle with daily generators at 1–7d horizons
- [ ] Promotion daemon refuses OOS validation when < 7 consecutive clean days are available
- [ ] Promotion daemon auto-transitions DISCOVERED → VALIDATED → PAPER_TRADING
- [ ] After **14 days** paper trading (G8), signal reaches APPROVAL_PENDING with all 5 G8 criteria recorded in metrics
- [ ] Kill-switch daemon: all 4 synthetic threshold breaches trigger correctly, Telegram alert < 60s, bridge skips cycles while halted
- [ ] Signal bridge daemon mode reads LIVE signals, sizes via `meta_portfolio.py` risk parity, places orders, logs fills for Q3.5 conditional-IC analysis
- [ ] Grafana signal lifecycle dashboard shows funnel
- [ ] Full `docker compose up` starts 16+ services, all healthy
- [ ] End-to-end: ingestion → discovery → OOS → paper → approval → live trade

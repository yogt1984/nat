# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NAT is a quantitative research platform for extracting alpha signals from Hyperliquid perpetual futures. Rust handles real-time ingestion and feature computation; Python handles analysis, backtesting, and ML training.

See `docs/OBJECTIVE.md` for the project objective and the end-to-end loop (Ingest → Discover → Cluster → Visualize → Generate → Validate → Deploy).

## CLI

The `nat` command is the primary interface (251 commands). Run `nat help` for full docs, `nat commands` for a structured list, `nat commands --json` for machine-readable output.

Key command groups: `start/stop/status`, `test`, `build`, `alpha`, `agent/mf-agent/macro-agent/meta-agent`, `discovery`, `backtest`, `algorithm`, `config`, `docker`, `swarm`, `evolve`.

JSON output: `nat --json <command>` for programmatic use.

## Build & Run

```bash
nat start               # Start ingestor + watchdog + dashboard
nat stop                # Stop everything
nat status              # Health check (JSON)
nat log                 # Tail ingestor log
nat run serve           # Ingestor + dashboard on http://localhost:8080
nat build               # Release build (all binaries, LTO enabled)
nat build debug         # Debug build (faster iteration)

# Cargo directly (for single-crate work):
cd rust && cargo build --release        # equivalent to nat build
cd rust && cargo build                  # equivalent to nat build debug
```

Under the hood, the ingestor runs `cd rust && ./target/release/ing ../config/ing.toml`. All relative paths in config resolve from `rust/` — that's why `data_dir = "../data/features"` in `config/ing.toml`.

## Testing

```bash
nat test                                # Rust unit tests (cargo test --package ing)
nat test verbose                        # With --nocapture
cd rust && cargo test -- test_name      # Single Rust test
nat test validate                       # Live API validation (4 binaries against Hyperliquid)
pytest scripts/tests/                   # Python tests
nat test dashboard                      # Dashboard endpoint tests
nat test pipeline                       # Pipeline state machine tests
nat test agent                          # Agent tests (350+ tests: unit + integration + logging + research output)
```

## Architecture

See `FEATURES.md` for the complete feature manifest with formulas, paper references, and interpretation.

### Data Flow

```
Hyperliquid WebSocket → HyperliquidClient (ws/client.rs)
    → MarketState (state/mod.rs) [OrderBook + TradeBuffer + MarketContext]
    → FeatureComputer (features/*.rs) [236 features across 21 categories]
    → mpsc channel → ParquetWriter (output/writer.rs)
    → data/features/YYYY-MM-DD/*.parquet (rotated hourly)
```

Each symbol (BTC, ETH, SOL) runs in its own tokio task. Features are emitted every 100ms per symbol. The writer buffers 10,000 rows (~5.5 min at 30 rows/sec for 3 symbols) then flushes a row group.

### Feature Vector Contract

`Features` (features/mod.rs) has 14 base categories (always computed) + 7 optional categories (whale_flow, liquidation_risk, concentration, regime, gmm_classification, cross_symbol, heatmap). `to_vec()` always returns exactly `count_all()` elements (236), padding NaN for missing optionals. `names_all()` must match `to_vec()` length exactly — the Parquet schema is built from `names_all()` in `output/schema.rs`.

When adding a new feature category:
1. Create struct with `count()`, `names()`, `to_vec()` methods
2. Add to `Features` struct in `features/mod.rs`
3. Add to `to_vec()`, `names_all()`, `count_all()` — if optional, use NaN padding pattern
4. Schema updates automatically via `create_schema()` in `output/schema.rs`

### Main Loop (`main.rs`)

`tokio::select! { biased; }` — WebSocket messages have priority over the emission ticker and health timer. The `biased` keyword is intentional to prevent data loss under load.

### Parquet Writer

`ArrowWriter` buffers internally — `writer.flush()` is called after each batch write to force data to disk immediately. Without this, files stay 0 bytes until `close()` (hourly rotation or shutdown).

### Pipeline State Machine (Python)

`scripts/pipeline_runner.py`: IDLE → BUILDING → INGESTING → COLLECTING → ANALYZING → DONE. State persisted in `data/pipeline_state.json` for resume-on-interrupt.

### Alpha Pipeline (Python)

`scripts/alpha/alpha_pipeline.py`: IDLE → SCREENING → COMBINING → SIZING → VALIDATING → REGIME → MULTI_FREQ → PORTFOLIO → PAPER → DEPLOYING → DONE. Chains 9 alpha modules with quality gates (PASS/WEAK/FAIL) between each step. Config in `config/alpha.toml`. State persisted in `data/alpha/pipeline_state.json`.

Steps: (1) feature screening with FDR, (2) signal combination, (3) position sizing, (4) walk-forward validation, (5) regime conditioning, (6) multi-frequency integration, (7) portfolio assembly, (8) paper trading simulation, (9) deployment readiness. Gate thresholds in `[gates]` section of config.

CLI: `nat pipeline start`, `nat pipeline resume` (with `--force-gate`), `nat pipeline status`, `nat pipeline gates`, `nat pipeline run-step N`.

### Alpha Discovery Orchestrator (Python)

`scripts/discovery_orchestrator.py`: Continuous daemon that sweeps (symbol, horizon) combos for alpha signals, then pipelines winners through train → backtest → validate. All child scripts called via subprocess (not imported) to prevent OOM. Config in `config/discovery.toml`. State persisted in `data/discovery/orchestrator_state.json`.

Cycle: DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING → ALPHA_PIPELINE → REPORTING → SLEEPING. Signal sweep uses `phase1_signal_test.py --json-report` for structured output. Gates at each step (PASS/WEAK/FAIL). CLI: `nat discovery start|once|status|stop`.

### Autonomous Research Agent (Python)

`scripts/agent/daemon.py`: MANIFEST → GENERATE → EXECUTE → FDR → MONITOR → SLEEP. State persisted in `data/agent/agent_state.json`. 5-gate protocol per hypothesis: discovery (IC+dIC) → cost → temporal replication → symbol replication → correlation dedup. FDR control (BH q=0.05) at end of each cycle. Computation cache in `scripts/agent/cache.py` (SHA-256 keys, 7-day TTL). Web dashboard in `scripts/agent_dashboard.py` (stdlib HTTP on port 8060).

Daemon architecture is consolidated: `ResearchAgent` (base.py) owns the full cycle loop, state machine, generator dispatch (via `generator_module_prefix`), FDR control, hypothesis chaining, promotion checks, and structured research output emission. Subclass daemons (`daemon.py`, `mf_daemon.py`, `macro_daemon.py`) are thin (~80-110 LOC each), overriding only config attributes and `create_runner()`.

Key files:
- `scripts/agent/base.py` — ResearchAgent ABC + BaseRunner ABC (full cycle loop, generator dispatch, FDR, chaining, promotions, research output emission)
- `scripts/agent/daemon.py` — MicrostructureAgent(ResearchAgent) thin subclass + CLI
- `scripts/agent/runner.py` — MicrostructureRunner(BaseRunner) 5-gate executor
- `scripts/agent/hypothesis_queue.py` — SQLite-backed priority queue
- `scripts/agent/mf_daemon.py` — MediumFrequencyAgent(ResearchAgent) thin subclass + CLI for 1min-1h signals
- `scripts/agent/mf_runner.py` — MediumFrequencyRunner(BaseRunner) 4-gate executor
- `scripts/agent/macro_daemon.py` — MacroAgent(ResearchAgent) thin subclass + CLI for 1h-24h signals
- `scripts/agent/macro_runner.py` — MacroRunner(BaseRunner) 4-gate executor
- `scripts/agent/meta_daemon.py` — MetaAgent orchestrator (cross-agent budget, correlation, portfolio)
- `scripts/agent/meta_portfolio.py` — Risk parity weights, portfolio metrics, promotion evaluation
- `scripts/agent/research_output.py` — Structured JSON emitter (per-hypothesis records + cycle summaries with LaTeX math)
- `scripts/agent/cache.py` — Deterministic command cache
- `scripts/logging_config.py` — Centralized JSON logging with correlation context (cycle_id, hypothesis_id)
- `scripts/agent_dashboard.py` — Agent web dashboard with IC heatmap
- `config/agent.toml` — Gate thresholds, promotion criteria, generator config (`[agent]` + `[agent_mf]` + `[agent_macro]` + `[meta_agent]`)

### Algorithm Contract (`scripts/algorithms/`)

Every algorithm must implement `MicrostructureAlgorithm` ABC from `scripts/algorithms/base.py`:

```python
class MyAlgorithm(MicrostructureAlgorithm):
    def name(self) -> str: ...                              # unique identifier
    def alg_features(self) -> list[AlgorithmFeature]: ...   # output descriptors
    def required_columns(self) -> list[str]: ...            # input feature names
    def step(self, tick: dict[str, float]) -> dict[str, float]: ...  # one tick
    def reset(self) -> None: ...                            # clear internal state
    # Optional: override run_batch(df) for vectorized path
```

**Rules:**
- Register via `@register` decorator from `scripts/algorithms/registry.py`
- `step()` must return exactly the keys from `alg_features()` — no more, no less
- Handle NaN inputs gracefully: if any required column is NaN, return NaN for all outputs
- Feature names must start with `alg_` prefix
- Include a docstring with mathematical formulation and references
- `run_batch()` default iterates rows; override with vectorized numpy/pandas for performance
- Warmup period: first `warmup` rows of `run_batch()` output are NaN-blanked automatically
- Parameters should be configurable via `config/algorithms.toml`

**Verification:** `pytest scripts/tests/test_algorithm_smoke.py -k <name>`, then `nat algorithm evaluate --algorithm <name> --symbol BTC` on real data.

**Current winners (tested via `nat oos30`):** `jump_detector` (Lee-Mykland), `optimal_entry` (SPRT/Kalman), `funding_reversion`, `surprise_signal` (entropy), `3f_liquidity` (composite).

## Cargo Workspace

Four crates in `rust/` (dependency chain: `ing-types` → `ing-features` → `ing`):
- `ing-types` — Shared data types (OrderBook, TradeBuffer, MarketContext, FeaturesConfig, Regime)
- `ing-features` — Feature computation (26 files, 14.8K LOC). Depends on `ing-types`.
- `ing` — Main ingestor binary + ML + WebSocket + hypothesis testing. Depends on `ing-features`. Includes validation binaries (validate_api, validate_positions, etc.)
- `api` — REST/WebSocket API server (Axum on port 3000). Research endpoints (`/api/research/*`) read structured JSON from `data/research/`. Config: `NAT_RESEARCH_DIR` env var.

Release profile: LTO, single codegen unit, panic=abort, stripped.

## Configuration

- `config/ing.toml` — Ingestor (WebSocket URL, symbols, emission interval, output format)
- `config/agent.toml` — Agent daemon (cycle interval, 5-gate thresholds, FDR q, promotion criteria)
- `config/pipeline.toml` — Pipeline orchestration (ingestion duration, analysis thresholds)
- `config/alpha.toml` — Alpha pipeline (gate thresholds G1-G8, step parameters, symbols)
- `config/hypothesis_testing.toml` — Hypothesis test parameters
- `config/costs.toml` — Fee parameters (taker/maker bps, slippage) — single source of truth
- `config/algorithms.toml` — Algorithm parameters (per-algorithm tuning)
- `config/symbols.toml` — Traded symbols list
- `config/it_engine.toml` — IT engine (MI/CMI horizons, stride, conditioning, cost gate)
- `config/swarm_ranges.toml` — Swarm parameter ranges (~35D: algo thresholds, ensemble, trading, feature selection)
- `config/discovery.toml` — Discovery orchestrator (sweep intervals, gates)
- `config/llm.toml` — LLM client (model, endpoint, API key reference)

Environment overrides: `RUST_LOG`, `REDIS_URL`, `ING_DASHBOARD_ENABLED`, `ING_PROMETHEUS_ADDR`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Hypothesis Testing

5 hypotheses validated before deployment (in `hypothesis/`): whale flow → returns (H1), entropy × whale interaction (H2), liquidation cascade prediction (H3), concentration → volatility (H4), persistence indicator (H5). Decision matrix in `final_decision.rs`: 0-1 pass = NOGO, 2-3 = PIVOT, 4-5 = GO.

## Multi-Machine Setup

The ingestor runs on a second machine (su-35). `nat start` kills stale processes before starting. Data written to `data/features/` at project root.

## Config Swarm & Evolutionary Optimization

Three-tier cloud deployment architecture for automated parameter optimization:

### Tier 1: Continuous Cloud + Observability
Docker stack with Prometheus metrics (5s scrape, 90d retention), Grafana dashboards, Caddy HTTPS reverse proxy, PostgreSQL state persistence. All services health-checked.

### Tier 2: Config Swarm (`scripts/swarm/`)
Shared ingestor writes Parquet once; N evaluators read and score configs in parallel.

- `parquet_reader.py` — Time-windowed data loading from latest available date
- `config_generator.py` — ~35D parameter space sampling (random, grid, Optuna trial)
- `evaluator.py` — Run 5 algorithms via registry → ensemble → fitness metrics (Sharpe, IC, drawdown, signal count, turnover) → SQLite storage
- `orchestrator.py` — Parallel evaluation via ProcessPoolExecutor, ranking, export

Config: `config/swarm_ranges.toml` defines parameter ranges (15 algo + 5 ensemble + 8 trading + 7 feature selection params). Each param has `{ min, max, type }` with optional `log = true` for log-scale sampling.

CLI: `nat swarm run [--instances N]`, `nat swarm status`, `nat swarm results`, `nat swarm best`, `nat swarm generate`.

### Tier 3: Evolutionary Optimization (`scripts/swarm/optuna_optimizer.py`)
Optuna-based optimizer replacing random/grid search with intelligent Bayesian optimization.

- `NATOptimizer` class with CMA-ES (continuous 35D), TPE (mixed), NSGA-II (multi-objective) samplers
- Walk-forward train/test split: algorithms run on full data for warmup, fitness measured on OOS portion only (default 2/3 train, 1/3 test)
- Guard rails: hard constraints (signal count > 50/day, turnover < 100/day, OOS Sharpe > 0), IS/OOS overfit detection (ratio > 3.0 triggers penalty), deflated Sharpe ratio (Bailey & Lopez de Prado 2014)
- Multi-objective Pareto front via NSGA-II (maximize Sharpe, minimize drawdown, maximize IC)
- SQLite default storage, PostgreSQL for distributed multi-machine studies
- MedianPruner for early stopping of bad trials

CLI: `nat evolve start [--trials N --sampler cma|tpe|nsga2]`, `nat evolve status`, `nat evolve best`, `nat evolve pareto`, `nat evolve export`.

## Docker

`docker-compose.yml` runs: redis (6379), ingestor (8080), api (3000), alerts, prometheus (9090), grafana (3002), postgres (5432), optuna-dashboard (8070), caddy (80/443). Build with `nat docker build`, run with `nat docker up`.

Grafana auto-provisions a "NAT Overview" dashboard (anonymous access, no login). Prometheus scrapes the ingestor's metrics endpoint every 5s with 90-day retention. Optuna dashboard shows live optimization history, parameter importance, and Pareto fronts.

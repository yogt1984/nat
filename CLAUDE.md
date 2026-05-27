# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NAT is a quantitative research platform for extracting alpha signals from Hyperliquid perpetual futures. Rust handles real-time ingestion and feature computation; Python handles analysis, backtesting, and ML training.

## Build & Run

```bash
make build              # Debug build (faster iteration)
make release            # Release build (all binaries, LTO enabled)
make run                # Build release + start ingestor (kills stale processes first)
make run_and_serve      # Ingestor + dashboard on http://localhost:8080
```

`make run` does `cd rust && ./target/release/ing ../config/ing.toml`. All relative paths in config resolve from `rust/` — that's why `data_dir = "../data/features"` in `config/ing.toml`.

## Testing

```bash
make test                               # Rust unit tests (cd rust && cargo test --package ing)
make test_verbose                       # With --nocapture
cd rust && cargo test -- test_name      # Single test
make validate                           # Live API validation (4 binaries against Hyperliquid)
pytest scripts/tests/                   # Python tests
make test_dashboard                     # Dashboard endpoint tests
make test_pipeline                      # Pipeline state machine tests
make test_agent                         # Agent tests (350 tests: unit + integration + logging + research output)
```

## Architecture

See `FEATURES.md` for the complete feature manifest with formulas, paper references, and interpretation.

### Data Flow

```
Hyperliquid WebSocket → HyperliquidClient (ws/client.rs)
    → MarketState (state/mod.rs) [OrderBook + TradeBuffer + MarketContext]
    → FeatureComputer (features/*.rs) [209 features across 19 categories]
    → mpsc channel → ParquetWriter (output/writer.rs)
    → data/features/YYYY-MM-DD/*.parquet (rotated hourly)
```

Each symbol (BTC, ETH, SOL) runs in its own tokio task. Features are emitted every 100ms per symbol. The writer buffers 10,000 rows (~5.5 min at 30 rows/sec for 3 symbols) then flushes a row group.

### Feature Vector Contract

`Features` (features/mod.rs) has 13 base categories (always computed) + 6 optional categories (whale_flow, liquidation_risk, concentration, regime, gmm_classification, cross_symbol). `to_vec()` always returns exactly `count_all()` elements (209), padding NaN for missing optionals. `names_all()` must match `to_vec()` length exactly — the Parquet schema is built from `names_all()` in `output/schema.rs`.

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

CLI: `start`, `resume` (with `--force-gate`), `status`, `gates`, `run-step N`. Makefile targets: `alpha_pipeline`, `alpha_pipeline_resume`, `alpha_pipeline_force`, `alpha_pipeline_status`, `alpha_pipeline_gates`, `alpha_pipeline_step`.

### Alpha Discovery Orchestrator (Python)

`scripts/discovery_orchestrator.py`: Continuous daemon that sweeps (symbol, horizon) combos for alpha signals, then pipelines winners through train → backtest → validate. All child scripts called via subprocess (not imported) to prevent OOM. Config in `config/discovery.toml`. State persisted in `data/discovery/orchestrator_state.json`.

Cycle: DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING → ALPHA_PIPELINE → REPORTING → SLEEPING. Signal sweep uses `phase1_signal_test.py --json-report` for structured output. Gates at each step (PASS/WEAK/FAIL). CLI: `start`, `once`, `status`, `stop`. Makefile targets: `discovery_start`, `discovery_once`, `discovery_status`, `discovery_stop`.

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

Two crates in `rust/`:
- `ing` — Main ingestor library + binary, plus validation binaries (validate_api, validate_positions, etc.)
- `api` — REST/WebSocket API server (Axum on port 3000). Research endpoints (`/api/research/*`) read structured JSON from `data/research/`. Config: `NAT_RESEARCH_DIR` env var.

Release profile: LTO, single codegen unit, panic=abort, stripped.

## Configuration

- `config/ing.toml` — Ingestor (WebSocket URL, symbols, emission interval, output format)
- `config/agent.toml` — Agent daemon (cycle interval, 5-gate thresholds, FDR q, promotion criteria)
- `config/pipeline.toml` — Pipeline orchestration (ingestion duration, analysis thresholds)
- `config/alpha.toml` — Alpha pipeline (gate thresholds G1-G8, step parameters, symbols)
- `config/hypothesis_testing.toml` — Hypothesis test parameters

Environment overrides: `RUST_LOG`, `REDIS_URL`, `ING_DASHBOARD_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Hypothesis Testing

5 hypotheses validated before deployment (in `hypothesis/`): whale flow → returns (H1), entropy × whale interaction (H2), liquidation cascade prediction (H3), concentration → volatility (H4), persistence indicator (H5). Decision matrix in `final_decision.rs`: 0-1 pass = NOGO, 2-3 = PIVOT, 4-5 = GO.

## Multi-Machine Setup

The ingestor runs on a second machine (su-35). `make run` kills stale processes before starting. Python defaults to `python3` (`PYTHON ?= python3` in Makefile). Data written to `data/features/` at project root.

## Docker

`docker-compose.yml` runs: redis (6379), ingestor, api (3000), alerts. Build with `make docker_build`, run with `make docker_up`.

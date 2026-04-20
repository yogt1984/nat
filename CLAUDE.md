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
```

## Architecture

See `FEATURES.md` for the complete feature manifest with formulas, paper references, and interpretation.

### Data Flow

```
Hyperliquid WebSocket → HyperliquidClient (ws/client.rs)
    → MarketState (state/mod.rs) [OrderBook + TradeBuffer + MarketContext]
    → FeatureComputer (features/*.rs) [191 features across 14 categories]
    → mpsc channel → ParquetWriter (output/writer.rs)
    → data/features/YYYY-MM-DD/*.parquet (rotated hourly)
```

Each symbol (BTC, ETH, SOL) runs in its own tokio task. Features are emitted every 100ms per symbol. The writer buffers 10,000 rows (~5.5 min at 30 rows/sec for 3 symbols) then flushes a row group.

### Feature Vector Contract

`Features` (features/mod.rs) has 10 base categories (always computed) + 5 optional categories (whale_flow, liquidation_risk, concentration, regime, gmm_classification). `to_vec()` always returns exactly `count_all()` elements (191), padding NaN for missing optionals. `names_all()` must match `to_vec()` length exactly — the Parquet schema is built from `names_all()` in `output/schema.rs`.

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

## Cargo Workspace

Two crates in `rust/`:
- `ing` — Main ingestor library + binary, plus validation binaries (validate_api, validate_positions, etc.)
- `api` — REST/WebSocket API server (Axum on port 3000)

Release profile: LTO, single codegen unit, panic=abort, stripped.

## Configuration

- `config/ing.toml` — Ingestor (WebSocket URL, symbols, emission interval, output format)
- `config/pipeline.toml` — Pipeline orchestration (ingestion duration, analysis thresholds)
- `config/hypothesis_testing.toml` — Hypothesis test parameters

Environment overrides: `RUST_LOG`, `REDIS_URL`, `ING_DASHBOARD_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Hypothesis Testing

5 hypotheses validated before deployment (in `hypothesis/`): whale flow → returns (H1), entropy × whale interaction (H2), liquidation cascade prediction (H3), concentration → volatility (H4), persistence indicator (H5). Decision matrix in `final_decision.rs`: 0-1 pass = NOGO, 2-3 = PIVOT, 4-5 = GO.

## Multi-Machine Setup

The ingestor runs on a second machine (su-35). `make run` kills stale processes before starting. Python defaults to `python3` (`PYTHON ?= python3` in Makefile). Data written to `data/features/` at project root.

## Docker

`docker-compose.yml` runs: redis (6379), ingestor, api (3000), alerts. Build with `make docker_build`, run with `make docker_up`.

# NAT

Quantitative research platform for extracting microstructure features from Hyperliquid perpetual futures. Rust handles real-time ingestion and feature computation; Python handles analysis, backtesting, and ML training.

## What It Does

NAT connects to Hyperliquid's WebSocket API, maintains L2 order book state, and computes 191 features at 100ms intervals per symbol. Features are written to hourly-rotated Parquet files for offline analysis.

```
Hyperliquid WebSocket → OrderBook + TradeBuffer + MarketContext
    → FeatureComputer (191 features, 15 categories)
    → Parquet files (data/features/YYYY-MM-DD/*.parquet)
```

Each symbol (BTC, ETH, SOL) runs in its own tokio task. The writer buffers 10,000 rows (~5.5 min at 30 rows/sec for 3 symbols) then flushes a row group.

## Feature Categories

191 features across 15 categories. See [`FEATURES.md`](FEATURES.md) for formulas, ranges, and paper references.

| Category | Count | Prefix | Key References |
|----------|-------|--------|----------------|
| Raw | 10 | `raw_` | Gatheral & Oomen (2010) |
| Imbalance | 8 | `imbalance_` | Cont, Stoikov & Talreja (2010) |
| Flow | 12 | `flow_` | — |
| Volatility | 8 | `vol_` | Parkinson (1980) |
| Entropy | 24 | `ent_` | Bandt & Pompe (2002) |
| Context | 9 | `ctx_` | — |
| Trend | 15 | `trend_` | Jegadeesh & Titman (1993) |
| Illiquidity | 12 | `illiq_` | Kyle (1985), Amihud (2002), Hasbrouck (2009) |
| Toxicity | 10 | `toxic_` | Easley et al. (2012) |
| Derived | 15 | `derived_` | — |
| Whale Flow | 12 | `whale_` | — |
| Liquidation | 13 | `liquidation_` | — |
| Concentration | 15 | `top`/mixed | — |
| Regime | 20 | `regime_` | — |
| GMM | 8 | `regime`/`prob_` | — |

123 base features are always computed. 68 optional features (whale flow, liquidation, concentration, regime, GMM) require additional data sources or warmup and are NaN-padded when absent. `Features::to_vec()` always returns exactly 191 elements.

**Placeholders**: `vol_spread_std_1m` and `vol_zscore` are hardcoded to 0.0 (need historical buffers not yet wired through).

## Build & Run

```bash
make build              # Debug build
make release            # Release build (LTO, stripped)
make run                # Build release + start ingestor
make run_and_serve      # Ingestor + dashboard on http://localhost:8080
```

`make run` does `cd rust && ./target/release/ing ../config/ing.toml`. Config paths resolve from `rust/`.

## Testing

```bash
make test                               # Rust unit tests (386 passing)
make test_verbose                       # With --nocapture
cd rust && cargo test -- test_name      # Single test
make validate                           # Live API validation (4 binaries)
pytest scripts/tests/                   # Python tests
```

## Configuration

- `config/ing.toml` — Ingestor config (WebSocket URL, symbols, emission interval, output)
- `config/pipeline.toml` — Pipeline orchestration
- `config/hypothesis_testing.toml` — Hypothesis test parameters

Environment overrides: `RUST_LOG`, `REDIS_URL`, `ING_DASHBOARD_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Hypothesis Testing

5 hypotheses validated against collected data before deployment:

| ID | Question | Success Criteria |
|----|----------|------------------|
| H1 | Does whale flow predict returns? | r > 0.05, p < 0.001, MI > 0.02 bits |
| H2 | Does entropy interact with whale flow? | Lift > 10%, p < 0.01 |
| H3 | Are liquidation cascades predictable? | Precision > 30%, Lift > 2x |
| H4 | Does concentration predict volatility? | r > 0.2, partial r > 0.1 |
| H5 | Does a persistence indicator work OOS? | WF Sharpe > 0.5, OOS/IS > 0.7 |

Decision matrix in `hypothesis/final_decision.rs`: 0-1 pass = NOGO, 2-3 = PIVOT, 4-5 = GO. Tests use Bonferroni correction, walk-forward validation, and OOS/IS ratio checks.

```bash
make test_hypotheses DATA=./data/features
```

## Pipeline

Python state machine in `scripts/pipeline_runner.py`:

```
IDLE → BUILDING → INGESTING → COLLECTING → ANALYZING → DONE
```

State persisted in `data/pipeline_state.json` for resume on interrupt.

```bash
make pipeline_start     # Start pipeline
make pipeline_status    # Check state
make pipeline_resume    # Resume after interrupt
```

## ML Infrastructure

Python-based training, scoring, backtesting, and serving pipeline.

```bash
# Train a model on a data snapshot
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=lightgbm

# Score new data
make score_data MODEL_PATH=./models/model.pkl

# Walk-forward backtest
make backtest_ml_validate ML_PREDICTIONS=./predictions.parquet

# Experiment tracking
make experiments_list
make experiments_best METRIC=sharpe_ratio

# Model serving (REST API on port 8000)
make serve_best METRIC=sharpe_ratio
```

REST endpoints: `/health`, `/models`, `/models/best`, `/predict`, `/predict/batch`, `/reload`.

## Dashboard

Real-time log monitoring via WebSocket at `http://localhost:8080`.

```bash
make run_and_serve      # Ingestor + dashboard
make tunnel             # Expose via cloudflared
```

## Data Validation

```bash
make validate_data                       # Validate all Parquet files
make validate_data_recent HOURS=24       # Last 24 hours only
```

Checks: file integrity, continuity gaps, NaN ratios, feature ranges, cross-symbol correlation, data rate, sequence monotonicity.

## Visualization

Python library for feature exploration in `scripts/viz/`:

```python
from scripts.viz import FeaturePlotter, EventPlotter, CorrelationAnalyzer

plotter = FeaturePlotter(df)
plotter.plot_feature_timeseries(['whale_net_flow_1h', 'vpin_10'])
```

```bash
make explore    # Launch Jupyter notebook
```

## Project Structure

```
rust/
  ing/          — Ingestor library + binaries
    src/
      main.rs           Entry point, tokio::select! loop
      ws/               WebSocket client (Hyperliquid)
      state/            OrderBook, TradeBuffer, MarketContext
      features/         15 feature modules (191 features)
      hypothesis/       H1-H5 statistical tests + decision matrix
      output/           Parquet writer (ArrowWriter)
      dashboard/        Axum WebSocket monitoring server
      ml/               GMM regime classifier
      whales/           Whale registry & classification
      positions/        Position tracking
  api/          — REST/WebSocket API server (Axum, port 3000)

scripts/
  pipeline_runner.py        Pipeline state machine
  experiment_governance.py  Data snapshots + experiment tracking
  model_serving.py          REST API for model predictions
  score_data.py             Prediction generation
  viz/                      Plotting library (features, events, correlations)
  tests/                    Python test suite

config/
  ing.toml                  Ingestor configuration
  pipeline.toml             Pipeline orchestration
  hypothesis_testing.toml   Hypothesis test parameters

data/features/              Parquet output (YYYY-MM-DD/*.parquet)
```

## Docker

```bash
make docker_build       # Build images
make docker_up          # Run: redis (6379), ingestor, api (3000), alerts
make docker_down        # Stop
```

## Multi-Machine Setup

The ingestor runs on a separate machine (`su-35`). `make run` kills stale processes before starting. Data is written to `data/features/` at project root.

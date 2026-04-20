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

## Feature Vector (191 dimensions)

123 base features are always computed. 68 optional features (categories 11-15) require additional data sources or warmup and are NaN-padded when absent. `Features::to_vec()` always returns exactly 191 elements. See [`FEATURES.md`](FEATURES.md) for the full manifest with Parquet column names.

### 1. Raw (10) — L2 order book observables

Direct measurements from the L2 snapshot. **Microprice** (Gatheral & Oomen 2010): P_micro = (V_ask · P_bid + V_bid · P_ask) / (V_bid + V_ask) — shifts toward the thinner side, reflecting higher probability of price movement in that direction.

`raw_midprice`, `raw_spread`, `raw_spread_bps`, `raw_microprice`, `raw_{bid,ask}_depth_{5,10}`, `raw_{bid,ask}_orders_5`

### 2. Imbalance (8) — Order book asymmetry

Ref: Cont, Stoikov & Talreja (2010). Volume imbalance: I = (Σ bid_vol - Σ ask_vol) / (Σ bid_vol + Σ ask_vol) at L1/L5/L10 depths. Pressure score: cumulative depth weighted by 1/(1 + distance_bps/10), giving exponentially decaying importance to liquidity further from midprice. Normalized to [0,1] by max(bid_pressure, ask_pressure).

`imbalance_qty_{l1,l5,l10}`, `imbalance_orders_l5`, `imbalance_notional_l5`, `imbalance_depth_weighted`, `imbalance_pressure_{bid,ask}`

### 3. Flow (12) — Trade arrival dynamics

Trade count, volume, and aggressor ratio at three windows: 1s (market-maker timescale), 5s (quote update cycle), 30s (informed trader execution horizon). VWAP deviation = (VWAP_5s - last_price) / last_price. Trade intensity = trades/sec via 5s EMA.

`flow_{count,volume}_{1s,5s,30s}`, `flow_aggressor_ratio_{5s,30s}`, `flow_vwap_5s`, `flow_vwap_deviation`, `flow_avg_trade_size_30s`, `flow_intensity`

### 4. Volatility (8) — Realized and range-based estimators

Ref: Parkinson (1980). Realized volatility: RV = sqrt(Σ r_i² / N) over 60 (1m) and 300 (5m) tick windows. Parkinson estimator: σ = ln(H/L) / sqrt(4·ln(2)) — single-window approximation using 300 ticks. Midprice std: sample standard deviation over 60 ticks.

`vol_returns_{1m,5m}`, `vol_parkinson_5m`, `vol_spread_mean_1m` (point-in-time, misnomer), `vol_spread_std_1m` (**placeholder: 0.0**), `vol_midprice_std_1m`, `vol_ratio_short_long`, `vol_zscore` (**placeholder: 0.0**)

### 5. Entropy (24) — Information content and predictability

Refs: Bandt & Pompe (2002), Shannon (1948), Zunino et al. (2009).

**Permutation entropy** (10): For embedding dimension m=3, count occurrences of each of the m!=6 ordinal patterns in windows of 8/16/32 ticks, compute Shannon entropy, normalize by ln(m!) to [0,1]. Applied to returns and L1 imbalance series. Distribution entropy: bin continuous values into N equal-width bins, H = -Σ p_i ln(p_i), normalize by ln(N). Applied to spreads (10 bins), trade volumes (10 bins), book depth proportions, trade sizes (5 bins).

**Tick entropy** (7): Shannon entropy of {up, down, neutral} trade direction counts within each window (1s/5s/10s/15s/30s/1m/15m). Range [0, ln(3)]. Low = trending, high = random.

**Volume-weighted tick entropy** (7): Same windows, but directions weighted by trade volume.

### 6. Context (9) — Hyperliquid market metadata

From the `activeAssetCtx` WebSocket channel. Funding rate and z-score, open interest (USD) with 5-minute absolute and percent change (60-sample lookback at ~5s updates), mark-index premium (bps), 24h volume and volume ratio, mark-oracle divergence.

`ctx_funding_rate`, `ctx_funding_zscore`, `ctx_open_interest`, `ctx_oi_change_{5m,pct_5m}`, `ctx_premium_bps`, `ctx_volume_24h`, `ctx_volume_ratio`, `ctx_mark_oracle_divergence`

### 7. Trend (15) — Persistence and mean-reversion detection

Refs: Jegadeesh & Titman (1993), Mandelbrot (1971). Momentum: linear regression slope over 60/300/600 ticks (~6s/30s/60s). R²: goodness of fit [0,1]. Monotonicity: fraction of ticks in majority direction, range [0.5, 1.0]. Hurst exponent via rescaled range: H < 0.5 = mean-reverting, H > 0.5 = trending. MA crossover: EMA(10) - EMA(50).

`trend_momentum_{60,300,600}`, `trend_momentum_r2_{60,300,600}`, `trend_monotonicity_{60,300,600}`, `trend_hurst_{300,600}`, `trend_ma_crossover`, `trend_ma_crossover_norm`, `trend_ema_{short,long}`

### 8. Illiquidity (12) — Market impact measures

Refs: Kyle (1985), Amihud (2002), Hasbrouck (2009).

- **Kyle's lambda**: λ = Cov(ΔP, signed_vol) / Var(signed_vol) — price impact per unit volume
- **Amihud**: Σ|r| / Σv × 10⁶ — return per dollar of volume (ratio of sums, not mean of ratios)
- **Hasbrouck**: permanent price impact via OLS
- **Roll spread**: S = 2·sqrt(-Cov(ΔP_t, ΔP_{t-1})) — implied spread from autocovariance

Each computed over 100-trade and 500-trade windows, plus kyle_ratio, amihud_ratio (short/long), composite score, and trade count.

### 9. Toxicity (10) — Adverse selection and informed flow

Refs: Easley et al. (2012), Glosten & Milgrom (1985). **VPIN** (Volume-Synchronized Probability of Informed Trading): Σ|V_buy - V_sell| / Σ(V_buy + V_sell) over 10 and 50 volume buckets. Effective spread: 2 × mean(|trade_price - VWAP|) using VWAP as midpoint proxy (price units). Realized spread: mean(direction × (trade_price - price_{t+5}) × 2) with 5-trade lookahead. Adverse selection = effective - realized. Composite toxicity index: weighted combination of VPIN, adverse selection, and flow imbalance, normalized to [0,1].

`toxic_vpin_{10,50}`, `toxic_vpin_roc`, `toxic_adverse_selection`, `toxic_effective_spread`, `toxic_realized_spread`, `toxic_flow_imbalance`, `toxic_flow_imbalance_abs`, `toxic_index`, `toxic_trade_count`

### 10. Derived (15) — Cross-category composites

Interaction terms combining entropy, trend, volatility, illiquidity, and toxicity. Key inputs: tick_entropy = ent_tick_entropy_30s, monotonicity = trend_monotonicity_{60,300}, vol = vol_returns_1m × 100 clamped to [0,1], kyle = illiq_kyle_100 / 100 clamped to [0,1].

- **Trend strength**: sign(momentum) × (monotonicity - 0.5)×2 × (1 - tick_entropy), range [-1, 1]
- **Regime type score**: vol × (1 - 2·tick_entropy) — positive = breakout, negative = chaotic volatility
- **Regime indicator**: (mean_revert_score - trending_score - flow_factor), clamped to [-1, 1], where trending = (1-ent)·(mono-0.5)·2·|strength|, mean_revert = ent·(1-mono)·2·(1-|strength|)
- **Toxicity-regime**: toxicity_index × tick_entropy (toxic in choppy markets)
- **Illiquidity-trend**: kyle · |momentum_60| × 1000 (informed directional flow)

### 11. Whale Flow (12) — Optional

Net position changes for large accounts across 1h/4h/24h windows. Normalized flow, momentum, intensity, rate of change, buy ratio, directional agreement, active count, total activity.

### 12. Liquidation Risk (13) — Optional

Liquidation volume mapped at ±1%/2%/5%/10% price moves. Asymmetry = (risk_above - risk_below) / total. Intensity = total risk / OI. Nearest cluster distance (%), positions at risk count, largest position at risk.

### 13. Concentration (15) — Optional

Position crowding via inequality metrics: top-{5,10,20,50} share of OI, Herfindahl index (Σ share_i²), Gini coefficient, Theil index (Σ share_i · ln(share_i / (1/N)), range [0, ln(N)]). Whale/retail ratios, concentration change over 1h, HHI rate of change, trend, position counts.

### 14. Regime Detection (20) — Optional

Accumulation/distribution cycle features at minute-level resolution:

- **Absorption**: volume / (|ΔPrice| + ε) at 1h/4h/24h + z-score. High absorption = volume absorbed without price move
- **Divergence**: actual ΔP - λ·signed_volume at 1h/4h/24h + z-score. Negative = price lagging volume (accumulation)
- **Churn**: (buy_vol + sell_vol) / (|buy_vol - sell_vol| + ε) at 1h/4h/24h + z-score. High = two-sided activity
- **Range position**: (price - min) / (max - min) at 4h/24h/1w + 24h range width
- **Composite**: accumulation_score, distribution_score, clarity

### 15. GMM Classification (8) — Optional

Gaussian Mixture Model fitted on [kyle_lambda, vpin, absorption_zscore, hurst, whale_net_flow_1h]. Outputs: regime label ∈ {0..4}, posterior probabilities for accumulation/markup/distribution/markdown/ranging, confidence = max(p_i), regime entropy = -Σ p_i ln(p_i).

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

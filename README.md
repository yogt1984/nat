# NAT

```
    _   _     _     _____
   | \ | |   / \   |_   _|
   |  \| |  / _ \    | |
   | |\  | / ___ \   | |
   |_| \_|/_/   \_\  |_|

   Quantitative Research Platform
   Hyperliquid Perpetual Futures
```

NAT is a quantitative research platform for extracting alpha signals from Hyperliquid perpetual futures. Rust handles real-time ingestion and feature computation; Python handles regime profiling, backtesting, and ML training.

## Why This System Exists

Market making on decentralized perpetual exchanges is a spread-setting problem under uncertainty. The market maker must continuously choose a half-spread $\delta$ that balances two competing objectives:

1. **Revenue from the spread**: wider $\delta$ captures more per round-trip
2. **Adverse selection**: wider $\delta$ reduces fill probability; narrow $\delta$ invites informed flow that moves through the quote

The classical result of Glosten & Milgrom (1985) proves that a competitive market maker must set:

$$\delta \geq \mu \cdot (V^+ - V^-) / (V^+ + V^-)$$

where $\mu$ is the probability of facing an informed trader, and $V^+, V^-$ are expected asset values conditional on a buy/sell. This is the **zero-profit condition** — any spread below this bound guarantees negative expected PnL due to adverse selection.

Avellaneda & Stoikov (2008) extend this to continuous-time with inventory:

$$\delta^*(q, \sigma, T) = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

where $\gamma$ is risk aversion, $\sigma$ is volatility, $q$ is inventory, and $k$ is fill intensity. The key insight: **optimal spread is a function of market state**, not a constant.

### The Entropy-Adaptive Hypothesis

NAT's core thesis is that **Shannon entropy of tick direction** is a sufficient statistic for the informed-trader intensity $\mu$ and therefore for optimal spread:

$$H(t) = -\sum_{d \in \{+,-,0\}} p_d(t) \ln p_d(t), \quad H \in [0, \ln 3]$$

**Theoretical justification**:

- When $H \to 0$ (trending): tick direction is predictable, implying informed flow dominates. Glosten-Milgrom says widen spread (high $\mu$).
- When $H \to \ln 3$ (random walk): no directional information in order flow, implying noise traders dominate. Narrow spread captures volume safely (low $\mu$).
- When $H$ is intermediate: regime is transitioning, spread should interpolate.

This maps to a testable statistical hypothesis:

> **H₀**: Optimal spread is independent of entropy regime  
> **H₁**: $E[\delta^* | H \in R_i] \neq E[\delta^* | H \in R_j]$ for at least one pair $(R_i, R_j)$

Tested via Kruskal-Wallis with effect size $\eta^2 > 0.01$ and $p < 0.01$.

### Why FPGA Without Co-location

Hyperliquid is a decentralized exchange — there is no co-location facility. All participants face the same network latency (~50-200ms to validators). The advantage of FPGA is not raw speed but **deterministic execution**: once a spread decision is made, the quote update executes in constant time regardless of system load, eliminating the tail-latency jitter that software market makers suffer during volatility spikes (exactly when correct spread-setting matters most).

The decision tree structure of LightGBM maps directly to FPGA lookup tables: each split is a comparator, each leaf is a fixed-point spread value. The full model evaluates in $O(\text{depth})$ clock cycles with no branch misprediction.

## Theoretical Foundations

| Concept | Reference | Application in NAT |
|---------|-----------|-------------------|
| Adverse selection & spread | Glosten & Milgrom (1985) | Zero-profit condition, spread lower bound |
| Inventory-optimal spread | Avellaneda & Stoikov (2008) | Inventory penalty $\gamma q^2 \sigma^2$, skew |
| Kyle's lambda (price impact) | Kyle (1985) | Feature: illiquidity measure |
| VPIN (informed flow) | Easley, Lopez de Prado & O'Hara (2012) | Feature: toxicity detection |
| Permutation entropy | Bandt & Pompe (2002) | Feature: nonlinear serial dependence |
| Tick entropy as regime indicator | Zunino et al. (2009) | Core thesis: entropy → optimal spread |
| Order flow imbalance | Cont, Kukanov & Stoikov (2014) | Feature: short-horizon prediction |
| Hurst exponent | Mandelbrot (1971) | Feature: persistence vs mean-reversion |
| Parkinson volatility | Parkinson (1980) | Feature: range-based vol estimator |
| GMM regime detection | Hamilton (1989), McLachlan & Peel (2000) | Phase 2: unsupervised regime discovery |
| Walk-forward validation | White (2000), Bailey et al. (2014) | No lookahead in all evaluations |

## Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Walk-forward edge (Phase 1) | +4.18% over base rate | Statistically significant signal in 123 features |
| Net PnL after costs (Phase 1) | -0.45 bps/trade | Signal alone insufficient; regime conditioning needed |
| Test coverage | 125 EAMM + 196 pipeline + 386 Rust | Comprehensive validation at every layer |
| Feature count | 191 (123 active) | Dense microstructure representation |
| Ingestion rate | 30 rows/sec × 3 symbols | 100ms resolution, sufficient for 5-min decisions |

## Architecture

```
Hyperliquid WebSocket --> OrderBook + TradeBuffer + MarketContext
    --> FeatureComputer (191 features, 15 categories)
    --> Parquet files (data/features/YYYY-MM-DD/*.parquet)
        --> Python Pipeline: bars --> derivatives --> PCA --> GMM --> regimes
            --> Validation (Q1-Q3) --> Online Classifier --> Trading Decisions
```

Each symbol (BTC, ETH, SOL) runs in its own tokio task. Features are emitted every 100ms per symbol. The writer buffers 10,000 rows (~5.5 min at 30 rows/sec for 3 symbols) then flushes a row group.

## Prototyping History

### Phase 1: Direct ML Attack (Completed)

**Goal**: Can a gradient-boosted model predict return sign from raw microstructure features?

**Method**: LightGBM binary classifier on 123 active features. Walk-forward validation on 5.5 days of BTC data (441,611 rows at 100ms). Tested 4 horizons (30s, 5min, 30min, 1h). Confidence-filtered trading at thresholds 0.50-0.80. Full transaction cost model: 8 bps round-trip taker, 1 bps maker.

**Results**:
- Walk-forward edge: +4.18% over base rate (statistically significant)
- Best accuracy: 54.2% at 0.80 confidence threshold
- Net P&L after costs: **-0.45 bps/trade (unprofitable)**
- Top features by importance: funding rate, realized vol (5m), momentum (60s), volume ratio, spread

**Conclusion**: Real signal exists in microstructure features but is too weak (~4% edge) to overcome transaction costs at 5-minute horizon with a flat feature-to-prediction approach. Signal sources align with published microstructure literature: funding rate (directional bias), volatility (continuation/reversal), momentum (short-term persistence), volume acceleration.

**Key insight**: A single global model averages over distinct market regimes. A trending market and a ranging market have identical feature distributions in some dimensions but completely different return dynamics. Regime conditioning should amplify the within-regime signal.

See [`docs/PHASE1_ALGORITHM.md`](docs/PHASE1_ALGORITHM.md) for full mathematical specification and results.

### Phase 2: Hierarchical Regime Profiling (Completed)

**Goal**: Discover natural market regimes from feature dynamics, validate their structural reality and predictive power, enable regime-conditioned trading.

**Method**: Unsupervised hierarchical clustering pipeline:

1. **Feature derivation** — Temporal derivatives (velocity, acceleration, z-score, slope, relative volatility) and spectral features (FFT low/high power, spectral ratio, dominant period) over rolling windows [5, 15, 30 bars]. Cross-feature derivatives (ratios, rolling correlations, divergences) between related feature pairs.

2. **Dimensionality reduction** — Variance filtering (drop <1% variance), correlation filtering (drop >0.95 correlated), PCA with Ledoit-Wolf shrinkage on the regularized covariance matrix. Retain components explaining 95% of variance.

3. **Structural break detection** — PELT (Pruned Exact Linear Time) changepoint detection on PCA-reduced space with BIC penalty. Segments the time series into statistically distinct episodes.

4. **Macro regime discovery** — Gaussian Mixture Model (GMM) fitted on PCA-reduced derivatives. Optimal k selected via BIC with block bootstrap stability validation (ARI > 0.6 required). Each regime is a multivariate Gaussian component representing a distinct market phase.

5. **Micro-state discovery** — Within each macro regime, fit a second-level GMM to discover sub-states (e.g., "early trending" vs "late trending" within a trend regime).

6. **Validation gates**:
   - Q1 (Structural): Silhouette score > 0.25, bootstrap ARI > 0.6
   - Q2 (Predictive): Kruskal-Wallis H-test on forward returns across states (p < 0.05, eta-squared > 0.01)
   - Q3 (Operational): Self-transition rate > 0.8, mean duration > 3 bars

7. **Decision logic**: Q1 fail = DROP (clusters aren't real), Q2 fail = COLLECT (need more data), Q3 fail = PIVOT (real but not tradeable), all pass = GO (deploy).

**Implementation**: 15 Python modules (8,300+ lines), 196 tests all passing. Includes online drift-detecting classifier with persistence (save/load fitted models), cross-symbol consistency analysis via pairwise Adjusted Rand Index, and automated markdown report generation.

See [`docs/specs/PROFILING_MATHEMATICAL_FORMULATION.md`](docs/specs/PROFILING_MATHEMATICAL_FORMULATION.md) for the full mathematical formulation.

### Phase 3: Alpha Discovery (Planned)

**Goal**: Convert regime detection into profitable trading signals.

**Algorithm groups under investigation**:

- **Regime-conditioned models** — Train separate predictors per detected regime. Within-regime signal should exceed the global 4% edge.
- **Funding rate mean-reversion** — Exploit extreme funding as a directional signal (8h Hyperliquid settlement cycle).
- **Order flow imbalance** — Cumulative multi-level imbalance for short-horizon prediction (Cont et al., 2014).
- **Maker-only execution** — Exploit Hyperliquid's 0% maker fee to reduce round-trip costs from 8 bps to ~1 bps.

## Feature Vector (191 dimensions)

123 base features are always computed. 68 optional features (categories 11-15) require additional data sources or warmup and are NaN-padded when absent. `Features::to_vec()` always returns exactly 191 elements. See [`FEATURES.md`](FEATURES.md) for the full manifest with Parquet column names.

### 1. Raw (10) — L2 order book observables

Direct measurements from the L2 snapshot. **Microprice** (Gatheral & Oomen 2010): P_micro = (V_ask * P_bid + V_bid * P_ask) / (V_bid + V_ask) — shifts toward the thinner side, reflecting higher probability of price movement in that direction.

`raw_midprice`, `raw_spread`, `raw_spread_bps`, `raw_microprice`, `raw_{bid,ask}_depth_{5,10}`, `raw_{bid,ask}_orders_5`

### 2. Imbalance (8) — Order book asymmetry

Ref: Cont, Stoikov & Talreja (2010). Volume imbalance: I = (bid_vol - ask_vol) / (bid_vol + ask_vol) at L1/L5/L10 depths. Pressure score: cumulative depth weighted by 1/(1 + distance_bps/10), giving exponentially decaying importance to liquidity further from midprice. Normalized to [0,1] by max(bid_pressure, ask_pressure).

`imbalance_qty_{l1,l5,l10}`, `imbalance_orders_l5`, `imbalance_notional_l5`, `imbalance_depth_weighted`, `imbalance_pressure_{bid,ask}`

### 3. Flow (12) — Trade arrival dynamics

Trade count, volume, and aggressor ratio at three windows: 1s (market-maker timescale), 5s (quote update cycle), 30s (informed trader execution horizon). VWAP deviation = (VWAP_5s - last_price) / last_price. Trade intensity = trades/sec via 5s EMA.

`flow_{count,volume}_{1s,5s,30s}`, `flow_aggressor_ratio_{5s,30s}`, `flow_vwap_5s`, `flow_vwap_deviation`, `flow_avg_trade_size_30s`, `flow_intensity`

### 4. Volatility (8) — Realized and range-based estimators

Ref: Parkinson (1980). Realized volatility: RV = sqrt(r_i^2 / N) over 60 (1m) and 300 (5m) tick windows. Parkinson estimator: sigma = ln(H/L) / sqrt(4*ln(2)) — single-window approximation using 300 ticks. Midprice std: sample standard deviation over 60 ticks.

`vol_returns_{1m,5m}`, `vol_parkinson_5m`, `vol_spread_mean_1m` (point-in-time, misnomer), `vol_spread_std_1m` (**placeholder: 0.0**), `vol_midprice_std_1m`, `vol_ratio_short_long`, `vol_zscore` (**placeholder: 0.0**)

### 5. Entropy (24) — Information content and predictability

Refs: Bandt & Pompe (2002), Shannon (1948), Zunino et al. (2009).

**Permutation entropy** (10): For embedding dimension m=3, count occurrences of each of the m!=6 ordinal patterns in windows of 8/16/32 ticks, compute Shannon entropy, normalize by ln(m!) to [0,1]. Applied to returns and L1 imbalance series. Distribution entropy: bin continuous values into N equal-width bins, H = -p_i ln(p_i), normalize by ln(N). Applied to spreads (10 bins), trade volumes (10 bins), book depth proportions, trade sizes (5 bins).

**Tick entropy** (7): Shannon entropy of {up, down, neutral} trade direction counts within each window (1s/5s/10s/15s/30s/1m/15m). Range [0, ln(3)]. Low = trending, high = random.

**Volume-weighted tick entropy** (7): Same windows, but directions weighted by trade volume.

### 6. Context (9) — Hyperliquid market metadata

From the `activeAssetCtx` WebSocket channel. Funding rate and z-score, open interest (USD) with 5-minute absolute and percent change (60-sample lookback at ~5s updates), mark-index premium (bps), 24h volume and volume ratio, mark-oracle divergence.

`ctx_funding_rate`, `ctx_funding_zscore`, `ctx_open_interest`, `ctx_oi_change_{5m,pct_5m}`, `ctx_premium_bps`, `ctx_volume_24h`, `ctx_volume_ratio`, `ctx_mark_oracle_divergence`

### 7. Trend (15) — Persistence and mean-reversion detection

Refs: Jegadeesh & Titman (1993), Mandelbrot (1971). Momentum: linear regression slope over 60/300/600 ticks (~6s/30s/60s). R-squared: goodness of fit [0,1]. Monotonicity: fraction of ticks in majority direction, range [0.5, 1.0]. Hurst exponent via rescaled range: H < 0.5 = mean-reverting, H > 0.5 = trending. MA crossover: EMA(10) - EMA(50).

`trend_momentum_{60,300,600}`, `trend_momentum_r2_{60,300,600}`, `trend_monotonicity_{60,300,600}`, `trend_hurst_{300,600}`, `trend_ma_crossover`, `trend_ma_crossover_norm`, `trend_ema_{short,long}`

### 8. Illiquidity (12) — Market impact measures

Refs: Kyle (1985), Amihud (2002), Hasbrouck (2009).

- **Kyle's lambda**: Cov(dP, signed_vol) / Var(signed_vol) — price impact per unit volume
- **Amihud**: |r| / v * 10^6 — return per dollar of volume (ratio of sums, not mean of ratios)
- **Hasbrouck**: permanent price impact via OLS
- **Roll spread**: S = 2*sqrt(-Cov(dP_t, dP_{t-1})) — implied spread from autocovariance

Each computed over 100-trade and 500-trade windows, plus kyle_ratio, amihud_ratio (short/long), composite score, and trade count.

### 9. Toxicity (10) — Adverse selection and informed flow

Refs: Easley et al. (2012), Glosten & Milgrom (1985). **VPIN** (Volume-Synchronized Probability of Informed Trading): |V_buy - V_sell| / (V_buy + V_sell) over 10 and 50 volume buckets. Effective spread: 2 * mean(|trade_price - VWAP|) using VWAP as midpoint proxy (price units). Realized spread: mean(direction * (trade_price - price_{t+5}) * 2) with 5-trade lookahead. Adverse selection = effective - realized. Composite toxicity index: weighted combination of VPIN, adverse selection, and flow imbalance, normalized to [0,1].

`toxic_vpin_{10,50}`, `toxic_vpin_roc`, `toxic_adverse_selection`, `toxic_effective_spread`, `toxic_realized_spread`, `toxic_flow_imbalance`, `toxic_flow_imbalance_abs`, `toxic_index`, `toxic_trade_count`

### 10. Derived (15) — Cross-category composites

Interaction terms combining entropy, trend, volatility, illiquidity, and toxicity. Key inputs: tick_entropy = ent_tick_entropy_30s, monotonicity = trend_monotonicity_{60,300}, vol = vol_returns_1m * 100 clamped to [0,1], kyle = illiq_kyle_100 / 100 clamped to [0,1].

- **Trend strength**: sign(momentum) * (monotonicity - 0.5)*2 * (1 - tick_entropy), range [-1, 1]
- **Regime type score**: vol * (1 - 2*tick_entropy) — positive = breakout, negative = chaotic volatility
- **Regime indicator**: (mean_revert_score - trending_score - flow_factor), clamped to [-1, 1]
- **Toxicity-regime**: toxicity_index * tick_entropy (toxic in choppy markets)
- **Illiquidity-trend**: kyle * |momentum_60| * 1000 (informed directional flow)

### 11. Whale Flow (12) — Optional

Net position changes for large accounts across 1h/4h/24h windows. Normalized flow, momentum, intensity, rate of change, buy ratio, directional agreement, active count, total activity.

### 12. Liquidation Risk (13) — Optional

Liquidation volume mapped at +/-1%/2%/5%/10% price moves. Asymmetry = (risk_above - risk_below) / total. Intensity = total risk / OI. Nearest cluster distance (%), positions at risk count, largest position at risk.

### 13. Concentration (15) — Optional

Position crowding via inequality metrics: top-{5,10,20,50} share of OI, Herfindahl index (share_i^2), Gini coefficient, Theil index (share_i * ln(share_i / (1/N)), range [0, ln(N)]). Whale/retail ratios, concentration change over 1h, HHI rate of change, trend, position counts.

### 14. Regime Detection (20) — Optional

Accumulation/distribution cycle features at minute-level resolution:

- **Absorption**: volume / (|dPrice| + eps) at 1h/4h/24h + z-score. High absorption = volume absorbed without price move
- **Divergence**: actual dP - lambda*signed_volume at 1h/4h/24h + z-score. Negative = price lagging volume (accumulation)
- **Churn**: (buy_vol + sell_vol) / (|buy_vol - sell_vol| + eps) at 1h/4h/24h + z-score. High = two-sided activity
- **Range position**: (price - min) / (max - min) at 4h/24h/1w + 24h range width
- **Composite**: accumulation_score, distribution_score, clarity

### 15. GMM Classification (8) — Optional

Gaussian Mixture Model fitted on [kyle_lambda, vpin, absorption_zscore, hurst, whale_net_flow_1h]. Outputs: regime label in {0..4}, posterior probabilities for accumulation/markup/distribution/markdown/ranging, confidence = max(p_i), regime entropy = -p_i ln(p_i).

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
pytest scripts/tests/                   # Python tests (196 passing)
make test_pipeline                      # Pipeline state machine tests
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
IDLE --> BUILDING --> INGESTING --> COLLECTING --> ANALYZING --> DONE
```

State persisted in `data/pipeline_state.json` for resume on interrupt.

```bash
make pipeline_start     # Start pipeline
make pipeline_status    # Check state
make pipeline_resume    # Resume after interrupt
```

## Profiling Pipeline

Hierarchical regime discovery in `scripts/cluster_pipeline/`:

```bash
# Run full profiling on collected data
python scripts/analyze_clusters.py

# Individual steps
pytest scripts/tests/                    # 196 tests across 29 files
make scan_schema                         # Schema + vector coverage
make validate_data                       # 7-point data quality check
```

Modules: `loader` (Parquet I/O), `preprocess` (bar aggregation), `derivatives` (temporal + spectral), `reduction` (PCA), `cluster` (GMM/HDBSCAN), `hierarchy` (macro/micro discovery), `characterize` (centroid profiling), `transitions` (Markov analysis), `validate` (Q1-Q3 quality gates), `online` (drift-detecting classifier), `report` (automated markdown reports).

## ML Infrastructure

```bash
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=lightgbm
make score_data MODEL_PATH=./models/model.pkl
make backtest_ml_validate ML_PREDICTIONS=./predictions.parquet
make serve_best METRIC=sharpe_ratio      # REST API on port 8000
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

## Project Structure

```
rust/
  ing/          -- Ingestor library + binaries
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
  api/          -- REST/WebSocket API server (Axum, port 3000)

scripts/
  cluster_pipeline/       Regime profiling system (15 modules, 8300+ lines)
  backtest/               Walk-forward backtesting engine
  eamm/                   Equilibrium agent market making (prototype)
  tests/                  Python test suite (196 tests)
  pipeline_runner.py      Pipeline state machine
  model_serving.py        REST API for model predictions

config/
  ing.toml                Ingestor configuration
  pipeline.toml           Pipeline orchestration
  hypothesis_testing.toml Hypothesis test parameters

docs/
  PHASE1_ALGORITHM.md     Phase 1 mathematical spec + results
  specs/                  Profiling specs, requirements, task tracking

data/features/            Parquet output (YYYY-MM-DD/*.parquet)
```

## Docker

```bash
make docker_build       # Build images
make docker_up          # Run: redis (6379), ingestor, api (3000), alerts
make docker_down        # Stop
```

## Multi-Machine Setup

The ingestor runs on a separate machine (`su-35`). `make run` kills stale processes before starting. Data is written to `data/features/` at project root.

## References

1. Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71-100.

2. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.

3. Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.

4. Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *The Review of Financial Studies*, 25(5), 1457-1493.

5. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.

6. Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.

7. Zunino, L., Zanin, M., Tabak, B. M., Perez, D. G., & Rosso, O. A. (2009). Forbidden patterns, permutation entropy and stock market inefficiency. *Physica A*, 388(14), 2854-2864.

8. Shannon, C. E. (1948). A mathematical theory of communication. *The Bell System Technical Journal*, 27(3), 379-423.

9. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *The Journal of Business*, 53(1), 61-65.

10. Mandelbrot, B. B. (1971). When can price be arbitraged efficiently? A limit to the validity of the random walk and martingale models. *The Review of Economics and Statistics*, 53(3), 225-236.

11. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

12. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

13. Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the AMS*, 61(5), 458-471.

14. Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56.

15. Hasbrouck, J. (2009). Trading costs and returns for US equities: Estimating effective costs from daily data. *The Journal of Finance*, 64(3), 1445-1477.

16. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65-91.

17. Cont, R., Stoikov, S., & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.

18. Gatheral, J., & Oomen, R. (2010). Zero-intelligence realized variance estimation. *Finance and Stochastics*, 14(2), 249-283.

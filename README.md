# NAT

```
    _   _     _     _____
   | \ | |   / \   |_   _|
   |  \| |  / _ \    | |
   | |\  | / ___ \   | |
   |_| \_|/_/   \_\  |_|

   Autonomous Alpha Discovery
   Crypto Perpetual Futures Microstructure
```

NAT is an autonomous research agent that discovers tradeable alpha signals from
Hyperliquid perpetual futures microstructure. A Rust ingestor computes 191
order book features at 100ms resolution; an autonomous Python agent generates
hypotheses, tests them through a 3-gate replication protocol, and registers
validated signals — without human intervention.

## System Overview

```
                         +-----------------------+
                         |   Hyperliquid L2 WS   |
                         +-----------+-----------+
                                     |
                                     v
                    +--------------------------------+
                    |     Rust Ingestor (ing)         |
                    |  191 features x 100ms x 3 sym  |
                    +----------------+---------------+
                                     |
                              Parquet files
                     data/features/YYYY-MM-DD/*.parquet
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +----------------+    +------------------+    +----------------+
     |  nat spannung  |    |  nat profile     |    | nat spannung   |
     |  regime        |    |  scalp           |    | spectral       |
     |  (IC screen)   |    |  (walk-forward)  |    | (PSD, ACF)     |
     +--------+-------+    +--------+---------+    +--------+-------+
              |                      |                      |
              +----------------------+----------------------+
                                     |
                                     v
                    +--------------------------------+
                    |        NAT AGENT DAEMON        |
                    |                                |
                    |  5 Generators --> Priority Q    |
                    |  Runner: 3-gate replication     |
                    |  Registry: validated signals    |
                    +----------------+---------------+
                                     |
                              +------+------+
                              |             |
                              v             v
                        Registry       Graveyard
                    (trade-ready)    (failed claims)
```

## The Agent

The agent runs an autonomous research loop: generate hypotheses about
microstructure predictability, test each claim against data, replicate
across time and assets, and register only the signals that survive all gates.

```bash
nat agent start       # launch daemon (cycles every hour)
nat agent stop        # graceful shutdown (finishes current experiment)
nat agent once        # single cycle (for testing)
nat agent status      # phase, cycle count, registry size, generator stats
nat agent queue       # queued hypotheses by priority
nat agent registry    # validated signals
nat agent graveyard   # failed hypotheses with reasons
```

### What It Tests

Each hypothesis has the form:

> *Signal F, gated by regime variable G at threshold T, predicts h-second forward returns.*

**Signal features** — order book imbalance at varying depth.
Formulation: Cont, Stoikov & Talreja (2010):

```
                V_bid(d) - V_ask(d)
  OBI(d)  =   ----------------------
                V_bid(d) + V_ask(d)
```

where `d in {1, 5, 10}` is depth. The agent tests `imbalance_qty_l1`,
`imbalance_qty_l5`, `imbalance_qty_l10`, `imbalance_depth_weighted`,
`imbalance_notional_l5`, `flow_aggressor_ratio_5s`, and `toxic_flow_imbalance`.

**Regime gates** — conditioning variables that partition data into
microstructure regimes where signal quality varies:

| Category | Features | Theory |
|----------|----------|--------|
| Entropy | `ent_book_shape`, `ent_tick_{5s,30s,1m}`, `ent_permutation_returns_16`, `ent_spread_dispersion` | Shannon (1948), Bandt & Pompe (2002) — low entropy = concentrated/predictable regime |
| Illiquidity | `illiq_kyle_100`, `illiq_composite`, `illiq_amihud_100` | Kyle (1985), Amihud (2002) — high lambda = thin market, signals have more impact |
| Toxicity | `toxic_vpin_50`, `toxic_adverse_selection`, `toxic_index` | Easley, Lopez de Prado & O'Hara (2012) — VPIN detects informed trading |
| Volatility | `vol_returns_{1m,5m}`, `vol_ratio_short_long` | Parkinson (1980) — vol regime determines mean-reversion vs momentum |
| Regime | `derived_regime_type_score`, `derived_regime_confidence` | Hamilton (1989) — HMM state posterior as gating variable |

**Search space**: 7 signals x 17 gates x 4 thresholds x 2 directions = 952 hypotheses.
Thresholds are unconditional quintile breakpoints: `{P20, P40, P60, P80}`.

### Three-Gate Replication Protocol

Every hypothesis must independently pass three gates before registration.
This controls the false discovery rate under multiple testing
(Harvey, Liu & Zhu, 2016).

```
DISCOVERY ──> TEMPORAL REPLICATION ──> SYMBOL REPLICATION ──> REGISTER
    |                 |                        |
    v                 v                        v
GRAVEYARD         GRAVEYARD                GRAVEYARD
```

**Gate 1 — Discovery.** Run `nat spannung regime` + `nat profile scalp --forward-test`
on BTC with latest data. Pass condition: Spearman rank IC >= 0.10.

```
                  6 * sum(d_i^2)
  IC  =  1  -  -----------------------       d_i = rank(signal_i) - rank(return_i)
                  n * (n^2 - 1)
```

**Gate 2 — Temporal replication.** Re-run on 2 other dates from the manifest.
Pass condition: IC gate holds on >= 1 additional date. A signal that works on
only one date is a statistical artifact (White, 2000).

**Gate 3 — Symbol replication.** Re-run on ETH and SOL.
Pass condition: IC gate holds on >= 1 other symbol. Cross-asset replication is
the strongest evidence of a structural microstructure effect vs. asset-specific
overfitting (Gueant, Lehalle & Fernandez-Tapia, 2012).

### Five Hypothesis Generators

| Generator | Schedule | What it does |
|-----------|----------|-------------|
| **Systematic** | Nightly | Exhaustive (feature x gate x threshold) search. Priority boost for `ent_book_shape` (empirically strongest gate). |
| **Spectral** | Daily | Monitors PSD slope and OU half-life against baselines. Flags anomalies: slope outside [-2.2, -1.5] (Mandelbrot & Van Ness, 1968), half-life outside [2, 15]s. |
| **Regime** | Daily | Tests whether IC improves after HMM state transitions (Rabiner, 1989). |
| **Cross-Asset** | Weekly | Tests lead-lag at the 68s coherence frequency: `C_xy(f) = |S_xy(f)|^2 / (S_xx * S_yy)` (Priestley, 1981). |
| **Recycler** | Weekly | Re-examines graveyard entries when failure conditions change (new data, new dates, complementary signals). |

Generator budget allocation uses a Beta-prior multi-armed bandit
(Thompson, 1933):

```
  weight(g) = (successes + 1) / (attempts + 2)       # E[Beta(a, b)]
```

This steers generation toward productive generators without abandoning
exploration.

## Data Ingestion Layer

The Rust ingestor (`ing`) subscribes to Hyperliquid L2 WebSocket and computes
191 features at 100ms resolution for BTC, ETH, and SOL.

```
Hyperliquid WebSocket --> OrderBook + TradeBuffer + MarketContext
    --> FeatureComputer (191 features, 15 categories)
    --> Parquet files (data/features/YYYY-MM-DD/*.parquet, rotated hourly)
```

Each symbol runs in its own tokio task. The `tokio::select! { biased; }` loop
prioritizes WebSocket messages over emission ticks to prevent data loss
under load.

### Feature Vector (191 dimensions)

| # | Category | Count | Key features | Reference |
|---|----------|-------|-------------|-----------|
| 1 | Raw | 10 | midprice, spread, microprice, depth | Gatheral & Oomen (2010) |
| 2 | Imbalance | 8 | `OBI(d)` at L1/L5/L10, pressure scores | Cont, Stoikov & Talreja (2010) |
| 3 | Flow | 12 | Trade count/volume at 1s/5s/30s, aggressor ratio, VWAP dev | — |
| 4 | Volatility | 8 | Realized vol (1m/5m), Parkinson, vol ratio | Parkinson (1980) |
| 5 | Entropy | 24 | Permutation entropy (m=3), tick entropy, book shape | Bandt & Pompe (2002) |
| 6 | Context | 9 | Funding rate, OI, premium, volume ratio | — |
| 7 | Trend | 15 | Momentum, R2, monotonicity, Hurst, MA crossover | Mandelbrot (1971) |
| 8 | Illiquidity | 12 | Kyle's lambda, Amihud ratio, Roll spread, Hasbrouck | Kyle (1985), Amihud (2002) |
| 9 | Toxicity | 10 | VPIN (10/50), adverse selection, effective/realized spread | Easley et al. (2012) |
| 10 | Derived | 15 | Trend strength, regime score, toxicity-regime interaction | — |
| 11 | Whale Flow | 12 | Net flow, momentum, intensity (1h/4h/24h) | Optional |
| 12 | Liquidation | 13 | Risk at +/-1%/2%/5%/10%, asymmetry, intensity | Optional |
| 13 | Concentration | 15 | Herfindahl, Gini, Theil, top-K share | Optional |
| 14 | Regime Detection | 20 | Absorption, divergence, churn, range position | Optional |
| 15 | GMM Classification | 8 | State posteriors, confidence, regime entropy | Optional |

123 base features are always computed. 68 optional features are NaN-padded when
absent. See [`FEATURES.md`](FEATURES.md) for the full manifest with formulas and
Parquet column names.

## NAT Analysis Tools

The agent orchestrates these tools autonomously. They can also be used
interactively for manual research.

| Command | Function | Output |
|---------|----------|--------|
| `nat spannung` | Grid search: feature x horizon IC screen | `reports/spannung/spannung_{sym}.json` |
| `nat spannung regime` | Quintile regime screener: gate x threshold x direction | `reports/spannung/regime_screen_{sym}.json` |
| `nat spannung spectral` | PSD, ACF, coherence, band-decomposed IC | `reports/spannung/spectral_{sym}.json` |
| `nat spannung backtest` | Cost-aware backtest with regime gating | `reports/spannung/backtest_{sym}.json` |
| `nat profile scalp` | Feature profiler with walk-forward validation | `reports/profiler/profile_{sym}_{tf}.json` |
| `nat cluster hmm-fit` | Gaussian HMM fitting (Baum-Welch EM) | `reports/hmm_fit.json` |
| `nat validate skeptical` | 20+ statistical tests (FDR, bootstrap, permutation) | `reports/skeptical_validation/` |

### Key Findings (Spannung Research Arc, Phases A-F)

| Finding | Evidence | Implication |
|---------|----------|-------------|
| OBI predicts 5s returns | IC = 0.19, 100% sign consistency across 5 walk-forward folds | Structural signal, not overfit |
| `ent_book_shape` is #1 regime gate | Independently #1 on BTC, ETH, and SOL | Universal microstructure property |
| Signal replicates cross-symbol | KEEP verdict on all 3 symbols, liquidity-ordered IC (SOL > ETH > BTC) | Genuine limit-order-book effect |
| Brown noise universality | PSD slope ~ -1.85 on all 3 symbols | Fractional Brownian motion microstructure |
| 68s coherence dominance | Single peak at 0.015 Hz carries all cross-signal IC | Natural market-making cycle |
| OU half-life orders by liquidity | BTC 7.3s > ETH 5.3s > SOL 3.3s | Informs per-symbol refresh rates |
| `_last` features replicate, `_mean`/`_std` don't | Instantaneous = KEEP, aggregated = DROP | Time-domain manifestation of spectral finding |

## Build & Run

```bash
# Rust ingestor
make build              # debug build
make release            # release build (LTO, stripped)
make run                # build + start ingestor
make run_and_serve      # ingestor + dashboard on :8080

# Agent
nat agent start         # launch autonomous research daemon
nat agent once          # single research cycle
nat agent status        # current state

# Manual research
nat spannung --symbol BTC               # signal grid search
nat spannung regime --symbol BTC        # regime screener
nat spannung spectral --symbol BTC      # spectral analysis
nat profile scalp --symbol BTC --forward-test   # walk-forward validation
```

## Testing

```bash
make test                               # Rust unit tests
cd rust && cargo test -- test_name      # single test
pytest scripts/tests/                   # Python tests
make validate                           # live API validation
make test_pipeline                      # pipeline state machine
```

## Configuration

| File | Purpose |
|------|---------|
| `config/ing.toml` | Ingestor: WebSocket URL, symbols, emission interval, output format |
| `config/agent.toml` | Agent: cycle interval, experiment budget, gate thresholds, promotion criteria |
| `config/pipeline.toml` | Pipeline orchestration: ingestion duration, analysis thresholds |

Environment: `RUST_LOG`, `REDIS_URL`, `ING_DASHBOARD_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Project Structure

```
rust/
  ing/src/
    main.rs               tokio::select! biased loop
    ws/                   Hyperliquid WebSocket client
    state/                OrderBook, TradeBuffer, MarketContext
    features/             15 feature modules (191 features)
    output/               Parquet writer (ArrowWriter, hourly rotation)
    dashboard/            Real-time monitoring (Axum WebSocket)
  api/                    REST/WS API server (Axum, port 3000)

scripts/
  agent/                  Autonomous research agent
    daemon.py               Main loop: manifest -> generate -> execute -> monitor
    hypothesis.py           Hypothesis, RegisteredSignal, GeneratorStats dataclasses
    queue.py                JSON-backed priority queue with dedup
    manifest.py             Data availability scanner
    runner.py               3-gate experiment executor (state machine)
    generators/             5 hypothesis generators
      systematic.py           Exhaustive feature x gate x threshold search
      spectral.py             PSD/OU anomaly detector
      regime.py               HMM transition detector
      cross_asset.py          Lead-lag prober (BTC/ETH/SOL)
      recycler.py             Graveyard re-evaluator
  spannung_regime_screener.py   Quintile regime IC screening
  spannung_spectral.py          Frequency-domain analysis
  scalping_profiler.py          Walk-forward feature profiler
  cluster_pipeline/             Unsupervised regime discovery (15 modules)
  backtest/                     Walk-forward backtesting engine
  eamm/                         Entropy-adaptive market making (prototype)
  pipeline_runner.py            Pipeline state machine

config/
  ing.toml                Ingestor configuration
  agent.toml              Agent daemon configuration
  pipeline.toml           Pipeline orchestration

data/
  features/               Parquet output (YYYY-MM-DD/*.parquet)
  agent/                  Agent state, hypotheses, registry, manifest
```

## Agent Development Network

The agent is designed as a composable network of generators, executors,
and validators that can be extended independently.

### Adding a New Generator

Create `scripts/agent/generators/my_generator.py`:

```python
def generate(manifest, queue, stats=None) -> list[Hypothesis]:
    """Emit hypotheses based on some signal or anomaly."""
    # 1. Inspect manifest for data availability
    # 2. Inspect queue to avoid duplicate claims
    # 3. Create Hypothesis objects with test_protocol (nat commands)
    # 4. Return list — daemon pushes to priority queue
```

Register in `config/agent.toml`:
```toml
generators_enabled = ["systematic", "spectral", "regime", "cross_asset", "recycler", "my_generator"]
```

Add lazy import in `daemon.py:_get_generator()`.

### Adding a New Gate Check

Add to `runner.py`:
```python
def check_my_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Return (passed, message)."""
```

Wire into `ExperimentRunner._check_gates()`.

### Agent State Machine

```
Per-cycle:   MANIFEST -> GENERATE -> EXECUTE (budget: 10 or 90min) -> MONITOR -> SLEEP

Per-hypothesis:
  SETUP -> DISCOVERY -> REPLICATE_TEMPORAL -> REPLICATE_SYMBOL -> REGISTER
    |          |               |                     |
    v          v               v                     v
  ABORT    GRAVEYARD       GRAVEYARD             GRAVEYARD
```

State persists to `data/agent/agent_state.json`. The daemon handles SIGTERM
gracefully (finishes current experiment before stopping).

### Signal Lifecycle

```
Hypothesis (queued) -> Discovery (IC check) -> Temporal replication (2+ dates)
  -> Symbol replication (ETH + SOL) -> Registry (validated)
  -> Paper trading -> Live (with human approval)
```

Registered signals are promoted through: `validated -> paper -> live -> retired`.
Paper-to-live promotion requires: 7-day Sharpe > 1.5, realized/predicted IC > 0.8,
max drawdown < 2%.

## Multi-Machine Setup

The Rust ingestor runs on a separate machine (`su-35`). The agent daemon
runs on the research machine, reading Parquet files from `data/features/`.
`make run` kills stale processes before starting.

## Docker

```bash
make docker_build       # build images
make docker_up          # redis (6379), ingestor, api (3000), alerts
make docker_down        # stop
```

## References

1. Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*, 5(1), 31-56.
2. Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
3. Bandt, C. & Pompe, B. (2002). Permutation entropy. *Physical Review Letters*, 88(17), 174102.
4. Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
5. Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457-1493.
6. Gatheral, J. & Oomen, R. (2010). Zero-intelligence realized variance estimation. *Finance and Stochastics*, 14(2), 249-283.
7. Glosten, L.R. & Milgrom, P.R. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71-100.
8. Gueant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). Dealing with the inventory risk. *Mathematics and Financial Economics*, 4(7), 477-507.
9. Hamilton, J.D. (1989). A new approach to nonstationary time series. *Econometrica*, 57(2), 357-384.
10. Harvey, C.R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
11. Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
12. Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions. *SIAM Review*, 10(4), 422-437.
13. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
14. Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
15. Rabiner, L.R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286.
16. Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
17. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3-4), 285-294.
18. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

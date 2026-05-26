# NAT

```
                                                                                
     ███╗   ██╗ █████╗ ████████╗                                                
     ████╗  ██║██╔══██╗╚══██╔══╝                                                
     ██╔██╗ ██║███████║   ██║                                                   
     ██║╚██╗██║██╔══██║   ██║                                                   
     ██║ ╚████║██║  ██║   ██║                                                   
     ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝                                                   
                                                                                
     ╔══════════════════════════════════════════════════════════════╗            
     ║  Autonomous Alpha Discovery for Crypto Perpetual Futures    ║            
     ║  ─────────────────────────────────────────────────────────── ║            
     ║  209 features · 100ms resolution · 27 algorithms            ║            
     ║  4 research agents · 5-gate replication · FDR control       ║            
     ║  From order book to deployment — zero human intervention    ║            
     ╚══════════════════════════════════════════════════════════════╝            
                                                                                
          ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             
          │  INGEST │───>│ DISCOVER│───>│REPLICATE│───>│ DEPLOY  │             
          │  (Rust) │    │ (Agent) │    │ (5-Gate)│    │ (Paper) │             
          └─────────┘    └─────────┘    └─────────┘    └─────────┘             
              ▲                                             │                   
              └─────────────── feedback loop ───────────────┘                   
```

NAT is a fully autonomous quantitative research platform that discovers tradeable alpha signals from [Hyperliquid](https://hyperliquid.xyz) perpetual futures microstructure. A Rust ingestor computes 209 order book features at 100ms tick resolution; four autonomous Python agents generate hypotheses, test them through a 5-gate replication protocol with FDR control, and register validated signals — without human intervention.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Ingestion Layer (Rust)](#data-ingestion-layer-rust)
- [Feature Vector (209 Dimensions)](#feature-vector-209-dimensions)
- [Microstructure Algorithm Library (27 Algorithms)](#microstructure-algorithm-library-27-algorithms)
- [Autonomous Research Agents](#autonomous-research-agents)
- [Alpha Pipeline](#alpha-pipeline)
- [Information-Theoretic Engine](#information-theoretic-engine)
- [Paper Trading & OOS Validation](#paper-trading--oos-validation)
- [Analysis & Profiling Tools](#analysis--profiling-tools)
- [Execution Layer](#execution-layer)
- [Web Dashboard & API](#web-dashboard--api)
- [Polymarket Integration](#polymarket-integration)
- [Entropy-Adaptive Market Making (EAMM)](#entropy-adaptive-market-making-eamm)
- [The `nat` CLI (163+ Commands)](#the-nat-cli-163-commands)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker](#docker)
- [Multi-Machine Setup](#multi-machine-setup)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [References](#references)

---

## Quick Start

```bash
# 1. Build the Rust ingestor
make release                          # LTO-optimized release build

# 2. Start data ingestion (production: tmux + watchdog + dashboard)
nat start                             # launches ingestor with auto-restart
nat log                               # tail live output
nat status                            # health check (JSON)
nat dashboard                         # web dashboard at :8050

# 3. Launch autonomous research agents
nat agent start                       # microstructure agent (tick-level)
nat mf_agent start                    # medium-frequency agent (1min-1h)
nat macro_agent start                 # macro agent (1h-24h)
nat meta_agent start                  # meta orchestrator (cross-agent)

# 4. Monitor research progress
nat agent status                      # phase, cycle count, registry size
nat agent dashboard                   # IC heatmap on :8060
nat agent registry                    # validated signals
nat agent report                      # full summary

# 5. Run paper trading (after 30 days of data)
nat oos30                             # all 5 winning algorithms, walk-forward

# 6. Manual research
nat spannung --symbol BTC             # signal IC grid search
nat spannung regime --symbol BTC      # regime screener
nat profile scalp --symbol BTC        # walk-forward profiling
```

### Makefile Quickref

```bash
make build              # debug build (faster iteration)
make release            # release build (LTO, stripped)
make run                # foreground ingestor (dev mode)
make run_and_serve      # ingestor + dashboard on :8080
make test               # Rust unit tests
make test_agent         # 350 agent tests (unit + integration)
pytest scripts/tests/   # Python test suite (68 test files)
make validate           # live API validation (4 binaries)
```

---

## Architecture

```
                              ┌──────────────────────────┐
                              │   Hyperliquid L2 WebSocket│
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │    Rust Ingestor (ing)    │
                              │  209 features × 100ms    │
                              │  × 3 symbols (BTC/ETH/SOL)│
                              │  tokio::select! biased   │
                              └────────────┬─────────────┘
                                           │
                                    Parquet files
                          data/features/YYYY-MM-DD/*.parquet
                                           │
           ┌───────────────┬───────────────┼───────────────┬───────────────┐
           │               │               │               │               │
           ▼               ▼               ▼               ▼               ▼
   ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │   NAT Agent  │ │  IT Engine  │ │  Alpha      │ │ Discovery│ │  Analysis    │
   │   Daemons    │ │  KSG MI     │ │  Pipeline   │ │ Orchstr  │ │  Tools       │
   │              │ │  CMI, TE    │ │  9-step     │ │  sweep   │ │              │
   │ 4 agents     │ │  greedy     │ │  w/ gates   │ │  train   │ │ spannung     │
   │ 6 generators │ │  selection  │ │  to deploy  │ │  backtest│ │ profiler     │
   │ 5-gate proto │ │             │ │             │ │          │ │ cluster      │
   └──────┬───────┘ └──────┬──────┘ └──────┬──────┘ └────┬─────┘ └──────┬───────┘
          │                │               │              │              │
          ▼                ▼               ▼              ▼              ▼
   ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │  Registry    │ │  Feature    │ │  Paper      │ │  Reports │ │  JSON/HTML   │
   │  (validated) │ │  Rankings   │ │  Trading    │ │  & Logs  │ │  Reports     │
   │  Graveyard   │ │  MI Scores  │ │  Execution  │ │          │ │              │
   │  (failed)    │ │             │ │  Bridge     │ │          │ │              │
   └──────────────┘ └─────────────┘ └─────────────┘ └──────────┘ └──────────────┘
          │                                │
          ▼                                ▼
   ┌────────────────────────────────────────────────┐
   │         Web Dashboard (Next.js)                │
   │  Hypothesis Explorer · IC Heatmap · Graveyard  │
   │  Research Network · Signal Table · Gate Funnel  │
   │         REST API (Axum, port 3000)              │
   │  /api/research/hypotheses · /cycles · /signals  │
   └────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Data ingestion | **Rust** (tokio, Arrow, Parquet, Axum) |
| Feature computation | **Rust** (209 features, 15 categories, 100ms emission) |
| Research agents | **Python** (autonomous daemons, SQLite state, FDR control) |
| ML & backtesting | **Python** (LightGBM, scikit-learn, pandas, numpy) |
| API server | **Rust** (Axum REST/WebSocket, port 3000) |
| Web dashboard | **Next.js** (TypeScript, Tailwind, React) |
| Agent dashboard | **Python** (stdlib HTTP server, port 8060) |
| Messaging | **Redis** (Pub/Sub + Streams, feature distribution, alerts) |
| State persistence | **SQLite** (hypothesis queue, agent state) + **JSON** (research output) |
| Alerting | **Telegram** (bot API for alerts) |
| Process management | **tmux** (sessions with watchdog auto-restart via cron) |
| Containerization | **Docker** (docker-compose: redis, ingestor, api, alerts, web) |

---

## Data Ingestion Layer (Rust)

The Rust ingestor (`rust/ing/`) subscribes to Hyperliquid L2 WebSocket and computes 209 features at 100ms resolution for BTC, ETH, and SOL.

```
Hyperliquid WebSocket ──▶ OrderBook + TradeBuffer + MarketContext
    ──▶ FeatureComputer (209 features, 15 categories)
    ──▶ Parquet files (data/features/YYYY-MM-DD/*.parquet, rotated hourly)
```

**Key design decisions:**

- Each symbol runs in its own `tokio` task
- `tokio::select! { biased; }` prioritizes WebSocket messages over emission ticks (prevents data loss under load)
- `ArrowWriter` with explicit `flush()` after each batch (prevents 0-byte files until close)
- Hourly file rotation with 10,000-row buffer flush (~5.5 min at 30 rows/sec)

### Validation Binaries

```bash
nat validate api                # Hyperliquid API connectivity
nat validate positions          # position tracking accuracy
nat validate whales             # whale identification pipeline
nat validate entropy            # entropy feature computation
nat run show BTC 10             # real-time feature display
```

---

## Feature Vector (209 Dimensions)

| # | Category | Count | Prefix | Key Features | Reference |
|---|----------|-------|--------|-------------|-----------|
| 1 | **Raw** | 10 | `raw_` | midprice, spread, microprice, depth L5/L10 | Gatheral & Oomen (2010) |
| 2 | **Imbalance** | 8 | `imbalance_` | OBI at L1/L5/L10, pressure scores | Cont, Stoikov & Talreja (2010) |
| 3 | **Flow** | 12 | `flow_` | trade count/volume 1s/5s/30s, aggressor ratio, VWAP deviation | — |
| 4 | **Volatility** | 9 | `vol_` | realized vol (1m/5m), Parkinson, Garman-Klass, vol ratio | Parkinson (1980), Garman & Klass (1980) |
| 5 | **Entropy** | 24 | `ent_` | permutation entropy (m=3), tick entropy, book shape, spread dispersion | Bandt & Pompe (2002), Shannon (1948) |
| 6 | **Context** | 9 | `ctx_` | funding rate, OI, premium, volume ratio | — |
| 7 | **Trend** | 15 | `trend_` | momentum, R², monotonicity, Hurst exponent, MA crossover | Mandelbrot (1971), Jegadeesh & Titman (1993) |
| 8 | **Illiquidity** | 12 | `illiq_` | Kyle's lambda, Amihud ratio, Roll spread, Hasbrouck | Kyle (1985), Amihud (2002) |
| 9 | **Toxicity** | 10 | `toxic_` | VPIN (10/50), adverse selection, effective/realized spread | Easley et al. (2012) |
| 10 | **Derived** | 15 | `derived_` | trend strength, regime score, toxicity-regime interaction | — |
| 11 | **Whale Flow** | 12 | `whale_` | net flow, momentum, intensity (1h/4h/24h) | Optional |
| 12 | **Liquidation** | 13 | `liquidation_` | risk at ±1%/2%/5%/10%, asymmetry, intensity | Optional |
| 13 | **Concentration** | 15 | `top`/`conc_` | Herfindahl, Gini, Theil, top-K share | Optional |
| 14 | **Regime** | 20 | `regime_` | absorption, divergence, churn, range position | Optional |
| 15 | **GMM** | 8 | `prob_` | 5-state posteriors, confidence, regime entropy | Optional |

138 base features always computed. 71 optional features NaN-padded when absent. Full manifest with formulas: [`FEATURES.md`](FEATURES.md).

### Feature Engineering Contract

```
Features::to_vec()    → always returns exactly 209 elements
Features::names_all() → matching column names (Parquet schema source)
Features::count_all() → 209
```

When adding a new feature category:
1. Create struct with `count()`, `names()`, `to_vec()`
2. Add to `Features` in `features/mod.rs`
3. Add to `to_vec()`, `names_all()`, `count_all()` (NaN padding for optional)
4. Schema auto-updates via `create_schema()` in `output/schema.rs`

---

## Microstructure Algorithm Library (27 Algorithms)

NAT includes 27 configurable microstructure algorithms that compute derived signals from the 209-dimensional base feature vector. Each algorithm implements the `MicrostructureAlgorithm` interface.

```bash
nat algorithm list                              # all algorithms and features
nat algorithm evaluate --all                    # IC/drift on all
nat algorithm evaluate --algorithm hawkes_intensity --symbol BTC
nat algorithm config                            # TOML configuration
nat backtest algorithm --algorithm weighted_ofi --symbol BTC
```

### Algorithm Catalog

| # | Algorithm | Method | Reference |
|---|-----------|--------|-----------|
| 1 | `kalman_imbalance` | OU Kalman filter on L1 imbalance | — |
| 2 | `regime_gated` | Entropy percentile gating | Bandt & Pompe (2002) |
| 3 | `multi_level_imb` | Weighted L1/L5/L10 composite | — |
| 4 | `weighted_ofi` | Depth-decay weighted OFI | Cont, Kukanov & Stoikov (2014) |
| 5 | `trade_through` | Queue depletion probability | Cont & de Larrard (2013) |
| 6 | `propagator` | Transient impact, power-law kernel | Bouchaud et al. (2004) |
| 7 | `hawkes_intensity` | Self-exciting trade arrival | Bacry, Mastromatteo & Muzy (2015) |
| 8 | `jump_detector` | Lee-Mykland nonparametric jump test | Lee & Mykland (2008) |
| 9 | `bipower_jump` | BV jump decomposition | Barndorff-Nielsen & Shephard (2004) |
| 10 | `vpin_regime` | VPIN-triggered regime switch | Easley, López de Prado & O'Hara (2012) |
| 11 | `spread_decomp` | Adverse selection decomposition | Hendershott, Jones & Menkveld (2011) |
| 12 | `entropy_momentum` | Entropy-gated momentum | Novel |
| 13 | `surprise_signal` | Entropy regime transition detection | Novel |
| 14 | `funding_reversion` | Funding rate mean-reversion | Crypto-specific |
| 15 | `oi_divergence` | Open interest vs price divergence | — |
| 16 | `switching_ou` | Two-regime OU, Bayesian filtering | Elliott et al. (2005), Hamilton (1989) |
| 17 | `optimal_entry` | SPRT on Kalman innovation | Wald (1947), Shiryaev (1978) |
| 18 | `online_ridge` | Online ridge regression meta-algorithm | Hoerl & Kennard (1970) |
| 19 | `convolver` | Kernel convolution feature discovery | Novel |
| 20 | `cascade_probability` | Liquidation cascade prediction | — |
| 21 | `spectral` | Spectral momentum extraction | — |
| 22-27 | Various | Additional signal algorithms | See `scripts/algorithms/` |

### Algorithm Evaluation Results (Experiment Report 2)

Walk-forward paper trading across BTC/ETH/SOL at 100min horizon, 5min bars, 1.61 bps fees:

| Tier | Algorithm | Total P&L (bps) | Best Sharpe |
|------|-----------|-----------------|-------------|
| **1 — Deployable** | `jump_detector` | **+23,199** | 6.2 (ETH/SOL) |
| **1 — Deployable** | `funding_reversion` | **+14,459** | 6.0 (ETH) |
| **1 — Deployable** | `optimal_entry` | **+13,679** | 5.2 (ETH) |
| **2 — Symbol-specific** | `surprise_signal` | +3,505 | 6.7 (SOL) |
| **Baseline** | `3f_liquidity` | — | 9.2 (BTC) |

The winning algorithms are complementary: 3f dominates BTC, jump/optimal dominate ETH/SOL.

---

## Autonomous Research Agents

NAT runs four autonomous research agents that continuously generate, test, and validate alpha hypotheses across different timeframes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MetaAgent (Orchestrator)                         │
│   Cross-agent budget allocation · Correlation dedup · Risk parity      │
├─────────────────────┬───────────────────┬───────────────────────────────┤
│  MicrostructureAgent│ MediumFreqAgent   │ MacroAgent                    │
│  Tick-level (1-10s) │ 1min-1h signals   │ 1h-24h signals               │
│  5-gate replication │ 4-gate replication│ 4-gate replication            │
│  6 generators       │ MF generators     │ Macro generators              │
└─────────────────────┴───────────────────┴───────────────────────────────┘
```

### Agent Commands

```bash
# Microstructure agent
nat agent start                 # launch daemon (cycles hourly)
nat agent stop                  # graceful shutdown
nat agent once                  # single cycle (testing)
nat agent status                # phase, cycle count, registry
nat agent queue                 # queued hypotheses by priority
nat agent registry              # validated signals
nat agent graveyard             # failed hypotheses with reasons
nat agent report                # full summary
nat agent dashboard             # web UI on :8060

# Medium-frequency agent
nat mf_agent start / stop / once / status / queue / registry / report

# Macro agent
nat macro_agent start / stop / once / status / queue / registry / report

# Meta orchestrator
nat meta_agent start / stop / once / status / portfolio / correlation / budget / report
```

### Consolidated Daemon Architecture

All agents share a common base (`ResearchAgent` ABC in `scripts/agent/base.py`) that owns:
- Full cycle loop and state machine
- Generator dispatch (lazy import via `generator_module_prefix`)
- FDR control (Benjamini-Hochberg, q=0.05)
- Hypothesis chaining and promotion logic
- Structured research output emission

Each agent subclass is a thin ~80-110 LOC file overriding only config attributes and `create_runner()`.

### Agent State Machine

```
Per-cycle:
  MANIFEST ──▶ GENERATE ──▶ ADAPTIVE IC ──▶ EXECUTE (budget: 10 or 90min)
  ──▶ FDR control (BH q=0.05) ──▶ STRUCTURED OUTPUT
  ──▶ MONITOR (decay + promotion) ──▶ SLEEP

Per-hypothesis:
  SETUP ──▶ DISCOVERY (IC+dIC) ──▶ COST ──▶ TEMPORAL ──▶ SYMBOL ──▶ CORRELATION
    │            │                   │          │           │            │
    ▼            ▼                   ▼          ▼           ▼            ▼
  ABORT      GRAVEYARD           GRAVEYARD  GRAVEYARD  GRAVEYARD    GRAVEYARD
             (no_effect)         (cost_killed)         (no_repl)   (redundant)
```

### Five-Gate Replication Protocol

Every hypothesis must independently pass five gates before registration. This controls the false discovery rate under multiple testing (Harvey, Liu & Zhu, 2016).

```
DISCOVERY ──▶ COST ──▶ TEMPORAL ──▶ SYMBOL ──▶ CORRELATION ──▶ REGISTER
    │           │          │           │             │
    ▼           ▼          ▼           ▼             ▼
GRAVEYARD  GRAVEYARD  GRAVEYARD  GRAVEYARD      GRAVEYARD

                  + FDR control (BH, q=0.05) at end of each cycle
```

**Gate 1 — Discovery (IC + dIC).** Spearman rank IC >= adaptive threshold, dIC >= 0.05:
```
                6 × Σ(dᵢ²)
  IC  =  1 − ─────────────────       dᵢ = rank(signalᵢ) − rank(returnᵢ)
                n × (n² − 1)
```

**Gate 2 — Cost.** Gross return per trade >= 0.1 bps.

**Gate 3 — Temporal replication.** IC holds on >= 1 additional date (White, 2000).

**Gate 4 — Symbol replication.** IC holds on >= 1 other symbol (Gueant et al., 2012).

**Gate 5 — Correlation deduplication.** max ρ < 0.7 vs all registry signals.

### Adaptive IC Threshold

The acceptance threshold rises as the registry accumulates strong signals:

```
  min_ic(t) = max( floor, median{ ICᵢ : i ∈ R(t) } × 0.8 )
```

Floor = 0.10 (configurable). Marginal signals cannot enter a strong registry.

### IC Decay Monitoring

Signals are continuously monitored. Auto-retired if rolling IC drops below 50% of discovery IC for 14 consecutive cycles:

```
  if IC_rolling(s) < IC_discovery(s) × 0.5   for 14 consecutive cycles:
      status(s) ← retired, reason ← ic_decay
```

### Six Hypothesis Generators

| Generator | Schedule | Strategy | Reference |
|-----------|----------|----------|-----------|
| **Systematic** | Nightly | Exhaustive (feature × gate × threshold) search | — |
| **Spectral** | Daily | PSD slope / OU half-life anomaly detection | Mandelbrot & Van Ness (1968) |
| **Regime** | Daily | IC improvement after HMM state transitions | Rabiner (1989) |
| **Cross-Asset** | Weekly | Lead-lag at 68s coherence frequency | Priestley (1981) |
| **Recycler** | Weekly | Re-examine graveyard on new data/conditions | — |
| **IT Discovery** | Daily | MI/CMI/II-driven hypotheses from IT engine | Kraskov et al. (2004) |

Budget allocation uses a Beta-prior Thompson bandit (Thompson, 1933):
```
  weight(g) = (successes + 1) / (attempts + 2)       # E[Beta(a, b)]
```

### Signal Lifecycle

```
Hypothesis (queued) ──▶ Discovery (IC+dIC) ──▶ Cost (gross >= 0.1 bps)
  ──▶ Temporal replication (2+ dates) ──▶ Symbol replication (ETH + SOL)
  ──▶ Correlation dedup (ρ < 0.7) ──▶ FDR control (BH q=0.05)
  ──▶ Registry (validated) ──▶ Paper trading ──▶ Live (human approval)
  ──▶ Retired (IC decay > 14 days)
```

Paper-to-live promotion: 7-day Sharpe > 1.5, realized/predicted IC > 0.8, max DD < 2%.

### Structured Research Output

After each hypothesis, a JSON record is emitted to `data/research/hypotheses/{id}.json` with:
- Claim, generator, status, gate results (metric, threshold, p-value per gate)
- LaTeX math derivation
- Features, regime gate, timestamps

Cycle summaries to `data/research/cycles/{cycle_id}.json` with aggregate stats.

### Computation Cache

Deterministic commands cached under `SHA-256(canonical_args)` with 7-day TTL.
Measured: 85% hit rate, 56% cycle-time reduction.

---

## Alpha Pipeline

Nine-step alpha signal pipeline with quality gates (PASS/WEAK/FAIL) between each step.

```bash
nat alpha_pipeline_start               # launch full pipeline
nat alpha_pipeline_resume              # resume from last checkpoint
nat alpha_pipeline_status              # current step and gate verdicts
nat alpha_pipeline_gates               # gate thresholds
nat alpha_pipeline_step 3              # run specific step
```

### Pipeline Steps

```
SCREENING ──▶ COMBINING ──▶ SIZING ──▶ VALIDATING ──▶ REGIME
    │             │            │            │              │
    ▼             ▼            ▼            ▼              ▼
 (FDR)       (weights)    (position)   (walk-fwd)    (HMM gate)
    │
    ▼
MULTI_FREQ ──▶ PORTFOLIO ──▶ PAPER ──▶ DEPLOYING ──▶ DONE
    │              │            │            │
    ▼              ▼            ▼            ▼
 (freqs)      (assembly)   (simulate)   (live exec)
```

| Step | Module | Function |
|------|--------|----------|
| 1. Screening | `screener.py` | Feature screening with FDR control |
| 2. Combining | `combiner.py` | Signal combination and weighting |
| 3. Sizing | `position.py` | Position sizing (Kelly / risk parity) |
| 4. Validating | Walk-forward | Out-of-sample stability check |
| 5. Regime | `regime_filter.py` | HMM/entropy regime conditioning |
| 6. Multi-Freq | `multi_freq.py` | Multi-frequency signal integration |
| 7. Portfolio | `portfolio.py` | Portfolio assembly and allocation |
| 8. Paper | `paper_trader.py` | Live paper trading simulation |
| 9. Deploy | `deployer.py` | Live deployment executor |

Config: `config/alpha.toml` (gate thresholds G1-G8).

---

## Information-Theoretic Engine

The IT Engine (`scripts/it_engine/`) provides rigorous mutual information estimation, entropy conditioning, and cost-aware feature selection across all 209 features.

```bash
nat it_engine start --symbol BTC               # live mode (Redis pub/sub)
nat it_engine start --symbol BTC --offline     # offline mode (parquet)
nat it_engine start --symbol BTC --dry-run     # single cycle
nat it_engine status --symbol BTC              # MI rankings
nat it_engine stop --symbol BTC                # shutdown
```

### Core Estimators

| Estimator | Formula | Purpose |
|-----------|---------|---------|
| **KSG MI** | Kraskov-Stögbauer-Grassberger k-NN (k=5) | I(f; r) — mutual information |
| **Conditional MI** | I(f; r \| H) via KSG in joint/marginal spaces | Proper IT formulation of entropy gating |
| **Interaction Info** | II(f;r;H) = I(f;r\|H) − I(f;r) | +synergy (gating helps), −redundancy |
| **Linear TE** | TE = 0.5 × log(σ²_reduced / σ²_full) | Causal information flow |
| **Cost threshold** | I_min = −0.5 × log₂(1 − (fee/σ_r)²) | Minimum MI to overcome costs |

### Greedy Feature Selection

Forward stepwise selection by conditional MI gain:
1. **Start**: f* = argmax_f I(f; r_k) across all features and horizons
2. **Step**: f_next = argmax_f I(f; r_k | S) — largest conditional MI gain given selected set S
3. **Stop**: when marginal gain < I_min(k) or max features reached

The `it_discovery` generator feeds cost-viable features into the agent hypothesis queue.

---

## Paper Trading & OOS Validation

### Walk-Forward Paper Trading

```bash
# Individual algorithms
python scripts/alpha/paper_trader_generic.py --algorithms jump_detector optimal_entry funding_reversion --save

# 3-feature liquidity signal (BTC specialist)
python scripts/alpha/paper_trader.py --save

# Surprise signal (SOL specialist)
python scripts/alpha/paper_trader_surprise.py --save

# All 5 winning algorithms at once (via nat CLI)
nat oos30
```

**Configuration:**
- Bar resolution: 5min (300s)
- Horizon: 20 bars (100min)
- Training window: 3-day rolling
- Entry thresholds: P20/P80 percentile z-score
- Fee model: 1.61 bps round-trip (Binance VIP9 taker)
- Walk-forward: true out-of-sample

### OOS Validation Workflow

```bash
nat start                  # start ingestor on production machine
# ... wait 30 days for out-of-sample data collection ...
nat oos30                  # runs all 5 winning algos in 3 steps:
                           #   1. 3f liquidity (BTC specialist)
                           #   2. jump_detector + optimal_entry + funding_reversion
                           #   3. surprise_signal (SOL specialist)
```

### Discovery Orchestrator

Continuous daemon that sweeps (symbol, horizon) combinations for alpha signals:

```bash
nat discovery start        # launch continuous sweep daemon
nat discovery once         # single sweep cycle
nat discovery status       # current state
nat discovery stop         # shutdown
```

Cycle: DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING → ALPHA_PIPELINE → REPORTING → SLEEPING.

---

## Analysis & Profiling Tools

The agent orchestrates these tools autonomously. They can also be used interactively.

| Command | Function | Output |
|---------|----------|--------|
| `nat spannung --symbol BTC` | Feature × horizon IC grid search | `reports/spannung/spannung_{sym}.json` |
| `nat spannung regime --symbol BTC` | Quintile regime screener | `reports/spannung/regime_screen_{sym}.json` |
| `nat spannung spectral --symbol BTC` | PSD, ACF, coherence, band-decomposed IC | `reports/spannung/spectral_{sym}.json` |
| `nat spannung backtest --symbol BTC` | Cost-aware backtest with regime gating | `reports/spannung/backtest_{sym}.json` |
| `nat profile scalp --symbol BTC` | Walk-forward feature profiler | `reports/profiler/profile_{sym}.json` |
| `nat cluster hmm` | Gaussian HMM fitting (Baum-Welch EM) | `reports/hmm_fit.json` |
| `nat validate skeptical` | 20+ statistical tests (FDR, bootstrap, permutation) | `reports/skeptical_validation/` |
| `nat scan --symbol BTC` | Signal discovery scan | JSON report |
| `nat macro --symbol BTC` | Macro regime analysis | JSON report |
| `nat kalman analysis --symbol BTC` | Kalman filter analysis | `reports/kalman/` |

### Cluster Pipeline

Unsupervised regime discovery via feature vector clustering:

```bash
nat cluster analyze                    # K-means/hierarchical clustering
nat cluster gmm                        # GMM-based regime classification
nat cluster quality                    # silhouette, Davies-Bouldin, Calinski-Harabasz
nat cluster explore                    # interactive exploration
nat cluster hmm                        # HMM state fitting
```

Modules: loader, preprocess, cluster, reduce (PCA/UMAP/t-SNE), characterize, hierarchy, transitions, online streaming, visualization.

### Backtesting Engine

```bash
nat backtest --symbol BTC              # generic backtest
nat backtest algorithm --algorithm weighted_ofi --symbol BTC
nat backtest ml --symbol BTC           # ML prediction backtest
nat backtest funding                   # funding rate reversion
nat backtest list                      # list available experiments
```

Walk-forward validation with configurable cost models (taker/maker at various fee tiers).

---

## Execution Layer

### Signal Bridge

`scripts/execution/signal_bridge.py` — translates validated signals into executable orders:
- Dry-run mode (logging only) and live mode
- Position sizing with risk limits
- Integration with paper trading results

### Hyperliquid Client

`scripts/execution/hyperliquid_client.py` — trading API wrapper for Hyperliquid exchange:
- Order placement (limit, market, stop)
- Position management
- Account state queries

---

## Web Dashboard & API

### Next.js Frontend (port 3001)

```bash
cd web && npm run dev                  # development server
```

| Page | URL | Description |
|------|-----|-------------|
| Homepage | `/` | System overview |
| Hypothesis Explorer | `/explorer` | Browse all hypotheses, filter by status/agent/generator |
| Hypothesis Detail | `/explorer/{id}` | Gate results, math derivation, feature data |
| IC Heatmap | `/heatmap` | Feature × horizon IC matrix visualization |
| Graveyard | `/graveyard` | Failed hypotheses with failure analysis |
| Research Network | `/network` | Graph visualization of hypothesis relationships |
| Signal Table | `/signals` | Active validated signals |
| Math Viewer | `/math` | LaTeX math rendering |

Components: hypothesis table, IC bar chart, cycle ring, gate funnel, agent cards, generator bars, weight treemap, signal table, filter bar, failure pie, near-miss analysis.

### Rust REST API (port 3000)

```bash
nat api start                          # launch API server
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research/hypotheses` | GET | Paginated list, filter by `?agent=`, `?generator=`, `?status=` |
| `/api/research/hypotheses/:id` | GET | Full detail (gates, math, thresholds) |
| `/api/research/cycles` | GET | Cycle summaries, filter by `?agent=` |
| `/api/research/signals` | GET | Registered signals only |
| `/api/research/stats` | GET | Aggregate counts by status, agent, generator |
| `/api/research/heatmap` | GET | Feature × horizon IC matrix |
| `/health` | GET | Health check |

### Agent Dashboard (port 8060)

```bash
nat agent dashboard                    # stdlib HTTP, dark theme, 10s auto-refresh
```

Panels: agent status, registry table, (signal × gate) IC heatmap, graveyard, queue, generator stats, cache statistics.

---

## Polymarket Integration

`scripts/polymarket/` — prediction market analysis module:

| Module | Purpose |
|--------|---------|
| `client.py` | Polymarket API client |
| `market_scanner.py` | Market opportunity scanner |
| `probability_model.py` | Prediction modeling |
| `edge_detector.py` | Edge detection in prediction markets |
| `backtest.py` | Strategy backtesting |

---

## Entropy-Adaptive Market Making (EAMM)

`scripts/eamm/` — market making strategy that adapts quotes based on entropy regime:

```bash
nat eamm run                           # simulation
nat eamm regime                        # regime analysis
nat eamm backtest                      # backtesting
```

| Module | Purpose |
|--------|---------|
| `simulator.py` | MM simulation engine |
| `features.py` | Feature engineering for MM |
| `labels.py` | Label generation (realized vs potential P&L) |
| `train.py` | Model training loop |
| `evaluate.py` | Evaluation metrics |
| `backtest.py` | Backtesting harness |
| `regime_analysis.py` | Regime-conditioned analysis |
| `export.py` | Model export |

---

## The `nat` CLI (163+ Commands)

`nat` is a unified research terminal (~4,400 lines) that replaces the Makefile for production use. All commands follow the pattern `nat <command> [subcommand] [flags]`.

### Command Reference

#### Ingestor Control
```bash
nat start                  # production launch (tmux + watchdog + dashboard + logging)
nat stop                   # graceful shutdown
nat status                 # health check (JSON-compatible)
nat log                    # tail latest log
nat dashboard              # start/show dashboard at :8050
nat health                 # system health overview
```

#### Research Agents
```bash
nat agent {start,stop,once,status,queue,registry,graveyard,report,dashboard}
nat mf_agent {start,stop,once,status,queue,registry,graveyard,report}
nat macro_agent {start,stop,once,status,queue,registry,graveyard,report}
nat meta_agent {start,stop,once,status,portfolio,correlation,budget,report}
```

#### Analysis
```bash
nat spannung [--symbol SYM]              # IC grid search
nat spannung regime [--symbol SYM]       # regime screener
nat spannung spectral [--symbol SYM]     # spectral analysis
nat profile [--symbol SYM]               # regime profiling
nat profile scalp [--symbol SYM]         # walk-forward profiler
nat scan [--symbol SYM]                  # signal discovery
nat macro [--symbol SYM]                 # macro regime analysis
nat kalman analysis [--symbol SYM]       # Kalman filter analysis
```

#### Algorithms
```bash
nat algorithm list                       # list all 27 algorithms
nat algorithm evaluate {--all|--algorithm NAME}
nat algorithm config                     # TOML config
nat backtest algorithm --algorithm NAME --symbol SYM
```

#### Alpha Pipeline
```bash
nat alpha_pipeline_start                 # start 9-step pipeline
nat alpha_pipeline_resume [--force-gate] # resume with optional gate override
nat alpha_pipeline_status                # current step
nat alpha_pipeline_gates                 # gate thresholds
nat alpha_pipeline_step N                # run specific step
nat screen [--symbol SYM]               # feature screening (step 1)
```

#### Backtesting
```bash
nat backtest [--symbol SYM]              # generic backtest
nat backtest algorithm --algorithm NAME  # algorithm-specific
nat backtest ml [--symbol SYM]           # ML prediction backtest
nat backtest funding                     # funding reversion
nat backtest list                        # list experiments
```

#### OOS Validation
```bash
nat oos30                                # all 5 winning algorithms
nat 15m                                  # 15-minute smoke test
nat 15m offline                          # offline smoke test
nat trade viz                            # trade visualization
```

#### Discovery & IT Engine
```bash
nat discovery {start,once,status,stop}   # continuous sweep daemon
nat it_engine {start,stop,status}        # information-theory engine
```

#### Cascade Validation
```bash
nat cascade {start,once,status,stop,report}
```

#### Models
```bash
nat model train                          # baseline ML training
nat model train gmm                      # GMM regime model
nat model list                           # list trained models
nat model score                          # prediction scoring
nat model serve                          # FastAPI model server
```

#### Clusters
```bash
nat cluster {analyze,gmm,all,quality,explore,hmm}
```

#### Experiments
```bash
nat exp {start,stop,status,check,midweek,analyze,dashboard,tunnel}
```

#### Build & Dev
```bash
nat build [debug|api|clean|fmt|lint|check]
nat test [verbose|hypotheses|validate|api|redis|integration|backtest|cluster|pipeline|dashboard|serving|eamm]
```

#### Infrastructure
```bash
nat api start                            # REST API server (port 3000)
nat api alerts                           # Telegram alert service
nat api serve all                        # API + alerts
nat docker {build,up,down,logs}          # Docker orchestration
nat config {show,get,validate}           # configuration management
nat log agent                            # agent log tail
nat reports [latest|show]                # report management
nat commands                             # list all commands
nat help                                 # usage help
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/ing.toml` | Ingestor: WebSocket URL, symbols, emission interval (100ms), output format |
| `config/agent.toml` | Agent: cycle interval, 5-gate thresholds, FDR q, generators, decay monitoring, promotion |
| `config/alpha.toml` | Alpha pipeline: gate thresholds G1-G8, step parameters, symbols |
| `config/pipeline.toml` | Pipeline orchestration: ingestion duration, analysis thresholds |
| `config/discovery.toml` | Discovery orchestrator: sweep config, training, backtesting |
| `config/algorithms.toml` | Algorithm parameters: per-algorithm constructor kwargs |
| `config/it_engine.toml` | IT engine: buffer size, KSG k, horizons, cost thresholds |
| `config/kalman.toml` | Kalman filter parameters |
| `config/hypothesis_testing.toml` | Hypothesis test parameters (H1-H5) |
| `config/symbols.toml` | Tradeable symbol list |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Rust logging level (e.g., `info`, `debug`) |
| `REDIS_URL` | Redis connection URL |
| `ING_DASHBOARD_ENABLED` | Enable ingestor dashboard |
| `NAT_RESEARCH_DIR` | Research data directory for API |
| `TELEGRAM_BOT_TOKEN` | Telegram alert bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat for alerts |

---

## Testing

```bash
# Rust
make test                               # unit tests (cargo test --package ing)
make test_verbose                       # with --nocapture
cd rust && cargo test -- test_name      # single test

# Python
pytest scripts/tests/                   # full suite (68 test files)
make test_agent                         # 350 agent tests (unit + integration + logging + research output)
make test_pipeline                      # pipeline state machine
make test_dashboard                     # dashboard endpoints
make test_eamm                          # EAMM module

# Validation
make validate                           # live API validation (4 binaries against Hyperliquid)
nat validate skeptical                  # 20+ statistical tests (FDR, bootstrap, permutation)

# Smoke tests
nat 15m                                 # 15-minute live smoke test
nat 15m offline                         # offline smoke test on parquet data
```

### Hypothesis Testing (H1-H5)

Five structural hypotheses validated before deployment:

| # | Hypothesis | Status |
|---|-----------|--------|
| H1 | Whale flow predicts returns | Confirmed |
| H2 | Entropy × whale interaction | Confirmed |
| H3 | Liquidation cascade prediction | Confirmed |
| H4 | Concentration → volatility | Confirmed |
| H5 | Trend persistence indicator | Confirmed |

Decision matrix in `hypothesis/final_decision.rs`: 0-1 pass = NOGO, 2-3 = PIVOT, 4-5 = GO.

---

## Docker

```bash
make docker_build                      # build all images
make docker_up                         # start full stack
make docker_down                       # stop
make docker_logs                       # tail logs
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **redis** | 6379 | Pub/sub, caching (256MB max, LRU) |
| **ingestor** | — | Market data collection (depends on redis) |
| **api** | 3000 | REST/WebSocket endpoints |
| **alerts** | — | Telegram alert service |
| **web** | 3001 | Next.js frontend (depends on api) |

---

## Multi-Machine Setup

The Rust ingestor runs on a separate machine (`su-35`) for low-latency data collection. The agent daemons and analysis tools run on the research machine, reading Parquet files from `data/features/`.

```
┌──────────────────┐         ┌──────────────────┐
│  su-35 (ingestor)│         │ research machine  │
│                  │  rsync  │                   │
│  nat start       │ ──────▶ │  nat agent start  │
│  Hyperliquid WS  │  data/  │  nat spannung     │
│  Parquet output  │         │  nat oos30         │
└──────────────────┘         └──────────────────┘
```

---

## Project Structure

```
nat/
├── nat                            # Unified CLI (4,400 lines, 163+ commands)
├── Makefile                       # Build/dev targets (compat, prefer nat CLI)
├── FEATURES.md                    # 209-feature manifest with formulas
├── CLAUDE.md                      # Architecture guide
│
├── rust/                          # Rust workspace
│   ├── ing/src/                   # Ingestor crate
│   │   ├── main.rs                # tokio::select! biased loop
│   │   ├── ws/                    # Hyperliquid WebSocket client
│   │   ├── state/                 # OrderBook, TradeBuffer, MarketContext
│   │   ├── features/              # 15 feature modules (209 features)
│   │   ├── output/                # Parquet writer (Arrow, hourly rotation)
│   │   ├── dashboard/             # Real-time monitoring (Axum WS)
│   │   ├── hypothesis/            # H1-H5 hypothesis tests
│   │   ├── ml/                    # GMM regime classification
│   │   └── bin/                   # Validation binaries (7 tools)
│   └── api/src/                   # API crate (Axum, port 3000)
│       ├── routes/                # REST endpoints (research, health, whales, regime)
│       └── bin/                   # Alert service binary
│
├── scripts/                       # Python analysis & ML
│   ├── agent/                     # Autonomous research agents
│   │   ├── base.py                # ResearchAgent ABC (full cycle loop, FDR, chaining)
│   │   ├── daemon.py              # MicrostructureAgent (tick-level)
│   │   ├── mf_daemon.py           # MediumFrequencyAgent (1min-1h)
│   │   ├── macro_daemon.py        # MacroAgent (1h-24h)
│   │   ├── meta_daemon.py         # MetaAgent (orchestrator)
│   │   ├── runner.py              # MicrostructureRunner (5-gate executor)
│   │   ├── mf_runner.py           # MediumFrequencyRunner (4-gate)
│   │   ├── macro_runner.py        # MacroRunner (4-gate)
│   │   ├── hypothesis_queue.py    # SQLite-backed priority queue
│   │   ├── research_output.py     # Structured JSON emitter (LaTeX math)
│   │   ├── cache.py               # SHA-256 computation cache (7-day TTL)
│   │   ├── meta_portfolio.py      # Risk parity portfolio optimization
│   │   └── generators/            # Hypothesis generators
│   │       ├── systematic.py      # Exhaustive feature × gate × threshold
│   │       ├── spectral.py        # PSD/OU anomaly detection
│   │       ├── regime.py          # HMM transition detection
│   │       ├── cross_asset.py     # Lead-lag probing (BTC/ETH/SOL)
│   │       ├── recycler.py        # Graveyard re-evaluation
│   │       ├── it_discovery.py    # IT engine integration (MI/CMI/II)
│   │       └── ensemble.py        # Ensemble prediction aggregation
│   │
│   ├── alpha/                     # Alpha signal pipeline (9 steps)
│   │   ├── alpha_pipeline.py      # State machine orchestrator
│   │   ├── screener.py            # FDR-controlled feature screening
│   │   ├── combiner.py            # Signal combination
│   │   ├── portfolio.py           # Portfolio assembly
│   │   ├── regime_filter.py       # HMM regime conditioning
│   │   ├── multi_freq.py          # Multi-frequency integration
│   │   ├── paper_trader.py        # 3f liquidity paper trader
│   │   ├── paper_trader_generic.py # Generic algo paper trader (17 algos)
│   │   ├── paper_trader_surprise.py # Surprise signal paper trader
│   │   └── deployer.py            # Live deployment executor
│   │
│   ├── algorithms/                # 27 microstructure algorithms
│   │   ├── base.py                # MicrostructureAlgorithm ABC
│   │   ├── registry.py            # @register decorator, auto-discovery
│   │   ├── runner.py              # AlgorithmRunner (parquet + dataframe)
│   │   ├── evaluate.py            # IC/drift evaluation harness
│   │   ├── jump_detector.py       # Lee-Mykland jump test
│   │   ├── funding_reversion.py   # Funding rate mean-reversion
│   │   ├── optimal_entry.py       # SPRT on Kalman innovation
│   │   ├── surprise_signal.py     # Entropy regime transition
│   │   ├── hawkes_intensity.py    # Self-exciting process
│   │   ├── convolver.py           # Kernel convolution discovery
│   │   └── ...                    # 17 more algorithms
│   │
│   ├── analysis/                  # Signal analysis (11 modules)
│   ├── backtest/                  # Backtesting engine (8 modules)
│   ├── cluster_pipeline/          # Unsupervised clustering (13 modules)
│   ├── cluster_quality/           # Quality metrics (5 modules)
│   ├── data/                      # Data utilities (6 modules)
│   ├── eamm/                      # Entropy-adaptive market making (10 modules)
│   ├── execution/                 # Signal bridge + Hyperliquid client
│   ├── experiment/                # Experiment monitoring (6 modules)
│   ├── it_engine/                 # Information theory engine (6 modules)
│   ├── polymarket/                # Prediction market integration (6 modules)
│   ├── utils/                     # Shared utilities
│   ├── viz/                       # Visualization modules
│   ├── workflows/                 # Validation workflows
│   └── tests/                     # 68 test files
│
├── web/                           # Next.js dashboard
│   └── src/
│       ├── app/                   # Pages (explorer, heatmap, graveyard, network, signals, math)
│       ├── components/            # UI (hypothesis table, IC chart, gate funnel, agent cards)
│       └── lib/                   # API client + WebSocket
│
├── config/                        # TOML configuration (10 files)
├── data/                          # Output (parquet, state, research)
├── reports/                       # Generated reports & analysis
├── docs/                          # Architecture, research papers, specs
├── docker/                        # Dockerfiles
├── docker-compose.yml             # Full stack orchestration
├── notebooks/                     # Jupyter notebooks
├── models/                        # Trained ML models
└── logs/                          # Ingestor & agent logs
```

---

## Key Findings

### Spannung Research Arc (Phases A-F)

| Finding | Evidence | Implication |
|---------|----------|-------------|
| OBI predicts 5s returns | IC = 0.19, 100% sign consistency across 5 folds | Structural signal, not overfit |
| `ent_book_shape` is #1 regime gate | Independently #1 on BTC, ETH, and SOL | Universal microstructure property |
| Signal replicates cross-symbol | KEEP on all 3 symbols, IC ordered by liquidity (SOL > ETH > BTC) | Genuine LOB effect |
| Brown noise universality | PSD slope ~ −1.85 on all 3 symbols | Fractional Brownian motion microstructure |
| 68s coherence dominance | Single peak at 0.015 Hz | Natural market-making cycle |
| OU half-life orders by liquidity | BTC 7.3s > ETH 5.3s > SOL 3.3s | Per-symbol refresh rates |
| `_last` features replicate, `_mean`/`_std` don't | Instantaneous = KEEP, aggregated = DROP | Time-domain spectral confirmation |

### Algorithm Sweep (17 Algorithms, 100min Horizon)

- **3f liquidity**: Sharpe 9.2 BTC / 7.8 ETH — strongest single-symbol algorithm
- **jump_detector**: +23,199 bps total — strongest cross-symbol algorithm
- **optimal_entry**: +13,679 bps — SPRT-based entry timing
- **funding_reversion**: +14,459 bps — crypto-native mean-reversion
- **surprise_signal**: Sharpe 6.7 SOL — entropy regime transitions

### Hypothesis Suite (H1-H6)

All six confirmed: directional, long-biased, no decay, 3-feature viable, maker viable.

---

## References

1. Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*, 5(1), 31-56.
2. Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
3. Bacry, E., Mastromatteo, I. & Muzy, J.F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity*, 1(1).
4. Bandt, C. & Pompe, B. (2002). Permutation entropy. *Physical Review Letters*, 88(17), 174102.
5. Barndorff-Nielsen, O.E. & Shephard, N. (2004). Power and bipower variation. *Econometrica*, 72(1), 1-37.
6. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289-300.
7. Bouchaud, J.P., Gefen, Y., Potters, M. & Wyart, M. (2004). Fluctuations and response in financial markets. *Quantitative Finance*, 4(2), 176-190.
8. Cont, R. & de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Math*, 4(1), 1-25.
9. Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
10. Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
11. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*. 2nd ed. Wiley.
12. Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity. *Review of Financial Studies*, 25(5), 1457-1493.
13. Elliott, R.J., Aggoun, L. & Moore, J.B. (2005). *Hidden Markov Models*. Springer.
14. Garman, M.B. & Klass, M.J. (1980). On the estimation of security price volatilities. *Journal of Business*, 53(1), 67-78.
15. Gatheral, J. & Oomen, R. (2010). Zero-intelligence realized variance estimation. *Finance and Stochastics*, 14(2), 249-283.
16. Glosten, L.R. & Milgrom, P.R. (1985). Bid, ask and transaction prices. *Journal of Financial Economics*, 14(1), 71-100.
17. Gueant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). Dealing with the inventory risk. *Mathematics and Financial Economics*, 4(7), 477-507.
18. Hamilton, J.D. (1989). A new approach to nonstationary time series. *Econometrica*, 57(2), 357-384.
19. Harvey, C.R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
20. Hendershott, T., Jones, C.M. & Menkveld, A.J. (2011). Does algorithmic trading improve liquidity? *Journal of Finance*, 66(1), 1-33.
21. Hoerl, A.E. & Kennard, R.W. (1970). Ridge regression. *Technometrics*, 12(1), 55-67.
22. Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65-91.
23. Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.
24. Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
25. Lee, S.S. & Mykland, P.A. (2008). Jumps in financial markets. *Review of Financial Studies*, 21(6), 2535-2563.
26. Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions. *SIAM Review*, 10(4), 422-437.
27. Parkinson, M. (1980). The extreme value method for estimating variance. *Journal of Business*, 53(1), 61-65.
28. Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
29. Rabiner, L.R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286.
30. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
31. Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
32. Shiryaev, A.N. (1978). *Optimal Stopping Rules*. Springer.
33. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3-4), 285-294.
34. Wald, A. (1947). *Sequential Analysis*. Wiley.
35. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

---

<p align="center">
<i>Built with Rust, Python, and relentless hypothesis testing.</i>
</p>

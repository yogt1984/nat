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
Hyperliquid perpetual futures microstructure. A Rust ingestor computes 209
order book features at 100ms resolution; an autonomous Python agent generates
hypotheses, tests them through a 5-gate replication protocol with FDR control,
and registers validated signals — without human intervention.

NAT is an autonomous microstructure alpha discovery agent that operates as a hypothesis-testing state machine over a 191-dimensional feature space extracted from Hyperliquid perpetual futures order books at 100ms resolution. The agent generates candidate hypotheses of the form *"signal feature F, conditioned on regime gate G < Q_p, predicts h-second forward returns"* and subjects each to a 5-gate sequential testing protocol. **Gate 1 (Discovery):** the agent computes the Spearman rank information coefficient `IC(F, r_{t+h} | G < Q_p) = 1 - 6 sum(d_i^2) / (n(n^2-1))` and requires both `IC >= min_ic(t)` and a differential improvement `dIC = IC_gated - IC_baseline >= 0.05`, where the acceptance threshold is adaptive: `min_ic(t) = max(0.10, median{IC_i : i in R(t)} * 0.8)`, rising as the registry `R(t)` accumulates stronger signals. **Gate 2 (Cost):** a backtest verifies gross per-trade edge `E[r | signal] >= 0.1 bps`. **Gates 3-4 (Replication):** the IC condition must hold on at least one additional date (temporal) and one additional symbol from `{BTC, ETH, SOL}` (cross-asset), guarding against overfitting (White, 2000) and asset-specific artifacts. **Gate 5 (Deduplication):** the candidate's regime-gated values must satisfy `max_j rho_S(F_candidate, F_j) < 0.7` against all registered signals `j`, preventing redundancy. At cycle end, Benjamini-Hochberg FDR control (`q = 0.05`) is applied across all tested hypotheses using IC p-values `p = erfc(|IC| sqrt(n) / sqrt(2))`, rejecting those that do not survive multiple-testing correction. Surviving signals enter a registry monitored for IC decay: if `IC_rolling(s) < IC_discovery(s) * 0.5` for 14 consecutive cycles, the signal is auto-retired. Generator budget allocation across the five hypothesis generators (systematic, spectral, regime, cross-asset, recycler) follows a Beta-prior Thompson bandit: `w(g) = (successes_g + 1) / (attempts_g + 2)`, steering exploration toward productive generators. Deterministic `nat` command outputs are cached under `SHA-256(canonical_args)` with 7-day TTL, yielding 85% hit rates and 56% cycle-time reduction.

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
              +----------------------+----------------------+
              |                                             |
              v                                             v
     +----------------------------+          +----------------------------+
     |     NAT AGENT DAEMON       |          |    IT ENGINE               |
     |                            |          |                            |
     | 6 Generators --> Priority Q|<---------|  KSG MI, CMI, TE          |
     | Runner: 5-gate replication |  IT      |  Greedy feature selection  |
     | Registry: validated signals|  hypoths |  Cost gate: I_min(k)      |
     +-------------+--------------+          +----------------------------+
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
nat agent report      # full summary (registry + graveyard + generators + cache)
nat agent dashboard   # web dashboard on :8060 with IC heatmap
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

### Five-Gate Replication Protocol

Every hypothesis must independently pass five gates before registration.
This controls the false discovery rate under multiple testing
(Harvey, Liu & Zhu, 2016).

```
DISCOVERY ──> COST ──> TEMPORAL ──> SYMBOL ──> CORRELATION ──> REGISTER
    |           |          |           |             |
    v           v          v           v             v
GRAVEYARD  GRAVEYARD  GRAVEYARD  GRAVEYARD      GRAVEYARD

                     + FDR control (BH, q=0.05) at end of each cycle
```

**Gate 1 — Discovery (IC + dIC).** Run `nat spannung regime` on BTC with latest data.
Extract gate-specific IC (not aggregate). Pass conditions:
- Spearman rank IC >= `min_ic` (adaptive, see below)
- dIC >= 0.05 (gated IC must exceed ungated baseline)

```
                  6 * sum(d_i^2)
  IC  =  1  -  -----------------------       d_i = rank(signal_i) - rank(return_i)
                  n * (n^2 - 1)
```

**Gate 2 — Cost.** Run `nat spannung backtest` and parse `avg_return_per_trade_bps`.
Pass condition: gross return per trade >= 0.1 bps. Signals that cannot cover
execution costs are parked (eligible for recycler re-evaluation).

**Gate 3 — Temporal replication.** Re-run on 2 other dates from the manifest.
Pass condition: IC gate holds on >= 1 additional date. A signal that works on
only one date is a statistical artifact (White, 2000).

**Gate 4 — Symbol replication.** Re-run on ETH and SOL.
Pass condition: IC gate holds on >= 1 other symbol. Cross-asset replication is
the strongest evidence of a structural microstructure effect vs. asset-specific
overfitting (Gueant, Lehalle & Fernandez-Tapia, 2012).

**Gate 5 — Correlation deduplication.** Compute max Spearman correlation of the
candidate signal against all existing registry signals (on regime-gated values
loaded from Parquet). Pass condition: max rho < 0.7. Redundant signals are
rejected to keep the registry diverse.

**FDR control.** At the end of each cycle, a Benjamini-Hochberg procedure is
applied across all tested hypotheses at q=0.05. IC p-values are computed via
z-test: `z = IC * sqrt(n)`, two-sided `p = erfc(z/sqrt(2))`. Hypotheses that
don't survive FDR correction are removed from the registry.

### Adaptive IC Threshold

The IC acceptance threshold rises as the registry accumulates high-quality signals:

```
  min_ic(t)  =  max( floor,  median{ IC_i : i in R(t), status != retired } * 0.8 )
```

where `R(t)` is the registry at cycle `t` and `floor = 0.10` (configurable).
With a single registered signal at IC = 0.569, the threshold rises from 0.10
to 0.455 — marginal signals cannot enter a strong registry. Computed once at
the start of each cycle and injected into every hypothesis before execution.

### IC Decay Monitoring

Registered signals are continuously monitored for degradation. Each cycle,
the MONITOR phase computes a rolling Spearman IC on the latest data:

```
  IC_rolling(s)  =  rho_S( F_s[mask_G], r_{t+h}[mask_G] )
```

where `F_s` is the signal feature, `mask_G` is the regime gate, and `r_{t+h}` is
the 5-second forward return. If `IC_rolling` drops below 50% of discovery IC
for 14 consecutive days, the signal is auto-retired:

```
  if IC_rolling(s) < IC_discovery(s) * 0.5   for 14 consecutive cycles:
      status(s) ← retired,  reason ← ic_decay
```

The decay counter resets on recovery. IC history (last 30 entries) is persisted
per signal for trend analysis. Both `ic_decay_ratio` (default 0.5) and
`consecutive_days_limit` (default 14) are configurable in `config/agent.toml`.

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

## Information-Theoretic Alpha Discovery Engine

The IT Engine (`scripts/it_engine/`) provides continuous, principled
information-theoretic analysis of all 209 features against forward returns
at multiple horizons. It replaces ad-hoc IC scanning with rigorous mutual
information estimation, proper entropy conditioning, and cost-aware feature
selection.

```bash
nat it-engine start --symbol BTC               # live mode (Redis pub/sub)
nat it-engine start --symbol BTC --offline      # offline mode (parquet files)
nat it-engine start --symbol BTC --dry-run      # single cycle and exit
nat it-engine status --symbol BTC               # show MI rankings, greedy selection
nat it-engine stop --symbol BTC                 # graceful shutdown
```

### Core Estimators

| Estimator | Formula | Purpose |
|-----------|---------|---------|
| **KSG MI** | Kraskov-Stögbauer-Grassberger k-NN (k=5) | I(f; r) — mutual information between feature and returns |
| **Conditional MI** | I(f; r \| H) via KSG in joint/marginal spaces | Proper IT formulation of entropy gating |
| **Interaction Info** | II(f;r;H) = I(f;r\|H) - I(f;r) | Positive → synergy (gating helps), Negative → redundancy |
| **Linear TE** | TE = 0.5 × log(σ²_reduced / σ²_full) | Causal information flow (Sherman-Morrison compatible) |
| **Cost threshold** | I_min = -0.5 × log₂(1 - (fee/σ_r)²) | Minimum MI (bits) to overcome transaction costs |

References: Kraskov, Stögbauer & Grassberger (2004), Schreiber (2000), Cover & Thomas (2006).

### Greedy Feature Selection

Forward stepwise selection by conditional MI gain — a tractable alternative
to full Partial Information Decomposition (NP-hard for >3 variables):

1. **Start**: f* = argmax_f I(f; r_k) across all features and horizons
2. **Step**: f_next = argmax_f I(f; r_k | S) — largest conditional MI gain given selected set S
3. **Stop**: when marginal gain < I_min(k) or max features reached

Output: ordered feature set with cumulative MI and cost viability flag.

### Agent Integration

The `it_discovery` generator reads IT engine state and creates hypotheses
for features with cost-viable MI and positive interaction information
(synergistic with entropy gating). Priority = CMI × (1 + II).

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

## Microstructure Algorithm Library

NAT includes 18 configurable microstructure algorithms that compute 59 derived
features from the base ingestor feature vector. Each algorithm implements
the `MicrostructureAlgorithm` interface with tick-by-tick `step()` and
vectorized `run_batch()` methods. Parameters are configurable via
`config/algorithms.toml`.

```bash
nat algorithm list                     # show all 18 algorithms and their features
nat algorithm evaluate --all           # IC/drift evaluation on all algorithms
nat algorithm evaluate --algorithm hawkes_intensity --symbol BTC
nat algorithm config                   # show TOML configuration
nat backtest algorithm --algorithm weighted_ofi --symbol BTC
```

| # | Algorithm | Features | Reference |
|---|-----------|----------|-----------|
| 1 | `kalman_imbalance` | 4 | Kalman OU filter on L1 imbalance |
| 2 | `regime_gated` | 3 | Entropy percentile gating — Bandt & Pompe (2002) |
| 3 | `multi_level_imb` | 3 | Weighted L1/L5/L10 composite |
| 4 | `weighted_ofi` | 3 | Depth-decay weighted OFI — Cont, Kukanov & Stoikov (2014) |
| 5 | `trade_through` | 3 | Queue depletion probability — Cont & de Larrard (2013) |
| 6 | `propagator` | 3 | Transient impact with power-law kernel — Bouchaud et al. (2004) |
| 7 | `hawkes_intensity` | 4 | Self-exciting trade arrival — Bacry, Mastromatteo & Muzy (2015) |
| 8 | `jump_detector` | 4 | Lee-Mykland nonparametric jump test — Lee & Mykland (2008) |
| 9 | `bipower_jump` | 4 | BV jump decomposition — Barndorff-Nielsen & Shephard (2004) |
| 10 | `vpin_regime` | 3 | VPIN-triggered regime switch — Easley, López de Prado & O'Hara (2012) |
| 11 | `spread_decomp` | 3 | Adverse selection decomposition — Hendershott, Jones & Menkveld (2011) |
| 12 | `entropy_momentum` | 3 | Entropy-gated momentum (novel) |
| 13 | `surprise_signal` | 3 | Entropy regime transition detection |
| 14 | `funding_reversion` | 3 | Funding rate mean-reversion (crypto-specific) |
| 15 | `oi_divergence` | 3 | Open interest vs price divergence |
| 16 | `switching_ou` | 4 | Two-regime OU with Bayesian filtering — Elliott et al. (2005), Hamilton (1989) |
| 17 | `optimal_entry` | 3 | SPRT on Kalman innovation — Wald (1947), Shiryaev (1978) |
| 18 | `online_ridge` | 3 | Online ridge regression meta-algorithm — Hoerl & Kennard (1970) |

Evaluation computes Spearman IC at horizons {1, 5, 10, 50, 100} ticks with
regime-gated IC (ent_book_shape < P30) and post-fill drift analysis.

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

# Agent (autonomous research)
make agent_start        # launch daemon in tmux (or: nat agent start)
make agent_stop         # graceful shutdown
make agent_status       # current state
make agent_report       # full summary (registry + graveyard + generators + cache)
make agent_dashboard    # web dashboard on :8060 with IC heatmap

# Agent watchdog (auto-restart via cron)
make agent_watchdog_install   # checks every 5 minutes
make agent_watchdog_remove    # remove cron entry

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
make test_agent                         # agent tests (350 tests: unit + integration + logging + research output)
pytest scripts/tests/test_it_estimators.py   # IT engine estimator tests (17 tests)
```

## Configuration

| File | Purpose |
|------|---------|
| `config/ing.toml` | Ingestor: WebSocket URL, symbols, emission interval, output format |
| `config/agent.toml` | Agent: cycle interval, experiment budget, 5-gate thresholds (IC, dIC, cost, FDR q), decay monitoring, promotion criteria |
| `config/pipeline.toml` | Pipeline orchestration: ingestion duration, analysis thresholds |
| `config/algorithms.toml` | Algorithm parameters: per-algorithm constructor kwargs (decay rates, windows, thresholds) |
| `config/it_engine.toml` | IT engine: buffer size, KSG k, horizons, entropy conditioning, cost thresholds |

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
    routes/research.rs      Research endpoints (hypotheses, cycles, signals, stats, heatmap)

scripts/
  agent/                  Autonomous research agent
    base.py                 ResearchAgent ABC + BaseRunner ABC (full cycle loop, FDR, chaining)
    daemon.py               MicrostructureAgent thin subclass + CLI
    mf_daemon.py            MediumFrequencyAgent thin subclass + CLI
    macro_daemon.py         MacroAgent thin subclass + CLI
    meta_daemon.py          MetaAgent orchestrator (cross-agent budget, correlation)
    runner.py               MicrostructureRunner 5-gate executor
    mf_runner.py            MediumFrequencyRunner 4-gate executor
    macro_runner.py         MacroRunner 4-gate executor
    hypothesis.py           Hypothesis, RegisteredSignal, GeneratorStats dataclasses
    hypothesis_queue.py     SQLite-backed priority queue with dedup
    research_output.py      Structured JSON per hypothesis + cycle summary (LaTeX math)
    manifest.py             Data availability scanner
    cache.py                Computation cache (SHA-256 keys, TTL-based, corruption-resilient)
    generators/             Hypothesis generators (lazy-imported via generator_module_prefix)
      systematic.py           Exhaustive feature x gate x threshold search
      spectral.py             PSD/OU anomaly detector
      regime.py               HMM transition detector
      cross_asset.py          Lead-lag prober (BTC/ETH/SOL)
      recycler.py             Graveyard re-evaluator
      it_discovery.py         IT engine-driven hypotheses (MI, CMI, II)
  logging_config.py         Centralized JSON logging with correlation context
  agent_dashboard.py        Agent web dashboard (stdlib HTTP, port 8060, IC heatmap)
  algorithms/                   Microstructure algorithm library (18 algorithms, 59 features)
    base.py                       MicrostructureAlgorithm ABC, AlgorithmFeature
    registry.py                   @register decorator, get_algorithm(), list_algorithms()
    runner.py                     AlgorithmRunner (parquet + dataframe modes)
    evaluate.py                   IC/drift evaluation harness
    autodiscover.py               importlib-based auto-registration
    tests/                        Parametrized smoke tests (195 tests)
  spannung_regime_screener.py   Quintile regime IC screening
  spannung_spectral.py          Frequency-domain analysis
  scalping_profiler.py          Walk-forward feature profiler
  cluster_pipeline/             Unsupervised regime discovery (15 modules)
  backtest/                     Walk-forward backtesting engine
    algorithm_strategy.py         Algorithm-to-backtest bridge
  eamm/                         Entropy-adaptive market making (prototype)
  it_engine/                IT alpha discovery engine
    daemon.py                 Main loop: Redis/parquet → MI/CMI/TE → greedy selection
    estimators.py             KSG MI, conditional MI, interaction info, linear TE
    state.py                  ITState dataclass + JSON persistence
    feature_selector.py       Greedy forward selection by conditional MI gain
    config.py                 TOML config loader
  pipeline_runner.py            Pipeline state machine

config/
  ing.toml                Ingestor configuration
  agent.toml              Agent daemon configuration
  pipeline.toml           Pipeline orchestration
  algorithms.toml         Algorithm parameters (18 sections)
  it_engine.toml          IT engine: buffer size, KSG k, horizons, cost thresholds

data/
  features/               Parquet output (YYYY-MM-DD/*.parquet)
  nat.db                  SQLite state store (agent state, hypotheses, registry)
  research/               Structured research output
    hypotheses/             Per-hypothesis JSON records (id, gates, math, status)
    cycles/                 Per-cycle JSON summaries (tested, registered, FDR, stats)
  logs/                   Structured JSON logs (nat.jsonl, daily rotation)
  agent/                  Agent state (legacy JSON, migrated to SQLite)
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

The generator is auto-discovered via `generator_module_prefix` — no import
boilerplate needed. Place the module in the correct package for the agent type:
- Microstructure: `scripts/agent/generators/my_generator.py`
- Medium-frequency: `scripts/agent/generators/medium_freq/my_generator.py`
- Macro: `scripts/agent/generators/macro/my_generator.py`

### Adding a New Gate Check

Add to `runner.py`:
```python
def check_my_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Return (passed, message)."""
```

Wire into `ExperimentRunner._check_gates()`.

### Consolidated Daemon Architecture

The agent daemon architecture is consolidated around `ResearchAgent` (base.py),
which owns the full cycle loop, state machine, generator dispatch, FDR control,
hypothesis chaining, promotion checks, and structured output emission. Each
agent subclass is a thin ~80-110 LOC file overriding only config attributes
and `create_runner()`. Adding a new agent requires only a TOML section and
~30 LOC subclass.

| Agent | Subclass | Timeframe | Generators |
|-------|----------|-----------|------------|
| Microstructure | `MicrostructureAgent` | Tick-level (1-10s) | `agent.generators.*` |
| Medium-Frequency | `MediumFrequencyAgent` | 1min-1h | `agent.generators.medium_freq.*` |
| Macro | `MacroAgent` | 1h-24h | `agent.generators.macro.*` |
| Meta | `MetaAgent` | Orchestrator | Cross-agent budget, correlation, portfolio |

Generator dispatch uses `generator_module_prefix` for lazy importing — no
registration boilerplate needed beyond placing a module with a `generate()` function.

### Agent State Machine

```
Per-cycle:   MANIFEST -> GENERATE -> ADAPTIVE IC -> EXECUTE (budget: 10 or 90min)
             -> FDR control (BH q=0.05) -> STRUCTURED OUTPUT -> MONITOR (decay + promotion) -> SLEEP

Per-hypothesis:
  SETUP -> DISCOVERY (IC+dIC) -> COST -> TEMPORAL -> SYMBOL -> CORRELATION -> REGISTER
    |          |                   |        |          |            |
    v          v                   v        v          v            v
  ABORT    GRAVEYARD          GRAVEYARD  GRAVEYARD  GRAVEYARD   GRAVEYARD
           (no_effect)        (cost_killed)        (no_repl)   (redundant)
```

State persists to `data/nat.db` (SQLite). The daemon handles SIGTERM
gracefully (finishes current experiment before stopping).

### Structured Research Output

After each hypothesis completes, the agent emits a self-contained JSON record
to `data/research/hypotheses/{id}.json` with full detail: claim, generator,
status, gate results (metric, threshold, p-value per gate), LaTeX math
derivation, features, regime gate, and timestamps. At cycle end, a summary
is emitted to `data/research/cycles/{cycle_id}.json` with aggregate stats
(tested, registered, FDR-rejected, chained) and per-generator hit rates.

LaTeX derivations are included for all 13 generator types (systematic, spectral,
regime, cross-asset, recycler, ensemble, momentum, vol_breakout, flow_cluster,
funding_meanrev, oi_divergence, whale_momentum, it_discovery).

### Structured Logging

All Python daemons use centralized logging (`scripts/logging_config.py`) with:
- **JSON format** to `data/logs/nat.jsonl` (daily rotation, 30-day retention)
- **Correlation context**: `cycle_id` and `hypothesis_id` attached to every log line
  within a cycle, enabling `grep hypothesis_id data/logs/nat.jsonl` to trace a
  single hypothesis through all gates
- **Human format** on stderr for interactive use
- Thread-local context via `set_context()` / `clear_context()`

### Research REST API

The Axum API server (port 3000) exposes research data through REST endpoints
that read from the structured JSON files emitted by the agent:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research/hypotheses` | GET | Paginated list, filterable by `?agent=`, `?generator=`, `?status=`, `?limit=`, `?offset=` |
| `/api/research/hypotheses/:id` | GET | Full detail for a single hypothesis (gates, math, thresholds) |
| `/api/research/cycles` | GET | Cycle summaries, filterable by `?agent=` |
| `/api/research/signals` | GET | Registered signals only (status=replicated) |
| `/api/research/stats` | GET | Aggregate counts by status, agent, generator |
| `/api/research/heatmap` | GET | Feature x horizon IC matrix for visualization |

Configure data directory via `NAT_RESEARCH_DIR` env var (default: `../data/research`).

### Computation Cache

Deterministic `nat` commands (`spannung regime`, `spannung spectral`,
`spannung backtest`, `profile scalp`) on the same (date, symbol) produce
identical output. The cache stores report JSONs keyed by
`SHA-256(canonical_command)` with TTL-based expiry (7 days default).
Flag order is canonicalized so `--symbol BTC --data X` and `--data X --symbol BTC`
produce the same cache key. Measured: 85% hit rate, 56% cycle speedup.

### Agent Dashboard

`nat agent dashboard` launches a stdlib HTTP server on port 8060 with:
- Agent status panel (phase, cycle count, queue depth)
- Registry table (validated signals with IC)
- (Signal x Gate) IC heatmap (green=registered, red=graveyard, gray=untested)
- Graveyard table with failure reasons
- Queue panel (next hypotheses by priority)
- Generator performance stats
- Cache statistics
- 10-second auto-refresh, dark theme

### Signal Lifecycle

```
Hypothesis (queued) -> Discovery (IC+dIC) -> Cost (gross edge >= 0.1 bps)
  -> Temporal replication (2+ dates) -> Symbol replication (ETH + SOL)
  -> Correlation dedup (rho < 0.7) -> FDR control (BH q=0.05)
  -> Registry (validated) -> Paper trading -> Live (with human approval)
```

Registered signals are promoted through: `validated -> paper -> live -> retired`.
Paper-to-live promotion requires: 7-day Sharpe > 1.5, realized/predicted IC > 0.8,
max drawdown < 2%. The MONITOR phase checks these criteria every cycle and
auto-retires signals whose rolling IC decays below 50% of discovery IC for 14
consecutive days.

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
4. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289-300.
5. Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
6. Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457-1493.
7. Gatheral, J. & Oomen, R. (2010). Zero-intelligence realized variance estimation. *Finance and Stochastics*, 14(2), 249-283.
8. Glosten, L.R. & Milgrom, P.R. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*, 14(1), 71-100.
9. Gueant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). Dealing with the inventory risk. *Mathematics and Financial Economics*, 4(7), 477-507.
10. Hamilton, J.D. (1989). A new approach to nonstationary time series. *Econometrica*, 57(2), 357-384.
11. Harvey, C.R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
12. Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
13. Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions. *SIAM Review*, 10(4), 422-437.
14. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
15. Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
16. Rabiner, L.R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286.
17. Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
18. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3-4), 285-294.
19. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
20. Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.
21. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
22. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*. 2nd ed. Wiley.

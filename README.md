# NAT

```
                                                                                
     ███╗   ██╗ █████╗ ████████╗                                                
     ████╗  ██║██╔══██╗╚══██╔══╝                                                
     ██╔██╗ ██║███████║   ██║                                                   
     ██║╚██╗██║██╔══██║   ██║                                                   
     ██║ ╚████║██║  ██║   ██║                                                   
     ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝                                                   
                                                                                
     ╔══════════════════════════════════════════════════════════════╗            
     ║  Alpha Discovery for Crypto Perpetual Futures                ║            
     ║  ─────────────────────────────────────────────────────────── ║            
     ║  239 features · 100ms ticks · 177-perp candle universe       ║            
     ║  32 algorithms · 15 processes · 4 research agents            ║            
     ║  Pre-registration · FDR control · adversarial kill gates     ║            
     ║  Deployable tier: EMPTY. The record says why.                ║            
     ╚══════════════════════════════════════════════════════════════╝            
                                                                                
          ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             
          │  INGEST │───>│ DISCOVER│───>│ REFUTE  │───>│ PROMOTE │             
          │  (Rust) │    │ (PROC)  │    │ (gates) │    │ (human) │             
          └─────────┘    └─────────┘    └─────────┘    └─────────┘             
              ▲                              │                                  
              └────────── what survives ─────┘                                  
```

NAT is a quantitative research platform for extracting alpha from
[Hyperliquid](https://hyperliquid.xyz) perpetual futures microstructure. A Rust ingestor
computes 239 order-book features at 100 ms resolution for BTC/ETH/SOL; a Python research
layer — a process/IT discovery engine, four autonomous hypothesis agents, and a
cross-sectional layer over the venue's full 177-perp universe — generates, tests, and
mostly *kills* candidate signals under pre-registration and FDR control.

**The honest headline: NAT has no deployable strategy.** Its five shipped "winners" were
refuted by its own adversarial kill gate on 2026-07-30, and passive quoting at BTC's touch
is negative at every fee tier the venue offers. What the platform has built is the
apparatus that established both facts, and a research program running on it. The full
record — including every failure, with the same care as the successes — is
[`docs/research/FINDINGS.md`](docs/research/FINDINGS.md).

> **Objective:** [`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) · **Method:**
> [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) · **Plan & backlog:**
> [`docs/PLAN.md`](docs/PLAN.md) / [`docs/TASKS.md`](docs/TASKS.md) ·
> **Shorthand decoder:** [`docs/GLOSSARY.md`](docs/GLOSSARY.md)

---

## Where the project stands  *(2026-08-08)*

Three results are settled and should not be re-litigated. Each is a measurement, not an
opinion; each names its section in `FINDINGS.md`.

| Result | Evidence | Consequence |
|---|---|---|
| **The taker path is arithmetically closed** | the 1–5 s move is 0.5–2 bps against ~11 bps round-trip cost (§2) | no tick-scale taker strategy can clear cost, at any accuracy |
| **All five shipped "winners" are refuted** | Q4 kill gate, 2026-07-30, **5/5 KILL** — wrong-venue cost tier (1.61 bps Binance VIP9) plus a sweep harness that never ran each algorithm's own entry logic (§4.6) | all REJECTED in the lifecycle; the deployable tier is empty |
| **Passive quoting at BTC's touch is negative at every reachable fee tier** | breakeven maker rate = E[adverse\|fill] − half-spread = **+0.144 bps**, against a zero-fee best case (§4.10–4.11) | the venue must *pay* ~0.15 bps before a resting BTC quote breaks even |

**What survives:** the instruments (A4 EV gate, queue simulators, the HF1 microprice
center), the **PROC discovery layer** (complete end-to-end 2026-08-05), the **XS
cross-sectional layer** (Track C passed its pre-registered kill test, §7.4), and the
methodology that caught all of the above.

**The binding constraint has moved.** It is no longer "clean days" in general — the current
research program runs entirely on data in hand. Data continuity now gates specifically
**paper trading, live capital, and the fill-economics verdict (X-3)**.

> **Hard rule:** zero contact with the `su-35` ingestor until a clean streak completes.
> Verify locally with `nat gap status` — it does not touch su-35.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Ingestion Layer (Rust)](#data-ingestion-layer-rust)
- [The Candle Universe](#the-candle-universe)
- [Feature Vector (239 Dimensions)](#feature-vector-239-dimensions)
- [Microstructure Algorithm Library (32 Algorithms)](#microstructure-algorithm-library-32-algorithms)
- [Process Discovery Layer (PROC)](#process-discovery-layer-proc)
- [Cross-Sectional Layer (XS / Class 3)](#cross-sectional-layer-xs--class-3)
- [Autonomous Research Agents](#autonomous-research-agents)
- [Alpha Pipeline](#alpha-pipeline)
- [Information-Theoretic Engine](#information-theoretic-engine)
- [Backtesting & OOS Validation](#backtesting--oos-validation)
- [Analysis & Profiling Tools](#analysis--profiling-tools)
- [Lifecycle, Promotion & Risk Automation](#lifecycle-promotion--risk-automation)
- [Web Dashboard & API](#web-dashboard--api)
- [Config Swarm & Evolutionary Optimization](#config-swarm--evolutionary-optimization)
- [Other Modules](#other-modules)
- [The `nat` CLI](#the-nat-cli)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker](#docker)
- [Multi-Machine Setup](#multi-machine-setup)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Current Direction: Three-Class Maker System](#current-direction-three-class-maker-system)
- [References](#references)

---

## Quick Start

```bash
# 1. Build
nat build                             # LTO release build (all binaries)
nat build debug                       # faster iteration

# 2. Start data ingestion (tmux + watchdog + dashboard)
nat doctor                            # preflight: data-dir ownership, binary, disk
nat start                             # launch ingestor with auto-restart
nat log                               # tail live output
nat status                            # health check (add --json for machine output)
nat gap status                        # ingestion freshness (local read, safe during a freeze)

# 3. Inspect & validate the captured parquet
nat data validate                     # whole feature dir (continuity / NaN / ranges)
nat data validate data/features/2026-08-07/20260807_140000.parquet   # one file → PASS/WARN/FAIL
nat viz render --tf 15m               # whole-day all-features PNG, auto path
nat viz render --last 15m --open      # freshest readable window + its age
nat viz3d --tf 15m --features entropy --open   # interactive 3D surface (Plotly)

# 4. Run a discovery process (the PROC layer)
nat process list                      # 15 registered processes
nat process run ic_horizon --symbol BTC
nat process results                   # past runs from the nat.db index
nat xs ledger                         # program-level multiple-testing ledger

# 5. Explore the cross-sectional layer (Class 3)
nat xs universe                       # candle archive + L2 sampler coverage
nat xs rank                           # rank-IC vs relative forward returns (XS-3)
nat xs trajectory                     # how much more data the rotation verdict needs

# 6. Launch autonomous research agents
nat agent start                       # microstructure agent (tick-level)
nat mf-agent start                    # medium-frequency agent (1 min–1 h)
nat macro-agent start                 # macro agent (1 h–24 h)
nat meta-agent start                  # meta orchestrator (cross-agent)
nat agent status                      # phase, cycle count, registry size
nat agent dashboard                   # IC heatmap on :8060

# 7. Promotion state
nat lifecycle status                  # where every signal sits in the state machine
nat lifecycle list                    # (the deployable tier is currently empty)
```

Note the command shapes: groups are **hyphenated** (`nat mf-agent`, `nat macro-agent`,
`nat meta-agent`, `nat it-engine`), and multi-word subcommands use hyphens too
(`nat alpha pipeline-start`). `nat help` and `nat commands` are authoritative.

---

## Architecture

```
                      ┌──────────────────────────────┐   ┌──────────────────────────┐
                      │  Hyperliquid L2 WebSocket    │   │  Hyperliquid REST        │
                      │  (BTC / ETH / SOL, 100ms)    │   │  (177 perps, candles+L2) │
                      └──────────────┬───────────────┘   └────────────┬─────────────┘
                                     │                                │
                      ┌──────────────▼───────────────┐   ┌────────────▼─────────────┐
                      │     Rust Ingestor (ing)      │   │  fetch_candles / fetch_l2│
                      │  239 features × 100ms × 3    │   │  708 series, 3.06M bars  │
                      │  tokio::select! { biased; }  │   │  zero gaps               │
                      └──────────────┬───────────────┘   └────────────┬─────────────┘
                                     │                                │
                        data/features/YYYY-MM-DD/*.parquet    data/candles/
                                     │                                │
        ┌──────────────┬─────────────┼──────────────┬─────────────────┴──────┐
        ▼              ▼             ▼              ▼                        ▼
 ┌────────────┐ ┌────────────┐ ┌───────────┐ ┌────────────┐          ┌────────────┐
 │ PROC layer │ │ IT Engine  │ │  Research │ │   Alpha    │          │  XS layer  │
 │ 15 processes│ │ KSG MI/CMI │ │  Agents   │ │  Pipeline  │          │ (Class 3)  │
 │ FDR ledger │ │ TE, greedy │ │ 4 daemons │ │  9 steps   │          │ rank/rotate│
 └─────┬──────┘ └─────┬──────┘ └─────┬─────┘ └─────┬──────┘          └─────┬──────┘
       │              │              │             │                       │
       └──────────────┴──────────────┴─────┬───────┴───────────────────────┘
                                           ▼
                          ┌────────────────────────────────┐
                          │   Signal Lifecycle (nat.db)    │
                          │  DISCOVERED → VALIDATED →      │
                          │  PAPER → APPROVAL_PENDING →    │
                          │  LIVE → MONITORING → RETIRED   │
                          │      (+ REJECTED)              │
                          └────────────────┬───────────────┘
                                           │  sole human gate: nat lifecycle approve
                          ┌────────────────▼───────────────┐
                          │  Promotion · Kill-switch ·     │
                          │  Signal bridge · Gap alert     │
                          │  Prometheus / Grafana / API    │
                          └────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Data ingestion | **Rust** (tokio, Arrow, Parquet, Axum) |
| Feature computation | **Rust** (239 features, 21 categories, 100 ms emission) |
| Research layer | **Python** (PROC processes, IT engine, autonomous daemons, SQLite state, FDR) |
| ML & backtesting | **Python** (LightGBM, scikit-learn, pandas, numpy) |
| Optimization | **Python** (Optuna CMA-ES/TPE/NSGA-II, walk-forward OOS) |
| API server | **Rust** (Axum REST/WebSocket, port 3000) |
| Web dashboard | **Next.js** (TypeScript, Tailwind, React) |
| Agent dashboard | **Python** (stdlib HTTP server, port 8060) |
| Messaging | **Redis** (Pub/Sub + Streams) |
| State persistence | **SQLite** (`nat.db`: lifecycle, hypothesis queue, process index) + **PostgreSQL** (Optuna) |
| Alerting | **Telegram** (bot API) |
| Process management | **tmux** + watchdog auto-restart; **systemd** units for cloud |
| Containerization | **Docker Compose** (redis, ingestor, api, postgres, optuna, grafana, caddy, daemons) |

---

## Data Ingestion Layer (Rust)

The ingestor (`rust/ing/`) subscribes to the Hyperliquid L2 WebSocket and computes 239
features at 100 ms resolution for BTC, ETH, and SOL.

```
Hyperliquid WebSocket ──▶ OrderBook + TradeBuffer + MarketContext
    ──▶ FeatureComputer (239 features, 21 categories)
    ──▶ Parquet (data/features/YYYY-MM-DD/*.parquet, rotated hourly)
```

**Key design decisions:**

- Each symbol runs in its own `tokio` task.
- `tokio::select! { biased; }` prioritizes WebSocket messages over emission ticks — this
  prevents data loss under load and is intentional, not incidental.
- `ArrowWriter` with an explicit `flush()` after each batch, or files stay 0 bytes until
  close (hourly rotation or shutdown).
- Hourly rotation, 10,000-row buffer (~5.5 min at 30 rows/sec across 3 symbols).

### Data integrity — the operational record

Data continuity has been the platform's most expensive recurring failure, and it is
documented as such (`FINDINGS.md` §7). The tick record carries real caveats: missing
calendar days, a historical zombie-ingestor gap that produced no error logs, and dead
(all-NaN) columns from unwired optional categories. `nat gap status` and `nat data
validate` exist because of those incidents. Treat any tick-window claim as gated on the
streak it was measured over.

### Validation binaries

```bash
nat test validate               # 4 binaries against the live Hyperliquid API
nat run show BTC 10             # real-time feature display
```

---

## The Candle Universe

A second data substrate, independent of the tick path and of `su-35`. It is what the
current research program actually runs on.

```bash
nat xs universe                 # coverage report: candle archive + L2 sampler
python scripts/data/fetch_candles.py --universe   # backfill (XS-1)
python scripts/data/fetch_l2.py                   # L2 half-spread sweep (XS-8)
```

**What was captured** (XS-1, `FINDINGS.md` §7.1): 177 listed perps (55 delisted excluded)
× {1m, 5m, 15m, 1h} = **708 series, 3,059,200 candles, 98 MB — every series 100 % complete
within its span, zero gaps anywhere.** It is the first dataset on the platform with no
integrity caveat attached.

**The constraint that matters more than the volume** — a ~5,000-bar retention cap per
interval, measured rather than inferred:

| interval | candles retained | span reachable | measured depth (177 pairs) |
|---|---|---|---|
| 1m | ~5,000 | ~3.5 d | **3.5 d** (0 pairs reach 7 d) |
| 5m | ~5,000 | ~17 d | 17.4 d |
| 15m | ~5,000 | ~52 d | 52 d (175/177 ≥ 30 d) |
| 1h | ~5,000 | ~208 d | 90 d requested, 175/177 full |

The cap is on bar *count*, so reachable history scales with bar size. **1-minute breadth
must be accrued, not fetched** — which is why `XS-7` refreshes on a schedule and treats a
missed window as data, not as a delay.

---

## Feature Vector (239 Dimensions)

| # | Category | Count | Prefix | Key features | Reference |
|---|----------|-------|--------|-------------|-----------|
| 1 | **Raw** | 10 | `raw_` | midprice, spread, microprice, depth L5/L10 | Gatheral & Oomen (2010) |
| 2 | **Imbalance** | 8 | `imbalance_` | OBI at L1/L5/L10, pressure scores | Cont, Stoikov & Talreja (2010) |
| 3 | **Flow** | 12 | `flow_` | trade count/volume 1s/5s/30s, aggressor ratio, VWAP deviation | — |
| 4 | **Volatility** | 9 | `vol_` | realized vol (1m/5m), Parkinson, Garman-Klass | Parkinson (1980), Garman & Klass (1980) |
| 5 | **Entropy** | 27 | `ent_` | permutation entropy (m=3), tick entropy, book shape, spread dispersion | Bandt & Pompe (2002), Shannon (1948) |
| 6 | **Context** | 12 | `ctx_` | funding rate + z-score, OI, premium, volume ratio | — |
| 7 | **Trend** | 15 | `trend_` | momentum, R², monotonicity, Hurst, MA crossover | Mandelbrot (1971), Jegadeesh & Titman (1993) |
| 8 | **Medium-Freq** | 16 | `mf_` | RSI, Bollinger, ATR-family at minute scale | Wilder (1978), Bollinger (2001) |
| 9 | **Illiquidity** | 12 | `illiq_` | Kyle's lambda, Amihud ratio, Roll spread, Hasbrouck | Kyle (1985), Amihud (2002) |
| 10 | **Toxicity** | 10 | `toxic_` | VPIN (10/50), adverse selection, effective/realized spread | Easley et al. (2012) |
| 11 | **Derived** | 15 | `derived_` | trend strength, regime score, toxicity-regime interaction | — |
| 12 | **Microstructure** | 5 | `micro_` | microprice deviation, tick-rule signs | Stoikov (2018) |
| 13 | **Hawkes** | 3 | `hawkes_` | self-excitation intensity, branching ratio | Bacry et al. (2015) |
| 14 | **Resilience** | 3 | `resilience_` | book refill rate after depletion | — |
| 15 | **Whale Flow** | 12 | `whale_` | net flow, momentum, intensity (1 h/4 h/24 h) | *optional* |
| 16 | **Liquidation** | 13 | `liquidation_` | risk at ±1/2/5/10 %, asymmetry, intensity | *optional* |
| 17 | **Concentration** | 15 | `top`/`conc_` | Herfindahl, Gini, Theil, top-K share | *optional* |
| 18 | **Regime** | 23 | `regime_` | absorption, divergence, churn, range position | *optional* |
| 19 | **GMM** | 8 | `regime`/`prob_` | 5-state posteriors, confidence, regime entropy | *optional* |
| 20 | **Cross-Symbol** | 3 | `cross_` | lead-lag, cross-book pressure | *optional* |
| 21 | **Heatmap** | 8 | `hm_` | depth-surface summaries | *optional* |

**157 base features** (always computed) + **82 optional** (NaN-padded when absent) = **239**.
Full manifest with formulas: [`FEATURES.md`](FEATURES.md).

> **Known documentation drift (2026-08-08):** `FEATURES.md` and the `ing-features` crate
> doc comment both still state **236**. The live contract is **239** — verified against the
> emitted Parquet schema (242 columns = 3 metadata + 239 features) and by summing
> `count_all()`. The counts in the table above are read from the Rust source, not from
> `FEATURES.md`. Reconciling the two is an open docs task; the schema is the authority.

### Feature engineering contract

```
Features::to_vec()    → always returns exactly count_all() elements
Features::names_all() → matching column names (the Parquet schema source)
Features::count_all() → to_vec().len(), enforced
```

Adding a feature category:
1. Create the struct with `count()`, `names()`, `to_vec()`.
2. Add it to `Features` in `features/mod.rs`.
3. Add it to `to_vec()`, `names_all()`, `count_all()` — NaN padding if optional.
4. Schema updates automatically via `create_schema()` in `output/schema.rs`.

> **Guardrail:** plan before any feature-vector or schema change. It ripples to every
> Parquet file and every reader.

---

## Microstructure Algorithm Library (32 Algorithms)

32 registered algorithms compute derived signals from the feature vector. Each implements
the `MicrostructureAlgorithm` ABC (`scripts/algorithms/base.py`).

```bash
nat algorithm list                              # all algorithms and their output features
nat algorithm evaluate --all                    # IC / drift across the library
nat algorithm evaluate --algorithm microprice --symbol BTC
nat backtest algorithm --algorithm weighted_ofi --symbol BTC
```

### Contract

- Register with the `@register` decorator (`scripts/algorithms/registry.py`).
- `step()` returns exactly the keys declared by `alg_features()` — no more, no less.
- Any NaN required column → NaN for every output. No silent imputation.
- Output names are prefixed `alg_`.
- Docstring carries the mathematical formulation and its reference.
- `run_batch()` defaults to row iteration; override it with vectorized numpy/pandas.
- The first `warmup` rows of `run_batch()` output are NaN-blanked automatically.
- Parameters live in `config/algorithms.toml`, never inline.

### Catalog

| Algorithm | Method | Reference |
|-----------|--------|-----------|
| `kalman_imbalance` | OU Kalman filter on L1 imbalance | — |
| `regime_gated` | Entropy-percentile gating | Bandt & Pompe (2002) |
| `weighted_ofi` | Depth-decay weighted order-flow imbalance | Cont, Kukanov & Stoikov (2014) |
| `trade_through` | Queue-depletion probability | Cont & de Larrard (2013) |
| `propagator` | Transient impact, power-law kernel | Bouchaud et al. (2004) |
| `hawkes_intensity` | Self-exciting trade arrival | Bacry, Mastromatteo & Muzy (2015) |
| `jump_detector` | Lee-Mykland nonparametric jump test | Lee & Mykland (2008) |
| `jump_detector_v2` | EVT (Gumbel) threshold, staggered bipower, directional routing | Lee & Mykland (2008) |
| `bipower_jump` | BV jump decomposition | Barndorff-Nielsen & Shephard (2004) |
| `vpin_regime` | VPIN-triggered regime switch | Easley, López de Prado & O'Hara (2012) |
| `spread_decomp` | Adverse-selection decomposition | Hendershott, Jones & Menkveld (2011) |
| `entropy_momentum` | Entropy-gated momentum | Novel |
| `surprise_signal` | Entropy regime-transition detection | Novel |
| `funding_reversion` | Funding-rate mean reversion | Crypto-specific |
| `funding_settlement` | Funding-settlement clock effects | Crypto-specific |
| `oi_divergence` | Open interest vs price divergence | — |
| `switching_ou` | Two-regime OU, Bayesian filtering | Elliott et al. (2005), Hamilton (1989) |
| `optimal_entry` | SPRT on Kalman innovation | Wald (1947), Shiryaev (1978) |
| `convolver` | Kernel-convolution feature discovery | Novel |
| `cascade_probability` | Liquidation-cascade prediction | — |
| `microprice` | HF1 microprice deviation — the maker fair-value center | Stoikov (2018) |
| `vwap_reversion` | VWAP-anchored reversion | — |
| `toxic_vwap_reversion` | VWAP reversion under a VPIN toxicity gate | Easley et al. (2012) |
| `vol_squeeze` | Volatility-compression breakout | Bollinger (2001) |
| `hierarchical_combiner` | Direction-gated two-level signal combination | — |
| `change_point_detector` | CUSUM + Bayesian online change detection | Page (1954), Adams & MacKay (2007) |
| `momentum_continuation` | Logistic-regression momentum classifier | Moskowitz, Ooi & Pedersen (2012) |
| `regime_state_machine` | 6-state threshold classifier | Hamilton (1989) |
| `mean_reversion_detector` | LightGBM false-breakout detector | Avellaneda & Lee (2010) |
| `meta_labeling` | De Prado meta-label precision filter | De Prado (2018) |
| `regime_conditioned_lgbm` | Per-regime LightGBM ensemble | Gu, Kelly & Xiu (2020) |
| `knn_retrieval` | Mahalanobis nearest-neighbour state retrieval | Cover & Hart (1967) |

`nat algorithm list` is the source of truth; the table above is a reading aid.

### Status of the library

**No algorithm is currently promoted.** The five once labelled winners —
`jump_detector`, `optimal_entry`, `funding_reversion`, `surprise_signal`, `3f_liquidity` —
were **all REFUTED** by the Q4 alpha-skeptic kill gate on 2026-07-30 and REJECTED in the
signal lifecycle. Their P&L was measured at 1.61 bps round-trip (Binance VIP9 — the wrong
venue; Hyperliquid reality is ~11 bps all-in) through a harness that applied one generic
P20/P80 entry to every candidate instead of each algorithm's own logic.

They remain registered for research. **Treat any historical Sharpe or P&L citation for them
as invalid.** The performance tables and the full mathematical derivations are preserved in
[`docs/research/ALGORITHMS.md`](docs/research/ALGORITHMS.md) as a mechanism record; the
refutation itself is `FINDINGS.md` §4.6.

### ML algorithms (wave-gated)

Implemented in waves with hard decision gates — each wave must show positive OOS alpha
before the next begins. Specs in `docs/research/new/ml_specs/`.

| Wave | Algorithm | Status |
|------|-----------|--------|
| 0 | Infrastructure (bar-level support, WelfordNormalizer) | Done |
| 1 | `change_point_detector` | Done |
| 1 | `momentum_continuation` | Done (awaiting training) |
| 2 | `regime_state_machine` | Gated |
| 2 | `mean_reversion_detector` | Done (awaiting training) |
| 2 | `meta_labeling` | Done (awaiting training) |
| 3 | `regime_conditioned_lgbm` | Done (awaiting training) |
| 3 | `knn_retrieval` | Done |
| 4 | `hmm_emissions`, `stacking_ensemble`, `online_learner` | Deferred — triggers in `DEFERRED_OVERVIEW.md` |

> **Open bug (BUG-1):** the three untrained ML algorithms keep their artifacts in
> `models/`, which is gitignored — so trained state is unauditable from a clean checkout.

**Bar-level algorithms** set `bar_level = True`; the runner calls `aggregate_bars()` before
`run_batch()` and forward-fills to tick level. No per-algorithm aggregation code.

```bash
python scripts/ml_health_check.py                    # nightly health check
python scripts/ml_rollback.py disable <algo>         # remove from trading
python scripts/ml_rollback.py rollback-model <algo>  # revert to previous model
```

---

## Process Discovery Layer (PROC)

A **process** is a reusable analytical unit with a declared contract — inputs, data level,
outputs, and a planted test written before the implementation. Where an *algorithm*
produces a tradeable signal, a *process* produces a **measurement**. The layer was
completed end-to-end on 2026-08-05 and is one of the three things that survived Q4.

```bash
nat process list                         # 15 registered processes
nat process run mi_ksg --symbol BTC      # run one on feature data
nat process results                      # past runs, indexed in nat.db
nat process show <run_id>                # one run record, provenance-stamped
nat process standing list|audit|run <name>   # standing (recurring) evaluations
```

### Registered processes

| Process | Measures |
|---|---|
| `ic_horizon` | IC across feature × horizon grids |
| `horizon_label_scan` | label construction vs horizon sensitivity |
| `mi_ksg` | Kraskov-Stögbauer-Grassberger mutual information |
| `mi_combiner` | synergy-aware MI combination (PROC-3) — synergy a greedy selector cannot see |
| `mi_stability` | MI durability, **one fold per day** rather than pooled (PROC-4) |
| `transfer_entropy` | directed information flow |
| `conditional_predictability` | predictability conditioned on regime |
| `residualize` | pure-innovation extraction; orthogonality that must survive a holdout (PROC-15) |
| `pca_combo` | principal-component feature combination |
| `spectral` | PSD, ACF, coherence, band-decomposed IC |
| `triple_barrier` | De Prado triple-barrier labelling |
| `persistence_stats` | momentum runs + band excursions (PROC-20) |
| `ml_importance` | model-based feature importance |
| `xs_rank_predictability` | cross-sectional rank-IC (XS-3) |
| `xs_persistence` | rank autocorrelation and half-life (XS-4) |

**Data levels:** `bars`, `ticks`, and — since PROC-19 — `candles`. The enumeration is
declared **once** (`VALID_DATA_LEVELS`); a test that carried its own copy had already
drifted, which is how that contract earned a single declaration.

### The FDR ledger (PROC-13)

Multiple testing is accounted **across the whole program**, not per study:
`data/processes/fdr_ledger.jsonl` records every sweep with its trial count and git SHA.

```bash
nat xs ledger                            # inspect the ledger
```

This exists because of §4.6 in miniature: winners selected from ~26 candidates with no
algorithm-level FDR or deflated-Sharpe accounting. A per-study q-value does not protect a
program that runs many studies on one window.

### Process → algorithm compiler (PROC-1)

`scripts/agent/` hosts a **refusal-first** compiler that turns a validated process result
into a registered algorithm skeleton. Refusal-first means it declines to emit anything when
the process result does not meet its declared promotion contract — the failure mode it is
designed against is a discovery loop that always produces something.

---

## Cross-Sectional Layer (XS / Class 3)

The Class-3 layer ranks the venue's **full perp universe** rather than three symbols. It is
the only branch of the current program that is fully data-independent — it runs on the
candle archive, needs no tick streak, and therefore leads.

```bash
nat xs                 # layer overview — every entry [PRELIM], nothing promoted
nat xs universe        # candle archive + L2 sampler coverage
nat xs capacity        # tradability curve (XS-5)
nat xs rank            # rank-IC vs relative forward returns (XS-3)
nat xs persistence     # rank autocorrelation half-life (XS-4)
nat xs trajectory      # standing t-stat trajectory (XS-10)
nat xs ledger          # program multiple-testing ledger (PROC-13)
```

**Everything in this layer is tagged `[PRELIM]`. Nothing is promoted.**

### The study chain, and what each one settled

| Study | Result |
|---|---|
| **XS-1** | Universe backfill: 708 series, 3.06 M candles, zero gaps; ~5,000-bar retention cap (§7.1) |
| **XS-8** | L2 half-spread sampler: universe median **1.372 bps = 17.7× BTC**; 169 of 177 pairs wider (§7.2) |
| **XS-2** | Permutation entropy carries **no** cross-sectional information at bar scale (IQR 0.0005–0.0025) — a negative that contradicts `specs/maker_system.md` §5 (§7.3) |
| **XS-3** | **Track C survives its pre-registered kill test.** `xs_vol` rank-IC −0.0690 (z −8.37), `xs_momentum` −0.0387 (z −4.56), both BH q 0.007 (§7.4) |
| **XS-4** | Only `vol` ranks persist: ρ(7 d) 0.691, half-life ~37.7 d. Momentum and Hurst decay in 1.4–1.5 d (§7.5) |
| **XS-5** | Capacity: touch notional is the wrong instrument; at 1 % of ADV, **117 pairs** support $1 k/day at ≤2 bps, only **52** support $10 k (§7.6) |
| **XS-6** | Rotation OOS: **0 of 6 configurations survive** the pre-registered criteria. It fails on **durability, not cost** (§7.7) |
| **XS-9** | Post-mortem: within-basket ρ 0.433 → 40 names = **≈2.2 effective bets**; a −0.33 beta tilt whose P&L is 0.802-correlated with a *static* low-beta-minus-high-beta position (§7.8) |
| **XS-10** | Standing trajectory: the wait now measures itself — **83 periods held, 325 needed, 242 remaining** |
| **A5** | Hysteresis bands: cost saving real and monotone, **net effect undecidable**; the apparent winner is reported and *not* adopted (§7.9) |
| **B-5a** | Wide-pair breakeven screen: reports the **indifference exponent**, not a survivor count (below) |

### Two results worth reading twice

**The signs are the finding (XS-3).** Both surviving scores are *negative*: low-volatility
pairs outperform high-volatility ones cross-sectionally, and recent winners *underperform*
— so the "momentum" score is a cross-sectional **mean-reversion** signal with its sign
inverted. That independently reproduces PROC-20's bar-scale anti-persistence result.

**The breadth was an illusion, but the signal is not (XS-9).** Daily rebalancing was
rearranging a position that was 80 % a standing factor bet — which is why IR = IC·√BR
overpredicted, why t = 0.49, and why one day carried 104 % of P&L. But the beta exposure
earns nothing (t −1.01) while the signal **survives neutralisation and sharpens** (t −5.48
vs −4.08 raw). An implementation defect, not a signal defect.

### B-5a — reporting the exponent instead of a verdict

§7.2 showed NAT has been studying the extreme tight tail of its own venue. The tempting
inference is that wide pairs cover adverse selection — and §7.2 named why that is a trap:
spreads are wide *because* makers price inventory and toxicity into them, so adverse
selection should scale with the spread.

So B-5a emits no verdict. Parameterising `E[adverse|fill](h) = A_btc·(h/h_btc)^β` and solving
for indifference gives, per pair:

```
  β* = ln( (h + rebate) / A_btc ) / ln( h / h_btc )
```

with β = 0 the optimistic reading and β = 1 the pessimistic one. A pair at the universe
median has **β\* = 0.69**: it survives if and only if adverse selection scales more slowly
than h^0.69. That is falsifiable by one tick-data measurement on one wide pair — which is
what B-5b is for. Reporting β\* rather than a count is the difference between a screen and
a claim, and a test asserts the summary carries no pooled survivor count at all.

Capacity is the second blade and cuts the other way: the widest pairs are nearly empty
(XAI 12.9 bps on $20), so admission runs through `xs.capacity.admit` — XS-5's floors, not a
second copy that can drift.

---

## Autonomous Research Agents

Four agents continuously generate, test, and validate hypotheses across timeframes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MetaAgent (Orchestrator)                         │
│   Cross-agent budget allocation · Correlation dedup · Risk parity       │
├─────────────────────┬───────────────────┬───────────────────────────────┤
│  MicrostructureAgent│ MediumFreqAgent   │ MacroAgent                    │
│  Tick-level (1-10s) │ 1min-1h signals   │ 1h-24h signals                │
│  5-gate replication │ 4-gate replication│ 4-gate replication            │
│  6 generators       │ MF generators     │ Macro generators              │
└─────────────────────┴───────────────────┴───────────────────────────────┘
```

```bash
nat agent {start,stop,once,status,queue,registry,graveyard,report,dashboard}
nat mf-agent {start,stop,once,status,queue,registry,graveyard,report}
nat macro-agent {start,stop,once,status,queue,registry,graveyard,report}
nat meta-agent {start,stop,once,status,portfolio,correlation,budget,report}
```

### Consolidated daemon architecture

All agents share `ResearchAgent` (ABC, `scripts/agent/base.py`), which owns the full cycle
loop and state machine, generator dispatch (lazy import via `generator_module_prefix`), FDR
control, hypothesis chaining, promotion logic, and structured research-output emission.
Each subclass is a thin ~80–110 LOC file overriding config attributes and `create_runner()`.

### Agent state machine

```
Per-cycle:
  MANIFEST ──▶ GENERATE ──▶ ADAPTIVE IC ──▶ EXECUTE (budget: 10 or 90 min)
  ──▶ FDR control (BH q=0.05) ──▶ STRUCTURED OUTPUT
  ──▶ MONITOR (decay + promotion) ──▶ SLEEP

Per-hypothesis:
  SETUP ──▶ DISCOVERY (IC+dIC) ──▶ COST ──▶ TEMPORAL ──▶ SYMBOL ──▶ CORRELATION
    │            │                   │          │           │            │
    ▼            ▼                   ▼          ▼           ▼            ▼
  ABORT      GRAVEYARD           GRAVEYARD  GRAVEYARD  GRAVEYARD    GRAVEYARD
             (no_effect)         (cost_killed)         (no_repl)   (redundant)
```

### Five-gate replication protocol

Every hypothesis must independently pass five gates before registration — false-discovery
control under multiple testing (Harvey, Liu & Zhu, 2016).

**Gate 1 — Discovery (IC + dIC).** Spearman rank IC ≥ adaptive threshold, dIC ≥ 0.05:

```
                6 × Σ(dᵢ²)
  IC  =  1 − ─────────────────       dᵢ = rank(signalᵢ) − rank(returnᵢ)
                n × (n² − 1)
```

**Gate 2 — Cost.** Gross return per trade ≥ 0.1 bps. All costs via `load_costs()` →
`config/costs.toml`. Never a hardcoded fee.

**Gate 3 — Temporal replication.** IC holds on ≥ 1 additional date (White, 2000).

**Gate 4 — Symbol replication.** IC holds on ≥ 1 other symbol (Guéant et al., 2012).

**Gate 5 — Correlation deduplication.** max ρ < 0.7 against every registry signal.

Plus BH FDR control (q = 0.05) at the end of each cycle.

### Adaptive IC threshold

The acceptance threshold rises as the registry accumulates strong signals, so marginal
signals cannot enter a strong registry:

```
  min_ic(t) = max( floor, median{ ICᵢ : i ∈ R(t) } × 0.8 )        floor = 0.10
```

### IC decay monitoring

```
  if IC_rolling(s) < IC_discovery(s) × 0.5   for 14 consecutive cycles:
      status(s) ← retired, reason ← ic_decay
```

### Six hypothesis generators

| Generator | Schedule | Strategy | Reference |
|-----------|----------|----------|-----------|
| **Systematic** | Nightly | Exhaustive feature × gate × threshold search | — |
| **Spectral** | Daily | PSD slope / OU half-life anomaly detection | Mandelbrot & Van Ness (1968) |
| **Regime** | Daily | IC improvement after HMM state transitions | Rabiner (1989) |
| **Cross-Asset** | Weekly | Lead-lag at the 68 s coherence frequency | Priestley (1981) |
| **Recycler** | Weekly | Re-examine the graveyard on new data | — |
| **IT Discovery** | Daily | MI/CMI/II-driven hypotheses from the IT engine | Kraskov et al. (2004) |

Budget allocation uses a Beta-prior Thompson bandit (Thompson, 1933):
`weight(g) = (successes + 1) / (attempts + 2)`.

### Structured research output

Per hypothesis, a JSON record lands in `data/research/hypotheses/{id}.json` — claim,
generator, status, per-gate results (metric, threshold, p-value), LaTeX derivation,
features, regime gate, timestamps. Cycle summaries in `data/research/cycles/{cycle_id}.json`.

### Computation cache

Deterministic commands cached under `SHA-256(canonical_args)`, 7-day TTL. Measured: 85 %
hit rate, 56 % cycle-time reduction.

---

## Alpha Pipeline

Nine steps with PASS/WEAK/FAIL quality gates between each.

```bash
nat alpha pipeline-start                 # launch the full pipeline
nat alpha pipeline-resume                # resume from checkpoint (--force-gate to override)
nat alpha pipeline-status                # current step + gate verdicts
nat alpha pipeline-gates                 # detailed gate report with metrics
nat alpha pipeline-step 3                # run one step

# individual steps also run standalone:
nat alpha combine|size|validate|regime|multi-freq|portfolio|paper|deploy
```

| Step | Module | Function |
|------|--------|----------|
| 1. Screening | `screener.py` | Feature screening with FDR control |
| 2. Combining | `combiner.py` | Signal combination and weighting |
| 3. Sizing | `position.py` | Position sizing (Kelly / risk parity) |
| 4. Validating | walk-forward | Out-of-sample stability |
| 5. Regime | `regime_filter.py` | HMM / entropy regime conditioning |
| 6. Multi-Freq | `multi_freq.py` | Multi-frequency integration |
| 7. Portfolio | `portfolio.py` | Portfolio assembly and allocation |
| 8. Paper | `paper_trader.py` | Paper-trading simulation |
| 9. Deploy | `deployer.py` | Deployment readiness |

Config: `config/alpha.toml`. **Gate thresholds G1–G8 are imported from ROADMAP, never
invented** — introducing a new number is a guardrail violation, not a tuning choice.

---

## Information-Theoretic Engine

`scripts/it_engine/` — mutual-information estimation, entropy conditioning, and cost-aware
feature selection across the full feature vector.

```bash
nat it-engine start --symbol BTC               # live mode (Redis pub/sub)
nat it-engine start --symbol BTC --offline     # offline mode (parquet)
nat it-engine start --symbol BTC --dry-run     # single cycle
nat it-engine status --symbol BTC              # MI rankings
nat it-engine stop --symbol BTC
```

| Estimator | Formula | Purpose |
|-----------|---------|---------|
| **KSG MI** | Kraskov-Stögbauer-Grassberger k-NN (k=5) | I(f; r) |
| **Conditional MI** | I(f; r \| H) via KSG in joint/marginal spaces | the proper IT formulation of entropy gating |
| **Interaction Info** | II(f;r;H) = I(f;r\|H) − I(f;r) | + synergy (gating helps), − redundancy |
| **Linear TE** | TE = ½·log(σ²_reduced / σ²_full) | causal information flow |
| **Cost threshold** | I_min = −½·log₂(1 − (fee/σ_r)²) | minimum MI to overcome costs |

**Greedy selection:** start at `f* = argmax_f I(f; r_k)`; step to
`f_next = argmax_f I(f; r_k | S)`; stop when the marginal gain falls below `I_min(k)`. The
`it_discovery` generator feeds cost-viable features into the agent hypothesis queue.

---

## Backtesting & OOS Validation

```bash
nat backtest --symbol BTC                    # generic backtest
nat backtest algorithm --algorithm NAME      # algorithm-specific
nat backtest ml --symbol BTC                 # ML prediction backtest
nat backtest list                            # available experiments
nat gauntlet run                             # multi-day OOS sweep across all algorithms
nat gauntlet report                          # latest gauntlet report
nat oos --window N                           # longitudinal OOS over trailing gauntlet P&L
```

### Costs — one source of truth

**All costs load via `load_costs()` (`scripts/costs.py` → `config/costs.toml`).** Never
hardcode a fee or a slippage figure anywhere in the stack. This is a hard guardrail, and it
exists because violating it is precisely what invalidated the 2026-05 sweep: it priced
trades at 1.61 bps round-trip (Binance VIP9) when Hyperliquid reality is ~11 bps all-in.

`config/costs.toml` carries the venue's real ladders, verified against the fee docs:
the HYPE-staking discount tier (`[hyperliquid_staked]`, X-1) and the maker
volume/rebate ladder (`[hyperliquid_maker_tiers]`, COST-5). COST-8 removed the last
hardcoded 0.2 bps rebate — the most load-bearing unvalidated number in the stack, worth
~1.7 bps/fill — so the maker preset now reads the SSOT and no longer inverts the rebate
sign.

> ⚠️ **`nat oos30` runs the historical 5-algorithm walk-forward.** All five algorithms it
> exercises were refuted on 2026-07-30 and REJECTED in the lifecycle. It is retained to
> reproduce the historical record, **not** as a validation path. Do not read its output as
> a current result.

### Deflated Sharpe

The G4 gate deliberately keeps a lenient `std_max` DSR; the full canonical DSR is
**reporting-only** in `nat oos --window`. Switching the gate is blocked on ~90 clean OOS
days — the gate and the report answer different questions and are not interchangeable.

### Discovery orchestrator

```bash
nat discovery {start,once,status,stop}
```

Cycle: DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING → ALPHA_PIPELINE → REPORTING →
SLEEPING, with gates at each step. Child scripts run via subprocess, not import, to prevent
OOM.

---

## Analysis & Profiling Tools

| Command | Function | Output |
|---------|----------|--------|
| `nat spannung --symbol BTC` | Feature × horizon IC grid search | `reports/spannung/spannung_{sym}.json` |
| `nat spannung regime --symbol BTC` | Quintile regime screener | `reports/spannung/regime_screen_{sym}.json` |
| `nat spannung spectral --symbol BTC` | PSD, ACF, coherence, band-decomposed IC | `reports/spannung/spectral_{sym}.json` |
| `nat spannung backtest --symbol BTC` | Cost-aware backtest with regime gating | `reports/spannung/backtest_{sym}.json` |
| `nat profile scalp --symbol BTC` | Walk-forward feature profiler | `reports/profiler/profile_{sym}.json` |
| `nat cluster hmm` | Gaussian HMM fitting (Baum-Welch EM) | `reports/hmm_fit.json` |
| `nat validate skeptical` | 20+ statistical tests (FDR, bootstrap, permutation) | `reports/skeptical_validation/` |
| `nat validate regression` | Skeptical regression battery (10 tests) | JSON report |
| `nat data validate [<file>]` | Schema / quality / continuity / NaN / ranges | PASS/WARN/FAIL, `--json`, nonzero exit on FAIL |
| `nat viz render --tf {1m,5m,15m} [N]` | Paged PNG viewer; `--features` scopes the panel grid | `reports/figures/snapshots/*.png` |
| `nat viz3d` / `nat mesh --tf {1m,5m,15m} [N]` | Interactive 3D feature surface (Plotly) | `reports/figures/mesh/*.html` |
| `nat scan --symbol BTC` | Signal discovery scan | JSON report |
| `nat macro --symbol BTC` | Macro regime analysis | JSON report |
| `nat kalman analysis --symbol BTC` | Kalman filter analysis | `reports/kalman/` |

### Cluster pipeline

```bash
nat cluster {analyze,gmm,all,quality,explore,hmm}
```

Unsupervised regime discovery: loader, preprocess, cluster, reduce (PCA/UMAP/t-SNE),
characterize, hierarchy, transitions, online streaming, visualization. Quality via
silhouette, Davies-Bouldin, Calinski-Harabasz.

---

## Lifecycle, Promotion & Risk Automation

The path from a validated signal to live capital is automated end-to-end behind a single
human gate and hard risk controls. Each stage is an independent daemon sharing the same
conventions: pidfile + heartbeat + `health` subcommand + graceful SIGTERM, **dry-run by
default**, and a Docker Compose service with a healthcheck. All gate thresholds are
**imported, not invented**.

### Signal lifecycle

`scripts/signal_lifecycle.py` is the single source of truth for promotion state, persisted
to `nat.db` (`signal_lifecycle` + `lifecycle_history`, shared migration framework in
`scripts/data/state.py`). Every transition is provenance-stamped with a git SHA and
recorded; illegal transitions raise.

```
DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING → LIVE → MONITORING → RETIRED   (+ REJECTED)
```

`approve()` (APPROVAL_PENDING → LIVE) is the **sole human gate**.

```bash
nat lifecycle status|list|history|approve|reject|seed
```

### Promotion daemon

`scripts/promotion_daemon.py` drives the lifecycle: a data-sufficiency + ≥7-clean-day
guard, then rigorous **G4** (walk-forward + deflated Sharpe) → VALIDATED, paper trading →
PAPER_TRADING, and **G8** (5 criteria over 14 days) → APPROVAL_PENDING. It never
auto-promotes to LIVE.

```bash
nat promotion status|once [--dry-run]|start|stop
```

### Kill-switch daemon

`scripts/risk/kill_switch.py` polls realized PnL/IC every 60 s and halts on ROADMAP Step-9
thresholds: daily loss > 1 % → `halt_24h`, weekly DD > 2 % → `halt_review`, monthly DD > 5 %
→ `kill_strategy` (retires the signal), IC < 0 for 5 days → `halt`. Publishes
`data/risk/halt_state.json` — the IPC the bridge reads before every cycle — and a Telegram
page within 60 s.

```bash
nat risk status|resume [--confirm]|start|stop
```

### Gap-alert daemon

`scripts/ops/gap_alert.py` pages via Telegram within minutes when ingestion stalls — the
real-time complement to the next-day report. Freshness = newest `*.parquet`/`*.parquet.tmp`
mtime; alerts once on gap-open and once on recovery. Read-only, so it is safe alongside a
streak-frozen ingestor.

```bash
nat gap status|check|start|stop
```

### Signal bridge

`scripts/execution/signal_bridge.py` reads LIVE signals from the lifecycle, **checks
`halt_state.json` before every cycle (cannot be skipped)**, sizes via `meta_portfolio` risk
parity (never independent per-signal), logs fills to `data/execution/fills_*.jsonl` for
fill-conditional IC, and rolls up `data/execution/daily_pnl.json` — the file the
kill-switch reads back, closing the loop. **Dry-run by default**; live requires an explicit
`mode=live` plus a healthy kill-switch.

```bash
nat bridge status|once [--dry-run]|start|stop
```

> **No live capital before G8 and a healthy kill-switch.** This is not configurable.

### Execution primitives

`scripts/execution/rebalance.py` provides hysteresis bands (Constantinides no-trade
boundary, "edge" and "full" modes) and TWAP/VWAP slicing. **Slicing ships as a primitive
with no performance claim** — it exists to reduce market impact, and NAT's cost model has
no impact term, so it measures as exactly zero. That is not a win left on the table; it is
unpriceable until X-3 has fill data.

### Approval-evidence visualization

```bash
nat viz paper <signal>      # cumulative P&L, IC decay, the 5-criterion G8 scorecard
nat viz portfolio --tab N   # P&L / exposure / cross-signal correlation (<0.35) / risk
nat viz features|algorithm  # per-feature and per-algorithm terminal views
```

### Observability

A pure-stdlib Prometheus exporter (`scripts/monitoring/metrics_exporter.py`, `:9094`) turns
lifecycle/paper/live-PnL state — SQLite and JSON, which Grafana cannot scrape directly —
into gauges (`nat_lifecycle_signals{state}`, `nat_paper_sharpe{signal}`,
`nat_live_cum_pnl_pct`, …), feeding three auto-provisioned Grafana dashboards.

---

## Web Dashboard & API

### Next.js frontend (port 3001)

```bash
cd web && npm run dev
```

| Page | URL | Description |
|------|-----|-------------|
| Homepage | `/` | System overview |
| Hypothesis Explorer | `/explorer` | Browse hypotheses, filter by status/agent/generator |
| Hypothesis Detail | `/explorer/{id}` | Gate results, math derivation, feature data |
| IC Heatmap | `/heatmap` | Feature × horizon IC matrix |
| Graveyard | `/graveyard` | Failed hypotheses with failure analysis |
| Research Network | `/network` | Graph of hypothesis relationships |
| Signal Table | `/signals` | Active validated signals |
| Math Viewer | `/math` | LaTeX rendering |

### Rust REST API (port 3000)

```bash
nat api start
```

| Endpoint | Description |
|----------|-------------|
| `/api/research/hypotheses` | Paginated list; `?agent=`, `?generator=`, `?status=` |
| `/api/research/hypotheses/:id` | Full detail (gates, math, thresholds) |
| `/api/research/cycles` | Cycle summaries; `?agent=` |
| `/api/research/signals` | Registered signals only |
| `/api/research/stats` | Aggregate counts by status, agent, generator |
| `/api/research/heatmap` | Feature × horizon IC matrix |
| `/health` | Health check |

Config: `NAT_RESEARCH_DIR` selects the research data directory.

### Agent dashboard (port 8060)

```bash
nat agent dashboard        # stdlib HTTP, dark theme, 10s auto-refresh
```

Panels: agent status, registry table, (signal × gate) IC heatmap, graveyard, queue,
generator stats, cache statistics, and a lifecycle tab (`/api/lifecycle`).

---

## Config Swarm & Evolutionary Optimization

### Tier 1 — Continuous cloud + observability

Docker stack with Prometheus (5 s scrape, 90 d retention), Grafana, Caddy HTTPS reverse
proxy, PostgreSQL state persistence. All services health-checked.

### Tier 2 — Config swarm

A shared ingestor writes Parquet once; N evaluators read and score configs in parallel over
a ~35-dimensional parameter space.

```bash
nat swarm run --instances 8          # parallel config evaluation
nat swarm {status,results,best,generate}
```

| Component | Function |
|-----------|----------|
| `parquet_reader.py` | Time-windowed reads from the latest available date |
| `config_generator.py` | Random, grid, or Optuna trial generation |
| `evaluator.py` | Algorithms → ensemble → fitness (Sharpe, IC, drawdown, turnover) |
| `orchestrator.py` | ProcessPoolExecutor, ranking, export |

Config: `config/swarm_ranges.toml` (15 algo + 5 ensemble + 8 trading + 7 feature-selection
params).

### Tier 3 — Evolutionary optimization (Optuna)

```bash
nat evolve start --trials 5000 --sampler cma    # CMA-ES over the continuous 35D space
nat evolve start --sampler nsga2                # multi-objective Pareto
nat evolve {status,best,pareto,export}
```

**Guard rails:** walk-forward OOS evaluation (train/test never overlap); IS/OOS overfit
detection (ratio > 3.0 penalized); deflated Sharpe (Bailey & López de Prado, 2014); hard
constraints (signal count > 50/day, turnover < 100/day, OOS Sharpe > 0); MedianPruner for
early stopping.

SQLite for single-machine, PostgreSQL for distributed studies. Dashboard on `:8070`.

---

## Other Modules

| Module | Path | Purpose |
|---|---|---|
| **EAMM** | `scripts/eamm/` | Entropy-adaptive market making — simulator, features, labels, training, regime analysis (`nat eamm run|regime|backtest`) |
| **Polymarket** | `scripts/polymarket/` | Prediction-market client, scanner, probability model, edge detector, backtest |
| **Hypothesis suite** | `rust/ing/src/hypothesis/` | H1–H5 structural tests; decision matrix in `final_decision.rs` (0–1 pass = NOGO, 2–3 = PIVOT, 4–5 = GO) |
| **Experiments** | `scripts/experiment/` | Experiment monitoring (`nat exp {start,status,analyze,dashboard,tunnel}`) |
| **Tournament** | — | Head-to-head algorithm comparison (`nat tournament`) |

---

## The `nat` CLI

`nat` is the primary interface: **340 commands across 72 groups**.

**[`docs/commands.md`](docs/commands.md) is the full reference — and it is generated, not
written.** `scripts/ops/gen_commands_doc.py` regenerates it from the live argparse tree.
Do not hand-edit it, and do not restate it here: it was hand-maintained until 2026-08-07,
by which point 26 groups were missing and the headline count was stale by 80. A reference
that disagrees with the CLI is worse than no reference, because it is trusted.

```bash
nat help                     # curated usage guide
nat commands                 # structured list of every command
nat commands --json          # machine-readable source of truth
```

Maturity tags (`[PRELIM]`, …) appear where a group declares one. **Absence of a tag is not
a claim of maturity.**

### The commands you actually need first

```bash
# ingestion
nat doctor · nat start · nat stop · nat status · nat log · nat gap status

# data
nat data validate · nat data schema · nat viz render --tf 15m · nat viz3d

# discovery
nat process list · nat process run <name> --symbol BTC · nat xs rank · nat xs ledger

# agents
nat agent start · nat agent status · nat agent dashboard

# promotion
nat lifecycle status · nat promotion status · nat risk status · nat bridge status

# build & test
nat build · nat test · nat test validate
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/ing.toml` | Ingestor: WebSocket URL, symbols, emission interval (100 ms), output format |
| `config/costs.toml` | **Fees, rebates, slippage, venue tier ladders — single source of truth** |
| `config/agent.toml` | Agents: cycle interval, gate thresholds, FDR q, generators, decay, promotion |
| `config/alpha.toml` | Alpha pipeline: gate thresholds G1–G8, step parameters, symbols |
| `config/pipeline.toml` | Pipeline orchestration: ingestion duration, analysis thresholds |
| `config/discovery.toml` | Discovery orchestrator: sweep config, training, backtesting |
| `config/algorithms.toml` | Per-algorithm constructor kwargs |
| `config/swarm_ranges.toml` | Swarm parameter ranges (~35D) |
| `config/it_engine.toml` | IT engine: buffer size, KSG k, horizons, cost thresholds |
| `config/kalman.toml` | Kalman filter parameters |
| `config/hypothesis_testing.toml` | Hypothesis test parameters (H1–H5) |
| `config/symbols.toml` | Tradeable symbol list |
| `config/llm.toml` | LLM client: model, endpoint, API key reference |
| `config/risk.toml` | Kill-switch: poll interval, paths (thresholds imported from `alpha.toml`) |
| `config/ops.toml` | Gap-alert: gap threshold, poll interval, watched data dirs |
| `config/promotion.toml` | Promotion daemon: poll interval, ≥7-clean-day guard, paths |
| `config/execution.toml` | Signal bridge: mode (dry-run default), account value, paths |
| `config/monitoring.toml` | Metrics exporter: port, refresh interval, data paths |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Rust logging level (`info`, `debug`, …) |
| `REDIS_URL` | Redis connection URL |
| `ING_DASHBOARD_ENABLED` | Enable the ingestor dashboard |
| `ING_PROMETHEUS_ADDR` | Bind address for the metrics endpoint |
| `NAT_RESEARCH_DIR` | Research data directory for the API |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alert routing |

---

## Testing

The full suite is **4,408 passing, 0 failed** as of 2026-08-07.

```bash
# Rust
nat test                                # cargo test --package ing
nat test verbose                        # with --nocapture
cd rust && cargo test -- test_name      # single test

# Python
pytest scripts/tests/                    # main suite
pytest scripts/algorithms/tests/ -v      # algorithm unit + integration + smoke
pytest scripts/tests/test_agent_*.py     # agent tests
pytest scripts/tests/test_signal_lifecycle.py scripts/tests/test_lifecycle_cli.py
pytest scripts/tests/test_bar_level_dispatch.py     # algorithm dispatch/conformance

# Live validation & smoke
nat test validate                        # 4 binaries against the live API
nat 15m                                  # 15-minute live smoke test
nat 15m offline                          # offline smoke test on parquet
nat validate skeptical                   # 20+ statistical tests

# Data sufficiency (before ML training)
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features
```

### Testing doctrine

Two rules are non-negotiable, and both were bought with real failures:

1. **Planted (synthetic) test before any real-data use.** Level-1 planted tests caught
   three estimator bugs that no amount of real-data eyeballing found. The test is written
   *first*, and it fails first.
2. **Real-parquet smoke before commit.** "Looks right" is not "is right"; correctness in
   alpha work is enforced by gates, not by inspection.

A conformance test that silently validates a subset is worse than one that fails — a real
incident here: XS processes were `@register`-ed but never imported, so the registry
conformance test iterated a set that excluded the very units it should have validated.

---

## Docker

```bash
nat docker build · nat docker up · nat docker down · nat docker logs
```

| Service | Port | Description |
|---------|------|-------------|
| **redis** | 6379 | Pub/sub, caching (256 MB max, LRU) |
| **ingestor** | 8080 | Market data collection + real-time dashboard |
| **api** | 3000 | REST/WebSocket endpoints |
| **alerts** | — | Telegram alert service |
| **kill-switch** | — | Risk kill-switch daemon |
| **gap-alert** | — | Data-gap alert daemon |
| **promotion** | — | Signal promotion daemon |
| **signal-bridge** | — | LIVE execution daemon (`depends_on` kill-switch healthy) |
| **metrics-exporter** | 9094 | Lifecycle/paper/live-P&L state → Prometheus gauges |
| **web** | 3001 | Next.js frontend |
| **prometheus** | 9090 | Metrics collection (90-day retention) |
| **grafana** | 3002 | Auto-provisioned dashboards (anonymous access) |
| **postgres** | 5432 | State persistence + Optuna study storage |
| **optuna-dashboard** | 8070 | Optimization history, parameter importance, Pareto fronts |
| **caddy** | 80/443 | HTTPS reverse proxy with auto-TLS |

**Ingestor metrics:** `ing_features_emitted_total`, `ing_errors_total`,
`ing_feature_latency_seconds`, `ing_update_latency_seconds`.
**Business-state metrics** (`:9094`): `nat_lifecycle_signals{state}`,
`nat_paper_sharpe{signal}`, `nat_paper_max_drawdown_bps{signal}`, `nat_live_cum_pnl_pct`.

Grafana dashboards: **NAT Overview**, **NAT Lifecycle Funnel**, **NAT Paper Performance**,
**NAT Live P&L**.

---

## Multi-Machine Setup

The Rust ingestor runs on a separate machine (`su-35`) for low-latency collection. Agents
and analysis run on the research machine, reading Parquet from `data/features/`.

```
┌──────────────────┐         ┌──────────────────────┐
│  su-35 (ingestor)│         │  research machine    │
│                  │  rsync  │                      │
│  nat start       │ ──────▶ │  nat process run     │
│  Hyperliquid WS  │  data/  │  nat agent start     │
│  Parquet output  │         │  nat xs rank         │
└──────────────────┘         └──────────────────────┘
```

> **Hard rule: zero su-35 contact until the clean-data streak completes.** The ingestor is
> the critical dependency for the streak milestone; deploy wiring and validation to the T0b
> cloud box instead (`nat deploy cloud <ip> --dry-run` first). Verify streak state locally
> with `nat gap status` — it reads local files and does not touch su-35.

Systemd units for the cloud box live in `deploy/`. Note `StartLimitIntervalSec` belongs in
`[Unit]`, not `[Service]` — placed wrongly it silently fails to disable the restart limiter.

---

## Project Structure

```
nat/
├── nat                            # Unified CLI (340 commands, 72 groups)
├── FEATURES.md                    # Feature manifest with formulas
├── CLAUDE.md                      # Architecture guide + guardrails
│
├── rust/                          # Cargo workspace (ing-types → ing-features → ing; + api)
│   ├── ing-types/src/             # Shared types (OrderBook, TradeBuffer, MarketContext)
│   ├── ing-features/src/          # Feature computation (21 categories, 239 features)
│   ├── ing/src/                   # Ingestor binary
│   │   ├── main.rs                # tokio::select! biased loop
│   │   ├── ws/                    # Hyperliquid WebSocket client
│   │   ├── state/                 # OrderBook, TradeBuffer, MarketContext
│   │   ├── output/                # Parquet writer (Arrow, hourly rotation)
│   │   ├── dashboard/             # Real-time monitoring (Axum WS)
│   │   ├── hypothesis/            # H1–H5 hypothesis tests
│   │   ├── ml/                    # GMM regime classification
│   │   └── bin/                   # Validation binaries
│   └── api/src/                   # API crate (Axum, port 3000)
│
├── scripts/                       # Python research layer
│   ├── processes/                 # PROC layer — 15 registered processes + registry
│   ├── xs/                        # Class-3 cross-sectional layer
│   │   ├── features.py            # per-pair scores (XS-2)
│   │   ├── rotation.py            # top-k rotation (XS-6)
│   │   ├── capacity.py            # tradability / admission floors (XS-5)
│   │   ├── breakeven.py           # wide-pair indifference exponent (B-5a)
│   │   └── trajectory.py          # standing t-stat trajectory (XS-10)
│   ├── agent/                     # Autonomous research agents
│   │   ├── base.py                # ResearchAgent ABC (cycle loop, FDR, chaining)
│   │   ├── daemon.py / mf_daemon.py / macro_daemon.py / meta_daemon.py
│   │   ├── runner.py / mf_runner.py / macro_runner.py
│   │   ├── hypothesis_queue.py    # SQLite priority queue
│   │   ├── research_output.py     # Structured JSON emitter (LaTeX math)
│   │   ├── cache.py               # SHA-256 computation cache (7-day TTL)
│   │   ├── meta_portfolio.py      # Risk-parity portfolio optimization
│   │   └── generators/            # systematic, spectral, regime, cross_asset, recycler, it_discovery
│   ├── algorithms/                # 32 microstructure algorithms + registry + runner
│   ├── alpha/                     # 9-step alpha pipeline
│   ├── data/                      # Data utilities, fetch_candles.py, fetch_l2.py, state.py
│   ├── execution/                 # Signal bridge, rebalance (bands/slicing), Hyperliquid client
│   ├── risk/                      # Kill-switch daemon
│   ├── ops/                       # Gap-alert daemon, gen_commands_doc.py
│   ├── monitoring/                # Prometheus metrics exporter
│   ├── it_engine/                 # Information-theory engine
│   ├── cluster_pipeline/          # Unsupervised clustering
│   ├── eamm/ · polymarket/ · swarm/ · experiment/ · backtest/ · analysis/ · viz/
│   ├── signal_lifecycle.py        # Lifecycle state machine (nat.db, provenance-stamped)
│   ├── promotion_daemon.py        # Auto-promotion daemon (G4/G8)
│   ├── provenance.py              # git-SHA + data-fingerprint stamping
│   └── tests/                     # Test suite
│
├── web/                           # Next.js dashboard
├── config/                        # TOML configuration
├── data/                          # features/ · candles/ · processes/ · research/ · risk/ · execution/
├── docs/                          # OBJECTIVE, METHODOLOGY, PLAN, TASKS, GLOSSARY, contracts/, specs/, research/
├── reports/                       # Generated reports & analysis
├── deploy/ · docker/ · docker-compose.yml
├── models/ · notebooks/ · logs/
```

---

## Key Findings

The complete record is [`docs/research/FINDINGS.md`](docs/research/FINDINGS.md). Findings
are point-in-time; each block there states its test window. What follows is a reading path.

### The central result

Order-book imbalance carries **IC ≈ 0.45 at 1–5 s** on all three symbols, 24/7, in both
volatility regimes, bootstrap CI width ~0.02. It is not noise and not symbol-specific.

**And it cannot be monetized by any fill model tested.** Taker: the 1–5 s move (0.5–2 bps)
is smaller than cost (~11 bps RT). Maker: conditioning on the directionally-correct
mid-cross fill collapses IC from ~0.45 to ~0.03 / −0.06 / −0.03. **Adverse selection is
structural, not a tuning problem.** This is the project's central result and its binding
research question (gate Q5).

### Spannung research arc

| Finding | Evidence | Implication |
|---------|----------|-------------|
| OBI predicts 5 s returns | IC 0.19, 100 % sign consistency across 5 folds | Structural, not overfit |
| `ent_book_shape` is the #1 regime gate | Independently #1 on BTC, ETH, SOL | Universal microstructure property |
| Signal replicates cross-symbol | KEEP on all 3; IC ordered by liquidity (SOL > ETH > BTC) | Genuine LOB effect |
| Brown-noise universality | PSD slope ≈ −1.85 on all 3 symbols | Fractional Brownian microstructure |
| 68 s coherence dominance | Single peak at 0.015 Hz | Natural market-making cycle |
| OU half-life orders by liquidity | BTC 7.3 s > ETH 5.3 s > SOL 3.3 s | Per-symbol refresh rates |
| `_last` replicates, `_mean`/`_std` don't | Instantaneous = KEEP, aggregated = DROP | Time-domain spectral confirmation |
| Regime gating measurably works | `ent_book_shape` lifts imbalance IC +22 % (low-entropy quintile) | Conditioning is real |
| Orthogonalization does **not** survive a holdout | 152-episode sweep; the control refuted the prior day's reading | A full-sample property, not a forward one |
| Bar-scale momentum is **anti-persistent** | PROC-20; no band cell clears the bar | Reversion, not continuation |

### The refutation record

| § | Date | Result |
|---|---|---|
| **4.6** | 2026-07-30 | **Q4 kill gate: all five "winners" REFUTED, 5/5 KILL.** Wrong-venue cost tier (1.61 bps Binance VIP9) + a harness that never ran each algorithm's own entry logic. `jump_detector`'s c = 3.0 threshold fires ~13,900×/day — a noise filter, not jump detection. `optimal_entry`'s SPRT logic was never executed. `funding_reversion` had n_eff ≈ 84 in a one-sided funding regime. `surprise_signal` drew 87.6 % of its edge from a single day. |
| **4.9** | 2026-07-31 | Touch-maker experiment, pre-registered, multi-day: **all 8 cells FAIL** on day-consistency and concentration |
| **7.3** | 2026-08-07 | Permutation entropy does not rank the universe — a clean negative that contradicts the spec |
| **7.7** | 2026-08-07 | Rotation OOS: **0 of 6** configurations survive their pre-registered criteria |

### The maker line

| § | Date | Result |
|---|---|---|
| **4.7** | 07-30 | Touch-joined postings are marginally +EV: capture 0.278 bps (rebate-carried) vs adverse 0.22–0.26 → EV +0.01–0.04 bps/posting. **The rebate, not the spread, carries maker economics.** |
| **4.7** | 07-30 | The A4 EV gate works as a filter: flips always-on touch per-fill **−1.66 → +0.67** (fills cut 55×) |
| **4.8** | 07-30 | Wide Avellaneda-Stoikov spreads are negative: −1.5 to −1.9 bps/fill, price-through-dominated. Quote at the touch; width is never derived. |
| **4.10** | 08-03 | **X-1: the maker line is fee-tier-invariant.** Staking discounts apply to fees *paid*, not to maker rebates — 179 day-symbol episodes re-priced, **no cell flips**. |
| **4.11** | 08-04 | **COST-5: zero fees are not free money.** Breakeven maker rate = E[adverse\|fill] − half-spread = **+0.144 bps (bid) / +0.159 (ask)**. At BTC's touch the half-spread (0.083 bps) is ~⅓ of adverse selection (0.228/0.242), so a zero-fee quote is still ~0.08 bps/posting under water. |
| **7.2** | 08-07 | Universe median half-spread **1.372 bps — 17.7× BTC**; 169 of 177 pairs wider. NAT has been studying the extreme tight tail of its own venue. |

### The cross-sectional line

| § | Date | Result |
|---|---|---|
| **7.1** | 08-07 | Candle universe: 708 series, 3.06 M candles, **zero gaps**; ~5,000-bar retention cap |
| **7.4** | 08-07 | **XS-3: Track C survives its kill test.** `xs_vol` rank-IC −0.0690 (z −8.37), `xs_momentum` −0.0387 (z −4.56), BH q 0.007. **Both signs negative.** |
| **7.5** | 08-07 | Only `vol` ranks persist (ρ₇d 0.691, half-life ~37.7 d); momentum and Hurst decay in ~1.4 d |
| **7.6** | 08-07 | Capacity: at 1 % of ADV, 117 pairs support $1 k/day at ≤2 bps; only 52 support $10 k |
| **7.7** | 08-07 | Rotation refuted — but **cost is not the killer**, and that was a validated prediction: turnover 0.17–0.49 against a max of 2.0, costs 1–2.7 % against 8.5 % gross |
| **7.8** | 08-07 | 40 names = **≈2.2 effective bets**; beta earns nothing (t −1.01) while the signal sharpens under neutralisation (t −5.48) |
| **7.9** | 08-07 | Hysteresis bands: cost saving real and monotone, **net effect undecidable** — the apparent winner is reported and not adopted |

### Data & sample-size arithmetic

- **Data continuity is the operational binding constraint** for paper/live/X-3. The tick
  record has missing days, a historical zombie-ingestor gap with no error logs, and dead
  all-NaN columns. The candle archive has none of this.
- **Sample-size arithmetic is unforgiving.** Convolver trap events need ~39 k candles;
  across-regime validation needs 6–24 months; hourly-pattern discovery is infeasible
  (~5 years). Claims must be sized to the data actually in hand — which is why XS-10 tracks
  **83 of 325** required periods rather than declaring a verdict.

### Hypothesis suite (H1–H6) — *historical, 2026-06*

All six confirmed on their contemporaneous windows: directional, long-biased, no decay,
3-feature viable, maker viable. These predate the Q4 kill gate and the maker-line
measurements above; read them as structural priors, not as current tradeable claims.

---

## Current Direction: Three-Class Maker System

Post-Q4, the platform runs a **maker-only doctrine** with a three-class architecture.
Design authority: [`docs/specs/maker_system.md`](docs/specs/maker_system.md).
Research program: [`docs/THREE_CLASS_RESEARCH_PROPOSAL.md`](docs/THREE_CLASS_RESEARCH_PROPOSAL.md)
(16 pre-registered studies, four tracks).

**Execution doctrine** — quotes are the only strategy; a taker order is an *emergency state
transition* (inventory unwind timeout, kill-switch, episode end), never an alpha decision.
Fee tiers are SSOT state in `config/costs.toml`.

**The three classes** (one regime router across them, hysteresis at every level):

| Class | Monetizes | Core | Status |
|---|---|---|---|
| **1 — Directional bias makers** | *persistent* book pressure | touch-pegged quotes, HF1 microprice center, agreement-gated combiner, VPIN/entropy/A4-EV vetoes | signal layer buildable; **economics closed at every reachable fee tier** (§4.10–4.11) — blocked on X-3 fill data |
| **2 — Oscillation harvesters** | *anti-persistent* amplitude | band ladder, geometry read off the spectral surface (never swept), admission by Hurst / band-power / OU-τ; LF7 is the founding member | admission + geometry studies in hand |
| **3 — Cross-sectional rotation** | *breadth* (IR ≈ IC·√breadth) | rank the full perp universe by entropy/momentum/vol vs each pair's own history → top-k weighted allocation → route Classes 1/2 onto selected pairs | **kill test passed (XS-3); rotation refuted (XS-6); cause diagnosed (XS-9); trajectory tracking 83/325 periods (XS-10)** |

**Where Class 3 actually stands.** It is the only data-independent branch, and it leads the
program — but it is *not* a working strategy. XS-6 refuted every rotation configuration
under pre-registered criteria; XS-9 showed why (≈2.2 effective bets, an uncompensated beta
tilt) *and* that the underlying signal sharpens once neutralised. XS-10 turned the
remaining question into arithmetic: at the measured Sharpe, **325 rebalance periods are
needed and 83 are held** — so the answer arrives on a schedule, or the Sharpe decays as n
grows and the rows make that visible in weeks. The sequence is the product, which is why it
appends rather than overwrites.

**Where Class 1 stands.** The maker economics are the tightest constraint in the project:
breakeven demands the venue *pay* ~0.15 bps, no fee tier reaches it, and no touch-maker
configuration survived pre-registration. The one hypothesis left alive is wider-spread
pairs — now reduced by B-5a to a single falsifiable exponent (β\* = 0.69 at the universe
median), resolvable by one tick-data measurement (B-5b).

**Combiner feature contract** — one representative per measured orthogonal axis: fast
direction (IC 0.40–0.47 @1–5 s), slow bias (0.15–0.21 @30 min–3 h), reversion anchor
(0.12–0.29 @10 s–5 min), carry tilt (hypothesis), three gates (VPIN / entropy / A4 EV), and
volatility strictly for sizing (zero directional IC, measured). Agreement-gating is
mandatory — the only structure with measured conditional IC above unconditional.

**Discipline.** Every study pre-registers its criteria before results (per-fill EV > 0,
positive-day share ≥ 0.55, max single-day ≤ 30 %, proxy-sensitivity stability). Signal
claims pass null calibration (z ≥ 3) and BH-FDR (q ≤ 0.05) against a **program-level** FDR
ledger. Failures are recorded in `FINDINGS.md` with the same care as successes — the
negatives in §7.3 and §7.7 were written up as fully as any positive. **No live capital
before G8 and a healthy kill-switch**; `nat lifecycle approve` remains the sole human gate.

---

## References

1. Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*, 5(1), 31-56.
2. Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
3. Bacry, E., Mastromatteo, I. & Muzy, J.F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity*, 1(1).
4. Bailey, D.H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94-107.
5. Bandt, C. & Pompe, B. (2002). Permutation entropy. *Physical Review Letters*, 88(17), 174102.
6. Barndorff-Nielsen, O.E. & Shephard, N. (2004). Power and bipower variation. *Econometrica*, 72(1), 1-37.
7. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289-300.
8. Bouchaud, J.P., Gefen, Y., Potters, M. & Wyart, M. (2004). Fluctuations and response in financial markets. *Quantitative Finance*, 4(2), 176-190.
9. Constantinides, G.M. (1986). Capital market equilibrium with transaction costs. *Journal of Political Economy*, 94(4), 842-862.
10. Cont, R. & de Larrard, A. (2013). Price dynamics in a Markovian limit order market. *SIAM J. Financial Math*, 4(1), 1-25.
11. Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
12. Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
13. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*. 2nd ed. Wiley.
14. Easley, D., López de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity. *Review of Financial Studies*, 25(5), 1457-1493.
15. Elliott, R.J., Aggoun, L. & Moore, J.B. (2005). *Hidden Markov Models*. Springer.
16. Garman, M.B. & Klass, M.J. (1980). On the estimation of security price volatilities. *Journal of Business*, 53(1), 67-78.
17. Gatheral, J. & Oomen, R. (2010). Zero-intelligence realized variance estimation. *Finance and Stochastics*, 14(2), 249-283.
18. Glosten, L.R. & Milgrom, P.R. (1985). Bid, ask and transaction prices. *Journal of Financial Economics*, 14(1), 71-100.
19. Guéant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). Dealing with the inventory risk. *Mathematics and Financial Economics*, 4(7), 477-507.
20. Hamilton, J.D. (1989). A new approach to nonstationary time series. *Econometrica*, 57(2), 357-384.
21. Harvey, C.R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
22. Hendershott, T., Jones, C.M. & Menkveld, A.J. (2011). Does algorithmic trading improve liquidity? *Journal of Finance*, 66(1), 1-33.
23. Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65-91.
24. Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.
25. Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
26. Lee, S.S. & Mykland, P.A. (2008). Jumps in financial markets. *Review of Financial Studies*, 21(6), 2535-2563.
27. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
28. Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions. *SIAM Review*, 10(4), 422-437.
29. Parkinson, M. (1980). The extreme value method for estimating variance. *Journal of Business*, 53(1), 61-65.
30. Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
31. Rabiner, L.R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286.
32. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
33. Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
34. Shiryaev, A.N. (1978). *Optimal Stopping Rules*. Springer.
35. Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative Finance*, 18(12), 1959-1966.
36. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3-4), 285-294.
37. Wald, A. (1947). *Sequential Analysis*. Wiley.
38. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

---

<p align="center">
<i>Built with Rust, Python, and relentless hypothesis testing —<br>
most of which refuted the hypothesis.</i>
</p>

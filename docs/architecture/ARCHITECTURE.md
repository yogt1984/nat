# NAT System Architecture

Full system map of the NAT quantitative research platform — from raw market data to live execution. Three major subsystems: Feature Ingestor (Rust), Signal Discovery (Python), and Execution (Python).

> Feature counts in the diagram below are illustrative — [`FEATURES.md`](../../FEATURES.md) is the
> authoritative manifest (236 features across 21 categories). Design rationale and the original
> (2026-01) vision are at the end of this file.

---

## Complete System Diagram

```
                        ┌──────────────────────────────────┐
                        │     HYPERLIQUID PERPETUALS        │
                        │     BTC / ETH / SOL               │
                        │     WebSocket (Book + Trades)      │
                        └───────────────┬──────────────────┘
                                        │
                    ════════════════════════════════════════════
                    ║          LAYER 1: DATA INGESTION         ║
                    ════════════════════════════════════════════
                                        │
                                        ▼
                ┌───────────────────────────────────────────────┐
                │            RUST INGESTOR (ing)                 │
                │                                               │
                │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
                │  │ BTC Task│  │ ETH Task│  │ SOL Task│      │
                │  └────┬────┘  └────┬────┘  └────┬────┘      │
                │       │            │            │             │
                │       ▼            ▼            ▼             │
                │  ┌──────────────────────────────────────┐    │
                │  │          MarketState (per symbol)     │    │
                │  │  OrderBook + TradeBuffer + Context    │    │
                │  └──────────────────┬───────────────────┘    │
                │                     │                         │
                │                     ▼ 100ms emission          │
                │  ┌──────────────────────────────────────┐    │
                │  │      FeatureComputer (217 features)   │    │
                │  │                                       │    │
                │  │  BASE (138):                          │    │
                │  │   Raw(10) Imbalance(8) Flow(12)       │    │
                │  │   Volatility(9) Entropy(24) Context(9)│    │
                │  │   Trend(15) Illiquidity(12)           │    │
                │  │   Toxicity(10) Derived(15)            │    │
                │  │   Micro(5) Resilience(3) Hawkes(3)    │    │
                │  │                                       │    │
                │  │  OPTIONAL (79, NaN if absent):        │    │
                │  │   WhaleFlow(12) Liquidation(13)       │    │
                │  │   Concentration(15) Regime(20)        │    │
                │  │   GMM(8) CrossSymbol(3) Heatmap(8)    │    │
                │  └──────────────────┬───────────────────┘    │
                │                     │                         │
                │                     ▼                         │
                │  ┌──────────────────────────────────────┐    │
                │  │   ParquetWriter                       │    │
                │  │   10k row batches, hourly rotation    │    │
                │  │   zstd compression                    │    │
                │  └──────────────────┬───────────────────┘    │
                └─────────────────────┼────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────────┐
                    │  data/features/YYYY-MM-DD/          │
                    │  Symbol_YYYY-MM-DD_HH_MM.parquet    │
                    │  ~30 rows/sec/symbol                │
                    └───────────────┬────────────────────┘
                                    │
         ┌──────────────────────────┼───────────────────────────────┐
         │                          │                               │
    ═════╪══════════════════════════╪═══════════════════════════════╪═════
    ║    │   LAYER 2: SIGNAL DISCOVERY & RESEARCH                  │    ║
    ═════╪══════════════════════════╪═══════════════════════════════╪═════
         │                          │                               │
         ▼                          ▼                               ▼
┌─────────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
│ PIPELINE RUNNER │   │ ALGORITHM FRAMEWORK   │   │   AGENT SYSTEM          │
│                 │   │                       │   │                          │
│ IDLE            │   │ 30+ algorithms:       │   │ ┌──────────────────┐    │
│  ↓              │   │                       │   │ │ Microstructure   │    │
│ BUILDING        │   │ ┌───────────────────┐ │   │ │ Agent (5s)       │    │
│  ↓              │   │ │ CONVOLVER         │ │   │ │ 6 generators     │    │
│ INGESTING       │   │ │                   │ │   │ │ 5-gate protocol  │    │
│  ↓              │   │ │ Offline:          │ │   │ └──────────────────┘    │
│ COLLECTING      │   │ │  Event detect     │ │   │ ┌──────────────────┐    │
│  ↓              │   │ │  SVD decompose    │ │   │ │ Medium-Freq      │    │
│ ANALYZING       │   │ │  IC gate + FDR    │ │   │ │ Agent (1m-1h)    │    │
│  ↓              │   │ │  → 6 kernels      │ │   │ │ 3 generators     │    │
│ DONE            │   │ │                   │ │   │ │ 4-gate protocol  │    │
│                 │   │ │ Online (8 feats): │ │   │ └──────────────────┘    │
│ Gates:          │   │ │  Tick accumulate  │ │   │ ┌──────────────────┐    │
│ silhouette,     │   │ │  60s candles      │ │   │ │ Macro Agent      │    │
│ bootstrap_ari,  │   │ │  Score kernels    │ │   │ │ (1h-24h)         │    │
│ temporal_ari    │   │ │  Cosine similarity│ │   │ │ 3 generators     │    │
│                 │   │ └───────────────────┘ │   │ │ 4-gate protocol  │    │
└─────────────────┘   │                       │   │ └──────────────────┘    │
                      │ ┌───────────────────┐ │   │          │              │
                      │ │ Other algorithms: │ │   │          ▼              │
                      │ │  jump_detector    │ │   │ ┌──────────────────┐    │
                      │ │  funding_reversion│ │   │ │ Meta Agent       │    │
                      │ │  entropy_momentum │ │   │ │ (orchestrator)   │    │
                      │ │  optimal_entry    │ │   │ │ Budget, correl,  │    │
                      │ │  ...              │ │   │ │ portfolio        │    │
                      │ │                   │ │   │ └──────────────────┘    │
                      │ └───────────────────┘ │   │                          │
                      └───────────┬───────────┘   └────────────┬───────────┘
                                  │                             │
                                  │    ┌────────────────────┐   │
                                  └───►│ Signal Registries   │◄──┘
                                       │ data/agent/         │
                                       │ data/agent_mf/      │
                                       │ data/agent_macro/    │
                                       │ data/alpha/          │
                                       └─────────┬──────────┘
                                                 │
    ═══════════════════════════════════════════════╪════════════════════
    ║              LAYER 3: LIQUIDITY ENGINE                          ║
    ═══════════════════════════════════════════════╪════════════════════
                                                  │
                                                  ▼
                    ┌────────────────────────────────────────────┐
                    │        ALPHA PIPELINE (9 Steps)             │
                    │                                             │
                    │  ┌───────┐  ┌───────┐  ┌───────┐          │
                    │  │SCREEN │→ │COMBINE│→ │ SIZE  │          │
                    │  │ FDR   │  │IC dedup│  │Kelly  │          │
                    │  │  G1   │  │  G2    │  │  G3   │          │
                    │  └───────┘  └───────┘  └───────┘          │
                    │       ↓                                     │
                    │  ┌───────┐  ┌───────┐  ┌───────┐          │
                    │  │VALIDAT│→ │REGIME │→ │MULTI_F│          │
                    │  │WF OOS │  │GMM cond│  │TF blend│         │
                    │  │  G4   │  │  G5    │  │  G6   │          │
                    │  └───────┘  └───────┘  └───────┘          │
                    │       ↓                                     │
                    │  ┌───────┐  ┌───────┐  ┌───────┐          │
                    │  │PORTFOL│→ │PAPER  │→ │DEPLOY │          │
                    │  │RiskPar│  │Live sim│  │Model  │          │
                    │  │  G7   │  │  G8    │  │  G9   │          │
                    │  └───────┘  └───────┘  └───────┘          │
                    └───────────────────┬────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │          3-FEATURE LIQUIDITY SIGNAL                   │
              │                                                       │
              │  Input: raw_spread_bps + raw_ask_depth_5 +           │
              │         flow_vwap_deviation                           │
              │                                                       │
              │  Method: Walk-forward z-scores (train on 3 prior     │
              │          dates), composite = mean(z_spread, z_depth, │
              │          z_vwap), entry on P80/P20 thresholds        │
              │                                                       │
              │  Output: direction (+1/-1/0), 100min horizon,        │
              │          Sharpe 5.6-11.8 (symbol-dependent)          │
              └──────────────────────┬──────────────────────────────┘
                                     │
    ═════════════════════════════════╪════════════════════════════════
    ║          LAYER 4: EXECUTION                                   ║
    ═════════════════════════════════╪════════════════════════════════
                                     │
         ┌───────────────────────────┼───────────────────────┐
         │                           │                       │
         ▼                           ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐   ┌───────────────────┐
│ PAPER TRADER    │   │ PORTFOLIO ASSEMBLY    │   │ SIGNAL BRIDGE     │
│                 │   │                       │   │ (Live Execution)  │
│ Modes:          │   │ Risk parity weights   │   │                   │
│  batch (replay) │   │ BTC/ETH/SOL           │   │ Cycle: 5min       │
│  watch (daemon) │   │ Correlation adjust    │   │ Modes:            │
│                 │   │ DD control:           │   │  dry-run           │
│ Gate G8:        │   │  50% scale if DD>2%  │   │  paper             │
│  Sharpe > 0.5×  │   │                       │   │  live              │
│  IC decay < 50% │   │ Output: allocation    │   │                   │
│  max loss < 2%  │   │ weights per symbol    │   │ Kill switches:     │
└────────┬────────┘   └───────────┬───────────┘   │  daily > 1%       │
         │                        │               │  weekly > 2%       │
         │                        ▼               │  monthly > 5%      │
         │              ┌──────────────────┐      │  IC neg 5 days     │
         │              │ Capital Scaling  │      │                   │
         └─────────────►│ 1% → 5% → 10%  │◄─────┘                   │
                        │ → 25% of equity  │      │                   │
                        └──────────────────┘      └─────────┬─────────┘
                                                            │
                                                            ▼
                                              ┌──────────────────────┐
                                              │ HYPERLIQUID API      │
                                              │ Maker orders (limit) │
                                              │ Position reconcile   │
                                              └──────────────────────┘
```

---

## Component Interaction Matrix

| Producer → Consumer | Data Format | Frequency |
|---|---|---|
| Hyperliquid → Ingestor | WebSocket JSON | ~10ms |
| Ingestor → Parquet | Arrow batches (10k rows) | ~5.5 min flush |
| Parquet → Algorithm Framework | DataFrame read | On-demand |
| Parquet → Agent Daemons | DataFrame read | 1h / 2h / 4h cycles |
| Parquet → Liquidity Engine | 5min bars | Walk-forward daily |
| Convolver Discovery → Kernel Files | .npz + .json | Offline (rare) |
| Convolver Online → Feature Vector | 8 floats per candle | Every 60s |
| Alpha Pipeline → Signal Registry | JSON | Per pipeline run |
| Agent → Signal Registry | JSON | Per hypothesis pass |
| Signal Registry → Paper Trader | Trade decisions | 5min |
| Paper Trader → Signal Bridge | Validated signal | 5min |
| Signal Bridge → Hyperliquid | REST API (orders) | 5min |

---

## Discovery Orchestrator (Cross-Cutting)

```
┌────────────────────────────────────────────────────────────┐
│          DISCOVERY ORCHESTRATOR (6h cycles)                  │
│                                                              │
│  DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING       │
│       │              │              │            │            │
│       ▼              ▼              ▼            ▼            │
│  Parquet fresh?   Screen all    Train model  Walk-forward    │
│                   (sym,horizon)  on winners   OOS validate   │
│                   combinations                               │
│                                                              │
│  → ALPHA_PIPELINE → REPORTING → SLEEPING                    │
│         │                │                                   │
│         ▼                ▼                                   │
│    Run full 9-step   JSON + plots                           │
│    on winners        to reports/                             │
└────────────────────────────────────────────────────────────┘
```

---

## Key Paths

| Component | Entry Point |
|---|---|
| Rust Ingestor | `rust/ing/src/main.rs` |
| Feature Modules (26) | `rust/ing/src/features/*.rs` |
| Convolver (online) | `scripts/algorithms/convolver.py` |
| Convolver (discovery) | `scripts/analysis/convolver_discovery.py` |
| Pipeline Runner | `scripts/pipeline_runner.py` |
| Alpha Pipeline | `scripts/alpha/alpha_pipeline.py` |
| Liquidity Backtest | `scripts/analysis/mf_liquidity_backtest.py` |
| Paper Trader | `scripts/alpha/paper_trader.py` |
| Signal Bridge | `scripts/execution/signal_bridge.py` |
| Agent (micro) | `scripts/agent/daemon.py` |
| Agent (MF) | `scripts/agent/mf_daemon.py` |
| Agent (macro) | `scripts/agent/macro_daemon.py` |
| Meta Agent | `scripts/agent/meta_daemon.py` |
| Discovery Orchestrator | `scripts/discovery_orchestrator.py` |
| Config (ingestor) | `config/ing.toml` |
| Config (alpha) | `config/alpha.toml` |
| Config (agents) | `config/agent.toml` |
| Config (algorithms) | `config/algorithms.toml` |

---

## Design Rationale

The system's shape follows five decisions from the original 2026-01 design. The *rationale* is
durable even where early implementation specifics have since evolved:

1. **Rust for latency-critical paths.** Ingestion + feature computation in Rust (`rust/ing-*`);
   Python for ML, analysis, orchestration. Determinism and memory safety over the hot path.
2. **Columnar, zero-copy hand-off.** Features cross the Rust→Python boundary as Arrow/Parquet, not a
   serialized message bus. (The original design floated shared-memory/ZeroMQ IPC; Parquet won for
   reproducibility and operational simplicity.)
3. **Deterministic replay.** All randomness is seeded and logged; feature extraction is a pure
   function of input, so any run replays exactly. This is what makes the planted-test discipline in
   [`METHODOLOGY.md`](../METHODOLOGY.md) possible.
4. **Multi-objective fitness, not Sharpe-only.** Optimize the Pareto front over Sharpe / drawdown / IC
   via NSGA-II (shipped in `nat evolve`), never a single scalar that invites overfitting.
5. **Incremental complexity.** Baseline (z-score / logistic) before ML; validate each unit
   independently — the feature / algorithm / process contracts in [`../contracts/`](../contracts/).

## Regime discovery: unsupervised > supervised

A methodological stance carried from the original design and still guiding the regime/process work:
do **not** define regimes by strategy profitability (circular — regimes become artifacts of the
strategy). Instead discover natural market states first (clustering / information-geometric
structure), **then** ask "what works in each state?" This is why NAT keeps both a `cluster_pipeline/`
and a conditional-predictability process (`PROC-6` in [`../TASKS.md`](../TASKS.md)): the regime is
*measured*, then conditioned on — not assumed.

## Historical vision (2026-01)

The full original design doc — the agent-swarm ecosystem diagram, the proposed
`{CATEGORY}_{SOURCE}_{TRANSFORM}_…` feature nomenclature (**never adopted**; live names are lowercase,
see [`FEATURES.md`](../../FEATURES.md)), the aspirational HFT specs (`<10μs`, `SCHED_FIFO`), the
polynomial-chaos / transfer-entropy / information-geometry "novel extensions", the genotype encoding,
and the 24-week phased methodology — is preserved for provenance at
[`../archive/architecture/ARCHITECTURE_vision_2026-01.md`](../archive/architecture/ARCHITECTURE_vision_2026-01.md).
Several of its ideas have since materialized elsewhere: multi-objective evolution as `nat evolve`,
the transfer-entropy causal graph as `PROC-9`, and the feature manifest as `FEATURES.md`.

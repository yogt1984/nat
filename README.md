```

    ███╗   ██╗ █████╗ ████████╗
    ████╗  ██║██╔══██╗╚══██╔══╝
    ██╔██╗ ██║███████║   ██║
    ██║╚██╗██║██╔══██║   ██║
    ██║ ╚████║██║  ██║   ██║
    ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝

    N E X T - G E N   A L P H A   T E C H N O L O G Y
    ─────────────────────────────────────────────────
    Hyperliquid Analytics & Signal Intelligence Layer

```

# NAT — Quantitative Research Infrastructure for Hyperliquid

**NAT** is a production-grade quantitative research platform designed for extracting alpha signals from Hyperliquid's perpetual futures market. Built in Rust for maximum performance, NAT provides real-time feature extraction, rigorous hypothesis testing, and statistically-validated trading signals.

[![Tests](https://img.shields.io/badge/tests-266%20passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.75+-orange)]()
[![License](https://img.shields.io/badge/license-proprietary-blue)]()

---

## Why NAT?

Most crypto analytics tools are **toys**. NAT is **infrastructure**.

| The Problem | NAT's Solution |
|-------------|----------------|
| Delayed data feeds | Sub-millisecond WebSocket ingestion |
| Basic indicators | 40+ institutional-grade features |
| No validation | Rigorous hypothesis testing with walk-forward validation |
| Overfitting | Bonferroni correction, OOS/IS ratio checks, MI thresholds |
| Black box signals | Full statistical transparency with confidence intervals |

---

## Core Capabilities

### Real-Time Feature Extraction Engine

NAT processes Hyperliquid's order book and trade stream in real-time, computing 40+ features across 8 categories:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WebSocket Stream ──► Order Book State ──► Feature Computation      │
│        │                    │                      │                │
│        ▼                    ▼                      ▼                │
│   ┌─────────┐        ┌───────────┐          ┌──────────┐           │
│   │ Trades  │        │ L2 Book   │          │ Parquet  │           │
│   │ Ticks   │        │ Snapshots │          │ Output   │           │
│   └─────────┘        └───────────┘          └──────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Feature Categories:**

| Category | Features | Signal Type |
|----------|----------|-------------|
| **Entropy** | Tick entropy (1s, 5s, 10s, 30s, 1m, 15m) | Regime detection, predictability |
| **Trend** | Momentum, monotonicity, Hurst exponent | Persistence, mean-reversion |
| **Illiquidity** | Kyle's λ, Amihud, Hasbrouck | Price impact, informed flow |
| **Toxicity** | VPIN, adverse selection | Order flow toxicity |
| **Order Flow** | Imbalance, pressure, aggressor ratio | Directional conviction |
| **Volatility** | Realized vol, Parkinson, Garman-Klass | Risk regime |
| **Whale Flow** | Net flow by wallet tier (1h, 4h, 24h) | Smart money tracking |
| **Concentration** | Gini, HHI, Top-10/20, Theil | Position crowding |

### Whale Intelligence System

Track and classify large players in real-time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WHALE CLASSIFICATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Tier 1: MEGA WHALE     │  $10M+ positions    │  Market movers    │
│   Tier 2: WHALE          │  $1M-$10M           │  Significant      │
│   Tier 3: LARGE TRADER   │  $100K-$1M          │  Notable          │
│   Tier 4: RETAIL         │  <$100K             │  Noise            │
│                                                                     │
│   Metrics: Net flow, position changes, entry/exit timing           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Liquidation Cascade Detection

Identify clustered liquidation risk before cascades occur:

- Real-time liquidation price mapping
- Cluster detection with configurable thresholds
- Lead-time analysis for actionable signals
- Directional prediction (long vs short squeeze)

---

## Hypothesis Testing Framework

NAT doesn't guess. NAT **validates**.

Every signal passes through a rigorous statistical gauntlet before deployment:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS TESTING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │    H1    │    │    H2    │    │    H3    │    │    H4    │     │
│  │  Whale   │    │ Entropy+ │    │ Liquid.  │    │ Concen.  │     │
│  │  Flow    │    │  Whale   │    │ Cascade  │    │   Vol    │     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│       │               │               │               │            │
│       └───────────────┴───────────────┴───────────────┘            │
│                           │                                        │
│                           ▼                                        │
│                    ┌──────────┐                                    │
│                    │    H5    │                                    │
│                    │ Persist. │                                    │
│                    │Indicator │                                    │
│                    └────┬─────┘                                    │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                               │
│              │   FINAL DECISION    │                               │
│              │  GO / PIVOT / NOGO  │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Five Hypotheses

| ID | Hypothesis | Success Criteria | Statistical Tests |
|----|------------|------------------|-------------------|
| **H1** | Whale flow predicts returns | r > 0.05, p < 0.001, MI > 0.02 bits | Pearson, Spearman, Walk-forward |
| **H2** | Entropy + whale interaction | Lift > 10%, p < 0.01 | Chi-squared, Contingency tables |
| **H3** | Liquidation cascades predictable | Precision > 30%, Lift > 2x | Classification metrics, Lead-time |
| **H4** | Concentration predicts volatility | r > 0.2, partial r > 0.1 | Partial correlation, Causality |
| **H5** | Persistence indicator works | WF Sharpe > 0.5, OOS/IS > 0.7 | Walk-forward, Regime analysis |

### Decision Framework

```
╔══════════════════════════════════════════════════════════════════╗
║                      DECISION MATRIX                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   0-1 hypotheses pass  ──►  NO-GO   (Insufficient alpha)         ║
║   2-3 hypotheses pass  ──►  PIVOT   (Focus on validated only)    ║
║   4-5 hypotheses pass  ──►  GO      (Full deployment)            ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Statistical Rigor

NAT implements institutional-grade statistical validation:

- **Bonferroni Correction** — Controls family-wise error rate across multiple tests
- **Walk-Forward Validation** — 5-fold expanding window, no look-ahead bias
- **Out-of-Sample Ratio** — OOS/IS > 0.7 required (detects overfitting)
- **Mutual Information** — Non-linear dependency detection beyond correlation
- **Confidence Intervals** — 95% CI on all correlation estimates
- **Regime Analysis** — Separate validation in low-vol vs high-vol environments

---

## Feature Redundancy Analysis

Not all features are created equal. NAT's feature analysis module:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ANALYSIS OUTPUT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✓ Correlation matrix (Pearson + Spearman)                         │
│  ✓ Mutual Information matrix                                        │
│  ✓ Hierarchical clustering with dendrogram                         │
│  ✓ Redundancy detection (|r| > 0.9)                                │
│  ✓ Feature ranking by predictive power                             │
│  ✓ Recommended subset (10-15 non-redundant features)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
nat/
├── rust/ing/                    # Core Rust engine
│   ├── src/
│   │   ├── main.rs              # Entry point & orchestration
│   │   ├── ws/                  # WebSocket client (Hyperliquid)
│   │   ├── rest/                # REST API client
│   │   ├── state/               # Order book state management
│   │   ├── features/            # Feature extraction (40+ features)
│   │   │   ├── entropy.rs       # Tick entropy features
│   │   │   ├── trend.rs         # Momentum, Hurst, monotonicity
│   │   │   ├── illiquidity.rs   # Kyle, Amihud, Hasbrouck
│   │   │   ├── toxicity.rs      # VPIN, adverse selection
│   │   │   ├── whale_flow.rs    # Whale tracking features
│   │   │   ├── concentration.rs # Position concentration
│   │   │   ├── liquidation.rs   # Liquidation mapping
│   │   │   └── ...
│   │   ├── hypothesis/          # Statistical testing framework
│   │   │   ├── stats.rs         # Core statistical functions
│   │   │   ├── h1_whale_flow.rs
│   │   │   ├── h2_entropy_whale.rs
│   │   │   ├── h3_liquidation_cascade.rs
│   │   │   ├── h4_concentration_vol.rs
│   │   │   ├── h5_persistence.rs
│   │   │   ├── feature_analysis.rs
│   │   │   └── final_decision.rs
│   │   ├── whales/              # Whale registry & classification
│   │   ├── positions/           # Position tracking
│   │   ├── output/              # Parquet writer
│   │   └── metrics/             # Prometheus metrics
│   └── config/                  # Configuration files
└── docs/                        # Documentation & research
```

---

## Performance

Built in Rust for production workloads:

| Metric | Performance |
|--------|-------------|
| Feature computation | < 1ms per tick |
| Memory footprint | ~50MB per symbol |
| Throughput | 10,000+ updates/sec |
| Output format | Parquet (columnar, compressed) |

---

## Quick Start

```bash
# Build
cargo build --release

# Run with default config
./target/release/ing config/ing.toml

# Run tests
cargo test

# 266 tests, all passing
```

---

## Configuration

```toml
[symbols]
assets = ["BTC", "ETH", "SOL"]

[features]
emission_interval_ms = 1000
entropy_windows = [1, 5, 10, 30, 60, 900]

[websocket]
url = "wss://api.hyperliquid.xyz/ws"
reconnect_delay_ms = 1000

[output]
path = "./data"
rotation_interval_secs = 3600
```

---

## Research Output

NAT generates structured output for downstream analysis:

**Parquet Schema:**
- Timestamp (ns precision)
- Symbol
- 40+ feature columns
- Metadata (sequence ID, data quality flags)

**Decision Report:**
- Hypothesis test results with confidence intervals
- Strategy estimates (Sharpe, capacity, alpha decay)
- Recommended feature subset
- Honest assessment & next steps

---

## Test Coverage

```
266 tests across:
├── Feature extraction (120+ tests)
├── Statistical functions (30+ tests)
├── Hypothesis H1-H5 (85+ tests)
├── Feature analysis (14 tests)
└── Final decision (12 tests)
```

---

## Roadmap

- [x] Real-time feature extraction
- [x] Whale tracking & classification
- [x] Hypothesis testing framework (H1-H5)
- [x] Feature redundancy analysis
- [x] GO/PIVOT/NO-GO decision engine
- [ ] Backtesting infrastructure
- [ ] Paper trading integration
- [ ] Live deployment

---

## Philosophy

> "In God we trust. All others must bring data." — W. Edwards Deming

NAT is built on the principle that **every trading signal must be statistically validated** before deployment. No hunches. No vibes. No "it worked in backtest."

The framework is intentionally skeptical — designed to reject weak signals and prevent overfitting, even at the cost of rejecting some potentially valid hypotheses.

---

## License

Proprietary. All rights reserved.

---

<p align="center">
  <b>NAT</b> — Where Alpha Meets Rigor
</p>

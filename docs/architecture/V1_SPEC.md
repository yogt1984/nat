# V1 Specification: Entropy-Based Regime Detection

## Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────┐
│                         V1 PRINCIPLES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PROVE BEFORE OPTIMIZE                                           │
│     └── Validate entropy-regime hypothesis first                    │
│                                                                      │
│  2. MINIMAL VIABLE FEATURE SET                                      │
│     └── 50-60 features, not 200                                     │
│                                                                      │
│  3. SINGLE EXCHANGE, FEW ASSETS                                     │
│     └── Hyperliquid: BTC, ETH, SOL                                  │
│                                                                      │
│  4. RUST CORE, PYTHON ANALYSIS                                      │
│     └── Rust: Ingest + features. Python: ML + research             │
│                                                                      │
│  5. FILE-BASED PERSISTENCE                                          │
│     └── Parquet files, no database for V1                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## V1 Feature Set (57 Features)

### Category 1: Raw Order Book (10 features)

```
RAW_MIDPRICE                    # (best_bid + best_ask) / 2
RAW_SPREAD                      # best_ask - best_bid
RAW_SPREAD_BPS                  # spread / midprice * 10000
RAW_MICROPRICE                  # (bid_qty * ask_px + ask_qty * bid_px) / (bid_qty + ask_qty)
RAW_BID_DEPTH_5                 # Sum of bid qty, levels 1-5
RAW_ASK_DEPTH_5                 # Sum of ask qty, levels 1-5
RAW_BID_DEPTH_10                # Sum of bid qty, levels 1-10
RAW_ASK_DEPTH_10                # Sum of ask qty, levels 1-10
RAW_BID_ORDERS_5                # Sum of order counts (n), levels 1-5
RAW_ASK_ORDERS_5                # Sum of order counts (n), levels 1-5
```

### Category 2: Order Book Imbalance (8 features)

```
IMBALANCE_QTY_L1                # (bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1)
IMBALANCE_QTY_L5                # Same for levels 1-5 aggregated
IMBALANCE_QTY_L10               # Same for levels 1-10 aggregated
IMBALANCE_ORDERS_L5             # (bid_orders - ask_orders) / total, levels 1-5
IMBALANCE_NOTIONAL_L5           # Notional-weighted imbalance
IMBALANCE_DEPTH_WEIGHTED        # Distance-weighted imbalance
IMBALANCE_PRESSURE_BID          # Cumulative bid pressure score
IMBALANCE_PRESSURE_ASK          # Cumulative ask pressure score
```

### Category 3: Trade Flow (12 features)

```
FLOW_COUNT_1S                   # Trade count, 1 second window
FLOW_COUNT_5S                   # Trade count, 5 second window
FLOW_COUNT_30S                  # Trade count, 30 second window
FLOW_VOLUME_1S                  # Trade volume, 1 second
FLOW_VOLUME_5S                  # Trade volume, 5 seconds
FLOW_VOLUME_30S                 # Trade volume, 30 seconds
FLOW_AGGRESSOR_RATIO_5S         # Buy aggressor volume / total volume
FLOW_AGGRESSOR_RATIO_30S        # Same, 30 second window
FLOW_VWAP_5S                    # VWAP over 5 seconds
FLOW_VWAP_DEVIATION             # (VWAP - midprice) / midprice
FLOW_AVG_TRADE_SIZE_30S         # Average trade size
FLOW_INTENSITY                  # Trades per second (EMA)
```

### Category 4: Volatility (8 features)

```
VOL_RETURNS_1M                  # Realized vol from 1-min returns
VOL_RETURNS_5M                  # Realized vol from 5-min returns
VOL_PARKINSON_5M                # Parkinson estimator (high-low)
VOL_SPREAD_MEAN_1M              # Average spread over 1 min
VOL_SPREAD_STD_1M               # Spread standard deviation
VOL_MIDPRICE_STD_1M             # Midprice standard deviation
VOL_RATIO_SHORT_LONG            # vol_1m / vol_5m (regime indicator)
VOL_ZSCORE                      # Current vol vs 1-hour mean
```

### Category 5: Entropy (10 features)

```
ENT_PERMUTATION_RETURNS_8       # Permutation entropy, embedding dim=3, len=8
ENT_PERMUTATION_RETURNS_16      # Permutation entropy, len=16
ENT_PERMUTATION_RETURNS_32      # Permutation entropy, len=32
ENT_PERMUTATION_IMBALANCE_16    # Permutation entropy of imbalance series
ENT_SPREAD_DISPERSION           # Entropy of spread distribution
ENT_VOLUME_DISPERSION           # Entropy of volume distribution
ENT_BOOK_SHAPE                  # Entropy of depth distribution across levels
ENT_TRADE_SIZE_DISPERSION       # Entropy of trade size distribution
ENT_RATE_OF_CHANGE_5S           # d(entropy)/dt over 5 seconds
ENT_ZSCORE_1M                   # Entropy z-score vs 1-minute mean
```

### Category 6: Market Context (9 features)

```
CTX_FUNDING_RATE                # Current funding rate
CTX_FUNDING_ZSCORE              # Funding vs 24h mean
CTX_OPEN_INTEREST               # Total OI
CTX_OI_CHANGE_5M                # OI change over 5 minutes
CTX_OI_CHANGE_PCT_5M            # OI % change
CTX_PREMIUM_BPS                 # (mark - oracle) / oracle * 10000
CTX_VOLUME_24H                  # 24h volume
CTX_VOLUME_RATIO                # Current rate vs 24h average
CTX_MARK_ORACLE_DIVERGENCE      # |mark - oracle| persistence
```

---

## Feature Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Raw Order Book | 10 | Base market state |
| Imbalance | 8 | Directional pressure |
| Trade Flow | 12 | Execution dynamics |
| Volatility | 8 | Risk regime |
| Entropy | 10 | Core regime signal |
| Market Context | 9 | Macro positioning |
| **Total** | **57** | |

---

## V1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            V1 ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌──────────────────┐                                │
│                         │   HYPERLIQUID    │                                │
│                         │   WebSocket API  │                                │
│                         └────────┬─────────┘                                │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           ING (Rust)                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │  WebSocket  │─▶│   Parser    │─▶│  Features   │─▶│   Writer    │   │ │
│  │  │   Client    │  │  (L2/Trade) │  │  Computer   │  │  (Parquet)  │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  │                                           │                            │ │
│  │                                           ▼                            │ │
│  │                                    ┌─────────────┐                     │ │
│  │                                    │ Ring Buffer │                     │ │
│  │                                    │  (History)  │                     │ │
│  │                                    └─────────────┘                     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌──────────────────┐                                │
│                         │   data/*.parquet │                                │
│                         │   (Feature Store)│                                │
│                         └────────┬─────────┘                                │
│                                  │                                          │
│          ┌───────────────────────┼───────────────────────┐                 │
│          ▼                       ▼                       ▼                 │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐           │
│  │   Labeler    │       │   Trainer    │       │  Backtester  │           │
│  │   (Python)   │       │   (Python)   │       │   (Python)   │           │
│  └──────────────┘       └──────────────┘       └──────────────┘           │
│          │                       │                       │                 │
│          └───────────────────────┼───────────────────────┘                 │
│                                  ▼                                          │
│                         ┌──────────────────┐                                │
│                         │  Regime Model    │                                │
│                         │  (XGBoost/ONNX)  │                                │
│                         └──────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. ING (Rust Ingestor)

**Responsibility**: Connect to Hyperliquid, compute features, write to Parquet.

```
rust/
├── Cargo.toml
└── ing/
    ├── Cargo.toml
    └── src/
        ├── main.rs              # Entry point, config loading
        ├── config.rs            # Configuration structs
        ├── ws/
        │   ├── mod.rs
        │   ├── client.rs        # WebSocket connection management
        │   └── messages.rs      # Message parsing (WsBook, WsTrade, etc.)
        ├── state/
        │   ├── mod.rs
        │   ├── order_book.rs    # Order book state (bid/ask levels)
        │   ├── trade_buffer.rs  # Rolling trade window
        │   └── context.rs       # Funding, OI, etc.
        ├── features/
        │   ├── mod.rs
        │   ├── raw.rs           # RAW_* features
        │   ├── imbalance.rs     # IMBALANCE_* features
        │   ├── flow.rs          # FLOW_* features
        │   ├── volatility.rs    # VOL_* features
        │   ├── entropy.rs       # ENT_* features
        │   └── context.rs       # CTX_* features
        ├── output/
        │   ├── mod.rs
        │   ├── schema.rs        # Arrow schema definition
        │   └── writer.rs        # Parquet writer (buffered)
        └── metrics.rs           # Latency tracking, counters
```

**Key Design Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State management | In-memory ring buffers | Simple, fast, sufficient for V1 |
| Output format | Parquet (row groups by time) | Columnar, compressed, Python-friendly |
| Feature emission | Every 100ms | Balances granularity vs storage |
| Reconnection | Automatic with exponential backoff | Robustness |

### 2. Feature Schema (Arrow/Parquet)

```rust
// Conceptual schema
pub struct FeatureRow {
    // Metadata
    pub timestamp_ns: i64,        // Nanosecond timestamp
    pub symbol: String,           // "BTC", "ETH", "SOL"
    pub sequence_id: u64,         // Monotonic sequence

    // Raw (10)
    pub raw_midprice: f64,
    pub raw_spread: f64,
    pub raw_spread_bps: f64,
    pub raw_microprice: f64,
    pub raw_bid_depth_5: f64,
    pub raw_ask_depth_5: f64,
    pub raw_bid_depth_10: f64,
    pub raw_ask_depth_10: f64,
    pub raw_bid_orders_5: u32,
    pub raw_ask_orders_5: u32,

    // Imbalance (8)
    pub imbalance_qty_l1: f64,
    pub imbalance_qty_l5: f64,
    pub imbalance_qty_l10: f64,
    pub imbalance_orders_l5: f64,
    pub imbalance_notional_l5: f64,
    pub imbalance_depth_weighted: f64,
    pub imbalance_pressure_bid: f64,
    pub imbalance_pressure_ask: f64,

    // Flow (12)
    pub flow_count_1s: u32,
    pub flow_count_5s: u32,
    pub flow_count_30s: u32,
    pub flow_volume_1s: f64,
    pub flow_volume_5s: f64,
    pub flow_volume_30s: f64,
    pub flow_aggressor_ratio_5s: f64,
    pub flow_aggressor_ratio_30s: f64,
    pub flow_vwap_5s: f64,
    pub flow_vwap_deviation: f64,
    pub flow_avg_trade_size_30s: f64,
    pub flow_intensity: f64,

    // Volatility (8)
    pub vol_returns_1m: f64,
    pub vol_returns_5m: f64,
    pub vol_parkinson_5m: f64,
    pub vol_spread_mean_1m: f64,
    pub vol_spread_std_1m: f64,
    pub vol_midprice_std_1m: f64,
    pub vol_ratio_short_long: f64,
    pub vol_zscore: f64,

    // Entropy (10)
    pub ent_permutation_returns_8: f64,
    pub ent_permutation_returns_16: f64,
    pub ent_permutation_returns_32: f64,
    pub ent_permutation_imbalance_16: f64,
    pub ent_spread_dispersion: f64,
    pub ent_volume_dispersion: f64,
    pub ent_book_shape: f64,
    pub ent_trade_size_dispersion: f64,
    pub ent_rate_of_change_5s: f64,
    pub ent_zscore_1m: f64,

    // Context (9)
    pub ctx_funding_rate: f64,
    pub ctx_funding_zscore: f64,
    pub ctx_open_interest: f64,
    pub ctx_oi_change_5m: f64,
    pub ctx_oi_change_pct_5m: f64,
    pub ctx_premium_bps: f64,
    pub ctx_volume_24h: f64,
    pub ctx_volume_ratio: f64,
    pub ctx_mark_oracle_divergence: f64,
}
```

### 3. Python Analysis Layer

```
python/
├── pyproject.toml
└── nat/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── loader.py            # Parquet loading utilities
    ├── labeling/
    │   ├── __init__.py
    │   ├── strategy_sim.py      # ASMM / TrendFollow simulation
    │   └── labeler.py           # Generate regime labels
    ├── regime/
    │   ├── __init__.py
    │   ├── features.py          # Feature selection, normalization
    │   ├── classifier.py        # XGBoost regime classifier
    │   └── evaluation.py        # Metrics, confusion matrix
    ├── backtest/
    │   ├── __init__.py
    │   ├── simulator.py         # Strategy backtester
    │   └── metrics.py           # Sharpe, drawdown, etc.
    └── visualization/
        ├── __init__.py
        └── plots.py             # Regime visualization, UMAP
```

### 4. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COLLECTION (Continuous)                                            │
│  ────────────────────────                                           │
│  Hyperliquid WS ──▶ ING ──▶ data/features/YYYY-MM-DD/               │
│                              ├── BTC.parquet                        │
│                              ├── ETH.parquet                        │
│                              └── SOL.parquet                        │
│                                                                      │
│  LABELING (Batch, Daily)                                            │
│  ───────────────────────                                            │
│  data/features/*.parquet                                            │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ Simulate    │────▶│  Compare    │────▶│   Label     │           │
│  │ ASMM(θ)     │     │  P&L        │     │   Windows   │           │
│  │ TrendF(θ)   │     │             │     │   MR/TF/NA  │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                 │                    │
│                                                 ▼                    │
│                                    data/labels/YYYY-MM-DD.parquet   │
│                                                                      │
│  TRAINING (Weekly)                                                  │
│  ─────────────────                                                  │
│  data/features/*.parquet + data/labels/*.parquet                    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  Join on    │────▶│   Train     │────▶│   Export    │           │
│  │  timestamp  │     │   XGBoost   │     │   ONNX      │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                 │                    │
│                                                 ▼                    │
│                                    models/regime_v1.onnx            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
nat/
├── PAPERS_IDEAS.md
├── ARCHITECTURE.md
├── HYPER_DOCS.md
├── V1_SPEC.md                    # This file
│
├── rust/
│   ├── Cargo.toml                # Workspace
│   └── ing/
│       ├── Cargo.toml
│       └── src/
│           └── ...               # As detailed above
│
├── python/
│   ├── pyproject.toml
│   └── nat/
│       └── ...                   # As detailed above
│
├── config/
│   └── ing.toml                  # Ingestor config
│
├── data/
│   ├── features/                 # Raw feature parquets
│   │   └── YYYY-MM-DD/
│   ├── labels/                   # Regime labels
│   └── models/                   # Trained models
│
└── scripts/
    ├── run_ing.sh                # Start ingestor
    ├── label.py                  # Generate labels
    ├── train.py                  # Train classifier
    └── backtest.py               # Run backtest
```

---

## Configuration

### ing.toml

```toml
[general]
log_level = "info"
data_dir = "./data/features"

[websocket]
url = "wss://api.hyperliquid.xyz/ws"
reconnect_delay_ms = 1000
max_reconnect_delay_ms = 30000

[symbols]
assets = ["BTC", "ETH", "SOL"]

[features]
emission_interval_ms = 100        # Emit features every 100ms
trade_buffer_seconds = 60         # Keep 60s of trades in memory
book_levels = 10                  # Use top 10 levels

[output]
format = "parquet"
row_group_size = 10000            # ~16 minutes at 100ms intervals
compression = "zstd"
rotate_interval = "1h"            # New file every hour
```

---

## Implementation Sequence

### Phase 1: Skeleton (Days 1-3)

```
□ Create Rust workspace
□ Implement WebSocket client (connect, subscribe, reconnect)
□ Parse WsBook, WsTrade, WsActiveAssetCtx
□ Print parsed messages to stdout (validation)
```

### Phase 2: State Management (Days 4-6)

```
□ Implement OrderBook state (update from WsBook)
□ Implement TradeBuffer (rolling window)
□ Implement ContextState (funding, OI)
□ Add ring buffers for historical values
```

### Phase 3: Feature Computation (Days 7-12)

```
□ Implement RAW_* features
□ Implement IMBALANCE_* features
□ Implement FLOW_* features
□ Implement VOL_* features
□ Implement ENT_* features (permutation entropy)
□ Implement CTX_* features
□ Unit tests for each feature
```

### Phase 4: Output (Days 13-15)

```
□ Define Arrow schema
□ Implement Parquet writer with buffering
□ File rotation logic
□ End-to-end test: WS → Features → Parquet
```

### Phase 5: Python Layer (Days 16-20)

```
□ Parquet loader
□ Simple ASMM simulator
□ Simple TrendFollow simulator
□ Labeling pipeline
□ XGBoost training script
□ Basic backtest
```

### Phase 6: Validation (Days 21-25)

```
□ Collect 1 week of data
□ Generate labels
□ Train initial model
□ Evaluate: Does entropy predict regime?
□ Iterate on feature selection
```

---

## Success Criteria for V1

| Metric | Target |
|--------|--------|
| Data collection uptime | > 99% over 1 week |
| Feature computation latency | < 10ms p99 |
| Regime classification accuracy | > 55% (better than random) |
| Sharpe ratio (regime-based vs random) | Positive delta |
| Backtest profit factor | > 1.0 |

---

## What V1 Explicitly Excludes

| Excluded | Reason | When to Add |
|----------|--------|-------------|
| L4 data | Overkill for detection | V3 (execution optimization) |
| Live trading | Prove offline first | V2 |
| Neural networks | Start simple | If XGBoost plateaus |
| Genetic evolution | Optimization, not detection | V2 |
| Multiple exchanges | Prove on one first | V2 |
| Spot markets | Focus on perps | V2 |
| Complex order types | Basic market/limit first | V2 |

---

## V1.x Extension Roadmap

After V1 base is validated, add extensions incrementally. See `EXTENSIONS.md` for full details.

### V1.1: Polynomial Chaos Expansion (Weeks 5-6)

**Additional Features (8):**
```
PCE_SOBOL_ENTROPY_FIRST          # First-order Sobol for entropy features
PCE_SOBOL_ENTROPY_TOTAL          # Total Sobol (includes interactions)
PCE_SOBOL_IMBALANCE_FIRST        # First-order for imbalance
PCE_SOBOL_IMBALANCE_TOTAL        # Total for imbalance
PCE_INTERACTION_ENT_IMB          # Entropy-imbalance interaction strength
PCE_COEFF_VARIANCE               # Variance explained by PCE
PCE_REGIME_UNCERTAINTY           # Prediction uncertainty
PCE_DOMINANT_MODE                # Which polynomial term dominates
```

**Value**: Uncertainty quantification + feature interaction discovery

### V1.2: Transfer Entropy Networks (Weeks 7-8)

**Additional Features (12):**
```
TE_NETWORK_DENSITY               # Fraction of edges present
TE_NETWORK_RECIPROCITY           # Bidirectional flow fraction
TE_TOTAL_INFORMATION_FLOW        # Sum of all TE values
TE_FLOW_ASYMMETRY               # Directional imbalance
TE_NETWORK_ENTROPY              # Edge weight entropy
TE_IMBALANCE_TO_RETURN          # Key causal: imbalance → return
TE_ENTROPY_TO_VOLATILITY        # Key causal: entropy → vol
TE_FLOW_TO_SPREAD               # Key causal: flow → spread
TE_RETURN_TO_IMBALANCE          # Feedback loop
TE_RETURN_PAGERANK              # Return node centrality
TE_DOMINANT_HUB                 # Most influential feature
TE_HIERARCHY_SCORE              # Causal hierarchy measure
```

**Value**: Causal structure discovery + regime-dependent topology

### V1.3: Information Geometry (Weeks 9-10)

**Additional Features (8):**
```
IG_HELLINGER_VELOCITY_1S         # Manifold velocity (1s)
IG_HELLINGER_VELOCITY_5S         # Manifold velocity (5s)
IG_FISHER_TRACE                  # Total information
IG_FISHER_DETERMINANT            # Information volume
IG_LOCAL_CURVATURE               # Regime boundary indicator
IG_GEODESIC_ACCELERATION         # Regime change acceleration
IG_DISTANCE_TO_CLUSTER_*         # Distance to each cluster centroid
```

**Value**: Geometric regime boundaries + natural distance metric

### V1.4: Unsupervised Regime Discovery (Weeks 11-12)

**Novel Contribution: Clustering on Information Manifold**

```
┌─────────────────────────────────────────────────────────────────────┐
│            UNSUPERVISED REGIME DISCOVERY (ORIGINAL)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INSTEAD OF:                                                        │
│  "Define MR/TF by strategy profitability → train classifier"        │
│                                                                      │
│  DO THIS:                                                           │
│  "Cluster on Fisher manifold → discover natural regimes             │
│   → THEN ask what works in each regime"                             │
│                                                                      │
│  WHY IT'S BETTER:                                                   │
│  ├── Discovers regimes you didn't know existed                     │
│  ├── No label leakage / circular reasoning                         │
│  ├── Uses geodesic distance, not Euclidean                         │
│  └── Strategy-agnostic regime definition                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Additional Features (K cluster distances):**
```
CLUSTER_0_DISTANCE               # Geodesic distance to cluster 0
CLUSTER_1_DISTANCE               # Geodesic distance to cluster 1
...
CLUSTER_K_DISTANCE               # Geodesic distance to cluster K
CLUSTER_ASSIGNMENT               # Current cluster ID
CLUSTER_CONFIDENCE               # Assignment confidence
```

**Pipeline:**
1. Select entropy-related features (10-15)
2. Estimate distribution in sliding window
3. Build geodesic (Hellinger) distance matrix
4. HDBSCAN clustering
5. Characterize each cluster
6. Backtest strategies per cluster
7. Deploy with cluster-based strategy selection

---

## Extended Feature Summary

| Phase | Category | Count | Cumulative |
|-------|----------|-------|------------|
| V1 Base | All base features | 57 | 57 |
| V1.1 | PCE | +8 | 65 |
| V1.2 | Transfer Entropy | +12 | 77 |
| V1.3 | Info Geometry | +8 | 85 |
| V1.4 | Clustering | +K (~5) | ~90 |

---

## Extended Directory Structure

```
nat/
├── PAPERS_IDEAS.md
├── ARCHITECTURE.md
├── HYPER_DOCS.md
├── V1_SPEC.md
├── EXTENSIONS.md                 # Detailed extension docs (NEW)
│
├── rust/
│   ├── Cargo.toml
│   └── ing/
│       └── src/
│           └── ...               # Base features (unchanged)
│
├── python/
│   ├── pyproject.toml
│   └── nat/
│       ├── __init__.py
│       ├── data/
│       │   └── loader.py
│       ├── labeling/
│       │   ├── strategy_sim.py
│       │   └── labeler.py
│       ├── regime/
│       │   ├── features.py
│       │   ├── classifier.py
│       │   └── evaluation.py
│       ├── extensions/           # NEW: Extension modules
│       │   ├── __init__.py
│       │   ├── pce.py            # Polynomial Chaos Expansion
│       │   ├── transfer_entropy.py  # TE network
│       │   ├── info_geometry.py  # Information geometry
│       │   └── clustering.py     # Unsupervised discovery
│       ├── backtest/
│       │   ├── simulator.py
│       │   └── metrics.py
│       └── visualization/
│           └── plots.py
│
├── config/
│   ├── ing.toml
│   └── extensions.toml           # NEW: Extension config
│
├── data/
│   ├── features/
│   ├── extended_features/        # NEW: With extension features
│   ├── clusters/                 # NEW: Cluster assignments
│   ├── labels/
│   └── models/
│
└── scripts/
    ├── run_ing.sh
    ├── compute_extensions.py     # NEW: Run extension pipeline
    ├── discover_regimes.py       # NEW: Unsupervised clustering
    ├── label.py
    ├── train.py
    └── backtest.py
```

---

## Originality Assessment

| Component | Originality | Notes |
|-----------|-------------|-------|
| Base features | LOW | Standard LOB features |
| Permutation entropy | LOW | Well-established |
| PCE for microstructure | **HIGH** | Novel application |
| Transfer entropy network | MEDIUM | Applied to LOB features |
| Information geometry | **HIGH** | Novel for market states |
| Clustering on Fisher manifold | **VERY HIGH** | Novel synthesis |
| Full integration | **HIGH** | Unique combination |

**With extensions, this project has significant publication potential.**

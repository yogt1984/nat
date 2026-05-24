# NAT User Manual — Data Collection to Cluster Analysis

## Overview

This guide covers the full workflow: collect live Hyperliquid data, validate quality, explore features, and determine whether natural market regime clusters exist.

**Data flow:**
```
Hyperliquid WebSocket → Rust Ingestor → Parquet files → Python analysis → Jupyter notebooks
```

**What gets collected:** 194 features across 14 categories (entropy, trend, illiquidity, toxicity, order flow, volatility, concentration, whale flow, liquidation risk, raw microstructure, trade flow, market context, derived signals, regime detection) for BTC, ETH, SOL at 100ms intervals.

---

## 1. Collect Data

### Start the Ingestor

```bash
# Quick start (builds release binary if needed, kills stale processes, starts ingestor)
make run

# To survive SSH disconnects (recommended for long collection):
tmux new-session -d -s ingestor 'cd /home/onat/nat && make run'
tmux attach -t ingestor     # to watch output
# Ctrl+B, D to detach without stopping
```

The ingestor connects to `wss://api.hyperliquid.xyz/ws`, subscribes to BTC/ETH/SOL, computes 194 features per tick, and writes parquet files to `data/features/YYYY-MM-DD/`.

### How Much Data Do You Need?

| Duration | Raw Rows  | 15-min Bars | Good For                          |
|----------|-----------|-------------|-----------------------------------|
| 6 hours  | ~648K     | ~72         | Quick sanity check                |
| 24 hours | ~2.6M     | ~288        | Initial cluster exploration       |
| 48 hours | ~5.2M     | ~576        | Weekday + weekend patterns        |
| 7 days   | ~18M      | ~2,016      | Full weekly cycle, robust results |

**Minimum for meaningful clustering: 24 hours** (~288 fifteen-minute bars).

### Monitor Collection

```bash
# Pipeline status (row counts, file counts, state)
make pipeline_status

# Watch files appear (rotated hourly, flushed every ~5.5 min)
ls -lh data/features/$(date +%Y-%m-%d)/

# Check ingestor is alive
pgrep -f "target/release/ing"
```

### Stop Collection

```bash
# Ctrl+C in the terminal, or:
pkill -f "target/release/ing"

# If using tmux:
tmux kill-session -t ingestor
```

### Configuration Reference

Edit `config/ing.toml` to change:
- `assets = ["BTC", "ETH", "SOL"]` — which symbols to collect
- `emission_interval_ms = 100` — feature computation frequency
- `row_group_size = 10000` — rows before flushing to disk (~5.5 min buffer)
- `rotate_interval = "1h"` — new parquet file every hour
- `compression = "zstd"` — parquet compression

---

## 2. Validate Data

### Quick Validation

```bash
# Validate all collected data (7 checks, verbose output)
make validate_data

# Validate only the last N hours
make validate_data_recent HOURS=24
```

### What Gets Checked

| Check                    | Threshold          | What It Catches                              |
|--------------------------|--------------------|----------------------------------------------|
| File Integrity           | All files readable | Corrupted parquet files (incomplete writes)   |
| Timestamp Continuity     | Gaps < 5 seconds   | Network drops, ingestor crashes               |
| NaN Ratio                | < 1% after warmup  | Missing feature computations                  |
| Feature Ranges           | Per-feature bounds  | Extreme/impossible values                     |
| Cross-Symbol Consistency | < 10% row diff     | Symbol-specific WebSocket issues              |
| Data Rate                | > 30,000 rows/hour | Ingestor slowdown or partial failure          |
| Sequence Monotonicity    | Strictly increasing | Data ordering/corruption                      |

### Schema & Vector Coverage

```bash
# Show parquet schema, row counts, and feature vector coverage
make scan_schema
```

Example output:
```
Files:          24
Total rows:     2,592,000
Columns:        194
Symbols:        ['BTC', 'ETH', 'SOL']

Vector           Found / Expected  Coverage
entropy             24 /       24      100%
trend               15 /       15      100%
volatility           8 /        8      100%
...
```

All 14 vectors should show 100% coverage. If any vector shows < 100%, some features may have been added/removed between ingestor versions.

---

## 3. Explore Features (Optional)

Before clustering, you may want to explore the raw feature distributions.

```bash
# Launch the exploration notebook
jupyter notebook notebooks/explore_features.ipynb
```

### What This Notebook Does

1. **Data Loading** — loads parquet files, lists all 194 columns
2. **Time Series** — plots feature evolution over time, rolling statistics
3. **Distributions** — histograms, Q-Q plots, outlier detection for each feature
4. **Correlations** — correlation matrix, highly correlated pairs (|r| > 0.7)
5. **Event Detection** — finds 2.5σ spikes, shows event studies (before/after)
6. **PCA** — explained variance plot, how many components for 90% variance

This is exploratory — no clustering yet. Use it to understand your data before making clustering decisions.

---

## 4. Cluster Analysis (Decision Gate)

This is the main analysis. It determines whether natural market regimes exist.

```bash
# Launch the decision gate notebook
jupyter notebook notebooks/cluster_analysis.ipynb
```

### Notebook Walkthrough

Run cells top to bottom. Each section builds on the previous.

#### Cell Group 1: Configuration

Set your parameters:
```python
DATA_DIR = "../data/features"
TIMEFRAME = "15min"           # Try: 5min, 15min, 1h, 4h
VECTORS_TO_TEST = [
    "entropy",      # 24 features: tick/permutation/conditional entropy
    "trend",        # 15 features: momentum, Hurst, R², moving averages
    "volatility",   # 8 features: realized vol, Parkinson, spread vol
    "toxicity",     # 10 features: VPIN, adverse selection, spreads
    "regime",       # 20 features: absorption, divergence, churn
    "micro",        # composite: entropy + volatility + flow (44 features)
]
K_RANGE = range(2, 11)       # Test k=2 through k=10
SCALER = "zscore"             # zscore, minmax, robust, none
```

#### Cell Group 2: Data Loading & Schema Check

Loads parquet files, validates schema, shows vector coverage. You'll see:
- Total rows loaded
- Which vectors are available
- Any NaN warnings

#### Cell Group 3: Bar Aggregation

Converts 100ms ticks into time bars. Smart aggregation per feature type:
- Price columns → OHLC (open, high, low, close)
- Volume/count → sum
- Entropy → mean + std + OLS slope (trend within bar)
- Whale flow → cumulative sum
- Default → mean + std + last

Also shows a multi-timeframe comparison table (5min/15min/1h/4h).

#### Cell Group 4: Per-Vector Cluster Analysis (Main Loop)

For each feature vector, runs:

1. **Preprocessing** — NaN removal, zero-variance filtering, outlier clipping (5σ), z-score scaling
2. **k-Sweep** — GMM with k=2..10, evaluates silhouette, BIC, Davies-Bouldin, Calinski-Harabasz
3. **Best-k Selection** — picks k by BIC elbow
4. **Quality Metrics** — silhouette score (how well-separated), Davies-Bouldin (inter/intra ratio)
5. **Bootstrap Stability** — re-clusters 50 random resamples, measures Adjusted Rand Index (ARI)
6. **Temporal Stability** — clusters first-half vs second-half, measures ARI
7. **Multimodality Scan** — Hartigan's dip test and bimodality coefficient per feature
8. **Dimensionality Reduction** — PCA + t-SNE for 2D visualization
9. **Diagnostic Plots** — scatter, silhouette, centroid heatmap, k-sweep, dendrogram, pairplot

**Key outputs per vector:**
```
VECTOR: entropy
  Shape after preprocess: (288, 24)
  Best k (BIC):       3
  Silhouette:         0.3142  (threshold: 0.25)
  Bootstrap ARI:      0.7234  (threshold: 0.60)
  Temporal ARI:       0.5891  (threshold: 0.50)
  Cluster sizes:      [98, 112, 78]
```

#### Cell Group 5: Predictive Quality

Tests whether clusters predict forward returns:
- **Kruskal-Wallis test** — are return distributions different across clusters?
- **Eta-squared** — how much variance in returns is explained by cluster membership?
- **Self-transition rate** — do regimes persist (> 70% same-cluster in next bar)?
- **Per-cluster Sharpe** — which clusters are profitable?

#### Cell Group 6: Comparison Grid

2×2 visualization for the best vector:
- Top-left: cluster labels on PCA projection
- Top-right: entropy overlay (continuous color)
- Bottom-left: forward return sign (+/-)
- Bottom-right: symbol coloring (BTC/ETH/SOL)

Shows whether cluster structure aligns with entropy, returns, and cross-asset behavior.

#### Cell Group 7: Multi-Timeframe

Runs the best vector across 5min, 15min, 1h, 4h. Answers: does regime structure depend on timeframe?

#### Cell Group 8: Decision Gate

Final summary table:

| Vector     | k  | Silhouette | Boot ARI | Temp ARI | Q1  | Q2  | Q3  |
|------------|----|-----------:|----------|----------|-----|-----|-----|
| entropy    | 3  | 0.3142     | 0.7234   | 0.5891   | YES | YES | YES |
| trend      | 2  | 0.2011     | 0.4512   | 0.3890   | no  | no  | no  |
| ...        |    |            |          |          |     |     |     |

**Decision rules:**

| Decision  | Condition                                           | Next Step                              |
|-----------|-----------------------------------------------------|----------------------------------------|
| **GO**    | Majority vectors pass Q1+Q2, at least one passes Q3 | Build online phase detector with GMM   |
| **PIVOT** | Clusters exist (Q1+Q2) but don't predict returns     | Try different features/timeframes      |
| **NO-GO** | No reliable clusters found                           | Longer collection or different approach |

---

## 5. Visualization Reference

All plots are generated automatically in the notebook. Here's what each one shows:

| Plot               | What It Shows                                                   | What to Look For                                  |
|--------------------|-----------------------------------------------------------------|---------------------------------------------------|
| **k-Sweep**        | Silhouette + BIC vs k                                           | Clear elbow in BIC, silhouette peak               |
| **Scatter 2D**     | PCA/t-SNE projection colored by cluster                         | Separated blobs = good; overlapping = weak        |
| **Silhouette**     | Per-sample silhouette grouped by cluster                        | All bars positive; uniform width across clusters   |
| **Centroid Heatmap** | Z-scored cluster centers (features × clusters)                 | Distinct color patterns per cluster                |
| **Dendrogram**     | Hierarchical clustering tree                                    | Clear branching = natural hierarchy                |
| **Pairplot**       | Scatter matrix of top features colored by cluster               | Visible separation in multiple feature pairs       |
| **Comparison Grid** | Same projection, 4 colorings (cluster/entropy/returns/symbol)  | Cluster coloring should correlate with returns     |

---

## 6. Available Feature Vectors

### Base Vectors (14)

| Vector        | Features | Description                                           |
|---------------|----------|-------------------------------------------------------|
| entropy       | 24       | Tick, permutation, conditional entropy; rate-of-change |
| trend         | 15       | Momentum, monotonicity, Hurst exponent, moving averages |
| illiquidity   | 12       | Kyle's lambda, Amihud, Hasbrouck, Roll spread         |
| toxicity      | 10       | VPIN, adverse selection, effective/realized spread     |
| orderflow     | 8        | L1/L5/L10 imbalance, pressure, depth-weighted         |
| volatility    | 8        | Realized vol, Parkinson, spread vol, vol ratio        |
| concentration | 15       | Gini, HHI, Top-K, Theil index, whale ratios          |
| whale         | 12       | Net flow 1h/4h/24h, momentum, intensity, buy ratio   |
| liquidation   | 13       | Risk mapping, cascade probability, cluster distance   |
| raw           | 10       | Midprice, spread, microprice, depths                  |
| flow          | 12       | Volume, VWAP, aggressor ratio, trade intensity        |
| context       | 9        | Funding rate, open interest, premium, basis           |
| derived       | 15       | Regime interactions, composite scores                 |
| regime        | 20       | Absorption, divergence, churn, range position         |

### Composite Vectors (3)

| Vector | Combines                          | Features |
|--------|-----------------------------------|----------|
| micro  | entropy + volatility + flow       | 44       |
| macro  | regime + whale + context          | 41       |
| full   | all 14 base vectors               | 183      |

---

## 7. Makefile Quick Reference

```bash
# ── Data Collection ──
make run                    # Build + start ingestor
make run_and_serve          # Ingestor + dashboard on :8080

# ── Validation ──
make validate_data          # Full 7-point validation
make validate_data_recent HOURS=24  # Last N hours only
make scan_schema            # Schema + vector coverage

# ── Analysis ──
make explore                # Launch explore_features.ipynb
make analyze_clusters SYMBOL=BTC HOURS=24  # Quick cluster quality check
make analyze_all_symbols    # Analyze BTC, ETH, SOL

# ── Pipeline (Automated) ──
make pipeline_start         # Full automated: ingest → validate → analyze
make pipeline_resume        # Resume after interruption
make pipeline_analyze       # Skip ingestion, analyze existing data
make pipeline_stop          # Stop ingestor, keep data
make pipeline_status        # Show current state

# ── Testing ──
make test_pipeline          # Run all 504 cluster pipeline tests
make test_pipeline_cov      # Tests with coverage report
```

---

## 8. Phase 1 — Signal Existence Test

Before building strategies, you need to answer one question: **do the features predict returns out-of-sample?** If not, nothing else matters.

### What It Does

Trains a LightGBM classifier to predict the direction (up/down) of the next N-second return using your 191 features. Runs three tests:

| Test | What It Measures | What to Look For |
|------|-----------------|------------------|
| **In-sample accuracy** | Can the model fit training data? | >55% = model can learn patterns |
| **Walk-forward validation** | Does it work on unseen future data? | Accuracy > base rate = real signal |
| **Confidence-filtered PnL** | Is signal profitable after costs? | Net(maker) > 0 at any threshold = tradeable |

### Run It

```bash
# Default: BTC, 5-minute horizon, all features
make signal_test

# Remove leaky features (midprice, OI, volume_24h) — the honest test
make signal_test REMOVE_LEAKY=1

# Test different symbols
make signal_test SYMBOL=ETH
make signal_test SYMBOL=SOL

# Test different horizons
make signal_test HORIZON=18000                # 30 minutes
make signal_test HORIZON=36000                # 1 hour

# Full sweep: all symbols, with and without leaky features
make signal_test_all
```

### Reading the Output

**Test 2 (Walk-Forward)** is the most important. Look at:
- `edge` = accuracy minus base rate. Positive = model learned something real.
- Consistency across splits — all positive edges is better than one huge and four negative.

**Test 3 (Confidence-Filtered)** answers "can I trade this?":
- `Net(taker)` > 0 at any threshold → profitable with market orders
- `Net(maker)` > 0 at any threshold → profitable with limit orders only
- Neither positive → signal exists but doesn't survive costs yet. Need more data or longer horizon.

### What We Know So Far (5.5 days of data, April 2026)

| Config | Walk-Forward Edge | Best Confidence Acc | Net(maker) at 0.80 |
|--------|-------------------|--------------------|--------------------|
| All features | +4.18% | 54.2% | -0.45 bps |
| Leaky removed | +4.61% | 50.5% | -0.76 bps |

**Interpretation:** Real signal exists in the microstructure features (+4.6% edge survives removing leaky features). Not yet profitable after costs — needs more data for confidence calibration to work.

---

## 9. One-Week Execution Plan

This is the concrete sequence to go from "data exists" to "signal validated." No ambiguity, just commands.

### Prerequisites

- Machine: su-35 (or wherever the ingestor runs)
- The ingestor binary builds and connects (`make run` works)
- You have screen/tmux for persistent sessions

### Day 0 (Now) — Start Collection

```bash
# Step 1: Open a persistent session so ingestor survives SSH disconnect
tmux new-session -d -s nat 'cd /home/onat/nat && make run'

# Step 2: Verify it's running (should see the ing process)
pgrep -f "target/release/ing" && echo "RUNNING" || echo "NOT RUNNING"

# Step 3: Verify data is being written (wait 6 minutes for first flush)
sleep 360 && ls -lh data/features/$(date +%Y-%m-%d)/

# Step 4: Attach to watch logs (Ctrl+B, D to detach)
tmux attach -t nat
```

**Leave it running. Do not touch it for 3 days.**

### Day 3 — Health Check

```bash
# Check ingestor is still alive
pgrep -f "target/release/ing" && echo "RUNNING" || echo "DEAD — restart with: tmux new-session -d -s nat 'cd /home/onat/nat && make run'"

# Check data volume (should be ~1.5 GB after 3 days)
du -sh data/features/
ls data/features/

# Validate recent data
make validate_data_recent HOURS=24

# Quick signal test (optional — just to see progress)
make signal_test HORIZON=3000
```

If the ingestor died, restart it immediately:
```bash
tmux new-session -d -s nat 'cd /home/onat/nat && make run'
```

### Day 5 — Mid-Week Check

```bash
# Same health check
pgrep -f "target/release/ing" && echo "RUNNING" || echo "DEAD"
du -sh data/features/

# Validate all collected data
make validate_data

# Check schema coverage
make scan_schema
```

### Day 7 — Full Analysis

```bash
# Step 1: Validate the full dataset
make validate_data
make scan_schema

# Step 2: Signal test — the main event
#   Run all three: default, leaky-removed, and full sweep
make signal_test HORIZON=3000
make signal_test HORIZON=3000 REMOVE_LEAKY=1
make signal_test HORIZON=18000
make signal_test HORIZON=18000 REMOVE_LEAKY=1

# Step 3: Test all symbols
make signal_test_all

# Step 4: Cluster analysis
make analyze_clusters SYMBOL=BTC HOURS=168
make analyze_clusters SYMBOL=ETH HOURS=168
make analyze_clusters SYMBOL=SOL HOURS=168

# Step 5: Full pipeline analysis (automated)
make pipeline_analyze
```

### Decision After Day 7

Read the signal test output and decide:

| Result | Meaning | Action |
|--------|---------|--------|
| Walk-forward edge > 3%, Net(maker) > 0 at any threshold | **Signal is tradeable** | Build strategy, paper trade |
| Walk-forward edge > 3%, Net(maker) < 0 | **Signal exists, not yet profitable** | Extend to 30 days, try longer horizons |
| Walk-forward edge < 1% | **No signal at this horizon** | Try 30min/1h horizons, add lagged features |
| Edge negative or inconsistent across splits | **No signal** | Need fundamentally different features or more data |

### DO NOT stop the ingestor after Day 7.

More data always helps. Keep collecting while you analyze. The goal is 30+ days covering multiple market regimes (trend, chop, crash, squeeze). Every day of data makes the next signal test more reliable.

---

## 10. Previous Typical Session

```bash
# 1. Start collecting data (let run 24+ hours)
tmux new-session -d -s ingestor 'cd /home/onat/nat && make run'

# 2. Check on it periodically
make pipeline_status
make validate_data_recent HOURS=6

# 3. After 24+ hours, stop and validate
pkill -f "target/release/ing"
make validate_data
make scan_schema

# 4. Run cluster analysis
jupyter notebook notebooks/cluster_analysis.ipynb
# → Run all cells top to bottom
# → Read the Decision Gate at the bottom: GO / PIVOT / NO-GO

# 5. If you want deeper feature exploration first
jupyter notebook notebooks/explore_features.ipynb
```

---

## 11. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `scan_schema` crashes | Corrupted parquet file | Move bad files to `data/quarantine/` |
| 0-byte parquet files | Ingestor killed before flush | Delete them; data is in other files |
| NaN ratio > 1% | Warmup period (first 60s) | Normal if only in first file |
| Low data rate | Network issues or API throttling | Check `RUST_LOG=debug make run` |
| `ModuleNotFoundError` | Wrong Python | Makefile uses conda python automatically |
| Notebook can't import cluster_pipeline | Path issue | Ensure you run from `notebooks/` directory |
| `signal_test` takes >10 min | Large dataset + 200 trees × 7 fits | Normal for 400k+ rows; be patient |
| Target distribution very skewed (e.g. 27/73) | Short horizon during a trend | Increase `HORIZON` to 3000+ to balance |
| Walk-forward edge is negative | Model is wrong about direction | Try longer horizon, remove leaky features |
| `ModuleNotFoundError: lightgbm` | Not installed in conda env | `conda install -c conda-forge lightgbm` |

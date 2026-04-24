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

## 8. Typical Session

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

## 9. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `scan_schema` crashes | Corrupted parquet file | Move bad files to `data/quarantine/` |
| 0-byte parquet files | Ingestor killed before flush | Delete them; data is in other files |
| NaN ratio > 1% | Warmup period (first 60s) | Normal if only in first file |
| Low data rate | Network issues or API throttling | Check `RUST_LOG=debug make run` |
| `ModuleNotFoundError` | Wrong Python | Makefile uses conda python automatically |
| Notebook can't import cluster_pipeline | Path issue | Ensure you run from `notebooks/` directory |

# Cluster Analysis Pipeline — Requirements Analysis

**Date:** 2026-04-13
**Goal:** Load parquet feature data, define feature vectors by category, cluster them using multiple methods, visualize clusters in 2D/3D.
**Deliverable:** Self-contained Python scripts the user owns and controls.

---

## 1. Problem Statement

We have 5 weeks of tick-level microstructure data (3 symbols: BTC, ETH, SOL) written as hourly parquet files by the Rust ingestor. Each row contains up to 183 features across 14 categories. We want to:

1. Define meaningful **feature vectors** (subsets of the 183 features grouped by domain)
2. **Cluster** these vectors using multiple algorithms
3. **Visualize** clusters in 2D and 3D projections
4. Determine whether natural market regimes exist in the data

---

## 2. Data Source

### 2.1 Parquet Schema (from Rust ingestor)

Each parquet file contains rows with these columns:

| Category | Prefix | Count | Column Names |
|----------|--------|-------|-------------|
| **Meta** | — | 3 | `timestamp_ns`, `timestamp`, `symbol` |
| **Raw** | `raw_` | 10 | midprice, spread, spread_bps, microprice, bid/ask_depth_5/10, bid/ask_orders_5 |
| **Imbalance** | `imbalance_` | 8 | qty_l1/l5/l10, orders_l5, notional_l5, depth_weighted, pressure_bid/ask |
| **Flow** | `flow_` | 12 | count_1s/5s/30s, volume_1s/5s/30s, aggressor_ratio_5s/30s, vwap_5s, vwap_deviation, avg_trade_size_30s, intensity |
| **Volatility** | `vol_` | 8 | returns_1m/5m, parkinson_5m, spread_mean/std_1m, midprice_std_1m, ratio_short_long, zscore |
| **Entropy** | `ent_` | 24 | permutation_returns_8/16/32, permutation_imbalance_16, spread/volume/trade_size_dispersion, book_shape, rate_of_change_5s, zscore_1m, tick_1s/5s/10s/15s/30s/1m/15m, vol_tick_1s/5s/10s/15s/30s/1m/15m |
| **Context** | `ctx_` | 9 | funding_rate, funding_zscore, open_interest, oi_change_5m, oi_change_pct_5m, premium_bps, volume_24h, volume_ratio, mark_oracle_divergence |
| **Trend** | `trend_` | 15 | momentum_60/300/600, momentum_r2_60/300/600, monotonicity_60/300/600, hurst_300/600, ma_crossover, ma_crossover_norm, ema_short, ema_long |
| **Illiquidity** | `illiq_` | 12 | kyle/amihud/hasbrouck/roll at 100/500 windows, kyle/amihud_ratio, composite, trade_count |
| **Toxicity** | `toxic_` | 10 | vpin_10/50, vpin_roc, adverse_selection, effective/realized_spread, flow_imbalance, flow_imbalance_abs, index, trade_count |
| **Derived** | `derived_` | 15 | entropy_trend_interaction/zscore, trend_strength_60/300/ratio, entropy_volatility_ratio, regime_type_score, illiquidity_trend, informed_trend_score, toxicity_regime, toxic_chop_score, trend_strength_roc, entropy_momentum, regime_indicator/confidence |
| **Whale Flow** | `whale_` | 12 | net_flow_1h/4h/24h, flow_normalized_1h/4h, flow_momentum/intensity/roc, buy_ratio, directional_agreement, active_count, total_activity |
| **Liquidation** | `liquidation_` | 13 | risk_above/below_1/2/5/10pct, asymmetry, intensity, positions_at_risk_count, largest_position_at_risk, nearest_cluster_distance |
| **Concentration** | — | 15 | top5/10/20/50_concentration, herfindahl_index, gini_coefficient, theil_index, whale_retail_ratio, whale_fraction, whale_avg_size_ratio, concentration_change_1h, hhi_roc, concentration_trend, position_count, whale_position_count |
| **Regime** | `regime_` | 20 | absorption_1h/4h/24h/zscore, divergence_1h/4h/24h/zscore, kyle_lambda, churn_1h/4h/24h/zscore, range_pos_4h/24h/1w, range_width_24h, accumulation/distribution_score, clarity |

**Total:** ~183 features + 3 meta columns

### 2.2 Data Volume Estimates (5 weeks)

- Emission rate: 100ms → 10 rows/sec
- Per day: ~864,000 rows/symbol
- Per 5 weeks: ~30M rows/symbol, ~90M rows total
- Hourly parquet files: ~840 files per symbol, ~2,520 total
- Compressed size: ~500MB–1.5GB total (zstd)

### 2.3 First Task: Verify Schema

Before any analysis, the pipeline must inspect one live parquet file and confirm which columns actually exist. The Rust ingestor has optional features (whale_flow, liquidation, concentration, regime) that may or may not be present depending on configuration. The synthetic data only has 25 columns — live data will differ.

---

## 3. Feature Vector Definitions

The core idea: instead of clustering all 183 features at once (curse of dimensionality), define **domain-specific feature vectors** that group related features. Each vector represents a different "lens" on market state.

### 3.1 Feature Vectors

| Vector Name | Features | Dim | Hypothesis |
|-------------|----------|-----|-----------|
| **entropy** | All 24 `ent_*` columns | 24 | Market uncertainty has discrete modes (ordered vs chaotic) |
| **volatility** | All 8 `vol_*` columns | 8 | Volatility regimes (calm, normal, stressed) exist |
| **flow** | All 12 `flow_*` columns | 12 | Trade flow patterns cluster into buying/selling/balanced |
| **orderbook** | All 8 `imbalance_*` + `raw_spread*`, `raw_*depth*` | ~14 | Book shape reveals supply/demand regimes |
| **toxicity** | All 10 `toxic_*` columns | 10 | Informed vs uninformed trading clusters |
| **trend** | All 15 `trend_*` columns | 15 | Trending vs mean-reverting vs choppy |
| **illiquidity** | All 12 `illiq_*` columns | 12 | Liquid vs illiquid regimes |
| **regime** | All 20 `regime_*` columns | 20 | Accumulation/distribution/ranging phases |
| **whale** | All 12 `whale_*` + 15 concentration cols | 27 | Smart money positioning clusters |
| **derived** | All 15 `derived_*` columns | 15 | Cross-domain interaction patterns |
| **micro** | entropy + volatility + flow (combined) | 44 | Core microstructure state |
| **macro** | regime + whale + context | 56 | Higher-level market structure |
| **full** | All available numeric features | ~183 | Full-dimensional clustering (baseline) |

### 3.2 Aggregation Requirement

Raw tick data (100ms) is too granular for meaningful clustering:
- Consecutive ticks are nearly identical (autocorrelation ~0.99)
- 90M rows won't fit in memory for clustering
- Tick-level noise dominates signal

**Required:** Aggregate to time bars before clustering.

| Timeframe | Rows per symbol (5wk) | Total rows (3 sym) | Use case |
|-----------|----------------------|--------------------|---------| 
| 5 min | ~10,080 | ~30,240 | Fine-grained intraday |
| 15 min | ~3,360 | ~10,080 | Primary analysis horizon |
| 1 hour | ~840 | ~2,520 | Swing-level regimes |
| 4 hour | ~210 | ~630 | Macro regime detection |

For each bar, aggregate each feature as: **mean, std, min, max, last** (or a subset). This multiplies dimensionality, so for clustering we primarily use **mean** (the bar's central tendency) and optionally **std** (intra-bar variability).

---

## 4. Clustering Methods

### 4.1 Algorithms to Implement

| Algorithm | Type | Strengths | Hyperparameters |
|-----------|------|-----------|-----------------|
| **GMM** (Gaussian Mixture) | Parametric, soft | Probabilistic assignments, BIC for model selection | n_components (1–10), covariance_type (full, diag, tied) |
| **K-Means** | Parametric, hard | Fast baseline, clear centroids | n_clusters (2–10) |
| **HDBSCAN** | Density-based | No k needed, finds noise, variable-density clusters | min_cluster_size, min_samples |
| **Spectral** | Graph-based | Finds non-convex clusters | n_clusters, affinity (rbf, nearest_neighbors) |
| **Agglomerative** | Hierarchical | Dendrogram visualization, no k needed | n_clusters or distance_threshold, linkage (ward, complete, average) |
| **DPGMM** (Dirichlet Process) | Bayesian | Automatic k selection | max_components, weight_concentration_prior |

### 4.2 Model Selection Criteria

For each vector + algorithm combination, compute:

| Metric | What it measures | Good value |
|--------|-----------------|-----------|
| **Silhouette score** | Cluster separation vs cohesion | > 0.25 meaningful, > 0.5 strong |
| **Davies-Bouldin index** | Inter/intra-cluster ratio | Lower is better, < 1.0 good |
| **Calinski-Harabasz** | Between/within variance ratio | Higher is better |
| **BIC** (GMM only) | Model complexity trade-off | Minimum across k |
| **Gap statistic** | Compares to uniform null | Largest gap |
| **Bootstrap ARI** | Cluster stability under resampling | > 0.6 stable |

### 4.3 Optimal k Selection

For each feature vector, sweep k from 1 to 10:
1. Fit GMM, record BIC → pick k at BIC minimum
2. Fit K-Means, record silhouette → pick k at silhouette maximum
3. Fit HDBSCAN (no k needed) → record number of clusters found
4. If all methods agree on k, strong evidence. If they disagree, investigate.

---

## 5. Dimensionality Reduction & Visualization

### 5.1 Projection Methods

| Method | Type | Preserves | Speed | Use |
|--------|------|-----------|-------|-----|
| **PCA** | Linear | Global variance | Fast | First look, explained variance analysis |
| **UMAP** | Nonlinear | Local + global topology | Medium | Primary visualization |
| **t-SNE** | Nonlinear | Local neighborhoods | Slow | Confirmation of UMAP structure |

### 5.2 Visualization Outputs

For each feature vector:

1. **2D scatter** — PCA, UMAP, t-SNE projections colored by cluster label
2. **3D interactive** — UMAP 3D with plotly, colored by cluster, hover shows feature values
3. **Cluster comparison grid** — same projection, different colorings:
   - Cluster label
   - Entropy level (continuous colormap)
   - Forward return sign (green/red)
   - Symbol (BTC/ETH/SOL)
   - Time-of-day or day-of-week
4. **Silhouette plot** — per-sample silhouette values, grouped by cluster
5. **Feature heatmap** — cluster centroids as rows, features as columns, z-scored
6. **Pairplot** — top 4-5 features, colored by cluster
7. **Dendrogram** — from agglomerative clustering

### 5.3 Interactive Requirements

Jupyter notebook visualizations must support:
- `plotly` for 3D rotation, zoom, hover tooltips
- `matplotlib` for static publication-quality plots
- `ipywidgets` (optional) for timeframe/vector/algorithm dropdowns

---

## 6. Pipeline Architecture

### 6.1 Scripts to Create

```
scripts/
  cluster_pipeline/
    __init__.py
    config.py            # Feature vector definitions, algorithm configs
    loader.py            # Load parquet, detect schema, select columns
    preprocess.py        # Handle NaN, scale, aggregate to bars
    cluster.py           # All clustering algorithms + metrics
    reduce.py            # PCA, UMAP, t-SNE wrappers
    viz.py               # All visualization functions
    report.py            # Generate summary tables + findings

notebooks/
    cluster_analysis.ipynb   # Main interactive notebook
```

### 6.2 Script Responsibilities

**`config.py`** — Single source of truth for feature vector definitions.
- Maps vector name → list of column name patterns (prefix matching)
- Algorithm hyperparameter grids
- Visualization color palettes
- No hardcoded column names scattered across files

**`loader.py`** — Data I/O.
- Scan parquet directory, load all files (lazy with polars for memory efficiency)
- Auto-detect which columns exist (handles optional features)
- Filter by symbol, date range
- Report schema summary (which vectors are available)

**`preprocess.py`** — Data preparation.
- Aggregate ticks to bars (5m, 15m, 1h, 4h) — **this is the critical piece**
  - For each feature: compute mean (primary), std (variability), last (close)
  - For price: OHLC
  - For volume: sum
  - Drop bars with >50% NaN
- StandardScaler normalization (fit on train, transform on test)
- NaN handling: drop rows with >20% NaN, impute remainder with column median
- Optionally compute forward returns (for later coloring)

**`cluster.py`** — Clustering engine.
- `fit_gmm(X, k_range)` → labels, probabilities, BIC curve
- `fit_kmeans(X, k_range)` → labels, silhouette curve
- `fit_hdbscan(X, min_cluster_size)` → labels, probabilities
- `fit_spectral(X, k_range)` → labels
- `fit_agglomerative(X, k_range)` → labels, linkage matrix
- `fit_dpgmm(X, max_k)` → labels, effective k
- `evaluate(X, labels)` → silhouette, DB, CH, gap statistic
- `sweep_k(X, method, k_range)` → dataframe of k vs all metrics
- `compare_methods(X, methods)` → comparison table

**`reduce.py`** — Dimensionality reduction.
- `pca_2d(X)`, `pca_3d(X)` → coords + explained variance
- `umap_2d(X)`, `umap_3d(X)` → coords
- `tsne_2d(X)` → coords
- All return a dataframe with projection columns appended

**`viz.py`** — Visualization.
- `plot_scatter_2d(coords, labels, title)` → matplotlib figure
- `plot_scatter_3d_interactive(coords, labels, features)` → plotly HTML
- `plot_comparison_grid(coords, colorings_dict)` → 2x2 subplot grid
- `plot_silhouette(X, labels)` → silhouette diagram
- `plot_centroids_heatmap(X, labels, feature_names)` → seaborn heatmap
- `plot_bic_curve(k_range, bic_values)` → elbow plot
- `plot_dendrogram(linkage_matrix)` → scipy dendrogram

**`report.py`** — Results summary.
- For each vector × algorithm: silhouette, DB, CH, optimal k
- Best vector/algorithm combination ranking
- Save as CSV + JSON

### 6.3 Notebook Flow

The Jupyter notebook `cluster_analysis.ipynb` calls these scripts:

```python
# Cell 1: Load data
from cluster_pipeline import loader, config
df = loader.load_parquet("./data/features")
loader.print_schema_summary(df)

# Cell 2: Preprocess — aggregate to 15m bars
from cluster_pipeline import preprocess
bars = preprocess.aggregate_to_bars(df, timeframe="15m")

# Cell 3: Pick a feature vector
vector_name = "entropy"  # change this to explore others
X, feature_names = preprocess.extract_vector(bars, vector_name)
print(f"{vector_name}: {X.shape[1]} features, {X.shape[0]} samples")

# Cell 4: Cluster sweep
from cluster_pipeline import cluster
results = cluster.sweep_k(X, method="gmm", k_range=range(1, 11))
cluster.plot_bic_curve(results)

# Cell 5: Fit best model
labels, probs = cluster.fit_gmm(X, k=3)
metrics = cluster.evaluate(X, labels)
print(metrics)

# Cell 6: Reduce + visualize
from cluster_pipeline import reduce, viz
coords_2d = reduce.umap_2d(X)
viz.plot_scatter_2d(coords_2d, labels, title=f"{vector_name} — UMAP 2D")

coords_3d = reduce.umap_3d(X)
viz.plot_scatter_3d_interactive(coords_3d, labels, hover_data=bars[["symbol", "timestamp"]])

# Cell 7: Compare across vectors
for vec in ["entropy", "volatility", "flow", "trend", "regime"]:
    X_v, _ = preprocess.extract_vector(bars, vec)
    labels_v, _ = cluster.fit_gmm(X_v, k="auto")
    score = cluster.evaluate(X_v, labels_v)
    print(f"{vec:15s} k={len(set(labels_v))} silhouette={score.silhouette:.3f}")
```

---

## 7. Dependencies

Single consolidated `requirements.txt` at project root:

```
# Core
numpy>=1.24
pandas>=2.0
polars>=0.20
pyarrow>=14.0
scipy>=1.11

# ML / Clustering
scikit-learn>=1.3
hdbscan>=0.8
umap-learn>=0.5

# Visualization
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15

# Notebook
jupyterlab>=4.0
ipywidgets>=8.0

# Testing
pytest>=7.0
```

**Note:** Remove the 5 existing fragmented requirements files and replace with this one.

---

## 8. Task Breakdown

### Phase 0: Data Access (prerequisite)
- [ ] **0.1** Sync live parquet data from remote machine to `./data/features/`
- [ ] **0.2** Inspect one live parquet file — confirm actual column names vs Section 2.1
- [ ] **0.3** Record which optional feature categories exist (whale, liquidation, concentration, regime)
- [ ] **0.4** Install dependencies: `pip install -r requirements.txt`

### Phase 1: Pipeline Foundation
- [ ] **1.1** Create `scripts/cluster_pipeline/config.py` — feature vector definitions (Section 3.1)
- [ ] **1.2** Create `scripts/cluster_pipeline/loader.py` — parquet loading with schema auto-detection
- [ ] **1.3** Create `scripts/cluster_pipeline/preprocess.py` — NaN handling, scaling, bar aggregation
- [ ] **1.4** Test: load data, aggregate to 15m bars, extract entropy vector, print shape

### Phase 2: Clustering Engine
- [ ] **2.1** Create `scripts/cluster_pipeline/cluster.py` — all 6 algorithms + metrics
- [ ] **2.2** Implement `sweep_k()` — iterate k=1..10 for GMM/KMeans, record BIC/silhouette/DB
- [ ] **2.3** Implement `compare_methods()` — run all algorithms on same data, tabulate results
- [ ] **2.4** Test: cluster entropy vector with GMM, print optimal k and silhouette

### Phase 3: Visualization
- [ ] **3.1** Create `scripts/cluster_pipeline/reduce.py` — PCA, UMAP, t-SNE wrappers
- [ ] **3.2** Create `scripts/cluster_pipeline/viz.py` — all plot functions
- [ ] **3.3** Test: entropy vector → UMAP 2D → scatter colored by GMM cluster
- [ ] **3.4** Test: 3D interactive plotly HTML generation

### Phase 4: Notebook Integration
- [ ] **4.1** Create `notebooks/cluster_analysis.ipynb` — full workflow (Section 6.3)
- [ ] **4.2** Run on entropy vector at 15m — document findings
- [ ] **4.3** Run on all 13 vectors — compare silhouette scores
- [ ] **4.4** Generate 3D interactive HTML for the best-clustering vector
- [ ] **4.5** Run at 5m, 1h, 4h — compare if clusters persist across timeframes

### Phase 5: Cross-Analysis
- [ ] **5.1** Per-symbol clustering: do BTC/ETH/SOL cluster differently?
- [ ] **5.2** Color clusters by forward return — do clusters predict direction?
- [ ] **5.3** Cluster transition matrix — how persistent are regimes?
- [ ] **5.4** Write findings to `reports/cluster_findings.md`

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Live schema differs from Rust code | Loader crashes, wrong columns | Auto-detect columns in loader.py, pattern-match prefixes |
| 90M rows won't fit in memory | OOM on load | Use polars lazy scan, aggregate before loading full dataset |
| All features are NaN for optional categories | Empty vectors | Skip vectors with <50% valid columns |
| Clusters are artifacts of normalization | False positives | Compare results with/without scaling, test on permuted data |
| UMAP is slow on large datasets | Long notebook cells | Subsample to 10K rows for visualization, full data for metrics |
| No real clusters exist | Wasted effort | Silhouette < 0.1 means no structure — accept the null result early |

---

## 10. Success Criteria

**Minimum viable result:** For at least one feature vector at one timeframe:
- GMM BIC selects k > 1
- Silhouette score > 0.2
- Clusters visible in UMAP 2D projection
- Two people looking at the plot would agree "there are groups"

**Strong result:** Additionally:
- Bootstrap ARI > 0.5 (clusters are stable)
- Clusters differ in forward return distribution (Kruskal-Wallis p < 0.05)
- Same structure appears at 2+ timeframes

**Null result (also valuable):** No feature vector produces silhouette > 0.15 at any timeframe. This means the feature space is continuous, not clustered — regimes may not exist in this data, or the features don't capture them. This saves months of building on a false premise.

# NAT Consolidated Specification

**Created:** 2026-04-15
**Status:** Living Document — consolidates all in-progress specs
**Sources:** 10 documents from `docs/in_progress/`

---

## 1. System Overview

NAT (Next-Gen Alpha Technology) is a crypto market microstructure data system that:

1. **Collects** tick-level data from Hyperliquid perpetual futures (BTC, ETH, SOL) via WebSocket
2. **Computes** 183 features across 14 categories in Rust at 100ms emission rate
3. **Stores** hourly-rotated zstd-compressed Parquet files
4. **Analyzes** feature vectors via clustering, dimensionality reduction, and statistical tests
5. **Decides** whether natural market regimes exist and predict returns (GO/PIVOT/NO-GO gate)

**Core philosophy:** Data dictates algorithms, not the reverse. Validate before building.

---

## 2. Feature Specification

### 2.1 Complete Feature Inventory (183 features, 14 categories)

| Category | Prefix | Count | Primary Use |
|----------|--------|-------|-------------|
| **Entropy** | `ent_` | 24 (20 active) | Regime detection, predictability measurement |
| **Trend** | `trend_` | 15 | Momentum vs mean-reversion classification |
| **Illiquidity** | `illiq_` | 12 | Price impact, informed flow detection |
| **Toxicity** | `toxic_` | 10 | Adverse selection, order flow quality |
| **Order Flow** | `imbalance_` | 8 | Short-term directional pressure |
| **Volatility** | `vol_` | 8 | Risk regime classification |
| **Concentration** | (mixed) | 15 | Position crowding, whale dominance (Hyperliquid-unique) |
| **Whale Flow** | `whale_` | 12 | Smart money tracking (Hyperliquid-unique) |
| **Liquidation** | `liquidation_` | 13 | Cascade prediction (Hyperliquid-unique) |
| **Raw** | `raw_` | 10 | Bid-ask dynamics, depth |
| **Flow** | `flow_` | 12 | Execution patterns, volume |
| **Context** | `ctx_` | 9 | Funding, OI, basis |
| **Derived** | `derived_` | 15 | Cross-domain interaction terms |
| **Regime** | `regime_` | 20 | Wyckoff accumulation/distribution phases |

**Meta columns:** `timestamp_ns`, `timestamp`, `symbol`

### 2.2 Known Issues

4 of 24 entropy features are hardcoded to 0.0 (need history buffers not yet implemented):
- `ent_permutation_imbalance_16`
- `ent_spread_dispersion`
- `ent_rate_of_change_5s`
- `ent_zscore_1m`

### 2.3 Hyperliquid-Unique Features

Concentration, Whale Flow, and Liquidation vectors require the PositionTracker polling wallet addresses via Hyperliquid's API. These may be N/A if not configured.

### 2.4 Feature Vector Definitions

14 primary vectors + 3 composites defined in `scripts/cluster_pipeline/config.py`:

| Vector | Dim | Hypothesis |
|--------|-----|-----------|
| **entropy** | 24 | Market uncertainty has discrete modes |
| **trend** | 15 | Trending vs mean-reverting vs choppy states |
| **illiquidity** | 12 | Liquid vs illiquid regimes |
| **toxicity** | 10 | Informed vs uninformed trading clusters |
| **orderflow** | 8 | Book shape reveals supply/demand regimes |
| **volatility** | 8 | Calm, normal, stressed volatility regimes |
| **concentration** | 15 | Position crowding patterns |
| **whale** | 12 | Smart money positioning clusters |
| **liquidation** | 13 | Liquidation risk landscape |
| **raw** | 10 | Price/depth microstructure |
| **flow** | 12 | Trade flow patterns |
| **context** | 9 | Market-level conditions |
| **derived** | 15 | Cross-domain interactions (warning: double-counts sources) |
| **regime** | 20 | Accumulation/distribution/ranging phases |

Composites:
- **micro** = entropy + volatility + flow (44d)
- **macro** = regime + whale + context (41d)
- **full** = all 183 features

### 2.5 Feature Algorithms (Key Details)

**Entropy:** Bandt-Pompe permutation entropy (ordinal patterns, normalized [0,1]), Shannon distribution entropy (spread, volume, book shape), tick entropy at 7 windows (1s-15m) both raw and volume-weighted. Multi-window = multi-resolution analysis similar to wavelet decomposition.

**Trend:** OLS regression slope + R-squared at 60/300/600 tick windows, monotonicity (directional consistency), Hurst exponent via R/S analysis (H>0.5=trending, H<0.5=mean-reverting, most spectral feature), EMA crossover.

**Illiquidity:** Kyle's lambda (Kyle 1985), Amihud (2002), Hasbrouck (2009), Roll spread (1984) at 100/500 trade windows + ratios + composite.

**Toxicity:** VPIN (Easley, Lopez de Prado, O'Hara 2012) at windows 10/50, adverse selection, effective/realized spread, flow imbalance, composite index.

**Regime:** Absorption (volume/price change), divergence (deviation from Kyle's lambda), churn (two-sided volume), range position (price within 4h/24h/1w range), composite accumulation/distribution scores.

---

## 3. Data Pipeline Specification

### 3.1 Data Volume (5 weeks collection)

- Emission: 100ms -> 10 rows/sec
- Per day: ~864K rows/symbol
- 5 weeks: ~30M rows/symbol, ~90M total
- ~2,520 hourly Parquet files total
- ~500MB-1.5GB compressed (zstd)

### 3.2 Aggregation Pipeline (THE BLOCKER)

Raw tick data must be aggregated to time bars before clustering. This is the single highest-priority missing component.

**Timeframes:**

| Timeframe | Rows/symbol (5wk) | Total (3 sym) | Use |
|-----------|-------------------|---------------|-----|
| 5 min | ~10,080 | ~30,240 | Fine-grained intraday |
| 15 min | ~3,360 | ~10,080 | Primary analysis horizon |
| 1 hour | ~840 | ~2,520 | Swing-level regimes |
| 4 hour | ~210 | ~630 | Macro regime detection |

**Aggregation rules per bar:**
- Features: mean (primary), std (variability), last (close)
- Price: OHLC
- Volume: sum
- Entropy: mean + slope
- Whale flow: sum (cumulative)

**Script:** `scripts/aggregate_bars.py` (TO BE IMPLEMENTED)

### 3.3 Cluster Pipeline Architecture

```
scripts/cluster_pipeline/
    __init__.py          # Package init (EXISTS)
    config.py            # 14 feature vectors + extraction API (EXISTS, 117 tests passing)
    loader.py            # Parquet loading, schema auto-detection (MISSING)
    preprocess.py        # NaN handling, scaling, bar aggregation (MISSING)
    cluster.py           # GMM, HDBSCAN, metrics, k-sweep (MISSING)
    reduce.py            # PCA, UMAP, t-SNE wrappers (MISSING)
    viz.py               # All visualization functions (MISSING)
    report.py            # Summary tables + findings (MISSING)

notebooks/
    cluster_analysis.ipynb  # Interactive notebook (MISSING)
```

---

## 4. Clustering Specification

### 4.1 Algorithm Selection

**Primary: GMM** (Gaussian Mixture Model)
- Soft assignments (probabilities), BIC for k selection, handles elliptical clusters
- Sweep k=1..10, pick minimum BIC

**Secondary: HDBSCAN**
- No k needed, identifies noise, variable-density clusters
- Validates GMM's cluster count (convergent evidence)

**Diagnostic: Agglomerative Ward**
- Dendrogram reveals natural hierarchy
- Don't use assignments directly

**Skip everything else initially.** Two algorithms that agree > ten that disagree.

### 4.2 Quality Metrics

**Internal (no ground truth):**

| Metric | Good Value |
|--------|-----------|
| Silhouette | > 0.25 meaningful, > 0.5 strong |
| Davies-Bouldin | < 1.0 |
| Calinski-Harabasz | Higher is better |
| BIC (GMM) | Minimum across k |
| Gap statistic | Largest gap |

**Stability:**

| Metric | Target |
|--------|--------|
| Bootstrap ARI (50 resamples) | > 0.6 |
| Temporal ARI (first/second half) | > 0.5 |
| Perturbation robustness | ARI > 0.7 |

**Predictive:**

| Metric | Target |
|--------|--------|
| Kruskal-Wallis on forward returns | p < 0.05 |
| Eta-squared (effect size) | > 0.01 |
| Cluster-conditional Sharpe | Meaningfully different |
| Self-transition rate | > 0.7 (regimes persist) |

**Multimodality (per feature):**

| Test | Threshold |
|------|-----------|
| Hartigan's dip test | p < 0.05 = multimodal |
| Bimodality coefficient | BC > 5/9 |
| GMM BIC k=1 vs k=2 | Direct cluster existence test |

### 4.3 Subspace Strategy

Cluster subspaces, not the full 183D space:

| Subspace | Expected Clustering |
|----------|-------------------|
| Entropy only | Moderate (2 clusters via BIC on synthetic) |
| Volatility only | Good (low-vol / high-vol well-documented) |
| Flow only | Weak at tick level |
| Entropy + volatility + whale flow | Best composite candidate |

---

## 5. Dimensionality Reduction & Visualization

### 5.1 Projection Methods

| Method | Preserves | Speed | Use |
|--------|-----------|-------|-----|
| **PCA** | Global variance | Fast | Always first |
| **UMAP** | Local + global topology | Medium | Primary visualization |
| **t-SNE** | Local neighborhoods | Slow | Confirmation |

**UMAP parameters:** n_neighbors=15, min_dist=0.1, metric="euclidean", n_components=2 or 3.

**Critical caveats:** Inter-cluster distances NOT meaningful. Cluster sizes NOT meaningful. Run multiple seeds.

### 5.2 Visualization Protocol

For each feature vector, generate:
1. 2D scatter (PCA, UMAP, t-SNE) colored by cluster
2. 3D interactive (UMAP + Plotly) with hover tooltips
3. Comparison grid: same projection, 4 colorings:
   - Cluster label
   - Entropy level (continuous)
   - Forward return sign (green/red)
   - Symbol (BTC/ETH/SOL)
4. Silhouette plot per cluster
5. Centroid heatmap (clusters x features, z-scored)
6. Pairplot of top 4-5 features
7. Dendrogram from agglomerative

**Key test:** If the same structure appears regardless of coloring, the clusters are real.

### 5.3 Frontend Stack (Progressive)

1. **Now:** Jupyter + Plotly + matplotlib (zero overhead)
2. **If clusters real:** Streamlit app (single Python file)
3. **If proven signal:** FastAPI + React + Three.js/Deck.gl (real-time dashboard)

**Do not build a web frontend before clusters are validated.**

---

## 6. Statistical Analysis Framework

### 6.1 Hierarchical Analysis Architecture

```
Level 0: Raw data ingestion
Level 1: Entropy distribution analysis (primary clustering)
    - Is entropy multimodal? GMM BIC, dip test, KDE modes
    - What are natural percentiles? (NOT 0.3/0.7)
    - How persistent? ACF half-life
    - Is it stationary? ADF test
Level 2: Within-cluster characterization
    - Volatility per cluster
    - Trend continuity per cluster
    - Return distribution per cluster
    - Feature correlations per cluster
Level 3: Cross-cluster dynamics
    - Transition probabilities
    - Change-point detection
    - Lead-lag relationships
Level 4: Predictive relationship analysis
    - Which features lead returns?
    - Cluster-conditional predictability
    - Causal validation (Granger, cross-correlation)
Level 5: Feature selection
    - PCA within clusters
    - Redundancy elimination (corr > 0.8)
    - Stability selection (bootstrap)
```

### 6.2 Data-First Philosophy

```
WRONG: Hypothesis -> Algorithm -> Data -> Validate
RIGHT: Data -> Statistical Analysis -> Pattern Discovery -> Hypothesis -> Algorithm
```

All thresholds must be empirically derived from data, not assumed.

---

## 7. Algorithmic Research Direction

### 7.1 The 8 Proposed Algorithms

| # | Algorithm | Type | Key Features | Model |
|---|-----------|------|-------------|-------|
| 1 | **Entropy-Gated Switcher** | Gating | entropy | Threshold routing |
| 2 | **Momentum Continuation** | Directional | momentum, whale flow, hurst | Logistic Regression |
| 3 | **Mean-Reversion Detector** | Fade | z-score, liq risk, concentration | LightGBM |
| 4 | **Meta-Labeling** | Filter | entropy, toxicity, volatility | Logistic Regression |
| 5 | **Regime State Machine** | State | all regime features | HMM / thresholds |
| 6 | **Market-Making Skew** | MM | whale flow, imbalance, VPIN | Linear model |
| 7 | **Change-Point Detector** | Transition | entropy, whale flow | CUSUM / Bayesian |
| 8 | **Nearest-Neighbor Retrieval** | Non-parametric | 7-feature state vector | KNN |

### 7.2 Entropy Gating Logic

```
Low entropy (data-derived threshold) -> Momentum algorithms
High entropy (data-derived threshold) -> Mean-reversion algorithms
Uncertain -> No trade
```

**Critical fix needed:** Replace arbitrary 0.3/0.7 thresholds with empirical percentiles from data analysis (Level 1).

### 7.3 Phase Detection Signatures

| Phase | Entropy | Whale Flow | Volume | Absorption | Divergence |
|-------|---------|-----------|--------|------------|-----------|
| **Accumulation** | Low, declining | Positive | Below avg | High | Negative |
| **Uptrend** | Low, stable | Positive, aligned | Rising | — | — |
| **Distribution** | Rising | Negative | High (churning) | High (sell side) | Positive |
| **Downtrend** | Low, stable | Negative, aligned | Rising | — | — |
| **Ranging** | High | Mixed | Normal | Low | Near zero |

**Key insight:** Accumulation and distribution look identical in price (range-bound) but opposite in flow features (whale buying vs selling, VPIN low vs high).

---

## 8. Critiques and Required Improvements

### 8.1 Seven Fundamental Weaknesses

| # | Weakness | Fix |
|---|----------|-----|
| 1 | **Arbitrary thresholds** (0.3/0.7 entropy) | Empirical percentiles from data |
| 2 | **Missing statistical rigor** | Formal H0/H1, power analysis, effect sizes |
| 3 | **Overfitting risk** (183 features) | 3-stage feature selection: redundancy -> relevance -> stability |
| 4 | **No baseline** | Beat buy-and-hold, random, simple MA crossover |
| 5 | **Ignored transaction costs** | 13 bps round-trip on Hyperliquid, include in all metrics |
| 6 | **Regime non-stationarity** | Rolling windows, regime persistence filters |
| 7 | **No causal framework** | Granger causality, cross-correlation, pre-register hypotheses |

### 8.2 Required Statistical Rigor

- **Power analysis:** For d=0.1, alpha=0.01, power=0.8 -> need ~1,570 observations (262 days at 4h bars)
- **Multiple testing:** 8 algorithms x 3 regimes x 5 features = 120 tests -> Bonferroni alpha = 0.000083
- **Validation protocol:** 60/40 discovery/validation split, pre-register hypotheses, single confirmatory test

### 8.3 Feature Selection Protocol

1. **Stage 1 — Remove redundancy:** Drop features with correlation > 0.8, keep one with higher mutual information
2. **Stage 2 — Relevance filter:** Keep only features with significant MI (permutation test, Bonferroni-corrected)
3. **Stage 3 — Stability selection:** Bootstrap 100x, keep features selected > 80% of the time

---

## 9. Architecture Specifications

### 9.1 MCP Server Architecture (Future)

Three MCP servers for Claude Code / Agent SDK integration:

| Server | Purpose | Transport |
|--------|---------|-----------|
| `nat-ingest` | Control Rust ingestor (start/stop/status) | stdio |
| `nat-research` | Run clustering, viz, statistics | stdio |
| `nat-resources` | URI-addressable artifacts (parquet, plots, models) | stdio |

Tools defined: `start_ingestion`, `stop_ingestion`, `get_ingest_status`, `list_available_data`, `run_clustering`, `run_visualization`, `get_snapshot`, etc.

### 9.2 Statistical Dashboard (Future)

- **Compute engine:** Realtime (1-5min), hourly, daily, weekly snapshot computers
- **Storage:** TimescaleDB (PostgreSQL + time-series extension)
- **API:** REST + WebSocket for real-time updates
- **Frontend:** React dashboard (only after signal validation)
- **Schema:** Snapshots table (parent) -> realtime_entropy, realtime_volatility, hourly_cluster_stats, daily_entropy_distribution, daily_feature_rankings, etc.
- **Principle:** Compute once, publish continuously, consume by many. Immutable snapshots with TTL.

### 9.3 NautilusTrader Integration (Future)

Production-grade backtesting framework (Rust/Cython + Python API):
- Custom FeatureBar data adapter
- Strategy implementations (EntropyGatedMomentumStrategy, etc.)
- Walk-forward validation: 60d train, 15d validate, OOS/IS > 0.7
- Paper trading 30 days before live deployment at 10% capital

---

## 10. Decision Gate

### 10.1 Questions to Answer with Data

**Q1: Do natural clusters exist?**
- Silhouette > 0.25 at any timeframe/subspace?
- GMM optimal k > 1?
- Dip test significant for any feature?

**Q2: Are clusters stable?**
- Bootstrap ARI > 0.6?
- Temporal ARI > 0.5?
- Self-transition rate > 0.7?

**Q3: Do clusters predict returns?**
- Kruskal-Wallis p < 0.05?
- Eta-squared > 0.01?
- Cluster-conditional Sharpes differ meaningfully?

**Q4: At which timeframe?**
- Best timeframe: ___
- Best feature subspace: ___
- Optimal cluster count: ___

### 10.2 Decision Rules

- **GO:** Majority YES on Q1-Q3 -> proceed to phase detection
- **PIVOT:** Clusters exist but don't predict returns -> try different features, timeframes, or transition derivatives
- **NO-GO:** No clusters found -> accept null result, treat as data engineering project

### 10.3 Algorithm Performance Targets (Post-GO)

| Metric | Target | Stretch |
|--------|--------|---------|
| Walk-Forward Sharpe | > 0.5 | > 0.8 |
| OOS/IS Ratio | > 0.7 | > 0.85 |
| Win Rate | > 52% | > 55% |
| Max Drawdown | < 20% | < 15% |
| Profit Factor | > 1.3 | > 1.5 |

---

## 11. Implementation Priority

| Priority | Task | Status | Dependency |
|----------|------|--------|-----------|
| **P0** | Collect 5+ weeks live data | DONE | — |
| **P0** | Feature vector config (`config.py`) | DONE (117 tests) | — |
| **P0** | Bar aggregation (`preprocess.py`) | **MISSING — BLOCKER** | Live data |
| **P0** | Parquet loader (`loader.py`) | MISSING | — |
| **P0** | Clustering engine (`cluster.py`) | MISSING | Aggregated bars |
| **P0** | Consolidated `requirements.txt` | MISSING | — |
| **P1** | Dim reduction (`reduce.py`) | MISSING | Clustering |
| **P1** | Visualization (`viz.py`) | MISSING | Clustering + reduction |
| **P1** | Jupyter notebook | MISSING | All pipeline scripts |
| **GATE** | Decision: GO / PIVOT / NO-GO | — | P0 + P1 complete |
| P2 | Phase detection prototype | — | Validated clusters |
| P3 | Algorithm prototyping (8 algorithms) | — | Decision gate = GO |
| P4 | NautilusTrader integration | — | Validated algorithms |
| P5 | MCP server implementation | — | Stable pipeline |
| P6 | Statistical dashboard | — | Proven signal |

---

## 12. Spectral / MA Extensions (Post-Validation Only)

| Extension | What it adds |
|-----------|-------------|
| Spectral entropy | Entropy of FFT power spectrum |
| Wavelet decomposition | Multi-resolution amplitude analysis |
| Autocorrelation function | Persistence at lag 1, 5, 10, 50, 100 |
| GARCH(1,1) | Conditional volatility clustering |
| Cross-spectral coherence | Frequency-domain co-movement |
| Hilbert transform | Instantaneous phase/amplitude, cycle detection |
| Approximate/Sample entropy | Self-similarity, better for short series |
| Transfer entropy | Information flow between features |

**Do not add until existing features cluster meaningfully.**

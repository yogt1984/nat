# Feature Vectors, Clustering, and Visualization — Reference Document

**Date:** 2026-04-14

---

## Part 0: Feature Vector Deep Dive — What Is Already Implemented

### 0.1 Entropy Vector (24 features)

**What it is:** Measures the randomness/predictability of market activity from multiple angles.

**Three sub-families implemented:**

**A) Permutation Entropy (4 features)**

Algorithm: Bandt-Pompe (2002). Takes a time series, extracts ordinal patterns of length `order=3` (how consecutive values rank relative to each other), counts pattern frequencies, computes Shannon entropy of the distribution. Normalized to [0, 1].

| Feature | Window | What it measures |
|---------|--------|-----------------|
| `ent_permutation_returns_8` | 8 ticks | Short-term return predictability |
| `ent_permutation_returns_16` | 16 ticks | Medium-term return predictability |
| `ent_permutation_returns_32` | 32 ticks | Longer-term return predictability |
| `ent_permutation_imbalance_16` | 16 ticks | Order book imbalance predictability |

*Interpretation:* Low permutation entropy = returns follow repeated patterns (trending/mean-reverting). High = random walk. The multi-window design captures scale: 8-tick patterns are microstructure noise, 32-tick patterns are more meaningful.

*Spectral character:* Permutation entropy is inherently non-spectral — it captures ordinal structure, not frequency. But it responds to periodic signals: a sine wave has low PE because the ordinal pattern repeats. Think of it as a non-parametric test for "is there any structure at this scale?"

*Currently broken:* `permutation_imbalance_16` is hardcoded to 0.0 (TODO in source). Needs imbalance history buffer.

**B) Distribution Entropy (4 features)**

Algorithm: Bin continuous values into histogram, compute Shannon entropy of bin counts. Normalized by log(n_bins).

| Feature | Input | Bins | What it measures |
|---------|-------|------|-----------------|
| `ent_spread_dispersion` | Spread history | — | Are spreads variable or constant? |
| `ent_volume_dispersion` | Trade sizes (30s) | 10 | Are trade sizes uniform or skewed? |
| `ent_book_shape` | Order book depths | n_levels | Is liquidity evenly distributed across levels? |
| `ent_trade_size_dispersion` | Trade sizes (30s) | 5 | Coarser view of trade size distribution |

*Interpretation:* Low book_shape entropy = liquidity concentrated at one level (wall/spoofing). High = evenly spread (normal market). Low volume_dispersion = institutional block trades (uniform size). High = mixed retail+institutional.

*Currently broken:* `spread_dispersion` is hardcoded to 0.0, `rate_of_change_5s` is 0.0, `zscore_1m` is 0.0 (all need history buffers not yet implemented).

**C) Tick Entropy (14 features: 7 raw + 7 volume-weighted)**

Algorithm: Classify each trade as up/down/neutral tick, count frequencies in a time window, compute Shannon entropy.

| Window | Raw | Vol-weighted | Interpretation |
|--------|-----|-------------|---------------|
| 1s | `ent_tick_1s` | `ent_vol_tick_1s` | Microstructure noise |
| 5s | `ent_tick_5s` | `ent_vol_tick_5s` | HFT timescale |
| 10s | `ent_tick_10s` | `ent_vol_tick_10s` | Short-term direction |
| 15s | `ent_tick_15s` | `ent_vol_tick_15s` | Intermediate |
| 30s | `ent_tick_30s` | `ent_vol_tick_30s` | Medium-term direction |
| 1m | `ent_tick_1m` | `ent_vol_tick_1m` | Intraday trend |
| 15m | `ent_tick_15m` | `ent_vol_tick_15m` | Swing-level direction |

*Interpretation:* Raw tick entropy treats all trades equally. Volume-weighted tick entropy weights by trade size — if one large trade goes up while 10 small trades go down, volume-weighted entropy is lower (more directional in dollar terms) while raw tick entropy is higher (more balanced in count terms). The difference between raw and volume-weighted at the same window is itself informative: large divergence means institutional flow disagrees with retail flow.

*Moving average / spectral relevance:* The 7 windows (1s to 15m) are effectively a multi-resolution analysis — similar in spirit to wavelet decomposition but using entropy instead of amplitude. Each window captures structure at a different frequency band. The ratio `ent_tick_1s / ent_tick_15m` is a crude spectral slope: if short-term entropy is high but long-term is low, there's noise at high frequency but structure at low frequency (trending with noise).

**What's missing from entropy that could help:**
- Spectral entropy (entropy of the power spectral density) — would directly measure whether energy is concentrated at specific frequencies
- Approximate entropy / sample entropy — measures self-similarity, better than PE for short series
- Transfer entropy — measures information flow *between* features (e.g., does order flow entropy predict price entropy?)

---

### 0.2 Trend Vector (15 features)

**What it is:** Measures whether price is trending, how strongly, and how persistent the trend is.

**A) Momentum (6 features)**

Algorithm: Ordinary least squares regression of price vs time index. Returns slope (momentum) and R-squared (fit quality).

| Feature | Window | What it measures |
|---------|--------|-----------------|
| `trend_momentum_60` | 60 ticks (~6s) | Immediate price direction |
| `trend_momentum_r2_60` | 60 ticks | How linear is the recent price path? |
| `trend_momentum_300` | 300 ticks (~30s) | Short-term trend |
| `trend_momentum_r2_300` | 300 ticks | Trend linearity over 30s |
| `trend_momentum_600` | 600 ticks (~1m) | Medium-term trend |
| `trend_momentum_r2_600` | 600 ticks | Trend linearity over 1m |

*Interpretation:* High momentum + high R2 = clean trend. High momentum + low R2 = volatile with a drift. Low momentum + high R2 = flat and quiet.

**B) Monotonicity (3 features)**

Algorithm: Fraction of consecutive ticks that move in the dominant direction. Range [0.5, 1.0].

*Interpretation:* Monotonicity > 0.7 = strong trend with few reversals. Monotonicity ~0.5 = random walk. This is a non-parametric complement to momentum — it doesn't care about magnitude, only direction consistency.

**C) Hurst Exponent (2 features)**

Algorithm: Rescaled range (R/S) analysis. Measures long-range dependence.

| Value | Interpretation |
|-------|---------------|
| H < 0.5 | Mean-reverting (anti-persistent) |
| H = 0.5 | Random walk |
| H > 0.5 | Trending (persistent) |

Only computed at 300 and 600 tick windows (needs sufficient data).

*Spectral connection:* Hurst exponent H relates to spectral density: `S(f) ~ f^(-(2H+1))`. For H > 0.5, low frequencies dominate (trending). For H < 0.5, high frequencies dominate (choppy). This is the most directly spectral feature you have.

**D) EMA Crossover (4 features)**

Algorithm: EMA(10) - EMA(50) and their absolute values. Classic momentum indicator.

*What's missing:* Spectral analysis (FFT/Welch), wavelet decomposition, autocorrelation function at multiple lags. These would tell you *at which frequency* price action has structure, rather than just "is there a trend?"

---

### 0.3 Illiquidity Vector (12 features)

**What it is:** Measures how much price moves per unit of trading activity. Core market microstructure.

**A) Kyle's Lambda (3 features)**

Algorithm: Regression of price change on signed volume. `lambda = delta_price / signed_volume`. Based on Kyle (1985) informed trading model.

*Interpretation:* High lambda = market is thin, trades move price easily = illiquid. Low lambda = deep market absorbs trades = liquid. The ratio `illiq_kyle_ratio = kyle_100 / kyle_500` measures whether illiquidity is changing (ratio > 1 = getting more illiquid recently).

**B) Amihud's Lambda (3 features)**

Algorithm: |return| / volume. Amihud (2002) illiquidity ratio.

*Interpretation:* Similar to Kyle but doesn't require signed volume. Higher = more illiquid.

**C) Hasbrouck's Lambda (2 features)**

Algorithm: Regression of absolute price change on square root of volume. Hasbrouck (2009).

**D) Roll Spread (2 features)**

Algorithm: `roll_spread = 2 * sqrt(-cov(delta_price_t, delta_price_{t-1}))`. Roll (1984) implied spread from price autocovariance.

*Interpretation:* Estimates the effective bid-ask spread from transaction prices alone, without order book data.

**Composite:** `illiq_composite` = weighted average of normalized Kyle, Amihud, Hasbrouck, and Roll.

*Spectral relevance:* Illiquidity features don't have spectral character themselves, but illiquidity *varies over time* and that variation has spectral content. An autocorrelation or spectral analysis of the illiquidity time series would tell you the typical duration of illiquid episodes.

---

### 0.4 Toxicity Vector (10 features)

**What it is:** Measures adverse selection — how much informed trading is happening.

**Key feature — VPIN (Volume-Synchronized Probability of Informed Trading):**

Algorithm: Based on Easley, Lopez de Prado, O'Hara (2012). Classifies trade volume into buyer/seller-initiated using the bulk classification method (tick rule), then measures the absolute imbalance. VPIN at windows 10 and 50 trades.

*Interpretation:* High VPIN = one side is aggressively taking liquidity = informed traders present. This historically spikes before large price moves and was notably high before the 2010 Flash Crash.

Other features: `toxic_adverse_selection` (permanent price impact), `toxic_effective_spread` (what traders actually pay), `toxic_realized_spread` (market maker's actual profit), `toxic_flow_imbalance` (net buy/sell pressure).

---

### 0.5 Order Flow Vector (8 features)

**What it is:** Order book imbalance at different depth levels.

| Feature | What it measures |
|---------|-----------------|
| `imbalance_qty_l1` | Best bid vs best ask quantity — immediate pressure |
| `imbalance_qty_l5` | Top 5 levels — short-term pressure |
| `imbalance_qty_l10` | Top 10 levels — medium-term pressure |
| `imbalance_orders_l5` | Number of orders (not quantity) — retail vs institutional |
| `imbalance_notional_l5` | Dollar-weighted imbalance |
| `imbalance_depth_weighted` | Distance-weighted: closer levels matter more |
| `imbalance_pressure_bid` | Absolute bid-side liquidity |
| `imbalance_pressure_ask` | Absolute ask-side liquidity |

*Moving average relevance:* These features are already instantaneous snapshots. An EMA or rolling mean of imbalance over time would measure *persistent* buying/selling pressure vs transient. The ingestor doesn't compute this — it's a candidate for the aggregation step.

---

### 0.6 Volatility Vector (8 features)

| Feature | Algorithm | Window |
|---------|-----------|--------|
| `vol_returns_1m` | Std(returns) | 1 minute |
| `vol_returns_5m` | Std(returns) | 5 minutes |
| `vol_parkinson_5m` | (High-Low)^2 / (4 ln2) | 5 minutes |
| `vol_spread_mean_1m` | Mean(spread) | 1 minute |
| `vol_spread_std_1m` | Std(spread) | 1 minute |
| `vol_midprice_std_1m` | Std(midprice) | 1 minute |
| `vol_ratio_short_long` | vol_1m / vol_5m | — |
| `vol_zscore` | (current_vol - mean) / std | — |

*Parkinson estimator* uses high-low range, which is more efficient than close-to-close for estimating true volatility (captures intra-period movement that close-to-close misses).

*Spectral relevance:* `vol_ratio_short_long` is a crude spectral ratio. If > 1, volatility is concentrated at high frequencies (choppy microstructure). If < 1, volatility is smooth (trending or calm). A wavelet-based vol decomposition would give you this at every scale.

*What's missing:* Garman-Klass estimator (uses OHLC, even more efficient), Yang-Zhang estimator (handles overnight jumps), realized volatility from intraday returns (Andersen & Bollerslev), GARCH-type conditional volatility.

---

### 0.7 Concentration Vector (15 features)

**What it is:** How concentrated are positions across market participants. Hyperliquid-unique.

Key concepts:
- `herfindahl_index` (HHI) = sum of squared position shares. HHI=1 means one entity holds everything.
- `gini_coefficient` = income inequality applied to position sizes. 0=equal, 1=maximally concentrated.
- `theil_index` = information-theoretic inequality measure (entropy-based).
- `top5/10/20/50_concentration` = fraction of total OI held by top N wallets.

*Why this matters:* When positions are concentrated (high Gini, high HHI), the market is fragile. A single whale exiting causes outsized price impact. When dispersed, the market is more resilient.

---

### 0.8 Whale Flow Vector (12 features)

**What it is:** Aggregate behavior of large position holders over time.

`whale_net_flow_1h/4h/24h` = net position change (in USD) by wallets above the whale threshold. Positive = whales are accumulating. Negative = distributing.

`whale_directional_agreement` = what fraction of active whales are moving in the same direction. High agreement = consensus. Low = disagreement (possibly a turning point).

`whale_flow_momentum` = acceleration of whale flow (are they speeding up or slowing down?).

---

### 0.9 Liquidation Vector (13 features)

**What it is:** Maps the landscape of positions at risk of liquidation.

`liquidation_risk_above/below_1/2/5/10pct` = total notional that would be liquidated if price moves up/down by X%. This creates a "liquidation heatmap" in feature form.

`liquidation_asymmetry` = ratio of upside vs downside liquidation risk. Positive = more longs at risk (price drop dangerous). Negative = more shorts at risk.

`nearest_cluster_distance` = how far is the nearest concentration of liquidation levels from current price.

---

### 0.10 Regime Vector (20 features)

**What it is:** Minute-scale features for detecting Wyckoff-style market phases.

**A) Absorption (4 features):** Volume per unit price change at 1h/4h/24h windows + z-score. High absorption = market absorbing selling without falling (accumulation) or buying without rising (distribution).

**B) Divergence (5 features):** Deviation from Kyle's lambda prediction at 1h/4h/24h + z-score. If price moves less than volume predicts, someone is absorbing.

**C) Churn (4 features):** Two-sided volume at 1h/4h/24h + z-score. High churn = active position transfer between participants (institutional rebalancing or whale rotation).

**D) Range Position (4 features):** Where current price sits within 4h/24h/1w range [0,1]. Near 0 = at bottom of range. Near 1 = at top.

**E) Composite (3 features):** `accumulation_score` and `distribution_score` combine absorption + divergence + churn + range position into single [0,1] scores. `regime_clarity` = max(accumulation, distribution) — how confident the detection is.

---

### 0.11 What's Already Computed But Hardcoded to 0.0

| Feature | Module | Issue |
|---------|--------|-------|
| `ent_permutation_imbalance_16` | entropy.rs | Needs imbalance history buffer |
| `ent_spread_dispersion` | entropy.rs | Needs spread history buffer |
| `ent_rate_of_change_5s` | entropy.rs | Needs entropy history buffer |
| `ent_zscore_1m` | entropy.rs | Needs entropy history buffer |

These 4 of the 24 entropy features will be 0.0 in your live data. The entropy vector effectively has 20 active features, not 24. This doesn't break clustering but you should be aware that 4 columns carry no information.

---

### 0.12 Spectral and MA Extensions That Could Add Value

| Extension | What it adds | Applicable to |
|-----------|-------------|---------------|
| **Spectral entropy** | Entropy of FFT power spectrum — is energy concentrated at specific frequencies? | Any time series |
| **Wavelet decomposition** | Multi-resolution amplitude (like your tick entropy windows, but for amplitude not entropy) | Price, volume, spread |
| **Autocorrelation function** | Correlation at lag 1, 5, 10, 50, 100 — measures persistence without assuming linearity | All features |
| **GARCH(1,1)** | Conditional volatility — captures volatility clustering | Returns |
| **Cross-spectral coherence** | At which frequencies do two features co-move? | Feature pairs |
| **Hilbert transform** | Instantaneous phase and amplitude — detects cycles | Price, entropy |

These are not implemented in the Rust ingestor. They could be computed in the Python aggregation step (on bars, not ticks). But don't add them until you've validated that the existing features cluster meaningfully.

---

## Part 1: Clustering Methods

### 1.0 Complete Catalog of Clustering Algorithms

There are roughly 30+ clustering algorithms in the literature. Here are the ones relevant to financial time series, grouped by type:

#### Partition-based (assign every point to exactly one cluster)

| Algorithm | How it works | Strengths | Weaknesses | Good for your data? |
|-----------|-------------|-----------|------------|-------------------|
| **K-Means** | Minimize within-cluster sum of squares. Iterates assign→update centroids. | Fast, interpretable centroids, scales to millions of rows | Assumes spherical clusters, requires k, sensitive to outliers | Baseline only — financial features have heavy tails |
| **K-Medoids (PAM)** | Like K-Means but uses actual data points as centers | Robust to outliers | Slow (O(n²k)) | Good for small datasets (<10K) |
| **Mini-Batch K-Means** | K-Means on random subsets per iteration | Scales to very large data | Slightly worse than full K-Means | Use when >100K rows |

#### Model-based (assume data comes from a mixture of distributions)

| Algorithm | How it works | Strengths | Weaknesses | Good for your data? |
|-----------|-------------|-----------|------------|-------------------|
| **GMM** (Gaussian Mixture) | EM algorithm. Assumes k Gaussian components. | Soft assignments (probabilities), BIC for k selection, models elliptical clusters | Assumes Gaussian components, can overfit in high dims | **Primary choice** — BIC gives principled k selection |
| **DPGMM** (Dirichlet Process) | Bayesian GMM with automatic k | No need to specify k, learns k from data | Slow, sensitive to concentration prior | Use to validate GMM's k choice |
| **HMM** (Hidden Markov Model) | GMM + temporal transitions | Models regime persistence and transitions | Requires temporal ordering, more parameters | Phase 2 — after clusters are validated |

#### Density-based (find regions of high density)

| Algorithm | How it works | Strengths | Weaknesses | Good for your data? |
|-----------|-------------|-----------|------------|-------------------|
| **DBSCAN** | Points in dense neighborhoods are clustered. Two params: eps, min_samples. | Finds arbitrary shapes, identifies noise/outliers, no k needed | Sensitive to eps, struggles with varying densities | Useful for outlier detection, not primary |
| **HDBSCAN** | Hierarchical DBSCAN. Builds cluster tree, extracts stable clusters. | No eps needed, handles varying densities, robust | Slow on large data, min_cluster_size matters | **Strong secondary choice** — good for finding natural structure |
| **OPTICS** | Orders points by density reachability | Visualizes cluster structure (reachability plot), handles varying density | Harder to extract flat clusters from | Diagnostic tool |

#### Hierarchical (build a tree of nested clusters)

| Algorithm | How it works | Strengths | Weaknesses | Good for your data? |
|-----------|-------------|-----------|------------|-------------------|
| **Agglomerative (Ward)** | Bottom-up merging, minimize within-cluster variance | Dendrogram visualization, no k needed (cut tree at desired level) | O(n²) memory, sensitive to linkage choice | Good for visualization and understanding cluster hierarchy |
| **Agglomerative (Complete)** | Merge clusters with smallest max inter-point distance | Finds compact clusters | Sensitive to outliers | Less relevant |
| **Agglomerative (Average)** | Merge clusters with smallest average distance | Balanced between single and complete | — | Less relevant |
| **BIRCH** | Builds CF-tree for large datasets, then clusters | Very fast, handles large data | Assumes spherical clusters | Only if data > 1M rows |

#### Graph-based

| Algorithm | How it works | Strengths | Weaknesses | Good for your data? |
|-----------|-------------|-----------|------------|-------------------|
| **Spectral Clustering** | Build similarity graph, cluster eigenvectors of graph Laplacian | Finds non-convex clusters, uses manifold structure | O(n³) for eigendecomp, need to choose affinity | Useful if UMAP shows non-convex structure |
| **Affinity Propagation** | Message passing to find exemplars | Automatically determines k | Very slow (O(n²)), often produces too many clusters | Rarely useful in practice |

#### Subspace / specialized

| Algorithm | How it works | Relevance |
|-----------|-------------|-----------|
| **Fuzzy C-Means** | Soft K-Means with membership degrees | Similar to GMM but without probabilistic interpretation |
| **Self-Organizing Maps** | Neural network for topology-preserving dimensionality reduction + clustering | Interesting for visualization but adds complexity |
| **Ensemble methods** | Run multiple algorithms, aggregate results | Validation technique, not primary clustering |

#### Recommendation: Do Not Diversify Blindly

**Primary: GMM.** It gives you soft assignments (probability of belonging to each cluster), BIC for model selection, and handles elliptical clusters. Start here.

**Secondary: HDBSCAN.** It finds clusters without specifying k and identifies outliers. Use it to validate GMM's cluster count — if GMM says k=3 and HDBSCAN finds 3 dense regions, you have convergent evidence.

**Diagnostic: Agglomerative (Ward).** Use the dendrogram to visualize the natural hierarchy. Don't use the cluster assignments directly.

**Skip everything else initially.** Running 10 algorithms on 14 vectors at 4 timeframes creates a massive multiple testing problem. You'll find "significant" clusters by chance. Two algorithms that agree are worth more than ten that each give different answers.

---

### 1.1 Cluster Separability — How to Quantify

#### Internal Metrics (no ground truth needed)

| Metric | Formula intuition | Range | Good value | Limitation |
|--------|------------------|-------|-----------|-----------|
| **Silhouette score** | (distance to nearest other cluster - distance within own cluster) / max | [-1, 1] | > 0.25 meaningful, > 0.5 strong | Biased toward spherical clusters |
| **Davies-Bouldin index** | Avg ratio of within-cluster scatter to between-cluster distance | [0, ∞) | < 1.0 good, lower is better | Assumes convex clusters |
| **Calinski-Harabasz** | Ratio of between-cluster variance to within-cluster variance | [0, ∞) | Higher is better, no absolute threshold | Biased toward spherical, favors more clusters |
| **BIC (GMM)** | Log-likelihood penalized by parameter count | (-∞, ∞) | Minimum across k | Only for GMM, assumes Gaussian |
| **Gap statistic** | Compare within-cluster dispersion to null (uniform random) | [0, ∞) | Largest gap = optimal k | Computationally expensive |
| **Dunn index** | Min inter-cluster distance / max intra-cluster diameter | [0, ∞) | Higher is better | Sensitive to outliers |

#### Stability Metrics (do clusters persist under perturbation?)

| Metric | Method | Good value |
|--------|--------|-----------|
| **Bootstrap ARI** | Resample data N times, recluster, compare via Adjusted Rand Index | > 0.6 stable |
| **Temporal ARI** | Cluster first half and second half independently, compare | > 0.5 |
| **Cross-validation stability** | K-fold: train on folds, predict on held-out, compare | > 0.6 |
| **Perturbation robustness** | Add small noise, recluster, compare | ARI > 0.7 |

#### Predictive Metrics (do clusters relate to something external?)

| Metric | Method | What it tells you |
|--------|--------|-----------------|
| **Kruskal-Wallis test** | Non-parametric ANOVA on forward returns by cluster | p < 0.05 = clusters differentiate returns |
| **Eta-squared (η²)** | Effect size for cluster→return relationship | > 0.01 small, > 0.06 medium |
| **Cluster-conditional Sharpe** | Sharpe ratio computed within each cluster | Different Sharpes = different risk/reward regimes |
| **Transition matrix entropy** | Entropy of row-normalized transition matrix | Low = predictable transitions, high = random |

#### Multimodality Tests (is the distribution clustered at all?)

| Test | What it detects | Good for |
|------|----------------|----------|
| **Hartigan's dip test** | Unimodality vs multimodality | Individual features — is entropy bimodal? |
| **Bimodality coefficient** | BC = (skewness² + 1) / kurtosis | Quick screen, BC > 5/9 suggests bimodality |
| **Silverman's bandwidth test** | Tests for number of modes in kernel density | More rigorous than dip test |
| **GMM BIC: k=1 vs k=2** | Is 2-component mixture better than 1? | Direct test of "do clusters exist?" |

---

### 1.2 Dimensionality Reduction and Visualization

#### Linear Methods

| Method | What it preserves | Computation | When to use |
|--------|------------------|-------------|------------|
| **PCA** | Maximum variance directions | Eigendecomposition of covariance matrix, O(d²n) | Always first — fast, interpretable, gives explained variance |
| **Factor Analysis** | Latent factors (like PCA but models noise separately) | EM algorithm | When features have different noise levels |
| **ICA** (Independent Component Analysis) | Statistical independence, not just uncorrelation | Iterative, maximizes non-Gaussianity | When you suspect non-Gaussian latent factors |
| **LDA** (Linear Discriminant Analysis) | Maximize between-class / within-class variance | Requires labels (cluster assignments) | After clustering — project to best separate known clusters |

**PCA specifics for your data:**
- 24d entropy vector → PCA will likely explain 80%+ variance in 3-4 components (the 7 tick entropy windows are highly correlated across windows)
- The first PC of entropy will roughly be "overall entropy level"
- The second PC will roughly be "short-term vs long-term entropy" (the multi-scale structure)
- Plot explained variance ratio — if first 3 PCs explain < 60%, the data doesn't have low-dimensional structure

#### Nonlinear Methods

| Method | What it preserves | Speed | Parameters | When to use |
|--------|------------------|-------|-----------|------------|
| **UMAP** | Local + global topology | Fast (minutes for 10K points) | `n_neighbors` (15), `min_dist` (0.1), `metric` (euclidean) | **Primary visualization** — best balance of speed, quality, and global structure |
| **t-SNE** | Local neighborhoods only | Slow (O(n²) naive, O(n log n) Barnes-Hut) | `perplexity` (30), `learning_rate` (200) | Confirmation of UMAP — if both show the same clusters, they're real |
| **Isomap** | Geodesic distances on manifold | Moderate | `n_neighbors` | When data lies on a curved surface |
| **LLE** (Locally Linear Embedding) | Local linear structure | Moderate | `n_neighbors` | When data lies on a smooth manifold |
| **Kernel PCA** | Nonlinear version of PCA | Moderate | Kernel choice, gamma | When PCA misses nonlinear structure |
| **Diffusion Maps** | Diffusion distances on data graph | Moderate | `alpha`, `n_neighbors` | When you want spectral-like decomposition of the geometry |
| **Trimap** | Like UMAP with better global structure | Fast | `n_inliers`, `n_outliers` | Newer alternative to UMAP |
| **PaCMAP** | Preserves both local and global structure | Fast | Minimal tuning | Newer, sometimes better than UMAP |

**UMAP parameter guidance for your data:**
- `n_neighbors=15`: default works well for 1K-30K points. Increase to 30-50 for more global structure.
- `min_dist=0.1`: how tight clusters are packed. Decrease to 0.01 for tighter clusters.
- `metric="euclidean"`: fine for standardized features. Try `"cosine"` if feature magnitudes vary wildly.
- `n_components=2` for 2D plots, `=3` for interactive 3D.

**Critical UMAP/t-SNE caveats:**
1. Distances between clusters are NOT meaningful. Two clusters far apart in UMAP are not necessarily more different than two clusters close together.
2. Cluster sizes are NOT meaningful. UMAP can make a real cluster look like a single point or spread a tight cluster out.
3. Run multiple times with different random seeds. If cluster structure changes, it's an artifact.

#### What 2D/3D Plots Can and Cannot Tell You

**Can tell you:**
- Whether discrete groups exist at all (vs one continuous blob)
- Relative positions of points within a cluster
- Outliers and anomalies
- Whether cluster assignments from GMM/HDBSCAN align with visual structure

**Cannot tell you:**
- Whether clusters predict anything useful
- The true distance between clusters
- Whether the projection is hiding structure (a 24d dataset projected to 2D loses information)
- Whether what you see is real or an artifact of the projection

---

### 1.3 Frontend and Visualization Tech Stack

#### Python-Native (Notebook-First)

| Library | Type | Best for |
|---------|------|---------|
| **Plotly** | Interactive HTML | 3D scatter, hover tooltips, rotation. Works in Jupyter. No server needed. This is your primary tool. |
| **Matplotlib** | Static publication plots | 2D scatter, heatmaps, histograms, dendrograms. Familiar, fast. |
| **Seaborn** | Statistical visualizations | Pair plots, violin plots, heatmaps with annotations. Built on matplotlib. |
| **Bokeh** | Interactive in-browser | More customizable than Plotly, good for dashboards. Steeper learning curve. |
| **HoloViews + Panel** | Declarative interactive | Compose interactive plots without boilerplate. Good for exploration. |

**Recommendation:** Start with Plotly for 3D interactive + matplotlib for static. Don't add complexity until you need a dashboard.

#### Full Web Stack (Only If You Build a Dashboard)

**If you reach the point where you want a persistent, shareable visualization tool:**

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | React + TypeScript | Industry standard, huge ecosystem |
| **3D Rendering** | Three.js or Deck.gl | WebGL-based 3D scatter plots, handles 100K+ points |
| **Charting** | D3.js or Visx (React wrapper for D3) | Maximum control over custom visualizations |
| **State management** | Zustand or Jotai | Lightweight, sufficient for dashboards |
| **Backend API** | FastAPI (Python) | Serves cluster data, runs analysis on demand |
| **Data layer** | DuckDB (embedded) or Polars | Query parquet files directly, no database server needed |
| **Deployment** | Docker + Caddy | Simple reverse proxy, HTTPS |

**Simpler alternative: Streamlit or Gradio**

| Tool | Pros | Cons |
|------|------|------|
| **Streamlit** | Python-only, fast to build, built-in plotly/matplotlib support, caching | Limited layout control, can feel toy-ish |
| **Gradio** | Even simpler than Streamlit, good for ML demos | Even more limited |
| **Panel** (HoloViz) | More flexible than Streamlit, better for data apps | Smaller community |
| **Dash** (by Plotly) | Full React-like components in Python, production-grade | More complex than Streamlit, callback-based |

**My recommendation for your stage:**

1. **Now:** Jupyter + Plotly + matplotlib. Zero infrastructure overhead. You're exploring, not deploying.
2. **If clusters are real (post decision gate):** Streamlit app. Single Python file, deploy anywhere, share with anyone. Good enough for 90% of needs.
3. **If you build a production system:** FastAPI + React + Three.js. Only when you have a proven signal and need a real-time dashboard.

**Do not build a web frontend before you know clusters exist.** This is a common trap — spending weeks on a beautiful dashboard for data that turns out to be noise.

---

## Summary: What to Do Next

1. **Sync your 5 weeks of data** from the remote machine
2. **Inspect the schema** — confirm which of the 183 features actually have data (vs hardcoded 0.0)
3. **Build `aggregate_bars.py`** — this is still the blocker
4. **Run the cluster pipeline** on entropy vector first (most promising), then volatility, then regime
5. **Use GMM + HDBSCAN**, not 10 algorithms
6. **Visualize with UMAP 2D/3D + Plotly**, not a custom web stack
7. **Respect the decision gate** — if silhouette < 0.15, accept the null result

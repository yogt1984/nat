# NAT MCP Server Specification

**Created:** 2026-04-12
**Status:** Architecture Specification
**Purpose:** Define three MCP servers that expose NAT's capabilities as composable tools

---

## Overview

NAT is decomposed into three MCP servers that separate concerns and allow independent
scaling, development, and composition with external quant/math MCP servers.

```
                          Claude Code / Agent SDK
                                   |
                  +----------------+----------------+
                  |                |                |
          +-----------+    +-----------+    +-----------+
          |  Ingest   |    | Research  |    | Resource  |
          |    MCP    |    |  /Viz MCP |    |   MCP     |
          +-----------+    +-----------+    +-----------+
               |                |                |
               v                v                v
          Rust Binary      Python Layer      URI Router
          (ing)            (scripts/)        (parquet/plots/models)
               |                |                |
               +-------+-------+--------+-------+
                       |                 |
                  ./data/features/    ./reports/
                  (Parquet files)     (artifacts)
```

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Binary stays binary** | The Rust ingestor is the hot path. MCP wraps it, doesn't replace it |
| **Python for analysis** | Clustering, visualization, statistics — all in Python via existing scripts |
| **URIs for everything** | Every artifact (parquet, plot, model, cluster label) is addressable |
| **Stateless tools, stateful resources** | Tools compute on demand; resources point to persisted artifacts |
| **Compose, don't monolith** | Each MCP server is independently useful and combinable with external servers |

---

## MCP Server 1: Ingest MCP (`nat-ingest`)

### Purpose
Control the Rust ingestor binary. Start/stop data collection, monitor stream health,
manage historical data acquisition.

### Transport
**stdio** — wraps the `./target/release/ing` binary and a thin Python control layer.

### Registration
```bash
claude mcp add --transport stdio nat-ingest -- python3 mcp/ingest_server.py
```

### Tools

#### `start_ingestion`
Start collecting tick-level data from Hyperliquid.

```yaml
name: start_ingestion
description: Start live market data ingestion from Hyperliquid
parameters:
  symbols:
    type: array
    items: { type: string }
    description: "Asset symbols to ingest"
    examples: [["BTC", "ETH", "SOL"], ["BTC"]]
    required: true
  source:
    type: string
    enum: [hyperliquid]
    default: hyperliquid
    description: "Data source"
  mode:
    type: string
    enum: [live, backfill, replay]
    description: |
      live     — connect to WebSocket, stream real-time
      backfill — fetch historical data from REST API, fill gaps
      replay   — replay existing parquet files as if live (for testing)
    default: live
  config_overrides:
    type: object
    description: "Override config/ing.toml values"
    properties:
      emission_interval_ms: { type: integer }
      book_levels: { type: integer }
      rotate_interval: { type: string, enum: ["1h", "1d"] }
      data_dir: { type: string }
returns:
  job_id: { type: string, description: "Unique ingestion job identifier" }
  pid: { type: integer, description: "OS process ID of the ingestor" }
  symbols: { type: array }
  started_at: { type: string, format: datetime }
  data_dir: { type: string, description: "Where parquet files are written" }
```

**Implementation:** Spawns `./target/release/ing` as subprocess with appropriate config.
Writes PID to `./data/.ingest_jobs/{job_id}.json` for tracking.

#### `stop_ingestion`
Gracefully stop an ingestion job.

```yaml
name: stop_ingestion
description: Stop a running ingestion job
parameters:
  job_id:
    type: string
    description: "Job ID from start_ingestion, or 'all' to stop everything"
    required: true
  flush:
    type: boolean
    default: true
    description: "Flush buffered data to parquet before stopping"
returns:
  stopped: { type: boolean }
  rows_written: { type: integer }
  duration: { type: string }
  files_created: { type: array, items: { type: string } }
```

**Implementation:** Sends SIGTERM to the process, waits for flush, reads final metrics.

#### `get_ingest_status`
Get health and metrics of a running ingestion job.

```yaml
name: get_ingest_status
description: Get status and metrics of an ingestion job
parameters:
  job_id:
    type: string
    required: true
returns:
  status: { type: string, enum: [running, stopped, error, starting] }
  uptime_seconds: { type: number }
  symbols:
    type: array
    items:
      type: object
      properties:
        symbol: { type: string }
        rows_emitted: { type: integer }
        last_emission_at: { type: string }
        features_computed: { type: integer, description: "Out of 183" }
        nan_rate: { type: number, description: "Fraction of NaN values" }
        avg_latency_ms: { type: number }
  disk:
    type: object
    properties:
      total_bytes: { type: integer }
      files_count: { type: integer }
      oldest_file: { type: string }
      newest_file: { type: string }
  errors: { type: array, items: { type: string } }
```

**Implementation:** Reads Prometheus metrics from the ingestor (if enabled) or
parses log output. Inspects `./data/features/` directory for disk stats.

#### `list_active_streams`
List all currently running ingestion jobs.

```yaml
name: list_active_streams
description: List all active ingestion streams
parameters: {}
returns:
  streams:
    type: array
    items:
      type: object
      properties:
        job_id: { type: string }
        symbols: { type: array }
        mode: { type: string }
        started_at: { type: string }
        status: { type: string }
        rows_total: { type: integer }
```

#### `list_available_data`
Inventory of all collected data.

```yaml
name: list_available_data
description: List all available parquet data files with metadata
parameters:
  symbol:
    type: string
    description: "Filter by symbol (optional)"
  start:
    type: string
    format: datetime
    description: "Start of time range (optional)"
  end:
    type: string
    format: datetime
    description: "End of time range (optional)"
returns:
  files:
    type: array
    items:
      type: object
      properties:
        path: { type: string }
        symbol: { type: string }
        start_time: { type: string }
        end_time: { type: string }
        rows: { type: integer }
        size_bytes: { type: integer }
        features: { type: integer, description: "Column count" }
  summary:
    type: object
    properties:
      total_rows: { type: integer }
      total_bytes: { type: integer }
      symbols: { type: array }
      time_range: { type: object, properties: { start: { type: string }, end: { type: string } } }
      total_duration: { type: string }
```

#### `aggregate_timeframe`
Aggregate tick-level data into bars at specified timeframe. This is the **critical
blocker** identified in the development methodology — without this, no clustering
or strategy work can proceed at meaningful horizons.

```yaml
name: aggregate_timeframe
description: |
  Aggregate tick-level parquet data into bars at a specified timeframe.
  For each feature, computes: mean, std, min, max, close (last value).
  Special aggregations for whale flow (cumulative sum), entropy (slope), etc.
parameters:
  symbol:
    type: string
    required: true
  timeframe:
    type: string
    enum: [1m, 5m, 15m, 1h, 4h, 1d]
    required: true
    description: "Bar duration"
  start:
    type: string
    format: datetime
    description: "Start of range (optional, defaults to all data)"
  end:
    type: string
    format: datetime
  features:
    type: array
    items: { type: string }
    description: "Subset of features to aggregate (optional, defaults to all)"
  output_dir:
    type: string
    default: "./data/bars/"
returns:
  output_file: { type: string }
  bars_created: { type: integer }
  timeframe: { type: string }
  features_aggregated: { type: integer }
  time_range: { type: object }
```

**Implementation:** Python script using pyarrow. Reads tick parquet, groups by time
window, computes statistics per feature, writes aggregated parquet.

**Aggregation rules:**
```
Standard features:
  {feature}_mean, {feature}_std, {feature}_min, {feature}_max, {feature}_close

Special aggregations:
  whale_net_flow_{window}  -> sum (cumulative over bar)
  ent_permutation_{n}      -> mean + slope (trend within bar)
  vol_realized_{n}         -> close (point estimate) + max (peak vol)
  regime_absorption_zscore -> mean + max
  composite_regime_signal  -> mean + close
  raw_mid_price            -> open, high, low, close (OHLC)
  flow_volume_5s           -> sum (total volume in bar)
```

---

## MCP Server 2: Research/Viz MCP (`nat-research`)

### Purpose
Clustering analysis, statistical validation, visualization, and regime detection.
Wraps the Python analysis scripts and the skeptical validation suite.

### Transport
**stdio** — Python process with access to numpy, scipy, sklearn, matplotlib.

### Registration
```bash
claude mcp add --transport stdio nat-research -- python3 mcp/research_server.py
```

### Tools

#### `cluster_features`
Run clustering on feature data. The core analytical operation.

```yaml
name: cluster_features
description: |
  Cluster market features using specified algorithm and evaluate quality.
  Tests whether natural market states exist in the data.
parameters:
  symbol:
    type: string
    required: true
  lookback:
    type: string
    description: "Time window to analyze"
    examples: ["24h", "7d", "30d", "all"]
    default: "all"
  timeframe:
    type: string
    enum: [tick, 1m, 5m, 15m, 1h, 4h, 1d]
    default: "15m"
    description: "Aggregation level before clustering"
  feature_set:
    type: string
    enum: [entropy_only, volatility_only, flow_only, microstructure, full, custom]
    default: full
    description: |
      entropy_only:     ent_permutation_8/16/32, ent_book_shape, ent_trade_size
      volatility_only:  vol_realized_20/100, vol_ratio
      flow_only:        whale_net_flow, flow_aggressor, imbalance_l5/l10
      microstructure:   raw_spread_bps, raw_bid/ask_depth, imbalance_persistence
      full:             all available features
      custom:           specify via custom_features parameter
  custom_features:
    type: array
    items: { type: string }
    description: "Feature names when feature_set=custom"
  algorithm:
    type: string
    enum: [gmm, kmeans, hdbscan, spectral, hmm, dpgmm, agglomerative]
    default: gmm
    description: |
      gmm:            Gaussian Mixture Model (soft assignments, BIC model selection)
      kmeans:          K-Means (hard assignments, fast baseline)
      hdbscan:         Hierarchical DBSCAN (auto cluster count, noise detection)
      spectral:        Spectral clustering (nonlinear boundaries)
      hmm:             Hidden Markov Model (temporal transitions)
      dpgmm:           Bayesian GMM with Dirichlet Process (auto cluster count)
      agglomerative:   Ward linkage hierarchical (dendrogram)
  n_clusters:
    type: integer
    description: "Number of clusters (ignored for hdbscan, dpgmm which auto-select)"
    default: null
  max_clusters:
    type: integer
    default: 8
    description: "Maximum clusters to test for BIC/gap model selection"
returns:
  n_clusters_found: { type: integer }
  cluster_labels: { type: string, description: "URI to cluster label resource" }
  quality:
    type: object
    description: "Cluster quality metrics"
    properties:
      silhouette_score: { type: number, description: "[-1,1], >0.25 reasonable, >0.5 strong" }
      calinski_harabasz: { type: number, description: "[0,inf), higher=better" }
      davies_bouldin: { type: number, description: "[0,inf), <1.0 good" }
      bic: { type: number, description: "GMM only, lower=better" }
      gap_statistic: { type: number }
  cluster_profiles:
    type: array
    items:
      type: object
      properties:
        cluster_id: { type: integer }
        size: { type: integer }
        pct_of_data: { type: number }
        centroid: { type: object, description: "Feature means" }
        feature_summary: { type: object, description: "mean/std per feature" }
        interpretation: { type: string, description: "Suggested regime label" }
  multimodality:
    type: object
    properties:
      dip_test_p: { type: number, description: "p<0.05 = multimodal" }
      bimodality_coefficient: { type: number, description: ">0.555 = bimodal" }
      n_modes_kde: { type: integer, description: "Modes via kernel density" }
  stability:
    type: object
    properties:
      bootstrap_ari_mean: { type: number }
      bootstrap_ari_std: { type: number }
      temporal_ari: { type: number }
      pct_stable: { type: number }
  transition_matrix:
    type: array
    description: "P(cluster_j | cluster_i) matrix. Only for temporal algorithms (HMM, sequential GMM)"
  return_differentiation:
    type: object
    description: "Do clusters predict different forward returns?"
    properties:
      kruskal_wallis_p: { type: number }
      eta_squared: { type: number }
      cluster_sharpes: { type: object }
  resource_uri: { type: string, description: "cluster://{symbol}/{date}/latest" }
```

#### `run_validation_suite`
Run the full skeptical validation (20 tests, 79+ assertions).

```yaml
name: run_validation_suite
description: |
  Run comprehensive skeptical validation of algorithmic hypotheses.
  Tests entropy distribution, persistence, predictability, feature correlations,
  baseline comparison, transaction cost survival, and more.
parameters:
  data_dir:
    type: string
    default: "./data/features"
  output_dir:
    type: string
    default: "./reports/skeptical_validation"
  timeframe:
    type: string
    enum: [tick, 5m, 15m, 1h, 4h]
    default: tick
    description: "Aggregate before testing (tick = raw data)"
returns:
  total_tests: { type: integer }
  survived: { type: integer }
  rejected: { type: integer }
  inconclusive: { type: integer }
  recommendation: { type: string, enum: [proceed, revisit, collect_more_data] }
  report_uri: { type: string }
  plot_uris: { type: array, items: { type: string } }
  critical_findings:
    type: array
    items:
      type: object
      properties:
        test: { type: string }
        verdict: { type: string }
        detail: { type: string }
```

**Implementation:** Wraps `scripts/skeptical_validation.py`.

#### `render_feature_dashboard`
Generate a multi-panel feature overview for a symbol.

```yaml
name: render_feature_dashboard
description: |
  Generate a comprehensive feature dashboard with distribution plots,
  time series, correlation heatmap, and regime overlay.
parameters:
  symbol:
    type: string
    required: true
  window:
    type: string
    default: "24h"
    description: "Time window to display"
  features:
    type: array
    items: { type: string }
    description: "Features to include (default: top 10 by variance)"
  include_panels:
    type: array
    items:
      type: string
      enum: [distributions, timeseries, correlations, pca, regime_overlay]
    default: [distributions, timeseries, correlations]
returns:
  plot_uri: { type: string, description: "plot://{symbol}/dashboard/{window}" }
  summary:
    type: object
    properties:
      n_features: { type: integer }
      n_rows: { type: integer }
      time_range: { type: string }
      nan_rate: { type: number }
```

#### `plot_entropy_regime`
Entropy-specific analysis with regime overlay.

```yaml
name: plot_entropy_regime
description: |
  Analyze entropy at specified horizon. Shows distribution, GMM clustering,
  persistence (ACF), and regime duration statistics.
parameters:
  symbol:
    type: string
    required: true
  horizon:
    type: string
    enum: [tick, 1m, 5m, 15m, 1h, 4h, 1d]
    default: "15m"
  entropy_measure:
    type: string
    enum: [permutation_8, permutation_16, permutation_32, book_shape, trade_size, all]
    default: permutation_16
returns:
  plot_uri: { type: string }
  distribution:
    type: object
    properties:
      mean: { type: number }
      std: { type: number }
      percentiles: { type: object }
      is_multimodal: { type: boolean }
      optimal_clusters: { type: integer }
      cluster_means: { type: array }
  persistence:
    type: object
    properties:
      acf_lag1: { type: number }
      half_life: { type: integer }
      is_stationary: { type: boolean }
  regime_durations:
    type: object
    properties:
      low_entropy_mean_duration: { type: number }
      high_entropy_mean_duration: { type: number }
```

#### `reduce_dimensions`
Project high-dimensional feature space to 2D/3D for visualization.

```yaml
name: reduce_dimensions
description: |
  Reduce feature space to 2D or 3D using PCA, UMAP, t-SNE, or Isomap.
  Color by cluster, entropy, volatility, or forward return.
parameters:
  symbol:
    type: string
    required: true
  method:
    type: string
    enum: [pca, umap, tsne, isomap]
    default: umap
    description: |
      pca:    Linear, preserves global variance. Fast. Good first look.
      umap:   Nonlinear, preserves global+local structure. Recommended primary.
      tsne:   Nonlinear, best local structure. Distances between clusters meaningless.
      isomap: Geodesic distance-preserving. Good for curved manifolds.
  dimensions:
    type: integer
    enum: [2, 3]
    default: 3
  color_by:
    type: string
    enum: [cluster, entropy, volatility, forward_return, regime_signal, none]
    default: cluster
    description: |
      Generate multiple colorings to cross-validate:
      if same structure appears regardless of coloring, clusters are real
  feature_set:
    type: string
    enum: [entropy_only, volatility_only, flow_only, full, custom]
    default: full
  timeframe:
    type: string
    default: "15m"
  interactive:
    type: boolean
    default: true
    description: "Generate interactive plotly HTML (3D rotation) or static PNG"
returns:
  plot_uri: { type: string, description: "plot://{symbol}/umap/3d/{color_by}" }
  explained_variance: { type: number, description: "PCA only" }
  n_points: { type: integer }
```

#### `compare_current_to_historical`
Compare current market state to historical analogs.

```yaml
name: compare_current_to_historical
description: |
  Find the K nearest historical states to the current market state.
  Reports what happened after similar states historically.
parameters:
  symbol:
    type: string
    required: true
  k:
    type: integer
    default: 20
    description: "Number of nearest neighbors"
  feature_set:
    type: string
    default: full
  distance_metric:
    type: string
    enum: [euclidean, mahalanobis, cosine]
    default: mahalanobis
  forward_horizons:
    type: array
    items: { type: string }
    default: ["5m", "1h", "4h", "24h"]
returns:
  current_state:
    type: object
    description: "Current feature vector summary"
  current_cluster: { type: integer }
  current_regime_probs: { type: object }
  neighbors:
    type: array
    items:
      type: object
      properties:
        timestamp: { type: string }
        distance: { type: number }
        cluster_at_time: { type: integer }
        forward_returns: { type: object, description: "Return at each horizon" }
  historical_outcome:
    type: object
    properties:
      mean_return: { type: object, description: "Mean return by horizon" }
      win_rate: { type: object, description: "Win rate by horizon" }
      sharpe: { type: object, description: "Sharpe by horizon" }
  plot_uri: { type: string }
```

#### `detect_phase`
Detect accumulation / uptrend / distribution / downtrend phases.

```yaml
name: detect_phase
description: |
  Classify current market phase using feature signatures.
  Uses the expected feature profiles defined in the development methodology spec.
parameters:
  symbol:
    type: string
    required: true
  timeframe:
    type: string
    default: "1h"
  confidence_threshold:
    type: number
    default: 0.6
    description: "Minimum confidence to assign a phase label"
returns:
  phase:
    type: string
    enum: [accumulation, uptrend, distribution, downtrend, ranging, noise, uncertain]
  confidence: { type: number }
  feature_evidence:
    type: object
    description: "Which features support the classification"
    properties:
      entropy: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      whale_flow: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      vpin: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      absorption: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      divergence: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      momentum: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      hurst: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
      concentration: { type: object, properties: { value: { type: number }, expected: { type: string }, matches: { type: boolean } } }
  transition_probability:
    type: object
    description: "Most likely next phase and probability"
  multi_pair:
    type: object
    description: "If multiple symbols ingested, cross-pair breadth analysis"
    properties:
      breadth: { type: number, description: "Fraction of pairs in same phase" }
      flow_alignment: { type: number, description: "Cross-pair whale flow correlation" }
      market_phase: { type: string }
```

---

## MCP Server 3: Resource MCP (`nat-resources`)

### Purpose
Expose all NAT artifacts (parquet files, plots, models, cluster labels) as
URI-addressable MCP resources. Provides the glue between Ingest and Research servers.

### Transport
**stdio** — lightweight Python process that routes URI lookups to filesystem.

### Registration
```bash
claude mcp add --transport stdio nat-resources -- python3 mcp/resource_server.py
```

### URI Scheme

```
nat://{resource_type}/{symbol}/{specifier}
```

### Resource Types

#### Data Resources (`data://`)

```yaml
# Raw tick-level parquet files
data://btcusdt/tick/2026-04-12
data://btcusdt/tick/latest
data://btcusdt/tick/2026-04-10..2026-04-12    # range

# Aggregated bars
data://btcusdt/5m/2026-04-12
data://btcusdt/1h/latest
data://btcusdt/4h/2026-04-01..2026-04-12

# Schema/metadata
data://btcusdt/schema                          # column names, types, counts
data://btcusdt/summary                         # row count, time range, NaN rates
data://_all/inventory                          # all available data across symbols
```

**Resolution:** Maps to `./data/features/` (tick) or `./data/bars/{timeframe}/` (aggregated).

**Template:**
```yaml
resource_template:
  uriTemplate: "data://{symbol}/{timeframe}/{date}"
  parameters:
    symbol: { type: string, enum: [btcusdt, ethusdt, solusdt] }
    timeframe: { type: string, enum: [tick, 1m, 5m, 15m, 1h, 4h, 1d] }
    date: { type: string, description: "YYYY-MM-DD, 'latest', or 'YYYY-MM-DD..YYYY-MM-DD'" }
  mimeType: application/parquet
```

#### Cluster Resources (`cluster://`)

```yaml
# Current cluster assignment
cluster://btcusdt/latest                       # latest cluster labels + probs
cluster://btcusdt/2026-04-12/latest            # latest for specific date

# Cluster model
cluster://btcusdt/model/gmm                    # fitted GMM parameters
cluster://btcusdt/model/hmm                    # fitted HMM parameters

# Cluster history
cluster://btcusdt/history/7d                   # cluster label time series, 7 days
cluster://btcusdt/transitions/7d               # transition matrix over 7 days

# Cluster quality
cluster://btcusdt/quality/latest               # silhouette, DB, CH, stability
cluster://btcusdt/quality/history              # quality metrics over time

# Multi-symbol
cluster://_all/breadth/latest                  # cross-pair cluster breadth
```

**Resolution:** Maps to `./data/clusters/{symbol}/` directory.

**Template:**
```yaml
resource_template:
  uriTemplate: "cluster://{symbol}/{specifier}"
  parameters:
    symbol: { type: string }
    specifier: { type: string, enum: [latest, model/gmm, model/hmm, quality/latest] }
  mimeType: application/json
```

#### Plot Resources (`plot://`)

```yaml
# Feature dashboards
plot://btcusdt/dashboard/24h                   # multi-panel feature overview
plot://btcusdt/dashboard/7d

# Entropy analysis
plot://btcusdt/entropy/distribution            # histogram + GMM overlay
plot://btcusdt/entropy/persistence             # ACF + regime duration
plot://btcusdt/entropy/regime/1h               # entropy regime at 1h horizon

# Dimensionality reduction
plot://btcusdt/umap/3d/cluster                 # 3D UMAP colored by cluster
plot://btcusdt/umap/3d/entropy                 # colored by entropy
plot://btcusdt/umap/3d/forward_return          # colored by forward return
plot://btcusdt/tsne/2d/cluster
plot://btcusdt/pca/2d/variance                 # scree plot + biplot

# Correlations
plot://btcusdt/correlations/heatmap            # feature correlation matrix
plot://btcusdt/correlations/by_cluster         # correlation within each cluster

# Validation
plot://btcusdt/validation/latest               # skeptical validation plots
plot://btcusdt/validation/feature_returns      # feature-return correlation plot
plot://btcusdt/validation/baselines            # strategy baseline comparison

# Phase detection
plot://btcusdt/phase/current                   # current phase with evidence
plot://btcusdt/phase/history/30d               # phase timeline
```

**Resolution:** Maps to `./reports/` directory. Plots generated on-demand if not cached.

**Template:**
```yaml
resource_template:
  uriTemplate: "plot://{symbol}/{plot_type}/{specifier}"
  parameters:
    symbol: { type: string }
    plot_type: { type: string, enum: [dashboard, entropy, umap, tsne, pca, correlations, validation, phase] }
    specifier: { type: string }
  mimeType: image/png
```

#### Replay Resources (`replay://`)

```yaml
# Replay a historical session
replay://btcusdt/session/2026-04-12T14:00Z     # replay from specific time
replay://btcusdt/session/2026-04-12T14:00Z/speed/10x  # 10x playback speed

# Replay with cluster overlay
replay://btcusdt/session/2026-04-12T14:00Z/with/clusters
replay://btcusdt/session/2026-04-12T14:00Z/with/phases
```

**Resolution:** Uses `start_ingestion(mode=replay)` with the specified parquet files.

#### Model Resources (`model://`)

```yaml
# Clustering models
model://btcusdt/gmm/latest                     # latest fitted GMM
model://btcusdt/hmm/latest                     # latest fitted HMM
model://btcusdt/gmm/params                     # means, covariances, weights

# Feature selection
model://btcusdt/feature_selection/latest        # which features survived selection
model://btcusdt/pca/components                  # PCA loadings

# Validation
model://btcusdt/validation/report               # JSON validation report
```

---

## Feature Set Reference

Available feature subsets for clustering and analysis tools. These map directly
to the Rust `FeatureComputer` modules.

### Predefined Sets

| Set Name | Features | Count | Module |
|----------|----------|-------|--------|
| `entropy_only` | ent_permutation_8/16/32, ent_book_shape, ent_trade_size, ent_rate_of_change, ent_zscore + 17 more entropy variants | 24 | `entropy.rs` |
| `volatility_only` | vol_realized_20/100, vol_ratio, parkinson_vol, garman_klass_vol + 3 more | 8 | `volatility.rs` |
| `flow_only` | flow_aggressor_ratio/momentum, flow_volume_5s, flow_trade_count_5s + 8 more | 12 | `flow.rs` |
| `microstructure` | raw_mid_price, raw_spread_bps, raw_bid/ask_depth_l5, imbalance_l5/l10 + 4 more | 10+8 | `raw.rs`, `imbalance.rs` |
| `trend` | momentum_60/300/600, hurst_300, r_squared_300, monotonicity_300 + 9 more | 15 | `trend.rs` |
| `illiquidity` | kyle_lambda_100/500, amihud_100/500, hasbrouck_lambda, roll_spread + 6 more | 12 | `illiquidity.rs` |
| `toxicity` | vpin_10/50, adverse_selection, informed_probability + 6 more | 10 | `toxicity.rs` |
| `whale_flow` | whale_net_flow_1h/4h/24h, whale_flow_intensity/momentum + 7 more | 12 | `whale_flow.rs` |
| `liquidation` | liq_risk_above/below_1/2/5/10_pct, liq_cluster_distance + 5 more | 13 | `liquidation.rs` |
| `concentration` | hhi, gini, top_5/10/20_pct, whale_dominance + 9 more | 15 | `concentration.rs` |
| `regime` | absorption_zscore, divergence_zscore, churn_4h, range_pos_24h + 16 more | 20 | `regime/*.rs` |
| `derived` | entropy_trend_interaction, flow_vol_interaction + 13 more | 15 | `derived.rs` |
| `context` | funding_rate, funding_zscore, open_interest, oi_change + 5 more | 9 | `context.rs` |
| `full` | All of the above | 183 | All modules |

### Clustering Algorithm Parameters

| Algorithm | Key Parameters | Auto Cluster Count | Temporal | Soft Assignments |
|-----------|---------------|-------------------|----------|-----------------|
| `gmm` | n_components, covariance_type | Via BIC | No | Yes (probabilities) |
| `kmeans` | n_clusters | Via elbow/gap | No | No |
| `hdbscan` | min_cluster_size, min_samples | Yes | No | Yes (probabilities) |
| `spectral` | n_clusters, affinity | Via eigengap | No | No |
| `hmm` | n_components | Via BIC | Yes (transitions) | Yes (posterior) |
| `dpgmm` | max_components, weight_concentration | Yes (Dirichlet) | No | Yes |
| `agglomerative` | n_clusters, linkage | Via dendrogram | No | No |

### Quality Metric Thresholds

| Metric | Poor | Acceptable | Good | Strong |
|--------|------|------------|------|--------|
| Silhouette | < 0.1 | 0.1 - 0.25 | 0.25 - 0.5 | > 0.5 |
| Davies-Bouldin | > 2.0 | 1.0 - 2.0 | 0.5 - 1.0 | < 0.5 |
| Bootstrap ARI | < 0.4 | 0.4 - 0.6 | 0.6 - 0.8 | > 0.8 |
| Return differentiation (p) | > 0.1 | 0.05 - 0.1 | 0.01 - 0.05 | < 0.01 |
| Effect size (eta^2) | < 0.01 | 0.01 - 0.06 | 0.06 - 0.14 | > 0.14 |

---

## Composition with External MCP Servers

### Quant / Finance MCP Servers

```bash
# Cross-reference NAT signals with external data
claude mcp add --transport http quantconnect <url>   # backtesting, portfolio opt
claude mcp add --transport http alphavantage <url>    # historical prices, forex
claude mcp add --transport http crypto-indicators <url> # technical analysis

# Example composed workflow:
# 1. nat-ingest: start_ingestion(["BTC", "ETH"])
# 2. nat-research: cluster_features("BTC", timeframe="1h", algorithm="gmm")
# 3. nat-research: detect_phase("BTC")
# 4. quantconnect: backtest(strategy based on cluster transitions)
# 5. nat-resources: cluster://btcusdt/quality/latest  (verify cluster quality)
```

### Math / Statistics MCP Servers

```bash
claude mcp add --transport http wolfram <url>         # symbolic math
claude mcp add --transport stdio jupyter <cmd>        # interactive notebooks

# Example: validate statistical formulas
# 1. nat-research: cluster_features(...) -> silhouette = 0.42
# 2. wolfram: "Is silhouette score 0.42 with n=5000 significantly > 0.25?"
```

---

## Implementation Roadmap

### Phase 1: Ingest MCP (Week 1)
- [ ] `mcp/ingest_server.py` — stdio MCP wrapper around Rust binary
- [ ] `start_ingestion`, `stop_ingestion`, `get_ingest_status`, `list_active_streams`
- [ ] `aggregate_timeframe` (the critical blocker)
- [ ] `.mcp.json` project config

### Phase 2: Resource MCP (Week 2)
- [ ] `mcp/resource_server.py` — URI router
- [ ] `data://` resolution (parquet file lookup)
- [ ] `plot://` resolution (on-demand generation + caching)
- [ ] Resource templates and completions

### Phase 3: Research MCP (Week 2-3)
- [ ] `mcp/research_server.py` — analysis wrapper
- [ ] `cluster_features` (all 7 algorithms + quality metrics)
- [ ] `run_validation_suite` (wraps existing script)
- [ ] `reduce_dimensions` (PCA, UMAP, t-SNE, Isomap)
- [ ] `render_feature_dashboard`, `plot_entropy_regime`

### Phase 4: Phase Detection (Week 3-4)
- [ ] `detect_phase` (accumulation/uptrend/distribution signatures)
- [ ] `compare_current_to_historical` (KNN state retrieval)
- [ ] Multi-pair breadth analysis

### Phase 5: External Composition (Week 4+)
- [ ] Register external quant MCP servers
- [ ] Build composed workflows (ingest -> cluster -> detect -> backtest)
- [ ] Claude Code skill that orchestrates the three MCP servers

---

## Project-Level MCP Configuration

### `.mcp.json` (project root)

```json
{
  "mcpServers": {
    "nat-ingest": {
      "type": "stdio",
      "command": "python3",
      "args": ["mcp/ingest_server.py"],
      "env": {
        "NAT_DATA_DIR": "./data/features",
        "NAT_BINARY": "./target/release/ing",
        "NAT_CONFIG": "./config/ing.toml"
      }
    },
    "nat-research": {
      "type": "stdio",
      "command": "python3",
      "args": ["mcp/research_server.py"],
      "env": {
        "NAT_DATA_DIR": "./data/features",
        "NAT_BARS_DIR": "./data/bars",
        "NAT_REPORTS_DIR": "./reports"
      }
    },
    "nat-resources": {
      "type": "stdio",
      "command": "python3",
      "args": ["mcp/resource_server.py"],
      "env": {
        "NAT_ROOT": "."
      }
    }
  }
}
```

### Claude Code Skill (`.claude/skills/nat/SKILL.md`)

```yaml
---
name: nat
description: |
  NAT market microstructure analysis. Use for: data ingestion from Hyperliquid,
  feature clustering and regime detection, entropy analysis, phase detection
  (accumulation/uptrend/distribution), dimensionality reduction visualization.
  183 features across 14 categories.
user-invocable: true
allowed-tools: |
  MCP(nat-ingest:*)
  MCP(nat-research:*)
  MCP(nat-resources:*)
  Bash(python3 scripts/*)
  Bash(./target/release/*)
  Read Glob Grep
---

## NAT Microstructure Analysis System

When invoked, analyze the request and route to the appropriate MCP server:

- Data collection / ingestion -> nat-ingest
- Clustering / statistics / visualization -> nat-research
- Fetching artifacts by URI -> nat-resources

Available commands: $ARGUMENTS
```

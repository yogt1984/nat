# NAT Daily Development Tasks

**Goal:** Collect live data on remote machine, transfer, validate, aggregate, cluster, visualize.
**Start date:** 2026-04-12
**Decision gate:** After Day 7-10 — do clusters exist and differentiate returns?

---

## Current State (What Exists)

| Component | Status | Location |
|-----------|--------|----------|
| Rust ingestor binary | Builds, runs | `rust/target/release/ing` |
| Config | Ready | `config/ing.toml` (BTC, ETH, SOL) |
| Parquet writer | Working | Hourly rotation, zstd compression |
| Data validation script | Working | `scripts/validate_data.py` |
| Cluster exploration script | Working | `scripts/explore_clusters.py` (PCA, UMAP, t-SNE, GMM, HDBSCAN) |
| Cluster quality metrics | Working | `scripts/cluster_quality/` (silhouette, DB, bootstrap ARI) |
| Visualization scripts | Working | `scripts/viz/` (features, correlations, distributions) |
| Skeptical validation suite | Working | `scripts/skeptical_validation.py` (79 tests) |
| Timeframe aggregation | **MISSING** | Needed: tick -> 5m/15m/1h/4h bars |
| Python requirements file | **MISSING** | Needed for remote machine setup |
| Data transfer script | **MISSING** | Needed: rsync from remote |
| Multi-timeframe clustering | **MISSING** | Needed: cluster at each horizon |

---

## Day 1: Remote Machine Setup & Start Ingestion

### Tasks

- [ ] **1.1** Create `requirements.txt` with all Python dependencies
  ```
  pyarrow>=14.0
  pandas>=2.0
  numpy>=1.24
  scipy>=1.11
  scikit-learn>=1.3
  matplotlib>=3.7
  seaborn>=0.12
  polars>=0.19
  ```
  Run `pip install -r requirements.txt` on both machines.

- [ ] **1.2** Copy project to remote machine
  ```bash
  rsync -avz --exclude='target/' --exclude='data/' \
    /home/onat/nat/ remote:/path/to/nat/
  ```

- [ ] **1.3** Build the ingestor on remote machine
  ```bash
  ssh remote "cd /path/to/nat && make release"
  ```

- [ ] **1.4** Start ingestion on remote machine
  ```bash
  ssh remote "cd /path/to/nat && nohup make run > ingest.log 2>&1 &"
  ```
  Verify it's running: check `data/features/` for new parquet files appearing.

- [ ] **1.5** Create `scripts/sync_data.sh` — data transfer script
  ```bash
  #!/bin/bash
  # Sync parquet data from remote ingestion machine
  REMOTE="user@remote-host"
  REMOTE_DIR="/path/to/nat/data/features"
  LOCAL_DIR="./data/features"
  rsync -avz --progress "$REMOTE:$REMOTE_DIR/" "$LOCAL_DIR/"
  echo "Synced. Local data:"
  du -sh "$LOCAL_DIR"
  find "$LOCAL_DIR" -name "*.parquet" -size +0 | wc -l
  ```

- [ ] **1.6** Verify first data arrives
  ```bash
  bash scripts/sync_data.sh
  python scripts/validate_data.py ./data/features --verbose
  ```

### Done when
- Ingestor running on remote, parquet files growing hourly
- Can sync data to local machine
- `validate_data.py` passes on synced data

---

## Day 2: Validate Live Data & Create Aggregation Pipeline

### Tasks

- [ ] **2.1** Sync overnight data
  ```bash
  bash scripts/sync_data.sh
  ```
  Expect: ~24 parquet files (1 per hour), several GB for 3 symbols at 100ms.

- [ ] **2.2** Run data validation
  ```bash
  python scripts/validate_data.py ./data/features --verbose
  ```
  Check: NaN rates, gaps, feature ranges, row counts per symbol.

- [ ] **2.3** Inspect live data schema — confirm column names match scripts
  ```python
  import pyarrow.parquet as pq
  t = pq.read_table("data/features/<latest_file>.parquet")
  print(t.schema)
  print(f"Rows: {t.num_rows}, Cols: {t.num_columns}")
  ```
  **Important:** Live data column names may differ from synthetic data.
  Update `ENTROPY_COLS` and `FEATURE_COLS_ALL` in `skeptical_validation.py`
  and `FEATURE_SUBSETS` in `explore_clusters.py` if needed.

- [ ] **2.4** Create `scripts/aggregate_bars.py` — the critical missing piece
  Takes tick parquet, outputs aggregated bars at specified timeframe.
  For each feature compute: mean, std, min, max, close (last value).
  Special: OHLC for midprice, sum for volume, slope for entropy.
  Input: `./data/features/*.parquet`
  Output: `./data/bars/{timeframe}/{symbol}_{date}.parquet`
  Timeframes: 5m, 15m, 1h, 4h

- [ ] **2.5** Run aggregation on collected data
  ```bash
  python scripts/aggregate_bars.py --input ./data/features --output ./data/bars --timeframes 5m,15m,1h,4h
  ```

- [ ] **2.6** Validate aggregated bars make sense
  Check: bar count matches expected (24h / timeframe), no NaN explosions,
  OHLC is consistent (high >= open,close; low <= open,close).

### Done when
- `aggregate_bars.py` works and produces clean bars at 4 timeframes
- Both tick and bar data validated

---

## Day 3: First Look — Entropy Distribution on Live Data

### Tasks

- [ ] **3.1** Sync latest data, aggregate to bars

- [ ] **3.2** Run entropy distribution analysis on live data
  Use the skeptical validation suite on live 15-min bars:
  ```bash
  python scripts/skeptical_validation.py --data ./data/bars/15m --output ./reports/live_15m_validation
  ```
  **Key questions to answer:**
  - Is entropy multimodal? (GMM BIC test)
  - What are the natural percentiles? (Not 0.3/0.7 — what does data say?)
  - How persistent is entropy? (ACF half-life — is it hours? minutes?)
  - Does entropy predict forward returns at 15m horizon?

- [ ] **3.3** Run entropy analysis at EACH timeframe
  ```bash
  for tf in 5m 15m 1h 4h; do
    python scripts/skeptical_validation.py --data ./data/bars/$tf --output ./reports/live_${tf}_validation
  done
  ```
  Compare results across timeframes. Which horizon shows the cleanest structure?

- [ ] **3.4** Generate entropy distribution plots for each symbol
  Visually inspect: do BTC, ETH, SOL have similar entropy distributions?
  Do they cluster similarly?

- [ ] **3.5** Record findings
  Write a short note in `reports/day3_entropy_findings.txt`:
  - Is entropy multimodal? At which timeframe?
  - What are the empirical thresholds (P25, P75)?
  - ACF half-life at each timeframe
  - Any notable differences between symbols

### Done when
- Entropy distribution analyzed at 4 timeframes for 3 symbols
- Know whether multimodality exists and at which horizon
- Findings documented

---

## Day 4: Feature Subspace Clustering

### Tasks

- [ ] **4.1** Sync + aggregate latest data

- [ ] **4.2** Run `explore_clusters.py` on the best timeframe from Day 3
  ```bash
  python scripts/explore_clusters.py --data-dir ./data/bars/15m --subset entropy
  python scripts/explore_clusters.py --data-dir ./data/bars/15m --subset volatility
  python scripts/explore_clusters.py --data-dir ./data/bars/15m --subset flow
  python scripts/explore_clusters.py --data-dir ./data/bars/15m --subset regime
  python scripts/explore_clusters.py --data-dir ./data/bars/15m --subset all
  ```

- [ ] **4.3** For each subspace, record:
  - Optimal cluster count (BIC/silhouette)
  - Silhouette score
  - Davies-Bouldin index
  - Whether clusters are visually separable in PCA 2D

- [ ] **4.4** Run GMM with BIC selection (1-8 components) on each subspace
  ```python
  from sklearn.mixture import GaussianMixture
  for n in range(1, 9):
      gmm = GaussianMixture(n_components=n, random_state=42)
      gmm.fit(X)
      print(f"k={n}: BIC={gmm.bic(X):.0f}")
  ```

- [ ] **4.5** Identify the top 2-3 subspaces with clearest clustering
  These become candidates for regime detection.

- [ ] **4.6** Record findings in `reports/day4_clustering_findings.txt`

### Done when
- Clustering tested on 5+ feature subspaces
- Know which subspaces cluster and which don't
- Have optimal cluster counts per subspace

---

## Day 5: Dimensionality Reduction & Visualization

### Tasks

- [ ] **5.1** Install UMAP if not present: `pip install umap-learn`

- [ ] **5.2** For the best-clustering subspace from Day 4, generate:
  - PCA 2D and 3D (fast, global structure)
  - UMAP 2D and 3D (nonlinear structure)
  - t-SNE 2D (local structure confirmation)

- [ ] **5.3** Color each projection by 4 different variables:
  1. Cluster assignment (from GMM)
  2. Entropy level (continuous)
  3. Forward return sign (positive/negative)
  4. Volatility regime (low/high)

  **If the same groupings appear across all colorings, clusters are real.**

- [ ] **5.4** Generate interactive 3D plotly HTML for UMAP
  ```python
  import plotly.express as px
  fig = px.scatter_3d(df, x='umap_0', y='umap_1', z='umap_2',
                      color='cluster', hover_data=['entropy', 'vol'])
  fig.write_html('reports/umap_3d_interactive.html')
  ```

- [ ] **5.5** Cross-timeframe comparison
  Run the same UMAP on 5m, 15m, 1h bars. Do the same clusters appear?
  This tests whether the structure is robust to aggregation level.

- [ ] **5.6** Save all plots to `reports/day5_visualization/`

### Done when
- Have 2D and 3D projections at multiple timeframes
- Know whether clusters are visually separable
- Interactive 3D HTML for exploration

---

## Day 6: Cluster Quality & Return Differentiation

### Tasks

- [ ] **6.1** Run full cluster quality analysis
  ```bash
  python scripts/analyze_clusters.py --data-dir ./data/bars/15m --symbol BTC
  ```

- [ ] **6.2** For the best clustering, compute:
  - Bootstrap ARI (50 resamples) — target > 0.6
  - Temporal ARI (first half vs second half) — target > 0.5
  - Kruskal-Wallis on forward returns by cluster — target p < 0.05
  - Eta-squared (effect size) — target > 0.01
  - Cluster-conditional Sharpe ratios

- [ ] **6.3** Transition analysis
  - Compute transition matrix: P(cluster_j at t+1 | cluster_i at t)
  - Self-transition rate: do clusters persist? (target > 0.7)
  - Mean regime duration per cluster
  - Are transitions predictable?

- [ ] **6.4** Multimodality tests on each feature individually
  For each of the 183 features (or available subset), compute:
  - Dip test p-value
  - Bimodality coefficient
  - GMM BIC (1 vs 2 components)
  Rank features by multimodality. The most multimodal features are
  the best candidates for regime detection.

- [ ] **6.5** Record findings in `reports/day6_quality.txt`

### Done when
- Cluster quality quantified with all metrics from the spec
- Know whether clusters differentiate forward returns
- Have ranked features by multimodality

---

## Day 7: Decision Gate

### Tasks

- [ ] **7.1** Compile all findings from Days 3-6

- [ ] **7.2** Answer these questions with data:

  **Q1: Do natural clusters exist?**
  - Silhouette > 0.25 at any timeframe/subspace? YES/NO
  - GMM optimal k > 1? YES/NO
  - Dip test significant for any feature? YES/NO

  **Q2: Are clusters stable?**
  - Bootstrap ARI > 0.6? YES/NO
  - Temporal ARI > 0.5? YES/NO
  - Self-transition rate > 0.7? YES/NO

  **Q3: Do clusters predict returns?**
  - Kruskal-Wallis p < 0.05? YES/NO
  - Eta-squared > 0.01? YES/NO
  - Cluster-conditional Sharpes differ meaningfully? YES/NO

  **Q4: At which timeframe?**
  - Best timeframe: ___
  - Best feature subspace: ___
  - Optimal cluster count: ___

- [ ] **7.3** Decision:
  - **GO:** Majority YES on Q1-Q3 → proceed to phase detection (Day 8+)
  - **PIVOT:** Clusters exist but don't predict returns → try different features, timeframes, or derivatives of cluster transitions
  - **NO-GO:** No clusters found → reassess entire approach, consider this a data engineering project with standalone value

- [ ] **7.4** Write decision document: `reports/decision_gate.md`

### Done when
- Clear GO / PIVOT / NO-GO decision with evidence

---

## Day 8-10 (If GO): Phase Detection & Multi-Pair Analysis

### Tasks (only if Day 7 = GO)

- [ ] **8.1** Map clusters to market phases
  Using the feature signatures from the MCP spec (accumulation, uptrend,
  distribution, downtrend, ranging), label each cluster.

- [ ] **8.2** Validate phase labels against price action
  - Does "accumulation" cluster precede price rises?
  - Does "distribution" cluster precede price drops?
  - What's the hit rate?

- [ ] **8.3** Multi-pair breadth analysis
  - Compute regime for BTC, ETH, SOL independently
  - Measure cross-pair agreement (breadth)
  - Does breadth > 0.6 correlate with market-wide moves?

- [ ] **8.4** Run skeptical validation on the best timeframe with live data
  ```bash
  python scripts/skeptical_validation.py --data ./data/bars/<best_tf>
  ```
  Compare results vs synthetic data run. How many more tests survive?

---

## Daily Routine (Every Day)

```bash
# 1. Sync data from remote (30 seconds)
bash scripts/sync_data.sh

# 2. Validate new data (1 minute)
python scripts/validate_data.py ./data/features --hours 24

# 3. Aggregate to bars (2 minutes)
python scripts/aggregate_bars.py --input ./data/features --output ./data/bars --timeframes 5m,15m,1h,4h

# 4. Quick check: is ingestion still running on remote?
ssh remote "ps aux | grep ing"
```

---

## Files to Create

| File | Day | Purpose |
|------|-----|---------|
| `requirements.txt` | 1 | Python dependencies |
| `scripts/sync_data.sh` | 1 | rsync from remote |
| `scripts/aggregate_bars.py` | 2 | Tick -> bar aggregation (THE blocker) |
| `reports/day3_entropy_findings.txt` | 3 | Entropy analysis results |
| `reports/day4_clustering_findings.txt` | 4 | Clustering results per subspace |
| `reports/day5_visualization/` | 5 | UMAP/PCA/t-SNE plots |
| `reports/day6_quality.txt` | 6 | Cluster quality metrics |
| `reports/decision_gate.md` | 7 | GO / PIVOT / NO-GO decision |

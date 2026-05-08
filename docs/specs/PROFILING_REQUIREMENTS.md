# NAT Profiling System — Requirements Document

**Version:** 1.0
**Date:** 2026-04-26
**Author:** Yigit Onat
**Status:** Draft

---

## 1. Purpose

The Profiling System extracts discrete market states from continuous microstructure observables collected via the NAT ingestor. Its output is a real-time classification: "the market is currently in state X with Y% confidence, and the most likely next state is Z." This classification forms the foundation for regime-aware trading algorithms.

---

## 2. Scope

### 2.1 In Scope

- Feature enrichment pipeline (temporal derivatives, spectral decomposition, multi-timeframe alignment)
- Unsupervised state discovery via clustering (GMM, with extensibility to HMM)
- State validation framework (separation, stability, persistence, recurrence, predictive quality)
- Transition modeling and state duration characterization
- Online regime detector for real-time state classification
- Visualization suite for human-in-the-loop state interpretation
- Research reporting and decision gate framework

### 2.2 Out of Scope

- Order execution and position management
- Strategy backtesting engine (separate system, consumes profiling output)
- Exchange connectivity (handled by ingestor)
- Portfolio optimization

---

## 3. System Context

```
Hyperliquid WebSocket
    → Rust Ingestor (194 features @ 100ms, BTC/ETH/SOL)
    → Parquet files (data/features/)
    → Profiling System (this document)
        → Feature Enrichment
        → State Discovery
        → Transition Model
        → Online Detector
    → State labels + confidence → Strategy Layer (future)
```

### 3.1 Upstream Dependencies

| Component | Interface | Format |
|-----------|-----------|--------|
| Rust Ingestor | Parquet files in `data/features/YYYY-MM-DD/` | 194 columns, 100ms emission, zstd compressed |
| Pipeline Config | `config/pipeline.toml` | TOML |

### 3.2 Downstream Consumers

| Consumer | Interface | Format |
|----------|-----------|--------|
| Strategy Layer | State labels + transition probabilities | JSON / DataFrame |
| Visualization | Figures + interactive dashboards | PNG / HTML |
| Research Reports | Decision gate summaries | Markdown + JSON |

---

## 4. Functional Requirements

### 4.1 Feature Enrichment Pipeline

#### FR-4.1.1 Temporal Derivatives

The system SHALL compute the following temporal derivatives for each base feature:

| Derivative | Description | Parameters |
|-----------|-------------|------------|
| Rolling mean | Low-pass filter at multiple windows | windows: [5, 10, 20, 50] bars |
| Rolling std | Local volatility estimate | windows: [5, 10, 20, 50] bars |
| First difference | Velocity of feature change | lag: 1 bar |
| Second difference | Acceleration of feature change | lag: 1 bar |
| Rate of change | Percentage change | lag: [1, 5, 10] bars |
| EMA | Exponential moving average | spans: [5, 10, 20, 50] bars |
| Z-score | Deviation from rolling mean | window: 20 bars |

#### FR-4.1.2 Spectral Decomposition

The system SHALL decompose selected features into frequency-domain representations:

- FFT magnitude spectrum per rolling window (window: 64 bars, hop: 1 bar)
- Dominant frequency extraction (top 3 peaks)
- Spectral entropy (distribution of energy across frequencies)
- Optional: wavelet decomposition (Daubechies-4) at 3 scale levels

#### FR-4.1.3 Multi-Timeframe Alignment

The system SHALL compute feature vectors at multiple timeframes simultaneously:

- Supported timeframes: 5min, 15min, 30min, 1h, 2h, 4h
- Cross-timeframe agreement metric: fraction of timeframes assigning the same state
- Fast/slow regime divergence signal: when 5min and 1h states disagree

#### FR-4.1.4 Cross-Feature Interactions

The system SHALL compute selected interaction terms:

- entropy x orderflow_imbalance (information x directional pressure)
- illiquidity x volatility (liquidity-adjusted risk)
- orderflow_imbalance x trend_momentum (confirmation signal)
- Interaction terms SHALL be configurable via pipeline.toml

### 4.2 State Discovery

#### FR-4.2.1 Clustering

The system SHALL support the following clustering methods:

| Method | Use Case | Parameters |
|--------|----------|------------|
| GMM (full covariance) | Primary state discovery | k: 2-10, n_init: 10, random_state: 42 |
| GMM (diagonal covariance) | High-dimensional vectors | Same as above |
| HDBSCAN | Noise-robust discovery | min_cluster_size: 5-20 |

The system SHALL perform k-sweep for GMM and select best k by:
- Primary: minimum BIC
- Secondary: maximum silhouette score
- Tie-breaking: prefer lower k

#### FR-4.2.2 Soft Assignment

The system SHALL output posterior probabilities P(state_i | observation) for each bar, not only hard labels. Transition zones (max probability < 0.7) SHALL be flagged as "uncertain."

#### FR-4.2.3 Preprocessing

For each vector-timeframe combination, the system SHALL:

1. Drop columns with > 50% NaN
2. Drop near-zero-variance columns (std < 1e-8)
3. Fill remaining NaN with column median
4. Clip outliers beyond 5 standard deviations
5. Apply z-score standardization
6. Log preprocessing decisions (columns dropped, NaN counts, clipping counts)

### 4.3 State Validation

#### FR-4.3.1 Quality Gates

Each discovered state configuration SHALL pass through three quality gates:

| Gate | Metric | Threshold | Interpretation |
|------|--------|-----------|----------------|
| Q1 — Existence | Silhouette score | > 0.25 | Clusters are geometrically separated |
| Q2 — Stability | Bootstrap ARI > 0.6 AND Temporal ARI > 0.5 | Both must pass | Clusters are reproducible and time-stable |
| Q3 — Predictive | Kruskal-Wallis p < 0.05 AND eta-squared > 0.01 | Both must pass | States predict forward return distributions |

#### FR-4.3.2 Bootstrap Stability

- Resample 80% of observations, re-cluster, measure ARI to original labels
- Minimum 30 resamples, report mean and std of ARI
- Random seed: configurable (default: 42)

#### FR-4.3.3 Temporal Stability

- Split data into first-half and second-half
- Cluster each half independently
- Measure ARI between the two halves

#### FR-4.3.4 Persistence Validation

Each state SHALL satisfy:

| Property | Metric | Threshold |
|----------|--------|-----------|
| Persistence | Self-transition rate | > 0.50 (minimum), > 0.70 (strong) |
| Recurrence | State appears in >= 3 non-contiguous segments | Required |
| Balance | No state contains > 80% of observations | Warning if violated |
| Duration | Median state duration > 2 bars | Required |

#### FR-4.3.5 Cross-Asset Validation

The system SHALL measure cross-asset state agreement:

- For each time step, compute fraction of symbols (BTC/ETH/SOL) in same state
- Report mean agreement and fraction of time at full agreement
- States driven by market-wide dynamics (agreement > 0.6) are preferred over symbol-specific artifacts

### 4.4 Transition Modeling

#### FR-4.4.1 Transition Matrix

The system SHALL compute the empirical transition matrix:

- P(state_t+1 | state_t) for all state pairs
- Report as heatmap with numerical values
- Identify forbidden transitions (P < 0.05)
- Identify asymmetric transitions (P(A→B) significantly differs from P(B→A))

#### FR-4.4.2 Hidden Markov Model

The system SHALL support fitting a formal HMM:

- Fit via Baum-Welch (EM) algorithm
- Compare HMM log-likelihood to null model (random state assignments)
- Compare HMM-predicted states to GMM labels (ARI)
- Out-of-sample: train on first 70%, predict on last 30%
- Report transition matrix, emission means/covariances, stationary distribution

#### FR-4.4.3 State Duration Distribution

For each state, the system SHALL report:

- Mean, median, min, max duration (in bars)
- Duration distribution (histogram/violin plot)
- Whether duration follows geometric distribution (expected for Markov) or has heavier tails

### 4.5 State Characterization

#### FR-4.5.1 Return Profiling

For each state, the system SHALL compute:

| Metric | Description |
|--------|-------------|
| Mean forward return | 1-bar, 3-bar, 5-bar horizons |
| Return std | Volatility within state |
| Skewness | Directional bias |
| Win rate | Fraction of positive returns |
| Sharpe ratio | Risk-adjusted return (annualized) |
| Max drawdown | Worst cumulative loss within state |

#### FR-4.5.2 Feature Signature

For each state, the system SHALL report:

- Centroid (mean feature values, z-scored)
- Top 5 most discriminative features (by F-ratio)
- Feature value ranges (5th-95th percentile)

#### FR-4.5.3 Entry/Exit Signatures

The system SHALL compute:

- Average feature trajectory 5 bars before entering each state
- Average feature trajectory 5 bars before exiting each state
- These signatures enable early transition detection

#### FR-4.5.4 Cross-Asset Behavior

For each state, report:

- Which symbols are typically in this state
- Lead/lag: does one symbol enter the state before others?
- Correlation of returns across symbols within this state

### 4.6 Online Regime Detector

#### FR-4.6.1 Real-Time Classification

The system SHALL classify incoming bars in real-time:

- Input: new bar of aggregated features
- Output: state label, posterior probabilities, confidence flag
- Latency: < 100ms per classification
- Model: pre-fitted GMM loaded from disk

#### FR-4.6.2 Model Management

- Models SHALL be serialized to disk (pickle or joblib)
- Models SHALL include metadata: training date, data range, k, vector, timeframe, quality metrics
- Models SHALL be versioned and reproducible from config + data

#### FR-4.6.3 Drift Detection

The system SHALL monitor for regime drift:

- Track rolling silhouette score on recent N bars (default: 50)
- If rolling silhouette drops below 0.15 for 20+ consecutive bars, flag "model drift"
- Track rolling ARI between current model labels and re-fitted labels on recent data
- If ARI < 0.4, flag "state structure changed — consider refitting"

### 4.7 Visualization

#### FR-4.7.1 Required Plots

The system SHALL generate the following visualizations:

| Plot | Purpose | Human Decision |
|------|---------|----------------|
| PCA scatter + loadings | See cluster geometry, which features drive separation | Name the states |
| Centroid heatmap | What makes each state distinct | Validate interpretability |
| State timeline + price | When do states occur, do they align with moves? | Detect time-of-day artifacts |
| Return distributions per state | Do states predict returns differently? | Decide if tradeable |
| Transition matrix heatmap | How does market move between states? | Find structural patterns |
| State duration violin | How long do states last? | Judge persistence |
| t-SNE 4-view (cluster/symbol/time/feature) | Manifold structure from multiple angles | Detect symbol vs regime effects |
| Cross-asset alignment | Do BTC/ETH/SOL share regimes? | Validate market-wide structure |
| Feature evolution (top 6) | Raw feature time series colored by state | Build raw intuition |
| Summary dashboard | All configs compared | Select best configuration |

#### FR-4.7.2 Visualization Format

- Output format: PNG (default), configurable DPI (default: 150)
- Output directory: `reports/figures/`
- File naming: `{plot_type}_{vector}_{timeframe}.png`

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Requirement |
|--------|-------------|
| Full sweep (9 vectors x 6 timeframes) | < 60 minutes on single machine |
| Single vector-timeframe analysis | < 5 minutes |
| Online classification latency | < 100ms per bar |
| Memory usage for 7-day dataset | < 8 GB peak |

### 5.2 Reproducibility

- All random operations SHALL use configurable seeds (default: 42)
- All results SHALL be reproducible from config + data
- All analysis runs SHALL log: timestamp, config hash, data hash, git commit

### 5.3 Configurability

All thresholds, parameters, and vector selections SHALL be configurable via `config/pipeline.toml`. No magic numbers in code.

### 5.4 Testing

- Unit tests for all preprocessing, clustering, and validation functions
- Integration test: synthetic data with known cluster structure, verify gates pass
- Regression test: fixed dataset produces identical results across code changes

---

## 6. Data Requirements

### 6.1 Minimum Data for Analysis

| Analysis Type | Minimum Data | Recommended |
|---------------|-------------|-------------|
| Initial exploration | 24 hours (~288 bars at 15min) | 48 hours |
| Decision gate | 48 hours | 7 days |
| HMM fitting | 7 days | 14+ days |
| Online detector training | 7 days | 30 days |
| Drift detection baseline | 30 days | 90 days |

### 6.2 Required Feature Vectors

Minimum viable set (from initial results):

| Priority | Vector | Justification |
|----------|--------|---------------|
| P0 | orderflow | Passes Q1+Q2 at 3 timeframes, highest temporal stability (0.806) |
| P0 | entropy | Passes Q1+Q2 at 5min, near-perfect bootstrap (0.984) |
| P1 | illiquidity | Highest separation at all timeframes (sil 0.54-0.58) |
| P1 | derived | Passes Q1+Q2 at 15min |
| P2 | volatility | Passes Q1+Q2 at 2h |
| P2 | trend | Borderline Q1, excellent bootstrap |

---

## 7. Validation Criteria

### 7.1 System Acceptance

The profiling system is considered validated when:

1. At least 2 vector-timeframe combinations pass all three gates (Q1+Q2+Q3) on 7-day data
2. The HMM log-likelihood exceeds the null model by > 2x
3. At least one state has self-transition rate > 0.70
4. Cross-asset agreement > 0.60 for the primary configuration
5. The online detector achieves ARI > 0.7 between online labels and batch-fitted labels on held-out data

### 7.2 Decision Gate

| Decision | Condition | Action |
|----------|-----------|--------|
| **GO** | >= 2 vectors pass Q1+Q2+Q3, HMM validated | Build strategy layer |
| **PIVOT** | Q1+Q2 pass but Q3 fails | Try feature enrichment, composite vectors, different timeframes |
| **NO-GO** | No reliable clusters on 7-day data | Fundamental reassessment of feature design |

---

## 8. Implementation Phases

### Phase 1: Foundation (current — complete)
- [x] Ingestor (194 features, 100ms, 3 symbols)
- [x] Parquet storage with hourly rotation
- [x] Bar aggregation (6 timeframes)
- [x] Preprocessing pipeline (z-score, clipping, NaN handling)
- [x] GMM clustering with k-sweep
- [x] Bootstrap and temporal stability
- [x] Quality gates Q1 and Q2
- [x] Full vector x timeframe sweep (54 combinations)
- [x] Visualization suite (10 plot types, 55 figures)
- [x] Academic report

### Phase 2: Predictive Quality
- [ ] Q3 implementation (Kruskal-Wallis, eta-squared, self-transition)
- [ ] Forward return profiling per state
- [ ] State duration analysis
- [ ] Entry/exit signature extraction
- [ ] Cross-asset lead/lag analysis

### Phase 3: Feature Enrichment
- [ ] Temporal derivatives (rolling stats, differences, EMA)
- [ ] Spectral decomposition (FFT, spectral entropy)
- [ ] Cross-feature interactions
- [ ] Multi-timeframe alignment signals
- [ ] Re-run sweep with enriched features

### Phase 4: Transition Modeling
- [ ] Empirical transition matrix
- [ ] HMM fitting (Baum-Welch)
- [ ] HMM vs GMM comparison
- [ ] Out-of-sample state prediction
- [ ] Forbidden/asymmetric transition analysis

### Phase 5: Online Detector
- [ ] Model serialization/loading
- [ ] Real-time bar classification
- [ ] Drift detection
- [ ] Model versioning
- [ ] Integration with ingestor pipeline

### Phase 6: Strategy Integration (future)
- [ ] State-conditional return signals
- [ ] Transition-based entry signals
- [ ] State-specific model activation (XGBoost per state)
- [ ] Risk management per regime

---

## 9. File Structure

```
nat/
├── config/
│   └── pipeline.toml              # All parameters and thresholds
├── scripts/
│   └── cluster_pipeline/
│       ├── config.py              # Feature vector definitions
│       ├── loader.py              # Parquet loading and schema validation
│       ├── preprocess.py          # Bar aggregation and preprocessing
│       ├── cluster.py             # GMM, k-sweep, quality metrics, stability
│       ├── enrich.py              # [Phase 3] Feature enrichment pipeline
│       ├── transitions.py         # [Phase 4] Transition matrix and HMM
│       ├── detector.py            # [Phase 5] Online regime detector
│       ├── reduce.py              # PCA, t-SNE dimensionality reduction
│       └── viz.py                 # Visualization functions
├── scripts/
│   ├── visualize_profiling.py     # Full visualization suite
│   └── pipeline_runner.py         # Orchestrator state machine
├── reports/
│   ├── cluster_sweep_results.json # Raw sweep results
│   ├── cluster_analysis_report.md # Academic report
│   └── figures/                   # All generated plots
└── data/
    ├── features/                  # Raw parquet from ingestor
    ├── models/                    # [Phase 5] Serialized GMM/HMM models
    └── pipeline_state.json        # Pipeline state machine
```

---

## 10. References

- Reynolds, D. (2009). Gaussian Mixture Models. Encyclopedia of Biometrics.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. Journal of Classification.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. Journal of Financial Econometrics.
- Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. Review of Financial Studies.

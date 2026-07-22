# System Architecture: Entropy-Based Regime Detection Agent Swarm

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AGENT SWARM ECOSYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   EXCHANGE   │───▶│     ING      │───▶│   ENTROPY    │───▶│   REGIME     │  │
│  │   (Data)     │    │  (Ingestor)  │    │   AGENTS     │    │   ROUTER     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         │                   ▼                   ▼                   ▼           │
│         │            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│         │            │   FEATURE    │    │   REGIME     │    │  STRATEGY    │  │
│         │            │    STORE     │◀──▶│    STORE     │◀──▶│   AGENTS     │  │
│         │            │  (RingBuf)   │    │   (State)    │    │  (MR/TF)     │  │
│         │            └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                                           │           │
│         │                   ┌──────────────────────────────────────┘           │
│         │                   ▼                                                   │
│         │            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│         │            │   ORDER      │───▶│  EXECUTION   │───▶│    P&L       │  │
│         └───────────▶│   MANAGER    │    │   GATEWAY    │    │   TRACKER    │  │
│                      └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                     │           │
│                      ┌──────────────────────────────────────────────┘           │
│                      ▼                                                          │
│               ┌──────────────┐    ┌──────────────┐                              │
│               │   GENETIC    │◀──▶│   GENOTYPE   │                              │
│               │   EVOLVER    │    │    STORE     │                              │
│               └──────────────┘    └──────────────┘                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. ING (Ingestor) - Rust

**Responsibility**: Transform raw market data into feature vectors with deterministic, reproducible, low-latency computation.

**Language**: Rust (mandatory for this component)

**Key Design Principles**:
- **Zero-copy parsing**: Use `zerocopy` or `bytemuck` for network packet parsing
- **Lock-free data structures**: `crossbeam` for SPMC (single-producer-multi-consumer) channels
- **Cache-aligned structures**: `#[repr(align(64))]` for hot data
- **SIMD acceleration**: `packed_simd` or `std::simd` for vectorized computations
- **Memory pools**: Pre-allocated buffers, no runtime allocation in hot path

```rust
// Conceptual structure
pub struct IngConfig {
    pub symbol: Symbol,
    pub feature_set: FeatureSet,
    pub buffer_depth: usize,        // Ring buffer size
    pub publish_interval_ns: u64,   // Feature emission rate
}

pub struct FeatureVector {
    pub timestamp_ns: u64,
    pub sequence_id: u64,
    pub raw: RawFeatures,           // ~50 features
    pub derived: DerivedFeatures,   // ~150 features
    pub entropy: EntropyFeatures,   // ~20 features
}
```

**Real-Time Specifications**:
- Target latency: < 10μs from packet arrival to feature emission
- Jitter budget: < 1μs (99th percentile)
- Use `SCHED_FIFO` with `mlockall()` for deterministic scheduling
- CPU isolation via `isolcpus` kernel parameter
- Disable hyperthreading on critical cores

---

### 2. Feature Taxonomy & Nomenclature

**Naming Convention**: `{CATEGORY}_{SOURCE}_{TRANSFORM}_{HORIZON}_{VARIANT}`

Example: `IMBALANCE_LOB_EMA_1000MS_DEPTH5`

#### Category Prefixes

| Prefix | Meaning | Examples |
|--------|---------|----------|
| `RAW_` | Direct observation | `RAW_BID_P1`, `RAW_ASK_Q1` |
| `IMBALANCE_` | Bid-Ask asymmetry | `IMBALANCE_LOB_VOLUME_DEPTH3` |
| `FLOW_` | Trade flow derived | `FLOW_VWAP_DELTA_5S` |
| `VOL_` | Volatility measures | `VOL_REALIZED_1MIN` |
| `ENT_` | Entropy measures | `ENT_PERMUTATION_64_5` |
| `SPEC_` | Spectral/frequency | `SPEC_PSD_LOWFREQ_RATIO` |
| `KALMAN_` | Kalman filtered | `KALMAN_MIDPRICE_STATE` |
| `CORR_` | Correlation based | `CORR_PRICE_VOLUME_ROLL60` |

#### Source Indicators

| Source | Meaning |
|--------|---------|
| `LOB` | Limit Order Book |
| `TRADE` | Trade/Execution data |
| `QUOTE` | Best bid/ask only |
| `DERIVED` | From other features |

#### Transform Indicators

| Transform | Meaning |
|-----------|---------|
| `RAW` | No transform |
| `LOG` | Log transform |
| `DIFF` | First difference |
| `RET` | Return (log or simple) |
| `EMA` | Exponential moving average |
| `SMA` | Simple moving average |
| `ZSCORE` | Z-score normalized |
| `RANK` | Rank normalized [0,1] |
| `FFT` | Fourier transform |
| `WAVELET` | Wavelet transform |

---

### 3. Complete Feature List

#### 3.1 Raw Order Book Features (~30 features)

```
RAW_LOB_BID_P{1-10}           # Best 10 bid prices
RAW_LOB_ASK_P{1-10}           # Best 10 ask prices
RAW_LOB_BID_Q{1-10}           # Best 10 bid quantities
RAW_LOB_ASK_Q{1-10}           # Best 10 ask quantities
RAW_LOB_MIDPRICE              # (best_bid + best_ask) / 2
RAW_LOB_SPREAD                # best_ask - best_bid
RAW_LOB_MICROPRICE            # Volume-weighted mid
```

#### 3.2 Order Book Imbalance Features (~25 features)

```
IMBALANCE_LOB_VOLUME_D{1,3,5,10}      # Volume imbalance at depth D
IMBALANCE_LOB_NOTIONAL_D{1,3,5,10}    # Notional imbalance at depth D
IMBALANCE_LOB_QUEUE_RATIO             # bid_qty_1 / (bid_qty_1 + ask_qty_1)
IMBALANCE_LOB_DEPTH_WEIGHTED          # Weighted average across depths
IMBALANCE_LOB_PRESSURE_BID            # Cumulative bid pressure
IMBALANCE_LOB_PRESSURE_ASK            # Cumulative ask pressure
IMBALANCE_LOB_SLOPE_BID               # Linear regression slope of bid curve
IMBALANCE_LOB_SLOPE_ASK               # Linear regression slope of ask curve
```

#### 3.3 Trade Flow Features (~30 features)

```
FLOW_TRADE_COUNT_{1S,5S,30S,1M}       # Trade count in window
FLOW_TRADE_VOLUME_{1S,5S,30S,1M}      # Volume in window
FLOW_TRADE_NOTIONAL_{1S,5S,30S,1M}    # Notional in window
FLOW_TRADE_VWAP_{1S,5S,30S,1M}        # VWAP in window
FLOW_TRADE_VWAP_DELTA_{1S,5S}         # VWAP - midprice
FLOW_TRADE_AGGRESSOR_RATIO_{1S,5S}    # Buy aggressor ratio
FLOW_TRADE_SIZE_AVG_{1S,5S}           # Average trade size
FLOW_TRADE_INTENSITY                  # Arrival rate (trades/sec)
FLOW_TRADE_TOXICITY_VPIN              # Volume-sync probability of informed trading
```

#### 3.4 Volatility Features (~20 features)

```
VOL_REALIZED_{1M,5M,15M,1H}           # Realized volatility
VOL_PARKINSON_{1M,5M,15M}             # Parkinson (high-low based)
VOL_GARMAN_KLASS_{1M,5M,15M}          # Garman-Klass estimator
VOL_YANG_ZHANG_{1M,5M,15M}            # Yang-Zhang (overnight gaps)
VOL_BIPOWER_{1M,5M}                   # Bipower variation (jump robust)
VOL_RATIO_{SHORT/LONG}                # Vol regime indicator
VOL_ZSCORE_REALIZED                   # Z-score of current vol
```

#### 3.5 Entropy Features (~25 features)

```
ENT_SHANNON_RETURN_{64,128,256}       # Shannon entropy of discretized returns
ENT_PERMUTATION_{3,4,5,6}             # Permutation entropy with embedding dim
ENT_SAMPLE_{M2_R02,M2_R015}           # Sample entropy (m=2, r=0.2σ etc)
ENT_APPROX_{M2_R02}                   # Approximate entropy
ENT_MULTISCALE_{SCALE1-5}             # Multi-scale entropy
ENT_SPECTRAL                          # Spectral entropy
ENT_TRANSFER_PRICE_VOLUME             # Transfer entropy (information flow)
ENT_CONDITIONAL_{PRICE|VOLUME}        # Conditional entropy
ENT_RATE_CHANGE_{1S,5S}               # Rate of entropy change
```

#### 3.6 Spectral/Frequency Features (~20 features)

```
SPEC_PSD_TOTAL                        # Total power spectral density
SPEC_PSD_BAND_{LOW,MID,HIGH}          # PSD in frequency bands
SPEC_PSD_RATIO_{LOW/HIGH}             # Frequency band ratios
SPEC_DOMINANT_FREQ                    # Dominant frequency
SPEC_DOMINANT_PERIOD                  # Dominant period
SPEC_HURST_EXPONENT                   # Hurst exponent (H)
SPEC_FRACTAL_DIM                      # Fractal dimension
SPEC_WAVELET_ENERGY_{SCALE1-5}        # Wavelet energy by scale
SPEC_COHERENCE_PRICE_VOLUME           # Coherence between series
```

#### 3.7 Kalman Filter Features (~15 features)

```
KALMAN_MIDPRICE_STATE                 # Filtered midprice
KALMAN_MIDPRICE_VELOCITY              # First derivative (drift)
KALMAN_MIDPRICE_ACCEL                 # Second derivative
KALMAN_MIDPRICE_UNCERTAINTY           # State covariance
KALMAN_SPREAD_STATE                   # Filtered spread
KALMAN_IMBALANCE_STATE                # Filtered imbalance
KALMAN_INNOVATION                     # Prediction error (surprise)
KALMAN_INNOVATION_ZSCORE              # Normalized innovation
```

#### 3.8 Cross-Feature Correlations (~15 features)

```
CORR_PRICE_VOLUME_{30S,1M,5M}         # Price-volume correlation
CORR_SPREAD_VOLATILITY_{1M,5M}        # Spread-vol correlation
CORR_IMBALANCE_RETURN_{1S,5S}         # Predictive correlation
CORR_AUTOCORR_RETURN_{LAG1-5}         # Return autocorrelation
CORR_CROSS_ENTROPY_IMBALANCE          # Entropy-imbalance relation
```

#### 3.9 Market Quality Features (~10 features)

```
QUALITY_QUOTED_SPREAD_BPS             # Spread in basis points
QUALITY_EFFECTIVE_SPREAD_BPS          # Effective spread from trades
QUALITY_DEPTH_BPS_{10,50,100}         # Depth within X bps
QUALITY_RESILIENCY                    # LOB recovery rate
QUALITY_KYLE_LAMBDA                   # Price impact coefficient
```

#### 3.10 Regime Indicator Features (~10 features)

```
REGIME_ENTROPY_ZSCORE                 # Standardized entropy level
REGIME_VOL_REGIME                     # High/Med/Low volatility
REGIME_TREND_STRENGTH                 # ADX-like measure
REGIME_MEAN_REVERSION_SCORE           # Ornstein-Uhlenbeck fit
REGIME_HMM_STATE_PROB_{1,2,3}         # Hidden state probabilities
```

---

## Process for Hypervolume Identification

### Step 1: Historical Labeling (Supervised Approach)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LABELING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Historical Data ──▶ Feature Extraction ──▶ Feature Matrix X    │
│                                                                  │
│  For each time window t:                                         │
│    1. Simulate ASMM(θ) → P&L_MR(t)                              │
│    2. Simulate TrendFollow(θ) → P&L_TF(t)                       │
│    3. Assign label:                                              │
│       - y(t) = MR  if P&L_MR(t) > τ                             │
│       - y(t) = TF  if P&L_TF(t) > τ                             │
│       - y(t) = NA  otherwise                                     │
│                                                                  │
│  Result: Labeled dataset (X, y)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 2: Dimensionality Reduction & Visualization

```python
# Conceptual flow
from sklearn.manifold import UMAP
from sklearn.decomposition import PCA

# 1. PCA for linear structure
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# 2. UMAP for nonlinear manifold
umap = UMAP(n_components=3, n_neighbors=50, min_dist=0.1)
X_umap = umap.fit_transform(X_pca)

# 3. Visualize with regime labels
# Look for cluster separation
```

### Step 3: Decision Boundary Learning

**Option A: Support Vector Machines (for interpretable boundaries)**

```python
from sklearn.svm import SVC

# RBF kernel for complex boundaries
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# Decision function gives distance to boundary
distances = svm.decision_function(X_test)
```

**Option B: Gradient Boosted Trees (for feature importance)**

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8
)
xgb.fit(X_train, y_train)

# Feature importance for regime detection
importance = xgb.feature_importances_
```

**Option C: Neural Network (for complex nonlinear boundaries)**

```python
import torch.nn as nn

class RegimeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 3))  # MR, TF, NA
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### Step 4: Hypervolume Characterization

Once boundaries are learned, characterize each regime:

```python
def characterize_hypervolume(X, y, regime_label):
    """
    Extract statistical properties of a regime's hypervolume.
    """
    X_regime = X[y == regime_label]

    return {
        'centroid': X_regime.mean(axis=0),
        'covariance': np.cov(X_regime.T),
        'hull_volume': ConvexHull(X_regime[:, :3]).volume,  # 3D projection
        'density': len(X_regime) / hull_volume,
        'key_features': identify_discriminative_features(X, y, regime_label),
        'boundary_samples': find_boundary_samples(X_regime, svm),
    }
```

### Step 5: Online Regime Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE DETECTION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Live Features x(t) ──▶ Classifier ──▶ P(regime | x(t))         │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │ P(MR) = 0.72    │                          │
│                    │ P(TF) = 0.15    │  ──▶ Select Strategy     │
│                    │ P(NA) = 0.13    │                          │
│                    └─────────────────┘                          │
│                                                                  │
│  Confidence threshold: Only act if max(P) > 0.6                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Software Architecture Components

### Technology Stack

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Ingestor (ING)** | Rust | Latency, determinism, memory safety |
| **Feature Store** | Rust + Arrow/Parquet | Zero-copy IPC, columnar storage |
| **Entropy Computation** | Rust | CPU-intensive, needs SIMD |
| **Regime Classifier** | Python + ONNX Runtime | ML ecosystem, deploy via ONNX |
| **Strategy Engine** | Rust | Latency critical |
| **Genetic Evolver** | Python | Rapid iteration, not latency critical |
| **Backtester** | Python + Rust core | Hybrid: Python orchestration, Rust simulation |
| **Monitoring** | Prometheus + Grafana | Industry standard |
| **Message Bus** | ZeroMQ or shared memory | Low latency IPC |

### Project Structure

```
nat/
├── PAPERS_IDEAS.md
├── ARCHITECTURE.md
├── Cargo.toml                    # Workspace root
├── rust/
│   ├── ing-core/                 # Core ingestor library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser/           # Protocol parsers (FIX, binary, etc.)
│   │       ├── features/         # Feature computations
│   │       │   ├── raw.rs
│   │       │   ├── imbalance.rs
│   │       │   ├── flow.rs
│   │       │   ├── volatility.rs
│   │       │   ├── entropy.rs
│   │       │   ├── spectral.rs
│   │       │   └── kalman.rs
│   │       ├── buffer/           # Ring buffers, memory pools
│   │       └── metrics/          # Latency histograms, counters
│   │
│   ├── ing-bin/                  # Ingestor binary
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   │
│   ├── strategy-core/            # Strategy implementations
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── asmm.rs           # Avellaneda-Stoikov MM
│   │       ├── trend.rs          # Trend following
│   │       └── portfolio.rs      # Position management
│   │
│   └── backtester/               # High-perf backtesting engine
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── simulator.rs
│           └── matching.rs
│
├── python/
│   ├── pyproject.toml
│   ├── nat/
│   │   ├── __init__.py
│   │   ├── regime/               # Regime detection ML
│   │   │   ├── __init__.py
│   │   │   ├── labeler.py        # Historical labeling
│   │   │   ├── classifier.py     # Boundary learning
│   │   │   └── online.py         # Live inference
│   │   │
│   │   ├── genetic/              # Evolutionary components
│   │   │   ├── __init__.py
│   │   │   ├── genotype.py       # Genotype encoding
│   │   │   ├── population.py     # Population management
│   │   │   ├── fitness.py        # Fitness evaluation
│   │   │   ├── selection.py      # Tournament, elitism
│   │   │   ├── crossover.py      # Recombination operators
│   │   │   └── mutation.py       # Mutation operators
│   │   │
│   │   ├── agents/               # Agent definitions
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Abstract agent
│   │   │   ├── regime_agent.py   # Regime detection agent
│   │   │   ├── mr_agent.py       # Mean reversion agent
│   │   │   └── tf_agent.py       # Trend following agent
│   │   │
│   │   └── analysis/             # Research & analysis
│   │       ├── __init__.py
│   │       ├── hypervolume.py    # Hypervolume analysis
│   │       └── visualization.py  # Plotting utilities
│   │
│   └── tests/
│
├── config/
│   ├── ing.toml                  # Ingestor config
│   ├── strategies.toml           # Strategy parameters
│   └── genetic.toml              # GA parameters
│
├── data/
│   ├── raw/                      # Raw market data
│   ├── features/                 # Computed features (Parquet)
│   └── models/                   # Trained models (ONNX)
│
├── scripts/
│   ├── train_regime.py           # Training script
│   ├── evolve.py                 # Genetic evolution script
│   └── backtest.py               # Backtesting script
│
└── infra/
    ├── docker/
    │   ├── Dockerfile.ing
    │   └── Dockerfile.python
    ├── k8s/                       # Kubernetes manifests
    └── terraform/                 # Infrastructure as code
```

### Genotype Structure

```python
@dataclass
class AgentGenotype:
    """
    Encodes all evolvable parameters of an agent.
    """
    # Feature selection (binary mask)
    feature_mask: np.ndarray  # shape: (n_features,), dtype: bool

    # Entropy computation
    entropy_method: str  # 'permutation', 'sample', 'multiscale'
    entropy_params: dict  # method-specific parameters

    # Regime thresholds
    regime_thresholds: dict  # {'mr_low': 0.3, 'tf_high': 0.7, ...}

    # Strategy parameters
    asmm_params: AvellanedaStoikovParams  # gamma, sigma, T, k
    trend_params: TrendFollowParams       # lookback, threshold, etc.

    # Meta-parameters
    confidence_threshold: float  # min P(regime) to act
    position_sizing: str         # 'fixed', 'kelly', 'volatility_scaled'

    def encode(self) -> np.ndarray:
        """Flatten to continuous vector for GA operations."""
        pass

    @classmethod
    def decode(cls, vector: np.ndarray) -> 'AgentGenotype':
        """Reconstruct from continuous vector."""
        pass

    def mutate(self, mutation_rate: float) -> 'AgentGenotype':
        """Apply Gaussian mutation to continuous params, bit flip to discrete."""
        pass

    def crossover(self, other: 'AgentGenotype') -> Tuple['AgentGenotype', 'AgentGenotype']:
        """Two-point crossover producing two offspring."""
        pass
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml (conceptual)
name: CI Pipeline

on: [push, pull_request]

jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Run tests
        run: cargo test --workspace
      - name: Run benchmarks
        run: cargo bench --workspace
      - name: Check latency regression
        run: ./scripts/check_latency.sh  # Fail if p99 > threshold

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ./python[dev]
      - name: Run tests
        run: pytest python/tests -v
      - name: Run type checking
        run: mypy python/nat

  integration-tests:
    needs: [rust-tests, python-tests]
    runs-on: ubuntu-latest
    steps:
      - name: Run integration tests
        run: ./scripts/integration_test.sh
      - name: Backtest validation
        run: python scripts/backtest.py --validate
```

---

## Development Methodology

### Phase 1: Foundation (Weeks 1-4)
1. Set up Rust workspace with `ing-core`
2. Implement basic feature extraction (raw LOB, trade flow)
3. Create Python skeleton with Parquet I/O
4. Establish CI pipeline

### Phase 2: Feature Engineering (Weeks 5-8)
1. Implement all entropy measures in Rust
2. Implement spectral features (FFT, Hurst)
3. Implement Kalman filter state estimation
4. Benchmark and optimize latency

### Phase 3: Regime Detection (Weeks 9-12)
1. Build historical labeling pipeline
2. Train initial classifier (XGBoost baseline)
3. Visualize hypervolumes with UMAP
4. Iterate on feature importance

### Phase 4: Strategy Integration (Weeks 13-16)
1. Implement ASMM in Rust
2. Implement trend-following baseline
3. Build backtesting framework
4. Validate regime-conditioned performance

### Phase 5: Genetic Evolution (Weeks 17-20)
1. Implement genotype encoding
2. Build population management
3. Implement fitness evaluation (parallel backtests)
4. Run initial evolution experiments

### Phase 6: Production Hardening (Weeks 21-24)
1. Paper trading deployment
2. Monitoring and alerting
3. Risk management integration
4. Documentation and handoff

---

## Key Design Decisions

### 1. Rust for Latency-Critical Paths
- Ingestor, feature computation, strategy execution in Rust
- Python for ML, analysis, orchestration
- ONNX Runtime bridges ML models to Rust inference

### 2. Shared Memory IPC
- Use `shared_memory` crate for zero-copy feature distribution
- Arrow format for columnar feature vectors
- Avoids serialization overhead

### 3. Deterministic Replay
- All randomness seeded and logged
- Feature extraction is pure function of input
- Enables exact replay for debugging

### 4. Multi-Objective Fitness
- Don't just optimize Sharpe; include:
  - Sortino ratio (downside risk)
  - Max drawdown
  - Win rate
  - Profit factor
- Use NSGA-II for Pareto frontier

### 5. Incremental Complexity
- Start with simple features, add complexity as needed
- Baseline with logistic regression before neural nets
- Validate each component independently

---

## Novel Extensions (Differentiators)

See `EXTENSIONS.md` for full implementation details.

### Extension Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXTENDED ARCHITECTURE WITH NOVEL COMPONENTS                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      LAYER 1: DATA INGESTION (Rust)                        │ │
│  │                                                                             │ │
│  │  Hyperliquid WS ──▶ Parser ──▶ Base Features (57) ──▶ Parquet             │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                         │
│                                       ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 2: NOVEL EXTENSIONS (Python)                      │ │
│  │                                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │ │
│  │  │     PCE     │  │  Transfer   │  │   Info      │  │ Unsupervised│       │ │
│  │  │   Engine    │  │  Entropy    │  │  Geometry   │  │  Clustering │       │ │
│  │  │             │  │  Network    │  │             │  │             │       │ │
│  │  │ • Sobol idx │  │ • Causal    │  │ • Fisher    │  │ • Geodesic  │       │ │
│  │  │ • UQ        │  │   graph     │  │   manifold  │  │   distance  │       │ │
│  │  │ • Interact. │  │ • Topology  │  │ • Curvature │  │ • HDBSCAN   │       │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │ │
│  │         │                │                │                │               │ │
│  │         └────────────────┼────────────────┼────────────────┘               │ │
│  │                          ▼                ▼                                 │ │
│  │                   Extended Features (~90 total)                            │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                         │
│                                       ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 3: REGIME DETECTION                               │ │
│  │                                                                             │ │
│  │   SUPERVISED PATH:                    UNSUPERVISED PATH (NOVEL):           │ │
│  │   ─────────────────                   ──────────────────────────           │ │
│  │   Extended Features                   Extended Features                     │ │
│  │         │                                   │                               │ │
│  │         ▼                                   ▼                               │ │
│  │   ┌───────────┐                      ┌───────────┐                         │ │
│  │   │  XGBoost  │                      │ Cluster on│                         │ │
│  │   │ Classifier│                      │  Fisher   │                         │ │
│  │   └─────┬─────┘                      │ Manifold  │                         │ │
│  │         │                            └─────┬─────┘                         │ │
│  │         ▼                                  │                               │ │
│  │   MR / TF / NA                             ▼                               │ │
│  │                                    Natural Regime Clusters                 │ │
│  │                                    (discovered, not defined)               │ │
│  │                                                                             │ │
│  │   HYBRID (RECOMMENDED):                                                    │ │
│  │   ─────────────────────                                                    │ │
│  │   Cluster assignment + Features ──▶ XGBoost ──▶ Strategy per cluster      │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                         │
│                                       ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 4: STRATEGY EXECUTION                             │ │
│  │                                                                             │ │
│  │   Cluster + Confidence ──▶ Lookup optimal strategy for cluster             │ │
│  │                               │                                             │ │
│  │                               ├──▶ ASMM(θ_cluster)                         │ │
│  │                               ├──▶ TrendFollow(θ_cluster)                  │ │
│  │                               └──▶ Abstain (low confidence)                │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Novel Contributions Summary

| Extension | Novelty Level | Description |
|-----------|---------------|-------------|
| **PCE for Microstructure** | HIGH | Polynomial chaos for feature interaction & UQ |
| **Transfer Entropy Networks** | MEDIUM-HIGH | Causal graph of LOB features |
| **Information Geometry** | HIGH | Fisher manifold for market states |
| **Clustering on Manifold** | **VERY HIGH** | Geodesic-based unsupervised regime discovery |

### Key Insight: Unsupervised > Supervised for Regime Discovery

```
SUPERVISED (Standard Approach):
├── Define regimes by strategy profitability (circular!)
├── Label data → Train classifier → Predict
└── Problem: Regimes are artifacts of your strategy definitions

UNSUPERVISED (Novel Approach):
├── Cluster on Fisher manifold using geodesic distance
├── Discover NATURAL market states
├── THEN ask: "What works in each state?"
└── Advantage: Strategy-agnostic regime definitions

This is a genuine methodological contribution.
```

### Publication Potential

With these extensions, viable venues include:
- **Quantitative Finance** (methodology paper)
- **ICAIF** (ACM AI in Finance conference)
- **NeurIPS AI4Finance Workshop**
- **Journal of Financial Econometrics**
- **Arxiv preprint** (establish priority)

---

## Documents Reference

| Document | Purpose |
|----------|---------|
| `PAPERS_IDEAS.md` | Literature review, 50+ references |
| `HYPER_DOCS.md` | Hyperliquid API documentation |
| `V1_SPEC.md` | V1 implementation specification |
| `EXTENSIONS.md` | Novel extension implementations |
| `ARCHITECTURE.md` | This file - overall system design |

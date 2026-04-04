# Algorithmic Research Direction

**Created:** 2026-04-04
**Status:** Strategic Proposal (No Implementation Yet)
**Priority:** Foundation for next iteration

---

## Executive Summary

NAT has collected **183 microstructure features** across 14 categories from Hyperliquid's perpetual futures market. We now propose a **hypothesis-driven algorithmic research program** focusing on **entropy-gated regime detection** and **whale flow tracking** rather than generic machine learning.

**Key Insight:** Markets alternate between **low-entropy (predictable) regimes** where momentum works and **high-entropy (random walk) regimes** where mean-reversion works. Use entropy as a **gating mechanism** to select the appropriate algorithm.

---

## The Feature Foundation

### 183 Features Across 14 Categories

Complete mathematical specifications available in:
- **README.md** — Lines 206-498 (complete feature reference with formulas)
- **docs/specs/ALGORITHM_DESIGN_PROPOSAL.md** — Detailed algorithmic usage

**Category Summary:**

| Category | Count | Key Insight | Primary Application |
|----------|-------|-------------|---------------------|
| **Entropy** (24) | Tick entropy, permutation entropy | **Regime detection** | Algorithm gating |
| **Trend** (15) | Momentum, Hurst, R² | Persistence measurement | Momentum signals |
| **Illiquidity** (12) | Kyle's λ, Amihud, Roll spread | Price impact | Capacity estimation |
| **Toxicity** (10) | VPIN, adverse selection | Flow quality | Entry timing |
| **Order Flow** (8) | L1/L5/L10 imbalance | Directional pressure | Short-term signals |
| **Volatility** (8) | Realized vol, Parkinson | Risk regime | Position sizing |
| **Concentration** (15) | Gini, HHI, Top-N | Position crowding | Contrarian signals |
| **Whale Flow** (12) | Net flow 1h/4h/24h | Smart money tracking | Trend confirmation |
| **Liquidation** (13) | Cluster detection | Cascade prediction | Tactical opportunities |
| **Microstructure** (10) | Midprice, spread, depth | Bid-ask dynamics | Market making |
| **Trade Flow** (12) | Volume, VWAP, intensity | Execution patterns | Volume confirmation |
| **Market Context** (9) | Funding, OI, basis | External conditions | Regime context |
| **Derived** (15) | Interaction terms | Combined signals | Alpha composites |
| **Regime Detection** (20) | Absorption, divergence, churn | Accumulation/distribution | Strategic positioning |

**Total:** 183 features, providing comprehensive market state representation.

---

## Critical Gap: Time Scale Mismatch

**Problem:** 75% of features are computed on **tick-level microstructure** (1s to 15m windows), but proposed strategies operate on **daily timeframes**.

**Solution (HIGHEST PRIORITY):**

```
Daily Aggregation Pipeline (NOT YET IMPLEMENTED)

Input:  Parquet files with tick-level features (1s intervals)
Output: Daily bars with feature statistics

For each feature:
  - daily_mean
  - daily_std
  - daily_min
  - daily_max
  - daily_close (last value of day)

Special aggregations:
  - whale_net_flow_24h = sum(whale_flow) over 24h
  - regime_accumulation_score = f(absorption, divergence, whale_flow)
  - entropy_mean_15m = mean(normalized_entropy_15m) over day
```

**Blocking:** All algorithmic work is blocked until daily aggregation is implemented.

**Estimated Effort:** 4-6 hours (Task 1 from TASKS_29_3_26.md)

---

## Proposed Algorithmic Approach

### Philosophy: Hypothesis-Driven, Not ML-First

**Traditional Approach (Avoid):**
```
Throw all 183 features into random forest
→ Overfit
→ No interpretability
→ Fails OOS
```

**NAT's Approach (Proposed):**
```
1. Form testable hypothesis (e.g., "low entropy + whale inflow → bullish")
2. Select minimal feature set (5-8 features)
3. Design interpretable algorithm (logistic regression, thresholds)
4. Validate rigorously (walk-forward, OOS/IS > 0.7)
5. Deploy incrementally (paper → small → scale)
```

### The 8 Algorithms

Detailed specifications in **docs/specs/ALGORITHM_DESIGN_PROPOSAL.md**

#### 1. Entropy-Gated Strategy Switcher (PRIMARY)

**Core Hypothesis:** Market predictability varies with entropy.

```
if normalized_entropy < 0.3:
    → LOW ENTROPY regime (predictable, trending)
    → Use MOMENTUM CONTINUATION CLASSIFIER

elif normalized_entropy > 0.7:
    → HIGH ENTROPY regime (random walk, noisy)
    → Use MEAN-REVERSION DETECTOR

else:
    → UNCERTAIN regime
    → NO TRADE
```

**Why This Matters:** Forces regime-appropriate algorithms instead of forcing one model across all conditions.

#### 2. Momentum Continuation Classifier

**Hypothesis:** In low-entropy regimes, momentum + whale alignment predicts continuation.

**Model:** Logistic Regression (interpretable)

**Features (ranked):**
1. `momentum_300 × (1 - H_norm)` — momentum in low entropy
2. `whale_net_flow_4h × sign(momentum_300)` — whale confirmation
3. `hurst_exponent` — persistence measure
4. `trend_strength = √(momentum² × R²)` — combined signal
5. `vpin_10` — toxicity (lower is better)

**Signal:**
```
P(continuation) = sigmoid(β₀ + Σ βᵢ × featureᵢ)

if P > 0.6 AND H_norm < 0.3:
    → GO LONG (momentum continuation expected)
```

**Target:** Sharpe > 0.5, precision > 55%, OOS/IS > 0.7

#### 3. Mean-Reversion / False-Breakout Detector

**Hypothesis:** In high-entropy regimes, extreme moves + liquidation clusters → reversion.

**Model:** LightGBM (capture non-linear interactions)

**Features:**
1. `mean_reversion_score = (P - MA) / σ` — z-score
2. `breakout_indicator = (P - min) / (max - min)` — range position
3. `liq_asymmetry` — liquidation risk skew
4. `concentration_momentum = HHI × momentum` — crowding
5. `whale_flow_momentum` (opposite sign check)

**Signal:**
```
if |z-score| > 2 AND P(reversion) > 0.65 AND H_norm > 0.7:
    → FADE THE MOVE (mean reversion trade)
```

#### 4. Meta-Labeling System

**Hypothesis:** Predict **if a simple signal will work**, not just direction.

```
Step 1: Generate base signal (MA crossover, whale flow alignment)
Step 2: Train classifier: Will this signal be profitable?
Step 3: Only take trades where P(signal_success) > threshold
```

**Advantage:** Combines simplicity (base signal) with ML precision filtering.

**Target:** Precision lift > 15% vs base signal alone.

#### 5. Regime State Machine

**Hypothesis:** Markets transition between discrete regimes with different optimal strategies.

**States (6 regimes):**
- **ACCUMULATION** — Low entropy, high absorption, whale buying → LONG bias
- **DISTRIBUTION** — Low entropy, high absorption, whale selling → SHORT bias
- **TRENDING_UP** — Low entropy, momentum > 0, whale aligned → Momentum long
- **TRENDING_DOWN** — Low entropy, momentum < 0, whale aligned → Momentum short
- **RANGING** — High entropy, low trend strength → Mean reversion
- **VOLATILE_NOISE** — High entropy, high toxicity → NO TRADE

**Implementation:** HMM or manual thresholds.

#### 6. Market-Making Skew Model

**Hypothesis:** Whale flow + liquidation asymmetry predict short-term pressure.

```
skew = α₁×whale_flow + α₂×imbalance + α₃×liq_asymmetry - α₄×vpin

bid = microprice × (1 - spread/2 - skew)
ask = microprice × (1 + spread/2 - skew)
```

**Use Case:** Provide liquidity while exploiting informed flow edge.

#### 7. Online Anomaly / Change-Point Detector

**Hypothesis:** Regime transitions detectable in real-time via entropy + whale flow changes.

**Algorithm:** CUSUM or Bayesian change-point detection

**Signals:**
- Entropy drop (0.7 → 0.3) → trending regime starting
- Whale flow surge (> 2σ) → accumulation/distribution phase
- Liquidation cluster formation → cascade risk building

#### 8. Nearest-Neighbor State Retrieval

**Hypothesis:** Current market state has historical analogs.

```
State Vector: [entropy, momentum, whale_flow, vpin, hurst, regime_score, vol]

Find K=20 nearest neighbors from history
expected_return = mean(neighbor_outcomes)
win_rate = count(neighbor_wins) / K

if expected_return > threshold AND win_rate > 0.6:
    → TRADE
```

**Advantage:** Non-parametric, adapts to new regimes automatically.

---

## NautilusTrader Integration

### Why NautilusTrader?

**Production-grade backtesting and live trading framework** (Rust/Cython + Python API).

**Key Features:**
- Realistic order matching (limit orders, slippage, latency)
- Strategy isolation (each algorithm = separate actor)
- Risk management (position limits, drawdown controls)
- Same code for backtest and production

### Architecture

```
NAT Feature Engine → Daily Aggregation → FeatureBar Objects → NautilusTrader Strategies

FeatureBar:
  - Standard OHLCV
  - Feature statistics (entropy_mean, whale_flow_sum, momentum_close, etc.)

Strategies:
  - EntropyGatedMomentumStrategy
  - MeanReversionStrategy
  - MetaLabelingStrategy
  - RegimeSwitchingStrategy
```

### Workflow

```bash
# 1. Collect tick data (done by NAT)
make run

# 2. Daily aggregation (TO BE IMPLEMENTED)
python scripts/aggregate_daily.py --input ./data/features/ --output ./data/daily_bars/

# 3. Backtest with NautilusTrader
python backtest_runner.py \
    --strategy EntropyGatedMomentumStrategy \
    --data ./data/daily_bars/ \
    --start 2024-01-01 --end 2024-12-31

# 4. Analyze results
python analyze_backtest.py --results ./backtests/latest/
```

**Validation Criteria:**
- Walk-forward Sharpe > 0.5
- OOS/IS ratio > 0.7
- Win rate > 52%
- Max drawdown < 20%

**If criteria met → Paper trading for 30 days → Live deployment at 10% capital**

---

## Liquidity Heatmap with Regime Detection

### Purpose

Visualize **where liquidity clusters** on the order book and **detect accumulation/distribution phases**.

### Design

```
Liquidity Heatmap:
  - Y-axis: Price levels
  - X-axis: Cumulative depth
  - Color: Blue (bids), Red (asks), Purple (liquidation clusters)

Regime Overlay:
  - Current regime (ACCUMULATION, DISTRIBUTION, TRENDING, RANGING, NOISE)
  - Confidence score [0, 1]
  - Feature values (absorption, divergence, whale_flow, range_position)
  - Interpretation and bias
```

### Regime Classification Logic

```python
def classify_regime(features):
    absorption = features['regime_absorption_zscore']
    divergence = features['regime_divergence_zscore']
    whale_flow = features['whale_net_flow_24h']
    range_pos = features['regime_range_pos_24h']
    entropy = features['normalized_entropy_15m']

    # ACCUMULATION: high absorption + negative divergence + whale buying + low range pos
    if (absorption > 1.5 and divergence < -1.0 and whale_flow > 0 and range_pos < 0.3):
        return "ACCUMULATION", confidence_score(features)

    # DISTRIBUTION: high absorption + positive divergence + whale selling + high range pos
    elif (absorption > 1.5 and divergence > 1.0 and whale_flow < 0 and range_pos > 0.7):
        return "DISTRIBUTION", confidence_score(features)

    # TRENDING: low entropy + high momentum + whale alignment
    elif (entropy < 0.3 and abs(momentum) > 0.01 and whale_aligned):
        return "TRENDING_" + direction, confidence_score(features)

    # RANGING: high entropy + low trend + high churn
    elif (entropy > 0.6 and trend_strength < 0.2 and churn > 1.0):
        return "RANGING", confidence_score(features)

    else:
        return "NOISE", 0.5
```

### Validation

**Backtest regime classifier on historical data:**
- ACCUMULATION → forward 7d return > 0? (target accuracy: >65%)
- DISTRIBUTION → forward 7d return < 0? (target accuracy: >65%)
- High confidence subset (>0.7): target accuracy >70%

**Implementation:** Python + FastAPI + React + D3.js (WebSocket updates)

---

## Research Lab Iteration Structure

### Continuous Alpha Search Pipeline

```
Phase 1: DATA COLLECTION
  ├─ NAT Ingestor → tick-level features
  ├─ Daily Aggregator → daily bars
  └─ Dataset Snapshots → versioned datasets

Phase 2: HYPOTHESIS GENERATION
  ├─ Feature analysis (correlation, MI, clustering)
  ├─ Regime classification
  ├─ Pattern discovery
  └─ Testable predictions

Phase 3: ALGORITHM DESIGN
  └─ Select from 8 algorithm catalog

Phase 4: MODEL TRAINING
  ├─ Load dataset snapshot
  ├─ Train model (sklearn, LightGBM)
  ├─ Hyperparameter tuning
  └─ Save model + metadata

Phase 5: WALK-FORWARD VALIDATION
  ├─ 5-fold expanding window (60d train, 15d validate)
  ├─ Generate OOS predictions
  ├─ Compute metrics (Sharpe, win rate, OOS/IS)
  └─ Regime-specific analysis

Phase 6: BACKTEST
  └─ NautilusTrader realistic execution simulation

Phase 7: DECISION GATE
  ├─ GO: Sharpe > 0.5, OOS/IS > 0.7, win rate > 52%, drawdown < 20%
  ├─ PIVOT: Sharpe 0.3-0.5 (weak signal, refine)
  └─ NO-GO: Sharpe < 0.3, OOS/IS < 0.5 (discard)

Phase 8: PAPER TRADING (if GO)
  └─ 30 days on testnet/paper account

Phase 9: LIVE DEPLOYMENT (if paper successful)
  └─ Start at 10% capital, scale if Sharpe > 0.5

Phase 10: CONTINUOUS MONITORING
  ├─ Daily performance tracking
  ├─ Regime drift detection
  ├─ Model retraining (weekly or on shift)
  └─ Graceful degradation
```

### Online Traceability

**Experiment Tracking:** Every hypothesis, algorithm, dataset, parameter, metric tracked in database.

```python
class ResearchExperiment:
    exp_id: str
    hypothesis: str
    algorithm: str
    dataset_snapshot: str
    features_used: List[str]
    model_params: dict
    training_metrics: dict
    validation_metrics: dict
    backtest_results: dict
    decision: str  # GO / PIVOT / NO-GO
    notes: str
    timestamp: datetime
```

**Dashboard:** View all experiments, compare, find best by metric, filter by date/algorithm/performance.

---

## Implementation Roadmap

### Phase 0: Foundation (Week 1) — HIGHEST PRIORITY

**Task 1: Daily Aggregation Pipeline (4-6h)**
- Aggregate tick-level features to daily bars
- Save to Parquet with NautilusTrader-compatible schema
- **BLOCKS ALL DOWNSTREAM WORK**

**Task 2: Regime Classification Validator (4-6h)**
- Backtest regime classifier on historical data
- Compute accuracy for ACCUMULATION/DISTRIBUTION
- Tune thresholds

### Phase 1: Algorithm Prototyping (Week 2-3)

- Entropy-gated strategy switcher
- Momentum continuation classifier
- Mean-reversion detector

### Phase 2: NautilusTrader Integration (Week 3-4)

- Custom data adapter
- Strategy implementations
- Backtesting harness

### Phase 3: Advanced Algorithms (Week 5-6)

- Meta-labeling system
- Regime state machine

### Phase 4: Visualization & Monitoring (Week 7-8)

- Liquidity heatmap
- Research dashboard

### Phase 5: Paper Trading (Week 9-12)

- Deploy best algorithm
- A/B testing framework

---

## Success Metrics

### Algorithm Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Walk-Forward Sharpe | > 0.5 | > 0.8 |
| OOS/IS Ratio | > 0.7 | > 0.85 |
| Win Rate | > 52% | > 55% |
| Max Drawdown | < 20% | < 15% |
| Profit Factor | > 1.3 | > 1.5 |

### Regime Classification Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Accumulation Accuracy | > 65% | > 70% |
| Distribution Accuracy | > 65% | > 70% |
| High Confidence Accuracy (>0.7) | > 70% | > 75% |

---

## Critical Path

```
Daily Aggregation (Week 1)
    ↓
Regime Classification Validation (Week 1)
    ↓
Entropy-Gated Switcher + Momentum Classifier (Week 2)
    ↓
NautilusTrader Integration (Week 3-4)
    ↓
Walk-Forward Validation (Week 4)
    ↓
Decision: GO / PIVOT / NO-GO
    ↓
If GO → Paper Trading (Week 9+)
```

**Blocking Task:** Daily aggregation pipeline (Task 1, TASKS_29_3_26.md, 4-6 hours)

---

## Key Documents

1. **README.md** (lines 206-498) — Complete feature reference with mathematical formulas
2. **docs/specs/ALGORITHM_DESIGN_PROPOSAL.md** — Detailed algorithmic specifications (this document expanded)
3. **docs/refined/unimplemented_specs_priorities.md** — Prioritized implementation sequence
4. **docs/specs/TASKS_29_3_26.md** — Week 1 critical tasks (includes daily aggregation)

---

## Philosophy

**Build Less, Validate More:**
- Start with simplest interpretable models (logistic regression, manual thresholds)
- Validate rigorously (walk-forward, regime-specific, OOS/IS > 0.7)
- Deploy incrementally (paper → small → scale)
- Track everything (experiment metadata, reproducibility)

**Hypothesis-Driven, Not Algorithm-Driven:**
- Form testable prediction first
- Select minimal feature set
- Design interpretable algorithm
- Validate with statistical rigor
- Only deploy if GO criteria met

**Entropy as Gating Mechanism:**
- Don't force one model across all regimes
- Low entropy → momentum algorithms
- High entropy → mean-reversion algorithms
- Uncertain entropy → no trade

---

## Conclusion

NAT has the **infrastructure** (183 features, hypothesis testing, ML pipeline) and the **data** (tick-level microstructure from Hyperliquid).

**What's Missing:**
1. **Daily aggregation** (CRITICAL, blocks everything, 4-6h effort)
2. **Algorithmic implementations** (8 algorithms specified, 2-6 weeks)
3. **NautilusTrader integration** (realistic backtesting, 1-2 weeks)
4. **Validation** (walk-forward, regime-specific, 1-2 weeks)

**What Happens Next:**
1. Implement daily aggregation (Week 1)
2. Validate regime classification (Week 1)
3. Prototype entropy-gated momentum classifier (Week 2)
4. Backtest with NautilusTrader (Week 3-4)
5. Decision: GO / PIVOT / NO-GO based on Sharpe > 0.5
6. If GO → Paper trading for 30 days
7. If successful → Live deployment at 10% capital

**Timeline to First Live Algorithm:** 9-12 weeks (assuming validation succeeds)

---

**Created:** 2026-04-04
**Next Review:** After Phase 0 completion (daily aggregation + regime validation)
**Status:** Specification complete, awaiting implementation approval

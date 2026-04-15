# Algorithm Design Proposal

**Status:** Specification (Not Implemented)
**Priority:** P0 (Critical for next iteration)
**Created:** 2026-04-04
**Purpose:** Speculate and design algorithms for hypothesis-driven alpha extraction from 183 microstructure features

---

## Executive Summary

This document proposes **8 algorithmic approaches** for extracting trading signals from NAT's 183 microstructure features. Rather than generic machine learning, we focus on **hypothesis-driven, interpretable models** that exploit specific market regimes and behavioral patterns.

**Key Principle:** Use entropy and regime features to **gate** which algorithm runs, rather than forcing a single model across all market conditions.

---

## Feature Foundation Analysis

### 14 Feature Categories Overview

NAT extracts **183 features** across 14 categories, providing comprehensive market state representation:

| Category | Count | Primary Use Case | Signal Type |
|----------|-------|------------------|-------------|
| **Entropy** | 24 | Regime detection, predictability measurement | Gating mechanism |
| **Trend** | 15 | Momentum vs mean-reversion classification | Directional signals |
| **Illiquidity** | 12 | Price impact, informed flow detection | Risk & capacity |
| **Toxicity** | 10 | Adverse selection, order flow quality | Entry timing |
| **Order Flow** | 8 | Short-term directional pressure | Execution signals |
| **Volatility** | 8 | Risk regime classification | Position sizing |
| **Concentration** | 15 | Crowding detection, whale dominance | Contrarian signals |
| **Whale Flow** | 12 | Smart money tracking | Trend confirmation |
| **Liquidation** | 13 | Cascade prediction | Tactical opportunities |
| **Microstructure** | 10 | Bid-ask dynamics, depth | Market making |
| **Trade Flow** | 12 | Execution patterns | Volume confirmation |
| **Market Context** | 9 | Funding, OI, basis | Regime context |
| **Derived** | 15 | Interaction terms, composites | Combined alpha |
| **Regime Detection** | 20 | Accumulation/distribution phases | Strategic positioning |

**Total:** 183 features, but 75% are **tick-level** (1s to 15m windows).

### Critical Insight: Time Scale Mismatch

**Problem:** Most features are computed on tick-level microstructure (1s-15m windows), but our proposed strategies operate on **daily timeframes**.

**Solution Required:** Daily aggregation pipeline to compute:
- Daily statistics (mean, std, min, max) of each feature
- Daily regime classification
- Daily whale flow accumulation
- Daily liquidation risk evolution

---

## Proposed Algorithms

### 1. Entropy-Gated Strategy Switcher

**Hypothesis:** Market predictability varies with entropy. Low entropy = trend continuation, high entropy = mean reversion.

**Mathematical Framework:**

```
Let H_norm = normalized_entropy ∈ [0, 1]

Strategy Selection:
├─ If H_norm < 0.3  → MOMENTUM_CLASSIFIER  (low entropy, high predictability)
├─ If H_norm > 0.7  → MEAN_REVERSION_DETECTOR  (high entropy, random walk)
└─ If 0.3 ≤ H_norm ≤ 0.7 → NO_TRADE  (uncertain regime)
```

**Features Used:**
- Primary: `normalized_entropy_1m`, `normalized_entropy_15m`, `entropy_rate`
- Secondary: `conditional_entropy`, `permutation_entropy`

**Implementation Approach:**
```
1. Compute daily entropy statistics (mean, std, trend)
2. Classify day into low/high/mixed entropy regime
3. Route to appropriate sub-algorithm
4. Track regime transition probabilities
```

**Target Prediction:** Binary regime classification → algorithm routing

**Why Not ML?** Interpretable thresholds allow manual override and regime-specific risk management.

---

### 2. Momentum Continuation Classifier

**Hypothesis:** In low-entropy regimes, momentum persists. Features like `whale_flow`, `trend_strength`, `hurst_exponent` predict continuation.

**Mathematical Framework:**

```
Target: y = sign(r_{t+1}) where r is next-period return

Features (ranked by expected importance):
1. momentum_300 × (1 - H_norm)  [momentum in low entropy]
2. whale_net_flow_4h × sign(momentum_300)  [whale confirmation]
3. hurst_exponent  [persistence measure]
4. trend_strength = √(momentum² × R²)  [combined signal]
5. r_squared_300  [trend linearity]
6. monotonicity_300  [directional consistency]
7. vpin_10  [toxicity - lower is better]
8. regime_accumulation_score  [accumulation phase]

Model: Logistic Regression or Linear SVM (interpretable)
Alternative: LightGBM (if interactions matter)
```

**Probability Calibration:**
```
P(continuation) = sigmoid(β₀ + Σ βᵢ × featureᵢ)

Trade Signal:
├─ If P > 0.6 and H_norm < 0.3  → GO LONG (continuation expected)
├─ If P < 0.4 and H_norm < 0.3  → GO SHORT (reversal expected)
└─ Otherwise → NO TRADE
```

**Walk-Forward Validation:**
- Train on 60 days, validate on 15 days (4:1 ratio)
- Retrain weekly
- Require OOS/IS performance ratio > 0.7

**Success Criteria:**
- Precision > 55% (better than random)
- Sharpe ratio > 0.5 (walk-forward OOS)
- Stable performance across volatility regimes

---

### 3. Mean-Reversion / False-Breakout Detector

**Hypothesis:** In high-entropy regimes, price excursions from MA are likely to revert. High `liq_risk` + high `concentration` + range extremes → reversal.

**Mathematical Framework:**

```
Target: y = sign(P_{t+Δ} - P_t) where Δ is reversion horizon (e.g., 4h)

Features:
1. mean_reversion_score = (P - MA) / σ  [z-score from MA]
2. breakout_indicator = (P - min) / (max - min)  [range position]
3. liq_risk_above_2pct / liq_risk_below_2pct  [asymmetry]
4. concentration_momentum = HHI × momentum  [crowding]
5. whale_flow_momentum (opposite sign check)  [whale divergence]
6. vpin_50  [toxicity suggests false move]
7. regime_distribution_score  [distribution phase]
8. realized_vol_5m  [high vol → noise]

Model: Gradient Boosted Trees (LightGBM) to capture non-linear interactions
```

**Signal Generation:**
```
P(reversion | overextension) = model(features)

Trade Signal:
├─ If |z-score| > 2 AND P(reversion) > 0.65 AND H_norm > 0.7
│   → FADE THE MOVE (mean reversion trade)
├─ If breakout_indicator > 0.9 AND liq_asymmetry > 0.5
│   → FADE (liquidity cluster, likely false breakout)
└─ Otherwise → NO TRADE
```

**Risk Management:**
- Stop loss at 1.5× recent ATR
- Target at z-score = 0 (return to mean)
- Position size inversely proportional to `realized_vol`

---

### 4. Meta-Labeling System

**Hypothesis:** Instead of predicting direction, predict **if a simple signal will work**. Improves precision by filtering bad setups.

**Framework:**

```
Step 1: Generate base signal (simple rule)
Example: BUY if MA_fast > MA_slow AND whale_net_flow_1h > 0

Step 2: Meta-label = Did this signal work? (binary classifier)
y_meta = 1 if signal led to profit, 0 otherwise

Step 3: Train classifier to predict y_meta from market state
Features:
1. entropy features (is regime predictable?)
2. toxicity features (is signal genuine?)
3. concentration features (is move crowded?)
4. volatility features (is risk acceptable?)

Model: Logistic Regression (interpretable feature importance)
```

**Implementation:**
```
Base Signals (simple, interpretable):
├─ MA crossover
├─ Whale flow alignment
├─ Liquidation cascade setup
└─ Funding rate extreme

Meta-Classifier Output:
P(signal_success | market_state) > threshold → TAKE TRADE
Otherwise → SKIP

Advantage: Combines simplicity (base signal) with ML precision filtering
```

**Evaluation:**
- Precision lift vs base signal alone (target: >15% lift)
- False positive reduction
- Out-of-sample validation

---

### 5. Regime State Machine

**Hypothesis:** Markets transition between discrete regimes. Explicit state modeling improves signal conditioning.

**State Space Design:**

```
States (6 regimes):
├─ ACCUMULATION (low entropy, high absorption, whale inflow)
├─ DISTRIBUTION (low entropy, high absorption, whale outflow)
├─ TRENDING_UP (low entropy, momentum > 0, whale alignment)
├─ TRENDING_DOWN (low entropy, momentum < 0, whale alignment)
├─ RANGING (high entropy, low trend_strength, high churn)
└─ VOLATILE_NOISE (high entropy, high vol, high toxicity)

Transition Model: Hidden Markov Model (HMM) or manual thresholds
```

**Feature-to-State Mapping:**

| State | Key Features | Thresholds |
|-------|-------------|------------|
| **ACCUMULATION** | `regime_accumulation_score > 0.7`, `whale_net_flow_24h > 0`, `regime_absorption_zscore > 1` | High absorption without price rise |
| **DISTRIBUTION** | `regime_distribution_score > 0.7`, `whale_net_flow_24h < 0`, `regime_absorption_zscore > 1` | High absorption without price fall |
| **TRENDING_UP** | `momentum_300 > 0`, `trend_strength > 0.5`, `H_norm < 0.3`, `whale_net_flow_4h > 0` | Persistent directional move |
| **TRENDING_DOWN** | `momentum_300 < 0`, `trend_strength > 0.5`, `H_norm < 0.3`, `whale_net_flow_4h < 0` | Persistent directional move |
| **RANGING** | `H_norm > 0.6`, `trend_strength < 0.2`, `regime_churn_4h > 1.5` | Two-sided flow, no direction |
| **VOLATILE_NOISE** | `realized_vol_5m > σ_threshold`, `vpin_50 > 0.8`, `H_norm > 0.7` | High noise, uninformative |

**Trading Rules per State:**

```
ACCUMULATION:
  → LONG bias (whale accumulation)
  → Wait for entropy drop (confirmation)
  → Target: distribution phase reversal

DISTRIBUTION:
  → SHORT bias (whale distribution)
  → Wait for entropy drop (confirmation)
  → Target: accumulation phase reversal

TRENDING_UP:
  → Momentum continuation trades
  → Use Momentum Classifier (Algorithm #2)
  → Fade only on extreme overextension

TRENDING_DOWN:
  → Momentum continuation trades (short)
  → Use Momentum Classifier (Algorithm #2)
  → Fade only on extreme oversold

RANGING:
  → Mean reversion trades (Algorithm #3)
  → Fade extremes, target midpoint
  → Reduce position size (low Sharpe regime)

VOLATILE_NOISE:
  → NO TRADE
  → Wait for regime transition
  → Monitor entropy for drop
```

**Implementation:**
- Compute regime scores daily
- Track state transitions (accumulation → trending → distribution → ranging)
- Use transition probabilities for timing

---

### 6. Market-Making Skew Model

**Hypothesis:** Whale flow and liquidation risk predict short-term directional pressure. Skew limit orders toward pressure direction.

**Framework:**

```
Goal: Quote bid/ask with skew based on predicted short-term imbalance

Fair Value: microprice = (bid×Q_ask + ask×Q_bid) / (Q_bid + Q_ask)

Skew Predictors:
1. whale_net_flow_1h (normalized)
2. imbalance_qty_l10 (order book imbalance)
3. liq_asymmetry (liquidation risk skew)
4. flow_imbalance (recent trade flow)
5. vpin_10 (toxicity - avoid toxic flow)

Skew Model:
skew = α₁×whale_flow + α₂×imbalance + α₃×liq_asymmetry + α₄×flow_imbalance - α₅×vpin

Quote Placement:
bid = microprice × (1 - spread/2 - skew)
ask = microprice × (1 + spread/2 - skew)

If skew > 0 (bullish pressure):
  → Tighter ask, wider bid (lean long)
If skew < 0 (bearish pressure):
  → Tighter bid, wider ask (lean short)
```

**Risk Management:**
- Maximum inventory limits
- Skew only within liquidity depth
- Stop quoting if `vpin > threshold` (toxic flow)

**Use Case:** Provide liquidity while exploiting informed flow edge

---

### 7. Online Anomaly / Change-Point Detector

**Hypothesis:** Regime transitions (accumulation → trending) are detectable in real-time via entropy + whale flow changes.

**Algorithms:**

```
Option A: CUSUM (Cumulative Sum Control Chart)

Sₜ = max(0, Sₜ₋₁ + (xₜ - μ) - k)  [detect upward shift]
If Sₜ > h → ALERT (regime change detected)

Apply to:
- entropy_rate (detect entropy drop → trending regime)
- whale_net_flow_1h (detect accumulation start)
- regime_absorption_zscore (detect absorption regime)

Option B: Bayesian Change-Point Detection

P(change_point | data) via Bayesian online inference
Update posteriors incrementally
Flag when P(change) > threshold
```

**Implementation:**
```
Monitor signals:
1. Entropy drop (H_norm: 0.7 → 0.3) → trending regime starting
2. Whale flow surge (flow > 2σ) → accumulation/distribution phase
3. Liquidation cluster formation → cascade risk building

Action:
- Entropy drop + whale inflow → PREPARE LONG
- Entropy drop + whale outflow → PREPARE SHORT
- Maintain state until reversal signal
```

**Advantage:** Real-time regime transition detection, not just static classification

---

### 8. Nearest-Neighbor State Retrieval

**Hypothesis:** Current market state has historical analogs. Find similar states and predict outcome based on historical behavior.

**Framework:**

```
State Vector: [entropy_norm, momentum_300, whale_flow_4h, vpin_50,
               hurst_exponent, regime_accumulation_score, realized_vol_5m]

Distance Metric: Weighted Euclidean or Mahalanobis distance

Algorithm:
1. Store historical state vectors with outcomes (forward returns)
2. For current state, find K nearest neighbors (e.g., K=20)
3. Aggregate neighbor outcomes:
   - Mean return
   - Win rate
   - Sharpe ratio
4. If neighbor performance > threshold → TRADE

Advantage: Non-parametric, adapts to new regimes automatically
```

**Feature Weighting:**
- High weight: entropy, regime scores (define regime)
- Medium weight: trend features (momentum, hurst)
- Low weight: volatility (too noisy)

**Decision Rule:**
```
neighbors = find_k_nearest(current_state, K=20)
expected_return = mean(neighbor_outcomes)
win_rate = count(neighbor_outcomes > 0) / K

If expected_return > threshold AND win_rate > 0.6:
  → TRADE in direction of expected_return
```

---

## NautilusTrader Integration

### Why NautilusTrader?

NautilusTrader is a **production-grade backtesting and live trading framework** built in Rust/Cython with Python API. It's designed for **low-latency execution** and **realistic backtesting** with proper order matching, slippage, and latency simulation.

**Key Advantages:**
1. **Realistic Backtesting:** Order-by-order matching, no look-ahead bias
2. **Strategy Isolation:** Each algorithm = separate strategy actor
3. **Risk Management:** Built-in position limits, drawdown controls, margin checks
4. **Execution Simulation:** Limit orders, market orders, stop-loss, take-profit
5. **Performance Analytics:** Sharpe, Sortino, drawdown, turnover, slippage analysis
6. **Live Trading Ready:** Same strategy code runs in backtest and production

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       NAUTILUS INTEGRATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NAT Feature Engine (Rust)          NautilusTrader (Rust/Python)│
│  ┌──────────────────┐                ┌─────────────────────┐    │
│  │  WebSocket Feed  │───────────────>│  Data Adapter       │    │
│  │  (Hyperliquid)   │   Tick/Bar     │  (Custom)           │    │
│  └──────────────────┘   Stream       └─────────────────────┘    │
│           │                                     │                │
│           ▼                                     ▼                │
│  ┌──────────────────┐                ┌─────────────────────┐    │
│  │ Feature Computer │                │  Strategy Actors    │    │
│  │ (183 features)   │───────────────>│  • EntropyGated     │    │
│  └──────────────────┘   Feature      │  • MomentumCont.    │    │
│           │              Vector       │  • MeanReversion    │    │
│           │                           │  • MetaLabeling     │    │
│           ▼                           │  • RegimeSwitch     │    │
│  ┌──────────────────┐                └─────────────────────┘    │
│  │ Daily Aggregator │                         │                 │
│  │ (rollup to bars) │                         ▼                 │
│  └──────────────────┘                ┌─────────────────────┐    │
│           │                           │  Execution Engine   │    │
│           └──────────────────────────>│  • Order matching   │    │
│                   Daily bars +        │  • Slippage model   │    │
│                   feature vector      │  • Position mgmt    │    │
│                                       └─────────────────────┘    │
│                                                │                 │
│                                                ▼                 │
│                                       ┌─────────────────────┐    │
│                                       │  Risk Manager       │    │
│                                       │  • Max position     │    │
│                                       │  • Drawdown stop    │    │
│                                       │  • Correlation      │    │
│                                       └─────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Adapter Design

**Challenge:** NAT produces 183 tick-level features, NautilusTrader expects OHLCV bars.

**Solution:** Custom data adapter that:
1. Ingests NAT Parquet files (tick-level features)
2. Aggregates to daily bars with feature statistics
3. Emits custom `FeatureBar` objects to strategies

**FeatureBar Schema:**
```python
class FeatureBar:
    timestamp: int64
    symbol: str

    # Standard OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Feature statistics (daily aggregates)
    entropy_mean: float
    entropy_std: float
    momentum_300_close: float
    whale_flow_4h_sum: float
    vpin_50_mean: float
    regime_accumulation_score: float
    regime_distribution_score: float
    liq_asymmetry_mean: float
    hurst_exponent: float
    trend_strength_close: float

    # ... (select subset of 183 features, aggregated daily)
```

### Strategy Implementation Example

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder

class EntropyGatedMomentumStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.entropy_threshold = 0.3
        self.momentum_threshold = 0.005
        self.position_size = 100  # contracts

    def on_bar(self, bar: FeatureBar):
        """Called on each daily bar with features"""

        # Entropy gate
        if bar.entropy_mean > self.entropy_threshold:
            return  # High entropy, skip

        # Momentum signal
        if bar.momentum_300_close > self.momentum_threshold:
            # Whale confirmation
            if bar.whale_flow_4h_sum > 0:
                # Low toxicity
                if bar.vpin_50_mean < 0.5:
                    # GO LONG
                    self.buy(bar.symbol, self.position_size)

        elif bar.momentum_300_close < -self.momentum_threshold:
            if bar.whale_flow_4h_sum < 0:
                if bar.vpin_50_mean < 0.5:
                    # GO SHORT
                    self.sell(bar.symbol, self.position_size)

        # Exit conditions
        if self.portfolio.is_long(bar.symbol):
            if bar.entropy_mean > 0.7:  # Regime change
                self.close_position(bar.symbol)

        if self.portfolio.is_short(bar.symbol):
            if bar.entropy_mean > 0.7:
                self.close_position(bar.symbol)
```

### Backtesting Workflow

```bash
# 1. NAT collects tick data → Parquet files (done)
make run

# 2. Daily aggregation (TO BE IMPLEMENTED)
python scripts/aggregate_daily.py \
    --input ./data/features/*.parquet \
    --output ./data/daily_bars/

# 3. Run NautilusTrader backtest
python backtest_runner.py \
    --strategy EntropyGatedMomentumStrategy \
    --data ./data/daily_bars/ \
    --start 2024-01-01 \
    --end 2024-12-31

# 4. Analyze results
python analyze_backtest.py --results ./backtests/latest/
```

**Output Metrics:**
- Sharpe ratio, Sortino ratio, max drawdown
- Win rate, profit factor, average trade P&L
- Turnover, slippage estimate, capacity estimate
- Regime-specific performance (low vs high entropy)

### Live Trading Deployment

Once validated in backtest:

```python
# Same strategy code, different execution mode
from nautilus_trader.live.node import TradingNode

node = TradingNode()
node.add_data_client(HyperliquidDataClient())  # Custom adapter
node.add_exec_client(HyperliquidExecClient())  # Order execution
node.add_strategy(EntropyGatedMomentumStrategy())
node.run()  # Live trading
```

**Risk Controls:**
- Max position per symbol
- Max total portfolio exposure
- Daily loss limit (stop trading)
- Correlation checks (avoid correlated positions)

---

## Liquidity Heatmap with Regime Detection

### Purpose

Visualize **where liquidity is concentrated** on the order book and **detect accumulation/distribution phases** based on whale behavior and volume absorption.

### Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIQUIDITY HEATMAP DISPLAY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Price Levels (Y-axis)         Liquidity Depth (X-axis)         │
│                                                                  │
│  $45,200 ████████████████████░░░░░░  <- Resistance cluster      │
│  $45,150 ████████░░░░░░░░░░░░░░░░░░                             │
│  $45,100 ██████░░░░░░░░░░░░░░░░░░░░                             │
│  $45,050 ████░░░░░░░░░░░░░░░░░░░░░░  <- Current price           │
│  $45,000 ██████████████░░░░░░░░░░░░  <- Support cluster         │
│  $44,950 ████████░░░░░░░░░░░░░░░░░░                             │
│  $44,900 ████████████████████████░░  <- Strong support          │
│                                                                  │
│  Color Coding:                                                   │
│  ■ Blue = Bid liquidity   ■ Red = Ask liquidity                 │
│  ■ Purple = Liquidation clusters                                │
│                                                                  │
│  Regime Overlay:                                                 │
│  ┌────────────────────────────────────────────────────┐          │
│  │ Regime: ACCUMULATION (confidence: 0.82)            │          │
│  │ Absorption: HIGH (z-score: +1.8)                   │          │
│  │ Divergence: NEGATIVE (-0.15)                       │          │
│  │ Whale Flow (24h): +$12.5M (net buying)             │          │
│  │ Range Position: 0.23 (near bottom)                 │          │
│  │                                                     │          │
│  │ Interpretation:                                     │          │
│  │ → Large volume absorbed without price rise          │          │
│  │ → Whales accumulating at support levels            │          │
│  │ → Low range position = near bottom                 │          │
│  │ → BIAS: BULLISH (expect upside breakout)           │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Inputs

**From NAT Features:**
1. `depth_bid_l1` to `depth_bid_l10` — Bid side order book depth
2. `depth_ask_l1` to `depth_ask_l10` — Ask side order book depth
3. `liq_risk_above_*` and `liq_risk_below_*` — Liquidation clusters
4. `regime_absorption_*` — Volume absorption metrics
5. `regime_divergence_*` — Price-volume divergence
6. `whale_net_flow_*` — Whale activity
7. `regime_range_pos_*` — Price position in range

**Aggregation:**
- Compute 1-minute snapshots of order book depth
- Track depth evolution over 24h
- Identify persistent clusters (depth > 2σ for > 1 hour)

### Regime Classification Logic

```python
def classify_regime(features: dict) -> str:
    """
    Classify regime based on NAT features

    Returns: ACCUMULATION, DISTRIBUTION, TRENDING, RANGING, NOISE
    """
    absorption = features['regime_absorption_zscore']
    divergence = features['regime_divergence_zscore']
    whale_flow = features['whale_net_flow_24h']
    range_pos = features['regime_range_pos_24h']
    churn = features['regime_churn_zscore']
    entropy = features['normalized_entropy_15m']

    # ACCUMULATION: high absorption + negative divergence + whale buying + low range pos
    if (absorption > 1.5 and divergence < -1.0 and
        whale_flow > 0 and range_pos < 0.3):
        return "ACCUMULATION", compute_confidence(features)

    # DISTRIBUTION: high absorption + positive divergence + whale selling + high range pos
    elif (absorption > 1.5 and divergence > 1.0 and
          whale_flow < 0 and range_pos > 0.7):
        return "DISTRIBUTION", compute_confidence(features)

    # TRENDING: low entropy + high momentum + whale alignment
    elif (entropy < 0.3 and abs(features['momentum_300']) > 0.01 and
          np.sign(whale_flow) == np.sign(features['momentum_300'])):
        direction = "UP" if whale_flow > 0 else "DOWN"
        return f"TRENDING_{direction}", compute_confidence(features)

    # RANGING: high entropy + low trend + high churn
    elif entropy > 0.6 and features['trend_strength'] < 0.2 and churn > 1.0:
        return "RANGING", compute_confidence(features)

    # NOISE: high volatility + high toxicity
    else:
        return "NOISE", 0.5


def compute_confidence(features: dict) -> float:
    """
    Confidence = how clearly features align with regime

    Returns: confidence score [0, 1]
    """
    # Check feature alignment
    score = 0.0

    # Absorption clarity
    if abs(features['regime_absorption_zscore']) > 2.0:
        score += 0.2

    # Divergence clarity
    if abs(features['regime_divergence_zscore']) > 1.5:
        score += 0.2

    # Whale flow magnitude
    if abs(features['whale_net_flow_24h']) > features['whale_flow_std'] * 2:
        score += 0.2

    # Range position extreme
    if features['regime_range_pos_24h'] < 0.2 or features['regime_range_pos_24h'] > 0.8:
        score += 0.2

    # Entropy clarity
    if features['normalized_entropy_15m'] < 0.25 or features['normalized_entropy_15m'] > 0.75:
        score += 0.2

    return min(score, 1.0)
```

### Visualization Implementation

**Technology Stack:**
- Backend: Python (FastAPI or Flask)
- Frontend: React + D3.js or Plotly Dash
- Real-time: WebSocket updates (1s interval)

**Heatmap Generation:**
```python
import plotly.graph_objects as go

def generate_liquidity_heatmap(order_book_snapshot, liquidation_data):
    """
    Generate 2D heatmap of liquidity distribution

    X-axis: Cumulative depth (normalized)
    Y-axis: Price levels
    Color: Bid (blue), Ask (red), Liquidation (purple)
    """

    # Extract price levels and depths
    bid_prices = [level['price'] for level in order_book_snapshot['bids']]
    bid_depths = [level['quantity'] for level in order_book_snapshot['bids']]

    ask_prices = [level['price'] for level in order_book_snapshot['asks']]
    ask_depths = [level['quantity'] for level in order_book_snapshot['asks']]

    # Liquidation clusters
    liq_prices = liquidation_data['cluster_prices']
    liq_sizes = liquidation_data['cluster_sizes']

    # Create figure
    fig = go.Figure()

    # Bid side (blue bars, left)
    fig.add_trace(go.Bar(
        y=bid_prices,
        x=[-d for d in bid_depths],  # negative for left side
        orientation='h',
        marker=dict(color='blue'),
        name='Bids'
    ))

    # Ask side (red bars, right)
    fig.add_trace(go.Bar(
        y=ask_prices,
        x=ask_depths,
        orientation='h',
        marker=dict(color='red'),
        name='Asks'
    ))

    # Liquidation clusters (purple dots)
    fig.add_trace(go.Scatter(
        y=liq_prices,
        x=[0] * len(liq_prices),  # at center
        mode='markers',
        marker=dict(
            color='purple',
            size=[min(s/1000, 50) for s in liq_sizes],  # scale by cluster size
            opacity=0.6
        ),
        name='Liquidations'
    ))

    # Current price (horizontal line)
    current_price = order_book_snapshot['midprice']
    fig.add_hline(y=current_price, line_dash="dash", line_color="green")

    # Layout
    fig.update_layout(
        title="Liquidity Heatmap",
        xaxis_title="Depth (contracts)",
        yaxis_title="Price",
        barmode='overlay',
        height=800
    )

    return fig
```

### Testing Regime Classification Accuracy

**Validation Approach:**

```python
def backtest_regime_classifier(historical_data):
    """
    Test regime classification accuracy on historical data

    Methodology:
    1. Classify each day's regime based on features
    2. Measure subsequent price behavior
    3. Check if behavior matches regime prediction

    Success Criteria:
    - ACCUMULATION → subsequent 7d return > 0 (accuracy > 65%)
    - DISTRIBUTION → subsequent 7d return < 0 (accuracy > 65%)
    - TRENDING_UP → continuation accuracy > 60%
    - TRENDING_DOWN → continuation accuracy > 60%
    """
    results = []

    for t in range(len(historical_data) - 7):
        # Current features
        features = historical_data[t]
        regime, confidence = classify_regime(features)

        # Forward return (7 days)
        fwd_return = (historical_data[t+7]['close'] -
                      historical_data[t]['close']) / historical_data[t]['close']

        # Check prediction accuracy
        if regime == "ACCUMULATION":
            correct = fwd_return > 0
        elif regime == "DISTRIBUTION":
            correct = fwd_return < 0
        elif regime == "TRENDING_UP":
            correct = fwd_return > 0
        elif regime == "TRENDING_DOWN":
            correct = fwd_return < 0
        else:
            correct = None  # RANGING/NOISE

        results.append({
            'regime': regime,
            'confidence': confidence,
            'fwd_return': fwd_return,
            'correct': correct
        })

    # Compute accuracy by regime
    df = pd.DataFrame(results)
    accuracy_by_regime = df.groupby('regime')['correct'].mean()

    print("Regime Classification Accuracy:")
    print(accuracy_by_regime)

    # High confidence subset
    high_conf = df[df['confidence'] > 0.7]
    print("\nHigh Confidence Accuracy:")
    print(high_conf.groupby('regime')['correct'].mean())

    return df
```

**Expected Results:**
- Baseline (random): 50% accuracy
- Target: >65% accuracy for ACCUMULATION/DISTRIBUTION
- High confidence (>0.7) subset: >70% accuracy

---

## Research Lab Iteration Structure

### From Feature Extraction → Continuous Alpha Search

**Proposed Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONTINUOUS RESEARCH PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: DATA COLLECTION                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ NAT Ingestor → Parquet (tick-level, 183 features)     │     │
│  │ Daily Aggregator → Daily bars with feature stats      │     │
│  │ Dataset Snapshots → Versioned, immutable datasets     │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 2: HYPOTHESIS GENERATION                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ 1. Feature Analysis (correlation, MI, clustering)      │     │
│  │ 2. Regime Classification (accumulation/distribution)   │     │
│  │ 3. Pattern Discovery (anomalies, state transitions)    │     │
│  │ 4. Hypothesis Formulation (testable predictions)       │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 3: ALGORITHM DESIGN                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Select algorithm from catalog:                         │     │
│  │ • Entropy-gated strategy switcher                      │     │
│  │ • Momentum continuation classifier                     │     │
│  │ • Mean-reversion detector                              │     │
│  │ • Meta-labeling system                                 │     │
│  │ • Regime state machine                                 │     │
│  │ • Market-making skew model                             │     │
│  │ • Anomaly detector                                     │     │
│  │ • Nearest-neighbor retrieval                           │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 4: MODEL TRAINING                                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ 1. Load dataset snapshot (versioned)                   │     │
│  │ 2. Train model (sklearn, LightGBM, custom)            │     │
│  │ 3. Hyperparameter tuning (grid search / Bayesian opt) │     │
│  │ 4. Save model + metadata (MLflow or custom)           │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 5: WALK-FORWARD VALIDATION                                │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ 1. Split data: train (60d) → validate (15d) × 5 folds │     │
│  │ 2. Retrain model each fold (expanding window)         │     │
│  │ 3. Generate OOS predictions                            │     │
│  │ 4. Compute metrics: Sharpe, win rate, OOS/IS ratio    │     │
│  │ 5. Regime-specific analysis (low vs high entropy)     │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 6: BACKTEST                                               │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ NautilusTrader backtesting engine:                     │     │
│  │ • Realistic order matching (limit orders, slippage)    │     │
│  │ • Risk management (position limits, stops)            │     │
│  │ • Transaction costs (fees, spread)                    │     │
│  │ • Regime-conditional performance                      │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 7: DECISION GATE                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Criteria for GO/PIVOT/NO-GO:                           │     │
│  │                                                         │     │
│  │ GO (deploy to paper trading):                          │     │
│  │ ✓ Walk-forward Sharpe > 0.5                            │     │
│  │ ✓ OOS/IS ratio > 0.7                                   │     │
│  │ ✓ Win rate > 52%                                       │     │
│  │ ✓ Max drawdown < 20%                                   │     │
│  │ ✓ Regime-stable (works in low & high entropy)         │     │
│  │                                                         │     │
│  │ PIVOT (adjust hypothesis):                             │     │
│  │ ✓ Sharpe 0.3-0.5 (weak signal)                        │     │
│  │ ✓ Regime-specific alpha (works only in low entropy)   │     │
│  │ → Refine feature set, adjust thresholds               │     │
│  │                                                         │     │
│  │ NO-GO (discard):                                       │     │
│  │ ✗ Sharpe < 0.3                                        │     │
│  │ ✗ OOS/IS < 0.5 (overfit)                              │     │
│  │ ✗ Inconsistent across regimes                         │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 8: PAPER TRADING                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Deploy to Hyperliquid testnet or paper account:       │     │
│  │ • Live execution with simulated capital               │     │
│  │ • Real-time monitoring (dashboard)                    │     │
│  │ • Slippage & latency measurement                      │     │
│  │ • Duration: 30 days minimum                            │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 9: LIVE DEPLOYMENT (if paper trading successful)         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ • Start with 10% of target capital                     │     │
│  │ • Monitor for 14 days                                  │     │
│  │ • Scale up if Sharpe > 0.5 and drawdown < 10%         │     │
│  └────────────────────────────────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  Phase 10: CONTINUOUS MONITORING & EVOLUTION                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ • Daily performance tracking (Sharpe, PnL, drawdown)   │     │
│  │ • Regime drift detection (entropy distribution change) │     │
│  │ • Model retraining (weekly or on regime shift)        │     │
│  │ • A/B testing (compare new vs old model)              │     │
│  │ • Graceful degradation (reduce size if Sharpe drops)  │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Online Traceability System

**Purpose:** Track every decision, every experiment, every parameter change for full reproducibility.

**Implementation:** Extend NAT's existing experiment tracking with:

```python
class ResearchExperiment:
    exp_id: str  # unique ID
    hypothesis: str  # what are we testing?
    algorithm: str  # which algorithm?
    dataset_snapshot: str  # versioned dataset
    features_used: List[str]  # subset of 183 features
    model_params: dict  # hyperparameters
    training_metrics: dict  # in-sample performance
    validation_metrics: dict  # walk-forward OOS
    backtest_results: dict  # NautilusTrader output
    decision: str  # GO / PIVOT / NO-GO
    notes: str  # manual observations
    timestamp: datetime

# Example usage
exp = ResearchExperiment(
    exp_id="exp_20260404_001",
    hypothesis="Low entropy + whale inflow predicts bullish continuation",
    algorithm="EntropyGatedMomentumClassifier",
    dataset_snapshot="snapshot_2024_01_01_to_2024_12_31",
    features_used=[
        "normalized_entropy_15m",
        "whale_net_flow_4h",
        "momentum_300",
        "vpin_50",
        "trend_strength"
    ],
    model_params={
        "model_type": "LogisticRegression",
        "C": 0.1,
        "entropy_threshold": 0.3,
        "momentum_threshold": 0.005
    },
    training_metrics={
        "precision": 0.58,
        "recall": 0.62,
        "f1": 0.60,
        "roc_auc": 0.64
    },
    validation_metrics={
        "walk_forward_sharpe": 0.52,
        "oos_is_ratio": 0.73,
        "win_rate": 0.55,
        "max_drawdown": 0.18
    },
    backtest_results={
        "total_return": 0.23,
        "sharpe_ratio": 0.48,
        "sortino_ratio": 0.72,
        "calmar_ratio": 2.67,
        "win_rate": 0.54,
        "profit_factor": 1.42,
        "max_drawdown": 0.14,
        "avg_trade_pnl": 0.0023,
        "num_trades": 87,
        "avg_holding_period_hours": 36
    },
    decision="GO",  # Meets criteria
    notes="Strong performance in low-entropy regimes. Reduce position size in high entropy.",
    timestamp=datetime.now()
)

# Save to experiment tracking database
save_experiment(exp)
```

**Dashboard View:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXPERIMENT TRACKING DASHBOARD                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Recent Experiments:                                             │
│                                                                  │
│  ID                  | Hypothesis              | Sharpe | Status │
│  ───────────────────────────────────────────────────────────────│
│  exp_20260404_001    | Entropy+Whale momentum  | 0.52   | GO     │
│  exp_20260403_002    | Mean reversion extreme  | 0.38   | PIVOT  │
│  exp_20260402_001    | Liquidation cascade     | 0.12   | NO-GO  │
│  exp_20260401_003    | Meta-labeling MA cross  | 0.61   | GO     │
│                                                                  │
│  Best Performing (by Sharpe):                                    │
│  1. exp_20260401_003 — Meta-labeling MA cross (0.61)           │
│  2. exp_20260404_001 — Entropy+Whale momentum (0.52)           │
│  3. exp_20260320_005 — Regime state machine (0.48)             │
│                                                                  │
│  Filter by:                                                      │
│  [ ] Algorithm type   [ ] Date range   [ ] Performance metric   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Continuous Alpha Search Loop

**Automation:**

```bash
# Cron job: daily alpha search
# Run new experiments automatically, testing variations

python research_loop.py \
    --dataset-snapshot snapshot_latest \
    --algorithms entropy_gated,momentum_cont,mean_reversion \
    --param-grid configs/param_grid.yaml \
    --decision-criteria configs/go_nogo_criteria.yaml \
    --output-dir experiments/

# Output:
# - experiments/exp_YYYYMMDD_NNN/
#   - model.pkl
#   - metadata.json
#   - backtest_results.json
#   - decision.txt (GO/PIVOT/NO-GO)
```

**Decision Criteria (YAML):**
```yaml
# configs/go_nogo_criteria.yaml

GO:
  walk_forward_sharpe:
    min: 0.5
  oos_is_ratio:
    min: 0.7
  win_rate:
    min: 0.52
  max_drawdown:
    max: 0.20
  num_trades:
    min: 20  # enough samples

PIVOT:
  walk_forward_sharpe:
    min: 0.3
    max: 0.5
  # Weak signal, adjust and retry

NO_GO:
  walk_forward_sharpe:
    max: 0.3
  # Discard hypothesis
```

---

## Implementation Roadmap

### Phase 0: Foundation (Week 1)
- [ ] **Daily aggregation pipeline** (CRITICAL, blocks everything)
  - Aggregate tick-level features to daily bars
  - Compute daily statistics (mean, std, min, max, close)
  - Save to Parquet with schema compatible with NautilusTrader
- [ ] **Regime classification validator**
  - Backtest regime classifier on historical data
  - Compute accuracy for ACCUMULATION/DISTRIBUTION detection
  - Tune thresholds for confidence scoring

### Phase 1: Algorithm Prototyping (Week 2-3)
- [ ] **Entropy-gated strategy switcher**
  - Implement entropy threshold logic
  - Route to momentum vs mean-reversion algorithms
- [ ] **Momentum continuation classifier**
  - Train logistic regression on low-entropy periods
  - Walk-forward validation
  - Feature importance analysis
- [ ] **Mean-reversion detector**
  - Train LightGBM for false-breakout detection
  - Test on high-entropy periods
  - Optimize z-score thresholds

### Phase 2: NautilusTrader Integration (Week 3-4)
- [ ] **Custom data adapter**
  - Ingest NAT Parquet files
  - Emit FeatureBar objects with daily aggregates
- [ ] **Strategy implementations**
  - Port algorithms to NautilusTrader strategy format
  - Add risk management (position limits, stops)
- [ ] **Backtesting harness**
  - Run walk-forward backtests with realistic execution
  - Generate performance reports

### Phase 3: Advanced Algorithms (Week 5-6)
- [ ] **Meta-labeling system**
  - Generate base signals (MA crossover, whale flow)
  - Train meta-classifier to filter signals
  - Measure precision lift
- [ ] **Regime state machine**
  - Implement HMM or manual threshold-based states
  - Define trading rules per state
  - Track state transition probabilities

### Phase 4: Visualization & Monitoring (Week 7-8)
- [ ] **Liquidity heatmap**
  - Real-time order book depth visualization
  - Liquidation cluster overlay
  - Regime detection display
- [ ] **Research dashboard**
  - Experiment tracking UI
  - Performance comparison charts
  - Feature importance visualizations

### Phase 5: Paper Trading (Week 9-12)
- [ ] **Deploy best algorithm to paper trading**
  - Hyperliquid testnet or simulated account
  - Real-time execution monitoring
  - Slippage measurement
- [ ] **A/B testing framework**
  - Compare multiple algorithms simultaneously
  - Track relative performance
  - Automatic kill switch if Sharpe < 0.3

---

## Success Metrics

### Algorithm Performance

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Walk-Forward Sharpe** | > 0.5 | > 0.8 |
| **OOS/IS Ratio** | > 0.7 | > 0.85 |
| **Win Rate** | > 52% | > 55% |
| **Max Drawdown** | < 20% | < 15% |
| **Profit Factor** | > 1.3 | > 1.5 |

### Regime Classification

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Accumulation Accuracy** | > 65% | > 70% |
| **Distribution Accuracy** | > 65% | > 70% |
| **High Confidence Accuracy** | > 70% | > 75% |

### System Performance

| Metric | Target |
|--------|--------|
| **Daily aggregation latency** | < 10 seconds |
| **Model inference latency** | < 100ms |
| **Backtest speed** | < 5 min per model |
| **Experiment tracking overhead** | < 1% of compute time |

---

## Risk Mitigation

### Overfitting Prevention
- Bonferroni correction for multiple hypothesis testing
- Walk-forward validation (no look-ahead bias)
- OOS/IS ratio > 0.7 requirement
- Regime-specific validation (low vs high entropy)

### Model Degradation Detection
- Daily Sharpe ratio monitoring
- Automatic position reduction if Sharpe < 0.3 for 7 days
- Entropy distribution drift detection
- Feature correlation matrix stability checks

### Execution Risk
- Start with 10% of target capital
- Position limits per symbol
- Daily loss limits (kill switch)
- Slippage tracking and modeling

---

## Open Questions

1. **Daily vs Intraday Timeframes:**
   - Current proposal: daily bars
   - Alternative: 4h or 1h bars for faster iteration
   - Trade-off: signal strength vs opportunity frequency

2. **Feature Selection:**
   - Use all 183 features or select subset?
   - Manual selection vs automatic (PCA, feature importance)
   - Risk: overfitting with too many features

3. **Regime Transition Timing:**
   - How to handle regime boundaries (accumulation → trending)?
   - Wait for confirmation or trade the transition?
   - Risk: late entry vs false signal

4. **Model Retraining Frequency:**
   - Weekly? Monthly? On regime shift?
   - Trade-off: adaptation vs stability

5. **Multi-Symbol Portfolio:**
   - Single-symbol strategies or portfolio approach?
   - Correlation management across BTC, ETH, SOL
   - Risk: correlation breakdown during market stress

---

## Appendix: Feature Usage Matrix

### Which Features for Which Algorithm?

| Algorithm | Primary Features | Secondary Features |
|-----------|------------------|-------------------|
| **Entropy-Gated Switcher** | `normalized_entropy_*`, `entropy_rate` | `conditional_entropy` |
| **Momentum Continuation** | `momentum_*`, `whale_net_flow_*`, `hurst_exponent`, `trend_strength` | `r_squared_*`, `monotonicity_*`, `vpin_*` |
| **Mean-Reversion** | `mean_reversion_score`, `breakout_indicator`, `liq_asymmetry`, `concentration_momentum` | `whale_flow_momentum`, `vpin_50`, `regime_distribution_score` |
| **Meta-Labeling** | `entropy_*`, `toxicity_*`, `concentration_*`, `volatility_*` | All features (market state) |
| **Regime State Machine** | `regime_accumulation_score`, `regime_distribution_score`, `whale_net_flow_24h`, `regime_absorption_zscore` | `regime_churn_*`, `regime_range_pos_*` |
| **Market-Making Skew** | `whale_net_flow_1h`, `imbalance_qty_l10`, `liq_asymmetry`, `flow_imbalance` | `vpin_10`, `microprice` |
| **Anomaly Detector** | `entropy_rate`, `whale_net_flow_*`, `regime_absorption_zscore` | `realized_vol_*`, `liq_intensity` |
| **Nearest-Neighbor** | `normalized_entropy_*`, `momentum_300`, `whale_flow_4h`, `vpin_50`, `hurst_exponent` | `regime_accumulation_score`, `realized_vol_5m` |

---

## Conclusion

This document proposes **8 hypothesis-driven algorithms** designed to exploit specific patterns in NAT's 183 microstructure features:

1. **Entropy-Gated Switcher** — Route to regime-appropriate algorithm
2. **Momentum Continuation** — Exploit low-entropy trend persistence
3. **Mean-Reversion** — Fade high-entropy overextensions
4. **Meta-Labeling** — Filter simple signals with ML
5. **Regime State Machine** — Explicit accumulation/distribution detection
6. **Market-Making Skew** — Lean into whale flow pressure
7. **Anomaly Detector** — Catch regime transitions early
8. **Nearest-Neighbor** — Non-parametric pattern matching

**Next Steps:**
1. Implement **daily aggregation pipeline** (CRITICAL, Week 1)
2. Validate **regime classification accuracy** (Week 1)
3. Prototype **entropy-gated switcher** + **momentum classifier** (Week 2)
4. Integrate with **NautilusTrader** for realistic backtesting (Week 3-4)
5. Deploy best algorithm to **paper trading** (Week 9+)

**Philosophy:** Start simple (logistic regression, manual thresholds), validate rigorously (walk-forward, OOS/IS), deploy incrementally (paper → small live → scale).

---

**Document Version:** 1.0
**Last Updated:** 2026-04-04
**Next Review:** After Phase 0 completion (daily aggregation)

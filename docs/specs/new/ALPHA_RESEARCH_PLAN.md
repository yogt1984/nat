# Alpha Research Plan

## Status Quo

Phase 1 (direct LightGBM on raw features) proved signal exists but is insufficient for profitability:
- +4.18% edge over base rate (real, statistically significant)
- -0.45 bps/trade net of costs (unprofitable)
- Signal sources: funding rate, realized vol, momentum, volume ratio, spread

Phase 2 (hierarchical regime profiling) built the infrastructure to detect market states but has not yet been validated on sufficient live data.

## The Core Problem

A single global model averages over distinct market regimes. Within a trending regime, momentum features are highly predictive. Within a ranging regime, mean-reversion features dominate. Averaging them produces a weak compromise signal.

## Algorithm Groups

### Group A: Regime-Conditioned Models (Priority 1)

**Thesis**: Within-regime predictability >> global predictability. If global edge is 4%, within-regime edge may be 8-15%, which clears the cost threshold.

| ID | Algorithm | Method | Horizon | Data Needed |
|----|-----------|--------|---------|-------------|
| A1 | Regime-filtered trading | Detect regime → map regime to position bias (no per-bar ML) | 1h-4h | 7+ days profiled |
| A2 | Regime-conditioned LightGBM | Separate model per macro regime, trained only on that regime's bars | 5min-1h | 14+ days profiled |
| A3 | HMM with emissions | Hidden Markov Model learns regimes + transition probabilities jointly | 30min-4h | 7+ days |

**Success criterion**: Net Sharpe > 0.5 out-of-sample after costs.

**Dependency**: Requires Phase 2 profiling to produce validated regime labels (Q1+Q2 pass).

### Group B: Microstructure Exploitation (Priority 2)

**Thesis**: 100ms feature granularity enables signals at timescales where competition is lower on Hyperliquid (most competitors use 1s+ data).

| ID | Algorithm | Method | Horizon | Data Needed |
|----|-----------|--------|---------|-------------|
| B1 | Order flow imbalance (OFI) | Multi-level cumulative imbalance predicts next-tick direction (Cont et al. 2014) | 1-30s | 24h tick data |
| B2 | Trade intensity bursts | Hawkes process on trade arrivals. Burst → momentum continuation | 5-60s | 48h tick data |
| B3 | Funding rate mean-reversion | Extreme funding → price reverts over settlement cycle | 1-8h | 30+ days |
| B4 | Spread regime exploitation | Spread widens before large moves. Wide spread + imbalance = entry signal | 10s-5min | 7 days |

**Success criterion**: Positive expectancy per trade after maker fees (1 bps round-trip).

**Dependency**: B1/B2/B4 need tick-level backtesting. B3 needs multi-week data.

### Group C: Cost Engineering (Priority 3)

**Thesis**: Phase 1's 4% edge IS profitable at maker fees (1 bps) instead of taker fees (8 bps). The problem was execution, not signal.

| ID | Algorithm | Method | Horizon | Data Needed |
|----|-----------|--------|---------|-------------|
| C1 | Maker-only limit orders | Place limits at predicted price +/- half-spread. Fill rate analysis. | Any | 7 days |
| C2 | Spread-conditioned entry | Only trade when spread < 0.5 bps (reduces cost by 50%+) | Any | Existing |
| C3 | Position sizing by confidence | Scale position by P(correct) - 0.5. Reduces average cost per unit of edge. | Any | Existing |

**Success criterion**: Effective round-trip cost < 2 bps consistently.

**Dependency**: Requires simulating limit order fills (market replay infrastructure).

### Group D: Ensemble and Adaptation (Priority 4)

**Thesis**: Markets change. Static models decay. Adaptive models maintain edge.

| ID | Algorithm | Method | Horizon | Data Needed |
|----|-----------|--------|---------|-------------|
| D1 | Stacking with regime meta-features | Use regime labels + confidence as features in a second-level model | 5min-1h | 14+ days |
| D2 | Online incremental learning | Retrain on rolling window. Already have OnlineClassifier. | Continuous | Ongoing |
| D3 | Multi-symbol consensus | Trade only when BTC+ETH+SOL regimes agree (cross-symbol consistency) | 1h-4h | 7+ days, 3 symbols |

**Success criterion**: OOS Sharpe degradation < 30% vs IS Sharpe over 30-day rolling windows.

## Execution Timeline

```
Week 1 (NOW):
  - Collect 7+ days of live data (ingestor on su-35)
  - Run profiling pipeline on collected data
  - Validate Q1+Q2 quality gates
  - Implement A1 (regime filter) as simplest test of regime value

Week 2:
  - If A1 shows promise: implement A2 (regime-conditioned LightGBM)
  - Start B3 (funding mean-reversion) — independent, well-understood signal
  - Implement C2 (spread filtering) — free improvement to any strategy

Week 3:
  - If profiling validates: implement D3 (multi-symbol consensus)
  - Begin C1 (maker execution simulation)
  - Backtest A2 walk-forward

Week 4:
  - Consolidate winners into composite strategy
  - Paper trade on live data (forward test)
  - Decision: GO (deploy live) / PIVOT (retry with modifications) / STOP
```

## Decision Gates

| Gate | Condition | Action if FAIL |
|------|-----------|----------------|
| G1: Regime reality | Q1 structural pass (sil > 0.25, ARI > 0.6) | Try different feature subsets, bar sizes |
| G2: Regime predicts | Q2 predictive pass (Kruskal p < 0.05) | Increase data collection, try HMM (A3) |
| G3: Strategy profitable | Net Sharpe > 0.5 OOS | Try cost engineering (Group C), longer horizons |
| G4: Strategy robust | OOS/IS ratio > 0.7, max drawdown < 5% | Reduce position size, add regime filter |
| G5: Live validation | Paper trade matches backtest within 2x | Deploy with minimum size |

## What NOT To Do

- Do not optimize hyperparameters until G1-G2 pass (no signal to optimize)
- Do not build production infrastructure until G3 passes (premature engineering)
- Do not trade real money until G5 passes (paper trade first)
- Do not add more features until existing 123 are validated (complexity without evidence)
- Do not use horizons < 30s until maker execution is proven (cost-dominated otherwise)

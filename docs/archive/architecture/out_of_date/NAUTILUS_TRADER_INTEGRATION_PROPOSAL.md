# NautilusTrader Integration Proposal

**Status:** Specification (Not Implemented)
**Priority:** P0 (Critical Infrastructure)
**Created:** 2026-04-05
**Author:** NAT Development Team
**Purpose:** Define architecture and implementation plan for integrating NAT's 183 microstructure features with NautilusTrader's execution framework

---

## Executive Summary

This document proposes the integration of NAT's proprietary microstructure feature engine with NautilusTrader, a production-grade algorithmic trading framework. The integration aims to combine NAT's **alpha generation capabilities** (183 features including liquidation mapping, whale flow tracking, and entropy-based regime detection) with NautilusTrader's **battle-tested execution infrastructure**.

**Core Thesis:** Standard technical indicators (RSI, MACD, Bollinger Bands) are commoditized and provide no edge. By augmenting these indicators with NAT's microstructure features—particularly liquidity heatmaps, whale flow, and regime detection—we can improve signal quality and filter false positives.

**Primary Use Case:** 15-minute candlestick trading with microstructure-enhanced signals for Hyperliquid perpetual futures.

### Key Deliverables

| Deliverable | Description | Priority |
|-------------|-------------|----------|
| Aggregation Pipeline | Transform tick features to 15m/1h/daily bars | P0 (Blocker) |
| Custom Data Adapter | Feed NAT features into NautilusTrader | P0 |
| Signal Gating Framework | Use NAT features to filter standard indicators | P0 |
| 5 Reference Strategies | Production-ready strategy implementations | P1 |
| Backtesting Harness | Walk-forward validation infrastructure | P1 |
| Live Data Bridge | Real-time feature streaming | P2 |

---

## Table of Contents

1. [Integration Rationale](#1-integration-rationale)
2. [Architecture Options](#2-architecture-options)
3. [Recommended Architecture](#3-recommended-architecture)
4. [Data Model & Feature Mapping](#4-data-model--feature-mapping)
5. [Aggregation Pipeline](#5-aggregation-pipeline)
6. [Strategy Templates](#6-strategy-templates)
7. [Implementation Plan](#7-implementation-plan)
8. [API Specifications](#8-api-specifications)
9. [Backtesting Framework](#9-backtesting-framework)
10. [Risk Management Integration](#10-risk-management-integration)
11. [Success Metrics](#11-success-metrics)
12. [Risk Analysis](#12-risk-analysis)
13. [Appendices](#appendices)

---

## 1. Integration Rationale

### 1.1 Why NautilusTrader?

NautilusTrader provides capabilities that would take 6-12 months to build from scratch:

| Capability | NAT Current State | NautilusTrader Provides |
|------------|-------------------|-------------------------|
| Backtesting Engine | Basic Python implementation | High-fidelity event-driven simulation |
| Order Management | Not implemented | Full OMS with order types, fills, cancellations |
| Position Tracking | Manual calculation | Automatic P&L, margin, exposure tracking |
| Risk Management | Not implemented | Configurable limits, circuit breakers |
| Execution | Not implemented | Multiple venue adapters, smart routing |
| Performance | Python (slow) | Rust core with Python bindings |
| Data Management | Parquet files | Built-in data catalog, caching |

### 1.2 Why NAT Features Matter

Standard indicators are **lagging** and **commoditized**. NAT features provide **leading** information:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INFORMATION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LEADING (NAT Features)          LAGGING (Standard Indicators)         │
│   ─────────────────────           ─────────────────────────────         │
│   • Liquidation clusters          • RSI (14-period price history)       │
│     → Where forced selling        • MACD (EMA difference)               │
│       WILL occur                  • Bollinger (std dev bands)           │
│                                                                         │
│   • Whale flow (4h/24h)           These tell you what HAPPENED          │
│     → Smart money positioning     NAT tells you what's ABOUT TO HAPPEN  │
│                                                                         │
│   • Entropy (regime clarity)                                            │
│     → Signal reliability                                                │
│                                                                         │
│   • VPIN (toxicity)                                                     │
│     → Adverse selection risk                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Combination Hypothesis

**Hypothesis:** Combining standard indicators with microstructure features improves precision.

| Signal Type | Standalone Precision | + NAT Gating | Improvement |
|-------------|---------------------|--------------|-------------|
| RSI Oversold (<30) | ~50% (coin flip) | ~58-62% | +8-12% |
| MACD Crossover | ~48% | ~55-58% | +7-10% |
| Bollinger Squeeze | ~52% | ~57-60% | +5-8% |

**Mechanism:** NAT features act as **filters**, not replacements:

```python
# Without NAT: Every RSI < 30 triggers a buy
if rsi < 30:
    buy()  # ~50% win rate

# With NAT: Only high-conviction signals pass
if rsi < 30:
    if entropy < 0.3:           # Signal is reliable (low noise)
        if whale_flow_4h > 0:   # Smart money agrees
            if liq_asym > 0:    # More shorts at risk (squeeze potential)
                buy()           # ~60% win rate
```

---

## 2. Architecture Options

### 2.1 Option A: Feature Injection (Tight Coupling)

NAT features embedded directly into NautilusTrader's Bar objects.

```
┌─────────────────────────────────────────────────────────────────┐
│                     NautilusTrader Process                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    EnhancedBar                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │   OHLCV     │  │  Standard   │  │  NAT Features   │  │  │
│  │  │  (native)   │  │  Indicators │  │  (injected)     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    Strategy.on_bar()                            │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Single unified data structure
- Lowest latency
- Simple strategy code

**Cons:**
- Tight coupling (changes require both systems)
- Must perfectly sync timestamps
- Harder to test independently

### 2.2 Option B: Signal Gating (Loose Coupling) ⭐ RECOMMENDED

NAT runs separately and provides gating decisions.

```
┌─────────────────────┐                    ┌─────────────────────┐
│   NautilusTrader    │                    │     NAT Engine      │
│                     │    Query/Response  │                     │
│  Standard Signals   │ ◄────────────────► │  Gate Conditions    │
│  • RSI < 30         │                    │  • entropy < 0.3    │
│  • MACD cross       │                    │  • whale_flow > 0   │
│                     │                    │  • liq_asym > 0     │
│         │           │                    │                     │
│         ▼           │                    │         │           │
│  if signal AND gate │                    │         ▼           │
│     → EXECUTE       │                    │  Feature Serving    │
└─────────────────────┘                    └─────────────────────┘
```

**Pros:**
- Clean separation of concerns
- Independent testing and development
- NAT can serve multiple strategies
- Easier to A/B test with/without gating

**Cons:**
- Slight latency overhead
- Requires synchronization logic

### 2.3 Option C: Hierarchical Meta-Model

NAT predicts regime → NautilusTrader selects strategy.

```
                    ┌─────────────────────────────┐
                    │     NAT Regime Classifier   │
                    │  Output: TRENDING | RANGING │
                    │          ACCUMULATING | ... │
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Momentum Strategy│    │ MeanRev Strategy │    │   No Trade       │
│ (when TRENDING)  │    │ (when RANGING)   │    │ (when UNCERTAIN) │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Pros:**
- Theoretically optimal (right strategy per regime)
- Clear decision boundaries

**Cons:**
- Complex to implement
- Regime classifier must be validated first
- More moving parts

### 2.4 Option D: Ensemble Voting

Multiple models vote independently, including NAT features.

```
┌────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE VOTER                          │
│                                                                │
│   ┌────────────┐  ┌────────────┐  ┌─────────────────────────┐ │
│   │  RSI Vote  │  │ MACD Vote  │  │    NAT Composite Vote   │ │
│   │  Weight: 1 │  │  Weight: 1 │  │    Weight: 2            │ │
│   │  BUY: 0.6  │  │  BUY: 0.4  │  │    BUY: 0.75            │ │
│   └─────┬──────┘  └─────┬──────┘  └───────────┬─────────────┘ │
│         └───────────────┼─────────────────────┘               │
│                         ▼                                     │
│          Weighted Score = (0.6×1 + 0.4×1 + 0.75×2) / 4        │
│                        = 0.625                                │
│          Threshold = 0.55 → EXECUTE BUY                       │
└────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Robust to single-signal failure
- Easy to add/remove signals
- Natural confidence scoring

**Cons:**
- Weight optimization is another overfitting vector
- Less interpretable than gating

### 2.5 Option E: Microservice Architecture (Production)

NAT as independent service with REST/WebSocket API.

```
┌─────────────────────┐         ┌─────────────────────────────────┐
│   NautilusTrader    │  REST   │         NAT Service             │
│                     │ ◄─────► │                                 │
│  on_bar():          │         │  GET /features?ts=1712345600    │
│    f = nat.query()  │   WS    │  GET /regime?symbol=BTC-PERP    │
│    if f.gate:       │ ◄─────► │  WS  /stream/features           │
│      execute()      │         │                                 │
│                     │         │  Rust Engine (real-time)        │
└─────────────────────┘         └─────────────────────────────────┘
```

**Pros:**
- Production-grade scalability
- Language-agnostic (Nautilus Python, NAT Rust)
- Can serve multiple consumers
- Independent deployment

**Cons:**
- Network latency
- Infrastructure complexity
- Requires monitoring, health checks

---

## 3. Recommended Architecture

### 3.1 Phased Approach

We recommend a **phased implementation** starting simple and adding complexity:

| Phase | Architecture | Timeline | Goal |
|-------|--------------|----------|------|
| Phase 1 | B: Signal Gating | Week 1-2 | Validate feature value |
| Phase 2 | C: Hierarchical | Week 3-4 | Regime-aware strategies |
| Phase 3 | E: Microservice | Week 5-8 | Production deployment |

### 3.2 Phase 1: Signal Gating Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐  │
│   │ Hyperliquid │     │    NAT      │     │     NautilusTrader      │  │
│   │  WebSocket  │────►│  Ingestor   │     │                         │  │
│   │             │     │  (Rust)     │     │  ┌───────────────────┐  │  │
│   └─────────────┘     └──────┬──────┘     │  │ Standard Indicators│  │  │
│                              │            │  │ RSI, MACD, BB, etc │  │  │
│                              ▼            │  └─────────┬─────────┘  │  │
│                       ┌──────────────┐    │            │            │  │
│                       │ Tick Parquet │    │            ▼            │  │
│                       │   Storage    │    │  ┌───────────────────┐  │  │
│                       └──────┬───────┘    │  │  Signal Generator │  │  │
│                              │            │  │  (raw signals)    │  │  │
│                              ▼            │  └─────────┬─────────┘  │  │
│                       ┌──────────────┐    │            │            │  │
│                       │  Aggregation │    │            ▼            │  │
│                       │   Pipeline   │    │  ┌───────────────────┐  │  │
│                       │ (15m/1h/1d)  │    │  │   Signal Gating   │◄─┼──┤
│                       └──────┬───────┘    │  │  (NAT features)   │  │  │
│                              │            │  └─────────┬─────────┘  │  │
│                              ▼            │            │            │  │
│                       ┌──────────────┐    │            ▼            │  │
│                       │FeatureStore │────►│  ┌───────────────────┐  │  │
│                       │ (Parquet/   │    │  │ Order Management  │  │  │
│                       │  SQLite)    │    │  │ & Execution       │  │  │
│                       └─────────────┘    │  └───────────────────┘  │  │
│                                          │                         │  │
│                                          └─────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow Specification

```
1. INGESTION (Real-time)
   Hyperliquid WS → NAT Rust Ingestor → Tick Parquet (1s resolution)

2. AGGREGATION (Batch, every 15 minutes)
   Tick Parquet → aggregate_features.py → 15m FeatureBars

3. FEATURE SERVING (On-demand)
   Strategy requests features → FeatureStore lookup → Return FeatureBar

4. SIGNAL GENERATION (Per bar)
   NautilusTrader computes RSI, MACD → Raw signal (BUY/SELL/HOLD)

5. SIGNAL GATING (Per signal)
   Raw signal + NAT FeatureBar → Gate logic → Filtered signal

6. EXECUTION (If gate passes)
   Filtered signal → Order Management → Exchange
```

---

## 4. Data Model & Feature Mapping

### 4.1 Core Data Types

#### FeatureBar (15-minute aggregated features)

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FeatureBar:
    """
    Aggregated feature bar combining OHLCV with NAT microstructure features.
    Designed for 15-minute timeframe but adaptable to other intervals.
    """
    # === Identifiers ===
    timestamp: int              # Unix timestamp (ms)
    symbol: str                 # e.g., "BTC-PERP"
    interval: str               # e.g., "15m", "1h", "1d"

    # === Standard OHLCV ===
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int

    # === Entropy Features (Gating) ===
    entropy_mean: float         # Mean normalized entropy over interval
    entropy_std: float          # Entropy stability
    entropy_trend: float        # Entropy direction (rising/falling)
    entropy_regime: str         # "LOW" | "MEDIUM" | "HIGH"

    # === Trend Features ===
    momentum_close: float       # Last momentum_300 value
    momentum_mean: float        # Mean momentum over interval
    hurst_exponent: float       # Persistence measure (>0.5 = trending)
    trend_strength: float       # sqrt(momentum^2 * r_squared)
    r_squared: float            # Trend linearity

    # === Whale Flow Features ===
    whale_flow_sum: float       # Net whale flow over interval
    whale_flow_direction: int   # +1 (accumulating) / -1 (distributing) / 0
    whale_intensity: float      # Absolute whale activity level
    mega_whale_active: bool     # Any mega whale trades?

    # === Liquidation Features ===
    liq_risk_above: float       # Liquidation risk if price rises 2%
    liq_risk_below: float       # Liquidation risk if price falls 2%
    liq_asymmetry: float        # (above - below) / (above + below)
    liq_cluster_distance: float # Distance to nearest cluster (%)
    cascade_probability: float  # Estimated cascade likelihood

    # === Toxicity Features ===
    vpin_mean: float            # Mean VPIN over interval
    vpin_max: float             # Peak toxicity
    adverse_selection: float    # Adverse selection score

    # === Regime Features ===
    regime_absorption: float    # Volume absorbed per price change
    regime_divergence: float    # Kyle's lambda deviation
    regime_churn: float         # Two-sided activity measure
    range_position: float       # Price position in range [0, 1]
    accumulation_score: float   # Composite accumulation indicator
    distribution_score: float   # Composite distribution indicator
    regime_label: str           # "ACCUMULATION" | "DISTRIBUTION" | "TRENDING" | "RANGING" | "NOISE"

    # === Concentration Features ===
    hhi: float                  # Herfindahl-Hirschman Index
    gini: float                 # Gini coefficient
    whale_dominance: float      # Top 10 concentration
    crowding_score: float       # Position crowding measure

    # === Volatility Features ===
    realized_vol: float         # Realized volatility
    vol_regime: str             # "LOW" | "NORMAL" | "HIGH" | "EXTREME"

    # === Market Context ===
    funding_rate: float         # Current funding rate
    open_interest: float        # Total open interest
    oi_change: float            # OI change over interval
    basis: float                # Spot-perp basis

    # === Composite Scores (Pre-computed) ===
    signal_quality: float       # Overall signal reliability [0, 1]
    conviction_score: float     # Trade conviction [0, 1]
    risk_score: float           # Current risk level [0, 1]


@dataclass
class GateConditions:
    """
    Pre-computed gate conditions for fast strategy evaluation.
    """
    # Entropy gates
    low_entropy: bool           # entropy_mean < 0.3
    high_entropy: bool          # entropy_mean > 0.7
    stable_entropy: bool        # entropy_std < 0.1

    # Whale gates
    whale_bullish: bool         # whale_flow_sum > threshold
    whale_bearish: bool         # whale_flow_sum < -threshold
    whale_neutral: bool         # abs(whale_flow_sum) < threshold

    # Liquidation gates
    squeeze_long_risk: bool     # liq_asymmetry < -0.3 (longs at risk)
    squeeze_short_risk: bool    # liq_asymmetry > 0.3 (shorts at risk)
    cascade_imminent: bool      # cascade_probability > 0.5

    # Toxicity gates
    low_toxicity: bool          # vpin_mean < 0.4
    high_toxicity: bool         # vpin_mean > 0.6

    # Regime gates
    accumulation_phase: bool    # regime_label == "ACCUMULATION"
    distribution_phase: bool    # regime_label == "DISTRIBUTION"
    trending_phase: bool        # regime_label == "TRENDING"
    ranging_phase: bool         # regime_label == "RANGING"

    # Composite gates
    high_conviction_long: bool  # Multiple bullish conditions aligned
    high_conviction_short: bool # Multiple bearish conditions aligned
    no_trade: bool              # Conflicting or uncertain signals
```

### 4.2 Feature Selection for 15-Minute Trading

From 183 tick-level features, we select **32 features** for 15-minute aggregation:

| Category | Selected Features | Aggregation Method |
|----------|-------------------|-------------------|
| **Entropy (4)** | normalized_entropy_15m, entropy_rate, conditional_entropy, permutation_entropy | mean, std, last |
| **Trend (5)** | momentum_300, momentum_600, hurst_exponent, r_squared_300, trend_strength | mean, last |
| **Whale (4)** | whale_net_flow_4h, whale_intensity, whale_momentum, mega_whale_count | sum, mean, max |
| **Liquidation (5)** | liq_risk_above_2pct, liq_risk_below_2pct, liq_asymmetry, nearest_cluster, cascade_prob | mean, last |
| **Toxicity (3)** | vpin_10, vpin_50, adverse_selection | mean, max |
| **Regime (6)** | absorption_4h, divergence_4h, churn_4h, range_pos_24h, accum_score, dist_score | mean, last |
| **Concentration (3)** | hhi, gini, whale_dominance | mean |
| **Volatility (2)** | realized_vol_5m, parkinson_vol | mean, max |

### 4.3 Standard Indicator Mapping

NautilusTrader will compute these standard indicators internally:

| Indicator | Parameters | Signal Logic |
|-----------|------------|--------------|
| RSI | period=14 | <30 oversold, >70 overbought |
| MACD | fast=12, slow=26, signal=9 | Crossover signals |
| Bollinger Bands | period=20, std=2 | Band touch/break |
| ATR | period=14 | Volatility measure |
| ADX | period=14 | Trend strength |
| OBV | - | Volume confirmation |
| Stochastic | k=14, d=3 | Momentum oscillator |

### 4.4 Combined Signal Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL COMBINATION MATRIX                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Standard Signal        NAT Gate                   Combined Action         │
│   ───────────────        ────────                   ───────────────         │
│   RSI < 30               entropy < 0.3              HIGH CONVICTION LONG    │
│   (oversold)           + whale_flow > 0                                     │
│                        + liq_asym > 0                                       │
│                        + vpin < 0.4                                         │
│                                                                             │
│   RSI < 30               entropy > 0.5              SKIP (noisy regime)     │
│   (oversold)             OR whale_flow < 0                                  │
│                                                                             │
│   RSI > 70               entropy < 0.3              HIGH CONVICTION SHORT   │
│   (overbought)         + whale_flow < 0                                     │
│                        + liq_asym < 0                                       │
│                                                                             │
│   MACD cross up          regime == ACCUMULATION    CONFIRMED LONG           │
│                        + hurst > 0.5                                        │
│                                                                             │
│   MACD cross up          regime == DISTRIBUTION    SKIP (distribution)      │
│                          OR hurst < 0.5                                     │
│                                                                             │
│   Bollinger squeeze      entropy dropping          BREAKOUT IMMINENT        │
│                        + absorption rising                                  │
│                                                                             │
│   Any signal             cascade_prob > 0.6        WAIT (cascade risk)      │
│                                                                             │
│   Any signal             vpin > 0.7                SKIP (toxic flow)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Aggregation Pipeline

### 5.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGGREGATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT                    PROCESSING                    OUTPUT             │
│   ─────                    ──────────                    ──────             │
│                                                                             │
│   ┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐     │
│   │ Tick Parquet│     │  1. Load & Validate  │     │  15m FeatureBar │     │
│   │ (1s data)   │────►│  2. Resample to 15m  │────►│  Parquet/DB     │     │
│   │ ~86,400     │     │  3. Aggregate stats  │     │  ~96 bars/day   │     │
│   │ rows/day    │     │  4. Compute derived  │     │                 │     │
│   └─────────────┘     │  5. Label regimes    │     └─────────────────┘     │
│                       │  6. Quality checks   │                             │
│                       └──────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation

**File:** `scripts/data/aggregate_to_15m.py`

```python
"""
Aggregation Pipeline: Tick Features → 15-Minute FeatureBars

This script transforms NAT's tick-level features (1s resolution) into
15-minute aggregated FeatureBars suitable for NautilusTrader strategies.

Usage:
    python scripts/data/aggregate_to_15m.py \
        --input ./data/features/ \
        --output ./data/bars_15m/ \
        --symbol BTC-PERP \
        --start 2026-01-01 \
        --end 2026-03-31
"""

import polars as pl
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import json

# Aggregation specifications for each feature
AGGREGATION_SPEC = {
    # Entropy features
    "normalized_entropy_15m": ["mean", "std", "last"],
    "entropy_rate": ["mean", "last"],
    "conditional_entropy": ["mean"],
    "permutation_entropy": ["mean"],

    # Trend features
    "momentum_300": ["mean", "last", "min", "max"],
    "momentum_600": ["mean", "last"],
    "hurst_exponent": ["mean", "last"],
    "r_squared_300": ["mean", "last"],
    "trend_strength": ["mean", "last"],

    # Whale features
    "whale_net_flow_4h": ["sum", "mean", "last"],
    "whale_intensity": ["mean", "max"],
    "whale_momentum": ["mean", "last"],

    # Liquidation features
    "liq_risk_above_2pct": ["mean", "last"],
    "liq_risk_below_2pct": ["mean", "last"],
    "liq_asymmetry": ["mean", "last"],
    "nearest_cluster_distance": ["min", "mean"],
    "cascade_probability": ["mean", "max"],

    # Toxicity features
    "vpin_10": ["mean", "max"],
    "vpin_50": ["mean", "max"],
    "adverse_selection": ["mean"],

    # Regime features
    "regime_absorption_4h": ["mean", "last"],
    "regime_divergence_4h": ["mean", "last"],
    "regime_churn_4h": ["mean"],
    "regime_range_pos_24h": ["last"],
    "regime_accumulation_score": ["mean", "last"],
    "regime_distribution_score": ["mean", "last"],

    # Concentration features
    "concentration_hhi": ["mean"],
    "concentration_gini": ["mean"],
    "whale_dominance_top10": ["mean"],

    # Volatility features
    "realized_vol_5m": ["mean", "max"],
    "parkinson_vol": ["mean"],

    # Market context
    "funding_rate": ["last"],
    "open_interest": ["last"],
    "oi_change_1h": ["sum"],
}


def build_aggregation_expressions(spec: dict) -> List[pl.Expr]:
    """Build Polars aggregation expressions from specification."""
    expressions = []

    for feature, agg_methods in spec.items():
        for method in agg_methods:
            col_name = f"{feature}_{method}"

            if method == "mean":
                expressions.append(pl.col(feature).mean().alias(col_name))
            elif method == "std":
                expressions.append(pl.col(feature).std().alias(col_name))
            elif method == "min":
                expressions.append(pl.col(feature).min().alias(col_name))
            elif method == "max":
                expressions.append(pl.col(feature).max().alias(col_name))
            elif method == "last":
                expressions.append(pl.col(feature).last().alias(col_name))
            elif method == "first":
                expressions.append(pl.col(feature).first().alias(col_name))
            elif method == "sum":
                expressions.append(pl.col(feature).sum().alias(col_name))
            elif method == "count":
                expressions.append(pl.col(feature).count().alias(col_name))

    return expressions


def compute_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Compute OHLCV from tick data."""
    return df.group_by_dynamic(
        "timestamp",
        every="15m",
        closed="left"
    ).agg([
        pl.col("midprice").first().alias("open"),
        pl.col("midprice").max().alias("high"),
        pl.col("midprice").min().alias("low"),
        pl.col("midprice").last().alias("close"),
        pl.col("volume_1s").sum().alias("volume"),
        pl.col("trade_count_1s").sum().alias("trade_count"),
    ])


def classify_entropy_regime(entropy_mean: float) -> str:
    """Classify entropy into regime."""
    if entropy_mean < 0.3:
        return "LOW"
    elif entropy_mean > 0.7:
        return "HIGH"
    else:
        return "MEDIUM"


def classify_market_regime(row: dict) -> str:
    """Classify overall market regime from features."""
    absorption = row.get("regime_absorption_4h_mean", 0)
    divergence = row.get("regime_divergence_4h_mean", 0)
    whale_flow = row.get("whale_net_flow_4h_sum", 0)
    range_pos = row.get("regime_range_pos_24h_last", 0.5)
    entropy = row.get("normalized_entropy_15m_mean", 0.5)

    # Accumulation: high absorption, negative divergence, whale buying, low range
    if absorption > 1.5 and divergence < -1.0 and whale_flow > 0 and range_pos < 0.3:
        return "ACCUMULATION"

    # Distribution: high absorption, positive divergence, whale selling, high range
    if absorption > 1.5 and divergence > 1.0 and whale_flow < 0 and range_pos > 0.7:
        return "DISTRIBUTION"

    # Trending: low entropy, strong momentum
    momentum = abs(row.get("momentum_300_mean", 0))
    if entropy < 0.3 and momentum > 0.005:
        return "TRENDING"

    # Ranging: high entropy, low trend
    if entropy > 0.6 and momentum < 0.002:
        return "RANGING"

    return "NOISE"


def compute_gate_conditions(feature_bar: dict) -> dict:
    """Compute pre-built gate conditions."""
    entropy = feature_bar.get("normalized_entropy_15m_mean", 0.5)
    whale_flow = feature_bar.get("whale_net_flow_4h_sum", 0)
    liq_asym = feature_bar.get("liq_asymmetry_mean", 0)
    vpin = feature_bar.get("vpin_50_mean", 0.5)
    regime = feature_bar.get("regime_label", "NOISE")

    return {
        "low_entropy": entropy < 0.3,
        "high_entropy": entropy > 0.7,
        "whale_bullish": whale_flow > 0.1,
        "whale_bearish": whale_flow < -0.1,
        "squeeze_short_risk": liq_asym > 0.3,
        "squeeze_long_risk": liq_asym < -0.3,
        "low_toxicity": vpin < 0.4,
        "high_toxicity": vpin > 0.6,
        "accumulation_phase": regime == "ACCUMULATION",
        "distribution_phase": regime == "DISTRIBUTION",
        "trending_phase": regime == "TRENDING",
    }


def aggregate_to_15m(
    input_path: Path,
    output_path: Path,
    symbol: str,
    start_date: str,
    end_date: str
) -> None:
    """
    Main aggregation function.

    Args:
        input_path: Directory containing tick Parquet files
        output_path: Directory for output 15m bars
        symbol: Trading symbol (e.g., "BTC-PERP")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    # Load tick data
    tick_files = list(input_path.glob(f"*{symbol}*.parquet"))
    if not tick_files:
        raise FileNotFoundError(f"No tick files found for {symbol}")

    df = pl.scan_parquet(tick_files).filter(
        (pl.col("timestamp") >= start_date) &
        (pl.col("timestamp") <= end_date)
    ).collect()

    print(f"Loaded {len(df)} tick records")

    # Build aggregation expressions
    agg_exprs = build_aggregation_expressions(AGGREGATION_SPEC)

    # Add OHLCV aggregations
    ohlcv_exprs = [
        pl.col("midprice").first().alias("open"),
        pl.col("midprice").max().alias("high"),
        pl.col("midprice").min().alias("low"),
        pl.col("midprice").last().alias("close"),
        pl.col("volume_1s").sum().alias("volume"),
        pl.col("trade_count_1s").sum().alias("trade_count"),
    ]

    # Perform aggregation
    bars = df.group_by_dynamic(
        "timestamp",
        every="15m",
        closed="left"
    ).agg(ohlcv_exprs + agg_exprs)

    print(f"Created {len(bars)} 15-minute bars")

    # Add derived columns
    bars = bars.with_columns([
        # Entropy regime
        pl.when(pl.col("normalized_entropy_15m_mean") < 0.3)
          .then(pl.lit("LOW"))
          .when(pl.col("normalized_entropy_15m_mean") > 0.7)
          .then(pl.lit("HIGH"))
          .otherwise(pl.lit("MEDIUM"))
          .alias("entropy_regime"),

        # Whale direction
        pl.when(pl.col("whale_net_flow_4h_sum") > 0.1)
          .then(pl.lit(1))
          .when(pl.col("whale_net_flow_4h_sum") < -0.1)
          .then(pl.lit(-1))
          .otherwise(pl.lit(0))
          .alias("whale_direction"),

        # Signal quality (composite)
        (1 - pl.col("normalized_entropy_15m_mean").abs()) *
        (1 - pl.col("vpin_50_mean")) *
        (1 + pl.col("regime_absorption_4h_mean").clip(0, 2) / 2)
        .alias("signal_quality"),
    ])

    # Write output
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{symbol}_15m_bars.parquet"
    bars.write_parquet(output_file)

    print(f"Written to {output_file}")

    # Write metadata
    metadata = {
        "symbol": symbol,
        "interval": "15m",
        "start": start_date,
        "end": end_date,
        "bar_count": len(bars),
        "features": list(AGGREGATION_SPEC.keys()),
        "aggregations": AGGREGATION_SPEC,
    }

    with open(output_path / f"{symbol}_15m_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate tick features to 15m bars")
    parser.add_argument("--input", type=Path, required=True, help="Input tick data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output bar directory")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    aggregate_to_15m(
        input_path=args.input,
        output_path=args.output,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end
    )
```

---

## 6. Strategy Templates

### 6.1 Base Strategy Class

**File:** `integrations/nautilus/strategies/base.py`

```python
"""
Base strategy class with NAT feature integration.
All strategies inherit from this to get gating capabilities.
"""

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from typing import Optional
import polars as pl


class NATGatedStrategy(Strategy):
    """
    Base strategy with NAT microstructure gating.

    Subclasses implement:
    - generate_signal(): Standard indicator logic
    - check_gates(): NAT feature gate conditions
    """

    def __init__(
        self,
        instrument_id: InstrumentId,
        feature_store_path: str,
        position_size: float = 1.0,
        max_positions: int = 1,
    ):
        super().__init__()
        self.instrument_id = instrument_id
        self.feature_store_path = feature_store_path
        self.position_size = position_size
        self.max_positions = max_positions

        # Feature cache
        self._feature_cache: Optional[pl.DataFrame] = None
        self._last_feature_load: int = 0

    def on_start(self):
        """Load feature store on strategy start."""
        self._load_features()
        self.log.info(f"Loaded {len(self._feature_cache)} feature bars")

    def _load_features(self):
        """Load or reload feature store."""
        self._feature_cache = pl.read_parquet(self.feature_store_path)
        self._last_feature_load = self.clock.timestamp_ns()

    def get_features(self, timestamp: int) -> Optional[dict]:
        """
        Get NAT features for a given timestamp.
        Returns the most recent feature bar at or before timestamp.
        """
        if self._feature_cache is None:
            return None

        # Find closest bar (round down to 15m boundary)
        bar_ts = (timestamp // 900_000_000_000) * 900_000_000_000

        row = self._feature_cache.filter(
            pl.col("timestamp") == bar_ts
        ).to_dicts()

        return row[0] if row else None

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """
        Generate raw signal from standard indicators.
        Override in subclass.

        Returns:
            OrderSide.BUY, OrderSide.SELL, or None
        """
        raise NotImplementedError

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """
        Check if NAT gates allow the signal.
        Override in subclass for custom gating logic.

        Returns:
            True if signal should execute, False to skip
        """
        raise NotImplementedError

    def on_bar(self, bar: Bar):
        """Main strategy loop with gating."""
        # 1. Generate raw signal
        signal = self.generate_signal(bar)
        if signal is None:
            return

        # 2. Get NAT features
        features = self.get_features(bar.ts_event)
        if features is None:
            self.log.warning(f"No features for timestamp {bar.ts_event}")
            return

        # 3. Check gates
        if not self.check_gates(signal, features):
            self.log.debug(f"Signal {signal} blocked by gates")
            return

        # 4. Execute
        self._execute_signal(signal, bar)

    def _execute_signal(self, signal: OrderSide, bar: Bar):
        """Execute the signal with position management."""
        current_position = self.portfolio.net_position(self.instrument_id)

        if signal == OrderSide.BUY:
            if current_position < self.max_positions:
                self.buy(
                    instrument_id=self.instrument_id,
                    quantity=self.position_size,
                )
                self.log.info(f"BUY executed at {bar.close}")

        elif signal == OrderSide.SELL:
            if current_position > -self.max_positions:
                self.sell(
                    instrument_id=self.instrument_id,
                    quantity=self.position_size,
                )
                self.log.info(f"SELL executed at {bar.close}")
```

### 6.2 Strategy 1: Entropy-Gated RSI

**File:** `integrations/nautilus/strategies/entropy_gated_rsi.py`

```python
"""
Entropy-Gated RSI Strategy

Standard RSI signals filtered by NAT entropy and whale flow.
Only trades when:
- Entropy is low (signal reliable)
- Whale flow confirms direction
- Toxicity is acceptable
"""

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from typing import Optional

from .base import NATGatedStrategy


class EntropyGatedRSI(NATGatedStrategy):
    """
    RSI signals with NAT microstructure gating.

    Signal Logic:
    - RSI < 30 → BUY signal
    - RSI > 70 → SELL signal

    Gate Conditions (ALL must pass):
    - entropy < 0.3 (low noise, reliable signal)
    - whale_flow aligned with signal direction
    - vpin < 0.5 (not toxic)
    - no imminent cascade
    """

    def __init__(
        self,
        instrument_id,
        feature_store_path: str,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        entropy_threshold: float = 0.3,
        vpin_threshold: float = 0.5,
        whale_threshold: float = 0.05,
        position_size: float = 1.0,
    ):
        super().__init__(
            instrument_id=instrument_id,
            feature_store_path=feature_store_path,
            position_size=position_size,
        )

        # RSI configuration
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Gate thresholds
        self.entropy_threshold = entropy_threshold
        self.vpin_threshold = vpin_threshold
        self.whale_threshold = whale_threshold

        # Indicator
        self.rsi = RelativeStrengthIndex(period=rsi_period)

    def on_start(self):
        super().on_start()
        self.register_indicator_for_bars(self.instrument_id, self.rsi)

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """Generate RSI signal."""
        if not self.rsi.initialized:
            return None

        rsi_value = self.rsi.value

        if rsi_value < self.rsi_oversold:
            return OrderSide.BUY
        elif rsi_value > self.rsi_overbought:
            return OrderSide.SELL
        else:
            return None

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """Check NAT gates for RSI signal."""

        # Gate 1: Entropy (signal quality)
        entropy = features.get("normalized_entropy_15m_mean", 1.0)
        if entropy > self.entropy_threshold:
            self.log.debug(f"Entropy gate failed: {entropy:.3f} > {self.entropy_threshold}")
            return False

        # Gate 2: Toxicity
        vpin = features.get("vpin_50_mean", 1.0)
        if vpin > self.vpin_threshold:
            self.log.debug(f"VPIN gate failed: {vpin:.3f} > {self.vpin_threshold}")
            return False

        # Gate 3: Whale flow alignment
        whale_flow = features.get("whale_net_flow_4h_sum", 0)

        if signal == OrderSide.BUY:
            if whale_flow < -self.whale_threshold:
                self.log.debug(f"Whale gate failed: flow {whale_flow:.3f} opposes BUY")
                return False
        elif signal == OrderSide.SELL:
            if whale_flow > self.whale_threshold:
                self.log.debug(f"Whale gate failed: flow {whale_flow:.3f} opposes SELL")
                return False

        # Gate 4: Cascade risk
        cascade_prob = features.get("cascade_probability_max", 0)
        if cascade_prob > 0.6:
            self.log.debug(f"Cascade gate failed: prob {cascade_prob:.3f}")
            return False

        # Gate 5: Liquidation asymmetry confirmation
        liq_asym = features.get("liq_asymmetry_mean", 0)

        if signal == OrderSide.BUY:
            # For longs, we want shorts at risk (positive asymmetry)
            if liq_asym < -0.2:
                self.log.debug(f"Liq asymmetry gate failed: {liq_asym:.3f} unfavorable for BUY")
                return False
        elif signal == OrderSide.SELL:
            # For shorts, we want longs at risk (negative asymmetry)
            if liq_asym > 0.2:
                self.log.debug(f"Liq asymmetry gate failed: {liq_asym:.3f} unfavorable for SELL")
                return False

        # All gates passed
        self.log.info(
            f"Gates passed: entropy={entropy:.3f}, vpin={vpin:.3f}, "
            f"whale={whale_flow:.3f}, cascade={cascade_prob:.3f}, liq_asym={liq_asym:.3f}"
        )
        return True
```

### 6.3 Strategy 2: Regime Momentum

**File:** `integrations/nautilus/strategies/regime_momentum.py`

```python
"""
Regime-Aware Momentum Strategy

Trades momentum only in favorable regimes:
- ACCUMULATION → Long bias
- DISTRIBUTION → Short bias
- TRENDING → Follow momentum
- RANGING/NOISE → No trade
"""

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from typing import Optional

from .base import NATGatedStrategy


class RegimeMomentum(NATGatedStrategy):
    """
    Momentum strategy gated by regime classification.

    Uses MACD for momentum detection, but only trades
    when regime aligns with signal direction.
    """

    def __init__(
        self,
        instrument_id,
        feature_store_path: str,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        hurst_threshold: float = 0.55,
        position_size: float = 1.0,
    ):
        super().__init__(
            instrument_id=instrument_id,
            feature_store_path=feature_store_path,
            position_size=position_size,
        )

        self.hurst_threshold = hurst_threshold
        self.macd = MovingAverageConvergenceDivergence(
            fast_period=macd_fast,
            slow_period=macd_slow,
            signal_period=macd_signal,
        )

    def on_start(self):
        super().on_start()
        self.register_indicator_for_bars(self.instrument_id, self.macd)

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """Generate MACD crossover signal."""
        if not self.macd.initialized:
            return None

        # MACD line crosses above signal line
        if self.macd.value > self.macd.signal and self.macd.value > 0:
            return OrderSide.BUY
        # MACD line crosses below signal line
        elif self.macd.value < self.macd.signal and self.macd.value < 0:
            return OrderSide.SELL
        else:
            return None

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """Check regime alignment."""

        regime = features.get("regime_label", "NOISE")
        hurst = features.get("hurst_exponent_mean", 0.5)
        entropy = features.get("normalized_entropy_15m_mean", 0.5)

        # No trade in noise or ranging regimes
        if regime in ["NOISE", "RANGING"]:
            self.log.debug(f"Regime gate failed: {regime}")
            return False

        # Check Hurst for persistence
        if hurst < self.hurst_threshold:
            self.log.debug(f"Hurst gate failed: {hurst:.3f} < {self.hurst_threshold}")
            return False

        # Regime-signal alignment
        if signal == OrderSide.BUY:
            if regime == "DISTRIBUTION":
                self.log.debug("Regime opposes BUY: DISTRIBUTION")
                return False
            # Prefer ACCUMULATION or TRENDING for buys
            if regime == "ACCUMULATION":
                self.log.info("Strong BUY: ACCUMULATION regime")

        elif signal == OrderSide.SELL:
            if regime == "ACCUMULATION":
                self.log.debug("Regime opposes SELL: ACCUMULATION")
                return False
            if regime == "DISTRIBUTION":
                self.log.info("Strong SELL: DISTRIBUTION regime")

        # Entropy check for signal reliability
        if entropy > 0.6:
            self.log.debug(f"High entropy ({entropy:.3f}): reducing confidence")
            # Could reduce position size here instead of blocking

        return True
```

### 6.4 Strategy 3: Liquidation Squeeze

**File:** `integrations/nautilus/strategies/liquidation_squeeze.py`

```python
"""
Liquidation Squeeze Strategy

Exploits liquidation asymmetry:
- When shorts heavily concentrated near price → Long (squeeze potential)
- When longs heavily concentrated near price → Short (dump potential)

Uses NAT's liquidation mapping as primary signal, not just a gate.
"""

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from typing import Optional

from .base import NATGatedStrategy


class LiquidationSqueeze(NATGatedStrategy):
    """
    Trade liquidation cascades.

    Primary Signal: Liquidation asymmetry from NAT features
    Confirmation: Whale flow direction
    Filter: Low entropy (clear signal)
    """

    def __init__(
        self,
        instrument_id,
        feature_store_path: str,
        asymmetry_threshold: float = 0.4,
        cluster_distance_threshold: float = 0.02,  # 2% from current price
        entropy_threshold: float = 0.4,
        cascade_threshold: float = 0.3,
        position_size: float = 1.0,
    ):
        super().__init__(
            instrument_id=instrument_id,
            feature_store_path=feature_store_path,
            position_size=position_size,
        )

        self.asymmetry_threshold = asymmetry_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        self.entropy_threshold = entropy_threshold
        self.cascade_threshold = cascade_threshold

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """
        Signal based on liquidation asymmetry.
        This strategy uses NAT features for signal generation, not just gating.
        """
        features = self.get_features(bar.ts_event)
        if features is None:
            return None

        liq_asym = features.get("liq_asymmetry_mean", 0)
        cluster_dist = features.get("nearest_cluster_distance_min", 1.0)
        cascade_prob = features.get("cascade_probability_mean", 0)

        # Short squeeze setup: More shorts at risk, cluster nearby
        if (liq_asym > self.asymmetry_threshold and
            cluster_dist < self.cluster_distance_threshold and
            cascade_prob > self.cascade_threshold):
            return OrderSide.BUY  # Expect short squeeze

        # Long squeeze setup: More longs at risk, cluster nearby
        if (liq_asym < -self.asymmetry_threshold and
            cluster_dist < self.cluster_distance_threshold and
            cascade_prob > self.cascade_threshold):
            return OrderSide.SELL  # Expect long liquidation

        return None

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """Confirm with whale flow and entropy."""

        entropy = features.get("normalized_entropy_15m_mean", 1.0)
        whale_flow = features.get("whale_net_flow_4h_sum", 0)

        # Entropy gate: Need clear signal
        if entropy > self.entropy_threshold:
            self.log.debug(f"Entropy too high for liquidation play: {entropy:.3f}")
            return False

        # Whale confirmation (optional but improves odds)
        if signal == OrderSide.BUY:
            if whale_flow < -0.1:
                self.log.debug("Whales opposing squeeze long")
                return False
        elif signal == OrderSide.SELL:
            if whale_flow > 0.1:
                self.log.debug("Whales opposing squeeze short")
                return False

        return True
```

### 6.5 Strategy 4: Whale Flow Follower

**File:** `integrations/nautilus/strategies/whale_flow.py`

```python
"""
Whale Flow Following Strategy

Follow smart money:
- Strong whale buying + supporting conditions → Long
- Strong whale selling + supporting conditions → Short

Simple but effective when whales are active.
"""

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from typing import Optional

from .base import NATGatedStrategy


class WhaleFlowFollower(NATGatedStrategy):
    """
    Follow whale flow when conditions align.
    """

    def __init__(
        self,
        instrument_id,
        feature_store_path: str,
        flow_threshold: float = 0.2,
        intensity_threshold: float = 0.5,
        entropy_threshold: float = 0.4,
        position_size: float = 1.0,
    ):
        super().__init__(
            instrument_id=instrument_id,
            feature_store_path=feature_store_path,
            position_size=position_size,
        )

        self.flow_threshold = flow_threshold
        self.intensity_threshold = intensity_threshold
        self.entropy_threshold = entropy_threshold

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """Generate signal from whale flow."""
        features = self.get_features(bar.ts_event)
        if features is None:
            return None

        whale_flow = features.get("whale_net_flow_4h_sum", 0)
        whale_intensity = features.get("whale_intensity_mean", 0)

        # Need significant flow AND activity
        if whale_intensity < self.intensity_threshold:
            return None  # Whales not active

        if whale_flow > self.flow_threshold:
            return OrderSide.BUY
        elif whale_flow < -self.flow_threshold:
            return OrderSide.SELL
        else:
            return None

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """Additional confirmation gates."""

        entropy = features.get("normalized_entropy_15m_mean", 1.0)
        vpin = features.get("vpin_50_mean", 0.5)
        regime = features.get("regime_label", "NOISE")

        # Entropy gate
        if entropy > self.entropy_threshold:
            return False

        # Toxicity gate
        if vpin > 0.6:
            return False

        # Regime alignment
        if signal == OrderSide.BUY and regime == "DISTRIBUTION":
            return False
        if signal == OrderSide.SELL and regime == "ACCUMULATION":
            return False

        return True
```

### 6.6 Strategy 5: Entropy Breakout

**File:** `integrations/nautilus/strategies/entropy_breakout.py`

```python
"""
Entropy Breakout Strategy

Trade compression → expansion:
- Entropy dropping (compression)
- Absorption rising (volume being absorbed)
- Bollinger squeeze
→ Breakout imminent, direction from momentum/whale flow
"""

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from typing import Optional

from .base import NATGatedStrategy


class EntropyBreakout(NATGatedStrategy):
    """
    Entropy-based breakout detection.
    """

    def __init__(
        self,
        instrument_id,
        feature_store_path: str,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_width: float = 0.02,  # 2% band width = squeeze
        entropy_drop_threshold: float = -0.1,
        absorption_threshold: float = 1.5,
        position_size: float = 1.0,
    ):
        super().__init__(
            instrument_id=instrument_id,
            feature_store_path=feature_store_path,
            position_size=position_size,
        )

        self.squeeze_width = squeeze_width
        self.entropy_drop_threshold = entropy_drop_threshold
        self.absorption_threshold = absorption_threshold

        self.bb = BollingerBands(period=bb_period, k=bb_std)

    def on_start(self):
        super().on_start()
        self.register_indicator_for_bars(self.instrument_id, self.bb)

    def generate_signal(self, bar: Bar) -> Optional[OrderSide]:
        """Detect breakout setup."""
        if not self.bb.initialized:
            return None

        features = self.get_features(bar.ts_event)
        if features is None:
            return None

        # Check for squeeze
        band_width = (self.bb.upper - self.bb.lower) / self.bb.middle
        if band_width > self.squeeze_width:
            return None  # No squeeze

        # Check entropy trend (dropping = compression)
        entropy_trend = features.get("entropy_trend", 0)
        if entropy_trend > self.entropy_drop_threshold:
            return None  # Entropy not dropping

        # Check absorption (volume being absorbed)
        absorption = features.get("regime_absorption_4h_mean", 0)
        if absorption < self.absorption_threshold:
            return None  # Not enough absorption

        # Direction from momentum and whale flow
        momentum = features.get("momentum_300_last", 0)
        whale_flow = features.get("whale_net_flow_4h_sum", 0)

        # Both must agree for signal
        if momentum > 0 and whale_flow > 0:
            return OrderSide.BUY
        elif momentum < 0 and whale_flow < 0:
            return OrderSide.SELL
        else:
            return None  # Conflicting signals

    def check_gates(self, signal: OrderSide, features: dict) -> bool:
        """Minimal gating since signal already uses features."""

        vpin = features.get("vpin_50_mean", 0.5)
        cascade_prob = features.get("cascade_probability_max", 0)

        # Just check toxicity and cascade risk
        if vpin > 0.7:
            return False
        if cascade_prob > 0.7:
            return False

        return True
```

---

## 7. Implementation Plan

### 7.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION TIMELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: Foundation (Week 1-2)                                           │
│   ├── Aggregation pipeline                                     [4-6h]      │
│   ├── FeatureBar data model                                    [2-3h]      │
│   ├── Feature store (Parquet/SQLite)                           [2-3h]      │
│   └── Unit tests                                               [2-3h]      │
│                                                      Subtotal: [10-15h]    │
│                                                                             │
│   PHASE 2: Integration (Week 2-3)                                          │
│   ├── NautilusTrader data adapter                              [4-6h]      │
│   ├── Base strategy class                                      [3-4h]      │
│   ├── Gate condition framework                                 [2-3h]      │
│   └── Integration tests                                        [2-3h]      │
│                                                      Subtotal: [11-16h]    │
│                                                                             │
│   PHASE 3: Strategies (Week 3-4)                                           │
│   ├── Strategy 1: Entropy-Gated RSI                            [3-4h]      │
│   ├── Strategy 2: Regime Momentum                              [3-4h]      │
│   ├── Strategy 3: Liquidation Squeeze                          [3-4h]      │
│   ├── Strategy 4: Whale Flow Follower                          [2-3h]      │
│   ├── Strategy 5: Entropy Breakout                             [3-4h]      │
│   └── Strategy tests                                           [3-4h]      │
│                                                      Subtotal: [17-23h]    │
│                                                                             │
│   PHASE 4: Validation (Week 4-5)                                           │
│   ├── Backtesting harness setup                                [4-6h]      │
│   ├── Walk-forward validation                                  [4-6h]      │
│   ├── Performance analysis                                     [3-4h]      │
│   └── Parameter sensitivity                                    [3-4h]      │
│                                                      Subtotal: [14-20h]    │
│                                                                             │
│   PHASE 5: Production (Week 5-8)                                           │
│   ├── Microservice architecture                                [8-12h]     │
│   ├── Real-time feature streaming                              [6-8h]      │
│   ├── Monitoring & alerting                                    [4-6h]      │
│   ├── Paper trading                                            [ongoing]   │
│   └── Documentation                                            [4-6h]      │
│                                                      Subtotal: [22-32h]    │
│                                                                             │
│   TOTAL ESTIMATE: 74-106 hours                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Detailed Task Breakdown

#### Phase 1: Foundation

| Task | Description | Dependencies | Hours |
|------|-------------|--------------|-------|
| 1.1 | Implement `aggregate_to_15m.py` | Existing tick Parquet | 4-6 |
| 1.2 | Define `FeatureBar` dataclass | None | 2-3 |
| 1.3 | Create feature store (read/write) | 1.1, 1.2 | 2-3 |
| 1.4 | Unit tests for aggregation | 1.1, 1.2, 1.3 | 2-3 |

**Deliverables:**
- `scripts/data/aggregate_to_15m.py`
- `integrations/nautilus/data/feature_bar.py`
- `integrations/nautilus/data/feature_store.py`
- `tests/integration/test_aggregation.py`

#### Phase 2: Integration

| Task | Description | Dependencies | Hours |
|------|-------------|--------------|-------|
| 2.1 | NautilusTrader data adapter | Phase 1 | 4-6 |
| 2.2 | `NATGatedStrategy` base class | 2.1 | 3-4 |
| 2.3 | Gate condition helpers | 2.2 | 2-3 |
| 2.4 | Integration tests | 2.1, 2.2, 2.3 | 2-3 |

**Deliverables:**
- `integrations/nautilus/adapters/hyperliquid.py`
- `integrations/nautilus/strategies/base.py`
- `integrations/nautilus/gates/conditions.py`
- `tests/integration/test_nautilus_adapter.py`

#### Phase 3: Strategies

| Task | Description | Dependencies | Hours |
|------|-------------|--------------|-------|
| 3.1 | Entropy-Gated RSI | Phase 2 | 3-4 |
| 3.2 | Regime Momentum | Phase 2 | 3-4 |
| 3.3 | Liquidation Squeeze | Phase 2 | 3-4 |
| 3.4 | Whale Flow Follower | Phase 2 | 2-3 |
| 3.5 | Entropy Breakout | Phase 2 | 3-4 |
| 3.6 | Strategy unit tests | 3.1-3.5 | 3-4 |

**Deliverables:**
- `integrations/nautilus/strategies/entropy_gated_rsi.py`
- `integrations/nautilus/strategies/regime_momentum.py`
- `integrations/nautilus/strategies/liquidation_squeeze.py`
- `integrations/nautilus/strategies/whale_flow.py`
- `integrations/nautilus/strategies/entropy_breakout.py`
- `tests/strategies/test_*.py`

#### Phase 4: Validation

| Task | Description | Dependencies | Hours |
|------|-------------|--------------|-------|
| 4.1 | Backtesting harness | Phase 3 | 4-6 |
| 4.2 | Walk-forward validation | 4.1 | 4-6 |
| 4.3 | Performance metrics | 4.1, 4.2 | 3-4 |
| 4.4 | Parameter sensitivity | 4.1, 4.2 | 3-4 |

**Deliverables:**
- `scripts/backtest/nautilus_runner.py`
- `scripts/backtest/walk_forward_nautilus.py`
- `scripts/analysis/strategy_performance.py`
- `docs/reports/backtest_results.md`

#### Phase 5: Production

| Task | Description | Dependencies | Hours |
|------|-------------|--------------|-------|
| 5.1 | Feature serving API | Phase 1 | 8-12 |
| 5.2 | Real-time streaming | 5.1 | 6-8 |
| 5.3 | Monitoring/alerting | 5.1, 5.2 | 4-6 |
| 5.4 | Paper trading setup | All above | ongoing |
| 5.5 | Documentation | All above | 4-6 |

**Deliverables:**
- `services/feature_server/` (FastAPI)
- `integrations/nautilus/adapters/realtime.py`
- `monitoring/dashboards/`
- `docs/deployment/`

---

## 8. API Specifications

### 8.1 Feature Server API

**Base URL:** `http://localhost:8000/api/v1`

#### GET /features

Get features for a specific timestamp.

```http
GET /features?symbol=BTC-PERP&timestamp=1712345600000
```

**Response:**
```json
{
  "timestamp": 1712345600000,
  "symbol": "BTC-PERP",
  "interval": "15m",
  "features": {
    "entropy_mean": 0.234,
    "whale_flow_sum": 0.156,
    "liq_asymmetry": 0.087,
    "regime_label": "ACCUMULATION",
    ...
  },
  "gates": {
    "low_entropy": true,
    "whale_bullish": true,
    "high_conviction_long": true,
    ...
  }
}
```

#### GET /gates

Get gate conditions only (faster).

```http
GET /gates?symbol=BTC-PERP&timestamp=1712345600000&signal=BUY
```

**Response:**
```json
{
  "timestamp": 1712345600000,
  "signal": "BUY",
  "gate_result": true,
  "gates_passed": ["low_entropy", "whale_bullish", "low_toxicity"],
  "gates_failed": [],
  "confidence": 0.78
}
```

#### GET /regime

Get current regime classification.

```http
GET /regime?symbol=BTC-PERP
```

**Response:**
```json
{
  "timestamp": 1712345600000,
  "symbol": "BTC-PERP",
  "regime": "ACCUMULATION",
  "confidence": 0.85,
  "supporting_features": {
    "absorption": 1.8,
    "divergence": -1.2,
    "whale_flow": 0.25,
    "range_position": 0.22
  },
  "recommended_strategies": ["momentum", "whale_flow"]
}
```

#### WebSocket /stream/features

Real-time feature streaming.

```javascript
ws://localhost:8000/ws/features?symbol=BTC-PERP

// Server sends on each new 15m bar:
{
  "type": "feature_bar",
  "data": { ... FeatureBar ... }
}

// Server sends on regime change:
{
  "type": "regime_change",
  "previous": "RANGING",
  "current": "ACCUMULATION",
  "confidence": 0.82
}
```

### 8.2 Data Adapter Interface

```python
class NATDataAdapter(Protocol):
    """Interface for NAT feature data."""

    def get_features(self, symbol: str, timestamp: int) -> Optional[FeatureBar]:
        """Get features for a specific timestamp."""
        ...

    def get_features_range(
        self,
        symbol: str,
        start: int,
        end: int
    ) -> List[FeatureBar]:
        """Get features for a time range."""
        ...

    def check_gates(
        self,
        symbol: str,
        timestamp: int,
        signal: OrderSide
    ) -> GateResult:
        """Check if gates allow a signal."""
        ...

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[FeatureBar], None]
    ) -> None:
        """Subscribe to real-time feature updates."""
        ...
```

---

## 9. Backtesting Framework

### 9.1 Backtest Configuration

```python
# config/backtest_config.py

BACKTEST_CONFIG = {
    "data": {
        "symbols": ["BTC-PERP", "ETH-PERP", "SOL-PERP"],
        "start_date": "2025-01-01",
        "end_date": "2026-03-31",
        "interval": "15m",
        "feature_store": "./data/bars_15m/",
    },

    "validation": {
        "method": "walk_forward",
        "train_days": 60,
        "test_days": 15,
        "retrain_frequency": "weekly",
        "min_trades_per_fold": 30,
    },

    "costs": {
        "maker_fee": 0.0002,      # 2 bps
        "taker_fee": 0.0006,      # 6 bps
        "slippage_bps": 2,        # 2 bps estimated slippage
        "funding_rate": True,     # Include funding in P&L
    },

    "risk": {
        "max_position_size": 1.0,
        "max_drawdown": 0.20,     # 20% max drawdown
        "stop_loss": 0.05,        # 5% stop loss
        "take_profit": 0.15,      # 15% take profit
    },

    "strategies": [
        {
            "name": "EntropyGatedRSI",
            "class": "integrations.nautilus.strategies.EntropyGatedRSI",
            "params": {
                "rsi_period": 14,
                "entropy_threshold": 0.3,
                "vpin_threshold": 0.5,
            }
        },
        {
            "name": "RegimeMomentum",
            "class": "integrations.nautilus.strategies.RegimeMomentum",
            "params": {
                "hurst_threshold": 0.55,
            }
        },
        # ... more strategies
    ],
}
```

### 9.2 Validation Criteria

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Walk-Forward Sharpe | > 0.3 | > 0.5 | > 0.8 |
| OOS/IS Ratio | > 0.6 | > 0.7 | > 0.85 |
| Win Rate | > 50% | > 55% | > 60% |
| Profit Factor | > 1.1 | > 1.3 | > 1.5 |
| Max Drawdown | < 25% | < 20% | < 15% |
| Avg Trade Duration | > 2h | > 4h | > 8h |
| Trades per Month | > 10 | > 20 | > 30 |

### 9.3 Comparison Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STRATEGY COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BASELINE (no NAT gating)          WITH NAT GATING                        │
│   ────────────────────────          ───────────────                        │
│   Strategy: RSI only                Strategy: RSI + NAT gates              │
│   Sharpe: 0.15                      Sharpe: 0.52                           │
│   Win Rate: 48%                     Win Rate: 57%                          │
│   Trades: 150/month                 Trades: 45/month                       │
│   Max DD: 28%                       Max DD: 18%                            │
│                                                                             │
│   IMPROVEMENT                                                               │
│   • Sharpe: +0.37 (+247%)                                                  │
│   • Win Rate: +9pp                                                         │
│   • Trades: -70% (filtered noise)                                          │
│   • Max DD: -10pp (risk reduction)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Risk Management Integration

### 10.1 Position Sizing

```python
def calculate_position_size(
    features: FeatureBar,
    base_size: float,
    max_size: float,
) -> float:
    """
    Dynamic position sizing based on signal quality.

    Higher conviction → Larger position
    Higher risk → Smaller position
    """

    # Signal quality factor [0.5, 1.5]
    quality_factor = 0.5 + features.signal_quality

    # Volatility adjustment [0.5, 1.5]
    if features.vol_regime == "EXTREME":
        vol_factor = 0.5
    elif features.vol_regime == "HIGH":
        vol_factor = 0.75
    elif features.vol_regime == "NORMAL":
        vol_factor = 1.0
    else:  # LOW
        vol_factor = 1.25

    # Cascade risk adjustment
    cascade_factor = 1.0 - features.cascade_probability

    # Combined sizing
    size = base_size * quality_factor * vol_factor * cascade_factor

    return min(size, max_size)
```

### 10.2 Dynamic Stop Loss

```python
def calculate_stop_loss(
    entry_price: float,
    side: OrderSide,
    features: FeatureBar,
    base_stop_pct: float = 0.05,
) -> float:
    """
    Dynamic stop loss based on liquidation clusters.

    Place stop just beyond nearest liquidation cluster
    to avoid being stopped out by cascades.
    """

    cluster_distance = features.liq_cluster_distance

    # At least base stop, but extend if cluster is nearby
    stop_distance = max(base_stop_pct, cluster_distance + 0.005)

    # Cap at reasonable maximum
    stop_distance = min(stop_distance, 0.10)  # Max 10%

    if side == OrderSide.BUY:
        return entry_price * (1 - stop_distance)
    else:
        return entry_price * (1 + stop_distance)
```

### 10.3 Circuit Breakers

```python
CIRCUIT_BREAKERS = {
    # Entropy spike: Market becoming unpredictable
    "entropy_spike": {
        "condition": lambda f: f.entropy_mean > 0.8 and f.entropy_std > 0.2,
        "action": "close_all_positions",
        "cooldown_minutes": 60,
    },

    # Cascade imminent: High liquidation risk
    "cascade_risk": {
        "condition": lambda f: f.cascade_probability > 0.7,
        "action": "reduce_exposure_50pct",
        "cooldown_minutes": 30,
    },

    # Toxic flow: Adverse selection risk
    "toxic_flow": {
        "condition": lambda f: f.vpin_max > 0.8,
        "action": "pause_new_entries",
        "cooldown_minutes": 15,
    },

    # Regime transition: Uncertainty
    "regime_transition": {
        "condition": lambda f: f.regime_label == "NOISE" and f.entropy_std > 0.15,
        "action": "pause_new_entries",
        "cooldown_minutes": 30,
    },
}
```

---

## 11. Success Metrics

### 11.1 Integration Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| Feature latency | < 10ms | Time from bar close to feature availability |
| Data completeness | > 99% | % of bars with complete features |
| Gate accuracy | > 80% | Gates correctly predict signal quality |
| System uptime | > 99.9% | Feature server availability |

### 11.2 Strategy Success

| Metric | Minimum | Target |
|--------|---------|--------|
| Gating Lift | > 5% | Win rate improvement from gating |
| Sharpe Improvement | > 0.2 | vs baseline without NAT features |
| Risk Reduction | > 10% | Max drawdown reduction |
| Trade Efficiency | > 20% | Fewer trades, same/better returns |

### 11.3 Business Success

| Metric | 3-Month Target | 6-Month Target |
|--------|----------------|----------------|
| Paper Trading Sharpe | > 0.4 | > 0.5 |
| Live Trading Sharpe | N/A | > 0.3 |
| Capital Deployed | $10K (paper) | $50K (live) |
| Monthly Return Target | 3-5% | 2-4% |

---

## 12. Risk Analysis

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Feature store corruption | Low | High | Checksums, backups, validation |
| Timestamp sync issues | Medium | High | NTP sync, tolerance windows |
| NautilusTrader API changes | Medium | Medium | Version pinning, adapter abstraction |
| Latency spikes | Medium | Medium | Caching, async processing |
| Memory exhaustion | Low | High | Streaming aggregation, limits |

### 12.2 Strategy Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting to backtest | High | High | Walk-forward validation, OOS checks |
| Feature decay | Medium | High | Continuous monitoring, retraining |
| Regime shift | Medium | High | Multiple strategies, regime detection |
| Correlation breakdown | Medium | Medium | Feature importance tracking |
| Whale behavior change | Medium | Medium | Adaptive thresholds |

### 12.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Exchange API changes | Medium | High | Abstraction layer, alerts |
| Liquidation cascade | Low | High | Circuit breakers, position limits |
| Network outage | Low | High | Redundancy, graceful degradation |
| Key compromise | Low | Critical | HSM, key rotation, 2FA |

---

## Appendices

### Appendix A: File Structure

```
nat/
├── integrations/
│   └── nautilus/
│       ├── __init__.py
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── hyperliquid.py      # Exchange adapter
│       │   ├── feature_store.py    # Feature data adapter
│       │   └── realtime.py         # Real-time streaming
│       ├── data/
│       │   ├── __init__.py
│       │   ├── feature_bar.py      # FeatureBar dataclass
│       │   └── gate_conditions.py  # GateConditions dataclass
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── base.py             # NATGatedStrategy base
│       │   ├── entropy_gated_rsi.py
│       │   ├── regime_momentum.py
│       │   ├── liquidation_squeeze.py
│       │   ├── whale_flow.py
│       │   └── entropy_breakout.py
│       └── gates/
│           ├── __init__.py
│           └── conditions.py       # Gate condition helpers
├── scripts/
│   └── data/
│       └── aggregate_to_15m.py     # Aggregation pipeline
├── services/
│   └── feature_server/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app
│       ├── routes/
│       │   ├── features.py
│       │   ├── gates.py
│       │   └── regime.py
│       └── websocket/
│           └── stream.py
├── config/
│   └── backtest_config.py
└── tests/
    ├── integration/
    │   ├── test_aggregation.py
    │   └── test_nautilus_adapter.py
    └── strategies/
        ├── test_entropy_gated_rsi.py
        └── ...
```

### Appendix B: Dependencies

```toml
# pyproject.toml additions

[project.dependencies]
nautilus-trader = ">=1.180.0"
polars = ">=0.20.0"
fastapi = ">=0.109.0"
uvicorn = ">=0.27.0"
websockets = ">=12.0"
httpx = ">=0.26.0"
pydantic = ">=2.5.0"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "hypothesis>=6.92.0",
]
```

### Appendix C: Configuration Schema

```python
# config/nautilus_config.py

from pydantic import BaseModel
from typing import List, Optional

class FeatureStoreConfig(BaseModel):
    path: str = "./data/bars_15m/"
    cache_size_mb: int = 512
    preload_symbols: List[str] = ["BTC-PERP", "ETH-PERP"]

class GateConfig(BaseModel):
    entropy_threshold: float = 0.3
    vpin_threshold: float = 0.5
    whale_threshold: float = 0.1
    cascade_threshold: float = 0.6

class StrategyConfig(BaseModel):
    name: str
    enabled: bool = True
    symbols: List[str]
    position_size: float = 1.0
    max_positions: int = 1
    gates: GateConfig = GateConfig()
    params: dict = {}

class NautilusIntegrationConfig(BaseModel):
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    strategies: List[StrategyConfig] = []
    risk_limits: dict = {
        "max_drawdown": 0.20,
        "max_position_value": 100000,
        "daily_loss_limit": 5000,
    }
```

### Appendix D: References

1. **NautilusTrader Documentation**: https://nautilustrader.io/docs/
2. **NAT Feature Specifications**: `docs/specs/new/ALGORITHM_DESIGN_PROPOSAL.md`
3. **Hyperliquid API**: `docs/HYPER_DOCS.md`
4. **Walk-Forward Validation**: `docs/user_guide/ML_BACKTESTING_COMPLETE.md`

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-05 | NAT Team | Initial proposal |

---

*End of Document*

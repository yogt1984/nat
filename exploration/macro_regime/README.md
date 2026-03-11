# Macro Regime Detection System

A hierarchical regime detection framework that combines:
1. **Business Cycle Analysis** (ISM PMI, Chicago PMI, Services PMI)
2. **Liquidity Conditions** (DXY, US10Y, Credit Spreads, Yield Curve)
3. **Real Economy Indicators** (Copper/Gold, Jobless Claims)
4. **Crypto-Specific Indicators** (ETH/BTC, BTC Dominance, OTHERS/BTC)
5. **On-Chain Metrics** (MVRV, Exchange Whale Ratio, Funding Rates, LTH Supply)

## Purpose

This system addresses the question: **"Where are we in the cycle?"** and translates that into actionable position sizing guidance.

It operates at a different timescale than the microstructure entropy-based regime detection in `../validation/`. This is the **macro overlay** that determines overall exposure, while the entropy system handles intra-day strategy selection.

## Quick Start

```bash
cd /home/onat/nat/exploration/macro_regime
python run_regime_check.py --example    # See demo with sample data
python run_regime_check.py --quick      # Quick check essential indicators
python run_regime_check.py --full       # Full analysis with persistence
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    INDICATOR INPUTS                            │
│  Business Cycle │ Liquidity │ Real Economy │ Crypto │ On-Chain │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                 REGIME DETECTOR                                 │
│  - Score each indicator [-1, +1]                               │
│  - Weight by category importance                               │
│  - Compute composite score                                     │
│  - Check kill switches                                         │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                 REGIME CLASSIFICATION                          │
│  EXPANSION │ LATE_CYCLE │ UNCERTAIN │ EARLY_RECOVERY │ CONTRACTION │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                 DECISION ENGINE                                 │
│  - Action recommendation (ACCUMULATE/HOLD/REDUCE/EXIT/WAIT)   │
│  - Position sizing guidance                                    │
│  - Risk guardrails                                             │
│  - Pre-commitment rule checks                                  │
└────────────────────────────────────────────────────────────────┘
```

## Regimes

| Regime | Score Range | Position Size | Description |
|--------|-------------|---------------|-------------|
| EXPANSION | > 0.3 | 100% | Risk-on, full exposure |
| LATE_CYCLE | 0.1 to 0.3 | 70% | Cautious, reduce leverage |
| UNCERTAIN | -0.1 to 0.1 | 50% | Mixed signals, half size |
| EARLY_RECOVERY | -0.3 to -0.1 | 60% | Building positions |
| CONTRACTION | < -0.3 | 25% | Defensive, preserve capital |

## Indicators

### Business Cycle (Leading)
- **ISM Manufacturing PMI**: >50 expansion, <50 contraction
- **ISM Services PMI**: Services sector activity
- **Chicago PMI**: Regional, often leads national
- **New Orders - Inventories Spread**: Positive = restocking cycle

### Liquidity Conditions
- **DXY (Dollar Index)**: Weaker dollar = more liquidity
- **US 10Y Yield**: Lower yields generally supportive
- **10Y-2Y Spread**: Inversion predicts recession
- **Credit Spreads (HY OAS)**: Wider = stress

### Crypto Indicators
- **ETH/BTC**: Rising = risk appetite, alt season approaching
- **OTHERS/BTC**: ATL often precedes massive alt surge
- **BTC Dominance**: Falling = capital rotating to alts

### On-Chain (CryptoQuant)
- **MVRV**: >3 overheated, <1 undervalued
- **Exchange Whale Ratio**: >0.5 = distribution
- **Funding Rates**: >0.05% = overleveraged longs
- **LTH Supply Change**: Rising = accumulation
- **Exchange Reserve**: Declining = bullish
- **SOPR**: <1 = capitulation, buying opportunity

## Kill Switches

Pre-defined conditions that force position reduction:

| Kill Switch | Trigger | Action |
|-------------|---------|--------|
| ISM PMI Crash | PMI < 45 | Reduce 50% |
| DXY Spike | DXY > 110 | Reduce 30% |
| Credit Blowout | HY Spread > 600bps | Reduce 50% |
| MVRV Extreme | MVRV > 3.5 | Reduce 30% |
| Whale Distribution | Whale Ratio > 0.55 | Reduce 20% |

## Pre-Commitment Framework

To prevent emotional decisions, the system enforces rules defined in advance:

```python
rules = {
    # Minimum time between actions
    "min_action_interval_hours": 24,

    # Don't chase pumps
    "no_buy_after_pump_pct": 10,   # No buying after >10% daily move

    # Don't panic sell dumps
    "no_sell_after_dump_pct": 15,  # No selling after >15% dump

    # Confidence requirements
    "min_confidence_to_act": 0.5,

    # Always scale into positions
    "always_scale_in": True,
    "min_tranches": 3,
}
```

## Usage Examples

### Basic Python Usage

```python
from regime_detector import MacroRegimeDetector
from decision_engine import DecisionEngine

# Create detector and engine
detector = MacroRegimeDetector()
engine = DecisionEngine(detector=detector)

# Update indicators
detector.update_indicator("ism_pmi", 51.2)
detector.update_indicator("dxy", 103.5)
detector.update_indicator("mvrv", 1.8)
# ... more indicators

# Get recommendation
engine.print_recommendation()
```

### With Data Persistence

```python
from data_manager import MacroDataManager, IndicatorInputHelper

dm = MacroDataManager()

# Add readings (automatically persisted)
dm.add_reading("ism_pmi", 51.2)
dm.add_reading("dxy", 103.5)

# Get statistics
stats = dm.get_statistics("ism_pmi", lookback_days=365)
print(f"ISM PMI: mean={stats['mean']:.1f}, current percentile={stats['percentile_current']:.0f}%")
```

### Import from CSV

```python
dm = MacroDataManager()
dm.import_from_csv("indicators.csv")

# CSV format:
# date,indicator,value,source,notes
# 2024-01-15,ism_pmi,50.3,manual,
```

## Answering Your Questions

### 1. Are we in a correction or bear market?

Use the system to score current conditions:
- If **composite score > 0** with **liquidity positive**: Likely correction in bull
- If **composite score < -0.3** with **multiple kill switches**: Bear market
- Key differentiator: Is DXY strengthening while credit spreads widening? That's bear.

### 2. How to avoid doing something stupid?

1. **Enter indicators weekly** (or when major data releases)
2. **Obey the pre-commitment rules** - the system will block emotional trades
3. **Use kill switches** - hard rules that force risk reduction
4. **Scale in/out** - never full position in one trade

### 3. ISM PMI to 54-56 in H2 2026?

Monitor:
- **New Orders - Inventories spread**: Should be positive and rising
- **Credit conditions**: Should be easing
- **Inventory cycle**: 12+ months of destocking typically precedes restocking

The system will show EARLY_RECOVERY → EXPANSION as this plays out.

## Files

| File | Description |
|------|-------------|
| `config.py` | Indicator definitions, thresholds, weights |
| `regime_detector.py` | Core regime detection logic |
| `decision_engine.py` | Action recommendations, pre-commitment |
| `data_manager.py` | Data persistence, statistics |
| `run_regime_check.py` | CLI interface |

## Integration with Entropy System

This macro regime system provides the **position sizing overlay**:

```
Macro Regime (this system)     →  How much to deploy (0-100%)
  ↓
Micro Regime (entropy system)  →  Which strategy (MR vs TF)
  ↓
Execution                      →  Order placement
```

When macro regime is CONTRACTION, reduce size regardless of micro regime signals.

## Data Sources

Currently manual input. Stubs exist for:
- **FRED API**: For macro indicators (ISM, yields, jobless claims)
- **CryptoQuant API**: For on-chain data

To implement automated data fetching, extend the classes in `data_manager.py`.

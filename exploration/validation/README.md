# NAT Week 1-2 Validation

## Overview

This validation answers the critical question: **Does entropy predict market regimes?**

If MI(entropy, regime) > threshold → Proceed to Phase 3
If MI(entropy, regime) < threshold → Pivot or investigate alternatives

## Quick Start

### 1. Setup Environment

```bash
cd /home/onat/nat/exploration/validation
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Quick Test (Synthetic Data)

Test the pipeline works with synthetic data that has KNOWN regime structure:

```bash
python run_validation.py --synthetic
```

Expected result: HIGH mutual information (since synthetic data has clear regimes)

### 3. Real Validation

#### Option A: Short Test (10 minutes)

```bash
python run_validation.py --duration 600
```

#### Option B: Proper Validation (1 hour minimum)

```bash
python run_validation.py --duration 3600
```

#### Option C: Full Validation (24+ hours recommended)

```bash
python run_validation.py --duration 86400
```

Or run in background:
```bash
nohup python run_validation.py --duration 86400 > validation.log 2>&1 &
```

## Pipeline Steps

### Step 1: Data Collection (`data_collector.py`)

- Connects to Hyperliquid WebSocket
- Collects order book snapshots and trades
- Computes features in real-time:
  - **Entropy features**: permutation_entropy at 8/16/32 samples
  - **Imbalance features**: order book imbalance, persistence
  - **Flow features**: aggressor ratio, momentum
  - **Volatility features**: realized vol at multiple timescales
- Saves to Parquet files

### Step 2: Regime Labeling (`regime_labeler.py`)

- Simulates ASMM (market making) on each window
- Simulates TrendFollow on each window
- Labels regime based on which strategy performs better:
  - **MR** (Mean-Reversion): ASMM outperforms
  - **TF** (Trend-Following): TrendFollow outperforms
  - **NA** (Neither): Neither significantly profitable

### Step 3: Hypothesis Validation (`validate_hypothesis.py`)

Computes:
- **Mutual Information**: MI(entropy, regime)
- **Conditional Entropy**: H(regime | entropy) vs H(regime)
- **Statistical Significance**: Permutation tests
- **Correlations**: Entropy vs strategy PnL

Makes **GO / NO-GO decision**:
- **GO**: Entropy MI > 0.05 → Proceed to classifier
- **GO (COMPOSITE)**: Composite signal works even if entropy alone weak
- **INVESTIGATE**: Some signal but needs more data
- **NO-GO**: No signal → Pivot to alternatives

## Interpreting Results

### Mutual Information Scores

| MI Score | Interpretation |
|----------|----------------|
| < 0.02 | No signal - likely noise |
| 0.02 - 0.05 | Weak signal - investigate further |
| 0.05 - 0.10 | Moderate signal - proceed with caution |
| > 0.10 | Strong signal - high confidence proceed |

### Expected Relationships

1. **Entropy vs ASMM**: Should be NEGATIVELY correlated
   - High entropy = bad for market making (unexpected moves)

2. **Entropy vs TrendFollow**: Could be either
   - If positive: High entropy = good for trend (strong moves)
   - If negative: Low entropy = predictable trends

3. **Imbalance Persistence**: Higher in TF regime
   - Trends have consistent order flow

4. **Aggressor Momentum**: More extreme in TF regime
   - Trends have consistent buyer/seller pressure

## Files Generated

```
results/
├── regime_labels.parquet   # Labeled regime data
├── regime_labels.csv       # Same in CSV for inspection
├── validation_results.json # MI scores and decision
└── validation_results.png  # Visualization
```

## Next Steps Based on Decision

### If GO:
1. Run `python -c "print('Proceed to Phase 3')"` (placeholder)
2. Build XGBoost regime classifier
3. Implement walk-forward validation
4. Backtest regime-conditioned strategies

### If NO-GO:
1. Try alternative entropy measures:
   - Sample entropy (ApEn)
   - Multi-scale entropy
   - Spectral entropy
2. Try transfer entropy networks
3. Pivot to funding rate prediction
4. Investigate information geometry approach

## Troubleshooting

### WebSocket Connection Fails

Check if Hyperliquid API is accessible:
```bash
curl https://api.hyperliquid.xyz/info
```

### Not Enough Data

Validation requires minimum 5000 samples (~8 minutes at 100ms).
Recommended: 36000+ samples (1 hour).

### MI Always Zero

1. Check feature computation (may have NaN/Inf values)
2. Verify regime labeling is producing varied labels
3. Try with synthetic data to verify pipeline works

## Configuration

Edit `config.py` to adjust:
- `MI_SIGNIFICANCE_THRESHOLD`: Default 0.05
- `REGIME_THRESHOLD_LOW/HIGH`: Entropy regime boundaries
- `EMISSION_INTERVAL_MS`: Feature frequency (default 100ms)

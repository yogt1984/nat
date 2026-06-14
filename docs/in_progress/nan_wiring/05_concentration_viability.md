# 05 — Concentration Feature Viability Gate

## What

After steps 01-04 are live, assess whether concentration features (15) produce
meaningful values or should remain NaN.

## Why

Concentration metrics (Herfindahl, Gini, top-N share) require position data from
enough accounts to be statistically representative. Hyperliquid has no public
endpoint for aggregate position distribution. We depend entirely on the wallets
discovered/configured in step 04.

This is a decision gate, not a code change.

## Decision Criteria

| Condition | Action |
|-----------|--------|
| 50+ wallets tracked, covering >20% of OI | Concentration features are viable |
| 20-50 wallets, 5-20% of OI | Features are noisy — add disclaimer in FEATURES.md |
| <20 wallets or <5% of OI | Keep features as NaN, document as unavailable |

## How to Measure

After running ingestor with position tracker enabled for 24h+:

```bash
# Check tracked wallet count
nat status --json | jq '.position_tracker.wallet_count'

# Check OI coverage
# Sum of tracked position values / total OI from context
# Log this in the position tracker consumer task
```

## If Viable

No code changes needed — concentration features are already wired in step 03.
Just verify values are reasonable:

```python
import pandas as pd
df = pd.read_parquet("data/features/latest/*.parquet")
print(df[["top5_concentration", "herfindahl_index", "gini_coefficient"]].describe())
# top5 should be 0.05-0.50, HHI 0.01-0.25, Gini 0.3-0.8
```

## If Not Viable

Two options:

### Option A: Keep NaN (recommended)
- No schema change (236 features preserved)
- All downstream code already handles NaN (algorithms, ML, parquet)
- Add comment in `FEATURES.md` under concentration section:
  "Requires per-account position data not available via public Hyperliquid API"

### Option B: Remove features
- Remove `ConcentrationFeatures` from `Features` struct optional fields
- Remove NaN padding in `to_vec()`
- Update `count_all()` from 236 to 221
- **Breaking change** — all parquet files, Python readers, algorithm inputs
  need schema update
- Not recommended unless storage/confusion cost is high

## Timeline

- Run assessment after step 04 has been live for 48h
- Make go/no-go decision based on wallet count + OI coverage
- If no-go, update FEATURES.md and close this item

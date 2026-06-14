# Concentration Feature Viability Assessment

## Status: Pending — requires 48h of live data with position tracker enabled

## What

Determine whether the 15 concentration features (Herfindahl, Gini, top-N share, etc.)
produce statistically meaningful values, given that they depend entirely on the number
of wallets discovered/tracked by the PositionTracker.

## Prerequisites

All code is already implemented (nan_wiring steps 01-04). To run the assessment:

```toml
# config/ing.toml — uncomment and enable:
[position_tracker]
enabled = true
poll_interval_secs = 60
max_concurrent_requests = 10
discover_from_trades = true
max_tracked_wallets = 50
# initial_wallets = ["0x...", "0x..."]  # optional seed wallets
```

Restart the ingestor and let it run for 48h+.

## Diagnostics to Check

### 1. Is WsTrade.users populated?

Check ingestor logs within the first 5 minutes:

- `WsTrade.users field IS populated — wallet discovery active` → discovery works
- `WsTrade.users field not populated after 5min` → discovery unavailable, need manual wallets

### 2. How many wallets are tracked?

```bash
# After 48h, check parquet for non-NaN concentration columns:
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/latest/')
cols = [c for c in df.columns if 'concentration' in c or 'herfindahl' in c or 'gini' in c]
print(df[cols].notna().mean())
print(df[cols].describe())
"
```

### 3. OI coverage

Compare tracked position value vs total OI from context features:

```python
# Sum of position values from tracked wallets / total OI
# This is logged by the position tracker consumer task
```

## Decision Matrix

| Wallets Tracked | OI Coverage | Action |
|----------------|-------------|--------|
| 50+ | >20% | Features viable — no changes needed |
| 20-50 | 5-20% | Features noisy — add disclaimer in FEATURES.md |
| <20 | <5% | Keep as NaN — document as unavailable |

## If Viable (no code changes)

Verify values are in reasonable ranges:

| Feature | Expected Range |
|---------|---------------|
| top5_concentration | 0.05 - 0.50 |
| top10_concentration | 0.10 - 0.60 |
| herfindahl_index | 0.01 - 0.25 |
| gini_coefficient | 0.30 - 0.80 |

## If Not Viable

Add to `FEATURES.md` under the concentration section:

> **Note:** Concentration features require per-account position data. Hyperliquid's
> public API does not expose aggregate position distribution. These features are
> populated only when the position tracker discovers enough wallets via trade stream
> addresses. If wallet discovery is unavailable (WsTrade.users not populated),
> these features remain NaN.

No schema change — the 236-feature vector keeps NaN padding for concentration.
This avoids breaking all downstream consumers (parquet readers, algorithms, ML).

## If WsTrade.users Is Never Populated

Fallback options:

1. **Manual wallet curation** — source whale addresses from Hyperliquid leaderboard
   or on-chain explorers, add to `initial_wallets` in config
2. **Accept NaN** — concentration features are genuinely unavailable without
   per-account data; document and move on
3. **Alternative signal** — use aggregate OI changes (already in context features)
   as a weaker proxy for concentration dynamics

## Timeline

- Enable tracker: now
- First check: after 24h (wallet count, users field status)
- Final decision: after 48h (OI coverage, feature quality)

# Q1.1 — Fix K2: Populate 56 Dead Features

**Phase**: Q1 — Foundation & Data Quality
**Priority**: 2 (after data accumulation)
**Status**: IN PROGRESS
**Effort**: ~8h
**Depends on**: nan_wiring docs, Hyperliquid API investigation

---

## Objective

Reduce NaN rate on all 191 ingestor features to <5%, unlocking 56 currently dead features across whale flow, liquidation risk, concentration, GMM regime, and heatmap categories.

## Context

56 of 239 features produce 100% NaN because their upstream data sources (exchange position/liquidation APIs) are not wired into the ingestor. This affects:
- **Quant path**: 23.5% of the feature vector is dead weight. Alpha screening (Q2.3) can only test 135/191 active features.
- **PhD path**: Incomplete feature vector undermines reproducibility claims.
- **Downstream**: GMM regime (8 features) and heatmap (8 features) auto-resolve once their upstream inputs (whale, liquidation, concentration) are populated.

| Category | Dead Count | Missing Data Source |
|----------|-----------|---------------------|
| Whale flow | 12 | Large-position tracking feed |
| Liquidation risk | 13 | Liquidation level / open interest API |
| Concentration | 15 | Position distribution data |
| GMM regime | 8 | Requires whale+concentration |
| Heatmap | 8 | Requires liquidation levels |

## Prerequisites

- K1 (Docker volume mount) — FIXED
- Hyperliquid API documentation reviewed for position/liquidation endpoints

## Scope

**In scope**:
- Investigate Hyperliquid API for position data, open interest, liquidation levels
- Wire available endpoints into the Rust ingestor
- Implement zero-cost heuristic for whale flow (trade-size classification, >$100K threshold) per `docs/nan_wiring/01_whale_flow_trade_classification.md`
- Mark genuinely unavailable features as deprecated

**Out of scope**:
- Adding new feature categories beyond the existing 56
- ML model retraining (deferred to Q2)
- Feature engineering on newly populated features

## Steps

1. Audit Hyperliquid API for available endpoints:
   - `/info` — open interest, funding rates (likely available)
   - `/clearinghouseState` — user positions (requires auth, may not provide aggregate data)
   - `/liquidations` — historical liquidation events
2. Implement whale flow heuristic in Rust (`rust/ing-features/`):
   - Classify trades > $100K USD as whale trades
   - Compute 12 whale flow features from existing trade stream
3. Implement concentration features from available position data:
   - If aggregate OI available: derive 15 concentration metrics
   - If not: mark as deprecated, remove NaN padding
4. Wire liquidation data if endpoint exists:
   - 13 liquidation risk features from event stream
   - 8 heatmap features from liquidation level density
5. Verify GMM regime features auto-populate (8 features depend on whale+concentration)
6. Update `Features::names_all()` and `count_all()` if any features are deprecated

## Acceptance Criteria

- [ ] `df.isna().all().sum() == 0` on a fresh 1-hour parquet file (zero fully-dead columns)
- [ ] NaN rate per feature < 5% across all 191 features (excluding warmup rows)
- [ ] Whale flow features show non-zero variance: `df[whale_cols].std().min() > 0.001`
- [ ] K3 (`regime_accumulation_score`) resolves to non-constant: `std > 0.01`
- [ ] No regression in existing 135 active features (values unchanged)
- [ ] `nat test` passes (Rust unit tests)
- [ ] Feature count in Parquet schema matches `Features::count_all()`

## Testing / Verification

```bash
# 1. Rust unit tests pass
nat test

# 2. Check dead feature count on fresh data
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/$(date +%Y-%m-%d)/*.parquet')
dead = df.isna().all().sum()
print(f'Dead features: {dead}')
assert dead == 0, f'{dead} features still dead'
"

# 3. Check NaN rate per feature
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/$(date +%Y-%m-%d)/*.parquet')
nan_rate = df.isna().mean()
bad = nan_rate[nan_rate > 0.05]
print(f'Features with >5% NaN: {len(bad)}')
for name, rate in bad.items():
    print(f'  {name}: {rate:.1%}')
assert len(bad) == 0
"

# 4. Verify regime_accumulation_score is non-constant
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/$(date +%Y-%m-%d)/*.parquet')
std = df['regime_accumulation_score'].std()
print(f'regime_accumulation_score std: {std:.6f}')
assert std > 0.01, 'Still constant'
"

# 5. Schema consistency
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/$(date +%Y-%m-%d)/*.parquet')
print(f'Columns: {len(df.columns)}')
"
```

## Key Files

- `rust/ing-features/src/whale_flow.rs` — whale flow feature computation
- `rust/ing-features/src/liquidation.rs` — liquidation risk features
- `rust/ing-features/src/concentration.rs` — concentration features
- `rust/ing/src/main.rs` — data source wiring
- `rust/ing-types/src/lib.rs` — feature struct definitions
- `docs/nan_wiring/01_whale_flow_trade_classification.md` — whale heuristic spec
- `docs/korrektur_tasks.md` — K2 original description

## References

- K2 in `docs/korrektur_tasks.md`
- Whale flow wiring: `docs/nan_wiring/01_whale_flow_trade_classification.md`
- Position tracker: `docs/nan_wiring/03_spawn_tracker.md`

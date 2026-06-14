# Q2.3 — Full Alpha Screen with FDR Correction

**Phase**: Q2 — Validation & Signal Strengthening
**Priority**: 2 (ROADMAP Step 1 — critical path)
**Status**: NOT STARTED
**Effort**: ~8h
**Depends on**: Q1.1 (clean features), Q1.2 (sufficient data)

---

## Objective

Screen all 191 features across 3 symbols, 4 timeframes, and 3 forward horizons for predictive power, applying Benjamini-Hochberg FDR correction. Output a ranked feature report that determines whether the quant path is viable.

## Context

This is ROADMAP Step 1 — the critical path. Everything downstream (combination, sizing, validation, paper trading, live) depends on finding features that survive FDR correction. A previous 551-feature scan at 1-day horizon found 0 significant features after FDR. This run uses the full 191-feature set at multiple horizons including shorter ones where microstructure features are expected to be strongest.

## Prerequisites

- Q1.1 (dead features fixed — testing all 191, not just 135)
- Q1.2 (7+ days data for meaningful IC windows)
- `scripts/alpha/screener.py` exists or needs to be created per ROADMAP spec

## Scope

**In scope**:
- Per-feature Spearman IC with forward returns at horizons {1h, 4h, 1d}
- Per-feature at timeframes {15min, 1h, 4h, daily bars}
- Per-symbol screening (BTC, ETH, SOL independently)
- Benjamini-Hochberg FDR correction (q=0.05) across all t-statistics
- IC mean, IC std, information ratio, IC autocorrelation, turnover, breakeven cost
- Nonlinear transforms if raw screen fails: rank, log, quantile
- Feature interaction tests for top pairs if needed

**Out of scope**:
- Feature combination (Step 2 — task Q2.4)
- ML models (only linear IC metrics)
- New feature engineering

## Steps

1. Implement `scripts/alpha/screener.py` per ROADMAP Step 1 spec:
   - Load aggregated bars from Parquet
   - Compute forward returns: `r(t,h) = price(t+h)/price(t) - 1`
   - For each feature × timeframe × horizon: rolling 7-day Spearman IC
   - Compute: IC mean, IC std, IR, IC autocorrelation, turnover, breakeven cost
   - t-statistic per feature
2. Apply BH FDR correction across all t-statistics
3. Generate `reports/alpha_screen.json`:
   ```json
   {
     "symbol": "BTC",
     "features": [
       {
         "name": "raw_spread_bps_mean",
         "timeframe": "15min",
         "horizon": "1h",
         "ic_mean": 0.139,
         "ic_std": 0.045,
         "ir": 3.08,
         "t_stat": 4.2,
         "p_adj": 0.003,
         "breakeven_bps": 8.5,
         "turnover": 0.34
       }
     ]
   }
   ```
4. Evaluate Gate G1
5. If G1 fails: retry with nonlinear transforms and lagged features
6. If G1 still fails: document and flag — PhD path becomes primary

## Acceptance Criteria

### Gate G1 — at least ONE must pass:
- [ ] 5+ features with adjusted p < 0.05 AND |IC_mean| > 0.015 AND breakeven > 2 bps
- [ ] 3+ features with adjusted p < 0.01 AND |IC_mean| > 0.025
- [ ] 1+ feature with |IC_mean| > 0.05

### Data quality:
- [ ] All 191 features screened (not just 135 active)
- [ ] All 3 symbols × 4 timeframes × 3 horizons = 36 runs completed
- [ ] `reports/alpha_screen.json` written with complete results
- [ ] FDR correction applied (adjusted p-values, not raw)
- [ ] Breakeven cost computed using `config/costs.toml` values (unified cost model)

### If G1 fails:
- [ ] Nonlinear transforms tested (rank, log, quantile)
- [ ] Lagged features tested (t-1, t-2)
- [ ] Top feature interactions tested
- [ ] Failure documented with specific reasons

## Testing / Verification

```bash
# 1. Run screener
python3 scripts/alpha/screener.py --symbols BTC ETH SOL --output reports/alpha_screen.json

# 2. Validate output
python3 -c "
import json
with open('reports/alpha_screen.json') as f:
    data = json.load(f)
for sym in ['BTC', 'ETH', 'SOL']:
    features = [f for f in data if f.get('symbol') == sym]
    print(f'{sym}: {len(features)} features screened')
    sig = [f for f in features if f['p_adj'] < 0.05]
    print(f'  Significant after FDR: {len(sig)}')
    for f in sig[:5]:
        print(f'    {f[\"name\"]} IC={f[\"ic_mean\"]:.3f} p={f[\"p_adj\"]:.4f}')
"

# 3. Gate G1 check
python3 -c "
import json
with open('reports/alpha_screen.json') as f:
    data = json.load(f)
g1a = [f for f in data if f['p_adj']<0.05 and abs(f['ic_mean'])>0.015 and f['breakeven_bps']>2]
g1b = [f for f in data if f['p_adj']<0.01 and abs(f['ic_mean'])>0.025]
g1c = [f for f in data if abs(f['ic_mean'])>0.05]
print(f'G1a (5+ sig, IC>0.015, BE>2bps): {len(g1a)} features')
print(f'G1b (3+ sig, IC>0.025): {len(g1b)} features')
print(f'G1c (1+ IC>0.05): {len(g1c)} features')
passed = len(g1a)>=5 or len(g1b)>=3 or len(g1c)>=1
print(f'G1 PASS: {passed}')
"
```

## Key Files

- `scripts/alpha/screener.py` — alpha screening implementation (new or existing)
- `reports/alpha_screen.json` — output
- `config/costs.toml` — cost parameters for breakeven calculation
- `docs/research/ROADMAP.md` — Step 1 specification

## References

- ROADMAP Step 1: `docs/research/ROADMAP.md` lines 54–96
- Prior 551-feature scan: `docs/tasks_assigned_12_6_26/situation_analysis.md` §II
- Existing IC scan: `docs/research/new/9_6/full_ic_scan_report.md`

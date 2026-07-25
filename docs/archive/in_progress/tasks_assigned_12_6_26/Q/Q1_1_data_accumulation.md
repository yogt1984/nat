# Q1.2 — Accumulate 7+ Days Continuous Data

**Phase**: Q1 — Foundation & Data Quality
**Priority**: 1 (highest — blocks both paths)
**Status**: IN PROGRESS (target: Jun 17, 2026)
**Effort**: Passive (pipeline running)
**Depends on**: K1 fixed, K5 watchdog running

---

## Objective

Collect 7+ consecutive days of clean, gap-free data at 100ms resolution across BTC, ETH, SOL to enable statistically meaningful revalidation of the hierarchical combiner and alpha screening.

## Context

The hierarchical combiner was evaluated on only 2 days of data (~576 bars at 5min). Results are promising (IC +0.18 to +0.36) but carry wide confidence intervals. The monotonically increasing IC across folds is suspicious and may reflect trending market bias rather than real signal.

7+ days provides:
- 2000+ bars per symbol (vs ~288/day)
- Enough walk-forward folds for meaningful OOS evaluation
- Spans at least one weekend (different market microstructure)
- Multiple regime transitions for robustness testing

## Prerequisites

- K1 (Docker volume mount) — FIXED
- K5 (watchdog health check) — FIXED
- Ingestor running on su-35 (separate machine)

## Scope

**In scope**:
- Monitor pipeline health daily
- Validate data quality on accumulated files
- Track K4 (WebSocket gaps) rate

**Out of scope**:
- Feature engineering changes during accumulation (risk breaking schema)
- Any ingestor code changes that require restart

## Steps

1. Verify ingestor is running: `nat status` shows `Last write: <5 min ago`
2. Set up daily validation cron or manual check:
   - Count parquet files per day per symbol
   - Check file sizes (>100 MB/day expected)
   - Verify no all-NaN columns appear mid-stream
3. Track WebSocket gap rate (K4): target < 5 gaps >10s per hour
4. Do NOT restart or modify the ingestor during this period
5. On Jun 17: validate 7-day dataset completeness

## Acceptance Criteria

- [ ] 7 consecutive days of data: `data/features/2026-06-11/` through `data/features/2026-06-17/` all present
- [ ] Each day has 20+ hourly parquet files per symbol (≥20h coverage)
- [ ] Each day's total size > 100 MB
- [ ] No day has > 2 hours of gap
- [ ] All 3 symbols (BTC, ETH, SOL) present in every file
- [ ] Feature count consistent across all files (no schema drift)

## Testing / Verification

```bash
# 1. Check date range coverage
ls -la data/features/ | grep "2026-06-1"

# 2. Validate per-day completeness
for day in $(seq -w 11 17); do
  dir="data/features/2026-06-$day"
  if [ -d "$dir" ]; then
    count=$(ls "$dir"/*.parquet 2>/dev/null | wc -l)
    size=$(du -sh "$dir" | cut -f1)
    echo "Jun $day: $count files, $size"
  else
    echo "Jun $day: MISSING"
  fi
done

# 3. Validate data quality on latest day
python3 -c "
import pandas as pd
from pathlib import Path
import glob

days = sorted(Path('data/features').glob('2026-06-1*'))
print(f'Days available: {len(days)}')
for d in days:
    files = list(d.glob('*.parquet'))
    if files:
        df = pd.read_parquet(files[0])
        dead = df.isna().all().sum()
        print(f'  {d.name}: {len(files)} files, {len(df.columns)} cols, {dead} dead')
"

# 4. Check for gaps > 1 hour within a day
python3 -c "
import pandas as pd
from pathlib import Path

day = sorted(Path('data/features').glob('2026-06-1*'))[-1]
files = sorted(day.glob('*.parquet'))
for f in files:
    df = pd.read_parquet(f, columns=['timestamp'])
    gaps = df['timestamp'].diff().dt.total_seconds()
    big_gaps = gaps[gaps > 3600]
    if len(big_gaps) > 0:
        print(f'{f.name}: {len(big_gaps)} gaps > 1h')
print('Gap check complete')
"
```

## Key Files

- `data/features/YYYY-MM-DD/*.parquet` — output data
- `config/ing.toml` — ingestor configuration
- `rust/ing/src/main.rs` — ingestor main loop

## References

- K5 in `docs/korrektur_tasks.md` — watchdog that prevents zombie gaps
- Hierarchical combiner report: `docs/research/new/10_6/hierarchical_combiner_report.md`

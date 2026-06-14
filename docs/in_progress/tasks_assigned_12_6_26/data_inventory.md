# Data Inventory & Accumulation Status

**Date:** 2026-06-12
**Total size:** 9.4 GB across `data/`
**Resolution:** 100ms (10 Hz), 36,000 rows/hour/symbol
**Symbols:** BTC, ETH, SOL
**Feature vector:** 236 columns per row (154 live, 82 NaN-padded)

---

## 1. Feature Data Timeline

**Directory:** `data/features/` (8.4 GB, 671 parquet files across 34 date directories)

### Daily Breakdown

| Date | Files | Size | Quality | Notes |
|------|-------|------|---------|-------|
| 2026-04-19 | 2 | 3 MB | Partial | Early collection, sparse |
| 2026-04-20 | 1 | 5 MB | Partial | Sparse |
| 2026-04-24 | 8 | 113 MB | OK | First full-ish day |
| 2026-04-25 | 8 | 181 MB | OK | Pipeline state: DONE |
| — | — | — | **GAP** | **Apr 26 - May 5 (10 days missing, K6)** |
| 2026-05-06 | 10 | 119 MB | OK | Collection resumes |
| 2026-05-07 | 12 | 131 MB | OK | OOS window starts |
| 2026-05-08 | 16 | 173 MB | OK | |
| — | — | — | **GAP** | **May 9 missing** |
| 2026-05-10 | 8 | 209 MB | OK | |
| 2026-05-11 | 21 | 481 MB | Good | First high-volume day |
| 2026-05-12 | 16 | 275 MB | Good | Also has -clean variant (96 MB) |
| — | — | — | **GAP** | **May 13 missing** |
| 2026-05-14 | 10 | 156 MB | OK | |
| 2026-05-15 | 5 | 125 MB | OK | |
| — | — | — | **GAP** | **May 16 missing** |
| 2026-05-17 | 3 | 8 MB | Partial | Very sparse |
| 2026-05-18 | 8 | 219 MB | Good | |
| 2026-05-19 | 21 | 585 MB | Good | IC scan reference day 1 |
| 2026-05-20 | 24 | 633 MB | Good | IC scan reference day 2 |
| 2026-05-21 | 21 | 590 MB | Good | IC scan reference day 3 |
| 2026-05-22 | 22 | 548 MB | Good | |
| 2026-05-23 | 25 | 670 MB | Good | OOS window ends. Peak day. |
| 2026-05-24 | 19 | 472 MB | Good | |
| 2026-05-25 | 22 | 680 MB | Good | Largest single day |
| 2026-05-26 | 14 | 405 MB | Good | |
| 2026-05-27 | 8 | 296 MB | OK | |
| 2026-05-28 | 13 | 434 MB | Good | |
| 2026-05-29 | 15 | 457 MB | Good | Last clean pre-gap day |
| — | — | — | **GAP** | **May 30 missing** |
| 2026-05-31 | 125 | 13 MB | **Bad** | Tiny fragments (K6: ~100KB avg) |
| 2026-06-01 | 57 | 6 MB | **Bad** | Tiny fragments |
| 2026-06-02 | 83 | 10 MB | **Bad** | Tiny fragments |
| 2026-06-03 | 60 | 140 MB | Partial | Recovery begins |
| 2026-06-04 | 11 | 505 MB | Good | |
| — | — | — | **GAP** | **Jun 5-9 zombie process (K5, FIXED)** |
| 2026-06-10 | 9 | 216 MB | Good | Watchdog added |
| 2026-06-11 | 10 | 372 MB | Good | Accumulation target day 1 |
| 2026-06-12 | 5 | 202 MB | In progress | Today (13:10 CEST) |

### Coverage Summary

| Metric | Value |
|--------|-------|
| Calendar range | Apr 19 - Jun 12 (54 calendar days) |
| Days with data | 34 |
| Days missing | 20 (~37%) |
| Days with good data (>200 MB) | 22 |
| Days with bad/partial data | 6 (Apr 19-20, May 17, May 31 - Jun 2) |
| Longest clean streak | May 18-29 (12 consecutive days) |
| Current streak | Jun 10-12 (3 days, targeting 7 by Jun 17) |

---

## 2. Accumulation Target

**Task:** `Q1_1_data_accumulation.md`
**Deadline:** Jun 17, 2026 (5 days from now)

**Acceptance criteria:**
- 7 consecutive clean days (Jun 11-17)
- 20+ hourly parquet files per day
- Each day >100 MB total size
- No day has >2 hours of gap
- All 3 symbols in every file
- Feature count consistent (no schema drift)

**Current progress:**

| Day | Date | Status | Size |
|-----|------|--------|------|
| 1 | Jun 11 | Done | 372 MB, 10 files |
| 2 | Jun 12 | In progress | 202 MB, 5 files (half day) |
| 3 | Jun 13 | Pending | — |
| 4 | Jun 14 | Pending | — |
| 5 | Jun 15 | Pending | — |
| 6 | Jun 16 | Pending | — |
| 7 | Jun 17 | Pending | — |

**Risk:** Zombie process (K5) or WebSocket outage could reset the streak. Watchdog now active.

---

## 3. Data Sufficiency for Key Analyses

### ML Training Requirements

From `docs/research/new/ml_specs/DATA_REQUIREMENTS.md`:

| Data Volume | 5-min Bars | Days | Models Safe |
|-------------|------------|------|-------------|
| <2,000 bars | <7 days | — | None |
| 2,000-4,000 | 7-14 days | — | LogReg only |
| 4,000-8,000 | 14-28 days | — | LogReg + LightGBM |
| >8,000 bars | >28 days | — | All (KNN, stacking, HMM) |

**Current clean data:** ~22 good days (non-contiguous) = ~6,300 bars at 5min = **LogReg + LightGBM safe**.

### Key Analysis Thresholds

| Analysis | Data Needed | Current | Status |
|----------|-------------|---------|--------|
| Hierarchical combiner revalidation | 7+ consecutive days | 3 days (Jun 10-12) | Waiting for Jun 17 |
| Walk-forward OOS (4-fold) | 4x 500 bars = 2,000+ | ~6,300 bars | Sufficient |
| OOS30 validation | 30 dates | 22 good dates | **Insufficient** |
| Convolver SVD training | 100+ events per type | Unknown | Need event count |
| Deferred HMM trigger | 60+ days | 22 good days | **Insufficient** |
| Stacking ensemble trigger | 4+ deployed ML algos | 1 (mean_reversion) | **Insufficient** |

### Signal Stability Requirements

| Signal | Minimum Window | Purpose |
|--------|----------------|---------|
| IC stability | 30 days rolling | Detect decay vs noise |
| ent_book_shape gating | 4+ weeks | Confirm +22% IC lift persists |
| Funding rate cycles | 7+ days (8h settlement) | Full funding cycle coverage |
| Cross-symbol correlation | 14+ days | Stable correlation estimates |

---

## 4. Data Rate & Schema

### Emission Profile

| Parameter | Value | Source |
|-----------|-------|--------|
| Emission interval | 100ms | `config/ing.toml` |
| Rows per second per symbol | 10 | Derived |
| Rows per hour per symbol | 36,000 | `scripts/data/catalog.py` |
| Rows per day per symbol | 864,000 | Derived (24h) |
| Rows per day all symbols | 2,592,000 | 3 symbols |
| Expected daily size | ~500-680 MB | Observed May 19-29 |
| Parquet row group size | 10,000 rows | `config/ing.toml` |
| Rotation interval | 1 hour | `config/ing.toml` |
| Compression | zstd | `config/ing.toml` |

### Schema

| Column Type | Count | Description |
|-------------|-------|-------------|
| Metadata | 3 | timestamp_ns, symbol, sequence_id |
| Base features | 154 | 14 categories, always computed |
| Optional features | 82 | 7 categories, NaN-padded |
| **Total columns** | **239** | |

File naming: `YYYYMMDD_HHMMSS.parquet` (e.g., `20260522_180000.parquet`)

---

## 5. Non-Feature Data

| Directory | Size | Files | Content | Span |
|-----------|------|-------|---------|------|
| `data/features/` | 8.4 GB | 671 parquet | Tick features at 100ms | Apr 19 - Jun 12 |
| `data/paper_trades/` | 1.2 MB | 58 JSON | Paper trade P&L per date/symbol | May-Jun |
| `data/paper_trades_surprise/` | 1.2 MB | 57 JSON | Surprise algo paper trades | May-Jun |
| `data/trades/` | 9.2 MB | parquet | Execution records | Jun 10-12 only |
| `data/candles/` | 904 KB | 6 parquet | OHLCV candles (1m, 15m per symbol) | — |
| `data/macro/` | 64 KB | 3 parquet | Daily macro indicators per symbol | 365 days |
| `data/agent/` | 3.1 MB | JSON | Micro agent state, 339 hypotheses | — |
| `data/agent_mf/` | 120 KB | JSON | MF agent state + hypotheses | — |
| `data/agent_macro/` | 12 KB | JSON | Macro agent state | — |
| `data/research/` | 96 KB | 22 JSON | Research cycle records (CYC-*) | — |
| `data/quarantine/` | 34 MB | — | Archived/problematic data | — |
| `data/swarm/` | 32 KB | TOML | Swarm optimization configs | — |
| `data/alpha/` | 16 KB | npy + JSON | Alpha signals (position, signal arrays) | — |
| `data/oos_validation/` | 132 KB | — | OOS validation results | — |
| `data/execution/` | 768 KB | 2 JSONL | Execution cycles | May 22-23 |
| `data/it_engine/` | 28 KB | JSON | IT engine state | — |
| `data/tournament/` | 44 KB | — | Tournament tracking | — |
| `data/features_clean/` | 240 KB | — | Cleaned feature variants (7 dates) | Apr-May |
| `nat.db` | 1.2 MB | SQLite | Research store | — |

---

## 6. Data Quality Monitoring

### Health Check Infrastructure

| Tool | File | Purpose |
|------|------|---------|
| Data validator | `scripts/validate_data.py` | Gap detection, NaN ratio, range checks |
| Data catalog | `scripts/data/catalog.py` | Manifest of available data per symbol |
| ML health check | `scripts/ml_health_check.py` | Model staleness, NaN monitoring |
| Sufficiency check | `scripts/check_data_sufficiency.py` | Minimum bar count validation |
| Daemon heartbeat | `scripts/utils/health.py` | Ingestor liveness |

### Quality Thresholds

| Check | Threshold | Current Status |
|-------|-----------|----------------|
| Max gap between records | 5.0s | K4: 10-12 gaps >1s/hr, max 13.4s |
| Min records per hour | 30,000 | Met on good days (36K expected) |
| Max NaN ratio (after 60s warmup) | 1% | 56 features at 100% NaN (K2) |
| Data freshness for trading | 2 hours staleness | OK (live ingestor running) |
| Model age warning | 14 days | — |
| Model age critical | 30 days | — |

### Active Issues

| Issue | Impact | Status | Fix |
|-------|--------|--------|-----|
| K2: 56 dead features | 82 NaN columns, blocks meta_labeling | OPEN | Wire Hyperliquid position API (~2-4h) |
| K3: regime scores constant | 2 features = 0.4429 always | OPEN | Depends on K2 |
| K4: WebSocket gaps | 10-12 gaps >1s/hr | MONITORING | Add Prometheus metrics |
| K6: Historical gaps | 17 days missing, unrecoverable | ACCEPTED | No backfill source |
| .parquet.tmp files | Incomplete writes in May 26 - Jun 12 | Low | Cleanup script needed |

---

## 7. When Do We Have Enough?

### Current vs Required

| Milestone | Required | Available | Gap | ETA |
|-----------|----------|-----------|-----|-----|
| Hierarchical combiner revalidation | 7 consecutive clean days | 3 days (Jun 10-12) | 4 days | **Jun 17** |
| Robust walk-forward (4-fold) | 2,000+ 5-min bars | ~6,300 bars | None | **Now** |
| OOS30 batch validation | 30 good dates | 22 good dates | 8 dates | **~Jun 20** |
| IC stability assessment (30-day rolling) | 30 clean days | 12 days (May 18-29) | 18 days | **~Jul 1** |
| Deferred HMM trigger | 60+ days | 22 days | 38 days | **~Aug 1** |
| Full statistical confidence | 90+ days | 22 days | 68 days | **~Sep 1** |

### Accumulation Rate

At current rate (1 good day per calendar day, assuming no outages):
- **Jun 17:** 7-day streak met, hierarchical combiner revalidation unblocked
- **Jun 20:** 30 total good dates, OOS30 validation feasible
- **Jul 1:** 30-day rolling IC window available
- **Jul 15:** LightGBM/KNN training with full confidence (8,000+ bars)
- **Aug 1:** HMM deferred trigger threshold (60 days)

**Critical dependency:** Ingestor uptime. K5 (zombie process) caused a 6-day outage. Watchdog is now active but has only 3 days of track record.

# Korrektur Tasks — Data Quality Issues

**Date:** 2026-06-10
**Source:** Data quality audit on fresh ingestor output (20260610_080000.parquet)

---

## K1. Docker Volume Mount — Data Not Persisting to Host

**Severity:** Critical
**Status:** Fixed (2026-06-10)

The ingestor config `data_dir = "../data/features"` resolves to `/data/features/` inside the container (workdir `/app`), bypassing the volume mount at `/app/data`. Data is lost on container restart.

**Fix options (pick one):**
1. Add symlink inside container: `ln -s /app/data /data` in Dockerfile entrypoint
2. Change config to absolute path: `data_dir = "/app/data/features"` and add Docker-specific config overlay
3. Change Dockerfile `WORKDIR` to `/app/rust` so `../data/features` resolves to `/app/data/features`

**Applies to both:** `data_dir = "../data/features"` and `data_dir = "../data/trades"`

**Verification:** `ls /home/onat/nat/data/features/2026-06-10/` shows parquet files on host after container restart.

---

## K2. 56 Dead Features — External Data Feeds Not Connected

**Severity:** Moderate
**Status:** Open

56 of 239 features produce 100% NaN. These depend on external data sources (exchange position/liquidation APIs) not wired into the ingestor.

| Category | Count | Missing Data Source |
|----------|-------|---------------------|
| Whale flow | 12 | Large-position tracking feed |
| Liquidation risk | 13 | Liquidation level / open interest API |
| Concentration | 15 | Position distribution data |
| GMM regime | 8 | Requires whale+concentration as inputs |
| Heatmap | 8 | Requires liquidation levels as inputs |

**Tasks:**
1. Investigate Hyperliquid API for position/liquidation data endpoints
2. If available: add data source to ingestor, wire into feature computation
3. If unavailable: mark features as deprecated in schema, remove NaN padding to reduce file size (~24% column bloat)
4. GMM regime and heatmap will auto-populate once their upstream dependencies (whale, liquidation, concentration) are resolved

**Verification:** `python3 -c "import pandas as pd; df = pd.read_parquet('<latest>.parquet'); print(df.isna().all().sum(), 'dead features')"` should report 0.

---

## K3. `regime_accumulation_score` Constant — Effectively Dead

**Severity:** Low-Moderate
**Status:** Open

Feature produces data (0% NaN) but with std=0.000000 (constant 0.4429). Provides zero information content despite being listed as a slow directional signal (IC 0.11 at 15m in the IC scan).

**Likely cause:** Upstream dependency (accumulation regime detection) requires whale flow or concentration data, which are all-NaN (see K2). The feature falls back to a default constant.

**Tasks:**
1. Trace `regime_accumulation_score` computation in Rust feature engine — confirm it depends on whale/concentration inputs
2. If confirmed: this resolves automatically when K2 is fixed
3. If independent: debug why the computation produces a constant

**Verification:** `df['regime_accumulation_score'].std() > 0.01` on fresh data after fix.

---

## K4. WebSocket Reconnect Gaps — 10-12 Gaps >1s Per Hour

**Severity:** Low
**Status:** Monitoring

Max gap 13.4s observed, with 10-12 gaps >1s per symbol per hour. Median cadence (100ms) is perfect. Gaps are likely WebSocket reconnections or API-side pauses.

**Tasks:**
1. Add gap counter to health metrics (Prometheus): count of gaps >1s, >5s, >10s per hour
2. Set alert threshold: >5 gaps of >10s per hour triggers warning
3. Investigate if gaps cluster at specific times (exchange maintenance windows)
4. Consider adding reconnect backoff logging to distinguish network vs API-side gaps

**Verification:** `docker logs nat-ingestor | grep -i reconnect` to correlate gap times with reconnect events.

---

## K5. 6-Day Data Gap (Jun 4 — Jun 10)

**Severity:** Moderate
**Status:** Resolved (ingestor restarted)

Ingestor process was alive but stopped writing data for ~6 days. Root cause unknown — zombie process with no error logs.

**Tasks:**
1. Add watchdog health check: if no new parquet file written in 30 minutes, restart container automatically
2. Add Prometheus metric for `last_write_timestamp` and alert if stale >15 min
3. Review Docker `restart: unless-stopped` policy — ensure container auto-restarts on crash
4. Investigate if the Jun 4 stoppage coincides with an OOM or disk full event: `docker inspect nat-ingestor --format '{{.State.OOMKilled}}'`

**Verification:** `nat status` shows `Last write: <5 min ago` consistently.

---

## K6. Historical Data Gaps — 17 Days Missing in Apr-May

**Severity:** Low
**Status:** Accepted

6 gaps totaling ~17 missing days across Apr 19 — Jun 4 range. Largest: Apr 25 — May 6 (10 days). Plus anomalous tiny files on May 31 — Jun 2 (125 files, 12.6 MB vs normal ~500 MB/day).

**Tasks:**
1. Investigate May 31 — Jun 2 anomaly: read those tiny parquet files to check if they contain valid data or are corrupt fragments
2. If corrupt: delete and mark dates as unavailable
3. For future analysis: maintain a data catalog (`data/catalog.json`) listing available date ranges per symbol with quality scores
4. Accept historical gaps as unrecoverable — no backfill source for tick-level data

**Verification:** `python3 -c "..."` script that validates each day directory has >100 MB and >10 hourly files.

---

## Priority Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | K1 — Volume mount fix | 10 min | Critical — data loss on restart |
| 2 | K5 — Watchdog/staleness alert | 30 min | Prevents future zombie gaps |
| 3 | K2 — Dead feature investigation | 2-4 hours | Unlocks 56 features + regime model |
| 4 | K3 — Accumulation score debug | 30 min | Likely resolves with K2 |
| 5 | K4 — Gap monitoring | 1 hour | Observability improvement |
| 6 | K6 — Historical gap audit | 30 min | Housekeeping |

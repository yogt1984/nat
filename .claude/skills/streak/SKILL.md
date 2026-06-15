# /streak — Data-continuity check (the binding constraint)

Data continuity is NAT's binding constraint: every clean day appreciates all prior research, and
N_eff / cluster stability need ≥ 7 clean days. This skill reports streak status **read-only**.

**Hard rule (non-negotiable):** zero su-35 contact until the streak completes. This skill reads
local parquet only — **never ssh, never touch su-35, never restart the ingestor.** If you cannot
answer from local files, say so; do not reach for the remote box.

## Step 1: Enumerate coverage

List the feature-data days and their sizes:
```bash
du -sh data/features/2026-* 2>/dev/null | sort
```
A day is **"good"** at > ~200 MB (per `STATE_*.md`); thin days (≪200 MB ≈ a few hours of ticks)
break the streak even though the directory exists.

## Step 2: Compute the streak

From the day list, compute the **current consecutive run of good days** ending at the latest date,
and the count of calendar days with any data vs. missing. Report:
- current consecutive-clean-day streak (N days)
- days remaining to the 7-day target
- any thin days inside the intended window (these are the silent streak-breakers)

## Step 3: Depth checks (optional, still local)

For a per-symbol sufficiency read (bars, NaN rates, fold sizes):
```bash
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features --json
```
For schema / parquet validation:
```bash
nat data schema          # parquet schema scan
nat data validate --hours 24
```

## Step 4: Cloud box (T0b), if configured

If a redundant cloud-ingestor data path is mounted/synced locally, repeat Steps 1–2 against it and
compare gap profiles. Do **not** SSH to su-35 to compare — the cutover comparison happens only after
the streak completes.

## Step 5: Report

Emit a short status block:
- **Streak:** N consecutive good days (latest: YYYY-MM-DD)
- **To target:** X days to 7-day clean
- **Risks:** thin days / gaps / missing dates listed explicitly
- **Verdict:** ON TRACK / AT RISK / BROKEN — and the single next action (which is *waiting*, not
  touching su-35, unless the streak is already broken and the user decides otherwise).

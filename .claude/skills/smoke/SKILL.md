# /smoke — Real-parquet smoke test for one unit (before commit)

Inner-loop step 5 / test-pyramid level 2: after the planted (synthetic) test is green, prove the
unit runs on the **latest real day** with the correct output shape and NaN discipline — *before*
committing. This is the wiring/contract gate, not the correctness gate (planted tests own that).

Usage: `/smoke <unit-name>` — infer the kind (process / algorithm / feature), or ask if ambiguous.

## Step 1: Find the latest good day

```bash
ls -d data/features/2026-* | tail -3      # pick the most recent good (>200 MB) day
```
Use a recent multi-day window if a single day is thin (a few partial-day bars under-tests).

## Step 2: Run the unit on real data

**process** (mirrors `scripts/tests/test_process_real_data.py`):
```bash
nat process run <name> --symbol BTC --start-date <DAY> --end-date <DAY>
```
Assert: exit 0; `summary.error` is null; **≥ 1 finding**; dead / K2 columns appear in
`features_skipped` **with a reason** (`all_nan` / `constant` / `n_valid=…`) — never a crash.

**algorithm** (pattern from `scripts/algorithms/tests/test_real_data.py` — note
`scripts/tests/test_algorithm_smoke.py` does NOT exist; ignore the CLAUDE.md reference):
```bash
nat algorithm evaluate --algorithm <name> --symbol BTC
```
Assert: runs without error; output keys == `alg_features()`; NaN inputs → NaN outputs (no exception).

**feature** (schema contract): confirm `names_all()` length == `to_vec()` length == 236 and the
new column is non-NaN on the latest day (or NaN-padded with a documented reason if its source is
absent):
```bash
nat data schema | grep -c .        # column count sanity
```

## Step 3: Report per-assertion

State each assertion and PASS/FAIL with the observed value (finding count, skipped-with-reason
count, runtime). On any FAIL, stop — do not commit. Smoke failure means the contract/wiring is
wrong even though the planted test passed.

## Step 4: Hand off

If green, the unit is ready for `/ship`. Note the runtime — a smoke that blows the ~120 s budget is
itself a finding (usually a KSG/`max_samples` or full-schema-load issue).

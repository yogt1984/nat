# NAT Platform Improvement Plan

**Date:** 2026-06-05
**Ref:** `suboptimalities__5_6_2026.md`

Work is organized into 4 waves by severity and dependency. Each item references the suboptimality it addresses.

---

## Wave 1 — Testing Foundation (High severity, unblocks everything)

### 1.1 Rust e2e ingestion test [I1]

Create a deterministic integration test that feeds canned JSON messages (book snapshots + trades) through a mock WebSocket into `MarketState` -> `FeatureComputer` -> `ParquetWriter`, then reads back the Parquet and asserts:

- Row count matches expected emission count
- Feature vector length = 236
- No unexpected NaN in base (non-optional) features
- Timestamp ordering is monotonic
- Symbol column is correct

**Where:** `rust/ing/src/tests/integration.rs` (new module)
**Fixtures:** Capture 60s of real WS messages via `scripts/capture_ws_fixture.py` (new), commit as `rust/ing/testdata/ws_fixture.json`. Replay deterministically in test.
**Verification:** `cargo test --package ing -- integration`

### 1.2 Real-market algorithm test fixtures [A1]

Capture and commit a small real-data fixture (1 hour, BTC only, ~36k ticks) that contains naturally occurring:
- At least 1 price jump (>3 sigma move)
- A funding rate swing
- Entropy regime transition
- Order book depth variation

**Where:** `scripts/algorithms/tests/fixtures/btc_1h_real.parquet`
**New tests in** `scripts/algorithms/tests/test_real_data.py`:
- `test_jump_detector_fires_on_real_jump` — verify `alg_jump_detected == 1` within 10 ticks of known jump timestamp
- `test_funding_reversion_fires_on_swing` — verify signal activates during captured funding swing
- `test_surprise_fires_on_regime_shift` — verify `alg_regime_transition_prob > 0.5` near known transition
- `test_optimal_entry_produces_signals` — verify at least 1 entry signal in the hour
- `test_all_winners_no_crash` — all 5 winners run without error on real data

**Verification:** `pytest scripts/algorithms/tests/test_real_data.py -v`

### 1.3 Python e2e pipeline test [A2]

Test that exercises: load Parquet -> AlgorithmRunner (all 5 winners) -> AlgorithmEvaluator -> JSON report. Uses the 1h fixture from 1.2.

**Where:** `scripts/tests/test_e2e_pipeline.py`
**Asserts:**
- Each algorithm produces output with correct column names
- IC report JSON is valid and contains all expected keys
- No algorithm crashes or produces 100% NaN post-warmup
- Evaluation completes in < 60s

**Verification:** `pytest scripts/tests/test_e2e_pipeline.py -v`

---

## Wave 2 — Infrastructure Hardening (Medium-High severity)

### 2.1 WebSocket and Redis test coverage [I2]

**WebSocket tests** (`rust/ing/src/ws/tests.rs`):
- `test_reconnect_on_stale` — mock stream goes silent, verify reconnect fires after `max_reconnect_delay_ms`
- `test_malformed_message` — inject invalid JSON, verify it's logged and skipped (no crash)
- `test_per_channel_staleness` — send trades but no books, verify `last_book_msg_at` diverges from `last_trade_msg_at`

**Redis tests** (`rust/ing/src/redis_publisher/tests.rs`):
- `test_publish_feature_summary` — mock Redis connection, verify correct channel and payload format
- `test_publish_on_redis_down` — verify graceful degradation (warn log, no panic)

**Verification:** `cargo test --package ing -- ws::tests redis`

### 2.2 NaN availability guard [I3]

Add a runtime check in `AlgorithmRunner.run_on_dataframe()` before algorithm execution:

```python
for col in algo.required_columns():
    nan_rate = df[col].isna().mean()
    if nan_rate > 0.95:
        warn(f"{algo.name()}: required column '{col}' is {nan_rate:.0%} NaN")
```

Log a warning, don't block execution. This surfaces the problem without changing behavior.

**Where:** `scripts/algorithms/runner.py` in `run_on_dataframe()`
**Verification:** Unit test with a DataFrame where one required column is all-NaN; verify warning is emitted.

### 2.3 Step/batch consistency tightening [A7]

Raise the consistency threshold from r > 0.7 to r > 0.9 for all algorithms except those with documented sequential state resets (regime_gated). For any algorithm that fails at r > 0.9, investigate and fix the divergence source rather than lowering the bar.

**Where:** `scripts/algorithms/tests/test_algorithms.py::TestStepBatchConsistency`
**Verification:** `pytest scripts/algorithms/tests/test_algorithms.py -k step_vs_batch -v`

---

## Wave 3 — Performance and Schema (Medium severity)

### 3.1 Emission budget benchmark [I4]

Add a Rust benchmark that measures feature computation latency for a single symbol under synthetic load:

- Baseline: empty book, no trades
- Normal: 10-level book, 30 trades/sec
- Stress: 10-level book, 300 trades/sec, all optional features enabled

Track p50, p95, p99 latency. Assert p99 < 80ms (80% of 100ms budget, leaving 20ms for Parquet write + channel send).

**Where:** `rust/ing-features/benches/feature_bench.rs` (Criterion.rs)
**Verification:** `cd rust && cargo bench --package ing-features`

### 3.2 Parquet schema versioning [I6]

Add a `schema_version` metadata key to every Parquet file written:

1. In `output/schema.rs`: add `schema_version: u32` to file metadata (key-value in Parquet footer)
2. Increment version when features are added/removed
3. In Python `scripts/data/schema.py`: on load, read `schema_version` from metadata. If missing (legacy file), infer version from column count. Pad missing columns with NaN, drop unknown columns.

**Where:** Rust: `rust/ing/src/output/schema.rs`, `rust/ing/src/output/writer.rs`. Python: `scripts/data/schema.py`
**Verification:** Write a Parquet with version N, add a feature (version N+1), verify Python loader reads both without error.

### 3.3 Research/production test bridge [C1]

Create a seeded data snapshot mechanism:

1. `nat test snapshot` — captures 1 hour of live data into `data/test_snapshots/latest.parquet`
2. `nat test regression` — runs all 5 winner algorithms on the latest snapshot, saves IC/signal JSON to `data/test_snapshots/latest_results.json`
3. On next run, compares current results against saved baseline. Flags if any algorithm's IC drops by > 0.01 or signal count changes by > 20%.

Not CI-friendly (requires data), but provides a repeatable regression gate before deployment.

**Where:** `scripts/test_regression.py` (new), `make/test.mk` additions
**Verification:** `nat test snapshot && nat test regression`

---

## Wave 4 — Signal Quality and Architecture (Medium to Low severity)

### 4.1 Cross-algorithm ensemble layer [A4]

Add an `Ensemble` class that consumes multiple algorithm outputs and produces combined signals:

```python
class Ensemble:
    def __init__(self, algorithms: list[str], method: str = "equal_weight"):
        ...
    def combine(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ...
```

Methods: `equal_weight` (mean of z-scored signals), `ic_weight` (weight by trailing IC), `regime_switch` (select algorithm based on regime column).

**Where:** `scripts/algorithms/ensemble.py` (new)
**Config:** `config/algorithms.toml` under `[ensemble]`
**Verification:** Unit test with mock algorithm outputs; e2e test combining jump_detector + optimal_entry on 1h fixture.

### 4.2 Warmup calibration study [A6]

Script that empirically measures warmup stability for each algorithm:

1. Run algorithm on 24h of real data
2. For each feature, compute rolling autocorrelation of output at lag=1 over sliding 100-tick windows
3. Mark warmup as the tick where autocorrelation stabilizes (drops below 0.99 and stays there)
4. Compare empirical warmup to declared warmup; flag discrepancies > 2x

**Where:** `scripts/algorithms/calibrate_warmup.py` (new)
**Output:** `reports/warmup_calibration.json`
**Verification:** `python scripts/algorithms/calibrate_warmup.py --data-dir data/features --symbol BTC`

### 4.3 Bar-level microstructure preservation [A5]

For bar-level algorithms, extend the aggregation to include microstructure summary statistics beyond mean/std/last/sum:

- `_jump_count`: number of ticks where `alg_jump_detected == 1` within bar
- `_max_abs_return`: largest absolute tick return within bar
- `_obi_range`: max - min of `imbalance_qty_l1` within bar
- `_trade_cluster_count`: number of trade bursts (>5 trades in 1s) within bar

**Where:** `scripts/cluster_pipeline/preprocess.py` in `aggregate_bars()`
**Verification:** Verify bar-level algorithms improve IC when given the extra columns (compare on 1h fixture).

### 4.4 CascadeProbability data improvement [A3]

Investigate Hyperliquid's `liquidations` WebSocket channel. If available:

1. Add subscription in `rust/ing/src/ws/client.rs`
2. Track liquidation volume in `MarketState`
3. Expose as `ctx_liquidation_volume` feature
4. Update CascadeProbability to consume real liquidation data instead of price proxy

If not available via API, document as a known limitation and close.

**Where:** Rust WS client + `ing-features/src/context.rs` + `scripts/algorithms/cascade_probability.py`
**Verification:** `cargo test` + algorithm smoke test + IC comparison (proxy vs real).

---

## Deferred (Low severity, opportunistic)

| ID | Item | Trigger | Status |
|----|------|---------|--------|
| I5 | Horizontal scaling | When symbol count > 10 | Design notes below; not needed yet |
| C2 | Perf regression tests | After 3.1 benchmark exists | **Done** — CI bench job added |
| C3 | Config combination tests | When a config-related bug is found | **Done** — 97 parametrized tests; found + fixed surprise_signal min_periods bug |

### I5 — Horizontal Scaling Design Notes

**Current state:** 3 symbols (BTC, ETH, SOL) running as tokio tasks in one process on su-35. Benchmark shows p99 feature computation at ~285µs/tick, well within the 100ms emission budget. No scaling pressure exists.

**Bottleneck when scaling to >10 symbols:** Cross-symbol features (`ing-features/src/cross_symbol.rs`) use `Arc<Mutex<HashMap>>` — contention grows linearly with symbol count. At ~30 symbols the lock becomes hot.

**Scaling strategy when needed:**

1. **Short-term (10-30 symbols):** Replace `Mutex` with `DashMap` (lock-free concurrent map) for cross-symbol state. Per-symbol tokio tasks already parallelize well. Expected capacity: ~30 symbols on current hardware.

2. **Medium-term (30-100 symbols):** Shard by symbol group. Run N ingestor processes, each handling a disjoint symbol set. Cross-symbol features either (a) compute within-shard only (loses cross-group correlation) or (b) use shared-memory ring buffer (e.g., memmap) for price feeds across shards.

3. **Long-term (100+ symbols):** Separate ingestion (WS → MarketState) from feature computation (MarketState → Features). Ingestion runs per-symbol. Feature computation runs on a dedicated machine with all MarketState snapshots replicated via Redis pub/sub (already in place for `FeatureSnapshot`). Parquet writing moves to a dedicated writer process consuming feature vectors from a channel.

**No code changes needed until symbol count exceeds 10.**

---

## Execution Order

```
Wave 1 (1.1, 1.2, 1.3)  — All parallelizable, no dependencies between them
    |
Wave 2 (2.1, 2.2, 2.3)  — 2.2 and 2.3 parallelizable; 2.1 independent
    |
Wave 3 (3.1, 3.2, 3.3)  — 3.1 independent; 3.2 and 3.3 parallelizable
    |
Wave 4 (4.1, 4.2, 4.3, 4.4) — All independent, order by interest
```

Wave 1 items should be completed before moving to Wave 2, since reliable testing infrastructure is prerequisite for validating the fixes in later waves.

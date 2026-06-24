# NAT Platform Suboptimalities

**Date:** 2026-06-05
**Scope:** Ingestion pipeline (Rust) + Algorithm framework (Python)

---

## Ingestion Pipeline (Rust)

### I1. No end-to-end integration test

There is no test that exercises WebSocket -> MarketState -> FeatureComputer -> ParquetWriter as a connected pipeline. Unit tests exist per component, but the handoffs between them are untested. The validation binaries (`validate_api`, `validate_positions`, etc.) require a live Hyperliquid connection and cannot run in CI.

**Impact:** Silent regressions in data flow handoffs go undetected until production.

### I2. WebSocket and Redis infrastructure barely tested

- REST client (`rust/ing/src/rest/client.rs`): 1 test (position size parsing only)
- Redis publisher (`rust/ing/src/redis_publisher.rs`): 1 test (config defaults only)
- No WebSocket reconnection test, no malformed message injection
- No concurrent multi-symbol stress test

**Impact:** Infrastructure failures in reconnection, message parsing, or pub/sub are only caught in production.

### I3. NaN propagation risk to downstream consumers

82 of 236 features (35%) are optional and NaN-padded when their data source is unavailable (whale_flow, liquidation, concentration, regime, GMM, cross-symbol, heatmap). Any downstream algorithm or ML model consuming these columns must handle NaN gracefully, but there is no runtime enforcement that a consumer's `required_columns()` are actually populated before execution.

**Impact:** Algorithms may silently receive all-NaN inputs for extended periods without warning.

### I4. 100ms emission budget is not load-tested

Feature computation + optional algorithm execution + Parquet buffering must complete within the 100ms `emission_interval_ms` per tick per symbol. No benchmark or profiling test exists. During volatile periods with 10k+ trades/min and full optional features enabled, this budget may be exceeded.

**Impact:** Unknown headroom. Potential tick drops or queue backpressure under load.

### I5. Single-process, no horizontal scaling

Each symbol runs as a tokio task in one process. Adding symbols is linear load increase on one machine. Cross-symbol features (`ing-features/src/cross_symbol.rs`) use `Arc<Mutex<>>` shared state, adding contention as symbol count grows.

**Impact:** Scaling beyond ~5-10 symbols requires architectural change.

### I6. No Parquet schema evolution strategy

Adding or removing features changes the Parquet column set. Historical files have different schemas than new files. There is no versioning, migration path, or backward-compatible read layer. Python consumers must handle schema mismatches themselves (missing columns, extra columns, reordering).

**Impact:** Breaking schema changes require manual migration or data discard.

---

## Algorithm Framework (Python)

### A1. All algorithm tests use synthetic data

`conftest.py` generates GBM prices with constant volatility, seeded with `seed=42`. No regime switching, no order book microstructure edge cases (zero depth, crossed books, flash crashes), no whale activity patterns. Algorithms pass smoke tests but are never tested against the behaviors they are designed to detect.

**Impact:** False confidence. An algorithm designed to detect jumps is tested on data that never contains jumps.

### A2. No end-to-end pipeline test

No automated test exercises: real Parquet data -> load -> AlgorithmRunner -> AlgorithmEvaluator -> signal output. The agent integration test (`test_daemon_integration.py`) uses synthetic parquet. The OOS validation (`nat oos30`) requires 30+ days of manually collected data and cannot run in CI.

**Impact:** Full pipeline regressions only caught during manual OOS runs.

### A3. CascadeProbability uses price proxy for liquidation volume

The algorithm notes at `cascade_probability.py:186`: "liquidation volume not available in tick data -- use price threshold as proxy." The primary signal it's designed to detect (liquidation cascades) is approximated through an indirect proxy.

**Impact:** Reduced model expressiveness for its core purpose.

### A4. No cross-algorithm ensemble layer

28 algorithms run independently. No framework for combining signals (weighted average, stacking, regime-conditional switching). The `meta_agent` handles budget and correlation at the hypothesis level, but there is no signal-level combination for live deployed algorithms.

**Impact:** Alpha left on table. Complementary signals (e.g., jump_detector + optimal_entry) are not systematically combined.

### A5. Bar-level algorithms lose intra-bar microstructure

`RegimeStateMachine`, `ChangePointDetector`, `KNNRetrieval`, `RegimeConditionedLGBM` operate on 5-min bars. The 100ms tick data is aggregated via mean/std/last/sum suffixes. Intra-bar order book dynamics, trade clustering, and sub-minute regime shifts are discarded.

**Impact:** These algorithms cannot detect events shorter than their bar timeframe.

### A6. Warmup periods are heuristic, not calibrated

Warmup values range from 50 to 3000 ticks across algorithms. These appear to be chosen by judgment, not by measuring when output stabilizes. Example: JumpDetector uses 100 ticks (10 seconds at 100ms) -- no study confirms bipower variation is stable at this scale for the assets traded.

**Impact:** Potential early-window signal quality issues. Over-conservative warmups waste data; under-conservative warmups emit noisy signals.

### A7. Step/batch consistency is approximate

The test suite accepts correlation > 0.7 between `step()` loop and `run_batch()` output. For `regime_gated`, consistency is skipped entirely. Ratio-based columns are also excluded.

**Impact:** Live tick-by-tick execution via `step()` may differ materially from backtested batch results via `run_batch()`.

---

## Cross-Cutting

### C1. Research-to-production gap in testing

The strongest validation (OOS on real data, hypothesis testing on real data) requires manual data collection and cannot be automated. The automated test suite (CI-friendly) uses only synthetic data. This creates a split where CI catches syntax/contract errors but not behavioral regressions.

### C2. No performance regression testing

Neither Rust feature computation nor Python algorithm execution has benchmark tests that track latency over time. A refactor that doubles computation time would not be caught by any existing test.

### C3. Configuration combination coverage

Only default/example configs are tested. No parametrized tests over realistic config variations (different symbol counts, different algorithm combinations, different `emission_interval_ms` values). Config interactions are untested.

---

## Severity Summary

| ID | Issue | Severity |
|----|-------|----------|
| I1 | No e2e ingestion test | High |
| A1 | Synthetic-only algorithm tests | High |
| A2 | No e2e pipeline test | High |
| I2 | Infrastructure barely tested | Medium-High |
| A4 | No ensemble layer | Medium |
| I3 | NaN propagation risk | Medium |
| I4 | No load benchmarks | Medium |
| I6 | No schema evolution | Medium |
| A7 | Step/batch divergence | Medium |
| C1 | Research/production test gap | Medium |
| A5 | Bar-level info loss | Low-Medium |
| A6 | Heuristic warmups | Low-Medium |
| A3 | Cascade price proxy | Low-Medium |
| I5 | No horizontal scaling | Low |
| C2 | No perf regression tests | Low |
| C3 | Config combinations untested | Low |

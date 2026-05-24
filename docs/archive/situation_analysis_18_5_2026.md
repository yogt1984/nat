# Situation Analysis — 18 May 2026

## Executive Summary

After 4 weeks of data collection (2026-04-19 to 2026-05-18) and systematic experimentation across BTC, ETH, and SOL perpetual futures on Hyperliquid, the research has produced one high-confidence scientific finding and zero deployable trading strategies.

**The core finding**: L1 book imbalance is a genuine, persistent microstructure signal with IC=0.44-0.48 at tick level, replicating across all three symbols and multiple out-of-sample dates. The signal is frequency-localized in the ultra-low band (0.005-0.1 Hz), mean-reverts with OU half-life 5-7s, and is best gated by book shape entropy (`ent_book_shape < P20-P40`).

**The core problem**: The edge is real but too small to trade profitably at any tested horizon. At tick level, the cost-to-edge ratio is ~20x. At medium frequency (5-30min), the signal either vanishes or produces negative PnL on independent non-overlapping trades.

---

## Experiment Registry

### EXP-1: Spannung Grid Search (2026-05-14)

**File**: [`experiment_results/result__2026-05-14__1933.md`](experiment_results/result__2026-05-14__1933.md)

Exhaustive grid search over 1,350 parameter combinations per symbol (3 flow metrics x 3 illiquidity denominators x 6 alpha x 5 beta x 5 horizons).

| Symbol | Best IC | Best Config | Go/No-Go Gate |
|--------|---------|-------------|---------------|
| BTC | 0.479 | imbalance_qty_l1, alpha=0.5s, beta=60s, 5s horizon | PASS (6-8x threshold) |
| ETH | 0.484 | same | PASS |
| SOL | 0.441 | same, alpha=0.5s, 1s horizon | PASS |

**Key result**: Raw `imbalance_qty_l1` dominates — the Spannung EWM formulation hurts IC by 8-10 points. OOS replication on 2026-05-11/10 confirms IC=0.42-0.47. 100% positive windows across 94 non-overlapping 5-min windows.

**Why it doesn't trade**: Per-trade edge = 0.17-0.37 bps vs 7.0 bps taker round-trip cost (20x gap). Every threshold loses money after fees. Regime gating adds +6% IC but is insufficient.

**Verdict**: Signal validated, not tradeable standalone.

---

### EXP-2: Spectral Analysis (2026-05-14)

**File**: [`experiment_results/result__2026-05-14__1945_spectral.md`](experiment_results/result__2026-05-14__1945_spectral.md)

Power spectral density, coherence, and band-filtered IC analysis of L1 imbalance signal.

| Band | Frequency Range | IC (5s fwd) | IR (5s) |
|------|----------------|-------------|---------|
| Ultra-low | 0.005-0.1 Hz | **+0.45** | **4.1** |
| Low | 0.05-0.5 Hz | +0.10 | 1.1 |
| Mid | 0.5-2.0 Hz | ~0 | — |
| High | 2.0-4.5 Hz | ~0 | — |

Brown noise (PSD slope -1.86), Hurst H=0.43, OU half-life 5-7s. Dominant coherence with returns at 0.015 Hz (~68s cycles).

**Key result**: Predictive power is frequency-localized. Bar aggregation fails because it mixes informative ultra-low frequencies with noisy mid/high components. Bandpass filtering preserves the signal.

**Implication**: Kalman filter on slow component could recover latency-lost IC, since ultra-low periods (10-200s) dwarf execution latency.

---

### EXP-3: Cross-Symbol Spectral + Regime Validation (2026-05-15)

**File**: [`experiment_results/result__2026-05-15__1115_spectral_regime_cross_symbol.md`](experiment_results/result__2026-05-15__1115_spectral_regime_cross_symbol.md)

Replicated spectral analysis and regime screening independently on BTC, ETH, SOL.

| Metric | BTC | ETH | SOL |
|--------|-----|-----|-----|
| PSD slope | -1.86 | -1.87 | -1.83 |
| Hurst H | 0.431 | 0.434 | 0.416 |
| OU half-life | 7.3s | 5.3s | **3.3s** |
| Ultra-low IC (5s) | +0.45 | +0.47 | +0.42 |
| Ultra-low IR (5s) | 4.1 | 4.75 | **5.21** |
| Dominant coherence | 68s | 68s | 68s |

**Best single regime factor** (independently on all 3 symbols): `ent_book_shape`

| Symbol | Best Condition | IC (gated) | dIC | Coverage |
|--------|---------------|------------|-----|----------|
| BTC | ent_book_shape < P40 | 0.544 | +0.089 | 40% |
| ETH | ent_book_shape < P20 | 0.557 | +0.091 | 20% |
| SOL | ent_book_shape < P20 | 0.484 | +0.068 | 20% |

**Best multi-factor combos** (Pareto-optimal):

| Symbol | Combo | IC | dIC | Coverage |
|--------|-------|-----|-----|----------|
| BTC | entropy + tick_entropy + regime_score | 0.634 | +0.179 | 7% |
| **ETH** | **vol_1m + toxicity + perm_entropy** | **0.712** | **+0.246** | **1%** |
| SOL | entropy + tick_30s + perm_entropy | 0.655 | +0.239 | 1% |

ETH multi-factor combo at IC=0.712 is the highest IC ever observed in this research.

**Key result**: Complete cross-symbol replication rules out data snooping. Same spectral structure, same dominant regime condition, same frequency-localized IC — structural property of crypto perpetual futures microstructure, not asset-specific. Publication-quality evidence.

**Regime persistence problem**: High-IC regimes (0.60+) last only 1-2s. Tradeable regimes (IC 0.49-0.51) last 5-17s. Fundamental tradeoff.

---

### EXP-4: Walk-Forward Cross-Symbol Validation (2026-05-15)

**File**: [`experiment_results/result__2026-05-15__1050_walk_forward_cross_symbol.md`](experiment_results/result__2026-05-15__1050_walk_forward_cross_symbol.md)

5-fold walk-forward validation at 1min bars across BTC (2467 bars), ETH (472 bars), SOL (472 bars).

| Feature | BTC OOS IC | ETH OOS IC | SOL OOS IC | Sign Consistency |
|---------|-----------|-----------|-----------|-----------------|
| imbalance_qty_l1_last | 0.185 | 0.177 | 0.110 | 80-100% |
| imbalance_qty_l10_last | 0.191 | 0.136 | 0.110 | 80-100% |
| imbalance_depth_weighted_last | 0.191 | 0.135 | 0.110 | 80-100% |

SOL shows striking IS-to-OOS IC increase (0.26 -> 0.52). Liquidity ordering: SOL (0.50+) > ETH (0.32) > BTC (0.27) on price features — less liquid instruments have stronger mean-reversion.

**Key result**: All `_last` imbalance features KEEP verdict on all 3 symbols. `_mean`/`_std` features degrade — confirms spectral explanation (bar averaging mixes informative low-freq with noisy high-freq). Net edge remains negative at 3.5 bps cost for all features on all symbols.

---

### EXP-5: Skeptical Validation Framework (2026-05-12)

**Files**: [`reports/skeptical_validation/validation_report.json`](reports/skeptical_validation/validation_report.json)

79-test skeptical battery on entropy features:

| Category | Survived | Rejected | Inconclusive |
|----------|----------|----------|-------------|
| Total | 28 | 45 | 6 |

Key passes:
- Entropy persistence: ACF(1)=0.77, half-life=69 ticks (SURVIVES)
- Entropy predicts 1-tick returns: KW=9.57, p=0.048, Spearman rho=0.023 (SURVIVES)
- ADF stationarity: stat=-36.0 (SURVIVES)

Key failures:
- Entropy does NOT predict 5-tick returns (p=0.53, REJECTED)
- Entropy does NOT predict 10-tick returns (p=0.95, REJECTED)

**Verdict**: Entropy is a valid regime conditioner at 1-tick horizon only. Predictive power decays rapidly.

---

### EXP-6: Regime Clustering (2026-05-06)

**Files**: [`data/experiment_state.json`](data/experiment_state.json), [`reports/cluster_sweep_results.json`](reports/cluster_sweep_results.json)

| Vector | k | Silhouette | Bootstrap ARI | Q1 Pass | Q2 Pass |
|--------|---|-----------|---------------|---------|---------|
| Entropy (5min) | 2 | 0.436 | 0.984 | Yes | Yes |
| Orderflow (5min) | 4 | 0.300 | 0.730 | Yes | Yes |
| Trend (5min) | 3 | 0.240 | 0.997 | No | Yes |
| Volatility (5min) | 9 | 0.130 | 0.510 | No | No |

Hopkins test H=0.89 (strongly clusterable). 5-state regime model achieves silhouette=0.47, bootstrap ARI=0.67.

**Verdict**: GO — entropy and orderflow vectors produce stable, reproducible clusters.

---

### EXP-7: Autonomous Agent Hypotheses (2026-05-15)

**File**: [`data/agent/hypotheses.json`](data/agent/hypotheses.json)

60 systematic hypotheses generated, 12 successful (~20% hit rate).

**Best replicated hypothesis**: HYP-SYS-e58cf26d
- Claim: `imbalance_qty_l1` gated by `ent_book_shape < P20` predicts 5s returns
- Regime IC: 0.569, Scalp IC: 0.614, dIC: +0.081
- Cost check: avg_ret=0.56 bps at threshold 0.8 — PASS
- Status: **REPLICATED**

---

### EXP-8: Spannung Grid Summary

**Files**: [`reports/spannung_grid.json`](reports/spannung_grid.json), [`reports/spannung/spannung_summary.json`](reports/spannung/spannung_summary.json)

| Symbol | Best IC | Best IR | Combos > 0.05 IC |
|--------|---------|---------|-----------------|
| BTC | 0.395 | 3.57 | 863/1350 (64%) |
| ETH | 0.386 | 3.96 | 956/1350 (71%) |
| SOL | 0.335 | 5.67 | 790/1350 (59%) |

SOL has highest IR (5.67) despite lowest IC — best signal-to-noise ratio.

---

### EXP-9: Discovery Orchestrator Test (2026-05-18)

**Files**: [`data/discovery/test_state.json`](data/discovery/test_state.json), [`reports/discovery_test/cycle_001/`](reports/discovery_test/cycle_001/)

Automated signal sweep found 1 winner from 1 combo tested:
- BTC 300s horizon: edge=0.133, Sharpe=11.95, accuracy=57.7%
- Gross: 0.71 bps, Net (taker): -7.29 bps

**Verdict**: Edge detected but not cost-viable at taker fees.

---

### EXP-10: Skeptical Regression Battery (2026-05-18)

**Files**: [`scripts/skeptical_regression_test.py`](scripts/skeptical_regression_test.py), JSON reports at `/tmp/skeptical_btc_30min.json`, `/tmp/skeptical_btc_5min.json`

10-test battery applied to LightGBM regression signals on BTC.

#### BTC 30min Horizon (18000 rows)

| Test | Verdict | Key Finding |
|------|---------|-------------|
| T1 Permutation | PASS | IC=0.23 beats null (p=0.000) |
| T2 Effective N | **FAIL** | 1.9 independent trades (p=0.74) |
| T3 Block Bootstrap | **FAIL** | Sharpe 5th pctile = -1.02 |
| T4 Feature Ablation | PASS | IC 0.25 -> 0.16 without funding (-33%) |
| T5 Temporal Stability | **FAIL** | Only 1 day with sufficient data |
| T6 Symbol Replication | PASS | ETH IC=0.24, SOL IC=0.35 |
| T7 Non-Overlapping | **FAIL** | 10 trades, gross=-27bp, 50% win rate |
| T8 Regime Split | PASS | IC=0.37 in negative-funding regime |
| T9 Cost Sensitivity | PASS | Breakeven=23bp, 15bp buffer |
| T10 Embargo Walk-Forward | PASS | IC=0.24 with full embargo |

**Overall: REJECT** (6/10 pass, but T2+T7 hard kill — overlapping IC=0.24 does not translate to profitable independent trades)

#### BTC 5min Horizon (3000 rows)

| Test | Verdict | Key Finding |
|------|---------|-------------|
| T1 Permutation | PASS | IC=0.10 beats null (p=0.000) |
| T2 Effective N | **FAIL** | 11 effective trades (p=0.93) |
| T3 Block Bootstrap | **FAIL** | Sharpe 5th pctile = -0.26 |
| T4 Feature Ablation | PASS | IC improves without funding (0.03 -> 0.06) |
| T5 Temporal Stability | **FAIL** | 1/2 days positive |
| T6 Symbol Replication | PASS | ETH IC=0.10, SOL IC=0.08 |
| T7 Non-Overlapping | **FAIL** | 56 trades, gross=+5.5bp, net=-2.5bp |
| T8 Regime Split | **FAIL** | Negative IC in both regimes |
| T9 Cost Sensitivity | **FAIL** | Gross is negative (-1.8bp) |
| T10 Embargo Walk-Forward | **FAIL** | IC collapses to 0.002 |

**Overall: REJECT** (3/10 pass, 7/10 fail)

---

## Consolidated Findings

### What is real

1. **L1 book imbalance predicts short-term returns** (IC=0.44-0.48 at tick level). This is structural — replicates across BTC/ETH/SOL, multiple dates, walk-forward validation, and survives permutation testing.

2. **Predictive power is frequency-localized** in the ultra-low band (0.005-0.1 Hz). Bar aggregation destroys it. Bandpass filtering preserves it.

3. **`ent_book_shape` is the universal best regime gate** — independently #1 on all three symbols, improving IC by 7-9 points.

4. **Cross-symbol spectral structure is universal**: brown noise, Hurst ~0.43, 68s dominant coherence, OU half-life scaling with liquidity (BTC 7.3s > ETH 5.3s > SOL 3.3s).

5. **Medium-frequency regression signals beat permutation null** and replicate across symbols (ETH IC=0.24, SOL IC=0.35 at 30min), but this does not translate to profitable independent trades.

### What is not tradeable

1. **Tick-level signal**: 20x cost-to-edge gap. Would need zero-fee venue or co-located market-making infrastructure.

2. **Bar-level signal**: IC degrades 2-3x with aggregation. Net edge negative at any cost structure tested.

3. **Medium-frequency regression (5-30min)**: Overlapping predictions inflate apparent IC. When forced to make independent non-overlapping trades, edge vanishes or goes negative. Only ~2-56 independent observations in 14 days depending on horizon.

4. **Regime-gated strategies**: High-IC regimes (0.60+) last only 1-2s — too short to trade. Tradeable regimes (5-17s) have insufficient IC improvement.

### What might work (not yet tested)

1. **Maker execution at tick level**: 0bp maker fees on Hyperliquid. If the 0.17-0.37 bps edge survives queue priority and adverse selection, market-making could be viable. Requires FPGA/co-location infrastructure.

2. **Kalman-filtered slow component**: Extract ultra-low frequency signal via Kalman filter, trade at 10-60s horizon. Could bridge the gap between tick-level IC and tradeable holding period.

3. **Negative-funding regime only at 30min**: IC=0.37 when funding rate is negative (negative carry = short squeeze risk = more predictable returns). Needs more data to confirm with sufficient independent trades.

4. **Multi-signal portfolio**: Combine imbalance, entropy, regime state, and medium-freq features in a unified model with proper walk-forward and embargo. The individual features are validated — the combination is not yet tested at portfolio level.

5. **More data**: 14 days is insufficient for medium-frequency strategies. 3-6 months would yield 100+ independent 30min observations, making the skeptical battery results statistically meaningful.

---

## Infrastructure Status

| Component | Status | Notes |
|-----------|--------|-------|
| Rust ingestor | Running on su-35 | 209 features, 10Hz, 3 symbols, hourly rotation |
| Data collection | 445K rows loaded (memory-limited) | ~14 days, 7 corrupted parquet files |
| Feature pipeline | Complete | 19 categories, 209 features |
| Hypothesis agent | Operational | 60 hypotheses generated, 12 successful, 1 replicated |
| Skeptical battery | Complete | 10-test battery, 36 unit tests passing |
| Alpha pipeline | Built, not yet run to completion | 9-step pipeline with quality gates |
| Discovery orchestrator | Built, tested once | Automated (symbol, horizon) sweep |

---

## Recommendation

**Do not deploy.** No strategy passes the skeptical battery at any tested horizon.

**Continue collecting data.** The ingestor should run for 2-3 more months to accumulate enough independent observations for medium-frequency analysis. The tick-level findings are scientifically valid and worth publishing, but translating them to a profitable strategy requires either (a) maker execution infrastructure or (b) significantly more data at longer horizons.

**Next experiments to prioritize:**
1. Horizon sweep at 10min/15min/20min with the skeptical battery once 30+ days of data exist
2. Kalman-filtered ultra-low band signal at 30-60s holding period
3. Maker execution simulation with realistic queue modeling
4. Multi-signal stacking with proper embargo walk-forward

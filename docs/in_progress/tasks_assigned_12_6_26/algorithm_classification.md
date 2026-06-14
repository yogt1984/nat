# Algorithm Classification — Logic × Time Horizon

**Date:** 2026-06-12
**Scope:** all 26 registered algorithms (`scripts/algorithms/`, verified against `@register`) + the ensemble combiner + 3f_liquidity (alpha pipeline) + the 12 literature candidates
**Companions:** `algorithms_report.md` (per-algorithm detail), `algorithm_candidates_literature.md` (candidate specs), `plan.md` §3.3 (agent horizon partition)

---

## 1. The Two Axes

### Logic family (what the signal exploits)

| Family | Exploits |
|--------|----------|
| **Order-flow / imbalance** | Book pressure predicts the next price move (queue dynamics, OFI, microprice) |
| **Event-driven** | Discrete events (jumps, change points, cascades, activity bursts) create predictable aftermath |
| **Mean-reversion** | Price displacement from fair value decays (impact relaxation, statistical reversion) |
| **Momentum / continuation** | Established moves persist (trend, autocorrelation) |
| **Carry / positioning** | The price of leverage and crowding (funding, OI, premium) forecasts flows |
| **Information / toxicity** | Information asymmetry measures (entropy, VPIN, spread decomposition) condition or direct |
| **Latent state / regime** | A hidden market state, once identified, changes the conditional return distribution |
| **Composite / execution** | Combination, sizing, or execution-layer logic on top of other signals |

### Time horizon (intrinsic vs tested)

Bands follow the agent partition: **Tick** (sub-s–min, micro agent), **MF** (min–1h), **Macro** (1h–24h), **Daily** (1–7d), **Cross-cutting** (gates/sizing/execution, no own horizon).

Every entry carries two horizons: **intrinsic** (the natural decay/holding scale of the logic — for imbalance signals the measured IC half-life is ~30s, `algorithms_report.md` §14) and **tested** (the evaluation horizon — 100min standard for the sweep, 50min for 3f, 5h for hierarchical). The `match` column flags where the test clock disagrees with the logic clock.

**Status tags:** `T1`/`T2` deployable tiers · `PRELIM` promising, data-starved · `ML✓`/`ML✗` deployed/failed ML · `T3✗` no edge after costs · `INFRA` non-trading · `CAND-n` literature candidate (priority rank from `algorithm_candidates_literature.md`)

---

## 2. Master Table

### Implemented — trading algorithms

| Algorithm | Status | Primary logic | Secondary | Intrinsic | Tested | Match | Agent |
|-----------|--------|---------------|-----------|-----------|--------|-------|-------|
| jump_detector | T1 | Event-driven | Mean-reversion (post-jump) | MF (min) | 100min | ✓ | micro/MF |
| 3f_liquidity | T1 | Composite | Order-flow (depth/VWAP) | MF (15–50min) | 50min | ✓ | MF |
| funding_reversion | T1 | Carry / positioning | Mean-reversion | Macro (1–8h) | 100min | ✓ | macro |
| optimal_entry | T1 | Order-flow / imbalance | Latent state (Kalman+SPRT) | Tick→MF | 100min | ~ (SPRT accumulation slows it) | micro |
| surprise_signal | T2 (ETH/SOL only) | Information / toxicity | Latent state (regime transition) | Tick→MF | 100min | ~ | micro |
| hierarchical_combiner | PRELIM | Composite | All three layers | MF→Macro (3–5h) | 5h | ✓ | MF/macro |
| mean_reversion_detector | ML✓ | Mean-reversion | ML (LightGBM), entropy gate | MF (100min by construction) | 100min | ✓ | MF |
| momentum_continuation | ML✗ (overfit) | Momentum | ML (LogReg), entropy gate | MF (100min by construction) | 100min | ✓ | MF |
| meta_labeling | ML✗ (K2 inputs NaN) | Latent state / regime | Meta-filter (ML) | follows base signal | 100min | n/a | cross |
| regime_conditioned_lgbm | ML✗ (regime samples) | Latent state / regime | ML (per-regime LGBM) | MF | 100min | ✓ | MF |
| oi_divergence | T3✗ | Carry / positioning | Momentum (divergence) | Macro (hours) | 100min | ~ | macro |
| regime_gated | T3✗ | Order-flow / imbalance | Information gate | **Tick (~30s)** | 100min | **✗** | micro |
| entropy_momentum | T3✗ | Momentum | Information gate | MF | 100min | ✓ | MF |
| propagator | T3✗ | Mean-reversion | Order-flow (impact decay) | **Tick→MF (min)** | 100min | **✗** | micro |
| hawkes_intensity | T3✗ | Event-driven | Order-flow (self-excitation) | **Tick (s)** | 100min | **✗** | micro |
| trade_through | T3✗ | Order-flow / imbalance | Event (queue depletion) | **Tick (s)** | 100min | **✗** | micro |
| weighted_ofi | T3✗ | Order-flow / imbalance | — | **Tick (~30s)** | 100min | **✗** | micro |
| switching_ou | T3✗ | Latent state / regime | Mean-reversion (OU) | **Tick→MF** | 100min | **✗** | micro |
| vpin_regime | T3✗ | Information / toxicity | Order-flow (gated imbalance) | **Tick** | 100min | **✗** | micro |
| kalman_imbalance | T3✗ | Order-flow / imbalance | Latent state (OU Kalman) | **Tick (s)** | 100min | **✗** | micro |
| bipower_jump | T3✗ | Event-driven | Volatility ratio | **Tick→MF** | 100min | **✗** | micro |
| spread_decomp | T3✗ | Information / toxicity | Adverse-selection estimate | **Tick** | 100min | **✗** | micro |

### Implemented — infrastructure (no own tradeable signal)

| Algorithm | Status | Primary logic | Role | Horizon |
|-----------|--------|---------------|------|---------|
| regime_state_machine | INFRA | Latent state / regime | 6-state classifier feeding regime_conditioned_lgbm, trade-allowed gate | Cross-cutting |
| change_point_detector | INFRA | Event-driven | CUSUM + Bayesian OCD regime-age features | MF→Macro |
| convolver | INFRA (pre-discovery) | Latent state / regime | SVD pattern kernels (6 event types) | MF (~30min windows) |
| knn_retrieval | INFRA | Latent state / regime | Analog state retrieval, fires only past cost bar | MF |
| cascade_probability | INFRA (K2-degraded) | Event-driven | Online cascade P() from heatmap features | MF (5min horizon) |
| ensemble | INFRA | Composite | equal/IC/regime-switch combination of members | Cross-cutting |

### Literature candidates

| Candidate | Status | Primary logic | Secondary | Intrinsic | Agent |
|-----------|--------|---------------|-----------|-----------|-------|
| HF1 microprice deviation | CAND-3 | Order-flow / imbalance | Maker fair-value anchor | Tick (1–60s) | micro |
| HF2 integrated OFI | CAND-7 | Order-flow / imbalance | Cross-asset (lasso) | MF (1–5min) | MF |
| HF3 Hawkes intensity imbalance | CAND-8 | Event-driven | Order-flow, regime-gated | Tick (10–60s) | micro |
| HF4 VPIN gate | CAND-1 | Information / toxicity | Gate on existing winners | Cross-cutting | shared |
| HF5 AS market making | CAND-10 | Composite / execution | Order-flow + toxicity inputs | Continuous | execution |
| HF6 lead-lag | CAND-11 | Order-flow / imbalance | Cross-symbol | Tick (sub-s–s) | micro |
| LF1 funding-settlement windows | CAND-2 | Carry / positioning | Event-time conditioning | Macro (1–8h) | macro |
| LF2 OI-positioning extremes | CAND-6 | Carry / positioning | Momentum/reversal switch | Macro→Daily (4–48h) | macro/daily |
| LF3 liquidation-cascade reversion | CAND-9 (K2-gated) | Event-driven | Mean-reversion (forced flow) | MF→Macro | MF/macro |
| LF4 volume-weighted TSM | CAND-4 | Momentum | Vol-scaled sizing | Daily (1–7d) | daily |
| LF5 weekend conditioning | CAND-4b | Momentum | Seasonality gate | Daily | daily |
| LF6 HAR-RV sizing | CAND-5 | Composite / execution | Volatility forecast (non-directional) | Daily forecast, cross-cutting use | infra |
| LF7 conditional carry | CAND-12 | Carry / positioning | Positioning-conditioned | Daily (1–7d) | daily |

---

## 3. Matrix View — Logic × Horizon

Placement by **primary logic** and **intrinsic horizon**. `†` = T3/ML failed, `◇` = candidate, `■` = infrastructure. Bold = deployable.

| | Tick (s–min) | MF (min–1h) | Macro (1h–24h) | Daily (1–7d) | Cross-cutting |
|---|---|---|---|---|---|
| **Order-flow / imbalance** | **optimal_entry**, weighted_ofi†, trade_through†, kalman_imbalance†, regime_gated†, ◇HF1, ◇HF6 | ◇HF2 | — *(structural)* | — *(structural)* | |
| **Event-driven** | hawkes_intensity†, bipower_jump†, ◇HF3 | **jump_detector**, ■cascade_probability, ■change_point_detector | ◇LF3 | — | |
| **Mean-reversion** | propagator† | **mean_reversion_detector** | *(gap)* | *(gap)* | |
| **Momentum** | — *(structural)* | entropy_momentum†, momentum_continuation† | *(gap)* | ◇LF4, ◇LF5 | |
| **Carry / positioning** | — *(structural)* | — | **funding_reversion**, oi_divergence†, ◇LF1, ◇LF2 | ◇LF7 | |
| **Information / toxicity** | vpin_regime†, spread_decomp†, surprise_signal | — | — | — | ◇HF4 (gate) |
| **Latent state / regime** | switching_ou† | regime_conditioned_lgbm†, ■convolver, ■knn_retrieval | *(gap)* | *(gap)* | ■regime_state_machine, meta_labeling† |
| **Composite / execution** | | **3f_liquidity**, hierarchical_combiner | | ◇LF6 (sizing) | ■ensemble, ◇HF5 (MM) |

---

## 4. Findings

### 4.1 The Tier 3 graveyard is mostly a clock mismatch, not dead logic

10 of the 12 Tier 3 failures sit in the **Tick column but were tested at 100min** — 12 to 200× their intrinsic signal half-life (imbalance IC: 0.45 @ 1s → 0.09 @ 5m → 0.06 @ 15m). weighted_ofi, trade_through, kalman_imbalance, regime_gated, vpin_regime, hawkes_intensity, spread_decomp, propagator, switching_ou, bipower_jump all encode logic the IC scan confirms is *real at its native horizon*.

The catch — and why "just retest faster" is not the fix: the adverse-selection finding (conditional IC on mid-cross fills ≈ 0, `algorithms_report.md` §14) means native-horizon **taker** execution is structurally impossible. The salvage path for this column is the maker side: HF1 (microprice quote placement), HF5 (market making), and HF4 (gating) are the literature candidates designed to *re-deploy the same tick logic via the only execution channel where it survives*. Q2.7 (horizon cross-validation per symbol) should still re-test the MF-capable ones (propagator, switching_ou, bipower_jump) at 5–30min horizons.

### 4.2 Coverage gaps — where no logic is deployed

- **Daily column is empty of implementations** — every entry is a candidate (LF4–LF7). This is the quantitative justification for the daily agent (plan.md §3.3).
- **Macro is thin**: funding_reversion is the only deployable. LF1/LF2/LF3 fill it; all use live features (LF3 K2-gated).
- **Mean-reversion and latent-state at macro/daily**: genuine gaps, no candidate yet — plausible future generators (multi-day basis reversion; weekly regime models) once 60+ days of data exist.
- **Structural non-cells** (correctly empty): order-flow logic at macro/daily (book pressure cannot forecast days), carry/momentum at tick (funding/trends don't exist at seconds).

### 4.3 The portfolio works because the winners span the matrix

The 4 deployable algorithms occupy **4 different logic families** and 3 horizon bands: jump_detector (event/MF), 3f_liquidity (composite/MF), funding_reversion (carry/macro), optimal_entry (flow/tick). The near-zero jump×funding correlation (~0.00) is the maximal case: different family *and* different horizon. The one pair above the 0.35 target (3f × surprise, 0.449 on ETH) is also explained: both load on MF book-state information. **Rule of thumb for `meta_portfolio.py` and future promotions: prefer adding signals from unoccupied (family × horizon) cells; same-cell additions need a correlation check first** — e.g. HF3 vs the failed hawkes_intensity share a cell, and HF3 must demonstrate its state-dependence + gating actually de-correlates it.

### 4.4 Agent ownership of the matrix

| Agent | Owns cells | Currently has | Should generate next |
|-------|-----------|---------------|---------------------|
| micro (tick) | Tick row | optimal_entry + 7 failures | HF1, HF3, HF6 — maker-framed only |
| MF (5min) | MF column | jump, 3f, mean_reversion_detector | HF2, LF3 prototype |
| macro (1h) | Macro column | funding_reversion | LF1, LF2 |
| daily (1–7d) | Daily column | nothing | LF4+LF5, LF7 |
| shared/infra | Cross-cutting | ensemble, RSM | HF4 gate, LF6 sizing |

# Unimplemented Specifications - Prioritized Sequence

**Generated:** 2026-04-02
**Source:** docs/refined/specs.md
**Total Unimplemented Items:** 86
**Total Estimated Effort:** 600+ hours

---

## Quick Summary

| Phase | Timeframe | Items | Total Effort | Goal |
|-------|-----------|-------|--------------|------|
| **Phase 0: Critical Path** | Week 1 | 5 tasks | 26-34h | Unblock all downstream work |
| **Phase 1: Validation** | Week 2 | 7 tasks | 30-38h | Validate strategy hypothesis |
| **Phase 2: Enhancement** | Week 3-4 | 12 tasks | 67h | Build regime detection |
| **Phase 3: Production** | Week 5-8 | 10 tasks | 74h | NautilusTrader + paper trading |
| **Phase 4: Agent System** | Month 2-3 | 25 tasks | 200h+ | Multi-agent evolution |
| **Phase 5: Advanced** | Month 4+ | 27 tasks | 200h+ | Research agents, LLM, etc. |

---

## PHASE 0: CRITICAL PATH (WEEK 1) ⚠️ HIGHEST PRIORITY

**Goal:** Unblock all downstream work
**Duration:** Week 1 (26-34 hours)
**Blocker Status:** Everything depends on these

### Sequential Order (Must be done in this sequence)

#### Task 1: Daily Aggregation Pipeline (4-6 hours)
- **ID:** #1
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Aggregate tick data to daily OHLCV
- **Blocks:** Tasks #21, #22, #23, hypothesis testing, all strategies
- **Dependencies:** None
- **Source:** specs/STRATEGY_IMPLEMENTATION_PLAN.md
- **Implementation:**
  ```bash
  scripts/data/aggregate_to_daily.py
  Makefile: aggregate_to_daily target
  ```

#### Task 2: Dataset Snapshot Versioning (2 hours)
- **ID:** #35
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Version control for datasets
- **Blocks:** Reproducibility, experiment tracking
- **Dependencies:** None
- **Source:** specs/TASKS_29_3_26.md (Task 3)

#### Task 3: Experiment Manifest Schema (2 hours)
- **ID:** #36
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** .qnat.yaml experiment configuration
- **Blocks:** Experiment governance
- **Dependencies:** Task #2
- **Source:** specs/TASKS_29_3_26.md (Task 3)

#### Task 4: Cluster Analysis Integration (4-6 hours)
- **ID:** #11
- **Priority:** P0
- **Status:** ❌ NOT STARTED (Framework ⚠️ exists)
- **Description:** Integration script for cluster quality framework
- **Blocks:** Regime validation
- **Dependencies:** None (framework already complete)
- **Source:** specs/TASKS_29_3_26.md (Task 1)
- **Implementation:**
  ```bash
  scripts/ml/analyze_clusters.py
  Integration with existing cluster quality framework
  ```

#### Task 5: Hypothesis Testing Data Loader (6-8 hours)
- **ID:** #21
- **Priority:** P0
- **Status:** ❌ NOT STARTED (Framework ⚠️ exists)
- **Description:** Load Parquet data into hypothesis testing framework
- **Blocks:** H1-H5 hypothesis validation, GO/NO-GO decision
- **Dependencies:** Task #1 (daily aggregation)
- **Source:** specs/TASKS_29_3_26.md (Task 2)
- **Implementation:**
  ```bash
  rust/src/bin/test_hypotheses.rs
  Load daily parquet → Run H1-H5 tests
  ```

#### Task 6: Baseline Models (8-12 hours total)
- **IDs:** #16, #17
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:**
  - Elastic Net baseline (4h)
  - LightGBM/XGBoost baseline (4h)
- **Blocks:** ML pipeline validation
- **Dependencies:** Task #1 (daily data)
- **Source:** specs/TASKS_29_3_26.md (Task 4)

**Week 1 Total:** 26-34 hours

**Success Criteria:**
- [ ] Daily data aggregation working
- [ ] Hypothesis tests run on real data
- [ ] Cluster analysis validates regime detection
- [ ] Baseline models trained and evaluated
- [ ] Experiment governance in place

---

## PHASE 1: VALIDATION (WEEK 2)

**Goal:** Validate trading strategy hypothesis
**Duration:** Week 2 (30-38 hours)
**Depends On:** Phase 0 complete

### Sequential Order

#### Task 7: Simple MA Crossover Implementation (4 hours)
- **ID:** #22
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** BTC MA44, ETH MA33 strategy
- **Blocks:** All strategy development
- **Dependencies:** Task #1 (daily aggregation)
- **Source:** specs/STRATEGY_IMPLEMENTATION_PLAN.md (Phase 0)
- **Implementation:**
  ```python
  scripts/strategies/simple_ma_crossover.py
  - MA44 for BTC
  - MA33 for ETH
  - Transaction cost: 8 bps
  ```

#### Task 8: Daily Backtester (4 hours)
- **ID:** #23
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Simple daily backtesting engine
- **Blocks:** Strategy validation
- **Dependencies:** Task #7 (MA strategy)
- **Implementation:**
  ```python
  scripts/backtest/simple_backtester.py
  - Daily execution at open
  - Transaction costs
  - Sharpe, drawdown, win rate
  ```

#### Task 9: Walk-forward Validation Harness (8 hours)
- **ID:** #18
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Walk-forward validation framework
- **Blocks:** Robust strategy validation
- **Dependencies:** Tasks #6, #8 (baseline models, backtester)
- **Implementation:**
  ```python
  scripts/validation/walk_forward.py
  - 6-month train, 1-month test
  - Rolling forward
  - OOS/IS ratio calculation
  ```

#### Task 10: Purged/Embargoed CV (6 hours)
- **ID:** #19
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Purged and embargoed cross-validation
- **Blocks:** Prevents leakage in validation
- **Dependencies:** Task #9 (walk-forward)

#### Task 11: Cost-aware Evaluation (4 hours)
- **ID:** #20
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Transaction cost-aware metrics
- **Dependencies:** Task #8 (backtester)

#### Task 12: 1-minute Bar Aggregation (4-6 hours)
- **ID:** #2
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** 1-minute bars for intraday regime features
- **Blocks:** Intraday regime detection
- **Dependencies:** Task #1 (daily aggregation - same pipeline)

#### Task 13: Data Quality Monitoring (Complete) (4 hours)
- **ID:** #5
- **Priority:** P1
- **Status:** 🔄 IN PROGRESS
- **Description:** Finish data quality monitoring
- **Dependencies:** None

**Week 2 Total:** 34-42 hours

**Success Criteria:**
- [ ] MA44/MA33 strategy validates (Sharpe > 0.3)
- [ ] Walk-forward OOS/IS ratio > 0.6
- [ ] Backtester produces reliable results
- [ ] Data quality monitoring complete

**Decision Point:**
- **If Sharpe > 0.5:** Proceed to Phase 2 (enhancement)
- **If Sharpe 0.3-0.5:** Try enhancements, re-evaluate
- **If Sharpe < 0.3:** STOP - rethink approach

---

## PHASE 2: REGIME DETECTION & ENHANCEMENT (WEEK 3-4)

**Goal:** Build regime detection and enhance strategies
**Duration:** Week 3-4 (67 hours)
**Depends On:** Phase 1 validates

### Week 3: Regime Features (37 hours)

#### Task 14: Absorption Ratio Features (12 hours)
- **ID:** #6
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** 12 features across 4 windows
- **Blocks:** GMM clustering, HMM
- **Dependencies:** Task #12 (1-minute bars)
- **Source:** specs/TASKS_23_3_2026.md (Task 1)

#### Task 15: Churn Rate Features (8 hours)
- **ID:** #7
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** 8 features for participant turnover
- **Dependencies:** Task #12
- **Source:** specs/TASKS_23_3_2026.md (Task 2)

#### Task 16: Volume-Price Divergence (10 hours)
- **ID:** #8
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** 12 divergence detection features
- **Dependencies:** Task #12
- **Source:** specs/TASKS_23_3_2026.md (Task 3)

#### Task 17: Range Position Features (7 hours)
- **ID:** #9
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** 12 Wyckoff-style features
- **Dependencies:** Task #12
- **Source:** specs/TASKS_23_3_2026.md (Task 4)

**Week 3 Total:** 37 hours (52 features)

### Week 4: Regime Classification (30 hours)

#### Task 18: GMM Clustering (8 hours)
- **ID:** #12
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** GMM in 5D feature space
- **Blocks:** HMM, regime-aware strategies
- **Dependencies:** Tasks #14-17 (regime features)
- **Source:** specs/TASKS_23_3_2026.md (Task 5)

#### Task 19: Composite Accumulation/Distribution Scores (6 hours)
- **ID:** #10
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** 8 composite regime scores
- **Dependencies:** Tasks #14-17, #18

#### Task 20: Regime-aware MA Crossover (6 hours)
- **ID:** #24
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** MA with regime filtering
- **Blocks:** Enhanced strategy performance
- **Dependencies:** Tasks #7, #18 (simple MA, GMM)
- **Implementation:**
  ```python
  scripts/strategies/regime_aware_ma.py
  - Only trade in trending/accumulation/distribution
  - Sit out ranging markets
  ```

#### Task 21: Bollinger Band Mean Reversion (4 hours)
- **ID:** #26
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** BB mean reversion strategy
- **Dependencies:** Task #8 (backtester)

#### Task 22: Volume Breakout Strategy (4 hours)
- **ID:** #27
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Volume breakout detection
- **Dependencies:** Task #8 (backtester)

#### Task 23: Feature Schema Versioning (2 hours)
- **ID:** #4
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Version control for feature schemas
- **Dependencies:** Tasks #14-17 (new features added)

**Week 4 Total:** 30 hours

**Phase 2 Total:** 67 hours

**Success Criteria:**
- [ ] 52 regime features implemented
- [ ] GMM clustering working (Silhouette > 0.3)
- [ ] Regime-aware MA improves Sharpe by +0.1-0.3
- [ ] Multiple strategy types validated

---

## PHASE 3: PRODUCTION READINESS (WEEK 5-8)

**Goal:** NautilusTrader integration, paper trading, live deployment
**Duration:** 4 weeks (74 hours)
**Depends On:** Phase 2 validates enhanced strategies

### Week 5: NautilusTrader Foundation (30 hours)

#### Task 24: Hyperliquid Execution Adapter (16 hours)
- **ID:** #28
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Nautilus adapter for Hyperliquid
- **Blocks:** All Nautilus integration
- **Dependencies:** None
- **Source:** specs/STRATEGY_IMPLEMENTATION_PLAN.md
- **Implementation:**
  ```python
  integrations/nautilus/hyperliquid_adapter.py
  - ExecutionClient implementation
  - Order routing
  - Fill simulation
  ```

#### Task 25: MA Strategy in Nautilus Framework (8 hours)
- **ID:** #29
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Implement MA in Nautilus
- **Dependencies:** Tasks #7, #24 (MA strategy, adapter)

#### Task 26: Backtest Engine Integration (6 hours)
- **ID:** #30
- **Priority:** P0
- **Status:** ❌ NOT STARTED
- **Description:** Nautilus backtest integration
- **Dependencies:** Task #24 (adapter)

**Week 5 Total:** 30 hours

### Week 6-7: Paper Trading (12 hours setup + validation)

#### Task 27: Paper Trading Harness (12 hours)
- **ID:** #31
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Paper trading system
- **Dependencies:** Tasks #24-26 (Nautilus integration)
- **Implementation:**
  ```python
  integrations/nautilus/run_live.py --testnet
  - Connect to Hyperliquid testnet
  - Run strategies in paper trading mode
  - 4-8 weeks validation
  ```

**Weeks 6-7:** Paper trading validation (no dev hours, just monitoring)

**Success Criteria for Paper Trading:**
- [ ] Paper trading Sharpe matches backtest (±20%)
- [ ] Execution slippage < 2 bps
- [ ] No system failures over 4 weeks
- [ ] Risk management working correctly

### Week 8: Live Deployment (8 hours + ongoing)

#### Task 28: Live Trading Deployment (8 hours)
- **ID:** #32
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Description:** Production deployment
- **Dependencies:** Task #27 (paper trading validates)
- **Implementation:**
  ```python
  integrations/nautilus/run_live.py --live
  - Start with $1K-5K
  - Scale if successful
  ```

#### Task 29: Adaptive MA (Spectral/Illiquidity) (8 hours)
- **ID:** #25
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Description:** Adaptive MA length optimization
- **Dependencies:** Tasks #7, #20 (strategies, regime detection)

#### Task 30: Real-time Regime Classifier (6 hours)
- **ID:** #14
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Real-time regime classification
- **Dependencies:** Task #18 (GMM clustering)

#### Task 31: Artifact Lineage Tracking (6 hours)
- **ID:** #37
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Track all artifact lineage
- **Dependencies:** Task #2 (dataset versioning)

#### Task 32: Deterministic Orchestration (8 hours)
- **ID:** #38
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Deterministic experiment runs
- **Dependencies:** Task #3 (manifest schema)

**Week 8 Total:** 36 hours

**Phase 3 Total:** 74 hours

**Success Criteria:**
- [ ] Paper trading successful (4-8 weeks)
- [ ] Live trading deployed with small capital
- [ ] Multiple strategies running
- [ ] Production monitoring in place

---

## PHASE 4: AGENT SYSTEM (MONTH 2-3) - OPTIONAL

**Goal:** Multi-agent evolution system
**Duration:** 2-3 months (200+ hours)
**Depends On:** Phase 3 deployed and profitable
**NOTE:** Only proceed if base strategies are profitable

### Infrastructure (40 hours)

#### Task 33: PostgreSQL + TimescaleDB Schema (8 hours)
- **ID:** #39
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Source:** specs/AGENT_SYSTEM_TASK_SEQUENCE.md (Task 1)

#### Task 34: Redis + Celery Task Queue (4 hours)
- **ID:** #40
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Source:** specs/AGENT_SYSTEM_TASK_SEQUENCE.md (Task 2)

#### Task 35: Base Agent Class (8 hours)
- **ID:** #41
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #34 (task queue)
- **Source:** specs/AGENT_SYSTEM_TASK_SEQUENCE.md

#### Task 36: Agent Orchestrator (12 hours)
- **ID:** #46
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #35 (base agent)

#### Task 37: Incremental Data Updates (8 hours)
- **ID:** #3
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Description:** Incremental data update pipeline

**Infrastructure Total:** 40 hours

### Genotype System (24 hours)

#### Task 38: Base Genotype Class (6 hours)
- **ID:** #47
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED

#### Task 39: MA Crossover Genotype (4 hours)
- **ID:** #48
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #38

#### Task 40: Mutation Operators (4 hours)
- **ID:** #49
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #38

#### Task 41: Crossover Operators (4 hours)
- **ID:** #50
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #38

#### Task 42: Fitness Evaluation (6 hours)
- **ID:** #51
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #38

**Genotype Total:** 24 hours

### Agent Types (50 hours)

#### Task 43: Evaluator Agent (12 hours)
- **ID:** #43
- **Priority:** P0 (for agents)
- **Status:** ❌ NOT STARTED
- **Dependencies:** Tasks #35, #38-42 (base agent, genotypes)

#### Task 44: Generator Agent (6 hours)
- **ID:** #42
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Tasks #35, #38 (base agent, genotypes)

#### Task 45: Evolver Agent (16 hours)
- **ID:** #44
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #43 (evaluator)

#### Task 46: Genetic Algorithm Core (12 hours)
- **ID:** #52
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Tasks #38-42 (genotype system)

#### Task 47: Monitor Agent (4 hours)
- **ID:** #45
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #35 (base agent)

**Agent Types Total:** 50 hours

### Evolution Engine (20 hours)

#### Task 48: Elitism Selection (2 hours)
- **ID:** #53
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #46 (GA core)

#### Task 49: Tournament Selection (2 hours)
- **ID:** #54
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #46

#### Task 50: HMM Regime Detection (16 hours)
- **ID:** #13
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Description:** Hidden Markov Model for regimes
- **Dependencies:** Task #18 (GMM clustering)

**Evolution Total:** 20 hours

### Dashboard (66 hours)

#### Task 51: FastAPI Backend (8 hours)
- **ID:** #56
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 52: Agent Status Display (4 hours)
- **ID:** #57
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #51

#### Task 53: Real-time Activity Feed (6 hours)
- **ID:** #58
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #51

#### Task 54: Strategy Leaderboard (4 hours)
- **ID:** #60
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #51

#### Task 55: Performance Analytics Page (8 hours)
- **ID:** #61
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #51

#### Task 56: Evolution Tree Visualization (12 hours)
- **ID:** #59
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #51

#### Task 57: React Frontend (24 hours)
- **ID:** #62
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Tasks #51-56 (all backend)

**Dashboard Total:** 66 hours

**Phase 4 Total:** ~200 hours

---

## PHASE 5: ADVANCED FEATURES (MONTH 4+)

**Goal:** Research agents, LLM integration, advanced ML
**Duration:** Open-ended (200+ hours)
**Depends On:** Phase 4 if building agents, or Phase 3 if skipping agents

### Advanced ML (42 hours)

#### Task 58: FTRL Online Linear Model (12 hours)
- **ID:** #76
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 59: Prequential Evaluation (6 hours)
- **ID:** #77
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 60: Contextual Bandits (16 hours)
- **ID:** #78
- **Priority:** P2
- **Status:** ❌ NOT STARTED

#### Task 61: Train Phase Classifier (LSTM) (12 hours)
- **ID:** #15
- **Priority:** P2
- **Status:** ❌ NOT STARTED

### Robustness Testing (20 hours)

#### Task 62: Threshold Perturbation Tests (4 hours)
- **ID:** #79
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 63: Noise Injection Tests (4 hours)
- **ID:** #80
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 64: Regime-split Analysis (6 hours)
- **ID:** #81
- **Priority:** P1
- **Status:** ❌ NOT STARTED
- **Dependencies:** Task #18 (regime classifier)

#### Task 65: Feature Stability Scoring (6 hours)
- **ID:** #82
- **Priority:** P1
- **Status:** ❌ NOT STARTED

### Productization (38 hours)

#### Task 66: Promotion Policy (8 hours)
- **ID:** #83
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 67: Signal Retirement/Drift Policy (6 hours)
- **ID:** #84
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 68: WebSocket/REST API Documentation (8 hours)
- **ID:** #85
- **Priority:** P2
- **Status:** ❌ NOT STARTED

#### Task 69: Auth/Billing Placeholders (16 hours)
- **ID:** #86
- **Priority:** P3
- **Status:** ❌ NOT STARTED

### Dashboard Enhancements (4 hours)

#### Task 70: Cloudflare Tunnel Setup (2 hours)
- **ID:** #33
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 71: Basic Authentication (2 hours)
- **ID:** #34
- **Priority:** P1
- **Status:** ❌ NOT STARTED

### LLM Integration (34 hours)

#### Task 72: Skill Discovery (6 hours)
- **ID:** #70
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 73: OpenAI/Anthropic Tool Mapping (8 hours)
- **ID:** #71
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 74: Skill Invocation Router (6 hours)
- **ID:** #72
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 75: Tool Loop Orchestration (8 hours)
- **ID:** #73
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 76: GET /skills Endpoint (2 hours)
- **ID:** #74
- **Priority:** P2
- **Status:** ❌ NOT STARTED

#### Task 77: POST /invoke Endpoint (4 hours)
- **ID:** #75
- **Priority:** P2
- **Status:** ❌ NOT STARTED

### Research Agents (96 hours)

#### Task 78: Experiment Designer (16 hours)
- **ID:** #65
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 79: Code Builder Agent (20 hours)
- **ID:** #66
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 80: Backtest Agent (12 hours)
- **ID:** #67
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 81: Critic Agent (12 hours)
- **ID:** #68
- **Priority:** P1
- **Status:** ❌ NOT STARTED

#### Task 82: Research Librarian (16 hours)
- **ID:** #63
- **Priority:** P2
- **Status:** ❌ NOT STARTED

#### Task 83: Hypothesis Agent (12 hours)
- **ID:** #64
- **Priority:** P2
- **Status:** ❌ NOT STARTED

#### Task 84: Reporter Agent (8 hours)
- **ID:** #69
- **Priority:** P2
- **Status:** ❌ NOT STARTED

### Advanced Evolution (18 hours)

#### Task 85: Multi-objective Optimization (16 hours)
- **ID:** #55
- **Priority:** P2
- **Status:** ❌ NOT STARTED
- **Dependencies:** Tasks #46, #48-49 (GA core, selection)

#### Task 86: (Reserved for future)

**Phase 5 Total:** ~250 hours

---

## CRITICAL PATH VISUALIZATION

```
WEEK 1 (Phase 0) - CRITICAL
├─ Daily Aggregation (4-6h) ────────────┐
├─ Dataset Versioning (2h)              │
├─ Experiment Manifest (2h)             │
├─ Cluster Integration (4-6h)           ├─→ UNBLOCKS ALL DOWNSTREAM
├─ Hypothesis Data Loader (6-8h)        │
└─ Baseline Models (8-12h) ─────────────┘
         ↓
WEEK 2 (Phase 1) - VALIDATION
├─ Simple MA Strategy (4h)
├─ Daily Backtester (4h)
├─ Walk-forward Validation (8h)
├─ Purged CV (6h)
├─ Cost-aware Eval (4h)
└─ 1-min Bars (4-6h)
         ↓
    DECISION POINT: Does MA work?
         ↓ YES (Sharpe > 0.3)
         ↓
WEEK 3-4 (Phase 2) - ENHANCEMENT
├─ Regime Features (37h)
│  ├─ Absorption (12h)
│  ├─ Churn (8h)
│  ├─ Divergence (10h)
│  └─ Range Position (7h)
├─ GMM Clustering (8h)
└─ Regime-aware MA (6h)
         ↓
WEEK 5-8 (Phase 3) - PRODUCTION
├─ NautilusTrader Adapter (16h)
├─ Nautilus Integration (14h)
├─ Paper Trading (4-8 weeks)
└─ Live Deployment
         ↓
    DECISION POINT: Profitable?
         ↓ YES
         ↓
MONTH 2-3 (Phase 4) - AGENTS [OPTIONAL]
└─ Multi-agent evolution system (200h)
         ↓
MONTH 4+ (Phase 5) - ADVANCED
└─ Research agents, LLM, advanced ML (250h)
```

---

## EFFORT SUMMARY BY PHASE

| Phase | Duration | Tasks | Total Effort | Cumulative |
|-------|----------|-------|--------------|------------|
| **Phase 0** | Week 1 | 6 | 26-34h | 26-34h |
| **Phase 1** | Week 2 | 7 | 34-42h | 60-76h |
| **Phase 2** | Week 3-4 | 12 | 67h | 127-143h |
| **Phase 3** | Week 5-8 | 10 | 74h | 201-217h |
| **Phase 4** | Month 2-3 | 25 | 200h | 401-417h |
| **Phase 5** | Month 4+ | 26 | 250h | 651-667h |

---

## RECOMMENDED APPROACH

### Conservative Path (Recommended)
1. **Complete Phase 0** (Week 1)
2. **Complete Phase 1** (Week 2)
3. **Evaluate:** If Sharpe < 0.3 → STOP
4. **Complete Phase 2** (Week 3-4)
5. **Complete Phase 3** (Week 5-8)
6. **Evaluate:** If profitable → Scale trading
7. **Skip Phase 4** (Agent system has low ROI for single strategy)
8. **Cherry-pick Phase 5** (Only what's needed)

**Total Time to Production:** 8 weeks (127-143 hours)
**Total Time if Including Agents:** 12-16 weeks (401-417 hours)

### Aggressive Path (Not Recommended)
1. Do everything in sequence
2. 6+ months to completion
3. High risk of building unused infrastructure

---

## DEPENDENCIES MAP

**Critical Dependencies (Must Complete First):**
- Daily Aggregation → Enables: Hypothesis testing, strategies, all ML
- Cluster Integration → Enables: Regime validation
- Hypothesis Data Loader → Enables: H1-H5 validation, GO/NO-GO
- Baseline Models → Enables: ML pipeline validation

**Secondary Dependencies:**
- Regime Features → GMM → HMM → Regime-aware strategies
- Simple MA → Regime-aware MA → Adaptive MA
- Nautilus Adapter → Nautilus Integration → Paper Trading → Live Trading
- Genotype System → Agents → Evolution Engine → Dashboard

---

## QUICK REFERENCE

**"What should I do TODAY?"**
→ Start Task #1: Daily Aggregation Pipeline (4-6h)

**"What should I do this WEEK?"**
→ Complete Phase 0: Tasks #1-6 (26-34h)

**"What should I do this MONTH?"**
→ Complete Phases 0-2: Tasks #1-23 (127-143h)

**"When can I start trading?"**
→ Week 8 (after Phase 3 completes and paper trading validates)

**"Should I build the agent system?"**
→ Only if Phase 3 is profitable and you want to explore multiple strategy families

---

**Last Updated:** 2026-04-02
**Source:** Extracted from docs/refined/specs.md
**Total Items:** 86 unimplemented specifications
**Critical Path:** 6 tasks blocking everything (Phase 0)

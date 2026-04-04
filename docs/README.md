# NAT Documentation

Comprehensive documentation for the NAT (Hyperliquid Analytics Layer) project.

---

## Quick Navigation

### 🎯 **I want to know what to implement next**
→ **[refined/specs.md](refined/specs.md)** - Single consolidated list with priorities and status

### 📋 **I want detailed specifications**
→ **[specs/](specs/)** - All technical specifications and task lists

### 📚 **I want guides and analysis**
→ Stay in this directory - Roadmaps, guides, and strategic analysis

### 👤 **I'm a user looking for how-to guides**
→ **[user_guide/](user_guide/)** - User-facing documentation

---

## Directory Structure

```
docs/
├── README.md (this file)
│
├── refined/                          # Consolidated documentation
│   ├── specs.md                      # ⭐ Single source of truth for all specs
│   └── README.md
│
├── specs/                            # Detailed specifications
│   ├── ALGORITHM_DESIGN_PROPOSAL.md  # 🆕 8 algorithmic approaches (detailed)
│   ├── TASKS_29_3_26.md              # ⚠️ Week 1 critical tasks (P0)
│   ├── TASKS_23_3_2026.md            # Regime features (Week 2-3)
│   ├── STRATEGY_IMPLEMENTATION_PLAN.md  # Trading strategies
│   ├── AGENT_BASED_RESEARCH_ARCHITECTURE.md  # Multi-agent system
│   └── [19 specification documents]
│
├── user_guide/                       # User documentation
│   ├── USER_MANUAL.md                # Complete user manual
│   ├── MODEL_SERVING_COMPLETE.md
│   └── [ML workflow guides]
│
└── [Analysis & Guides]               # Strategic docs (this directory)
    ├── ALGORITHMIC_RESEARCH_DIRECTION.md  # 🆕 Algorithmic strategy proposal
    ├── EXECUTIVE_SUMMARY.md          # Project state summary
    ├── IMMEDIATE_ACTION_PLAN.md      # Get out of analysis paralysis
    ├── NEXT_STEPS_ROADMAP.md         # Comprehensive roadmap
    ├── PROJECT_MATURITY_ANALYSIS.md  # 80-85% mature assessment
    ├── QUICK_START_DAILY_TRADING.md  # 4-week implementation guide
    ├── FEATURE_STRATEGY_MISMATCH_ANALYSIS.md  # 183 features analysis
    ├── TIME_HORIZON_ANALYSIS.md      # Why daily > HFT
    └── ORIGINALITY_AND_REALITY_CHECK.md  # Honest assessment

```

---

## Documents in This Directory (Main docs/)

### Strategic Analysis

**EXECUTIVE_SUMMARY.md** (4KB)
- Three-sentence project summary
- 80-85% mature, 0% validated
- Critical decision points

**PROJECT_MATURITY_ANALYSIS.md** (17KB)
- Component-by-component maturity assessment
- Infrastructure vs validation gap
- Comparison to industry standards

**ORIGINALITY_AND_REALITY_CHECK.md** (15KB)
- Honest assessment of originality (30-40%)
- Alpha probability estimates (40-60%)
- Cost-benefit analysis of evolution system

### Algorithmic Research (NEW - 2026-04-04)

**ALGORITHMIC_RESEARCH_DIRECTION.md** (10KB) 🆕 **START HERE FOR NEXT PHASE**
- Executive summary of algorithmic approach
- 8 proposed algorithms (entropy-gated, momentum, mean-reversion, meta-labeling, etc.)
- NautilusTrader integration architecture
- Liquidity heatmap with regime detection
- Continuous research pipeline (10 phases)
- Success metrics and decision criteria

**specs/ALGORITHM_DESIGN_PROPOSAL.md** (40KB+) 🆕 **DETAILED TECHNICAL SPECS**
- Complete mathematical specifications for 8 algorithms
- Feature usage matrix (which features for which algorithm)
- NautilusTrader data adapter design
- Regime classification logic with validation
- Research lab iteration structure
- Implementation roadmap (Phase 0-5)

**Key Insight:** Use **entropy as a gating mechanism** to select regime-appropriate algorithms:
- Low entropy (<0.3) → Momentum continuation strategies
- High entropy (>0.7) → Mean-reversion strategies
- Uncertain entropy → No trade

### Implementation Guides

**IMMEDIATE_ACTION_PLAN.md** (18KB) ⚠️ **START HERE IF PARALYZED**
- Get out of high-entropy decision state
- Phase 0: Validate MA hypothesis (TODAY, 4-6 hours)
- Clear decision tree for every outcome
- Complete backtest implementation code

**QUICK_START_DAILY_TRADING.md** (9KB)
- 4-week roadmap: validation → production
- Week-by-week actionable tasks
- Risk management guidelines
- Expected performance metrics

**NEXT_STEPS_ROADMAP.md** (60KB)
- Comprehensive 3-6 month roadmap
- Addresses all 7 user questions
- Phase-by-phase implementation
- Multiple strategy families

### Technical Analysis

**FEATURE_STRATEGY_MISMATCH_ANALYSIS.md** (15KB)
- Why 183 HFT features don't match daily strategies
- 75% of features wrong time scale
- Daily aggregation pipeline design
- Liquidation cascade integration

**TIME_HORIZON_ANALYSIS.md** (17KB)
- Mathematical proof: daily > HFT
- Sharpe ratio vs time horizon (0.05 → 0.5-1.0)
- Transaction cost analysis
- Signal-to-noise scaling (38× improvement)

**STRATEGY_IMPLEMENTATION_PLAN.md** → Moved to specs/

### Status & Integration

**INTEGRATION_COMPLETE.md**
- ML pipeline integration status
- What's working, what's not
- Next integration steps

---

## Current Project Status (2026-04-02)

### What's Implemented ✅
- Real-time data ingestion (Hyperliquid WebSocket)
- 183 microstructure features (tick-level)
- Hypothesis testing framework (H1-H5)
- Cluster quality measurement
- Real-time dashboard (partial)
- ML experiment tracking
- Model serving API

### What's NOT Implemented ❌
- Daily data aggregation (CRITICAL)
- Regime detection features (52 features)
- Any trading strategies
- Baseline ML models
- NautilusTrader integration
- Agent evolution system
- Paper trading infrastructure

### Critical Path (Next 4 Weeks)

**Week 1:** TASKS_29_3_26.md (26-34 hours)
- Daily aggregation
- Cluster analysis integration
- Hypothesis testing data loader
- Experiment governance
- Baseline models

**Week 2-3:** TASKS_23_3_2026.md (37 hours)
- Regime features (52 features)
- GMM clustering
- HMM foundation

**Week 4:** STRATEGY_IMPLEMENTATION_PLAN.md
- Simple MA backtest validation
- If works → enhance
- If fails → pivot

---

## How to Use This Documentation

### Scenario 1: "I don't know what to implement next"
1. Read **IMMEDIATE_ACTION_PLAN.md** (if feeling paralyzed)
2. Check **refined/specs.md** for prioritized list
3. Start with Week 1 tasks in **specs/TASKS_29_3_26.md**

### Scenario 2: "I want to understand project state"
1. Read **EXECUTIVE_SUMMARY.md** (5 min read)
2. Read **PROJECT_MATURITY_ANALYSIS.md** (detailed)
3. Check **refined/specs.md** for completion %

### Scenario 3: "I want to implement trading strategies"
1. **NEW APPROACH:** Read **ALGORITHMIC_RESEARCH_DIRECTION.md** (8 algorithms, entropy-gated)
2. Read **specs/ALGORITHM_DESIGN_PROPOSAL.md** for detailed specs
3. Alternative: Read **IMMEDIATE_ACTION_PLAN.md** Phase 0 (simple MA validation)
4. Alternative: Read **QUICK_START_DAILY_TRADING.md** (4-week guide)

### Scenario 3b: "I want to understand the algorithmic approach"
1. Read **ALGORITHMIC_RESEARCH_DIRECTION.md** (executive summary, 10 min read)
2. Read **specs/ALGORITHM_DESIGN_PROPOSAL.md** (detailed technical specs, 30 min)
3. Key concepts: Entropy gating, regime detection, NautilusTrader integration
4. Critical path: Daily aggregation → regime validation → algorithm prototyping

### Scenario 4: "I want to build agent system"
1. **STOP** - Read **ORIGINALITY_AND_REALITY_CHECK.md** first
2. Validate base strategy first (IMMEDIATE_ACTION_PLAN.md)
3. Only proceed if alpha validated
4. Then: **specs/AGENT_BASED_RESEARCH_ARCHITECTURE.md**

### Scenario 5: "I'm looking for API/user documentation"
→ **user_guide/USER_MANUAL.md**

---

## Documentation Philosophy

**refined/** - What to build (consolidated priorities)
**specs/** - How to build it (detailed specifications)
**docs/** (this directory) - Why and when to build it (strategy & analysis)
**user_guide/** - How to use what's built (end-user docs)

---

## Key Insights from Documentation

1. **Infrastructure ≠ Alpha** (ORIGINALITY_AND_REALITY_CHECK.md)
   - 80-85% infrastructure complete
   - 0% alpha validation
   - Build less, validate more

2. **Time Scale Mismatch** (FEATURE_STRATEGY_MISMATCH_ANALYSIS.md)
   - 183 tick features built for HFT
   - Planning daily strategies
   - Need aggregation pipeline

3. **Daily > HFT** (TIME_HORIZON_ANALYSIS.md)
   - Daily Sharpe: 0.5-1.0
   - HFT Sharpe: 0.05-0.15
   - Transaction costs kill HFT

4. **Validate First** (IMMEDIATE_ACTION_PLAN.md)
   - Don't build agents before proving MA works
   - 1 day validation vs 3 months building
   - Infrastructure scales proven alpha

5. **Evolution ROI is Low** (ORIGINALITY_AND_REALITY_CHECK.md)
   - Evolution: +0.1 Sharpe for 2 months work
   - Grid search: 0.65 Sharpe in 3 days
   - Opportunity cost matters

6. **Entropy as Regime Detector** (ALGORITHMIC_RESEARCH_DIRECTION.md) 🆕
   - Low entropy (<0.3) = predictable regimes → momentum strategies work
   - High entropy (>0.7) = random walk → mean-reversion strategies work
   - Use entropy to gate which algorithm runs, not force one model everywhere
   - Regime-specific validation required (test separately in each regime)

7. **Hypothesis-Driven > ML-First** (ALGORITHM_DESIGN_PROPOSAL.md) 🆕
   - Form testable prediction first, then select algorithm
   - Use interpretable models (logistic regression, manual thresholds) before complex ML
   - Validate rigorously: walk-forward, OOS/IS > 0.7, regime-specific
   - Only deploy if GO criteria met (Sharpe > 0.5, win rate > 52%)

8. **NautilusTrader for Realistic Backtesting** (ALGORITHM_DESIGN_PROPOSAL.md) 🆕
   - Same code for backtest and live trading
   - Realistic order matching, slippage, latency simulation
   - Risk management built-in (position limits, drawdown controls)
   - Production-ready execution engine (Rust/Cython)

---

## Maintenance

**Last Reorganized:** 2026-04-02
**Last Major Update:** 2026-04-04 (Added algorithmic research direction)
**Maintained By:** Auto-generated from project state
**Update Frequency:** After major milestones

**To update specs.md:**
1. Modify files in specs/
2. Re-run consolidation (manual for now)
3. Update implementation status

---

**Questions?** See IMMEDIATE_ACTION_PLAN.md for decision tree
**Stuck?** See refined/specs.md for what's actually important
**Ready to code?** See specs/TASKS_29_3_26.md for Week 1 tasks
**Want algorithmic strategy?** 🆕 See ALGORITHMIC_RESEARCH_DIRECTION.md (executive summary)
**Need detailed algorithm specs?** 🆕 See specs/ALGORITHM_DESIGN_PROPOSAL.md (8 algorithms)

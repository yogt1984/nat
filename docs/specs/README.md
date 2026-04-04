# Specification Documents

This directory contains all technical specifications, feature designs, and implementation task lists.

## Quick Reference

**For prioritized, consolidated view:** See `../refined/specs.md`
**For detailed specifications:** Browse files in this directory

---

## Document Categories

### Core Specifications

**AGENT_BASED_RESEARCH_ARCHITECTURE.md** (39KB)
- Multi-agent system architecture
- Genotype/phenotype framework
- Evolution engine design
- Web dashboard specifications

**AGENT_SYSTEM_TASK_SEQUENCE.md** (23KB)
- Detailed implementation tasks (Phase 1-4)
- Task dependencies and sequence
- Complete code examples
- Success criteria for each task

**STRATEGY_IMPLEMENTATION_PLAN.md** (26KB)
- Daily trading strategy implementation
- NautilusTrader integration
- 4-phase roadmap (validation → production)
- Risk management guidelines

### Feature Specifications

**TASKS_23_3_2026.md** (52KB)
- Regime detection features (52 features)
- Absorption, churn, divergence, range position
- HMM specifications
- Estimated: 37 hours implementation

**TASKS_20_3_2026.md** (27KB)
- Regime feature extraction design
- Real-time classification
- Integration with existing pipeline

**TASKS_3_5_26.md** (54KB)
- Original 24-task Hyperliquid roadmap
- Microstructure features (70+ features)
- Hypothesis testing framework
- Status: ~50% implemented

### Testing & Validation

**CLUSTER_QUALITY_MEASUREMENT_FRAMEWORK_SPECS.md**
- Cluster validation metrics
- Silhouette, Davies-Bouldin, Calinski-Harabasz
- Bootstrap stability, temporal stability
- Status: Framework complete

**CLUSTER_QUALITY_MEASUREMENT_TASKS.md**
- Implementation tasks for cluster analysis
- Integration requirements

**HYPOTHESIS_TESTING_GUIDE.md**
- H1-H5 hypothesis specifications
- Statistical testing framework
- GO/PIVOT/NO-GO decision logic
- Status: Framework complete, integration pending

### Critical Tasks

**TASKS_29_3_26.md** (45KB) ⚠️ **HIGHEST PRIORITY**
- Critical integration tasks (Week 1 priority)
- Cluster analysis integration (4-6h)
- Hypothesis testing data loader (6-8h)
- Experiment governance (4-6h)
- Baseline models (8-12h)

**Total:** 26-34 hours blocking all downstream work

### Project Planning

**PROJECT_PROPOSAL_0__6_3_2026.md**
- 10-phase master project plan
- Milestones and deliverables
- Timeline and dependencies

**PROJECT_STATE_AND_ALGO_EXPERIMENT_PLAN_2026-03-23.md**
- Algorithm selection framework
- Online learning specifications
- Evaluation methodology

**PRIORITIES_23_3_26.md**
- Historical priority decisions
- Feature prioritization rationale

### Advanced Systems

**MULTI_AGENT_PROPOSAL.md**
- Research agent specifications
- Agent roles and orchestration
- Inter-agent communication

**LLM_BRIDGE_SKILL_SPEC.md**
- LLM integration architecture
- Skill discovery and invocation
- Tool mapping specifications

### Historical Task Lists

**TASKS_12_3_26.md** - Hypothesis testing implementation
**TASKS_13_3_26.md** - Data collection roadmap
**TASKS_20_3_2026__2.md** - Dashboard implementation
**X_0.md** - Pending features summary

---

## Implementation Status Overview

| Category | Total Items | Implemented | % Complete |
|----------|-------------|-------------|------------|
| Data Infrastructure | 10 | 6 | 60% |
| Feature Engineering | 95 | 70 | 74% |
| ML Pipeline | 20 | 8 | 40% |
| Testing/Validation | 12 | 7 | 58% |
| Trading Strategies | 12 | 0 | 0% |
| Agent Systems | 30 | 0 | 0% |
| **Overall** | **206** | **94** | **46%** |

---

## Navigation

- **Start Here:** Read `../refined/specs.md` for consolidated view
- **Critical Tasks:** See `TASKS_29_3_26.md` (Week 1 priority)
- **Regime Features:** See `TASKS_23_3_2026.md` (Week 2-3)
- **Trading:** See `STRATEGY_IMPLEMENTATION_PLAN.md` (Week 4+)
- **Agents:** See `AGENT_BASED_RESEARCH_ARCHITECTURE.md` (Month 2-3)

---

**Last Updated:** 2026-04-02
**Total Specification Lines:** ~15,000
**Total Documents:** 19

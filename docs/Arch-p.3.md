# Architecture Phase 3 — Code Quality & Research Integrity

Actionable fixes for structural weaknesses in the Python codebase. Ordered by dependency chain and impact on research velocity.

**Prerequisite:** Arch-p.1.md complete.

---

## Priority 1 — Python Packaging & Organization

### 1.1 Add pyproject.toml and __init__.py Files ✅

**Problem:** Zero `__init__.py` files in the entire `scripts/` tree. 133 `sys.path.insert` hacks across 127 files. Imports are fragile — renaming a directory silently breaks downstream code with no traceback pointing to the actual cause.

**Implementation:**
1. Create `scripts/pyproject.toml` with a `nat` package:
   ```toml
   [project]
   name = "nat-research"
   version = "0.1.0"
   requires-python = ">=3.12"

   [tool.setuptools.packages.find]
   where = ["."]
   include = ["agent*", "data*", "alpha*", "backtest*", "execution*",
              "it_engine*", "algorithms*", "utils*"]
   ```
2. Add `__init__.py` to every package directory: `agent/`, `data/`, `alpha/`, `backtest/`, `execution/`, `it_engine/`, `algorithms/`, `utils/`.
3. Install in dev mode: `pip install -e scripts/` so all imports resolve without `sys.path`.
4. Replace `sys.path.insert(0, ...)` with proper relative or absolute imports in all production modules. Keep `sys.path` hacks only in standalone CLI scripts and tests (via `conftest.py`).
5. Add a `scripts/tests/conftest.py` that sets up the import path once for all tests.

**Files:** `scripts/pyproject.toml` (new), `scripts/*/\_\_init\_\_.py` (new), `scripts/tests/conftest.py` (new), 127 files with `sys.path` removal (incremental)

**Effort:** ~3h

---

### 1.2 Organize Root-Level Scripts

**Problem:** 42 `.py` files in `scripts/` root. Mix of:
- **Production services:** `pipeline_runner.py`, `discovery_orchestrator.py`, `agent_dashboard.py`, `logging_config.py`, `config_utils.py`
- **CLI tools:** `phase1_signal_test.py`, `run_backtest.py`, `score_data.py`, `validate_data.py`
- **One-off exploration:** `15m_test.py`, `15m_visualize.py`, `explore_clusters.py`, `spannung_*.py`, `visualize_*.py`

No way to tell which scripts are active vs abandoned without reading each one.

**Implementation:**
1. Keep in `scripts/` root: production services and shared utilities only (`pipeline_runner.py`, `discovery_orchestrator.py`, `agent_dashboard.py`, `logging_config.py`, `config_utils.py`, `phase1_signal_test.py`).
2. Move CLI tools to `scripts/cli/`: `run_backtest.py`, `run_backtest_tracked.py`, `run_experiment.py`, `score_data.py`, `validate_data.py`, `train_baseline.py`, `train_regime_gmm.py`.
3. Move exploration/viz scripts to `exploration/`: `15m_*.py`, `spannung_*.py`, `visualize_*.py`, `explore_clusters.py`, `analyze_clusters.py`, `scalp_edge_scanner.py`, `scalping_profiler.py`, `oos_*.py`, `q3_*.py`, `audit_*.py`, `skeptical_*.py`.
4. Update Makefile targets that reference moved scripts.
5. Add a `scripts/README.md` with a one-liner per remaining root script.

**Files:** Move ~30 scripts, update Makefile

**Effort:** ~2h

---

## Priority 2 — Module Structure

### 2.1 Split base.py (1,550 lines → 3 modules)

**Problem:** `scripts/agent/base.py` owns the cycle loop, state machine, gate functions, config loading, FDR control, hypothesis chaining, monitoring, and research output emission. After 3 months away, this file is the first one that becomes opaque.

**Target split:**
```
agent/
  base.py          — ResearchAgent ABC, run(), run_cycle() (~400 lines)
  runner.py        — BaseRunner ABC (already exists, keep)
  gates.py         — Gate protocol + check functions (already exists from P4.2)
  state_machine.py — AgentPhase, AgentState, load_agent_config, validate_config (~300 lines)
  monitor.py       — run_monitor(), rolling IC, decay tracking (~200 lines)
```

**Implementation:**
1. Extract `AgentPhase`, `AgentState`, `load_agent_config`, `validate_config`, `_deep_merge` to `agent/state_machine.py`.
2. Extract `run_monitor()`, `_compute_rolling_ic()`, `_check_decay()` to `agent/monitor.py`.
3. Move remaining free functions (`_find_gate_entry`, `_ic_pvalue`, `check_ic_gate`, `check_dIC_gate`, `check_cost_gate`, `check_coverage_gate`, `check_walkforward_gate`, `check_correlation_gate`, `apply_fdr`) into `agent/gates.py` (extend existing).
4. Keep `ResearchAgent` and `BaseRunner` in `base.py` — now a thin orchestration layer.
5. Update all imports. Re-export from `base.py` for backward compat during transition.

**Files:** `scripts/agent/state_machine.py` (new), `scripts/agent/monitor.py` (new), `scripts/agent/gates.py` (extend), `scripts/agent/base.py` (shrink)

**Effort:** ~3h

---

### 2.2 Unify Cost Model Across Backtest and Paper Trading

**Problem:** Three independently implemented fill/cost models:
- `backtest/costs.py`: `CostModel(fee_bps=5.0, slippage_bps=2.0)` — taker, conservative
- `alpha/paper_trader_generic.py`: `FEE_BPS = 1.61` — hardcoded Binance VIP9, no slippage
- `execution/signal_bridge.py`: `MAKER_FEE_BPS = 0.30` — maker rebate
- `kalman/fill_sim.py`: Sophisticated passive fill simulation (queue priority, latency) — unused by main pipeline

A hypothesis can show Sharpe 2.0 in backtest (7 bps costs) and Sharpe 4.0 in paper trading (1.61 bps costs) — or the reverse. No way to diagnose without manually checking which cost model each module uses.

**Implementation:**
1. Move `CostModel` from `backtest/costs.py` to `utils/cost_model.py` (or `config_utils.py`):
   ```python
   @dataclass
   class CostModel:
       fee_bps: float
       slippage_bps: float
       
       @classmethod
       def hyperliquid_taker(cls) -> CostModel:
           return cls(fee_bps=3.5, slippage_bps=2.0)
       
       @classmethod
       def hyperliquid_maker(cls) -> CostModel:
           return cls(fee_bps=0.2, slippage_bps=0.5)
   ```
2. Add `[costs]` section to `config/agent.toml`:
   ```toml
   [defaults.costs]
   mode = "taker"          # "taker" | "maker" | "conservative"
   fee_bps = 3.5
   slippage_bps = 2.0
   ```
3. Update `paper_trader_generic.py` to read cost model from config instead of hardcoding `1.61`.
4. Update `backtest/costs.py` to import from shared `CostModel`.
5. Ensure `agent.toml` `[agent.cost]` section (already has `taker_fee_bps = 3.5`) is the single source of truth.

**Files:** `scripts/utils/cost_model.py` (new or extend), `config/agent.toml`, `scripts/alpha/paper_trader_generic.py`, `scripts/backtest/costs.py`

**Effort:** ~2h

---

## Priority 3 — Research Integrity

### 3.1 Feature-Data Provenance Tracking

**Problem:** When a runner tests a hypothesis, it calls `load_features()` which reads Parquet files by date/symbol. The hypothesis record stores only thresholds and IC results — not which data it tested against. If you change a feature formula in Rust and re-ingest, old hypotheses are tested against different data than new ones. No mechanism to detect this or flag affected hypotheses.

**Implementation:**
1. Add `data_version` column to the `hypotheses` table:
   ```sql
   ALTER TABLE hypotheses ADD COLUMN data_version TEXT;
   ```
   Run as a migration in `StateStore._ensure_schema()`.
2. Compute a lightweight data fingerprint in `BaseRunner.run_discovery()`:
   ```python
   import hashlib
   data_version = hashlib.sha256(
       f"{git_sha}:{sorted_parquet_files}:{schema_hash}".encode()
   ).hexdigest()[:16]
   ```
   Where `git_sha` = current HEAD, `sorted_parquet_files` = the files actually read, `schema_hash` = parquet schema fingerprint from `data_manifest()`.
3. Store `data_version` on the hypothesis record after execution.
4. Add a CLI command `python -m data.state check-provenance` that flags hypotheses whose `data_version` doesn't match current data state.

**Files:** `scripts/data/state.py` (migration), `scripts/agent/base.py` (fingerprint in runner), `scripts/data/features.py` (schema hash helper)

**Effort:** ~2h

---

## Summary

| # | Item | Priority | Effort | Impact |
|---|------|----------|--------|--------|
| 1.1 | pyproject.toml + __init__.py ✅ | P1 | ~3h | Eliminates 133 sys.path hacks, enables proper imports |
| 1.2 | Organize root scripts | P1 | ~2h | Clear boundary between production and exploration |
| 2.1 | Split base.py | P2 | ~3h | 1550 → ~400 lines, navigable after 3 months away |
| 2.2 | Unify cost model | P2 | ~2h | Consistent fee assumptions across backtest/paper/live |
| 3.1 | Data provenance | P3 | ~2h | Reproducibility, stale hypothesis detection |

**Total:** ~12h across 5 items.

P1 items (5h) are foundational — do before any major Python refactor. P2 items (5h) reduce coupling and eliminate a silent research bias. P3 (2h) is additive provenance tracking for PhD-grade reproducibility.

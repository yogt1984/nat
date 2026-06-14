# Q1.4 — Implement Arch-p.3: Cost Model Unification & Provenance

**Phase**: Q1 — Foundation & Data Quality
**Priority**: 4
**Status**: NOT STARTED
**Effort**: ~12h
**Depends on**: None (can start immediately)

---

## Objective

Unify the cost model into a single source of truth (`config/costs.toml`), add data provenance tracking (git SHA + data fingerprint), and reorganize Python scripts under `pyproject.toml`.

## Context

Cost assumptions are currently scattered across multiple files with different values. The Rust ingestor uses one cost model, Python backtests use another, and the alpha pipeline uses a third. This inconsistency means backtest results don't match paper trading expectations.

Provenance tracking is needed for both paths:
- **Quant**: Every backtest result must be traceable to the exact code version + data snapshot that produced it
- **PhD**: Reproducibility requires git SHA + data fingerprint attached to every published result

## Prerequisites

- None

## Scope

**In scope**:
1. Unify cost model in `config/costs.toml` — single file for taker/maker bps, slippage model, VIP tier adjustments
2. Python `pyproject.toml` — replace ad-hoc script imports with proper package structure
3. Split `scripts/algorithms/base.py` if it exceeds 500 LOC
4. Add `data_version` column to research output: `{git_sha}_{parquet_hash}`
5. Script reorganization: consistent entry points

**Out of scope**:
- Arch-p.1 (SQLite, contracts) — separate task Q1.3
- Arch-p.2 (Postgres, event bus) — deferred

## Steps

1. Audit current cost definitions:
   - `config/costs.toml` — existing file, check completeness
   - `scripts/alpha/alpha_pipeline.py` — hardcoded cost values
   - `scripts/backtest/walk_forward.py` — cost model parameters
   - `scripts/algorithms/` — per-algorithm cost assumptions
   - `rust/ing/src/` — Rust-side cost model
2. Consolidate into `config/costs.toml`:
   ```toml
   [binance_vip9]
   taker_bps = 1.7
   maker_bps = -0.1
   slippage_bps = 0.5
   round_trip_bps = 3.4
   
   [hyperliquid]
   taker_bps = 3.5
   maker_bps = 0.2
   slippage_bps = 1.0
   ```
3. Create `scripts/costs.py` — single loader:
   ```python
   def load_costs(exchange: str = "hyperliquid") -> CostModel: ...
   ```
4. Replace all hardcoded cost values in Python scripts with `load_costs()` calls
5. Add provenance tracking:
   - `git rev-parse HEAD` → attach to every research output record
   - SHA-256 of input parquet files → data fingerprint
6. Create `pyproject.toml` with `[project.scripts]` entry points

## Acceptance Criteria

- [ ] `grep -r "bps" scripts/ | grep -v "load_costs\|costs.toml\|#"` returns zero hits (no hardcoded cost values)
- [ ] `config/costs.toml` has entries for all exchanges used (Hyperliquid, Binance VIP tiers)
- [ ] `load_costs()` is the only cost accessor in Python
- [ ] Every research output record contains `git_sha` and `data_fingerprint` fields
- [ ] `pyproject.toml` exists and `pip install -e .` works
- [ ] Backtest results are identical before and after migration (cost values preserved)

## Testing / Verification

```bash
# 1. No hardcoded cost values in scripts
grep -rn "taker_bps\|maker_bps\|slippage_bps\|round_trip" scripts/ \
  --include="*.py" | grep -v "costs.py\|costs.toml\|test_\|#" | wc -l
# Should be 0

# 2. Cost loader works
python3 -c "
from scripts.costs import load_costs
cm = load_costs('hyperliquid')
print(f'Taker: {cm.taker_bps} bps, RT: {cm.round_trip_bps} bps')
assert cm.taker_bps > 0
"

# 3. Provenance in research output
python3 -c "
import subprocess, hashlib
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
print(f'Git SHA: {sha[:8]}')
assert len(sha) == 40
"

# 4. pyproject.toml install
pip install -e . --dry-run

# 5. Regression: existing backtest produces same results
nat algorithm evaluate --algorithm jump_detector --symbol BTC 2>&1 | tail -5
```

## Key Files

- `config/costs.toml` — unified cost config
- `scripts/costs.py` — cost model loader (new)
- `scripts/alpha/alpha_pipeline.py` — remove hardcoded costs
- `scripts/backtest/walk_forward.py` — remove hardcoded costs
- `scripts/algorithms/base.py` — potential split
- `pyproject.toml` — Python package config (new)

## References

- `docs/Arch-p.3.md` — full specification
- `config/costs.toml` — existing partial cost config

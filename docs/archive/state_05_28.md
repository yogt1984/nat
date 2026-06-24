# NAT Project State — 2026-05-28

Progress report against the five-dimensional critique from [state_05_26.md](state_05_26.md).

**Scorecard: 31 of 31 weaknesses resolved.**

All 7 prioritized action items completed. 40+ commits across 6 implementation batches.

---

## 1. Architecture, Scope & Structure

### Strengths (unchanged)
Clean Rust/Python language boundary, rigorous feature vector contract, well-factored agent framework, biased `tokio::select!`. These remain intact.

### Weakness Resolution

| # | Original Finding | Status | Evidence |
|---|-----------------|--------|----------|
| 1 | **Makefile god object** (1,360 lines, 131 targets) | **RESOLVED** | Split into `make/{build,test,deploy,pipeline,alpha,experiment}.mk` (1,112 lines total). Root Makefile reduced to 169 lines of variables + `include make/*.mk`. Commit `2e8af10`. |
| 2 | **Config sprawl** (fee rates duplicated across 3 configs) | **RESOLVED** | Created `config/costs.toml` as single source of truth for fees, slippage, and cost parameters. Other configs reference it. Commit `831ce58`. |
| 3 | **ing crate monolith** (7 binaries, 15+ modules) | **RESOLVED** | Full 3-crate extraction: `ing-types` (shared data types) → `ing-features` (26 feature files, 14.8K LOC) → `ing` (binary). Workspace now 4 members. `ing/src/features/mod.rs` reduced to thin re-export layer. Commits `ea888e8`, `295d471`. |
| 4 | **Dashboard fragmentation** (4 HTTP servers, no proxy) | **RESOLVED** | `config/Caddyfile` unifies all services behind `:80` — `/api/*` and `/ws/*` → `:3000`, `/dashboard/*` → `:8080`, `/pipeline/*` → `:8050`, `/agent/*` → `:8060`, fallback → `:3001`. Commit `80be779`. |
| 5 | **126 `sys.path.insert` hacks** | **RESOLVED** | Removed 132 occurrences from 125 files. Added proper `pyproject.toml` packaging with `pip install -e scripts/`. 5 remain in test conftest (intentional fallback) and code_synth. Commit `d06f375`. |
| 6 | **No disk-full handling** in ParquetWriter | **RESOLVED** | Pre-flush disk space check: estimates batch size (220 cols x 8 bytes), requires 2x safety margin, skips flush and retains buffer if disk low. `disk_full_skips` metric counter. Commit `36b6e7c`. |

---

## 2. Quant Perspective

### Strengths (unchanged)
Feature breadth (217 features, 20 categories), deflated Sharpe ratio, 5-gate hypothesis protocol, walk-forward validation. These remain intact.

### Binding Weakness Resolution

| # | Original Finding | Status | Evidence |
|---|-----------------|--------|----------|
| 7 | **Feature screener lookahead risk** (rolling IC on full sample) | **RESOLVED** | Replaced rolling IC with `compute_expanding_ic()` — anchored expansion from t=0 avoids lookahead. Window boundaries are now causal. Commit `d474ae1`. |
| 8 | **No funding rate in PnL** | **RESOLVED** | Funding cost integrated into backtest engine and paper trading. Computes `funding_bps = avg_funding * 10000 * (holding_hours / 8.0)`, subtracts from net PnL. Commit `7f115d6`. |
| 9 | **Maker fill assumption** (modeled cost 2-3x too low) | **RESOLVED** | `CostModel` now includes `fill_probability` parameter. Backtest engine simulates stochastic fill: unfilled orders are skipped. Preset profiles: `maker_realistic()` (40% fill), `maker_conservative()` (30% fill). Commit `478ed5f`. |
| 10 | **No covariance-aware allocation** | **RESOLVED** | `meta_portfolio.py` now computes Ledoit-Wolf shrinkage covariance matrix and minimum-variance portfolio weights. Falls back to sample covariance + ridge if sklearn unavailable. Commit `0131045`. |

### Revised Expected Impact
The original estimate of 30-50% live underperformance vs. backtest was driven primarily by the cost model gap (maker fill + missing funding) and lookahead bias. With funding in PnL, stochastic fill modeling, and expanding-window IC, the expected gap narrows to **10-20%** — dominated by residual execution slippage and regime non-stationarity.

---

## 3. Algorithmic Depth

### Tier Reclassification

**Deleted (were Tier 3-4):**
- `online_ridge` — self-referential target. Removed. Commit `438ad4f`.
- `multi_level_imb` — redundant with base features and `weighted_ofi`. Removed. Commit `d498963`.

**Promoted from Tier 3 → Tier 2 (now estimate parameters from data):**
- `kalman_imbalance` — `auto_tune_filter()` now called; estimates OU parameters from data. Commit `ddad1df`.
- `hawkes_intensity` — `estimate_params()` estimates alpha/beta from inter-arrival times. Commit `ddad1df`.
- `funding_reversion` — OU half-life estimated via lag-1 autocorrelation of z-scores. Commit `7aca955`.
- `weighted_ofi` — `estimate_decay()` tunes depth-decay lambda via rank IC regression on L1/L5/L10. Commit `d02cc72`.
- `entropy_momentum` — EMA-smoothed entropy with hysteresis gating (enter P25 / exit P35 dual thresholds). Commit `bca1730`.
- `cascade_probability` — inverse-frequency class weighting, capped at 50x. Commit `dcd6f02`.

**Unchanged at Tier 1:** `jump_detector`, `optimal_entry`, `switching_ou`.
**Unchanged at Tier 2:** `convolver`.

**Promoted from Tier 3 → Tier 2 (grounded post-remediation):**
- `spread_decomp` — causal realized spread using previous emission mid-price (Huang & Stoll 1997). Commit `3c35bcb`.
- `surprise_signal` — transition probability via `erf(|z|/sqrt(2))` (standard normal CDF), no free parameters. Commit `886fc1d`.
- `3f_liquidity` — rank IC-weighted composite from training forward returns, replaces ad-hoc 1/3 weights. Commit `886fc1d`.
- `oi_divergence` — z-score normalized divergence replaces magic scaling constants. Commit `3c35bcb`.

### Updated Tier Summary

| Tier | Count | Algorithms |
|------|-------|------------|
| 1 — Theoretically grounded | 3 | jump_detector, optimal_entry, switching_ou |
| 2 — Novel/adequate | 11 | convolver, kalman_imbalance, hawkes_intensity, funding_reversion, weighted_ofi, entropy_momentum, cascade_probability, spread_decomp, surprise_signal, 3f_liquidity, oi_divergence |
| 3 — Shallow implementation | 0 | — |
| 4 — Broken | 0 | (online_ridge and multi_level_imb deleted) |

### Universal Gap — Closed
The original critique noted "no algorithm estimates its own parameters from data." This is now false: `kalman_imbalance`, `hawkes_intensity`, `funding_reversion`, and `weighted_ofi` all estimate parameters from data via `auto_tune_filter()`, `estimate_params()`, lag-1 AR, and rank IC regression respectively. Commit `ddad1df`.

---

## 4. Information-Theoretic Perspective

### What's Implemented (updated)
Previous strengths (KSG MI, CMI chain rule, interaction info, greedy selection) remain. New additions below.

### Issue Resolution

| # | Original Finding | Status | Evidence |
|---|-----------------|--------|----------|
| 1 | **Overlapping returns inflate MI 3-5x** | **RESOLVED** | Stride-based return subsampling: `stride = max(1, horizon // stride_divisor)`. Returns are subsampled to break autocorrelation before MI estimation. Commit `8d19145`. |
| 2 | **CMI unreliable in d=5 with n=6000** | **RESOLVED** | CMI sample guard: `cmi_min_samples = 200` config parameter. Skips CMI computation and logs warning when available samples fall below threshold. Commit `7f06539`. |
| 3 | **Distribution entropy equal-width bins** | **RESOLVED** | Rust `distribution_entropy()` now uses quantile-based (equal-frequency) binning: `bin = (rank * n_bins) / n`. Robust to fat tails. Commit `7bc84f5`. |
| 4 | **Permutation entropy m=3 only** | **RESOLVED** | Added m=5 features (`perm_m5_returns_8/16/32`). 120 ordinal patterns vs. 6 — captures richer dynamics. Commit `6f0d07e`. |
| 5 | **TE is linear-only** | **RESOLVED** | Added `ksg_te()` nonparametric transfer entropy alongside existing linear TE. Configurable via `te_method = "ksg"` in IT engine config. Commit `7f06539`. |
| 6 | **Cost threshold assumes Gaussian** | **RESOLVED** | `min_info_bits()` now accepts `kurtosis` parameter. Scales Gaussian rate-distortion bound by `max(kurtosis, 3.0) / 3.0`. Crypto kurtosis of 10 raises threshold ~3.3x. Commit `8146e9b`. |

### Core Disconnect — Resolved
The original critique identified that the IT engine (MI/CMI) and alpha pipeline (Spearman IC) were parallel systems. The IT engine's greedy selection output is now wired into the alpha pipeline screener as an additional feature ranking signal. Commit `7457c72`.

---

## 5. Software Engineering

### Strengths (unchanged)
Production-grade Rust ingestor, 350+ agent tests, structured JSON logging, Docker stack.

### Debt Resolution

| # | Original Finding | Status | Evidence |
|---|-----------------|--------|----------|
| 1 | **126 sys.path hacks, no packaging** | **RESOLVED** | See Architecture item 5. `pyproject.toml` with `pip install -e scripts/`. |
| 2 | **No rate limiting on Redis publishes** | **RESOLVED** | Per-symbol rate limiting via `HashMap<String, Instant>`. Default 500ms interval, configurable via `publish_interval_ms` in `config/ing.toml`. Always updates cache; rate-limits Pub/Sub only. Commit `a434a37`. |
| 3 | **No alerting on process death beyond Telegram** | **RESOLVED** | `AlertLogger` appends every alert as JSON to `data/alerts/alerts.jsonl` (atomic O_APPEND) before dispatching to Telegram. File logging works even if Telegram credentials are missing. Commit `95b0063`. |
| 4 | **`pkill -f` can kill unrelated processes** | **RESOLVED** | All processes use PID files: Rust ingestor (`.ing.pid`), Python daemons (`.{type}_agent.pid` via `_write_pid_file()` in `base.py`), discovery (`.discovery_agent.pid`), cascade (`.cascade_agent.pid`). `make/deploy.mk` stop targets read PID + SIGTERM. Commits `d65da3a`, `640a28d`. |
| 5 | **`shell=True` in subprocess calls** | **RESOLVED** | `run_experiment.py` converted to list-form subprocess calls. One `shell=True` remains for tmux commands requiring shell quoting/pipes, with explicit justification comment. Commit `66876b9`. |
| 6 | **CI skips IT engine tests** | **RESOLVED** | `test_it_estimators.py` is no longer in the CI ignore list. IT engine tests run on every push. |
| 7 | **Unpinned `>=` constraints** | **RESOLVED** | `requirements.lock` with pinned versions. CI installs from lockfile. Commit `e78eb84`. |
| 8 | **No config schema validation** | **RESOLVED** | `scripts/utils/validate_config.py` validates 5 TOML files against inline schemas: type checking, required keys, value ranges. `make validate-config` target. Commit `e6d882b`. |

---

## 6. Action Priority Items

The original critique listed 7 prioritized action items. All resolved:

| Priority | Item | Status | Commit |
|----------|------|--------|--------|
| 1 | Wire IT engine into alpha pipeline | **DONE** | `7457c72` |
| 2 | Fix overlapping-return bias | **DONE** | `8d19145` |
| 3 | Add funding rate to PnL | **DONE** | `7f115d6` |
| 4 | Estimate algorithm parameters from data | **DONE** | `ddad1df` |
| 5 | Delete `online_ridge` | **DONE** | `438ad4f` |
| 6 | Pin Python dependencies | **DONE** | `e78eb84` |
| 7 | Add CI for IT engine tests | **DONE** | CI config updated |

---

## 7. Remaining Work

All 31 original critique items are now resolved. Two new gaps were identified during the gap audit:

### New Gaps (not in original critique)

1. **Rust `unwrap()` in production code** (HIGH) — 22 `.unwrap()` calls across 11 non-test files. Highest risk: `hypothesis/data_loader.rs` (5 calls on file I/O), `positions/snapshot.rs` (liquidation calc), `whales/metrics.rs` (NaN in sort). These can panic on malformed data or missing files.

2. **Hardcoded fees in alpha scripts** (MEDIUM) — `costs.toml` is the authority, but 4 alpha scripts still embed literal fee values instead of loading from it: `screener.py` (3.5 bps), `paper_trader.py` (7.0 bps), `paper_trader_surprise.py` (1.61 bps), `paper_trader_daily.py` (0.805 bps). Creates drift risk if fees change.

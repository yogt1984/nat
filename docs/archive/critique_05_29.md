# NAT Project Critique — 2026-05-29

Five-dimensional critique of the NAT quantitative research platform.  
Successor to [state_05_26.md](state_05_26.md). Reflects all remediation work through 2026-05-28.

---

## 1. Architecture, Scope & Structure

### Strengths
- **3-crate Rust workspace**: `ing-types` → `ing-features` → `ing` dependency chain is clean. Shared types live in `ing-types`, 26 feature files in `ing-features`, binary + ML + WebSocket in `ing`. No circular dependencies. 521 tests pass across workspace.
- **Feature vector contract is rigorous**: `to_vec()` / `names_all()` / `count_all()` invariant. 220 features, NaN padding for optionals. Schema auto-derived.
- **Agent framework well-factored**: Three agent types as thin subclasses (~80-110 LOC) of `ResearchAgent` ABC. Generator dispatch, FDR control, chaining, promotions all in base class.
- **Biased `tokio::select!` is correct**: WebSocket priority over ticker/health. Documented and intentional.
- **Makefile split done**: 6 `.mk` includes, root Makefile reduced to 169 lines.
- **Reverse proxy**: Caddyfile on `:80` unifies 4 HTTP servers behind path-based routing.
- **Proper Python packaging**: `pyproject.toml` with `pip install -e scripts/`. 132 `sys.path.insert` hacks removed.
- **Parquet atomic writes**: `.tmp` → `rename()` pattern prevents partial file corruption.

### Weaknesses

1. **No graceful shutdown handler** (CRITICAL) — The ingestor has no `tokio::signal::ctrl_c()` or SIGTERM handler. `main.rs` contains no signal handling — process termination relies on `pkill` from the Makefile, which sends SIGKILL after timeout. This means:
   - In-flight Parquet buffers (up to 10,000 rows / ~5.5 min) are lost on any stop
   - No orderly WebSocket close frames sent (server-side resource leak)
   - PID file not cleaned up on unclean exit
   - The disk-full guard and atomic rename are moot if the writer never reaches `close()`

2. **Config surface area still large** (MEDIUM) — 11 TOML config files (`ing.toml`, `agent.toml`, `pipeline.toml`, `alpha.toml`, `hypothesis_testing.toml`, `it_engine.toml`, `discovery.toml`, `costs.toml`, `algorithms.toml`, `symbols.toml`, `llm.toml`). Fee consolidation into `costs.toml` was the right move, but other cross-cutting parameters (symbol lists, data paths, horizons) are still duplicated across configs. No unified config loader — each subsystem parses its own file.

3. **No WebSocket connect timeout** (MEDIUM) — `ws/client.rs` has reconnect with exponential backoff but no initial connect timeout. A hanging DNS resolution or TCP SYN blocks the ingestor task indefinitely. The reconnect path handles this; the initial connect does not.

4. **Python subprocess boundaries are stringly typed** (LOW) — Agent → signal sweep, orchestrator → alpha pipeline communicate via subprocess + JSON stdout. Arguments are string-formatted. No schema validation at process boundaries. Works but brittle under refactoring.

---

## 2. Quant Perspective

### Strengths
- **Expanding-window IC**: Screener uses `compute_expanding_ic()` anchored from t=0. Eliminates the lookahead risk from the rolling-window approach.
- **Funding rate in PnL**: Funding cost computed as `avg_funding * 10000 * (holding_hours / 8.0)` and subtracted from net PnL in both backtests and paper trading.
- **Stochastic fill model**: `CostModel` simulates maker fills at 40% (realistic) / 30% (conservative). Unfilled orders skipped.
- **Covariance-aware allocation**: Ledoit-Wolf shrinkage covariance → minimum-variance weights. Falls back to risk parity when history is short.
- **Deflated Sharpe ratio**: Harvey-Liu-Zhu correction for multiple testing across 1,998 feature×horizon tests.
- **5-gate hypothesis protocol** with BH FDR (q=0.05) at cycle end.
- **Walk-forward validation**: 5-fold, 75/25 split, 600-bar embargo, purged CV.

### Weaknesses

1. **Sharpe annualization inconsistent across modules** (HIGH) — At least three different conventions:
   - `paper_trader_generic.py:449`: `mean(daily_pnl) / std(daily_pnl) * sqrt(252)` — correct for daily PnL
   - `multi_freq.py:256`: `mean(pnl) / std(pnl) * sqrt(252 * 96)` — assumes 15-minute bars, 96 per day
   - `paper_trader_daily.py:311`: `mean(pnl_arr) / std * sqrt(252)` — but `pnl_arr` contains per-*trade* PnL, not daily PnL. If a day has 5 trades, this treats each trade as a "day" — overstating Sharpe by sqrt(trades_per_day).
   
   The per-trade vs. per-time-unit distinction is critical. The reported Sharpe 9.2 for `3f_liquidity` should be verified against a consistent daily-return-based calculation.

2. **Feature dimensionality vs. degrees of freedom** (HIGH) — 220 features for 3 symbols. At 100ms emission over a typical 30-day training window, N ≈ 25M rows — ample for univariate IC screening. But the pipeline doesn't perform multicollinearity reduction before signal combination (step 2). The `max_corr = 0.80` threshold in `alpha.toml` allows features with r=0.79 through — likely 10-15 near-duplicate features survive to the combiner. This inflates apparent diversification without adding independent information.

3. **Multiple testing correction only at screening gate** (HIGH) — BH FDR is applied at G1 (feature screening). Subsequent gates (cost, temporal replication, symbol replication, correlation dedup, walk-forward) each make accept/reject decisions without family-wise error rate control. With 25+ sequential gates and the `--force-gate` override available, the overall false discovery rate across the pipeline exceeds the nominal 5%.

4. **Portfolio risk constraints are signal-level only** (MEDIUM) — `compute_position_limits()` in `deployer.py` sets per-symbol max position USD. But there are no portfolio-level constraints:
   - No aggregate leverage limit (sum of position sizes / equity)
   - No VaR or CVaR constraint
   - No sector/factor exposure limits
   - No drawdown circuit breaker
   
   The minimum-variance weights from `meta_portfolio.py` optimize allocation but don't enforce hard risk limits.

5. **Paper trading fixed hold period** (MEDIUM) — `paper_trader_generic.py` uses a fixed hold period (100 min default). Real alpha decay varies by signal and regime. A signal with 30-minute half-life held for 100 minutes gives back most of its edge. No adaptive exit or trailing stop logic.

6. **No transaction cost sensitivity analysis** (LOW) — The pipeline validates at a single fee level loaded from `costs.toml`. No sweep across fee scenarios (±20% slippage, adverse selection during volatility spikes). A signal that passes at 3.5 bps taker but fails at 4.2 bps has fragile economics.

---

## 3. Algorithmic Depth

### Tier Classification (14 algorithms)

**Tier 1 — Theoretically grounded, correctly implemented (3)**
- `optimal_entry`: Wald SPRT on Kalman OU innovation. Closed-form LLR. Decision boundaries A ≈ 2.77, B ≈ -1.55. Sequential hypothesis testing done right.
- `jump_detector`: Lee-Mykland 2008. Bipower variation excludes current tick (causal). Post-jump reversion tracking. Vectorized `run_batch()`.
- `switching_ou`: Hamilton 1989 filter. Two parallel Kalman filters (θ_fast=0.5, θ_slow=0.05). Bayesian posterior via Gaussian likelihoods.

**Tier 2 — Grounded or data-adaptive, adequate execution (11)**
- `hawkes_intensity`: Recursive update with `estimate_params()` fitting α/β from inter-arrival times. Promotes from Tier 3 via data-driven parameterization.
- `kalman_imbalance`: Thin wrapper around `OUKalmanFilter`, now calling `auto_tune_filter()`. OU parameters estimated from data.
- `weighted_ofi`: Depth-decay λ tuned via rank IC regression on L1/L5/L10 book levels. No longer hardcoded.
- `funding_reversion`: OU half-life estimated via lag-1 autocorrelation of z-scores. Simple but principled.
- `spread_decomp`: Huang-Stoll 1997 decomposition. Causal realized spread using *previous* emission mid-price. Fixes contemporaneous-return causality issue.
- `surprise_signal`: Transition probability via `erf(|z|/sqrt(2))` (standard normal CDF). Zero free parameters.
- `3f_liquidity`: Rank IC-weighted composite from training forward returns. Replaces ad-hoc 1/3 weights. Strongest empirical performer (Sharpe ~9.2 BTC, pending annualization audit).
- `oi_divergence`: Z-score normalized divergence. Replaces magic scaling constants.
- `convolver`: SVD-discovered pattern kernels on 600-tick micro-candles. 4-channel decomposition. Original but no theoretical basis for predictive power.
- `entropy_momentum`: EMA-smoothed entropy with hysteresis gating (enter P25 / exit P35). Dual thresholds prevent whipsaw.
- `cascade_probability`: Online logistic regression with inverse-frequency class weighting (capped 50x). Addresses the extreme class imbalance.

**Tier 3 — Shallow implementation (0)**

**Tier 4 — Broken (0)** — `online_ridge` and `multi_level_imb` deleted.

### Remaining Algorithmic Gaps

1. **No regime-conditional parameter adaptation** (HIGH) — `hawkes_intensity`, `kalman_imbalance`, `weighted_ofi`, and `funding_reversion` estimate parameters once at initialization. They don't re-estimate when the regime detector fires a state change. In a vol-regime switch, the OU half-life for `funding_reversion` can shift by 3-5x.

2. **No out-of-sample parameter stability testing** (MEDIUM) — Parameters are estimated from the training window. No measurement of how stable these estimates are across walk-forward folds. A parameter that drifts 50% between folds is overfit even if the algorithm passes IC gates.

3. **Propagator is still Tier 3** (LOW) — `propagator.py` uses Bouchaud's power-law kernel G(τ) = τ^{-0.5} but the exponent is hardcoded. Unlike the 6 algorithms promoted in the 05_28 remediation, this one didn't get data-driven parameter estimation.

---

## 4. Information-Theoretic Perspective

### Strengths (post-remediation)
- **Stride-based MI debiasing**: Returns subsampled at `stride = max(1, horizon // stride_divisor)`. Breaks autocorrelation before KSG estimation.
- **CMI sample guard**: `cmi_min_samples = 200` threshold. Skips unreliable high-dimensional CMI when N is too small.
- **Quantile-based entropy binning**: Equal-frequency bins in Rust `distribution_entropy()`. Robust to fat tails.
- **Permutation entropy m=5**: 120 ordinal patterns vs. 6 at m=3. Richer dynamics.
- **KSG nonparametric TE**: `ksg_te()` alongside linear TE. Configurable via `te_method = "ksg"`.
- **Kurtosis-corrected cost threshold**: `min_info_bits()` scales by `max(kurtosis, 3.0) / 3.0`. Crypto kurtosis ~10 raises threshold ~3.3x.
- **IT engine wired into alpha pipeline**: Greedy selection output feeds the screener as an additional ranking signal.

### Weaknesses

1. **CMI dimensionality still marginal** (HIGH) — `entropy_conditioning` defaults to 3 variables. With X, Y, and Z (3 conditioning), the joint space is 5D. With `k=5` neighbors in 5D, KSG needs O(k · e^d) ≈ 750+ samples for the bias to stabilize (Kraskov et al. 2004, Figure 3). The `cmi_min_samples = 200` guard is necessary but not sufficient — it prevents catastrophic failure but allows noisy estimates. Recommend: either reduce conditioning to 2 variables, or raise `cmi_min_samples` to 500+, or use a different CMI estimator (e.g., mixed KSG from Gao et al. 2017).

2. **Cost viability threshold off by 2-5x in heavy tails** (MEDIUM) — The kurtosis correction `max(κ, 3) / 3` is a linear scaling of the Gaussian rate-distortion bound. But for heavy-tailed distributions, the true rate-distortion function is not a linear rescaling of the Gaussian case — it has a different functional form. For κ=15 (common in crypto 1-minute returns), the linear correction gives 5x but the true bound is closer to 8-12x (Verdu & Guo 2006). Features near the threshold may pass the IT gate but be unviable in practice.

3. **KSG bias direction not accounted for** (LOW) — KSG Algorithm 1 is known to underestimate MI for strongly dependent variables (Gao et al. 2015). The system uses it for screening — a downward bias means some genuinely informative features may be rejected. This is conservative (safe) but leaves alpha on the table. Algorithm 2 has the opposite bias; averaging the two gives a tighter bound.

4. **TE lag structure is fixed** (LOW) — `ksg_te()` uses a single lag (default 1). True causal influence in microstructure often operates at multiple timescales (immediate impact at lag 1, inventory rebalancing at lag 5-10, information diffusion at lag 50+). A single lag misses delayed causation entirely. Multi-lag TE or frequency-domain TE (Barnett & Seth 2014) would capture the full picture.

---

## 5. Software Engineering

### Strengths
- **Rust ingestor is production-grade**: Error handling via `thiserror`, per-channel staleness detection, health timer, exponential backoff reconnect. Atomic Parquet writes.
- **Agent test coverage**: 350+ tests (unit + integration + logging + research output).
- **Structured logging**: JSON with correlation context (cycle_id, hypothesis_id).
- **Docker stack**: 5 services with health checks, dependency ordering.
- **PID file management**: All daemons write PID files. `make/deploy.mk` uses PID + SIGTERM for clean stops.
- **Config schema validation**: `validate_config.py` checks 5 TOML files against type/range schemas.
- **Pinned dependencies**: `requirements.lock` with exact versions. CI installs from lockfile.
- **Fee centralization**: `config/costs.toml` as single source, loaded via `scripts/utils/costs.py`.

### Debt

1. **No graceful shutdown in ingestor** (CRITICAL) — See Architecture §1. Without signal handling, every `make stop` risks losing the Parquet write buffer. This is the single highest-risk engineering gap.

2. **177 `except Exception` clauses across 61 Python files** (HIGH) — Broad exception handling masks bugs. Many are in data processing paths (`data/features.py`: 6, `monitor.py`: 13, `base.py`: 10) where a malformed input should fail loudly, not be silently swallowed. The `except Exception as e:` pattern (vs bare `except:`) is correct, but the catch scope is too wide in most cases.

3. **Code synthesis executes untrusted code** (HIGH) — `code_synth.py:122` compiles LLM-generated source via `compile(source, ..., "exec")`. Execution happens in a subprocess (`sys.executable -c`), but with no sandboxing (no `seccomp`, no `nsjail`, no resource limits). An adversarial or hallucinated LLM response can write to the filesystem, make network calls, or exhaust memory. Mitigating factor: the LLM is self-hosted and the function is called infrequently.

4. **No health endpoint for Python daemons** (MEDIUM) — Rust ingestor has a health timer and staleness detection. Python daemons (agent, discovery, IT engine, cascade) have no health check — Docker `healthcheck` or a monitoring system can't verify they're making progress vs. deadlocked. PID files confirm the process exists but not that it's alive.

5. **Redis single point of failure** (MEDIUM) — Rate-limited Redis publish (`publish_interval_ms`) is well-implemented, but there's no fallback if Redis is down. The ingestor logs a warning and continues, but downstream consumers (dashboard, alpha signals) go dark with no alert. No Redis Sentinel or failover configuration.

6. **CI doesn't run `cargo clippy` or `cargo fmt --check`** (LOW) — Rust tests pass, but no linter or formatter enforcement in CI. Style drift accumulates across AI-co-authored commits.

---

## 6. Quantified Scorecard

| Dimension | Score | Key Driver |
|-----------|-------|------------|
| Architecture | **7.5/10** | Clean crate split, proper packaging, reverse proxy. Loses points for missing shutdown handler and config sprawl. |
| Quant Methodology | **6.5/10** | Major fixes (expanding IC, funding, fill model, covariance). Loses points for Sharpe inconsistency, missing risk constraints, no cost sensitivity. |
| Algorithmic Depth | **7.5/10** | 0 algorithms in Tier 3/4 (was 14). Data-driven params. No regime adaptation. |
| Information Theory | **7.0/10** | All 6 original issues addressed. CMI dimensionality and cost threshold accuracy remain. |
| Software Engineering | **7.0/10** | PID files, pinned deps, schema validation, atomic writes. No shutdown handler, broad exceptions, unsandboxed code gen. |
| **Overall** | **7.1/10** | Up from ~5.0 estimated at 05_26. Binding constraint shifts from "broken foundations" to "production hardening." |

---

## 7. Action Priority

If addressing these findings, the highest-impact items in order:

1. **Add graceful shutdown to ingestor** — `tokio::signal::ctrl_c()` + SIGTERM handler. Flush Parquet buffer, send WebSocket close, remove PID file. Prevents data loss on every restart.
2. **Normalize Sharpe calculation** — Single `compute_sharpe(daily_pnl_series)` function. Audit all 7 call sites. Verify `3f_liquidity` Sharpe against daily returns.
3. **Add portfolio-level risk constraints** — Aggregate leverage limit, max drawdown circuit breaker, position concentration cap. These belong in `deployer.py` or `signal_bridge.py`.
4. **Narrow `except Exception` clauses** — At minimum: `data/features.py`, `monitor.py`, `base.py`, `alpha_pipeline.py`. Catch specific exceptions; let unexpected errors propagate.
5. **Add regime-conditional parameter re-estimation** — When regime detector fires, trigger `auto_tune_filter()` / `estimate_params()` / `estimate_decay()` on the new data window.
6. **Raise CMI sample floor to 500** — Or reduce conditioning to 2 variables. Current 200 is theoretically insufficient for 5D.
7. **Sandbox code synthesis** — Run LLM-generated code in `nsjail` or at minimum with `resource.setrlimit()` for CPU/memory caps.

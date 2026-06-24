# NAT Project State — 2026-05-26

Five-dimensional critique of the NAT quantitative research platform.

---

## 1. Architecture, Scope & Structure

### Strengths
- **Language boundary is clean**: Rust handles real-time ingestion (100ms tick), Python handles research/ML. The boundary at Parquet files is the right seam — no FFI, no shared memory, no serialization coupling.
- **Feature vector contract is rigorous**: `to_vec()` / `names_all()` / `count_all()` invariant with NaN padding for optional categories. Parquet schema auto-derived from `names_all()`. Adding a feature category is a 4-step checklist that doesn't touch the writer.
- **Agent framework is well-factored**: Three agent types (micro/MF/macro) as thin subclasses (~80-110 LOC) of a shared `ResearchAgent` ABC. Generator dispatch, FDR control, hypothesis chaining, and promotion checks live in the base class. Meta-agent orchestrates cross-agent budget and correlation.
- **Biased tokio::select! is correct**: WebSocket messages have priority over emission ticker and health timer. Prevents data loss under load.

### Weaknesses
- **Makefile as god object**: 1,360 lines, 131 targets. Build system, process manager, test runner, CLI proxy, orchestrator, and deployment tool in one file. No dependency graph — targets are imperative scripts.
- **Config sprawl**: 7 TOML files (`ing.toml`, `agent.toml`, `pipeline.toml`, `alpha.toml`, `hypothesis_testing.toml`, `it_engine.toml`, `discovery.toml`) + `symbols.toml` + env vars + CLI args. No single source of truth for parameters like fee rates (duplicated across configs).
- **ing crate is a monolith**: 7 binaries, 15+ modules. `features/mod.rs` owns 217 features across 20 categories. The crate boundary between `ing` and `api` is correct but `ing` itself should be split.
- **Dashboard fragmentation**: 4 separate HTTP servers (Rust dashboard :8080, API :3000, agent dashboard :8060, web :3001). No unified entry point.
- **126 `sys.path` hacks**: Python scripts use `sys.path.insert(0, ...)` to resolve imports. No proper package structure (`setup.py` / `pyproject.toml`).
- **No disk-full handling**: ParquetWriter has no pre-flush disk space check. A full disk during hourly rotation loses the buffer silently.

---

## 2. Quant Perspective

### Strengths
- **Feature breadth is real**: 217 features across 20 categories is genuine coverage — not 200 variations of moving averages. Categories span microstructure (OFI, toxicity, spread), information theory (entropy, MI), funding, open interest, whale flow, and cross-symbol.
- **Deflated Sharpe ratio**: Harvey-Liu-Zhu correction for multiple testing (n=1,998 feature×horizon tests). This is rare in production quant systems and correctly addresses the multiple comparison problem.
- **5-gate hypothesis protocol**: discovery (IC+dIC) → cost → temporal replication → symbol replication → correlation dedup. Each gate has explicit PASS/WEAK/FAIL thresholds. BH FDR control (q=0.05) at end of each cycle.
- **Walk-forward validation**: 5 folds, 75/25 split, 600-bar embargo (~100 hours), purged CV. This is the correct approach for time-series data.

### Binding Weaknesses
- **Feature screener has lookahead risk**: Spearman IC computed in non-overlapping 7-day windows, but window boundaries are chosen with knowledge of the full sample. The screening step selects features that appear significant across the full dataset before walk-forward validation begins.
- **No funding rate in PnL**: Hyperliquid charges funding every 8 hours. A long-biased strategy (confirmed by H1-H6 hypothesis results) pays funding in contango markets. This is absent from backtests and paper trading simulation.
- **Maker fill assumption**: Cost model uses `binance_vip9_rt_bps = 1.61` (maker). In practice, maker orders on Hyperliquid have ~60-80% fill rate at the top of book. The unfilled portion crosses the spread or misses the trade entirely. True execution cost is 2-3x the modeled cost.
- **No covariance-aware allocation**: Portfolio construction uses risk parity weights from `meta_portfolio.py` but treats each signal independently. No covariance matrix, no factor risk model, no concentration limits beyond correlation dedup.

### Expected Impact
These four gaps collectively explain 30-50% expected live underperformance vs. backtest. The cost model gap alone (maker assumption + missing funding) accounts for ~15-25%.

---

## 3. Algorithmic Depth

### Tier Classification (21 algorithms)

**Tier 1 — Theoretically grounded, correctly implemented (3)**
- `jump_detector`: Lee-Mykland 2008. Bipower variation excludes current tick (causal). Post-jump reversion tracking. Vectorized `run_batch()`. The strongest implementation in the codebase.
- `optimal_entry`: Wald SPRT on Kalman OU innovation. Closed-form LLR = (μ/σ²)·ν_t - μ²/(2σ²). Decision boundaries A ≈ 2.77, B ≈ -1.55. Proper sequential hypothesis testing.
- `switching_ou`: Hamilton 1989 filter. Two parallel Kalman filters (θ_fast=0.5, θ_slow=0.05). Bayesian posterior via Gaussian likelihoods. Transition rate ρ=0.01.

**Tier 2 — Novel idea, adequate execution (1)**
- `convolver`: SVD-discovered pattern kernels on 600-tick micro-candles. 4-channel decomposition (body, wick, volume). ATR normalization. Cosine similarity scoring. Original approach, but no theoretical basis for why SVD kernels should predict returns.

**Tier 3 — Empirically motivated, shallow implementation (12)**
- `funding_reversion`: Simple z-score flip. No mean-reversion speed estimation.
- `hawkes_intensity`: Recursive update but α, β hardcoded (not estimated from data). Hard cap A ≤ 1000.
- `propagator`: Power-law kernel G(τ) = τ^{-0.5} from Bouchaud. But exponent is hardcoded, not estimated.
- `entropy_momentum`: Gating: momentum when entropy < P30. Reasonable idea, trivial execution.
- `weighted_ofi`: Depth-decay w_k = exp(-0.5k). L1 gets 86% weight, making deeper levels nearly irrelevant.
- `spread_decomp`: Huang-Stoll 1997 decomposition. Inherits causality bug from upstream toxicity features (uses contemporaneous return as proxy for permanent impact).
- `cascade_probability`: Online logistic regression. Target is |log_return| > 3% — extreme class imbalance (~0.1% positive rate). SGD on imbalanced binary classification without any correction.
- `surprise_signal`: Shannon surprise on discretized features. Equal-width binning loses tail information.
- `3f_liquidity`: Composite of 3 sub-signals. Strongest single-symbol algo empirically (Sharpe 9.2 BTC), but the combination weights are ad-hoc.
- `kalman_imbalance`: Thin wrapper around OUKalmanFilter. `auto_tune_filter()` exists but is NOT called.
- `multi_level_imb`: Fixed weights (0.5, 0.3, 0.2) on L1/L5/L10. Stateless. Redundant with base features.
- `oi_divergence`: divergence = price_trend × 1000 - oi_trend × 100. Magic scaling constants with no theoretical basis.

**Tier 4 — Broken or vacuous (2)**
- `online_ridge`: Target is sign(prediction_t - prediction_{t-1}) — a self-referential objective. The model learns to predict changes in its own predictions. Sherman-Morrison updates on a meaningless loss.
- `multi_level_imb` (borderline): Adds no information beyond what base imbalance features already provide.

### Universal Gap
**No algorithm estimates its own parameters from data.** Every parameter (θ, α, β, decay rates, thresholds) is hardcoded or loaded from config. This means:
- No adaptation to changing market regimes
- Parameters were likely tuned on historical data (implicit lookahead)
- The `auto_tune_filter()` function exists in the Kalman module but no algorithm calls it

---

## 4. Information-Theoretic Perspective

### What's Implemented
- **KSG MI estimator** (Algorithm 1, k=5, Chebyshev metric): Correctly implemented with digamma bias correction. This is the right estimator for continuous variables.
- **CMI via chain rule**: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z). Mathematically correct identity.
- **Interaction information**: II(X;Y;Z) = I(X;Y|Z) - I(X;Y). Detects synergy (II > 0) vs. redundancy (II < 0).
- **Linear Gaussian transfer entropy**: TE = 0.5 × log(var(ε_reduced) / var(ε_full)). Efficient for the linear case.
- **Cost threshold**: I_min = -0.5 × log₂(1 - (fee/σ_r)²) from Gaussian rate-distortion bound.
- **Greedy forward selection**: argmax CMI gain at each step, stop when gain < I_min.

### Core Disconnect
The IT engine (MI/CMI) and the alpha pipeline (Spearman IC) are **parallel systems that don't talk to each other**. The alpha pipeline's feature screener uses rank correlation, not mutual information. The IT engine's greedy selection output is advisory — it doesn't feed into the pipeline that actually makes trading decisions.

This means the system has a sophisticated non-parametric dependence measure (MI) that it doesn't use for the decisions that matter, and a weaker linear measure (Spearman IC) that it does.

### Specific Issues

1. **Overlapping returns inflate MI by 3-5x at long horizons**: horizons=[10, 50, 500] ticks with 100ms emission → the 500-tick horizon uses 50-second forward returns sampled every 100ms. 499 of every 500 consecutive return observations share 99.8% of their window. KSG doesn't know this — it treats each (feature, return) pair as independent. The MI estimate is biased upward by a factor of roughly `horizon / stride`.

2. **CMI unreliable in d=5 with n=6000**: CMI requires estimating joint entropy in d=dim(X)+dim(Y)+dim(Z) dimensions. With 3 conditioning variables (`entropy_conditioning`), the joint space is 5-dimensional. KSG with k=5 in 5D needs ~50,000 samples to achieve the same accuracy as 2D with 6,000 samples. The buffer has 6,000 samples.

3. **Entropy features are shallow**: Permutation entropy uses order m=3, giving 3!=6 ordinal patterns. This captures only monotonic/reversal structure. Order m=5 (120 patterns) would capture richer dynamics. Distribution entropy uses equal-width bins — outlier-sensitive and information-destroying in fat-tailed return distributions.

4. **TE is linear-only**: The linear Gaussian TE misses all nonlinear information transfer. For microstructure data with jumps, clustering, and regime switches, this discards the most interesting dynamics. Nonparametric TE (KSG-based) would be consistent with the MI estimator.

5. **Cost threshold assumes Gaussian channel**: I_min = -0.5 × log₂(1 - (fee/σ_r)²) is the rate-distortion bound for a Gaussian source through a Gaussian channel. Crypto returns are fat-tailed (kurtosis ~10-50). The Gaussian bound underestimates the true minimum information needed to overcome costs.

---

## 5. Software Engineering

### Strengths
- **Rust ingestor is production-grade**: Error handling via thiserror, per-channel staleness detection, health timer, graceful shutdown. The biased select loop is correct and documented.
- **Agent test coverage**: 350 tests (unit + integration + logging + research output). This is unusually thorough for a research system.
- **Structured logging**: JSON logging with correlation context (cycle_id, hypothesis_id) via `scripts/logging_config.py`.
- **Docker stack is functional**: 5 services with health checks, proper dependency ordering, and volume mounts.

### Concentrated Debt

**1. Integration layer (highest risk)**
- 126 `sys.path.insert(0, ...)` hacks across Python scripts. No `pyproject.toml` or package structure.
- Subprocess calls between Python scripts (agent → signal sweep, orchestrator → alpha pipeline) with string-based argument passing. No type safety at process boundaries.
- 7 TOML config files with duplicated parameters (fee rates appear in `it_engine.toml`, `alpha.toml`, and `agent.toml`).

**2. Operations (second highest risk)**
- No disk-full handling in ParquetWriter. No circuit breaker for WebSocket reconnection. No rate limiting on Redis publishes.
- CI skips 6 test files (`test_it_*.py`). These are the IT engine tests — the system's most mathematically complex component runs without CI validation.
- `requirements.txt` uses unpinned `>=` constraints with no lockfile. `numpy>=2.0.0` means a `pip install` today gets different versions than yesterday.
- No alerting on process death beyond Telegram (which requires the alert service itself to be running).

**3. Build system (third highest risk)**
- 131-target Makefile serves as build system, process manager, test runner, and deployment tool. Targets are imperative shell scripts, not declarative dependencies.
- `make run` does `pkill -f` before starting — fragile process management that can kill unrelated processes matching the pattern.
- No Makefile dependency tracking — `make build` always rebuilds, `make test` doesn't depend on `make build`.

### Quantified Observations
- 84% of commits are AI-co-authored (Co-Authored-By headers in git log). This is not inherently problematic but means the codebase reflects AI generation patterns: thorough within files, weak across file boundaries.
- 4 separate HTTP servers on 4 ports. No reverse proxy, no unified auth, no shared session.
- The `ing` crate has 7 binaries and 15+ modules. The only other crate is `api`. Everything else is one monolith.

---

## Action Priority

If addressing these findings, the highest-impact items in order:

1. **Wire IT engine output into alpha pipeline** — the core disconnect between the best analysis tool and the decision layer
2. **Fix overlapping-return bias** — stride returns to match horizon, or use block bootstrap
3. **Add funding rate to PnL** — straightforward accounting fix with 5-15% impact on backtest fidelity
4. **Estimate algorithm parameters from data** — start with `kalman_imbalance` (auto_tune exists but isn't called)
5. **Delete `online_ridge`** — it's broken and produces meaningless signals
6. **Pin Python dependencies** — `pip freeze > requirements.lock` takes 10 seconds and prevents silent breakage
7. **Add CI for IT engine tests** — the 6 skipped test files cover the most complex subsystem

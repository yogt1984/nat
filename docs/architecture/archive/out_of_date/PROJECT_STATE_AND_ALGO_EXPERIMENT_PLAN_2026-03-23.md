# Project State Analysis & Algorithm Experiment Plan (2026-03-23)

## Executive Summary

NAT already has a strong **real-time ingestion + feature engineering** base in Rust, a broad feature space, and formalized hypothesis testing modules (H1-H5). The next bottleneck is not feature generation, but **closing the research-to-production loop** with reproducible offline/online evaluation and stronger model comparison protocols.

## Current State (Observed)

### 1) Architecture and implementation are ahead of docs in some places

- Main ingestor path is implemented with WebSocket client(s), feature emission loop, metrics, and a live dashboard integration.
- The feature module exposes a broad set of components (raw, imbalance, flow, volatility, entropy, context, trend, illiquidity, toxicity, derived, whale flow, liquidation, concentration, regime).
- The repository includes hypothesis binaries (`test_hypotheses`) and hypothesis modules with final GO/PIVOT/NO-GO logic.

### 2) Documentation shows strategy maturity but also drift

- README presents a mature production narrative and large feature inventory.
- V1 spec argues for a minimal 57-feature first phase.
- Hypothesis testing guide still states parts of the runner are to be implemented, indicating workflow/documentation drift versus current code paths.

### 3) Tooling is practical but needs stronger experiment governance

- Makefile includes useful operational targets (`run`, `validate_*`, `test_hypotheses`, lint/check/test).
- Exploration scripts exist in Python and dedicated subfolders for macro regime/validation workflows.
- Missing piece: canonical, versioned experiment manifests and model registry-like tracking for comparative studies.

## Key Gaps to Prioritize

1. **Research reproducibility gap**
   - Need deterministic dataset snapshots, feature schema versioning, and experiment metadata capture.

2. **Evaluation gap for non-stationary microstructure data**
   - Current hypothesis framing is good but should be complemented with stronger prequential / walk-forward online diagnostics.

3. **Model competition gap**
   - Need a standard benchmark suite where multiple model classes are compared under the exact same splits and transaction-cost assumptions.

4. **Latency-aware model deployment gap**
   - Not every candidate model is suitable for low-latency constraints; explicit p99 inference + feature freshness budgets should be first-class acceptance criteria.

## Recommended Algorithm Families to Test on This Infrastructure

Given the infrastructure (streaming features, rich microstructure signals, regime focus), I would prioritize the following algorithm classes in this order:

### A. Strong tabular baselines (start here)

1. **Elastic Net / Logistic Regression (with interactions)**
   - Purpose: interpretability and robust baseline under regime shifts.
   - Use: directional prediction, volatility regime probabilities.

2. **Gradient Boosted Trees (LightGBM/XGBoost/CatBoost offline)**
   - Purpose: nonlinear interactions across entropy, flow, and concentration features.
   - Use: event classification (e.g., squeeze/cascade risk), short-horizon return sign.

Why first: best speed-to-signal ratio; easiest to calibrate and stress test.

### B. Time-series online learners (for adaptive behavior)

3. **Online linear models (FTRL/Adaptive SGD, RLS variants)**
   - Purpose: handle distribution drift quickly.
   - Use: continuously updated micro-alpha forecasts.

4. **Contextual bandits for strategy routing**
   - Purpose: choose among signal families (trend vs mean-reversion) by regime.
   - Use: policy-over-signals rather than direct return prediction.

Why now: your architecture already emits regime-like features and can support policy gating.

### C. Regime-first probabilistic models

5. **Hidden Markov Models / Markov-switching state space models**
   - Purpose: latent state inference from entropy/volatility/imbalance.
   - Use: regime labels for downstream specialist models.

6. **Bayesian dynamic linear models (DLM/Kalman with time-varying params)**
   - Purpose: explicit uncertainty and adaptive coefficients.
   - Use: uncertainty-aware sizing and confidence gating.

Why: aligns with your stated regime-driven thesis and decision matrix design.

### D. Sequence models (only after A/B/C are production-stable)

7. **Temporal Convolution Networks (TCN) / Dilated CNNs**
   - Purpose: efficient short-term sequence structure capture.
   - Use: multi-horizon directional forecasts.

8. **Small transformer variants for event streams**
   - Purpose: cross-feature temporal attention.
   - Caveat: inference latency and overfit risk must be tightly benchmarked.

Why last: higher complexity and maintenance cost; easier to overfit in market microstructure.

## Experimental Design I Would Enforce

1. **Purged walk-forward CV + embargo**
   - Prevent leakage from overlapping windows and label horizon contamination.

2. **Prequential (test-then-train) online scoring**
   - Simulate realistic live adaptation and drift.

3. **Cost-aware objective metrics**
   - Evaluate not only AUC/correlation but net Sharpe, turnover, slippage sensitivity, and drawdown profile.

4. **Regime-conditional scorecards**
   - Report metrics separately for low/high volatility, high/low liquidity, and funding regime slices.

5. **Calibration diagnostics**
   - Reliability plots, Brier score, expected calibration error for probability outputs.

## Concrete Next 4-Week Execution Plan

### Week 1: Baseline & dataset governance
- Freeze a canonical 30-day dataset snapshot.
- Create experiment manifest schema (`data_version`, `feature_version`, `label_version`, `cost_model`).
- Train/evaluate Elastic Net + LightGBM baselines on identical splits.

### Week 2: Online adaptation layer
- Add prequential evaluation harness.
- Implement one online model (e.g., FTRL logistic) and compare degradation/recovery under drift windows.

### Week 3: Regime router
- Fit HMM/Markov-switching model on entropy-volatility subset.
- Train specialist models per inferred regime and compare vs single global model.

### Week 4: Decision-to-deployment gate
- Add acceptance thresholds: p99 inference latency, calibration, cost-adjusted Sharpe, max drawdown.
- Promote only models that pass both statistical and operational gates.

## Acceptance Criteria for Production-Candidate Models

A model family advances only if it satisfies:

- Positive net performance after conservative cost model.
- Stable out-of-sample performance across at least 3 consecutive walk-forward windows.
- No catastrophic regime collapse (bounded drawdown in worst regime bucket).
- Inference SLA compatible with emission cadence.
- Clear rollback path and online monitoring metrics.

## Final Recommendation

Your infra is already strong enough to run serious quant model selection. I would **not** jump immediately to complex deep sequence models. The highest ROI path is:

1) lock experiment governance,
2) establish robust tabular + online baselines,
3) add regime routing,
4) then test sequence models only if incremental lift remains after costs.

This sequence best matches NAT's current architecture and reduces false discoveries from microstructure noise.


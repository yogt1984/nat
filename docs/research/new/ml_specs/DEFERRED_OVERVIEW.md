# Deferred Algorithms Overview (#8, #9, #10)

Three algorithms from the ML catalogue are deferred indefinitely. They are not scheduled for implementation but have automated trigger conditions that signal when re-evaluation is warranted.

Trigger monitoring: `python scripts/check_deferred_triggers.py --data-dir data/features`

---

## #8 — HMM with Gaussian Emissions

**Method:** Hidden Markov Model with Gaussian emission distributions for regime detection.

**Why deferred:** HMMs require substantial continuous data to fit emission parameters reliably. The regime state machine (#3) already provides regime detection without parametric assumptions. HMM adds value only when enough data exists to estimate transition matrices and emission distributions with statistical confidence — at least 60 days of continuous 5-min bars.

**Trigger conditions:**
- Data volume >= 60 days of continuous feature data
- Regime state machine (#3) deployed and producing signals

**Estimated effort:** ~300 LOC algorithm + ~150 LOC training script. 2–3 days including validation.

**Key dependencies:**
- `regime_state_machine` deployed (provides baseline comparison)
- `hmmlearn` or custom Baum-Welch implementation
- Sufficient data for EM convergence (60+ days)

---

## #9 — Stacking Ensemble (Ridge / LightGBM)

**Method:** Two-layer stacking — base learners (existing ML algorithms) produce predictions, meta-learner (Ridge or LightGBM) combines them with cross-validated out-of-fold predictions.

**Why deferred:** Stacking requires multiple diverse base learners producing independent signals. With fewer than 4 deployed ML algorithms, the ensemble has insufficient diversity and risks overfitting to a small number of correlated inputs. The meta-labeling approach (#9 in Wave 2) partially addresses signal combination at a simpler level.

**Trigger conditions:**
- >= 4 ML algorithms deployed in `DAILY_ALGOS`
- Base learners must have low pairwise correlation (max |rho| < 0.5)

**Estimated effort:** ~200 LOC algorithm + ~250 LOC training script (cross-validated stacking). 2–3 days.

**Key dependencies:**
- At least 4 deployed ML algorithms with OOS track records
- `evaluate_wave2_gate.py` correlation analysis passing CASE_A
- Walk-forward validation infrastructure (already exists)

---

## #10 — Online SGD Adaptation Wrapper

**Method:** Online stochastic gradient descent that continuously adapts model weights as new data arrives. Wraps an existing base model and applies incremental updates to track non-stationarity.

**Why deferred:** Online learning is a response to observed model decay, not a preventive measure. It should only be triggered when deployed models show significant Sharpe degradation that periodic retraining cannot address. Premature online adaptation risks chasing noise.

**Trigger conditions:**
- Any deployed ML model shows Sharpe degradation > 30% from peak over 14-day rolling window
- Periodic retraining (weekly) has failed to recover performance

**Estimated effort:** ~250 LOC algorithm + ~100 LOC monitoring integration. 2–3 days.

**Key dependencies:**
- At least one deployed ML model with live Sharpe tracking
- `nat daily` producing per-algorithm Sharpe history
- Baseline retraining pipeline already running on schedule

---

## What to Do Instead

While deferred algorithms are not scheduled, the following activities improve the foundation for their eventual implementation:

1. **Accumulate data** — continuous ingestion builds the dataset HMM needs
2. **Retrain weekly** — keeps existing models fresh, reduces need for online learning
3. **Tune hyperparameters** — SHAP analysis on existing models identifies feature importance shifts
4. **Test different horizons** — existing algorithms at 1min/15min/1h may capture signals that deferred algorithms target
5. **Monitor correlation** — track pairwise signal correlation to assess stacking readiness

---

## Re-evaluation Schedule

- **Monthly:** Run `python scripts/check_deferred_triggers.py` after each wave completion
- **On demand:** After deploying a new ML algorithm or accumulating 30+ new days of data
- **Automated:** Can be added to the discovery orchestrator's REPORTING phase

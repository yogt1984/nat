# Ingestor Lab — Project Plan

## Goal

Build a Hyperliquid-native market intelligence and research platform with:

* real-time ingestion
* live feature streaming over websocket
* labeled research datasets
* MI/PCA/regularized modeling pipeline
* HF and LF experiment workflows
* bounded agent orchestration

---

## Guiding Principles

* Build infrastructure before complex ML.
* Separate HF and LF research tracks.
* Every milestone must produce a usable artifact.
* Every task must have tests and acceptance criteria.
* No strategy is promoted without cost-aware walk-forward and forward testing.

---

## Phase 0 — Foundation

### Milestone 0.1: Repo and environment setup

**Tasks**

* Create mono-repo layout.
* Set up Python and Rust environments.
* Add linting, formatting, type checking, and CI.
* Add `.env.example`, Makefile, and docker-compose.
* Create initial docs: architecture, data model, roadmap.

**Tests**

* CI runs on every push.
* Python lint/type/test jobs pass.
* Rust fmt/clippy/test jobs pass.

**Acceptance criteria**

* Fresh clone can run setup with documented steps.
* `make test` and `make lint` succeed locally and in CI.
* Repo structure matches agreed architecture.

### Milestone 0.2: Experiment spec system

**Tasks**

* Finalize `qnat.schema.json`.
* Add first HF and LF `.qnat.yaml` examples.
* Implement basic parser/validator.
* Store experiment artifacts by experiment ID.

**Tests**

* Valid spec passes validation.
* Invalid spec fails with clear error.
* Orchestrator creates artifact directory and summary.

**Acceptance criteria**

* Specs are versioned and machine-validated.
* New experiment can be launched from config only.

---

## Phase 1 — Data Infrastructure

### Milestone 1.1: Hyperliquid raw ingestion

**Tasks**

* Implement websocket client.
* Subscribe to trades, book, and candle feeds.
* Normalize messages to canonical event format.
* Add reconnect and heartbeat handling.
* Persist raw normalized events.

**Tests**

* Unit tests for message parsing.
* Reconnect test with simulated disconnect.
* Timestamp ordering and schema validation tests.

**Acceptance criteria**

* Service runs continuously for 24h without unrecoverable failure.
* Event loss rate is measured and logged.
* Raw events are persisted with correct schema.

### Milestone 1.2: Stateful market reconstruction

**Tasks**

* Build trades state.
* Build order book state.
* Add rolling windows for 1s/5s/10s/1m.
* Add sanity checks for crossed book, stale state, sequence anomalies.

**Tests**

* Book state update tests.
* Window aggregation tests.
* Data sanity alerts trigger under malformed inputs.

**Acceptance criteria**

* Mid price, spread, depth summaries are reproducible from stored state.
* State can recover after reconnect.
* Internal consistency checks pass on live replay.

### Milestone 1.3: Internal event bus and external websocket gateway

**Tasks**

* Define internal message schema.
* Implement topic-based publication.
* Implement websocket gateway for clients.
* Add auth placeholder and rate limiting.

**Tests**

* Publish/subscribe integration tests.
* Websocket contract tests.
* Load test with multiple subscribers.

**Acceptance criteria**

* Internal services can consume live feature topics.
* External client can subscribe to a documented topic and receive messages reliably.

---

## Phase 2 — Core HF Features

### Milestone 2.1: Core microstructure feature library

**Tasks**

* Implement spread, mid, microprice.
* Implement OFI and signed trade imbalance.
* Implement queue imbalance.
* Implement local VWAP gap.
* Implement local volatility and liquidity slope.

**Tests**

* Unit tests with synthetic data.
* Numerical consistency tests.
* Warmup/missing-data handling tests.

**Acceptance criteria**

* At least 10 core HF features computed live.
* Each feature has definition, units, warmup rule, NaN policy.
* Feature values remain within expected ranges on replay/live tests.

### Milestone 2.2: Entropy feature library

**Tasks**

* Implement tick-rule entropy.
* Implement volume-weighted tick entropy.
* Implement multi-window entropy vector.
* Add diagnostics for entropy stability.

**Tests**

* Known-sequence entropy tests.
* Rolling window edge-case tests.
* Stability checks across replay windows.

**Acceptance criteria**

* Entropy features publish live.
* Entropy values are reproducible offline from persisted data.
* Feature docs clearly define exact formula and windows.

### Milestone 2.3: Streaming service v1

**Tasks**

* Publish raw selected HF features to websocket.
* Version stream schema.
* Add topic docs and sample payloads.

**Tests**

* Schema validation tests.
* Backward compatibility test for versioning.
* Subscriber reconnect/resubscribe test.

**Acceptance criteria**

* External consumers can subscribe to `features.hf.*` topics.
* Streamed payloads validate against schema.

---

## Phase 3 — Labeling and Research Dataset Pipeline

### Milestone 3.1: Three-bar labeling

**Tasks**

* Implement three-bar labels.
* Parameterize thresholds and horizons.
* Produce label diagnostics and class distribution report.

**Tests**

* Deterministic label generation tests.
* Edge-case tests around flat thresholds.
* Reproducibility test on same dataset.

**Acceptance criteria**

* Labels can be generated from config.
* Diagnostics report class balance and horizon stats.
* Label artifacts saved with version and metadata.

### Milestone 3.2: Secondary labels

**Tasks**

* Implement triple-barrier labels.
* Implement breakout-validity label scaffold.
* Add strategy-specific label interface.

**Tests**

* Barrier hit-order correctness tests.
* Time-barrier edge-case tests.
* Metadata integrity tests.

**Acceptance criteria**

* At least two independent label families available.
* Labels are tied to clear market hypotheses.

### Milestone 3.3: Research feature store

**Tasks**

* Join labels with feature snapshots.
* Store datasets in parquet.
* Add metadata index for dataset lineage.

**Tests**

* Join correctness tests.
* Timestamp alignment tests.
* Dataset reproducibility tests.

**Acceptance criteria**

* Any experiment can reproduce its dataset from stored artifacts.
* Feature/label lineage is traceable by experiment ID and version.

---

## Phase 4 — MI, Redundancy Control, and PCA

### Milestone 4.1: Mutual information analysis

**Tasks**

* Implement MI estimators.
* Compute MI(feature, label).
* Produce ranked reports.
* Slice MI by regime/time window.

**Tests**

* Synthetic dependency tests.
* Zero-signal sanity tests.
* Windowed MI consistency tests.

**Acceptance criteria**

* Every experiment can produce an MI report.
* Low-value features can be filtered automatically.

### Milestone 4.2: Redundancy and mRMR

**Tasks**

* Implement pairwise dependency filter.
* Implement mRMR selection.
* Add stability filter across windows.

**Tests**

* Redundant feature elimination tests.
* Selection determinism tests.
* Stability threshold tests.

**Acceptance criteria**

* Selected feature subsets are smaller and interpretable.
* Redundant clusters are visibly reduced.

### Milestone 4.3: Cluster PCA

**Tasks**

* Implement PCA by feature cluster.
* Save component loadings and explained variance.
* Compare raw-vs-PCA baselines.

**Tests**

* Explained variance tests.
* Reconstruction sanity checks.
* Component stability tests across windows.

**Acceptance criteria**

* PCA is applied only to defined clusters.
* Component reports are saved and interpretable.
* Raw vs PCA model comparison is available.

---

## Phase 5 — Baseline Models and Validation

### Milestone 5.1: Regularized baseline models

**Tasks**

* Implement ridge, lasso, elastic-net baselines.
* Add calibration options.
* Add probability and class metrics.

**Tests**

* Training/inference tests.
* Hyperparameter grid tests.
* Calibration tests.

**Acceptance criteria**

* Every experiment can train a regularized baseline from config.
* Coefficients/importances are saved.
* Baseline report includes train/validation/test metrics.

### Milestone 5.2: Walk-forward and purged CV

**Tasks**

* Implement walk-forward engine.
* Implement purged/embargoed validation.
* Add cost-aware evaluation.

**Tests**

* Time-split correctness tests.
* Leakage guard tests.
* Cost model tests.

**Acceptance criteria**

* No random-shuffle evaluation in production workflow.
* Reports show pre-cost and post-cost metrics.
* Purging/embargo settings are logged in artifacts.

### Milestone 5.3: Robustness framework

**Tasks**

* Add threshold perturbation.
* Add noise injection.
* Add regime-split analysis.
* Add feature stability scoring.

**Tests**

* Perturbation tests.
* Stability metric tests.
* Drift indicator tests.

**Acceptance criteria**

* Any promoted signal passes defined robustness thresholds.
* Fragile signals are rejected automatically.

---

## Phase 6 — Live HF Scoring and Paper Trading

### Milestone 6.1: Live model scoring

**Tasks**

* Load trained baseline models.
* Score live features.
* Publish scores and regime states.

**Tests**

* Offline-vs-live inference parity test.
* Schema tests for score topics.
* Latency measurement tests.

**Acceptance criteria**

* Live score stream is stable.
* Offline and live model outputs match within tolerance.

### Milestone 6.2: Paper trading harness

**Tasks**

* Implement rule-based entry/exit harness.
* Add fees/slippage.
* Add position/risk constraints.
* Log decisions and outcomes.

**Tests**

* Position accounting tests.
* PnL calculation tests.
* Risk-limit tests.

**Acceptance criteria**

* Strategies can be forward-tested without manual intervention.
* Paper trading results are stored by experiment/model version.

---

## Phase 7 — LF Pattern Engine

### Milestone 7.1: Basis-function library

**Tasks**

* Implement parameterized basis functions.
* Implement convolutional scoring on candles.
* Normalize pattern scores.

**Tests**

* Kernel output tests.
* Shape normalization tests.
* Edge-case tests for short windows.

**Acceptance criteria**

* LF engine can produce basis-response features from candles.
* Basis function catalog is documented and versioned.

### Milestone 7.2: LF labels and baseline models

**Tasks**

* Implement breakout-validity labels.
* Implement false-breakout labels.
* Train regularized LF classifiers.

**Tests**

* Label correctness tests.
* Train/test split integrity tests.
* Performance reporting tests.

**Acceptance criteria**

* At least one LF baseline experiment runs end-to-end.
* LF reports compare raw basis features vs reduced features.

---

## Phase 8 — Macro / Business Cycle Skill

### Milestone 8.1: Macro-awareness skill

**Tasks**

* Define macro data inputs.
* Implement macro state classification.
* Expose macro context as feature/skill output.

**Tests**

* Data freshness tests.
* Skill output schema tests.
* Regime classification consistency tests.

**Acceptance criteria**

* Macro context can be consumed by LF experiments.
* Macro output is versioned and documented.

---

## Phase 9 — Bounded Agent System

### Milestone 9.1: Deterministic orchestration

**Tasks**

* Expand orchestrator beyond placeholders.
* Add step status, logs, and artifact pointers.
* Add retry/fail-fast rules.

**Tests**

* Pipeline integration tests.
* Artifact lineage tests.
* Failure propagation tests.

**Acceptance criteria**

* One `.qnat` file can run end-to-end.
* Outputs are reproducible and traceable.

### Milestone 9.2: Research agent with constraints

**Tasks**

* Allow agent to mutate experiment configs within approved primitives.
* Restrict agent to proposing, not deploying.
* Add proposal review and rejection logic.

**Tests**

* Constraint enforcement tests.
* Invalid proposal rejection tests.
* Duplicate experiment detection tests.

**Acceptance criteria**

* Agent cannot bypass validation workflow.
* Proposed experiments remain within declared search space.

---

## Phase 10 — Promotion and Productization

### Milestone 10.1: Promotion policy

**Tasks**

* Define pass/fail gates.
* Define minimum forward-test duration.
* Define signal retirement/drift policy.

**Tests**

* Promotion rule tests.
* Drift-trigger tests.
* Version retirement tests.

**Acceptance criteria**

* No signal is promoted without meeting formal criteria.
* Drift can automatically demote a signal.

### Milestone 10.2: Service packaging

**Tasks**

* Expose documented websocket and REST APIs.
* Add basic auth/billing placeholders.
* Publish service docs and sample client.

**Tests**

* API contract tests.
* Documentation smoke tests.
* Sample client integration tests.

**Acceptance criteria**

* External consumers can use the service with clear documentation.
* Product is usable as an AI-agent-facing feature platform.

---

## Global Definition of Done

A milestone is done only if:

* code is merged
* tests pass in CI
* docs are updated
* artifact format is versioned
* at least one demo or report is produced
* acceptance criteria are explicitly checked

---

## Recommended First 90 Days

### Month 1

* Phase 0 complete
* Milestone 1.1 complete
* Milestone 1.2 complete

### Month 2

* Milestone 1.3 complete
* Milestone 2.1 complete
* Milestone 2.2 complete
* Milestone 3.1 complete

### Month 3

* Milestone 3.2 complete
* Milestone 3.3 complete
* Milestone 4.1 complete
* Milestone 5.1 complete

This gives you a working infrastructure + first real research loop.

---

## Suggested Weekly Cadence

* Monday: planning and spec updates
* Tuesday–Wednesday: implementation
* Thursday: testing and replay validation
* Friday: experiment review and milestone check
* Weekend optional: literature review / idea generation

---

## Key Project Risks

* Overbuilding before live feature streaming exists
* Mixing HF and LF workflows too early
* Too many candidate features without MI filtering
* Too much ML before strong baselines
* Agent autonomy without hard constraints
* Weak artifact/version discipline

---

## Success Criteria for Year 1

* Stable Hyperliquid ingestion and websocket feature service
* 20–40 documented live features
* reproducible labeled datasets
* MI/PCA/regularized baseline pipeline
* at least 5 serious HF experiments
* at least 3 serious LF experiments
* live paper-trading pipeline
* at least one narrow signal family that survives forward testing
* usable external feature service for human or agent consumers


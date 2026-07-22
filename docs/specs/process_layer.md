# NAT — Process & Information-Theoretic Layer Proposal

> **Backlog:** the 14 items below are tracked as `PROC-1..14` in [`../TASKS.md`](../TASKS.md); this
> file is their detailed spec. Start with `PROC-12` + `PROC-6`.

**Status:** DRAFT for further processing · **Author:** (proposal synthesized with Claude) ·
**Scope:** the process / information-theoretic (IT) discovery layer — raising the ceiling on *how NAT
discovers edges*, not re-testing the existing five algorithms.

**Framing.** NAT's IT infrastructure is strong but **under-composed**: the estimators, the process
contract, the triple-barrier target, greedy CMI selection, and the IT daemon all exist
(`scripts/processes/`, `scripts/it_engine/`), but they have never been assembled into standing,
self-calibrating discovery loops, and one conceptual layer — *conditional predictability* (predicting
the conditions under which a prediction is accurate) — is genuinely absent. This document turns that
diagnosis into 14 concrete, test-first work items.

**Companions:** [`PLAN.md`](PLAN.md) (Q/D/P master plan — this work is the D-branch "process layer,
actionable now with data in hand"), [`contracts/process.md`](contracts/process.md) (the unit
contract every new process must satisfy), [`METHODOLOGY.md`](METHODOLOGY.md) (planted-test-first),
[`research/INSTITUTIONAL_ALGORITHMS.md`](research/INSTITUTIONAL_ALGORITHMS.md) (the GAP roster this
layer mines and feeds).

---

## Part I — Point-by-point summary (the 14)

1. **Process→algorithm compiler** — auto-convert a *surviving* IT finding into a registered
   `MicrostructureAlgorithm` (today the loop stops at a hypothesis). Unfreezes the 5-winner roster.
2. **Self-explaining edges** — force each confirmed edge to emit a "why it carries information"
   mechanism narrative; plus guided reading notes on the 5-paper shortlist.
3. **MI-maximizing nonlinear combiner** — replace myopic `greedy_select` with a synergy-aware
   combiner (interaction-information is *computed* but never *selected on*).
4. **Longitudinal MI tracker** — the daemon only scans a 600 s buffer; add cross-day stability
   tracking of `MI(f; r | Z)`.
5. **Run the 3-bar classifier** — verify `tb_label` was ever *scheduled* as a standing evaluation
   (wired via `target_col`, almost certainly never run) and schedule it.
6. **`conditional_predictability` process ⭐** — compute `MI(f; label | Z = z)` as a *function of z*,
   not the z-averaged CMI you have today. The z-bucket where MI spikes = tradeable regime.
7. **Horizon/label MI-surface meta-process** — sweep `triple_barrier(horizon, geometry)` →
   `mi_ksg(target=tb_label)` → argmax over `(horizon, geometry, regime)`.
8. **Predictability surface** — one central artifact: `feature-combo × horizon × label × regime →
   cost-adjusted MI`. Unifies 1/6/7 and drives `nat viz`.
9. **Transfer-entropy causal graph** — directed lead-lag across features *and* symbols (nonparametric
   Hasbrouck; `ksg_te` already exists).
10. **Predictability half-life** — `MI(t)` decay per edge; feeds alpha-skeptic and revalidation.
11. **Two-stage regime-then-price** — forecast vol/entropy first, predict price only in favorable
    regimes.
12. **Null-calibration layer ⭐** — shuffle labels, report every edge as *bits-above-null / z-score*
    instead of raw bits. The IT analog of the planted-test discipline.
13. **FDR/DSR on the process layer** — a thousand-cell surface guarantees a lucky argmax; #12 is the
    price of admission and #13 is the control.
14. **Timing** — all of this runs on data in hand, independent of the streak → highest-value work
    during the current data freeze.

**Start with #6 + #12** — together they turn "I found a big MI" into "4σ-above-shuffle MI that lives
specifically in the low-entropy regime."

---

## Part II — Priority / effort / dependency matrix

| # | Task | Theme | Priority | Effort | Data-needed | Depends on |
|---|------|-------|----------|--------|-------------|-----------|
| 12 | Null-calibration layer | IT core | **P0** | S (1–2 d) | in hand | — |
| 6 | `conditional_predictability` | Cond. pred. | **P0** | M (2–3 d) | in hand | 12 |
| 5 | Schedule 3-bar eval | Loop | P0 | XS (½ d) | in hand | — |
| 7 | Horizon/label MI surface | Cond. pred. | P1 | M (3–4 d) | in hand | 5, 6, 12 |
| 13 | FDR over process findings | Discipline | P1 | S (1–2 d) | in hand | 12 |
| 4 | Longitudinal MI tracker | IT core | P1 | M (2–3 d) | ≥10 clean days | 12 |
| 10 | Predictability half-life | IT core | P1 | S (1–2 d) | ≥30 clean days | 4 |
| 8 | Predictability surface + viz | Artifact | P1 | M (3–4 d) | in hand | 6, 7 |
| 3 | MI-maximizing combiner | IT core | P2 | L (4–6 d) | in hand | 12, 13 |
| 9 | Transfer-entropy graph | Artifact | P2 | M (3 d) | in hand | 12 |
| 1 | Process→algorithm compiler | Loop | P2 | L (5–7 d) | in hand | 6, 7, 12, 13 |
| 11 | Two-stage regime-then-price | Algorithm | P2 | L (5–7 d) | in hand | 6 |
| 2 | Self-explaining edges + notes | Understanding | P2 | M (ongoing) | in hand | — |
| 14 | Timing / sequencing | Meta | — | — | — | — |

Effort key: XS ≤ ½ d · S 1–2 d · M 2–4 d · L 4–7 d. "Data-needed = in hand" means it runs on the
existing parquet and does **not** depend on the Q-branch data streak.

---

## Part III — Cross-cutting methodology (applies to every task)

Every item below is delivered under the same non-negotiable methodology (from `METHODOLOGY.md` +
`contracts/process.md`):

1. **Planted-test-first (red → green).** Before any implementation, write a failing synthetic test
   whose ground-truth answer is *known by construction* (e.g. inject a feature that carries exactly
   `b` bits about the label only in a specified regime). Three estimator bugs in Stage 1 were caught
   only this way. Use the `planted-test-author` agent. Test lands **before** the module.
2. **Contract conformance.** New processes implement `EvaluationProcess.evaluate()` or
   `TransformProcess.transform()`, register via `@register`, key == `config/processes.toml` section,
   emit `ProcessResult` with `Finding`s. Verified by extending `test_process_base.py` /
   `test_bar_level_dispatch.py`.
3. **Costs only via `load_costs()`** (`config/costs.toml`). Every information gate is cost-adjusted
   (`I_min`); never hardcode a fee.
4. **Null-calibration is universal (task 12).** No finding is reported as "informative" on raw bits
   alone — it must clear the shuffle-null z-score threshold. This is a *library gate*, injected once
   and reused by every process.
5. **FDR/DSR at the surface (task 13).** Any process that sweeps a grid reports Benjamini–Hochberg
   q-values; the argmax cell is reported *with* its multiple-testing correction, never raw.
6. **Real-parquet smoke before commit.** After planted-green, run the process on real data via
   `nat process run <name> --symbol BTC` and the `/smoke` skill; only then commit
   (conventional commit, feat branch, `merge --no-ff`, no PRs).
7. **Reproducibility.** Every `ProcessResult` is provenance-stamped (git SHA, params, run_id) and
   emitted as structured JSON via `research_output.py` so findings are auditable and diffable.

**Definition of Done (per task):** planted test green · real-parquet smoke clean · null-calibrated ·
FDR-corrected if it sweeps · `ProcessResult` JSON emitted · contract test extended · documented in
this file's status column.

---

## Part IV — Detailed task specifications

> Each task: **Gap → Proposal → Plugs into → Design sketch → Dev methodology → Test coverage →
> Acceptance criteria → Effort/Deps.**

### 1. Process→algorithm compiler *(theme: loop closure · P2 · L)*

**Gap.** `scripts/agent/generators/it_discovery.py` turns IT-engine findings into *hypotheses*
(queue entries), and the hypothesis runner scores them, but nothing turns a *surviving* hypothesis
into a registered `MicrostructureAlgorithm`. The roster of tradeable algorithms is therefore
hand-authored and frozen at ~5.

**Proposal.** A code-generating step `scripts/agent/algo_synth.py` that consumes a promoted finding
`(feature-combo, horizon, regime-gate, polarity, sizing)` and emits a conforming algorithm stub under
`scripts/algorithms/generated/`, registered via `@register`, wired to the same
`AlgorithmFeature`/`step()`/`run_batch()` contract, with the finding's provenance in the docstring.

**Plugs into.** `scripts/algorithms/base.py` (ABC), `registry.py` (`@register`), `it_discovery.py`
(promotion signal), `scripts/algorithms/generated/` (already exists in the tree).

**Design sketch.**
```
promoted_finding → template(kind = {threshold, regime_gated, combiner}) →
  render step()/run_batch() from finding.feature_combo + finding.regime_gate →
  write scripts/algorithms/generated/<name>.py (+ docstring w/ MI, z-null, horizon, git_sha) →
  auto-run nat algorithm evaluate --algorithm <name> --symbol BTC (OOS) →
  emit ProcessResult(kind="algo_synth")
```

**Dev methodology.** Planted test: feed a synthetic finding with a *known* linear rule; assert the
generated algorithm reproduces the rule's IC on planted data. Then real finding → generated algo →
`nat oos30` matches the finding's forecast IC within tolerance.

**Test coverage.** `test_algo_synth.py` (planted rule round-trip) · conformance via
`test_bar_level_dispatch.py` (generated algo passes dispatch) · regression: generated algo's OOS IC
≥ finding's discovery IC − slack.

**Acceptance.** A promoted finding produces a green, contract-conformant, OOS-evaluated algorithm
with zero hand-editing; provenance chain finding→algo is queryable.

**Deps:** 6, 7, 12, 13 (only *validated, null-calibrated, FDR-passed* findings may be compiled).

---

### 2. Self-explaining edges + guided reading *(theme: understanding · P2 · ongoing)*

**Gap.** The single honest ceiling is operator understanding. Findings today emit numbers and LaTeX
math (`research_output.py`) but not *mechanism* — *why* a feature carries information.

**Proposal.** (a) Extend `research_output.py` so every surviving edge emits a structured
**mechanism annotation**: hypothesized microstructure cause, the paper it maps to
(`INSTITUTIONAL_ALGORITHMS.md`), and the failure mode that would kill it. (b) Produce
`docs/research/reading_notes/` — one note per shortlist paper (microprice, Cartea–Jaimungal,
queue-reactive, rough vol, DeepLOB): assumptions → estimator → what breaks in crypto perps → which
NAT feature/process it maps to.

**Plugs into.** `scripts/agent/research_output.py`, `docs/research/`.

**Dev methodology.** Not code-first; template-first. Define the annotation schema, backfill it for
the 5 current winners, then require it on all future promotions (schema-validated).

**Test coverage.** `test_research_output.py` extended to assert the mechanism annotation schema is
present and non-empty on promotion records.

**Acceptance.** Every promoted edge carries a human-readable mechanism + paper link; the 5 reading
notes exist and each cross-links a codebase GAP.

**Deps:** none (parallelizable, high learning ROI).

---

### 3. MI-maximizing nonlinear combiner *(theme: IT core · P2 · L)*

**Gap.** `it_engine/feature_selector.greedy_select` maximizes *marginal* CMI gain — it is myopic and
**misses synergy**. `interaction_info` (synergy/redundancy) is *computed* in `mi_ksg` but never
*selected on*. `pca_combo` is only the linear composition. "Indicators as a combination of features"
done right is a nonlinear combiner whose objective is information about the label.

**Proposal.** A `TransformProcess` `mi_combiner` that fits a small model (GBDT or shallow MLP) mapping
a candidate feature set → a scalar `combo` column, with objective = maximize
`CMI(combo; label | Z)` under the cost gate, regularized against redundancy via interaction
information. Emits the `combo` column as a first-class derived feature (chainable into `mi_ksg`,
`ic_horizon`, algorithms).

**Plugs into.** `scripts/it_engine/estimators.py` (`cmi`, `interaction_info`, `min_info_bits`),
`scripts/processes/pca_combo.py` (sibling), `scripts/processes/base.py` (`TransformProcess`,
`ProcessResult.derived`).

**Design sketch.**
```
candidate set S (top-k by task-6 conditional MI) →
  fit g_θ: features_S → combo, loss = −CMI(combo; label | Z) + λ·redundancy(S) →
  cross-fit (purged K-fold, no leakage) → emit combo column + selected S + achieved bits
```

**Dev methodology.** Planted test: build 3 features where the label is a *synergistic* XOR-like
function of two (each individually ~0 bits, jointly high). Assert `mi_combiner` recovers the pair and
`greedy_select` does *not* (documents the improvement). Purged K-fold to prevent overlap leakage.

**Test coverage.** `test_mi_combiner.py` (synergy recovery vs greedy baseline) · leakage test
(shuffled label ⇒ combo carries ≤ null bits) · real-parquet smoke.

**Acceptance.** On planted synergy, `mi_combiner` achieves ≥ X bits where greedy achieves ≈ 0; on
real data the combo's null-calibrated bits exceed the best single feature's.

**Deps:** 12 (null), 13 (FDR on candidate search).

---

### 4. Longitudinal MI tracker *(theme: IT core · P1 · M)*

**Gap.** The IT daemon (`scripts/it_engine/daemon.py`) recomputes on a **600 s rolling buffer**
(`buffer_size = 6000` @ 100 ms). It answers "is there MI *right now*" — never "is `MI(f; r | Z)`
*stable across days and regimes*, or a within-day mirage?"

**Proposal.** A batch evaluation process `mi_stability` that computes per-edge MI on each clean day
(or walk-forward fold) over the OOS window and reports the time series `MI_d`, its mean, dispersion,
and a stationarity/trend statistic. An edge that is only informative on scattered days is flagged
non-durable.

**Plugs into.** `scripts/processes/info_theory.py` (reuse estimators), `scripts/processes/runner.py`
(walk-forward folds, same machinery as `ic_horizon`), `config/it_engine.toml` (horizons, gate).

**Design sketch.**
```
for fold d in walk_forward(window_days):
    MI_d = mi_ksg(f; r | Z) on fold d   (null-calibrated, task 12)
report {mean, cv, slope, frac_days_informative, per-day series}
```

**Dev methodology.** Planted test: synthesize an edge present in folds 1–5 and absent in 6–10; assert
`frac_days_informative ≈ 0.5` and the trend statistic is negative. Requires ≥10 clean days of real
data for the real smoke (data-gated — but the planted path runs now).

**Test coverage.** `test_mi_stability.py` (planted intermittent edge) · integration with
`test_process_real_data.py`.

**Acceptance.** Produces a per-edge stability report; correctly flags a planted intermittent edge as
non-durable; runs over the `nat oos --window` fold set (PLAN task Q2).

**Deps:** 12; real run gated on ≥10 clean days.

---

### 5. Schedule the 3-bar classifier as a standing evaluation *(theme: loop · P0 · XS)*

**Gap.** `TripleBarrierProcess` (`triple_barrier`, `scripts/processes/labeling.py`) is fully
implemented — outputs `tb_label / tb_ret / tb_hit_bars` — and its docstring *explicitly* says to
chain it via `target_col="tb_label"` into `ic_horizon`/`ml_importance`/`mi_ksg`. But it is almost
certainly **never run as a standing evaluation**: it's a target you *can* score against, not one
anything schedules. Free alpha-surface sitting on the shelf.

**Proposal.** (a) Audit: grep the research output / findings store for `tb_label` to confirm it has
never been scored. (b) Add a standing config entry so `mi_ksg` and `ic_horizon` run with
`target_col="tb_label"` across the configured horizons, on every discovery cycle.

**Plugs into.** `config/processes.toml` / `config/it_engine.toml` (`--score-with`), the process
`runner.py`, `scripts/cli/process.py` (`nat process run`).

**Dev methodology.** No new estimator — this is composition + scheduling. Verify the chained run
produces `Finding`s keyed on `tb_label` and that path-dependent labels differ from fixed-horizon
returns (they should, because of stop-outs).

**Test coverage.** `test_process_transforms.py` extended: assert `triple_barrier → mi_ksg(target=
tb_label)` yields findings and that `tb_label` findings ≠ raw-return findings on the same features.

**Acceptance.** `tb_label` appears as a scored target in the standing discovery output; the audit
result (was it ever run before?) is recorded here.

**Deps:** none. **Do first — cheapest item on the list.**

---

### 6. `conditional_predictability` process ⭐ *(theme: conditional predictability · P0 · M)*

**Gap — the one genuine conceptual hole.** Today `mi_ksg` computes `CMI(f; r | Z)` — MI **averaged
over** the conditioning variable Z. What's missing is `MI(f; label | Z = z)` **as a function of z**:
the *predictability profile across the conditioning variable*. The z-bucket where MI spikes is the
*tradeable regime*. This is "predictability-of-predictability," and it upgrades the `regime_gated`
heuristic (low-entropy→trend, high-entropy→revert) from a guess to a **measured** gate.

**Proposal.** An `EvaluationProcess` `conditional_predictability` that, given a conditioning column Z
(vol, entropy, estimated entropy — the existing `entropy_conditioning` set is the default), buckets
observations by z (quantile bins), computes null-calibrated MI (task 12) per bucket, and reports the
per-bucket MI profile plus the argmax bucket and its bits-above-null.

**Plugs into.** `scripts/processes/info_theory.py` (new sibling process), `it_engine/estimators.py`
(`ksg_mi`, `min_info_bits`), `config/it_engine.toml` (`entropy_conditioning`,
`cmi_max_z_dims`), `regime_gated.py` (consumer of the measured gate).

**Design sketch.**
```
for z_bin in quantile_bins(Z, n_bins):
    subset = df[Z ∈ z_bin]
    if len(subset) ≥ cmi_min_samples:
        MI_bin = ksg_mi(f, label; subset)  −  null(f, label; subset)   # bits-above-null
report profile {z_bin → MI_bin}, argmax_bin, contrast = MI_max − MI_median
informative iff MI_max ≥ I_min AND contrast ≥ κ·σ_null
```

**Dev methodology.** Planted test (author first): inject a feature that carries `b` bits about the
label **only** when `Z < P30` and ≈ 0 bits elsewhere; assert the process localizes the edge to the
correct bucket and reports ≈ 0 in the others. This is the acceptance-defining test.

**Test coverage.** `test_conditional_predictability.py` (planted regime-localized edge) · min-samples
guard test (sparse bucket ⇒ NaN, never a spurious spike) · real-parquet smoke on BTC with
`Z = ent_tick_5s`.

**Acceptance.** Recovers a planted regime-localized edge to the correct z-bucket; on real data emits
a per-regime MI profile with FDR-corrected bucket significance; the argmax bucket is consumable by
`regime_gated` as a data-driven gate.

**Deps:** 12 (null-calibration is inside the loop). **Top pick — start here with #12.**

---

### 7. Horizon/label MI-surface meta-process *(theme: conditional predictability · P1 · M)*

**Gap.** Nobody has assembled the pieces into a scan over *what to predict × how far × in which
regime*. Each ingredient exists; the orchestrator does not.

**Proposal.** A meta-process `horizon_label_scan` that, for each horizon `h` and barrier geometry
`g = (pt_mult, sl_mult, max_holding)`, runs `triple_barrier(h, g)` → `conditional_predictability`
(task 6) and assembles an **MI surface** over `(h, g, regime)`. The argmax names the best
`(target, horizon, regime)` triple; the profile shows which horizons are predictable *at all* — the
measured answer to the MF-vs-macro-agent split.

**Plugs into.** `scripts/processes/runner.py` (orchestration), `labeling.py` (parametric
`triple_barrier`), task 6, `config/it_engine.toml` (`bar_horizons = [5min, 25min, 50min]`).

**Design sketch.**
```
grid = product(horizons, barrier_geometries)
surface = {}
for (h, g) in grid:
    tb = triple_barrier(h, g)
    surface[(h,g)] = conditional_predictability(f, tb.tb_label, Z)   # per-regime, null-calibrated
report argmax_{h,g,regime} with BH q-value (task 13); export surface for task 8
```

**Dev methodology.** Planted test: construct data where a feature predicts the label best at exactly
one `(h, g)` cell; assert the scan's argmax is that cell and neighbors are lower. Bound grid size and
`log()` anything truncated (no silent caps).

**Test coverage.** `test_horizon_label_scan.py` (planted single-cell optimum) · FDR-correction test
(all-null grid ⇒ no cell passes after BH) · real smoke (small grid on BTC).

**Acceptance.** Locates the planted optimum cell; on real data emits an FDR-corrected surface; the
argmax triple is reproducible across runs (deterministic given data + seed).

**Deps:** 5, 6, 12, 13.

---

### 8. Predictability surface artifact + viz *(theme: artifact · P1 · M)*

**Gap.** Findings are scattered across process runs; there is no single object the platform revolves
around, and nothing for the PLAN's `nat viz` D1 task to render.

**Proposal.** Define the **predictability surface** as a first-class persisted artifact with axes
`feature-combo × horizon × label-definition × regime`, value = cost-adjusted, null-calibrated,
FDR-corrected MI. It is the output aggregator for tasks 6/7, the input to the task-1 compiler, and the
data source for `nat viz predictability`.

**Plugs into.** `scripts/agent/research_output.py` (persistence), a new `nat viz predictability`
(PLAN D1), the `api` crate `/api/research/*` (already reads `data/research/`).

**Design sketch.**
```
schema: surface_cell{combo_id, horizon, label_def, regime_bin, mi_bits, z_null, bh_q, git_sha, run_id}
aggregate all task-6/7 findings → surface.parquet → viz renders a heatmap (regime × horizon),
  color = bits-above-null, opacity/× = FDR pass, drill-down = combo mechanism (task 2)
```

**Dev methodology.** Schema-first (planted rows → render round-trip). Follow the `dataviz` skill for
the heatmap; theme-aware, accessible.

**Test coverage.** `test_predictability_surface.py` (schema + aggregation determinism) · viz endpoint
test (`nat test dashboard` pattern).

**Acceptance.** A single queryable surface artifact exists; `nat viz predictability` renders it;
maturity tags (`[PROVEN]/[PRELIM]/[SPEC]`) from PLAN D1 attach per cell.

**Deps:** 6, 7.

---

### 9. Transfer-entropy causal graph *(theme: artifact · P2 · M)*

**Gap.** `ksg_te` / `linear_te` exist (directed information flow with a reverse-direction control),
but there is no *graph* — no view of which features/symbols *lead* which, and redundancy pruning is
still marginal-CMI based. Cross-symbol lead-lag is the `PARTIAL` gap in `INSTITUTIONAL_ALGORITHMS.md`
(Hasbrouck info-share); TE is its nonparametric cousin.

**Proposal.** An `EvaluationProcess` `te_graph` that builds a directed graph over features (and over
symbols BTC/ETH/SOL): edge `i→j` weighted by null-calibrated `TE(i→j)` where `TE_fwd > TE_rev`.
Report source nodes (drivers), prune features dominated by an upstream source, and surface
cross-symbol lead-lag as candidate signals.

**Plugs into.** `it_engine/estimators.py` (`ksg_te`, `linear_te`, `min_info_bits`),
`scripts/processes/info_theory.py` (`TransferEntropyProcess` sibling), cross-symbol features.

**Design sketch.**
```
for ordered pair (i, j): te = ksg_te(i→j) − null; keep if te ≥ I_min AND te > ksg_te(j→i)
build DiGraph; sources = nodes with in-degree 0 and high out-strength
prune: drop j if ∃ i: te(i→j) explains ≥ ρ of j's MI to the label
```

**Dev methodology.** Planted test: a driver series `x_t` and a lagged copy `y_t = x_{t−1}+noise`;
assert edge `x→y` present, `y→x` absent. Cross-symbol planted lead-lag as a second case.

**Test coverage.** `test_te_graph.py` (planted directed edge + reverse-control) · symmetry/noise
null (independent series ⇒ no edges after calibration).

**Acceptance.** Recovers planted direction; on real data emits a stable driver set and at least one
FDR-passing cross-symbol lead-lag candidate.

**Deps:** 12.

---

### 10. Predictability half-life *(theme: IT core · P1 · S)*

**Gap.** No decay model per edge. An edge with a 3-day MI half-life is untradeable regardless of peak
IC — and nothing measures that today. This is the quantitative feed into Q4/alpha-skeptic and the
PLAN's revalidation.

**Proposal.** Extend `mi_stability` (task 4): fit `MI(t)` decay (exponential / changepoint) per edge
and report a **half-life** in days. Edges with half-life below a configured floor are auto-demoted
before they ever reach paper.

**Plugs into.** task 4 (`mi_stability`), `scripts/agent/` promotion checks, alpha-skeptic gate (Q4).

**Design sketch.**
```
series MI_d (from task 4) → fit MI(t) = MI_0 · 2^(−t / τ) (robust) → half-life = τ
demote if τ < τ_min (config); attach τ to every surface cell (task 8)
```

**Dev methodology.** Planted test: synthesize `MI_d` with a known decay constant; assert recovered τ
within tolerance. Real path gated on ≥30 clean days (walk-forward).

**Test coverage.** `test_mi_halflife.py` (planted decay recovery) · robustness to a single outlier
day.

**Acceptance.** Recovers planted τ; produces a per-edge half-life column feeding promotion/skeptic.

**Deps:** 4; real run gated on ≥30 clean days.

---

### 11. Two-stage regime-then-price system *(theme: algorithm · P2 · L)*

**Gap.** Point 4 taken literally: signals fire in all regimes. But predicting the *condition* (future
vol/entropy) is itself a process, and a good condition forecast tells you *when to turn signals on* —
often worth more than a marginally better signal.

**Proposal.** A composite `MicrostructureAlgorithm` `regime_then_price`: **stage 1** forecasts the
favorable regime (predict future vol/entropy bucket, using the "estimated entropy" the operator
raised); **stage 2** emits the price signal *only* when stage 1 says the regime is in the task-6
argmax bucket. Sizing scales with stage-1 confidence.

**Plugs into.** `scripts/algorithms/base.py`, `regime_gated.py` / `regime_state_machine.py` (existing
regime machinery), task 6 (which bucket is favorable), entropy features.

**Design sketch.**
```
stage1: ŝ_{t+h} = forecast(vol_t, entropy_t, …)         # predict the condition
gate = 1[ ŝ_{t+h} ∈ favorable_bucket(task 6) ]
stage2: signal_t = base_price_signal_t · gate · confidence(ŝ)
```

**Dev methodology.** Planted test: data where the price signal is informative only in regime R and
stage-1 can predict R; assert gated IC > ungated IC and that gating off R removes the loss. Purged
K-fold; cost-gated.

**Test coverage.** `test_regime_then_price.py` (planted gating uplift) · conformance via
`test_bar_level_dispatch.py` · OOS via `nat oos30`.

**Acceptance.** Gated OOS Sharpe/IC exceeds the ungated base signal on real data, net of costs; ablating
the gate degrades performance (proves stage 1 adds value).

**Deps:** 6.

---

### 12. Null-calibration layer ⭐ *(theme: IT core discipline · P0 · S)*

**Gap — the trust-defining one.** `info_theory.py` already documents that KSG has a spurious
~0.07-bit noise floor (hence the rank/copula transform). But findings are still reported in raw bits,
leaving "is 0.05 bits real?" unanswered. This is the IT analog of the planted-test discipline and the
precondition for trusting *every other task here*.

**Proposal.** A shared utility `it_engine/null_calibration.py`: for any estimator call, build the
**null distribution** by shuffling the label / circular-shifting to break dependence while preserving
marginals (`n_shuffles` configurable), and report each finding as **bits-above-null**, a **z-score**,
and an empirical **p-value** — not raw bits. Injected once into `mi_ksg`, `transfer_entropy`, tasks
3/4/6/7/9/10.

**Plugs into.** `scripts/it_engine/estimators.py` (wrap `ksg_mi`/`cmi`/`ksg_te`),
`scripts/processes/info_theory.py`, `config/it_engine.toml` (`n_shuffles`, `null_z_threshold`).

**Design sketch.**
```
null = [ estimator(f, shuffle(label)) for _ in range(n_shuffles) ]
z = (raw − mean(null)) / std(null);  p = mean(null ≥ raw)
report {raw_bits, bits_above_null = raw − mean(null), z, p}
informative iff bits_above_null ≥ I_min AND z ≥ null_z_threshold
```

**Dev methodology.** Planted test (author first): (a) pure noise ⇒ z ≈ 0, p ≈ 0.5, bits_above_null ≈
0 (kills the 0.07-bit floor as a false positive); (b) a known-`b`-bit planted edge ⇒ large z, small
p. This is the estimator-honesty gate that protects the whole document.

**Test coverage.** `test_null_calibration.py` (noise ⇒ null; planted ⇒ significant) · determinism
(seeded shuffles reproducible — note the workflow constraint: seed passed in, not `Math.random`).

**Acceptance.** Pure-noise inputs never report as informative; planted edges clear the z-threshold;
all downstream processes consume `bits_above_null`, not raw bits.

**Deps:** none. **Do first with #6 — this is the foundation.**

---

### 13. FDR/DSR on the process layer *(theme: discipline · P1 · S)*

**Gap.** A surface with thousands of `(combo × horizon × label × regime)` cells *guarantees* a
great-looking argmax by chance. FDR/DSR is enforced on features (`alpha/screener.py`) but not yet on
the process-layer sweeps (tasks 6/7/9).

**Proposal.** A shared `process_fdr` step: collect all cell p-values from a sweep (via task 12),
apply Benjamini–Hochberg at `q = 0.05`, and report every cell **with** its q-value. The argmax is only
ever surfaced as "argmax, BH-q = …". Reuse the existing FDR machinery; extend it to `ProcessResult`
findings.

**Plugs into.** `scripts/alpha/screener.py` (existing BH), tasks 6/7/9 sweeps, `research_output.py`.

**Design sketch.**
```
pvals = [cell.p for cell in sweep]              # p from task 12
q = benjamini_hochberg(pvals, alpha=0.05)
annotate each cell with q; report n_discoveries, argmax + its q
```

**Dev methodology.** Planted test: an all-null grid ⇒ expected discoveries ≤ q·m; a grid with k
planted true cells ⇒ those k recovered with controlled false-discovery proportion.

**Test coverage.** `test_process_fdr.py` (all-null false-discovery rate ≤ q; planted power) ·
integration with tasks 6/7.

**Acceptance.** All-null sweeps yield ≈ 0 discoveries; every reported cell carries a BH q-value; no
argmax is ever reported without its correction.

**Deps:** 12.

---

### 14. Timing / sequencing *(theme: meta)*

**Gap.** Risk of deferring this behind the Q-branch data streak.

**Proposal / rationale.** Every task 1–13 runs on **data already in hand** (planted paths need no
data at all; real smokes use existing parquet). `PLAN.md` explicitly lists the process layer under
*actionable-now, no data needed*. While the Q-branch is frozen on data continuity, this layer is the
**highest-value work available** — it raises the ceiling on *all future* discovery instead of
re-testing the same five algorithms. The two data-gated items (4 real-run ≥10 days, 10 real-run ≥30
days) have their planted paths delivered now and their real runs deferred to the streak.

**Acceptance.** This document is adopted as the D-branch process-layer backlog; #5, #6, #12 are
scheduled immediately.

---

## Part V — Sequencing & phased roadmap

```
Phase 0 (foundation, ~1 wk, data in hand):
   #12 null-calibration  →  #6 conditional_predictability  →  #5 schedule 3-bar
   (kill false positives, get the one missing concept, harvest the shelved target)

Phase 1 (composition, ~1–2 wk):
   #13 FDR  →  #7 horizon/label surface  →  #8 surface artifact + viz
   #4 longitudinal tracker (planted now; real run when ≥10 clean days land)

Phase 2 (leverage, ~2–3 wk):
   #3 MI combiner   #9 TE graph   #10 half-life (real @ ≥30 days)
   #1 process→algorithm compiler (needs 6/7/12/13 mature)
   #11 two-stage regime-then-price
   #2 mechanism notes + reading notes (parallel throughout)
```

**Critical path:** `#12 → #6 → #7 → #8 → #1`. Everything else hangs off `#12`.

## Part VI — Risks & open questions

- **Overfitting the surface.** The central risk: more MI mining = more multiple testing. Mitigated by
  #12 (null) + #13 (FDR) — these are non-negotiable, not optional polish.
- **Sample sufficiency for conditional MI.** Bucketing by regime (#6) shrinks per-bucket N; KSG needs
  `cmi_min_samples` (≥500). Sparse regimes must return NaN, never a spurious spike — enforced by test.
- **Leakage in combiners (#3, #11).** Purged K-fold + embargo mandatory; overlapping triple-barrier
  labels are the classic leak.
- **Is `0.15` (PLAN D1 gate) reachable via this layer?** Open. The surface tells you the *best
  achievable* null-calibrated conditional MI on current features; if the argmax cell still implies
  fill-conditional IC < 0.15, the honest answer is "widen features (institutional GAPs) or the gate is
  not met" — which is exactly the go/no-go this layer exists to answer.
- **Estimator cost.** KSG kd-tree ball queries are the slow path; shuffle-null (#12) multiplies cost
  by `n_shuffles`. Budget subsampling (`max_samples`) and cache (SHA-256, existing) are required.

---

*End of proposal. Next action on adoption: schedule #5, #6, #12; open a `feat/process-layer` branch
per task; planted-test-author first on each.*

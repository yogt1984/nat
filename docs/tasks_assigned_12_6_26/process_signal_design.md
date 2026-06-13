# Process Set Design — Extracting Uncorrelated Signals, Information-Theoretically

**Date:** 2026-06-13 · **Status:** DESIGN (Stage 1+2 framework is the substrate, see `process_concept.md`)
**Thesis being operationalized:** edge = (information exists) AND (you actually found it, not noise)
AND (it survives regimes/decay) AND (uncorrelated pieces are combined into breadth). One process
layer per clause; the capstone transform hands a decorrelated signal book to the algorithm layer.

The fundamental law is the budget: IR ≈ IC_bar · sqrt(N_eff), with N_eff = N / (1 + (N-1)·rho_bar).
Every process below either raises IC_bar (find real information), raises N_eff (kill redundancy,
verify orthogonality where it matters), or protects both from estimation error and decay.

---

## Layer 1 — Existence with redundancy control (raises N_eff at the source)

### S1. `cmi_select` (evaluation) — redundancy-aware information ranking  [~6h]
The core "uncorrelated extraction" machine. Greedy forward selection in information space:

    f_1     = argmax_f  I(f ; r)
    f_{k+1} = argmax_f  I(f ; r | S_k)          (conditional on the already-selected set)
    stop when marginal gain < max(I_min(fee, sigma_r, kurt), epsilon)

A feature correlated with an already-selected one contributes ~zero CMI and is never picked —
selection is uncorrelated-by-construction in the information metric, which is stronger than
linear decorrelation (catches nonlinear redundancy PCA misses).
- Wraps `it_engine.feature_selector.greedy_select` + `estimators` (rank-transform inputs — the
  Stage-1 KSG scale-bias fix applies verbatim).
- Findings per selected feature: metric `cmi_gain_bits`, extras: rank, cumulative MI curve,
  `redundancy_bits = I(f;r) − I(f;r|S)` for the rejected runner-ups.
- Params: `horizon`, `max_selected` (default 10), `ksg_k`, `max_samples`, `epsilon_bits`.
**Planted test:** frame with signal A, an exact noisy copy A', and an independent signal B →
must select {one of A/A', B}, never both copies; cumulative-MI curve must flatten after 2.

### S2. `interaction_synergy` (evaluation) — pairwise synergy mining  [~5h]
Finds information that only exists in combinations — invisible to any single-feature screen:

    S(X,Y) = I([X,Y] ; r) − I(X ; r) − I(Y ; r)     (> 0: synergy, < 0: redundancy)

Run over the top-K features from S1/ic_horizon only (K≈20 → 190 pairs; KSG cost is the
constraint, `max_samples` ≈ 4000). Synergistic pairs are candidate engineered composites
(products, conditionals) to feed back as derived features.
- Findings per pair: metric `synergy_bits`, threshold I_min; extras: marginal MIs, joint MI.
**Planted test:** XOR construction — r driven by sign(x)·sign(y): marginal MI ≈ 0 for each,
joint MI > gate → must flag the pair and ONLY the pair.

### S3. `lead_lag_te` (evaluation) — cross-symbol directed information  [~6h]
Cross-asset breadth is the cheapest decorrelation there is. TE matrix across symbols:

    TE(x_A → r_B, lag L)  vs  TE(r_B → x_A)        for A,B in {BTC, ETH, SOL}, L in {1..L_max}

`linear_te` default (the Stage-1 KSG-CMI bias finding applies). Edges above I_min with
directionality ratio > 1 form the lead-lag network; an edge A→B that survives S5/S6 is a
signal for B uncorrelated with B's own microstructure family.
- Needs a small runner extension: multi-symbol load + per-pair alignment on bar_start
  (currently the runner filters to one symbol; ~30 lines).
**Planted test:** synthetic pair where A's feature drives B's returns at lag 2 → edge A→B at
lag 2 flagged, reverse and lag-1 not.

---

## Layer 2 — Estimation hygiene (protects IC_bar from being fiction)

### S4. `permutation_null` (evaluation, meta-process) — the reality check  [~6h]
The backstop against manufactured discoveries. For a candidate set (or another process's run_id):
rebuild the null by **stationary block bootstrap of returns** (preserves autocorrelation and
overlap structure — naive shuffling destroys exactly the dependence that inflates metrics),
M=500 resamples, recompute the headline metric per feature:

    p_perm(f) = #{ metric_null >= metric_obs } / M
    excess discoveries = n_obs(passing) − E[n_null(passing)]     (White's Reality Check flavor)

A set whose excess is ~0 is indistinguishable from luck regardless of per-feature p-values.
- Input modes: `--features ...` fresh, or `--from-run <run_id>` to audit an existing record.
- Findings: per feature `p_perm`; summary: `excess_discoveries`, null distribution quantiles.
**Planted test:** pure-noise frame → passing fraction ≈ alpha (calibration); planted frame →
signal's p_perm < 0.01 and excess ≥ 1.

### S5. `stability_decay` (evaluation) — non-stationarity audit  [~5h]
Signals are perishable inventory; this is the freshness meter. Per feature:
- Rolling **calendar-windowed** IC series (not expanding — expanding hides death by averaging).
- Break detection: CUSUM / Page–Hinkley on the IC series; flag location of first break.
- Sub-period sign consistency (the cross-symbol KEEP rule applied across time): fraction of
  windows with sign(IC_w) == sign(IC_overall); decay half-life of |IC| vs window age.
- Informative = no detected break AND sign consistency ≥ threshold (default 0.7).
**Planted test:** signal constructed to die at the sample midpoint → break flagged within ±10%
of the true location, informative=False; persistent twin passes.

---

## Layer 3 — Conditional structure (orthogonality that survives)

### S6. `regime_conditional` (evaluation) — conditional IC and stress correlation  [~6h]
Two questions per candidate, conditioned on live regime variables (`ent_book_shape`,
`vol_zscore`, `regime_*`, `trend_hurst_*` — all in the current 154):
1. **Conditional IC:** IC within conditioning-variable quantile buckets; lift = IC_active/IC_uncond
   (the preprint's 0.45→0.67 ent_book_shape gating, generalized to any feature × any regime).
2. **Stress correlation of the SET:** pairwise signal correlations computed per bucket —
   specifically the high-vol bucket. A signal book whose calm-weather rho_bar = 0.1 but
   stress rho_bar = 0.7 has N_eff ≈ 1 exactly when it matters. Report rho_bar and N_eff per bucket.
- Findings per (feature, regime_var): metric `ic_lift`; set-level summary: `n_eff_by_bucket`.
**Planted test:** signal active only when planted vol regime is low → lift localized to the
correct bucket; two signals sharing a stress factor → high-vol N_eff collapses, flagged.

### S7. `residualize` (transform) — pure-innovation extraction  [~4h]
Orthogonalization as a first-class transform: for targets F and conditioning set Z (e.g. the
market mode pc_1, or `imbalance_qty_l1` to extract what a cousin feature adds beyond it):

    res_f(t) = f(t) − beta' Z(t),   beta fit on the training prefix only (no lookahead)

Output `res_<name>` columns; chain `--score-with ic_horizon` to ask "what does f know that Z
doesn't?" — the per-feature analogue of S1's set-level CMI, but it produces a *tradeable series*
rather than a ranking.
**Planted test:** f = unique_signal + market_mode; residualizing vs market_mode preserves the
unique IC and drives holdout corr(res_f, Z) ≈ 0; prefix-only fit verified by holdout perturbation
(same no-lookahead test pattern as pca_combo).

### S8. `pca_combo` upgrade: Marchenko–Pastur denoising (param, not a new process)  [~3h]
Eigenvalues of a T×N correlation matrix from pure noise fall below
lambda_plus = (1 + sqrt(N/T))²; with ~100-300 bars and 50+ features, MOST raw eigenvectors are
noise dressed as structure. Add `denoise="mp"` to pca_combo: clip the noise band, keep signal
eigenvectors, report `n_signal_eigenvalues` and holdout orthogonality raw-vs-denoised.
**Planted test:** 3 true factors + 40 noise features → n_signal ≈ 3; denoised holdout
orthogonality strictly better than raw.

---

## Layer 4 — Combination (the bridge to algorithms)

### S9. `signal_book` (transform, capstone) — the decorrelated book + its honest N_eff  [~8h]
Composition, not new math. Inputs: run_ids of upstream evidence (S1 selection, S5 stability,
S6 regime verdicts). Pipeline:
1. Take features passing: selected by S1 ∩ stable per S5 ∩ (regime-robust OR explicitly gated) per S6.
2. Residualize sequentially against each other (S7 logic, selection order) → mutually orthogonal.
3. Walk-forward z-score → `sig_*` columns, emitted as derived parquet.
4. Manifest in extras: per-signal provenance chain (every upstream run_id — the full audit trail
   from raw feature to book entry), pairwise corr (full sample AND high-vol bucket), and:

       N_eff = N / (1 + (N−1)·rho_bar),    IR_expected = IC_bar · sqrt(N_eff · bars_per_year)

   computed for BOTH calm and stress rho_bar — the gap between the two IS the risk disclosure.
- The output contract is what a combiner algorithm consumes: a `MicrostructureAlgorithm` whose
  `required_columns()` are `sig_*` and whose `step()` is risk-parity weighting (Q2.8/meta_portfolio
  logic) — algorithms combine; processes certify what is combinable.
**Planted test:** 3 independent planted signals → N_eff ≈ 3 and book IC_bar ≈ individual ICs;
add a duplicated signal → N_eff must NOT rise; one unstable signal → excluded by the S5 gate.

---

## Sequencing, effort, dependencies

| Order | Process | Layer | Effort | Depends on |
|-------|---------|-------|--------|-----------|
| 1 | S1 cmi_select | existence | 6h | it_engine.feature_selector (exists) |
| 2 | S5 stability_decay | hygiene | 5h | — |
| 3 | S4 permutation_null | hygiene | 6h | — (audits everything else; land early) |
| 4 | S7 residualize | structure | 4h | — |
| 5 | S6 regime_conditional | structure | 6h | live regime columns (exist) |
| 6 | S8 MP-denoise param | structure | 3h | pca_combo (exists) |
| 7 | S2 interaction_synergy | existence | 5h | S1 (top-K input) |
| 8 | S3 lead_lag_te | existence | 6h | runner multi-symbol (~30 lines) |
| 9 | S9 signal_book | combination | 8h | S1+S5+S6+S7 |

Total ~49h. S1+S5+S4 (17h) already yield the minimum honest loop: select uncorrelated → check
stable → check not luck. S9 is only worth building on ≥7 clean days of data (post-Jun-17, plan
T11b window) — N_eff estimates on 100 bars are themselves noise.

## Risks / known constraints
- KSG cost explodes on pairs (S2) and conditioning sets (S1): hard caps via `max_samples`,
  top-K pre-filter, and `linear_te` defaults where Gaussian is tolerable.
- S6 stress buckets are thin until more data accumulates — report bucket sizes, refuse below
  min_obs (the framework's existing conservatism conventions).
- S3 multi-symbol runner extension touches shared load path — keep single-symbol default intact.
- Every process ships with its planted-signal contract in `synthetic.py` extensions BEFORE
  real-data use (the Stage-1 lesson: three estimator bugs were caught only by planted tests).

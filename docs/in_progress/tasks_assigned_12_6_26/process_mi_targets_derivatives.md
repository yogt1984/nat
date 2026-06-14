# MI ↔ Price Action — Lineage, Process Object, Target Taxonomy, Derivative Operators

**Date:** 2026-06-14 · **Status:** DESIGN / reference (doc-only)
**Companions:** `process_concept.md` (the three citizens), `process_signal_design.md` (processes S1–S9)
**Scope note:** Part A citations are canonical works stated from knowledge — **verify exact
volume/year before any go into the preprint.** Parts B–D bind to the shipped framework and this
session's two code audits; "shipped" vs "proposed" is marked throughout. Nothing here is built yet.

---

## Part A — The MI ↔ price-action research lineage

Why mutual information is the right currency, as a thread rather than a flat list:

**1. Why MI at all (the theoretical floor).**
- Shannon (1948), *A Mathematical Theory of Communication* — entropy/MI, the foundation.
- Cover & Thomas, *Elements of Information Theory* — **Fano's inequality** and **rate–distortion**:
  MI lower-bounds achievable prediction error. This is *the* reason MI, not correlation, is the
  right metric — and the formal parent of the shipped `min_info_bits` cost gate (a signal needs
  ≥ I_min bits to pay the round-trip toll). Correlation answers "linear and how much"; MI answers
  "is there *any* exploitable dependence, at all."

**2. Entropy of price series (the object being measured).**
- Pincus (1991, PNAS), Approximate Entropy; Richman & Moorman (2000), Sample Entropy — regularity
  of financial series; the ancestors of the live `ent_*` features.
- Cont (2001), *Empirical properties of asset returns* — the stylized facts (fat tails, vol
  clustering, near-zero return autocorrelation) that explain *why* return MI is small and why the
  honest gates in `process_signal_design.md` exist.

**3. MI estimation (the instrument).**
- Kraskov–Stögbauer–Grassberger (2004, Phys Rev E) — the KSG k-NN estimator wrapped in
  `it_engine/estimators.py`. Practical footnote from our Stage-1 work: rank/copula-transform the
  inputs first — raw-scale KSG manufactures a ~0.07-bit noise floor when feature and return scales
  differ (verified on planted data).

**4. Directed information in markets (which way it flows).**
- Schreiber (2000, PRL), *Measuring Information Transfer* — transfer entropy.
- Marschinski & Kantz (2002, Eur Phys J B) and Dimpfl & Peter (2013) — effective TE *between
  financial series*; the applied parent of S3 `lead_lag_te`. Granger (1969, Econometrica) is the
  linear ancestor (`linear_te`).

**5. MI feature selection (how to pick uncorrelated signals).**
- Battiti (1994) MIFS → Peng–Long–Ding (2005) **mRMR** → Brown et al. (2012, JMLR) unifying
  framework — the relevance-minus-redundancy lineage behind S1 `cmi_select`.
- Williams & Beer (2010), Partial Information Decomposition — synergy/redundancy, behind S2.

**6. Microstructure information ↔ price (where the bits originate).**
- Kyle (1985, Econometrica) — λ, the price impact of informed trading; lives as `illiq_kyle_*`.
- Hasbrouck (1995, J Finance) — information share / price discovery.
- Easley–Kiefer–O'Hara (PIN) and Easley–López de Prado–O'Hara (2012, RFS, **VPIN**) — flow
  toxicity as a carrier of directional information; lives as `toxic_vpin_*`.

**7. Predictability limits & the honesty thread (why measured MI lies).**
- Lo & MacKinlay (1988) variance-ratio; Lo (2004) Adaptive Markets — predictability is real but
  regime-dependent and decays (motivates S5/S6).
- Bailey–Borwein–López de Prado–Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism* +
  the Deflated Sharpe Ratio — the thesis statement for Layer-2 estimation hygiene (S4).

**One-line takeaway:** the field has converged on MI/TE as the dependence currency, k-NN/copula
as the instrument, mRMR/CMI as the selector, and data-snooping correction as the mandatory
counterweight. The framework already instantiates all four — the gaps are the *target* (Part C)
and the *feature space* (Part D).

---

## Part B — The Process object, refined

Shipped today (`scripts/processes/base.py`): `Process` → {`EvaluationProcess`,
`TransformProcess`}; `ProcessContext` (symbol, timeframe, price_col, horizons, costs, target_col,
extra_sources); `Finding`; `ProcessResult`; `partition_usable_columns` (the NaN/K2 guard);
`describe()` (machine-readable params); registry/config/CLI/persistence uniformity. **Keep all of
this verbatim** — it works.

Three refinements the maturity discussion surfaced:

**B1. Promote the target to a first-class object (proposed).** Today the prediction target is a
buried `target_col: str | None`, defaulting to simple forward returns, and — the real gap —
**only `ic_horizon` and `ml_importance` honor it; `mi_ksg`, `transfer_entropy`, `spectral`, and
the IT engine silently use forward returns regardless.** Replace `target_col` with
`ProcessContext.target: Target` (Part C), so *every* process computes `I(feature ; Target)` against
one explicit, registry-resolved target. This is the single highest-leverage change: it makes "MI
between a feature and price action" honest about *which* price action.

**B2. Adopt the typed-node view as the honest model (proposed framing).** The
evaluation/transform split is already leaking — `signal_book` (S9) is a "transform" doing portfolio
construction; `triple_barrier` produces a *target* that is really neither. The mature model is a
typed computation graph of node kinds — **feature-node, transform-node, evaluation-node,
target-node** — not a 2-kind stack. Treat the leak as the architecture earning its next revision,
not a defect. Do *not* refactor preemptively; record the target-node as the first new kind (B1).

**B3. Program-level multiple-testing ledger (proposed hook).** BH-FDR is per-run; the number that
decides whether discoveries are real is the *total* hypotheses the platform ever tested. Add a
ledger the runner appends to (process, target, n_tested, timestamp, git_sha) — `describe()` already
exposes enough for an honest count. The per-run index stays a research log; this is the meta-gate
(White's reality check at program scope, S4 generalized).

Refined ABC sketch (proposed deltas only):
```
class Target(ABC):                 # NEW node kind (Part C)
    name: str; kind: str           # "continuous" | "binary" | "categorical"
    def build(self, prices, bars, horizon) -> np.ndarray: ...
class ProcessContext:
    target: Target                 # replaces target_col; default = ForwardReturn
    ...                            # everything else unchanged
```

---

## Part C — Classifying price action (the target side)

"Price action" is not one thing. The MI literature has studied features against many targets; the
codebase studies essentially one (forward returns). A 2-D taxonomy — **aspect × horizon** — with
the research that owns each cell and the current support status:

| Aspect ↓ / Horizon → | tick–s | min–bar | multi-day | Literature | Status |
|---|---|---|---|---|---|
| **Direction** sign(r) | ✓ | ✓ | ✓ | Lo–MacKinlay; classification ML | binary in `ml_importance` only |
| **Magnitude** \|r\|, r | ✓ | ✓ | ✓ | Fano/rate-distortion | **shipped** (forward returns) |
| **Volatility** RV/range | – | ✓ | ✓ | Cont; HAR (Corsi 2009) | not a target (only a scaler) |
| **Path** triple-barrier, MFE/MAE, time-to-touch | – | ✓ | ✓ | López de Prado (AFML Ch.3) | `tb_label` exists, partial |
| **Regime transition** P(switch) | – | ✓ | ✓ | Hamilton (1989); Ang–Bekaert | none |
| **Jump / tail** | ✓ | ✓ | – | Lee–Mykland (2008); Aït-Sahalia | `jump_detector` algo only |
| **Microstructure** mid → micro → **fill-conditional** | ✓ | ✓ | – | Kyle; Hasbrouck; Stoikov | mid only |

Two cells matter most because they are both unbuilt and high-value:
- **Volatility-as-target** — features predicting *future realized vol* (not direction) is a whole
  tradeable family (vol-squeeze, sizing) the platform can't currently screen for.
- **Fill-conditional move** — the bridge to the execution-realism gap. The 12 dead algorithms lost
  *between mid-price signal and fill*; a target defined on the price you actually transact at is
  where that loss becomes measurable. Highest-value missing target.

**Proposed (design, not build): `scripts/processes/targets.py`** — a `Target` registry mirroring
the process registry:
```
@register_target
class ForwardReturn(Target):   kind="continuous"   # default; today's behavior
class Direction(Target):       kind="binary"        # sign(r_h)
class RealizedVol(Target):     kind="continuous"    # future RV over h
class TripleBarrier(Target):   kind="categorical"   # wraps labeling.py
class FillConditional(Target): kind="continuous"    # needs execution model (gap)
```
Wiring is uniform: `ProcessContext.target.build(...)` replaces the per-process forward-return call.
Closing B1's gap is ~1 line per process. The taxonomy *is* the config surface: a run is
(process × target × horizon × symbol).

---

## Part D — Derivative-feature operators (`feature_ops` transform-process)

**Design constraint from the audit:** much is already built — do NOT re-propose `trend_momentum_*`
(slopes), `trend_hurst_300/600` (R/S), `vol_parkinson/garman_klass` (realized vol), the 15
`derived_*` interactions (incl. ROC-style `trend_strength_roc`, `entropy_momentum`,
`toxic_vpin_roc`), EMAs, `vol_zscore`, or the bar-aggregation `_slope`. Those exist.

**Proposed:** one Python `TransformProcess` named `feature_ops` (streak-safe — zero ingestor
contact; chainable: `feature_ops` → `cmi_select`/`ic_horizon` via `--score-with`). It applies an
operator spec to a chosen base-column set and emits `op_<operator>_<source>` columns. Winners get a
documented promotion path to Rust ingestor features post-streak. Only the *missing* operators below:

**D1. Temporal calculus.**
- **Fractional differentiation** `(1−B)^d`, 0<d<1 (López de Prado, AFML Ch.5) — *highest value,
  not built*: makes a level series stationary while preserving memory the integer diff destroys.
  Consumers: every evaluation process (stationarity ↑ → IC/MI estimates honest). Planted test:
  fractional-Brownian input → ADF stationarity at min d, memory retained vs full diff.
- **Leaky / cumulative integration** `y_t = ρ y_{t-1} + x_t` — accumulates flow/imbalance into a
  pressure state (raw imbalance is memoryless). Consumer: ic_horizon at longer horizons.
- **Higher-order ROC / jerk** — 2nd/3rd differences of slow features (only 1st-order exists).

**D2. Spectral-as-feature** (audit: spectral lives only in `spannung_spectral.py` *analysis*, never
as persisted features). Rolling, causal, windowed:
- bandpower per frequency band, dominant frequency, **rolling spectral entropy**, wavelet energy.
- Consumer: `spectral` process (tick-level) and ic_horizon (bar-level rolled). Planted test:
  injected band-limited oscillation → bandpower spikes in the right band only.

**D3. Normalization / shape.** rolling rank/quantile (distribution-free, fat-tail-robust), robust
z (median/MAD vs the existing mean/std z), bounded logit/sigmoid. Consumer: `mi_ksg` (rank already
helps KSG), `ml_importance`.

**D4. Cross-feature.** synergy products/ratios `x·y`, `x/y` for candidate pairs (feeds S2
`interaction_synergy`); rolling cross-correlation between categories (e.g. flow↔vol). Note:
orthogonalization is already S7 `residualize` — don't duplicate.

**D5. Path / persistence, generalized.** rolling Hurst/**DFA** and autocorrelation-at-lag applied
to *any* column (today Hurst is hard-wired to price/trend only). Consumer: S5 `stability_decay`,
S6 `regime_conditional`. Refs: Peng et al. (1994) DFA; Kantelhardt et al. (2002) MFDFA.

**D6. Nonlinear basis.** polynomial/spline expansion so *linear* processes (linear_te, IC) catch
curvature; sign×magnitude split (a feature may inform direction and size differently). Consumer:
ic_horizon, transfer_entropy.

**Standing caveat:** operators multiply the column space (one base feature → many `op_*`), so the
program-level FDR ledger (B3) matters *more*, not less. `feature_ops` runs must log how many columns
they minted so the meta-gate counts them.

---

## How it assembles (and what's still missing)

```
feature_ops (D: expand the space)
   → cmi_select (S1: pick uncorrelated)
   → Target family (C: choose which price action)
   → evaluation processes (ic_horizon / mi_ksg / te / spectral / ml_importance: score)
   → signal_book (S9: orthogonal book + honest calm/stress N_eff)
   → algorithm (risk-parity combiner)
```

Two maturity gaps this doc names but does not close (consistent with the architecture assessment):
1. **Execution realism** — no fill/cost-model node; the fill-conditional target (C) is the hook,
   but the model behind it is the real missing fourth citizen, and where the 12 algos actually died.
2. **Program-level multiple testing** — B3's ledger is the design; until it exists, every IC/MI
   number is un-budgeted against the platform's total search.

Sequencing if/when built: `targets.py` (B1/C, ~1 day, unblocks honest MI on all processes) →
`feature_ops` D1+D3 first (fractional-diff + robust-norm, the cheapest high-value pair) → D2/D5 →
D4/D6. All Python, all streak-safe, all behind the planted-signal gate before real data.

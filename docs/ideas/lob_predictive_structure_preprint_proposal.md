# Preprint Proposal — Predictive Structure of the Perp Order Book

**Status:** PROPOSAL v2 (2026-07-26; supersedes execution_overlay_preprint_proposal.md — overlay
demoted to an implications section) · **Owner:** Onat · **Parent record:** `../research/FINDINGS.md`
**Relation to existing preprints:** standalone; sharpens the empirical core of
`microstructure_alpha_preprint` into a measurement-first paper. Cross-cite both ways.

---

## One-line thesis

The limit order book of cryptocurrency perpetual futures carries **two empirically orthogonal
predictive channels** — a directional family (order-book imbalance, rank IC ≈ 0.45 at 1–5 s) and a
volatility family (arrival-intensity/activity, IC ≈ 0.35 at 5 s) — that are universal across
symbols, spectrally localized, decaying toward efficiency, and cleanly bounded in what they can
and cannot deliver at the point of execution.

## Positioning (the novelty ledger — write the related-work section from this)

Known (must cite, not re-claim): imbalance/OFI predicts short-horizon returns
(Cont–Kukanov–Stoikov 2014; deep-LOB literature). **New here:**
1. **Orthogonality factorization** — direction features carry zero vol-IC and volatility features
   zero directional IC, consistently on BTC/ETH/SOL: sign and magnitude of short-horizon returns
   are independently predictable from disjoint feature sets.
2. **Universality at scale** — 207-feature census, 6 horizons, 3 symbols, 100 ms cadence,
   bootstrap CIs, compressing to eight independent directional axes.
3. **Spectral localization** — the entire directional IC lives in the 0.005–0.1 Hz band; OU
   half-life 5–7 s; dominant cross-coherence at 68 s; time-domain aggregation destroys the signal.
4. **Efficiency drift measured in real time** — signal half-life shortening May→June (5 s IC
   0.45→0.29 while 1 s holds/improves) as spreads tighten in a maturing venue.
5. **The execution boundary** (mechanism section, not headline): directional fill-conditioning
   collapses IC 0.45 → ≈ 0.03 while any-fill conditioning *raises* it to 0.52 — adverse selection,
   not decay, bounds monetization; regime gating (`ent_book_shape`, +22 %) partially reopens it.

## Working titles

1. *Two Orthogonal Channels: Directional and Volatility Information in Cryptocurrency Perpetual
   Futures Order Books*
2. *Predictive Structure of the Perpetual-Futures Limit Order Book: Universality, Orthogonality,
   and the Execution Boundary*
3. *Sign and Size: Independent Predictability of Direction and Volatility at Sub-Minute Horizons*

## Draft abstract (numbers to re-verify against FINDINGS.md before use)

> Using a 236-feature representation of the limit order book computed at 100 ms cadence on BTC,
> ETH and SOL perpetual futures (2.17 M ticks/symbol), we document the predictive structure of the
> book at horizons of 1 s to 100 min. Directional information is large (rank IC up to 0.47 at
> 1–5 s), universal across symbols and volatility regimes, concentrated in eight independent
> feature axes led by order-book imbalance, and spectrally localized in the 0.005–0.1 Hz band with
> OU half-life 5–7 s. Volatility information (peak IC 0.35) resides in a disjoint feature family —
> arrival intensity, activity counts, toxicity — and the two channels are empirically orthogonal:
> directional features carry no volatility information and vice versa, on all three symbols. The
> signal decays toward efficiency in real time (half-life shortening as spreads tighten) and is
> sharply bounded at execution: conditioning on directionally-consistent fills collapses IC to
> ≈ 0.03, isolating adverse selection as the binding constraint. All results replicate on an
> out-of-time month. We discuss implications: two-block (sign × size) forecasting architectures,
> entropy-gated conditioning, and execution-timing overlays for slower strategies.

## Contributions

- **C1 — The census**: prediction power, mapped. 207 live features → IC by horizon/symbol with
  bootstrap CIs and FDR discipline; eight independent directional axes; per-symbol decay
  half-lives (30 s BTC / 20 s ETH / 15 s SOL).
- **C2 — Orthogonality**: the direction ⊥ volatility factorization (the headline novelty).
- **C3 — Localization & drift**: spectral band, OU dynamics, and the measured efficiency drift.
- **C4 — The execution boundary**: fill-conditioning decomposition + regime-gate recovery
  (+22 % lift, cross-symbol) — what the ICs do and do not imply.
- **C5 — Out-of-time replication**: every claim re-estimated on the fresh month (repairs the one
  failing cell of the old validation matrix: temporal-OOS delta −0.17).

## Section map — evidence in hand vs. new experiments

| Section | In hand (FINDINGS.md) | New (post-collection) |
|---|---|---|
| Census (C1) | §1 full scan, CIs, axes, decay | Phase 1 re-scan on fresh month |
| Orthogonality (C2) | §1 direction-vs-vol IC, 3 symbols | Confirm on fresh month (same scan, free) |
| Spectral + drift (C3) | §5 Spannung spectral; §3 drift May→Jun | Third time-point from new month |
| Execution boundary (C4) | §2 conditioning table; §5 entropy gate | Phase 2 conditional-IC replication; gate re-test |
| Replication (C5) | — | Phases 0–2 battery (data gate → scan → conditioning) |
| Implications (overlay, two-block sizing) | §4.1 OOS book | Optional: overlay backtest — include if ready, else future work |

Hygiene: BH-FDR across the battery, embargoed walk-forward, costs via `load_costs()`, planted test
before any new estimator.

## Timeline

- **Aug w1–3:** Phase 0 data gate → Phase 1–2 (re-scan, conditioning replication, gate re-test).
- **Aug w4 – Sep w1:** draft to camera-ready; numbers audited against FINDINGS.md; reproducibility
  appendix (git SHA, data fingerprints, script commands).
- **Mid-Sep:** SSRN (JEL G14/G12/C58; e-journals: Capital Markets: Market Microstructure +
  Financial Engineering) → Tier-1 outreach per P3/P4. Overlay backtest continues in parallel; if
  it lands early it joins the paper, otherwise it seeds the follow-up email and the next paper.

## Research Program (thesis chapters; doubles as §3 of the P4 research statement)

Each chapter is stated as a pre-registered hypothesis with existing evidence, the deciding
experiment, and a kill criterion. Thresholds are **imported at pre-registration time** from the
existing gate machinery (`config/agent.toml` 5-gate protocol, `config/alpha.toml` G-gates) — never
invented per-experiment. Ordering is by evidence strength, and that ordering is deliberate: it
demonstrates that capital-flavored ideas are queued behind their evidence, not ahead of it.

### Ch. 1 — The measurement foundation (this preprint)

The census, orthogonality, localization, drift, and the execution boundary (C1–C5). Everything
below consumes these estimates. Status: evidence in hand + one-month replication in progress.

### Ch. 2 — Signal-driven market making (strongest evidence)

- **Hypothesis:** a two-block extension of Avellaneda–Stoikov — directional family skews the
  reservation price (microprice-style), volatility family sets spread width and inventory limits,
  entropy gate arms/disarms quoting — reduces adverse-selection cost per fill vs. unconditioned
  A-S under an identical queue-simulation harness.
- **Existing evidence:** direction ⊥ volatility factorization (C2); entropy-gate IC lift (+22 %,
  cross-symbol); toxicity predictors (`toxic_effective_spread`, `hawkes_intensity`) in the census.
- **Experiment:** sim-first (backlog HF1 → HF4 → HF5); baseline-controlled comparison on recorded
  raw trades; costs via `load_costs()` only; planted test before the estimator touches real data.
- **Kill criterion:** no measurable adverse-selection reduction vs. baseline at imported gate
  thresholds → the two-block architecture is rejected, the census stands.
- **Literature anchor:** Avellaneda–Stoikov (2008); Guéant–Lehalle–Fernandez-Tapia;
  Cartea–Jaimungal (MM with alpha signals); Stoikov (microprice).

### Ch. 3 — The execution boundary: queue-priority pre-positioning (thesis core)

- **Hypothesis:** the fill-conditioning collapse (IC 0.45 → ≈ 0.03) is a property of *reactive*
  limit orders; *ex-ante* orders holding queue priority, with a toxicity veto, retain a
  materially positive fill-time IC because front-of-queue fills sample noise flow, not only
  adverse flow.
- **Existing evidence:** the decomposition itself (§2 of FINDINGS.md) — any-fill IC 0.52 vs.
  directional-fill ≈ 0.03 — which locates the loss at fill selection, not signal decay.
- **Experiment:** queue-position fill simulation on recorded raw trades (collecting since
  2026-06-09): IC conditional on queue rank at fill, with and without the veto.
- **Kill criterion:** if fill-time IC under best-case queue priority stays at the collapsed level,
  ex-ante placement is dead for this signal class and the chapter's contribution becomes the
  impossibility result — publishable either way.
- **Literature anchor:** queue value (Moallemi–Yuan); adverse selection and informed-trading
  measurement (Collin-Dufresne). *This is the open question the PhD proposes to answer.*

### Ch. 4 — Timescale structure: VWAP-deviation oscillation (partially confirmed)

- **Hypothesis:** VWAP deviation, treated as an (oscillatory) OU process rather than a level,
  occupies a spectral band distinct from the imbalance family's 0.005–0.1 Hz band, adding a
  timescale-diversified combiner input rather than a redundant one.
- **Existing evidence:** `flow_vwap_deviation` is already independent axis #5 of the census
  (IC −0.29/−0.21/−0.19 @1 s, mean-reverting) — orthogonality in the cross-section is confirmed;
  spectral distinctness is not.
- **Experiment:** PSD + OU fit of the deviation process per symbol; band-overlap test vs.
  imbalance; if disjoint, combiner ablation with/without the oscillation feature.
- **Kill criterion:** band overlap → redundant with axis 1; keep as census entry, no algorithm.
  (Explicitly *not* claimed as a standalone "superior algorithm" — its value is combiner
  orthogonality.)
- **Literature anchor:** the spectral/OU toolkit of Ch. 1 (C3).

### Ch. 5 — Cross-sectional microstructure momentum (contingent on infrastructure)

- **Hypothesis:** persistent MF/macro-horizon microstructure strength (slow features:
  `regime_divergence_1h`, funding, OI — *not* the 1–5 s family) disperses across a 30–100 perp
  universe and ranks tradably in the cross-section.
- **Existing evidence:** the liquidity ordering SOL > ETH > BTC (thinner book → stronger signal)
  is measured cross-sectional dispersion on n=3 — the necessary condition, at anecdote scale.
- **Experiment:** universe expansion of the ingestor (n=3 → n≥30), then FDR-disciplined
  cross-sectional ranking backtests with embargoed walk-forward.
- **Kill criterion / honest gate:** blocked until the data layer holds a clean streak at scale —
  the 37 % missing-day record on three symbols multiplies with the universe; this chapter is
  sequenced last *because* its infrastructure precondition is the project's known binding
  constraint.
- **Literature anchor:** cross-sectional crypto pricing (Malamud, Leippold).

## Supervisor fit

- **Bühlmann (ETH):** C1/C5 — high-dimensional census with FDR + stability/replication design.
- **Bölcskei (ETH):** C2/C3 — information factorization, spectral localization.
- **Malamud (EPFL):** C2 — orthogonal factor structure feeding pricing kernels.
- **Collin-Dufresne (EPFL):** C4 — adverse selection bounding visible information (keep his email's
  hook on the conditioning table; it remains a full section).
- **Teichmann (ETH):** implications — two-block architectures, learned execution.
- **Hugonnier (EPFL):** the instrument itself (perp funding mechanics in the census).

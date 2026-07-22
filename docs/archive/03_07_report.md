# NAT — Development Directions Report (2026-07-03)

Synthesis of the current documentation (`PLAN.md`, `METHODOLOGY.md`, `GLOSSARY.md`, `contracts/`,
`in_progress/`, `research/`, `ideas/`, `specs/`, `backlog/`): what directions the project can take
from here, what each costs, and what gates each one.

---

## The frame the docs impose

Everything in `docs/` orbits two facts:

1. **One binding constraint — data continuity.** The Jun-17 7-day clean streak was missed;
   re-establishing it (now via the T0b Hetzner cloud box, with su-35 frozen) gates the whole quant
   branch.
2. **One binding finding — execution, not signal discovery, is the blocker.** Raw IC ≈ 0.45 on L1
   imbalance collapses to ≈ 0.03 under realistic fills (adverse selection). The go/no-go for the
   entire trading business is the conditional-IC > 0.15 verdict (D1, ~Aug).

`PLAN.md` consolidates work into three official branches; beyond those, the docs contain five
substantial directions that are spec'd but unscheduled.

---

## The three official branches (`PLAN.md`)

### 1. Q — prove and trade the edge (production trading)

Streak → T0b cloud ingest box → longitudinal `nat oos --window` tool → revalidate the 5 winners on
≥30 clean days → adversarial `alpha-skeptic` kill gate → G8 14-day paper → live capital Sep–Oct at
1%→25%, human-gated, kill-switch first.

**Status:** mostly blocked on data. The plan itself recommends running the Q4 skeptic pass **now**
on existing data so ~90 days aren't accumulated for edges that die under refutation. The
kill-switch (Q3.1, ~6h) has no dependencies and must ship before anything live.

### 2. D — harden and ship the platform

D1 unified `nat viz *` + maturity tags (NAT9), D2 modularize the 5.1k-line `nat` monolith, D3
apt-installable packaging (gated on relocatable paths), D4 harden the discovery orchestrator +
agents into a cloud research lab surfaced via the `api` crate + Next.js.

**Status:** the old engineering backlog is essentially done (unified data layer, Sharpe
standardization, research API, website pages) — what remains is packaging and the viz/tag layer.
Fully unblocked, pure engineering.

### 3. P — publish, then PhD outreach

Five preprints exist: convolver (complete, the flagship), microstructure-alpha (complete — it *is*
the adverse-selection thesis), liquidity-heatmap, prism, and process (all three methods-only,
empirical validation pending). Track: polish → SSRN → arXiv `q-fin.TR` → 13 professor emails
(ETH/EPFL Tier-1) → applications; EPFL EDFI deadline **Jan 15 2027**.

**Status:** entirely data-independent, actionable today. As of this writing `process_related.tex`
is untracked in git and `PLAN.md`'s artifact list still says four preprints — the fifth needs
committing and a `PLAN.md` touch-up.

---

## Documented but unscheduled directions

### 4. Execution research — beat adverse selection (the make-or-break one)

The Spannung arc ended with a validated signal that's untradeable at taker fees, and a documented
viable path that nobody has built: Kalman extraction of the ultra-low band (0.005–0.1 Hz),
`ent_book_shape` regime gating (lifts IC 0.45 → 0.55–0.67), maker/zero-fee market-making at the
~7s OU half-life. Q2.5 exists as a task file but isn't in the do-now sequence.

Since every other Q-branch outcome is downstream of the conditional-IC verdict, this is arguably
the highest-leverage research direction in the repo. The liquidity-heatmap cascade model is its
sibling — blocked on the whale-data (K2) viability verdict.

### 5. ML wave portfolio — code done, data-gated

`research/new/ml_specs/`: Waves 0–3 (change-point, momentum, regime state machine, mean-reversion
LightGBM, meta-labeling, regime-conditioned LGBM, kNN retrieval) are **implemented and waiting for
~14 days of continuous bars to train**. Hard decision gates with an explicit "CASE_D = stop ML
work" outcome, so this direction self-terminates cleanly if the data doesn't support it. Also in
`PLAN.md`'s open bugs: the 3 existing ML algos need retraining against the current schema.

### 6. Autonomy scale-up — give the platform a brain

`ideas/microstructure_research_agent.md` specs a hypothesis engine (5 generators + bandit
meta-learning) on top of the execution layer, ~80% of which already exists as the 3 agents +
meta-agent. The cloud swarm (Tier 2) and Optuna evolution (Tier 3) are partially built with CLI
already wired (`nat swarm`, `nat evolve`). Direction: deepen from "daemons that sweep gates" to
"system that invents its own hypotheses." Blocked mostly by compute budget and the same data
constraint.

### 7. Novel-methods research — feeds both P and Q

`architecture/EXTENSIONS.md` has reference implementations (not integrated) for transfer-entropy
networks, information-geometry features, and HDBSCAN regime discovery on the Fisher manifold —
self-rated as the most publishable novelty in the repo. Prism (news-narrative pressure metrics)
opens a second data domain entirely — exogenous signals beyond microstructure. These are
paper-#2/#3 material for the P-track and regime-machinery for the Q-track.

### 8. Public research product

`specs/WEBSITE_SPEC.md`: a self-updating "living research paper" — LLM agents read papers,
implement them as strategies, backtest, and publish results including failures. Full 6-phase spec,
zero build. Plus the greenfield 3D mesh-graph viz (React Three Fiber behind the `/api/research/*`
contract). Payoff is visibility and PhD-outreach credibility rather than direct alpha; the largest
greenfield cost of anything here.

---

## How they interact

The docs' own decision structure: **D1 (~Aug, conditional-IC)** decides whether Q or P becomes
primary; **D2 (~Nov, ≥2 interested professors)** sizes further P effort. Directions 4 and 5 are
what make D1 winnable; 7 and 8 make D2 winnable; 6 and the D-branch compound everything else.

**Actionable today without any data:** T0b provisioning, the entire P-branch, the Q4 skeptic pass
on existing data, the kill-switch, D2/D3 packaging, and the three open P1 bugs (`nat agent status`
import error, GMM regime classifier, ML retrain).

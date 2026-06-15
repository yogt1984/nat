# NAT Automation Layer — Skills & Agents

**Date:** 2026-06-15 · **Companions:** `METHODOLOGY.md` (the loop this mechanizes),
`PRIORITIES.md` (tiers), `PLAN.md` (build order), `STATE_14_6_2026.md` (the binding constraint).

## Premise

`METHODOLOGY.md` specifies the development process tightly enough to **mechanize** it: a 7-step
inner loop, a 5-kind unit taxonomy with explicit contracts, a 6-rung maturity ladder tied to gates,
and a fixed guardrail set. So the automation layer is a direct translation:

- **Skills** = invocable workflows that mechanize the *loop steps* and *guardrails*.
- **Agents** = delegated specialist roles the loop keeps invoking (independent perspective, parallelizable).

The two highest-leverage targets, straight from the docs: automate **"gates not inspection"** and
**"planted-test-first,"** and point an adversarial agent at the **binding constraint** — the
IC≈0.45 edge collapses to ~0.03 under realistic fills, and the D1 go/no-go *is* the conditional-IC
verdict.

## Skills (workflows you invoke)

| Skill | Mechanizes | Tier | Status |
|---|---|---|---|
| **`/streak`** | Binding constraint (data continuity) + the su-35 no-touch guardrail | **P0** | **built** |
| **`/smoke`** | Inner-loop step 5 + test-pyramid L2 (real-parquet smoke) | **P0** | **built** |
| **`/ship`** | Inner-loop step 7 + git guardrails (re-check state, feat branch, maturity-tag-in-same-change, `merge --no-ff`) | **P0** | **built** |
| **`/slice`** | The whole 7-step inner loop for one unit (orchestrates `planted-test-author` → `/smoke` → `/ship`) | P1 | proposed |
| **`/gate`** | "Gates not inspection" — G1 (FDR) / G4 (walk-forward + deflated Sharpe) / G8 (paper), imported thresholds | P1 | proposed |
| **`/maturity`** | Promotion ladder as single source of truth (`SPEC→PRELIM→BETA→PROVEN→LIVE→RETIRED`, NAT9) | P1 | proposed |
| **`/schema-change`** | The "plan before any feature-vector/schema change" guardrail (236-vector ripple analysis) | P1 | proposed |

## Agents (delegated specialist roles)

| Agent | Role | Serves | Tier | Status |
|---|---|---|---|---|
| **`planted-test-author`** | Writes the failing planted test *first* (red) encoding a unit's contract, before implementation | Methodology step 2 | **P0** | **built** |
| **`alpha-skeptic`** | Adversarial validator — *tries to kill* a claimed edge (look-ahead, overlap, multiple-testing, regime, cost, **fill-conditional collapse**, decay, data-sufficiency) | The binding constraint / D1 | **P0** | **built** |
| **`gate-runner`** | Executes the scientific/economic gates, returns a verdict with imported thresholds (engine behind `/gate`) | Outer loop | P1 | proposed |
| **`contract-conformance-auditor`** | Audits a unit against its definition-of-done (contract methods, `@register`, command + maturity tag in same change, schema auto-sync) | Unit taxonomy + DoD | P1 | proposed |
| **`viz-frontend-engineer`** | Greenfield 3D track (R3F/Three.js/GLSL) against the `/api/research/*` JSON contract + vitest | Visualize stage | P3 | deferred (gated on conditional-IC > 0.15) |

## Reuse — do not rebuild

- **`/milestone`** — the cadence driver METHODOLOGY already names; `/slice` feeds into it.
- **`/deep-research`** — citation verify-before-preprint (P-track) + the literature→process pipeline.
- **`/code-review`**, **`/verify`** — engineering-correctness layer, distinct from `/smoke` (research-data correctness).
- **Agents** `Explore` (sweeps), `Plan` (slice specs / schema-change plans), `math-documentation-writer`
  (feature/process math + preprint), `plan-implementation-gap-scanner` (audit T0–T21 plan vs code).

## How they compose with the loop

```
INNER LOOP (per unit, human cadence)
  spec ─▶ planted-test-author (red) ─▶ implement ─▶ register+command+maturity tag
       ─▶ /smoke (real parquet) ─▶ snapshot ─▶ /ship (commit, merge --no-ff)
  └────────────────────── /slice orchestrates the above ──────────────────────┘

OUTER LOOP (autonomous daemons + gates)
  merged unit ─▶ /gate (G1→G4→G8 via gate-runner) ─▶ /maturity advances the tag
              ─▶ alpha-skeptic adversarially audits any "winner" before it earns trust

GUARDRAILS (always on)
  /streak (data continuity, no su-35)   /schema-change (plan before 236-vector edits)
```

## Build order (tied to the 30-day PLAN)

- **Stage A (now):** `/streak` (daily), then the inner-loop primitives `planted-test-author` →
  `/smoke` → `/ship`, then `alpha-skeptic` (it's on the D1 critical path — ready it for the
  conditional-IC verdict). **← this pass builds exactly these five.**
- **Stage B (the G1 gate):** `/gate` + `gate-runner` + `/maturity` so FDR screen + combiner
  revalidation produce mechanical verdicts.
- **As needed:** `/schema-change` (first feature-vector edit), `contract-conformance-auditor`
  (when slice cadence rises), `/slice` last (it only composes primitives that must exist first).
- **Deferred:** `viz-frontend-engineer` until conditional-IC is positive (PRIORITIES non-goal).

## Design notes

- **Skill vs agent split is deliberate.** A skill is a deterministic workflow you invoke; an agent
  is delegated reasoning with constrained tools. `/gate` (workflow) and `gate-runner` (executor) are
  the same logic at two entry points — matching how the outer loop already separates CLI from daemon.
- **`/slice` is built last on purpose** — it is an orchestrator over the three primitives; building
  it before they exist would just be a stub.
- **Gates imported, never invented** — every gate skill/agent reads thresholds from the canonical
  sources (G4 = walk-forward + deflated Sharpe; G8 = 14-day paper, 5 criteria; kill = ROADMAP Step 9).
- **`alpha-skeptic` defaults to "not proven"** under uncertainty and prefers to refute — the
  methodology's "looks right ≠ is right," operationalized.

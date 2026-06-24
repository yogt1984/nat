# Master Plan — Development Direction (from 2026-06-14)

**Companions:** `OBJECTIVE.md` (what) · `METHODOLOGY.md` (how we build) · `STATE_14_6_2026.md` (where we
are) · `PRIORITIES.md` (what matters) · `PLAN.md` (next 30 days). This file is the ≤200-line direction
summary; the others hold the detail.

## The thesis
We have a real edge (validated IC≈0.45) and the open question is no longer *"is there signal?"* but
*"can we capture it?"*. The entire direction is therefore to **turn validated information into captured,
risk-managed, live edge** — by building the autonomous pipeline that proves each signal through hard
gates (discover → validate → paper → live) — while a parallel academic track packages the same work
into a PhD application.

## Where we are going — Q-track (production)
1. **Solve execution feasibility first.** Build the trade-flow fill model on the raw trades; revalidate
   `hierarchical_combiner` on clean data. Go/no-go on conditional-IC > 0.15. *Nothing scales until this
   is positive.*
2. **Earn a clean substrate.** Protect the Jun-17 7-day streak; stand up a redundant cloud ingestor; wire
   the 82 dead features; unify costs + provenance; harden storage. Data continuity is the binding
   constraint — every clean day appreciates all prior research.
3. **Make promotion autonomous.** Signal-lifecycle spine + promotion daemon + kill-switch + risk-parity
   bridge, so discovered signals flow G1→G4→G8 to a human-approved LIVE with no manual shepherding.
4. **Prove, then deploy.** FDR alpha screen (G1) → combine/size/walk-forward (G4) → 14-day paper (G8) →
   live at 1% maker-only, scaling 1%→25% over 4+ months behind tier gates.
5. **Surface it.** Consolidate the visual layer under `nat viz *`; the greenfield 3D mesh-graph view comes
   online once there are validated signals worth rendering.

## Where we are going — P-track (academic)
Let data accumulate while drafting the preprint (*Event-Aligned SVD for pattern-kernel discovery*).
Provenance now (shared with Q1.4); camera-ready ~Aug; SSRN/arXiv ~Sep; professor outreach Sep–Nov;
EPFL/ETH applications Jan–Mar 2027; offer Apr–May 2027. The two tracks share data, methodology, and
infra — they compete only for *time*.

## The arc (four stages, then live)
- **Stage A (Jun 12–17)** — pure-Python + cloud: dead-features verdict, cloud ingestion, costs/provenance,
  lifecycle spine, CLI/viz foundations. Touch nothing on su-35.
- **Stage B (Jun 17–24)** — first real validation: FDR screen (G1), combiner revalidation, process
  transforms, daily agent; su-35 cuts over to the wired binary.
- **Stage C (Jun 24–Jul 8)** — automation + risk: promotion daemon, kill-switch, approval viz, signal bridge.
- **Stage D (Jul 8 →)** — observability, the 14-day paper window, polish.
- **Live (Sep–Oct)** — first human-approved LIVE at 1%, scaling through Q4 tier gates.

## Decision points
- **D1 (~Aug)** — "is there a trading business?" — conditional-IC verdict + first G8 window. The pivot.
- **D2 (~Nov)** — ≥2 professors interested (P-track viability).
- **D3 (paper)** — paper Sharpe within 2× backtest before scaling.
- **D4 (Apr–May 2027)** — PhD offer.

## Guardrails (non-negotiable)
Data continuity over research velocity · gates imported not invented (G4/G8/kill = ROADMAP) · all costs
via `load_costs()` · no live capital before G8 + a healthy kill-switch · plan before any
feature-vector/schema change · conventional commits / feat branches / `merge --no-ff`.

## What "done" looks like in ~90 days
A clean ≥30-day dataset; the dead-feature and conditional-IC verdicts recorded; the autonomous lifecycle
(discover → paper) running unattended behind a kill-switch; ≥1 signal through a 14-day paper window; the
preprint camera-ready. **Live capital only if D1 is positive.**

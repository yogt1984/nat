# NAT — Documentation Restructure Proposal

**Status:** DRAFT for approval · **Goal:** collapse the `docs/` sprawl into **one strategic spine
(`PLAN.md`) + a small, typed, meaningfully-named companion set**, with every task description living
in exactly one place.

> This is a *plan for the docs*, not a docs edit. Nothing moves until you approve the migration map
> in Part 5. Execution is staged (archive → merge → dedupe) so it is reversible at every step.

---

## 1. The problem, in numbers

| Symptom | Evidence |
|---|---|
| Sheer volume | ~**86,500** lines of markdown under `docs/` |
| One folder is a landfill | `docs/in_progress/` = **89 files**; its own `INDEX.md` says Tier-D directives (37) are "breadth-via-restatement", **N_eff ≪ 89** |
| Task specs scattered | `07_26_TASKS.md`, `backlog/*` (incl. 1,764-line `detailed_task_descriptions.md`), `in_progress/tasks_assigned_12_6_26/*`, `in_progress/korrektur_tasks.md`, root `tasks_05_28.md`, `critique_05_29.md` |
| Duplicate files | `architecture/architecture.md` (21 KB) **vs** `architecture/ARCHITECTURE.md` (38 KB) — a case-collision hazard |
| Two indexes | `PLAN.md` **and** `in_progress/INDEX.md` both try to be the map |
| Two archives | `docs/archive/` **and** `docs/architecture/archive/` (the latter is enormous) |
| Orphaned newer docs | `07_26_TASKS.md`, `03_07_report.md`, `PROPOSAL.md`, `INSTITUTIONAL_ALGORITHMS.md` — not indexed by `PLAN.md` |

**Root cause:** documents are created **by date and by topic-of-the-week** ("tasks_22_6", "03_07_report",
"tasks_assigned_12_6_26"), never merged back. The fix is to organize **by document *type* and
*lifecycle*, not by date** — and to enforce one canonical home per type.

---

## 2. Design principles

1. **One spine.** `PLAN.md` is the *only* living index and the single entry point. If a doc matters,
   `PLAN.md` links it in one line. There is no second index.
2. **One canonical doc per type.** Each *kind* of knowledge has exactly one living home (below).
   Everything else is either merged into it, archived, or deleted.
3. **Three lifecycle states only:** `living` (canonical, maintained) · `archived` (snapshot, frozen,
   read-only) · `deleted` (pure redundancy). A doc is always in exactly one.
4. **No dates in living filenames.** A dated name (`tasks_05_28`, `gauntlet_..._06_01`) is by
   definition a snapshot → it lives in `archive/`, never at top level.
5. **`ALL_CAPS.md` = authoritative living doc; `lowercase` = supporting reference.** The casing tells
   you at a glance what to trust.
6. **Reuse the classification we already have.** `in_progress/INDEX.md` already tiers 89 docs as
   A/B/C/D. That tiering *is* the migration key (Part 4). We don't re-classify; we route.

---

## 3. Target structure (the "structured set with meaningful names")

Eight living homes. Everything routes into one of them or into `archive/`.

```
docs/
  PLAN.md            ← THE SPINE. Strategy, gates, Current Focus, + the one index of all docs below.
  TASKS.md           ← THE one backlog. Every actionable task, one entry each. (new — absorbs 6 sources)
  OBJECTIVE.md       ← Mission. (durable, keep)
  METHODOLOGY.md     ← How we build/validate/promote. (durable, keep; absorb root CONVENTIONS.md)
  GLOSSARY.md        ← Shorthand decoder. (durable, keep)
  ARCHITECTURE.md    ← THE system reference. (merge the 2 dup files + Arch-p.1/2/3 + EAMM/V1/EXTENSIONS/PHASE1)
  commands.md        ← CLI reference. (durable, keep)

  contracts/         ← Unit contracts: feature / algorithm / process / viz. (already clean — keep as-is)

  research/          ← Empirical + academic knowledge (Tier A + B + papers)
    FINDINGS.md            ← merge the Tier-A reports (IC scans, gauntlet, combiner, feature/algo catalogues)
    ALGORITHMS.md          ← implemented-algorithm catalogue (keep)
    INSTITUTIONAL_ALGORITHMS.md ← the institutional gap audit (keep)
    PAPERS.md              ← merge PAPERS_IDEAS.md + per-paper reading notes
    papers/                ← the preprint .tex/.pdf artifacts (move loose research/*.tex|pdf here)
    math/                  ← durable theory (it_engine_mathematical_foundations, profiling math)

  specs/             ← ACTIVE forward blueprints only (Tier C that is NOT yet built)
    WEBSITE_SPEC.md · DASHBOARD_BUILD_PLAN.md · PROFILING_*.md · process_layer.md (←PROPOSAL.md)
                       (every dead/realized spec → archive/)

  runbooks/          ← Operational procedures (move root HETZNER_DEPLOYMENT_PLAN.md; cloud_deployment/*)

  ideas/             ← Speculative / P-branch seeds (spannung*). Small. (keep)

  archive/           ← THE one archive. (merge docs/archive/ + docs/architecture/archive/ + everything superseded)
```

Retired after migration: `docs/in_progress/` (drained into the homes above), `docs/backlog/`,
`docs/agent_specifications/`, `docs/convolver_docs/`, `docs/test_docs/`, `docs/requirements/`, the
second archive, and `in_progress/INDEX.md` (its job passes to `PLAN.md`).

---

## 4. The routing rule (drive the migration off `INDEX.md`'s tiers)

`in_progress/INDEX.md` already labels every file Tier A/B/C/D. Route mechanically:

| Tier (from INDEX.md) | Meaning | Destination |
|---|---|---|
| **A — Empirical findings** (10) | Measured truth about data/market/code | `research/FINDINGS.md` (merge; keep provenance + date inline) |
| **B — Reference / theory** (9) | Timeless method / math | `research/math/` or `contracts/` (whichever it defines) |
| **C — Design / architecture spec** (33) | Blueprint | If **not yet built** → `specs/` or `ARCHITECTURE.md`. If **built or dead** → `archive/` |
| **D — Directive / task-spec** (37) | Prescriptive intent (high redundancy) | If **still actionable** → one entry in `TASKS.md` (dedupe hard). If **done** → `archive/` |

The A/B docs are the ones with irreplaceable information — **merge, never delete**. The D docs are
mostly restatement — **collapse to one `TASKS.md` line each, then archive the source**.

---

## 5. Task consolidation (your specific ask)

All task/backlog/directive content collapses into **`TASKS.md`**, a single backlog with a fixed
schema, owned and indexed by `PLAN.md`:

```
## <Branch Q/D/P> · <ID> · <title>              [status: TODO|WIP|BLOCKED|DONE]
Priority: P0/P1/P2 · Effort: XS/S/M/L · Data-needed: in-hand | streak
Gate/Dep: <what unblocks it>
Spec: <link to the one spec doc if the detail is long, else inline>
```

Sources that merge into `TASKS.md` (then archive the originals):

| Source | Action |
|---|---|
| `docs/07_26_TASKS.md` | merge → TASKS.md, archive source |
| `docs/backlog/*` (4 files, incl. 1,764-line detail) | merge the *live* items; long detail → `specs/`; archive |
| `docs/in_progress/tasks_assigned_12_6_26/*` (Tier D) | merge live items; archive done ones |
| `docs/in_progress/korrektur_tasks.md`, `test_plan.md` | merge → TASKS.md |
| `docs/in_progress/convolver_implementation/*` (14), `nan_wiring/*` (5) | if built (per memory: nan-wiring verified, convolver published) → **archive**; else one TASKS.md entry each |
| root `tasks_05_28.md`, `critique_05_29.md`, `docs/03_07_report.md` | dated snapshots → **archive** |
| `docs/PROPOSAL.md` (14-point process plan) | its 14 items → TASKS.md entries; the detail → `specs/process_layer.md` |

`PLAN.md` then holds only the **strategic view** (Q/D/P gates, Current Focus) and links `TASKS.md`
for the itemized list — so "read `PLAN.md` first" gives you everything with one hop.

---

## 6. Full migration map (top level)

| Current | Action | New home |
|---|---|---|
| `architecture/architecture.md` + `ARCHITECTURE.md` | **de-dup + merge** | `ARCHITECTURE.md` |
| `architecture/{Arch-p.1,2,3, EAMM_SPEC, V1_SPEC, EXTENSIONS, PHASE1_ALGORITHM}.md` | fold live parts in; archive rest | `ARCHITECTURE.md` / `archive/` |
| `architecture/USER_MANUAL.md` | merge | `commands.md` |
| `architecture/archive/**` | **merge archives** | `archive/` |
| `agent_specifications/*` | fold into | `ARCHITECTURE.md` (agent section) |
| `cloud_deployment/*`, `in_progress/cloud_deployment/*`, root `HETZNER_DEPLOYMENT_PLAN.md` | consolidate | `runbooks/` |
| `research/*.tex|*.pdf` (loose preprints) | move | `research/papers/` |
| `research/{PAPERS_IDEAS}.md` | merge | `research/PAPERS.md` |
| `research/{EXP_0_INTRODUCTIONS, HYPER_DOCS, gauntlet_..._06_01, liquidity_heatmap_model}.md` | Tier A/B route | `research/FINDINGS.md` / `research/math/` |
| `specs/{PROFILING_*, WEBSITE_SPEC, DASHBOARD_BUILD_PLAN, ALPHA_RESEARCH_PLAN}.md` | keep if active | `specs/` |
| `in_progress/**` (89) | drain by tier (Part 4) | research / specs / TASKS / archive |
| `ideas/*` | keep | `ideas/` |
| `convolver_docs/`, `test_docs/`, `requirements/` | fold/archive | `research/` / `archive/` |
| root `CONVENTIONS.md` | merge | `METHODOLOGY.md` |
| root `FEATURES.md` | keep at root (referenced by CLAUDE.md), link from PLAN | — |

**Do not touch:** `contracts/`, the preprint sources, `OBJECTIVE/METHODOLOGY/GLOSSARY` (only additive
merges). These are already canonical.

---

## 7. Anti-sprawl conventions (so it doesn't regrow)

Add these to `METHODOLOGY.md` and enforce them:

1. **One-backlog rule.** New tasks go into `TASKS.md`. No new `*_TASKS.md` / `tasks_<date>.md` files —
   ever. A dated task file is an automatic smell.
2. **PLAN registration rule.** No new top-level doc without a one-line entry in `PLAN.md` the same
   commit. (The preprints already follow this — generalize it.)
3. **No dates in living names.** Dated = snapshot = `archive/`.
4. **Merge-on-touch.** When you revisit a topic, update its canonical doc; don't spawn a v2.
5. **`in_progress/` is deprecated.** Work-in-progress notes go in `TASKS.md` (status: WIP) or a branch,
   not a parallel doc tree with its own index.
6. **Quarterly archive sweep.** Anything `DONE` in `TASKS.md` and any realized spec → `archive/`.

---

## 8. Phased execution (safe, reversible, git-disciplined)

Each phase is one `feat`/`docs` branch → `merge --no-ff`. Content-preserving moves use `git mv` so
history follows the file.

- **Phase 0 — freeze & snapshot** (½ d): tag current state; confirm `in_progress/INDEX.md` tiering is
  still accurate (spot-check 10 files). *No deletions yet.*
- **Phase 1 — archive consolidation** (½ d): merge `architecture/archive/` → `archive/architecture/`;
  move superseded dated snapshots (`tasks_05_28`, `critique_05_29`, `03_07_report`) → `archive/`.
  Purely additive/relocating — zero information loss. **Note:** dated *findings* (e.g.
  `gauntlet_analysis_2026_06_01`) are Tier A — they are **not** archived here; they merge into
  `research/FINDINGS.md` in Phase 4 (never bury a finding).
- **Phase 2 — de-dup** (1 d): resolve `architecture.md` vs `ARCHITECTURE.md` into one; merge the
  architecture spec fragments. Diff-review before deleting either source.
- **Phase 3 — build `TASKS.md`** (1–2 d): drain all Tier-D directive sources into the single backlog;
  archive each source as it's absorbed. This is the big one and your primary goal.
- **Phase 4 — research routing — DONE (2026-07-25):** Tier-A findings merged into
  `research/FINDINGS.md` (all numbers + provenance preserved); Tier-B theory →
  `research/math/` (IT-engine math foundations, IC reference); drained residue + `INDEX.md` →
  `archive/in_progress/`; **`in_progress/` retired**; `gauntlet_analysis_2026_06_01` merged +
  archived. Remaining (optional polish): preprints→`papers/`, ops→`runbooks/`.
- **Phase 5 — rewrite `PLAN.md` as the sole index** (½ d): add the "Documentation Map" section
  linking the 8 homes + `TASKS.md`; retire `in_progress/INDEX.md`.
- **Phase 6 — front door, front-matter & conventions** (structure + readability, *no information
  reduction*):
  - **(a) `docs/README.md` front door — DONE.** A single navigable entry point: reading paths by
    role (new / operator / researcher / builder / PhD / planner) + a map by *document type*
    (a light [Diátaxis](https://diataxis.fr/) split: explanation / reference / how-to / research /
    spec / planning). Replaces the stale 49-line index (strict superset — nothing lost).
  - **(b) Front-matter schema.** Standard YAML header (`title/purpose/type/status/maturity/branch/
    updated`) on every living doc → self-describing + machine-readable, with `[PROVEN]/[PRELIM]/
    [SPEC]/[LIVE]` maturity badges (same tags as `nat viz` / D1).
  - **(c) Generated map.** Once (b) is stamped, generate the `README.md` map + a `PLAN.md`
    doc-index from front-matter so they never drift; retire `in_progress/INDEX.md` (Phase 5).
  - **(d) Conventions.** Append the §7 anti-sprawl rules to `METHODOLOGY.md`.
  - **Future readability (roadmap, not this phase):** `mkdocs-material` static site (search + nav,
    zero source change), ASCII→Mermaid diagrams, generate `commands.md`/`FEATURES.md` from source,
    and a `TL;DR` box + auto-TOC on long docs. All form-only, information-preserving.

**Rollback:** every phase is a merge commit; revert the merge to undo. `git mv` preserves blame.

---

## 9. End state (what "done" looks like)

Reading order for a newcomer becomes trivial and total:

```
PLAN.md  →  (strategy + gates + Current Focus + Documentation Map)
   ├─ TASKS.md         what to do now
   ├─ OBJECTIVE / METHODOLOGY / GLOSSARY   why / how / vocabulary
   ├─ ARCHITECTURE / commands / contracts  how the system + its units work
   ├─ research/        what we've learned + the papers
   ├─ specs/           what we're about to build
   └─ runbooks/        how to operate it
```

From **~89 in-progress files + 6 scattered task backlogs + 2 indexes + 2 archives**
→ **1 spine + 1 backlog + 8 typed homes + 1 archive**, no duplicates, no dated stragglers,
one index.

---

*Next action on approval: I execute Phase 0–1 first (pure archive/relocate, zero risk) and pause for
review before the merges in Phase 2–3.*

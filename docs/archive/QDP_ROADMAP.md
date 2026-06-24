# Q / D / P Roadmap — Consolidated Task Plan

**Date:** 2026-06-18
**Supersedes:** `in_progress/tasks_assigned_12_6_26/phd_vs_quant_roadmap.md` (2026-06-12, two-track Q/P).
This refresh **adds the D (development) track**, **corrects stale status labels** (the Jun-12 doc
predates ~400 commits), and pins the **PhD artifact locations** so they don't get lost again.

Three branches, one platform: **Q**uant (prove the edge), **D**evelopment (build + ship the tool),
**P**hD (publish + apply). They share infrastructure, data, and methodology — they compete for
*time allocation*, not resources.

> ⏰ **Time-sensitive:** the Jun-17 "7 clean days" data gate was **yesterday**. Verify the outcome
> (`nat gap status` / `/streak`, local read — does **not** contact su-35) **before** planning any Q
> execution. A passed streak lifts the su-35 freeze and unblocks the entire Q2 chain.

---

## 0. Status corrections since the Jun-12 roadmap

| Item | Jun-12 said | Reality (Jun-18) |
|---|---|---|
| SVD convolver preprint | "NOT STARTED" (`docs/preprint.tex`) | **Drafted** — `docs/research/convolver_preprint.tex` (1,172 lines) + compiled PDF. The expected `docs/preprint.tex` was never the real path. |
| Prof list / email template / pub strategy | implied future work | **Written + compiled** — `docs/phd_related/phd_application_guide.{tex,pdf}` (Jun-15). |
| `nat viz features\|algorithm`, `help --grep`, group help | proposed | **Shipped** (NAT4/5, T6). |
| `nat monitor` live probe, `nat doctor`, `viz render --last` | not in plan | **Shipped** (last week). |
| su-35 freeze ("zero contact until Jun-17") | active | **Re-evaluate** — gate date has passed; confirm streak first. |

---

## Branch Q — Quant: longitudinal revalidation of the winners

**Aim:** test the promising algorithms over a *longer* process with a dedicated terminal tool and a
dedicated ingest box, then see if they're really as good as they look.
**Hard constraint:** no live capital until paper trading passes G8 (14+ days). Gates imported, not invented.

| Task | What | Deliverable | Depends on | Status |
|---|---|---|---|---|
| **Q0** | Verify the Jun-17 streak (`nat gap status` / `/streak`) | 7-day clean-data verdict; freeze lifted or extended | — | ⏳ do first (gating) |
| **Q1** | Allocate the dedicated ingest box — provision T0b Hetzner per `HETZNER_DEPLOYMENT_PLAN.md`; 24/7 ingestor + Telegram <5min gap alert | Redundant independent feed; removes single-point su-35 risk | — | specced |
| **Q2** | Define the longitudinal terminal tool — generalize `nat oos30` → `nat oos --window <N>d` (a.k.a. `nat validate longitudinal`); walk-forward folds + deflated Sharpe; `--json` | One command runs the 5 winners over full clean history | Q0 | new (extends OOS harness) |
| **Q3** | Run extended revalidation — `3f_liquidity`, `jump_detector`, `funding_reversion`, `optimal_entry`, `surprise_signal` on ≥30 clean days | Per-symbol OOS Sharpe, IC decay, complementarity (<0.35 corr still holds?) | Q1+Q2+data | blocked on data |
| **Q4** | Adversarially kill each survivor — run the `alpha-skeptic` agent on every Q3 winner | Refutation report per algo (fill-conditional collapse, look-ahead, regime overfit) **before** any trust/paper decision | Q3 | new gate |

> **Framing (from `STATE_14_6_2026.md`):** signal *quality* looks solved, but IC≈0.45 collapses to
> ~0.03 under realistic fills. **Q2/Q3 only matter if Q4 survives** — build the skeptic in, don't
> bolt it on. Maps to existing detail docs `Q/Q2_1_hierarchical_revalidation.md`,
> `Q/Q2_2_alpha_screen_fdr.md`, `Q/Q3_2_paper_trading_deployment.md`.

---

## Branch D — Dev: harden `nat`, ship it, converge to the cloud lab

**Aim:** keep building the tool that continuously hunts microstructure alpha (solid terminal backend
that later converges to a cloud research lab), and make `nat` shippable as a native Debian tool.
Most of the Jun-12 CLI plan already shipped; what remains:

| Task | What | Deliverable | Status |
|---|---|---|---|
| **D1** | Finish the viz set + maturity tags — `nat viz portfolio`, `nat viz paper`, `nat viz spectral/regime`; `[PROVEN]/[PRELIM]/[SPEC]/[LIVE]` tags | Closes `nat_cli_improvement_plan.md` §VII items 5–8 (~20h) | unstarted |
| **D2** | Modularize the 5,113-line `nat` monolith → `scripts/cli/*.py` with a `register()` protocol (plan §V) | Per-domain files; prerequisite for clean packaging (~12h) | unstarted |
| **D3** | **Ship `nat` as an apt-installable tool** (see advice below) | `apt install nat` (or interim `pipx install nat`) | new |
| **D4** | Continuous-discovery backend → cloud research lab | Harden `discovery_orchestrator` + the 4 agents as always-on search; surface via the `api` crate + Next.js web (per `cloud_deployment/`) | partially built |

### D3 — Packaging advice (Python CLI + Rust binaries)

`nat` is a Python CLI that shells out to compiled Rust binaries (`./target/release/ing`, `api`, …)
and assumes CWD = repo root (e.g. `data_dir = "../data/features"`). That mixed-language, path-coupled
shape is what makes a clean `.deb` fiddly. Recommendation, phased:

1. **Prerequisite (do regardless of target): make `nat` relocatable.** Move to XDG-style paths
   (`~/.config/nat`, `~/.local/share/nat`, `/etc/nat`) with a `NAT_HOME` override; stop assuming
   CWD = repo. **Nothing installs cleanly until this is done** — it's the real blocker, and it pairs
   naturally with D2.
2. **Interim (days, not weeks): `pipx` / wheel.** Package via `pyproject.toml`; vendor the prebuilt
   `amd64` Rust binaries as package data. Gets you a one-command install now. *Not literally apt, but
   unblocks distribution immediately.*
3. **Destination: native `.deb` via `fpm` + a self-hosted apt repo** (or Cloudsmith/packagecloud).
   `fpm` bundles a Python venv + the Rust binaries into `/opt/nat` with a `/usr/bin/nat` wrapper —
   far less painful than raw `debhelper`/`dh-virtualenv` for a mixed payload. Build per-arch (amd64)
   in CI; the Rust binaries must be rebuilt per target. This delivers the true `apt install nat`
   experience.

**Verdict:** native `.deb` is the right destination, but it's gated on step 1 (relocatable paths) and
worth shipping step 2 (`pipx`) as the interim so distribution isn't blocked on the full repo.

---

## Branch P — PhD: publish the preprint, then reach out

**Goal:** PhD at ETH Zürich / EPFL in spectral microstructure / quant finance.
**Hard deadlines:** EPFL EDFI **Jan 15 2027** (Round 1) / **Mar 31 2027** (Round 2), Sep-2027 entry; ETH rolling.

### Where everything is (so it's never lost again)

| Artifact | Location |
|---|---|
| **Paper proposal / preprint** | `docs/research/convolver_preprint.tex` (+ `convolver_preprint.pdf`) — *Event-Aligned SVD for Pattern Kernel Discovery in Crypto Perps* |
| **Email template** | `docs/phd_related/phd_application_guide.tex` — **Part III, line 253** |
| **Prof list + profiles** | same file — **Part II, lines 87–225**; "Top Matches" 226, "Co-Supervision Pairings" 240 |
| **Condensed prof list + pitch angles** | `in_progress/tasks_assigned_12_6_26/P/P3_1_professor_outreach.md` (13 profs) |
| **One-pager** | `docs/phd_related/phd_application_summary.{tex,pdf}` |
| **Submission strategy (SSRN/arXiv)** | guide **Part I** (SSRN 35, arXiv 52, strategy 69) + `P/P2_1_publication_pipeline.md` |
| **Research framing** | `docs/ideas/spannung_phd_brief.md`, `docs/ideas/spannung.md` |

⚠️ Prof **email addresses are not stored** — only names/profiles; gathering them is part of P3.1.
⚠️ The `P/*.md` docs reference the old path `docs/phd_application_guide.tex`; files moved to `docs/phd_related/`.

### Corrected status

- ✅ Convolver preprint drafted; prof list + email template + SSRN/arXiv strategy written.
- ❌ Remaining: polish pass, SSRN upload, arXiv endorsement, the actual outreach emails; (optional)
  Spannung spectral/regime/cross-symbol sections (P1.2–1.4).

### Key decision

The Jun-12 plan wanted to *expand* the preprint with three more Spannung sections before publishing.
But the convolver paper is **already a complete, self-contained contribution.**
**Recommendation:** publish the convolver preprint **now** to establish presence; develop Spannung as
**paper #2.** This unblocks outreach months earlier instead of gating it behind ~26h more writing.

### The pipeline (next 5 actions)

| # | Action | Effort |
|---|---|---|
| P1 | Polish pass on `convolver_preprint.tex` → recompile → freeze camera-ready PDF | ~1 day |
| P2 | SSRN upload (guide Part I): independent-researcher profile, JEL G14/G12/C58, keywords → permanent URL | 1–3 business days |
| P3 | arXiv `q-fin.TR`: request endorsement (or via a responding prof) → cross-post | variable |
| P4 | Gather Tier-1 emails (Teichmann, Collin-Dufresne, Hugonnier, Malamud, Bölcskei); fill Part III template + per-prof pitch; send with SSRN link + PDF | ~5h |
| P5 | Track responses; Tier-2 (Acciaio, Cheridito, Bühlmann, Farkas, Leippold, Zdeborova, Krzakala, Kuhn) staggered +2 weeks; milestone = 2+ interested → formal applications | ongoing |

---

## Synergies & decision points (unchanged from Jun-12, still valid)

- **Shared work:** provenance/cost unification (P1.5 = Q1.4); Spannung spectral findings feed both the
  paper and the Kalman strategy (Q2.6); FDR methodology is identical for academic and production rigor.
- **D1 (Aug):** "Is there a trading business?" — Q gates G1–G4. If G1 fails, the research still has
  academic value → P becomes primary.
- **D2 (Nov):** "Do professors want this?" — P3 outreach. If zero interest, broaden beyond ETH/EPFL
  (Imperial, Oxford-Man, KTH, TU Delft) or double down on Q.

## Detail-doc pointers

- Q: `in_progress/tasks_assigned_12_6_26/Q/Q1_1 … Q3_3*.md`
- D: `in_progress/tasks_assigned_12_6_26/nat_cli_improvement_plan.md` + `nat_cli_tasks/NAT*.md`
- P: `in_progress/tasks_assigned_12_6_26/P/P1_1, P2_1, P3_1, P4_1*.md`
- Context: `STATE_14_6_2026.md`, `PRIORITIES.md`, `OBJECTIVE.md`, `METHODOLOGY.md`

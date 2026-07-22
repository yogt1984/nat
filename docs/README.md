# NAT Documentation — Front Door

Quantitative research platform for extracting alpha signals from Hyperliquid perpetual futures
(BTC / ETH / SOL). This page is the **single entry point** to every doc: start with a reading path
for your role, or jump to the map by document type. The strategic spine is
**[`PLAN.md`](PLAN.md)**; the itemized backlog is **[`TASKS.md`](TASKS.md)**.

> **Read first, always:** [`PLAN.md`](PLAN.md) (strategy, gates, Current Focus) →
> [`TASKS.md`](TASKS.md) (what to do now). Everything else below is reference for those two.

---

## Start here — reading paths by role

| If you are… | Read in this order |
|---|---|
| **New to NAT** | [`OBJECTIVE.md`](OBJECTIVE.md) → [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) → [`METHODOLOGY.md`](METHODOLOGY.md) → [`contracts/README.md`](contracts/README.md) → [`GLOSSARY.md`](GLOSSARY.md) |
| **Operating the system** | [`commands.md`](commands.md) → [`../HETZNER_DEPLOYMENT_PLAN.md`](../HETZNER_DEPLOYMENT_PLAN.md) → [`AUTOMATION.md`](AUTOMATION.md) → `TASKS.md` §REL (reliability) |
| **Doing research** | [`research/ALGORITHMS.md`](research/ALGORITHMS.md) → [`research/INSTITUTIONAL_ALGORITHMS.md`](research/INSTITUTIONAL_ALGORITHMS.md) → [`specs/process_layer.md`](specs/process_layer.md) → [`in_progress/INDEX.md`](in_progress/INDEX.md) |
| **Building a unit** | [`contracts/`](contracts/) (feature / algorithm / process / viz) → [`METHODOLOGY.md`](METHODOLOGY.md) → `TASKS.md` (pick an ID) |
| **On the PhD track** | `research/*_preprint.{tex,pdf}` → [`phd_related/`](phd_related/) → [`ideas/spannung.md`](ideas/spannung.md) |
| **Planning / leading** | [`PLAN.md`](PLAN.md) → [`TASKS.md`](TASKS.md) → [`DOCS_RESTRUCTURE_PROPOSAL.md`](DOCS_RESTRUCTURE_PROPOSAL.md) |

---

## Documentation map by type

Docs are grouped by **what they are for** (a light [Diátaxis](https://diataxis.fr/) split), not by date.

### Planning & strategy
| Document | Purpose |
|---|---|
| [`PLAN.md`](PLAN.md) | The Q/D/P strategic spine — gates, Current Focus, milestones. |
| [`TASKS.md`](TASKS.md) | The single itemized backlog (~80 tasks: Q/QA/D/P/REL/INF/BUG/PROC). |
| [`OBJECTIVE.md`](OBJECTIVE.md) | Mission + the end-to-end loop (Ingest → Discover → … → Deploy). |

### Explanation — understand *why / how it works*
| Document | Purpose |
|---|---|
| [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) | Current 4-layer system map, component matrix, key paths, design rationale. |
| [`METHODOLOGY.md`](METHODOLOGY.md) | How a capability is built, tested (planted-first), and promoted. |
| [`GLOSSARY.md`](GLOSSARY.md) | Decodes the planning shorthand (`T0`, `Q1.1`, `G4`, `su-35`, conditional-IC…). |

### Reference — look *up* a fact
| Document | Purpose |
|---|---|
| [`commands.md`](commands.md) | The `nat` CLI reference (~260 commands). |
| [`../FEATURES.md`](../FEATURES.md) | Authoritative feature manifest (236 features, formulas, references). |
| [`contracts/`](contracts/) | Per-unit contracts: [feature](contracts/feature.md) · [algorithm](contracts/algorithm.md) · [process](contracts/process.md) · [viz](contracts/viz.md). |
| [`research/ALGORITHMS.md`](research/ALGORITHMS.md) | Implemented-algorithm catalogue with OOS results. |
| [`research/INSTITUTIONAL_ALGORITHMS.md`](research/INSTITUTIONAL_ALGORITHMS.md) | Institutional-algo survey + HAVE/PARTIAL/GAP audit. |

### How-to / operations
| Document | Purpose |
|---|---|
| [`../HETZNER_DEPLOYMENT_PLAN.md`](../HETZNER_DEPLOYMENT_PLAN.md) | Cloud ingest-box deployment runbook (T0b / Q1). |
| [`AUTOMATION.md`](AUTOMATION.md) | Scheduled/automated workflows. |
| [`cloud_deployment/`](cloud_deployment/) | Docker stack, Prometheus/Grafana, observability & E2E. |
| [`test_docs/`](test_docs/) · [`requirements/`](requirements/) | Viz-validation testing guide + parquet-viz requirements. |

### Research & findings
| Document | Purpose |
|---|---|
| [`research/`](research/) | Experiment intros, `HYPER_DOCS`, liquidity-heatmap model, **preprints** (`*_preprint.{tex,pdf}`). |
| [`research/PAPERS_IDEAS.md`](research/PAPERS_IDEAS.md) | Literature review / reading bibliography. |
| [`synthesis/`](synthesis/) | Consolidated academic findings + build-implementation spec. |
| [`in_progress/INDEX.md`](in_progress/INDEX.md) | A/B/C/D information-content index of remaining in-progress docs (findings + references). |

### Specs — forward blueprints (what we're about to build)
| Document | Purpose |
|---|---|
| [`specs/process_layer.md`](specs/process_layer.md) | The 14-point process/IT discovery layer (PROC-1..18). |
| [`specs/`](specs/) | `WEBSITE_SPEC`, `DASHBOARD_BUILD_PLAN`, `PROFILING_*`, `ALPHA_RESEARCH_PLAN`. |
| [`agent_specifications/`](agent_specifications/) · [`convolver_docs/`](convolver_docs/) | Agent + convolver design detail. |

### PhD
| Document | Purpose |
|---|---|
| `research/*_preprint.{tex,pdf}` | The 5 preprints (convolver, microstructure, liquidity-heatmap, prism, prism-signal). |
| [`phd_related/`](phd_related/) | Application guide, prof list, email template, one-pager. |
| [`ideas/`](ideas/) | Speculative seeds — `spannung`, research-agent, terminal reference. |

### Meta — about the docs themselves
| Document | Purpose |
|---|---|
| **`README.md`** | This front door. |
| [`DOCS_RESTRUCTURE_PROPOSAL.md`](DOCS_RESTRUCTURE_PROPOSAL.md) | The docs-restructure plan (phases 0–6) + anti-sprawl conventions. |
| [`archive/`](archive/) | Superseded/historical docs, frozen for provenance (incl. the old `ROADMAP.md`, `MASTER_PLAN.md`, prior planning layers). |

---

## Front-matter convention *(new — enables a generated map)*

Every living doc should carry this header so it is **self-describing and machine-readable**. The
map above is hand-maintained today; once headers are stamped, it can be **generated** from them
(so it never drifts — the failure mode the restructure exists to prevent):

```yaml
---
title: <short title>
purpose: <one line — what this doc is for>
type: explanation | reference | how-to | tutorial | planning | research | spec
status: living | reference | archived
maturity: PROVEN | PRELIM | SPEC | LIVE      # optional; same tags as nat viz / D1
branch: Q | D | P | cross                    # which Q/D/P branch owns it
updated: YYYY-MM-DD
---
```

**Rules (see [`DOCS_RESTRUCTURE_PROPOSAL.md`](DOCS_RESTRUCTURE_PROPOSAL.md) §7):** one entry per task
in `TASKS.md` (never a new `*_TASKS.md`); register every new top-level doc here **and** in `PLAN.md`
the same commit; no dates in living filenames (a dated name → `archive/`).

---

## Related directories (repo root)

- [`../FEATURES.md`](../FEATURES.md) — feature manifest · [`../CLAUDE.md`](../CLAUDE.md) — project conventions for Claude Code · [`../CONVENTIONS.md`](../CONVENTIONS.md) — coding conventions.
- `../config/` — runtime config (ingestor, agent, pipeline, alpha, costs, …).
- `../reports/` — experiment results, JSON data, paper-trade comparisons.
- `../scripts/`, `../rust/` — the implementation (Python research + Rust ingestor).

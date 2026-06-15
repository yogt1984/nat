# NAT Glossary — decode the planning shorthand

The strategic docs (`STATE_*`, `PRIORITIES`, `MASTER_PLAN`, `PLAN`, `in_progress/*`) use a dense
private shorthand. This is the decoder. Read it once; it makes every plan doc legible at speed.

## Machines & infra

| Term | Meaning |
|---|---|
| **su-35** | Second machine running the production ingestor. **HARD RULE: zero contact until the Jun-17 7-day clean streak completes** — its uptime is the milestone's critical dependency. |
| **T0b cloud box** / Hetzner AX52 | The redundant cloud ingestor (Tier-1 docker stack + wired binary). Where wiring/validation is deployed *instead of* su-35 during the streak freeze. |
| **the streak** | A run of consecutive clean-data days. Target: **7 consecutive (Jun 11–17)** — needed because N_eff and cluster stability require ≥7 clean days. |

## The edge (the core finding)

| Term | Meaning |
|---|---|
| **IC** | Information Coefficient — rank correlation between a feature and the future return. The headline edge: **IC≈0.45** on `imbalance_qty_l1` (1–5s, ~30s half-life). |
| **conditional-IC** | IC *after* realistic mid-cross fills. **The go/no-go number: tradeable iff > 0.15.** Raw IC≈0.45 collapses to **~0.03** on taker fills — the binding problem. |
| **adverse selection** | The reason conditional-IC collapses: you get filled precisely when the signal is about to move against you. The whole "execution feasibility" question. |
| **fill model** | Trade-flow model (on raw trades collected since Jun-09) that estimates realistic fills → yields conditional-IC. |
| **N_eff** | Effective sample size after autocorrelation — why overlapping windows don't give "free" data. |

## Gates (imported, never invented)

| Term | Meaning |
|---|---|
| **G1** | BH-FDR alpha screen across 191 feats × 3 symbols. The first real-data validation gate. |
| **G4** | Walk-forward + deflated Sharpe: OOS Sharpe>0.5, OOS/IS>0.7, deflated p<0.05, maxDD<5%, ≥30 trades, PF>1.2. |
| **G8** | 14-day paper window, 5 criteria. Calendar-bound; first APPROVAL_PENDING ~Aug. |
| **G1–G9** | The alpha-pipeline quality gates (`scripts/alpha/alpha_pipeline.py`). |
| **5-gate protocol** | The agents' per-hypothesis ladder: discovery (IC+ΔIC) → cost → temporal → symbol → correlation-dedup, then BH-FDR. |
| **BH-FDR** | Benjamini-Hochberg false-discovery-rate control (q=0.05) — kills false positives across many tests. |
| **deflated Sharpe** | Bailey & López de Prado Sharpe, penalized for the number of trials. |
| **kill thresholds** | The 4 risk-daemon triggers (ROADMAP Step 9). Imported. |

> **Rule:** gates and thresholds are **imported from ROADMAP, never invented.** "Looks right" ≠ "is right."

## Maturity ladder

`SPEC → PRELIM → BETA → PROVEN → LIVE → RETIRED` — the single source of truth for how far a unit has
earned its way. See `contracts/README.md` for what earns each tag. (Surfacing in `nat help` lands
with NAT9.)

## Task / work IDs

| Prefix | Meaning | Source |
|---|---|---|
| **T0, T0b, T1…T21** | Reconciled task sequencer | `in_progress/tasks_assigned_12_6_26/plan.md` (rev 6). T0=resolve 82 dead features; T0b=cloud ingestion; T11=alpha screen; T14=promotion daemon; T16=kill-switch; T21=live deploy. |
| **Q1.1, Q2.3…** | Production-track gate items | `docs/in_progress/Q/*` |
| **P1, P1.5…** | Academic-track (PhD preprint) items | `docs/in_progress/P/*` |
| **D1–D4** | Decision points | D1 (~Aug)=conditional-IC verdict, "is there a trading business?"; D2 (~Nov)=≥2 professors; D3=paper Sharpe within 2× backtest; D4 (Apr–May 2027)=PhD offer. |
| **NAT1–NAT10** | CLI/integration infra | NAT1/2=auto-help; NAT3–8=unified `nat viz *`; NAT9=maturity tags in help; NAT10=modularize the `nat` monolith into `nat/commands/*`. |
| **HF*/LF*/MF*** | Feature/algo candidates by frequency | HF=high-freq (e.g. HF1 microprice, HF3 Hawkes, HF4 VPIN gate); LF=low-freq (LF1 funding-settlement, LF3 cascade); MF=medium-freq. |
| **F1–F5** | Feature-gap candidates | settlement-clock (F1), microprice-deviation (F2), multi-band OFI (F3), HAR-RV (F4)… |
| **M2** | Milestone: continuous cloud ingestion | `PLAN.md` |

## Data-quality shorthand

| Term | Meaning |
|---|---|
| **dead / NaN-padded features** | Of 236 features, ~82 emit NaN because their source isn't wired. Committed wiring unlocks 40 (whale 12 / liquidation 13 / concentration 15), pending a **48h viability verdict**. |
| **K2 columns** | A class of known-dead columns processes must *skip with a reason*, never crash on. |
| **K5 zombie process** | The stuck-process incident that caused the Jun 4–10 (6-day) data outage; the watchdog now guards against it. |
| **viability verdict** | The T0 decision on dead features: **viable / noisy / unavailable** (per `tasks_assigned_12_6_26/01_concentration_viability_assessment.md`). |

## The OBJECTIVE loop (capability stages)

`Ingest → Discover → Cluster → Visualize → Generate → Validate → Deploy` — see `OBJECTIVE.md`. Each
has a maturity in `STATE_*.md`'s subsystem table.

## Citizens & contracts

**feature / algorithm / process** (three code citizens) + **cli-capability / viz-component**. Each has
a contract in [`contracts/`](contracts/README.md). The methodology is **contract-first vertical
slices** — see `METHODOLOGY.md`.

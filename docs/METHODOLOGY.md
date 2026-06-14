# NAT Development Methodology

**Contract-First Vertical Slices on a Continuous Research Substrate.**

## In one paragraph
We build NAT one small vertical slice at a time, where every slice is a single contract-bound
unit — a feature, an algorithm, or a process — that gets written test-first (a planted synthetic
signal must pass before any real data touches it), implemented to its contract, registered and
surfaced as a `nat` command with a maturity tag in the same change, smoke-tested on the latest
real parquet before commit, then shipped on a feat branch through CI. Once merged, the unit isn't
"done" — it flows automatically into the always-on daemons that sweep it through the gate ladder
(IC → cost → temporal → symbol → FDR → paper → live), and its maturity tag
(`SPEC → PRELIM → BETA → PROVEN → LIVE`) is the honest, single source of truth for how far it has
earned its way. To keep that loop cheap we first harden the integration substrate (CLI maturity
tags, auto-help, modularization, a unified `nat viz *` group), test in a pyramid (planted →
real-data smoke → snapshot → CI, then the scientific/economic gates), and keep the one greenfield
piece — the 3D mesh-graph visualization — decoupled behind the `/api/research/*` JSON contract so
the frontend never blocks the research loop.

## Core principle
One interface (`nat`), three first-class citizens (feature / algorithm / process), one
definition-of-done. Every increment is a *registered, contract-bound, dual-tested (planted +
real-data), maturity-tagged* unit that the autonomous daemons can immediately sweep. Research
correctness is enforced by **gates, not inspection** — in alpha work, "looks right" ≠ "is right."

## Unit taxonomy
What "develop a feature" concretely means — each increment is exactly one of:

| Unit | Lives in | Contract | Registry |
|------|----------|----------|----------|
| **feature** | `rust/ing-features` | `count()` / `names()` / `to_vec()`; Parquet schema auto-syncs (`output/schema.rs`) | field in the `Features` struct |
| **algorithm** | `scripts/algorithms/` | `alg_features()` / `required_columns()` / `step()` / `run_batch()` | `@register` |
| **process** | `scripts/processes/` | `evaluate()` (Evaluation) or `transform()` (Transform) → `ProcessResult` | `@register` |
| **cli-capability** | `nat` (→ `nat/commands/*` after NAT10) | parser + handler + maturity tag | `build_parser()` |
| **viz-component** | `web/src/components/` | reads the `/api/research/*` JSON contract | `nat viz *` |

## Inner loop — per unit, contract-first / test-first
1. **Spec one slice** — the smallest shippable unit, exactly one citizen.
2. **Write the planted test first (red)** — a planted synthetic frame (`scripts/processes/synthetic.py` for processes, a planted-IC frame for algorithms, a schema-count test for features). *Lesson from Stage 1: three estimator bugs were caught only by planted tests.*
3. **Implement to the contract** until the planted test is green.
4. **Register and surface in `nat`** — `@register` (auto-discovery) + one command (parser + handler) + a **maturity tag**, all in the same change.
5. **Real-parquet smoke** on the latest day — assert no-error and correct output shape; dead / K2 columns are skipped with a reason, never crashed. (Smoke before commit.)
6. **Snapshot / regression** baseline if existing outputs are touched.
7. **Commit** — feat branch, conventional message, minimal diff, `merge --no-ff` to master, CI green.

## Outer loop — continuous and autonomous
A merged unit is **not done**; it is done when it has flowed through the gates to a maturity
verdict. Registered processes and algorithms automatically become candidates the daemons sweep:
- `scripts/discovery_orchestrator.py` — DATA_HEALTH → SIGNAL_SWEEP → TRAINING → BACKTESTING → ALPHA_PIPELINE → REPORTING.
- the three research agents + `scripts/agent/meta_daemon.py` — 5-gate protocol with BH-FDR control.
- `scripts/alpha/alpha_pipeline.py` — G1–G9 quality gates.

Continuous development is the human inner-loop cadence **feeding** an autonomous outer loop that
runs the `docs/OBJECTIVE.md` pipeline 24/7.

## Promotion ladder = maturity
Maturity is the single source of truth, mechanically tied to the gates:

`SPEC → PRELIM → BETA → PROVEN → LIVE → RETIRED`

- **PRELIM** — planted + smoke pass, merged.
- **BETA** — passes discovery-IC + cost gate on real data.
- **PROVEN** — temporal + symbol replication + FDR-survived + ≥ `nat oos30` / paper Sharpe.
- **LIVE** — paper→live thresholds met + human approval.
- **RETIRED** — IC decay (> 14 days).

The tag is surfaced in `nat help` / `nat commands --json` (NAT9) so a SPEC command never *looks*
PROVEN.

## Enabling infrastructure first
The integration substrate — not the features — is the bottleneck, so front-load it:
1. **NAT9 maturity tags + NAT1/2 auto-help / group-help** — kills help-text drift; integration becomes one step.
2. **NAT10 modularize** the 5.4k-line `nat` monolith into `nat/commands/*.py` — removes the merge-conflict tax.
3. **NAT3–8 unified `nat viz *`** — one home for the visual layer, including the greenfield 3D track.

Cheap integration *is* the point of "integrating features into the nat terminal tool."

## Test pyramid
**Engineering correctness (before merge):**
1. Planted-signal (synthetic) — correctness gate; mandatory before any real-data use.
2. Real-parquet smoke (latest day) — wiring / contract gate; before commit.
3. Snapshot / regression — output stability.
4. CI (`.github/workflows/ci.yml`) — fmt/clippy/`cargo test`, `pytest`, `vitest`, criterion 10% regression gate.

**Economic correctness (after merge, autonomous):**
5. Scientific gates — IC / cost / temporal / symbol / FDR (agents) + alpha G1–G9 + deflated Sharpe.
6. Paper → live — economic gate, human-in-the-loop.

Both halves are required; level 1 before any real-data use is non-negotiable.

## Greenfield track — 3D visualization
The interactive 3D mesh-graph visualization (React + React Three Fiber / Three.js with custom GLSL
shaders) is the one major greenfield piece. It feeds the **JSON contract** exposed by the `api`
crate (`/api/research/*`) plus structured research JSON — never the database directly. It is
launched via `nat viz *` and tested with `vitest` against the `ProcessResult` shape. The JSON
contract is the only coupling, so the frontend stack never blocks the research inner loop.

## Cadence & governance
- Milestone-driven (`/milestone`); data-gated work is sequenced around the 7-day clean-data streak (N_eff and cluster stability need ≥ 7 clean days).
- **Plan before any feature-vector / schema change** — it ripples to Parquet and every downstream reader.
- Git: conventional commits, feat branches, `merge --no-ff`, no PRs; re-check git state before each commit (concurrent activity in this checkout).

## Why this is optimal
- **Single interface + contracts** → minimal integration friction; units are parallelizable.
- **Planted + real + gates** → false discoveries die early — the expensive failure mode in alpha research.
- **Units feed the daemons** → every increment compounds into 24/7 research instead of shelfware.
- **Maturity ladder** → an honest readiness signal from raw idea to live capital.

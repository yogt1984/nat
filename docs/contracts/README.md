# NAT Contracts — the contract-first SSOT

> One interface (`nat`), **three code citizens** (feature / algorithm / process) plus two
> integration citizens (cli-capability / viz-component), **one definition-of-done**.
> This directory is the executable form of `../METHODOLOGY.md` — read the one contract for the
> unit you're building, then follow its checklist. Don't reconstruct the contract from prose.

## Unit taxonomy

| Unit | Lives in | Contract file | Registry |
|------|----------|---------------|----------|
| **feature** | `rust/ing-features` | [`feature.md`](feature.md) | field in the `Features` struct |
| **algorithm** | `scripts/algorithms/` | [`algorithm.md`](algorithm.md) | `@register` |
| **process** | `scripts/processes/` | [`process.md`](process.md) | `@register` |
| **cli-capability** | `nat` (→ `nat/commands/*` after NAT10) | *(2nd-wave)* | `build_parser()` |
| **viz-component** | `web/src/components/` | *(2nd-wave; P3-deferred)* | `nat viz *` |

Every increment is **exactly one** citizen — registered, contract-bound, dual-tested
(planted + real-data), maturity-tagged.

## Definition of Done (the inner loop, as a checklist)

The same 7 steps apply to every code unit. The per-unit contract file gives the exact test
commands and skeleton; this is the shared spine.

- [ ] **1. Spec one slice** — the smallest shippable unit, exactly one citizen.
- [ ] **2. Planted test first (red)** — a synthetic frame with a *known* answer, written before any
      real parquet is touched. *(Stage-1 lesson: three estimator bugs were caught only this way.)*
- [ ] **3. Implement to the contract** until the planted test is green.
- [ ] **4. Register + surface in `nat`** — `@register` (or struct field) **+** one command **+** a
      **maturity tag**, all in the same change.
- [ ] **5. Real-parquet smoke** on the latest day — assert no-error and correct output shape; dead /
      unavailable columns are skipped with a reason, never crashed. *(Smoke before commit.)*
- [ ] **6. Snapshot / regression** baseline if you touched existing outputs.
- [ ] **7. Commit** — feat branch, conventional message, minimal diff, `merge --no-ff` to master,
      CI green. **Re-check `git status` first** (concurrent activity in this checkout).

A merged unit is **not done** — it is done when it has earned a maturity verdict through the gates.

## Maturity ladder (the single source of truth for readiness)

`SPEC → PRELIM → BETA → PROVEN → LIVE → RETIRED`

| Tag | Earned when |
|-----|-------------|
| **SPEC** | designed, not yet implemented |
| **PRELIM** | planted + smoke pass, merged |
| **BETA** | passes discovery-IC + cost gate on real data |
| **PROVEN** | temporal + symbol replication + FDR-survived + ≥ `nat oos30` / paper Sharpe |
| **LIVE** | paper→live thresholds met + human approval |
| **RETIRED** | IC decay (> 14 days) |

> **Tagging mechanism note:** the *surfacing* of tags in `nat help` / `nat commands --json` lands
> with **NAT9** (not built yet). Until then, record the tag in the unit's docstring / spec and the
> lifecycle DB; the contract still requires you to *assign* one at merge (PRELIM).

## Hard rules that touch every contract

See `../../CLAUDE.md` → **Guardrails**. The two that bite contract work most:
- **Plan before any feature-vector / schema change** — it ripples to Parquet and every reader.
- **Gates imported, not invented** · **costs only via `load_costs()`** — never hardcode a threshold
  or a fee.

Acronyms in any plan/spec you hit: see [`../GLOSSARY.md`](../GLOSSARY.md).

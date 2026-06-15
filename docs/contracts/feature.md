# Contract: Feature

**Lives in** `rust/ing-features` · **registry** field in the `Features` struct (`features/mod.rs`)

A **feature** is a number computed from market state every 100ms and written to Parquet. The feature
vector is a **hard contract**: `to_vec()` length == `count_all()` == `names_all()` length == the
Parquet schema. Get this wrong and every downstream reader (parquet loaders, algorithms, ML) breaks.

> ⚠️ **PLAN BEFORE ANY FEATURE-VECTOR / SCHEMA CHANGE.** It ripples to Parquet and every reader.
> This is a Guardrail (see `../../CLAUDE.md`), not a suggestion.

## Signature — adding a feature category

```rust
// 1. The category struct
struct MyCategory { /* … */ }
impl MyCategory {
    fn count() -> usize { N }
    fn names() -> Vec<String> { vec!["my_feat_a".into(), /* … N total */] }
    fn to_vec(&self) -> Vec<f64> { vec![/* … exactly N, NaN where unavailable */] }
}
```
Then in `features/mod.rs`:
1. Add the field to the `Features` struct (base = always computed; optional = NaN-padded when source absent).
2. Wire into `to_vec()`, `names_all()`, `count_all()` — **optional categories use the NaN-padding pattern**.
3. Schema auto-syncs via `create_schema()` in `output/schema.rs` — do **not** hand-edit the schema.

**Invariants**
- `to_vec()` always returns exactly `count_all()` elements (currently the full vector; verify with the
  schema-count test, don't hardcode the number here — it drifts).
- `names_all()` order == `to_vec()` order, exactly.
- Optional/unavailable sources → **NaN padding**, never a panic, never a shortened vector.

## Tests

```bash
# 1. Planted: schema-count test — names_all().len() == to_vec().len() == count_all():
cd rust && cargo test -p ing-features            # includes the count/contract tests
cd rust && cargo test -p ing-features -- <name>  # single test

# 2. Real-parquet smoke after a build, before commit:
nat build && nat run serve        # then confirm the new columns appear, correct NaN behavior
# read the latest parquet and check: column present, non-NaN when source live, NaN-padded otherwise
```

## Definition of Done (+ the shared 7 in [`README.md`](README.md))

- [ ] **Planned** the schema change (it's a Guardrail).
- [ ] Schema-count planted test green: `names_all().len() == to_vec().len() == count_all()`.
- [ ] Optional category uses NaN-padding; base category always computes.
- [ ] `output/schema.rs` auto-synced (not hand-edited); benches/tests using `FeaturesConfig` updated.
- [ ] Real smoke: column present in latest parquet, NaN behavior correct on dead sources.
- [ ] Maturity tag (PRELIM on merge) · feat branch · `merge --no-ff` · CI green (fmt/clippy/test + criterion 10% gate).

> Counts (feature total, category count) live in the code (`names_all()`), **not** in prose docs —
> CLAUDE.md and the strategic docs point at the SSOT rather than hardcoding, because they drift.

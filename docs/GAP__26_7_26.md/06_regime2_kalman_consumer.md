# 06 · REGIME-2 — Give `config/kalman.toml` a Consumer

**Prio** P1 · **Effort** M · **Depends** in-hand · **Status** TODO
Source: `docs/TASKS.md` (REGIME-2).

## Methodology

`config/kalman.toml` already declares the thesis — `[kalman.ou_prior]` (θ≈0.1, half-life ≈7s,
`sigma_process`, `sigma_obs`) and `[kalman.regime]` (`feature = "ent_book_shape"`,
`percentile = 30`) — but **nothing in `scripts/` or `rust/` loads it** (verified by grep).
`kalman_imbalance` runs on hardcoded constructor defaults, and the `ent_book_shape` gate is
inert. Wire the config as the single source of truth for the filter and its regime gate, or
retire the file so it can't rot.

## Bottom line

Closes dead wiring: `ent_book_shape` is the mechanism that lifts **IC 0.45 → 0.55** (Spannung
Phase E). Until a consumer exists, neither `kalman_imbalance` nor Q2.6 (05) can be config-driven
or regime-gated. Small, but a prerequisite for 05 and 04.

## Implementation

- Load `config/kalman.toml` in `scripts/algorithms/kalman_imbalance.py` (mirror the
  `config/algorithms.toml` consumer pattern already in the repo).
- Source `theta`, `sigma_process`, `sigma_obs`, `dt`, `horizons` from `[kalman]`/`[kalman.ou_prior]`
  instead of constructor literals.
- Implement `[kalman.regime]`: blank/NaN the algorithm outputs when
  `ent_book_shape ≥ P{percentile}` (gate closed) — same NaN-graceful contract as every algorithm.
- **Retire path (if chosen):** delete `config/kalman.toml`, remove the dangling reference, note it
  in `docs/TASKS.md`. Prefer wiring — the gate is proven.

## Verification

- **Planted test first:** set a distinct `theta_init` in a temp config → assert the filter's `theta`
  equals it (config actually flows); feed rows straddling `ent_book_shape` P30 → outputs blank above,
  populate below.
- **Conformance:** `pytest scripts/tests/test_bar_level_dispatch.py` (algorithm dispatch/contract).
- **Smoke:** `nat algorithm evaluate --algorithm kalman_imbalance --symbol BTC` with vs without the
  gate on real parquet → conditional IC rises in the gated run.

## Acceptance

- [ ] Config values demonstrably drive filter params (planted assert)
- [ ] `ent_book_shape` percentile gate blanks outputs when closed
- [ ] Contract test + real-parquet smoke green; gated conditional IC > ungated
- [ ] (Or) config retired and all references removed, recorded in TASKS.md

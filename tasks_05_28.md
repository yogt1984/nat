# NAT Tasks — 2026-05-28

Remaining work from the state_05_26.md critique remediation and new items surfaced during implementation.

---

## Open from Original Critique

- [ ] **Dashboard reverse proxy** — Add Caddyfile for unified routing (:80 → :8050/:8060/:3000). Low priority; multi-port works for single-user dev.
- [ ] **Full ing-features crate extraction** — Extract `features/` (13.6K LOC, 26 files) into `ing-features` workspace crate. Requires untangling deep coupling with state/, ws/, ml/, config/. Dedicated session.

## New — Surfaced During Remediation

- [x] **Tier 3 algorithm depth** — all 4 algorithms fixed:
  - [x] `spread_decomp` — causal realized spread via prev emission mid-price (commit `3c35bcb`)
  - [x] `3f_liquidity` — IC-weighted composite from training forward returns (commit `886fc1d`)
  - [x] `oi_divergence` — z-score normalized divergence replaces magic constants (commit `3c35bcb`)
  - [x] `surprise_signal` — erf(|z|/sqrt(2)) transition probability, no free params (commit `886fc1d`)
- [x] **Python daemon PID files** — All daemons now write PID files via `_write_pid_file()` (commit `640a28d`)
- [x] **Config duplication cleanup** — Scanner/profiler load taker_bps from costs.toml (commit `7e2bd7c`)

## Completed (state_05_28.md)

- [x] Makefile split into make/*.mk (commit `2e8af10`)
- [x] Config consolidation → costs.toml (commit `831ce58`)
- [x] ing-types crate extraction (commit `ea888e8`)
- [x] sys.path.insert removal + pyproject.toml (commit `d06f375`)
- [x] Disk-full guard in ParquetWriter (commit `36b6e7c`)
- [x] Expanding-window IC (commit `d474ae1`)
- [x] Funding rate in PnL (commit `7f115d6`)
- [x] Maker fill probability in cost model (commit `478ed5f`)
- [x] Covariance-aware portfolio weights (commit `0131045`)
- [x] Delete online_ridge (commit `438ad4f`)
- [x] Delete multi_level_imb (commit `d498963`)
- [x] kalman_imbalance auto_tune (commit `ddad1df`)
- [x] entropy_momentum hysteresis (commit `bca1730`)
- [x] hawkes_intensity param estimation (commit `ddad1df`)
- [x] funding_reversion OU half-life (commit `7aca955`)
- [x] weighted_ofi decay estimation (commit `d02cc72`)
- [x] cascade_probability class weighting (commit `dcd6f02`)
- [x] MI stride subsampling (commit `8d19145`)
- [x] CMI sample guard (commit `7f06539`)
- [x] Quantile-based entropy binning (commit `7bc84f5`)
- [x] Permutation entropy m=5 (commit `6f0d07e`)
- [x] KSG nonparametric TE (commit `7f06539`)
- [x] IT cost kurtosis correction (commit `8146e9b`)
- [x] Wire IT engine into alpha pipeline (commit `7457c72`)
- [x] Redis publish rate limiting (commit `a434a37`)
- [x] Alert file fallback (commit `95b0063`)
- [x] Makefile pkill → PID file (commit `d65da3a`)
- [x] shell=True minimization (commit `66876b9`)
- [x] Config schema validation (commit `e6d882b`)
- [x] IT engine tests in CI
- [x] Pin Python dependencies (commit `e78eb84`)

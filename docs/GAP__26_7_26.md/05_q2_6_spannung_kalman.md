# 05 · Q2.6 — Spannung Kalman Backtest on Zero-Fee Pairs

**Prio** P1 · **Effort** ~8h · **Depends** data-in-hand (runs on existing parquet) · **Status** NOT STARTED
Source: `docs/archive/.../Q/Q2_5_spannung_kalman_filter.md`, `docs/ideas/spannung.md` (Phase D–E).

## Methodology

Spannung Phase D found the entire **IC ≈ 0.45 lives in the ultra-low band (0.005–0.1 Hz,
periods 10–200s)**; mid/high bands are noise. The band has **OU dynamics** (half-life 5–7s,
Hurst 0.43, dominant return-coherence at 0.015 Hz ≈ 68s). A Kalman filter extracts that slow
component, where 100ms–1s latency is negligible:

- **State** `x(t)` = latent OU imbalance; **observation** `y(t)` = raw `imbalance_qty_l1_last` (noisy).
- Trade the **mean reversion** of `x_filtered`: entry when `|x_filtered| > 1σ`, exit on return to
  mean, size ∝ deviation.
- **Gate on `ent_book_shape < P30`** (Phase E: lifts IC 0.45→0.55+).
- Evaluate at **0 / 0.5 / 1.5 bps** — the 60s-hold edge may clear cost where the 5s-take edge (7 bps
  RT taker) never could.

## Bottom line

The single experiment that answers **"does the viable path actually make money?"** Its result
feeds Q5 (conditional-IC gate) directly and doubles as the empirical core of the
microstructure-alpha preprint. Not blocked on the data streak — actionable now.

## Implementation

- **Reuse** `scripts/kalman/ou_filter.py` (`OUKalmanFilter`, `estimate_ou_params`, `auto_tune_filter`).
- **New** `scripts/backtest/spannung_kalman.py`:
  1. Fit AR(1) `x_{t+dt}=μ+φx_t+σε` → OU `θ=-ln φ/dt`, half-life `ln2/θ` (validate 4–8s).
  2. Run Kalman filter → `x_filtered`.
  3. Mean-reversion backtest (entry/exit/size as above).
  4. Apply `ent_book_shape < P30` gate.
  5. Score at 3 cost levels via `load_costs()`.
- **Output** `reports/spannung_kalman_results.md`.

## Verification

- **Planted test first:** synthetic OU series (known half-life 6s) → estimator recovers 4–8s;
  synthetic `ultra-low sinusoid + white noise` → filter isolates the < 0.1 Hz component (energy check).
- **Real-parquet smoke:** run on `data/features/<clean date>/` for BTC/ETH/SOL before commit.
- **Gates (imported):** filtered IC > 0.3 @10s (> 0.45 with gate); Sharpe > 2 @0bps; Sharpe > 0 @0.5bps.

## Acceptance

- [ ] OU half-life estimate ∈ 4–8s
- [ ] Filtered signal IC > 0.3 @10s fwd; > 0.45 with `ent_book_shape` gate
- [ ] Spectral check: filtered energy concentrated < 0.1 Hz
- [ ] Sharpe > 2.0 @0bps · Sharpe > 0 @0.5bps · loss quantified @1.5bps
- [ ] `reports/spannung_kalman_results.md` written; planted + smoke green

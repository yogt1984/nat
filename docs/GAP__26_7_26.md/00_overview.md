# GAP — 2026-07-26 · Execution capture: the units that decide Q5

**One gate governs the whole trading business — Q5**, the conditional-IC > 0.15 go/no-go (~Aug):
raw `imbalance_qty_l1` / `flow_vwap_deviation` score **IC ≈ 0.25–0.48**, but **naive execution
collapses that to ~0.03** — taker is arithmetically impossible (0.5–2 bps move vs ~11 bps RT), and
naive maker bleeds it away through adverse selection. Every unit here must state *how it survives
that*. Each file is self-contained (< 200 lines): **methodology · bottom line · implementation ·
verification**.

The folder has two layers, simplest → deepest:

**A. Toxicity × VWAP — the simplest profit attempts (01–04).** Design principle: *VWAP deviation
supplies the direction; toxicity supplies the permission.* From [`../research/FINDINGS.md`](../research/FINDINGS.md):
`flow_vwap_deviation` is a real mean-reverting axis (IC ≈ 0.25 @1s, already ⅓ of `3f_liquidity`);
toxicity (`toxic_vpin_50`) carries **zero directional IC** — a *condition* variable, not a
direction one (the `vpin_regime` signal lost −7,331 bps OOS). Using VPIN to *veto* is its
institutionally-correct use; using it to *pick direction* is the mistake already falsified.

**B. Spannung / Kalman execution research — the Q5-deciding experiments (05–07).** Deeper
extraction of the same edge: Kalman-filter the ultra-low OU band, wire the proven `ent_book_shape`
regime gate, and price the queue. These answer whether the 0.45 is maker-capturable *at all*.

## Index

| # | File | Unit | One-liner | TASKS.md |
|---|------|------|-----------|----------|
| 01 | [vwap_reversion](01_vwap_reversion.md) | `vwap_reversion` | Fade z-scored deviation from rolling VWAP (bar-scale baseline) | new QA row |
| 02 | [toxicity_gate](02_toxicity_gate.md) | `toxicity_gate` | Trade only when VPIN is low — a shared veto, not a strategy | **HF4** |
| 03 | [toxic_vwap_reversion](03_toxic_vwap_reversion.md) | `toxic_vwap_reversion` | Fade VWAP deviation *only* in low-toxicity states (headline) | new QA row |
| 04 | [microprice_maker_sim](04_microprice_maker_sim.md) | `microprice_maker_sim` | Quote around microprice, pull on toxicity; degenerate A/S → **HF5** | **HF1** (+HF5) |
| 05 | [q2_6_spannung_kalman](05_q2_6_spannung_kalman.md) | Q2.6 | OU-Kalman ultra-low-band backtest on zero-fee pairs | Q2.6 |
| 06 | [regime2_kalman_consumer](06_regime2_kalman_consumer.md) | REGIME-2 | Wire `kalman.toml` + the `ent_book_shape` gate (IC 0.45→0.55) | **REGIME-2** |
| 07 | [a4_queue_value](07_a4_queue_value.md) | A4 | Expected value of a resting limit order (queue-reactive) | **A4** |

## Build sequence

```
Data in hand — start here:
  06 wire kalman.toml gate ─┐
  02 toxicity gate (HF4) ───┼─→ 05 Kalman backtest (Q2.6)      ← the Q5 experiment
                            └─→ 03 toxic_vwap_reversion         ← headline
  01 vwap_reversion baseline ──→ 03

Execution capture (sim-first):
  02 + 04 microprice_maker_sim (HF1) ─→ 07 queue-value (A4) ─→ HF5 full Avellaneda–Stoikov
```

Expected quality ordering (falsifiable): **03 > 01** on net PnL (the gate removes knife-catching);
**02** improves any base signal it wraps; **04 / 05** are the only ones attacking the
fill-conditional collapse head-on (the Q5 experiment in miniature). **05 and 03 run on data already
in hand — start there.**

## Rules that bind every design here

- **Contract:** `MicrostructureAlgorithm` ABC (`scripts/algorithms/base.py`), `@register`, `alg_`
  prefix, `step()` returns exactly `alg_features()` keys, NaN-in → NaN-out, warmup blanking, params
  in `config/algorithms.toml`.
- **Planted (synthetic) test first** (red → green; `planted-test-author`). Each file specifies its test.
- **All costs via `load_costs()`** — no literal fees anywhere.
- **Thresholds are percentile params in config, never invented constants** (gates imported).
- **Verification chain:** `pytest scripts/tests/test_bar_level_dispatch.py` → real-parquet smoke
  (`/smoke`) → `nat algorithm evaluate --algorithm <name> --symbol BTC` → `nat oos30`.
- Execution units (**04 / 05 / 07**) are **sim-first** — no live capital before G8 + a healthy kill-switch.
- Adoption = one row each in [`../TASKS.md`](../TASKS.md); this folder is design detail referenced
  from those rows.

**Input columns (verified vs `FEATURES.md`):** `flow_vwap_5s`, `flow_vwap_deviation`,
`toxic_vpin_50` / `_10` / `_roc`, `toxic_flow_imbalance`, `toxic_effective_spread`, `raw_spread_bps`,
`vol_returns_1m` / `_5m`, `imbalance_qty_l1_last`, `ent_book_shape` (+ Python-side `microprice`,
`scripts/features/microprice.py`).

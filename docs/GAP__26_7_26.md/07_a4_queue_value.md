# 07 · A4 — Queue-Value Execution Model (sim-first)

**Prio** P2 · **Effort** M · **Depends** HF1 · **Status** TODO · **sim-first**
Source: `docs/TASKS.md` (A4). Ref: Huang, Lehalle & Rosenbaum (2015), *queue-reactive model*.

## Methodology

The micro-decision under every market-making quote: **is this resting limit order worth its
adverse-selection risk?** Expected value of an order posted at level `L`, queue position `k`:

```
EV = P(fill | k, arrival/cancel rates) · spread_capture
   − P(adverse fill) · adverse_selection_cost
```

Fill probability comes from queue dynamics (arrivals ahead/behind, cancellations); the
adverse-selection cost comes from **post-fill drift** — how far price moves against you right after
a fill. When `EV < 0`, don't post (or cross instead). This gives HF5 per-order granularity: *where*
in the book, and *whether*, posting is +EV.

## Bottom line

Answers **"is this resting order +EV after adverse selection?"** — the decision underneath every
A/S quote. Makes HF5's fills smarter and quantifies the queue edge on zero-fee books. Complements
04 rather than competing with it.

## Implementation

- **New** `scripts/execution/queue_value.py`:
  - Queue-position tracker reconstructed from book updates.
  - Fill-probability model with intensities estimated from data.
  - Adverse-selection cost from post-fill drift (reuse `scripts/kalman/drift_analysis.py`).
  - `EV(post at level L, queue pos k)` → used as an A/S quote filter or a standalone signal.
- **Costs via `load_costs()`**; sim-first (no live orders).

## Verification

- **Planted test first:** deterministic queue where fill is certain and drift is zero ⇒
  `EV = spread_capture`; certain adverse fill ⇒ `EV < 0` (order suppressed).
- **Sim smoke:** replay real parquet book updates ⇒ compare realized vs predicted fill rate and
  check EV calibration (predicted-vs-realized PnL per posted order).
- **Gate:** sim-only; no live capital path.

## Acceptance

- [ ] EV sign correct on planted certain-fill / certain-adverse cases
- [ ] Predicted fill rate ≈ realized on real-parquet replay (calibration check)
- [ ] Integrates as an HF5 quote filter; costs via `load_costs()`; sim-only

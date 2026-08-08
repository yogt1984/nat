# NAT — Findings Summary

**One page over [`FINDINGS.md`](FINDINGS.md).** Every row links to the section carrying the
method, sample and caveats; nothing here is a source. When the two disagree, `FINDINGS.md`
wins. *Compiled 2026-08-08.*

---

## The one-sentence state

A large, universal, structurally-validated signal exists (order-book imbalance, IC ≈ 0.45 at
1–5 s, all three symbols, both vol regimes) and **no execution path tested so far can monetise
it** — as of 2026-08-08 nothing capital-relevant in the record is unrefuted.

## The pattern behind every failure

Every refuted result shares one property: **the predicted move is comparable to the cost.**
That is the diagnosis, not a mood — and it is why the surviving candidates all live at
horizons or instrument counts where the ratio is 5–100× rather than ~0.1×.

---

## 1. What has been refuted

| # | Tried | Result | Mechanism | §|
|---|---|---|---|---|
| 1 | Taker capture at 1–5 s | **dead by arithmetic** | 0.5–2 bps move vs 11 bps RT | §2 |
| 2 | Naive maker fills | IC 0.45 → **0.03 / −0.06 / −0.03** | fill requires the adverse move | §2 |
| 3 | The 5 shipped "winners" | **5/5 killed** | wrong venue cost tier; harness never ran their own entry logic | §4.6 |
| 4 | VWAP reversion (taker), 58 d | −25k…−41k bps | ~50 trades/day × fees | §4.5 |
| 5 | Textbook Avellaneda–Stoikov | −1.89 bps/fill | risk-widened quotes sit *behind* the touch → price-through fills, adverse by construction | §4.8 |
| 6 | Touch quoting, 8 cells × 179 episodes | **0 survive** | fails day-consistency + concentration, not sign | §4.9 |
| 7 | HYPE-staking fee tiers | **tier-invariant** | discounts apply to fees, never to rebates | §4.10 |
| 8 | Maker fee ladder | breakeven **+0.144 bps** vs a zero-fee ceiling | and the EV gate is *non-monotone* in the tier | §4.11 |
| 9 | Momentum persistence, 1m/5m | **anti-persistent**, 34/36 cells negative | bar returns mean-revert | §5 |
| 10 | VWAP band excursions, 330 cells | **0 survive** | **~1.5 events/day** — underpowered, *not* absent | §5 |
| 11 | Permutation entropy as a xs score | IQR 0.0005 | the middle half of the universe is indistinguishable | §7.3 |
| 12 | Rotation OOS, 6 configurations | 0/6 → **4/6** beta-neutral | fails DSR + OOS/IS; power needs ~0.89 yr | §7.7, §7.8 |
| 13 | Hierarchical combiner | **loses to one of its own inputs** | weights fitted *after* the window they were scored on | §5.1 |
| 14 | Agreement gating | **harmful** | the *disagreement* subset carries the signal | §5.1 |

## 2. What survived — the actual toolkit

| Asset | Measured effect | Caveat | §|
|---|---|---|---|
| **A4 EV gate** | flips per-fill sign −1.66 → **+0.67** | non-monotone in fee tier; re-derive per tier | §4.9, §4.11 |
| **HF1 microprice centre** | liquidation cost **−25…−40 %** | effect is on *inventory*, not per-fill PnL | §4.7, §4.8 |
| **HF4 VPIN gate** | Sharpe lift 3/3 symbols | **still not a registered unit** | §4.5 |
| **`ent_book_shape`** | +22 % IC lift, low-entropy quintile | replicated cross-symbol | §5 |
| **XS-3 rank predictability** | **survives its kill test** | 83 non-overlapping folds | §7.4 |
| **`imbalance_qty_l1` @5 s** | durable **77–86 %** of 62 days | MI is *information*, not direction — VPIN scores 44 % on the same test | §5 |
| Direction ⟂ volatility | vol-IC 0.29–0.35, direction-IC **exactly 0** | the only *proven* orthogonality | §1 |

## 3. Structural facts that constrain everything

- **NAT has been studying the tight tail of its own venue.** Universe median half-spread
  **1.372 bps = 17.7× BTC**; 169/177 pairs are wider, and SOL is in the tightest five. (§7.2)
- **The whole universe is thin at the touch**, and spread/depth are *uncorrelated*
  (ρ = −0.107). Median touch notional $33–$391. Only 4 of 177 pairs hold $5 k. (§7.10)
- **Venue history is capped** at ~5000 bars/interval: 1 m expires in ~3.5 days and **cannot be
  backfilled, ever** — it can only be accumulated. (§7.1)
- **Data continuity is the binding constraint** and has been since June. (§7)
- **Orthogonality is a full-sample property, not a forward one** — it holds with a fat tail, and
  raw depth asymmetry is mechanically redundant with imbalance. (§5)

## 4. Still open — and why they are *time*-blocked, not work-blocked

| Question | State | What it needs |
|---|---|---|
| **B-5a** — does adverse selection scale slower than `h^0.70`? | median β\* = 0.698; capacity caps it at ~10 wide pairs × ~$1 k | **one tick measurement** on one wide pair (`B-5b`) |
| **Track C** — beta-neutral rotation | 4/6 criteria; Sharpe 1.06 → 2.12, turnover *fell* | ~325 rebalances ≈ **0.89 yr** (`n ∝ 1/SR²`) |
| **Oscillation harvesting** | not refuted — starved of events | 177 pairs ⇒ ~265 events/day instead of 1.5 |
| **Trend continuation** | refuted at 1m/5m on 3 symbols | untested at 15m/1h/4h × 177 pairs |

## 5. The method is the durable asset

Three suggestive numbers died under their own tests **in a single week** — a 0.192 orthogonality
drift (89th percentile of its own distribution), a 23 % durability smoke (did not replicate at
full budget), and a +45.8 bps band markout (q = 0.33, zero durable days). A fourth was mine: an
IC of 0.39–0.46 that was *higher* than the claim it audited, produced purely by overlapping
forward windows.

What catches these is now standard equipment: null calibration against the right
exchangeability (PROC-12), BH-FDR across the whole grid (PROC-13), per-day durability rather
than pooled means (PROC-4), pre-registered criteria committed before the run, and controls
designed to make the method fail if it is broken. **In every case this week, the control was
what killed the hypothesis — not the reasoning.**

---

*Sources: [`FINDINGS.md`](FINDINGS.md) §§1–8. Plans live in [`../PLAN.md`](../PLAN.md) and
[`../TASKS.md`](../TASKS.md) — nothing here is a plan.*

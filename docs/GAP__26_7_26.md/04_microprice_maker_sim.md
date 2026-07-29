# ALG-4 · `microprice_maker_sim` — the primitive quoting engine (TASKS.md **HF1**, seed of HF5)

**Idea:** the only design in this folder that attacks the §2 fill-conditional collapse directly.
Quote *both* sides around a forward-looking fair price (microprice, not VWAP), pull quotes when
toxicity is high, skew on inventory. Simulation-first — this is the **Q5 experiment in miniature**,
not a deployable.

## Why microprice replaces VWAP here

VWAP is a *lagging* average of where trades happened; the **microprice** (Stoikov 2018) is a
queue-imbalance-weighted forecast of where mid is *going*:

```
P_micro = P_ask · Q_bid/(Q_bid+Q_ask) + P_bid · Q_ask/(Q_bid+Q_ask)
```

It embeds exactly the imbalance information that carries IC 0.45 — as a *price* rather than a
signal. Quoting around a center that already leans with imbalance is the textbook mechanism for
reducing adverse selection on fills: your bid fades away from toxic sellers before they hit it.
The feature exists (`scripts/features/microprice.py`, F2 done); nothing consumes it — that gap is
HF1.

## Formulation (degenerate Avellaneda–Stoikov: fixed risk aversion, no terminal time)

```
center_t = P_micro,t − η · inv_t · σ_t          # inventory skew: long inventory → lower quotes
δ_t      = ½·spread_t + c · σ_t                 # half-width: spread floor + vol cushion
bid_t    = center_t − δ_t     ask_t = center_t + δ_t

quote both sides iff  alg_tox_gate = 1  (ALG-2)  — else cancel both ("pull on toxicity")
inventory hard cap ±I_max; breach → passive unwind toward flat
```

Free parameters: `η` (skew), `c` (vol cushion), `θ` (gate percentile), `I_max`. All in config;
`σ_t` from `vol_returns_1m`. Full A-S (γ, T, k calibration) is HF5 — deliberately out of scope.

## What it measures (success ≠ Sharpe yet)

Run against recorded book/trades (`data/trades/`, collected since 2026-06-09) with the existing
`MakerFillSimulator` (`scripts/kalman/fill_sim.py`) extended to two-sided quoting. Report:

1. **Fill markout** — E[mid_{t+h} − fill price | fill, side] for h ∈ {1s, 5s, 30s, 5m}. The §2
   record says naive maker markout ≈ negative-to-zero (conditional IC +0.03/−0.06/−0.03). The
   question: does microprice-centering + toxicity-pulling move markout **positive at any h**?
2. **Adverse-selection decomposition** — markout split by `alg_tox_gate` state at fill and by
   quoting center (microprice vs mid vs VWAP — 3-way ablation, same sim, same window).
3. Spread capture vs adverse selection vs inventory cost — the three P&L components separately.

**The Q5 linkage is explicit:** if markout stays ≤ 0 under every (center, gate, δ) configuration
tried, that is strong evidence the 0.45 is not maker-capturable *by us* and Q5 leans no-go —
capital pivots to the MF/macro book. A positive markout at any horizon is the first crack in the
§2 wall and defines exactly what HF5 should industrialize.

## Planted test (write first)

Deterministic replayed book with scripted informed bursts (marked) and noise flow: assert
(a) with a perfectly-informed oracle gate, markout > 0 by construction (sim plumbing correct);
(b) with gate off, markout reproduces the §2-style negative drag; (c) inventory never exceeds
±I_max; (d) fills only occur when the replayed trade price crosses the resting quote (no
optimistic fills — the exact failure mode §2 exists to prevent).

## Honesty constraints

- **Sim only.** No paper/live path until Q5 passes gates — this folder does not amend the ladder.
- Fill model conservatism: assume queue-back position (fill only when price *trades through* the
  quote, not touches), zero rebates unless `load_costs()` says otherwise.
- Latency: assume ≥1 tick (100 ms) reaction lag everywhere; NAT is not fast — the sim must not
  pretend otherwise.

**Evaluate:** planted → sim on ≥5 clean days of `data/trades/` → ablation table (center × gate)
into `research/FINDINGS.md` as a new dated section — this result is a finding either way.

## Graduation → HF5 (full Avellaneda–Stoikov)

If markout turns positive under any (center, gate, δ), industrialize this primitive into the full
inventory-optimal maker (**HF5** · P2 · L · depends HF1 · HF4 · kill-switch · sim-first):

- **Reservation price** `r(s,t) = s − q·γ·σ²·(T−t)` — skews quotes to mean-revert inventory to 0.
- **Optimal spread** `δ_ask + δ_bid = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/κ)`.

vs this file's degenerate form (fixed γ, no terminal `T`): HF5 calibrates `γ, κ, T` from data and
leans `r` on the Kalman ultra-low signal (`05`). Build in
`scripts/execution/avellaneda_stoikov.py` on the same `fill_sim`/`limit_order_sim` harness.
**No live capital before G8 + a healthy kill-switch.**

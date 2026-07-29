# ALG-3 · `toxic_vwap_reversion` — the combination (headline of this folder)

**Idea:** fade the oscillation around VWAP **only when the deviation is noise, not information** —
and let VPIN be the classifier. Direction from ALG-1, permission from ALG-2, in one unit.

## The mechanism, stated precisely

A deviation `d_t` from micro-VWAP has two generators:

| Generator | What happens next | Correct action | Toxicity fingerprint |
|---|---|---|---|
| **Liquidity noise** (inventory shocks, sloppy flow) | reverts to fair value | fade (profit) | VPIN low |
| **Informed flow** (someone knows something) | price *continues* to a new fair value | stand aside | VPIN high, rising |

Fading indiscriminately (ALG-1 alone) earns the noise-reversion and pays the informed-continuation
— the classic adverse-selection tax, the same force behind the §2 conditional-IC collapse. The gate
splits the two populations *ex ante*. This is Easley–López de Prado–O'Hara's use of VPIN
(liquidity providers withdrawing under toxic flow) applied to a reversion book.

## Formulation

```
z_t        = rolling z-score of flow_vwap_deviation          (window W_z)
vpin_pct_t = rolling percentile of toxic_vpin_50             (window W_p)

signal_t = −sign(z_t) · 1[|z_t| > k_entry]                  # ALG-1 direction
           · 1[vpin_pct_t < θ]                              # ALG-2 permission
           · 1[raw_spread_bps < P90]                        # sanity: no quoting into blown spreads

size_t   = base · (1 − vpin_pct_t)                          # optional: continuous toxicity sizing
exit: |z_t| < k_exit, or vpin_pct_t ≥ θ (gate closes → flatten), or T_max bars
```

The **gate-closes-flatten** rule matters: if toxicity spikes mid-hold, the noise hypothesis is
dead — exit, don't hope.

## Falsifiable predictions (the algorithm's own hypothesis test)

1. `IC(−z → fwd return | vpin_pct < θ)` > `IC(−z)` > `IC(−z | vpin_pct ≥ θ)` — monotone in the gate.
2. Ungated ALG-1's worst-decile losses concentrate in high-VPIN states (that's *where* the knife
   lives); ALG-3 removes most of that tail.
3. Net (after `load_costs()`): ALG-3 > ALG-1 on the same window, driven by loss removal, not
   trade-count reduction.

If (1) fails on real data, the whole folder's premise is wrong for this market — that is a useful
finding (it would say Hyperliquid's VPIN doesn't separate informed flow at these scales) and it
feeds `research/FINDINGS.md` either way. Note this is exactly a **`conditional_predictability`**
question (PROC-6) with Z = toxicity instead of entropy — when PROC-6 ships, this prediction
becomes one call of that process.

## Contract sketch

```python
@register
class ToxicVwapReversion(MicrostructureAlgorithm):
    """VPIN-gated VWAP mean-reversion. Fade noise-driven deviation; stand aside
    for informed flow. Refs: Easley/LdP/O'Hara 2012; FINDINGS §1 axis-5, §2."""
    def required_columns(self):
        return ["flow_vwap_deviation", "toxic_vpin_50", "raw_spread_bps"]
    def alg_features(self):
        return [AlgorithmFeature("alg_txvr_z",       warmup=W_z),
                AlgorithmFeature("alg_txvr_gate",    warmup=W_p),
                AlgorithmFeature("alg_txvr_signal",  warmup=max(W_z, W_p))]
    # step(): any required NaN → all outputs NaN. run_batch(): vectorized pandas.
```

Params (`config/algorithms.toml [toxic_vwap_reversion]`): `W_z=96`, `W_p=288`, `k_entry=P80`,
`k_exit=P50`, `θ=P70`, `T_max=12` — all percentiles/windows, zero absolute constants.

## Planted test (write first — this one is the folder's Level-1 test)

Synthesize a two-generator process: deviations tagged noise revert with OU half-life h; deviations
tagged informed continue for T bars; a planted VPIN series correlates ρ≈0.8 with the tag. Assert:
gated PnL > 0, ungated PnL indistinguishable from 0 or negative, and per-bucket ICs satisfy
prediction (1). Then break the correlation (ρ=0) and assert the improvement vanishes (null control
— the PROC-12 discipline applied locally).

**Evaluate:** planted → smoke → `nat algorithm evaluate --algorithm toxic_vwap_reversion --symbol
BTC` → `nat oos30` three-way: ALG-1 vs ALG-3 vs 3f (does the gate generalize beyond its own book?).
**Mode:** 5-min bars, taker costs, exactly like the deployables — no maker assumptions anywhere.

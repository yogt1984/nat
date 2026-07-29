# ALG-1 · `vwap_reversion` — fade the oscillation around micro-VWAP (baseline)

**Idea (simplest possible):** price oscillates around rolling volume-weighted fair value; when it
stretches, fade it. This is the standalone version of the signal that already earns its keep as ⅓
of `3f_liquidity`, isolated so its contribution is measurable on its own.

## Mathematical formulation

Let `d_t = flow_vwap_deviation` (price − rolling VWAP, precomputed in the ingestor). Standardize
over a rolling window `W`:

```
z_t = (d_t − μ_W(d)) / σ_W(d)

signal_t =  −1   if z_t > +k_entry     (price above VWAP → short, expect reversion)
            +1   if z_t < −k_entry     (price below VWAP → long)
             0   otherwise; exit when |z_t| < k_exit or after T_max bars
```

Evidence for polarity: scan IC of `flow_vwap_deviation` is **negative** (−0.287/−0.206/−0.188 @1 s
BTC/ETH/SOL) — high deviation predicts *down* forward return ⇒ mean-reversion, fade is correct.
Spannung localizes the dynamics: OU half-life 5–7 s at tick scale, dominant coherence ~68 s.

## Two operating points — only one is honest for taker execution

| Mode | Bars | Execution | Verdict from the record |
|---|---|---|---|
| **MF (recommended)** | 5-min bars | taker viable | Cost-viable: this is where 3f lives (net + after 1.61 bps RT) |
| Tick (1 s–68 s) | ticks | **maker only** | §2 applies: naive maker IC ≈ 0.03 — do NOT ship without ALG-4's fill model |

The bar-scale mode is the deliverable; the tick mode exists only as an input to ALG-3/ALG-4.

## Contract sketch

```python
@register
class VwapReversion(MicrostructureAlgorithm):
    """Fade z-scored deviation from rolling VWAP. Polarity: high_short.
    Refs: FINDINGS.md §1 axis-5; Spannung spectral (OU t½ 5–7s)."""
    PARAMS via config/algorithms.toml [vwap_reversion]:
        z_window   (default 96 bars)   # μ, σ estimation window
        k_entry    (default P80 abs-z as percentile, not a constant)
        k_exit     (default P50)
        max_hold   (default 12 bars)
    def required_columns(self): return ["flow_vwap_deviation"]
    def alg_features(self):
        return [AlgorithmFeature("alg_vwaprev_z", warmup=z_window),
                AlgorithmFeature("alg_vwaprev_signal", warmup=z_window)]
    # step(): NaN in flow_vwap_deviation → NaN out for both keys.
```

## Planted test (write first)

1. **Recovery:** synthesize OU around a known VWAP path (`d_t = OU(θ, σ)`); assert the algorithm's
   signal has planted-IC > threshold against 1-bar forward reversion, correct sign.
2. **Polarity trap:** feed a pure trend (deviation grows monotonically, never reverts); assert the
   fade signal *loses* — and that losses are bounded by `max_hold` (timeout works).
3. **NaN discipline:** NaN deviation rows → NaN outputs, exactly per contract.

## Failure modes to expect

- Trends: fading a persistent move loses by construction — this is precisely the loss ALG-3's
  toxicity gate should remove (that comparison is the point of building #1 and #3 separately).
- Regime drift: 3-week-old z-window parameters aged IC 0.48 → 0.36 in the Spannung arc — keep
  `z_window` short and re-fit walk-forward, never globally.

**Evaluate:** planted → `nat algorithm evaluate --algorithm vwap_reversion --symbol BTC` →
`nat oos30` vs the 3f baseline (it must add information *beyond* 3f or it stays a component).

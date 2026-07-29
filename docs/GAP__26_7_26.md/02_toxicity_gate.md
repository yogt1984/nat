# ALG-2 · `toxicity_gate` — the shared VPIN veto (TASKS.md **HF4**)

**Idea (simplest correct use of toxicity):** never a strategy — a *permission bit*. Multiply any
base signal by `1[toxicity low]`. The record shows toxicity has zero directional IC and that using
it as a signal fails (`vpin_regime` −7,331 bps OOS); its information is *"informed traders are
active now"* — which is exactly when a reversion/maker strategy should stand down.

## Formulation

```
vpin_pct_t = rolling percentile rank of toxic_vpin_50 over window W_p

g_t = 1[ vpin_pct_t < θ ]                      # base gate
g_t = g_t · 1[ toxic_vpin_roc ≤ 0 or vpin_pct_t < θ_strict ]   # optional: veto rising toxicity

gated_signal_t = base_signal_t · g_t
```

- `θ` is a **percentile parameter** in `config/algorithms.toml [toxicity_gate]` (default P70),
  never an absolute VPIN constant — VPIN levels are not comparable across symbols/regimes.
- Optional refinement, same shape: add `toxic_flow_imbalance_abs` and `toxic_effective_spread`
  percentiles as an OR-veto (a 3-condition gate is still "simplest"; anything more is ALG-4).

## Why a gate and not a signal — the mechanism

VPIN estimates the *probability the counterparty is informed*. Direction-neutral by construction
(informed traders buy or sell). But every strategy that provides liquidity or fades price — ALG-1,
`3f_liquidity`, any maker logic — loses specifically to informed flow. Removing those periods
should cut the left tail without touching the right one. Precedent inside NAT: the `ent_book_shape`
gate lifts imbalance IC +22 % in the calm quintile — same logic, different condition variable.

## Contract sketch

Implemented as a **transform-style algorithm** so every consumer shares one definition:

```python
@register
class ToxicityGate(MicrostructureAlgorithm):
    """Shared VPIN permission gate (HF4). Emits the gate, not a trade signal."""
    def required_columns(self): return ["toxic_vpin_50", "toxic_vpin_roc"]
    def alg_features(self):
        return [AlgorithmFeature("alg_tox_gate", warmup=W_p),        # 0/1
                AlgorithmFeature("alg_tox_vpin_pct", warmup=W_p)]    # continuous, for sizing
```

Consumers (`vwap_reversion`, `3f_liquidity`, ALG-3/4, the ensemble) read `alg_tox_gate` as an
input column — one gate, audited once, reused everywhere (the reason HF4 is "shared").

## Planted test (write first)

1. **Selective-removal:** synthesize a base signal profitable when planted toxicity is low and
   loss-making when high; assert gated PnL > ungated PnL and gated IC ≳ 2× ungated on the mix.
2. **Null-neutrality:** if the planted toxicity is independent of PnL, the gate must not *improve*
   results beyond chance (guards against the gate laundering multiple-testing luck).
3. **Percentile sanity:** constant VPIN input → gate ≡ 1 (P-rank degenerate), no NaN leakage.

## Success criterion (falsifiable, on real data)

For each wrapped strategy: net Sharpe(gated) > net Sharpe(ungated) **and** the improvement comes
from the loss tail (worst-decile trade PnL improves), not from fewer good trades. If the gate only
reduces trade count without reshaping the tail, it is noise — reject.

**Evaluate:** planted → wrap ALG-1 → `nat oos30` A/B (gated vs ungated), same window, same costs.

## Lineage — this is HF4 (shared)

One definition, audited once, reused everywhere: `03_toxic_vwap_reversion` (direction + permission),
`04_microprice_maker_sim` (pull quotes on toxicity), and `07_a4_queue_value` (the adverse-selection
term) all consume `alg_tox_gate`. Threshold `θ` stays a config percentile — never an absolute VPIN
level (VPIN is not comparable across symbols/regimes).

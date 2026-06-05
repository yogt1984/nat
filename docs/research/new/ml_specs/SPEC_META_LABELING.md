# Specification: Meta-Labeling Precision Filter (#3)

Predicts P(trade success) from non-directional market state features. Acts as a precision filter on base algorithm signals — does not predict direction, only whether conditions favor profitable execution. Trained on triple-barrier labels from the 5 winner base algorithms.

---

## Thesis

Base algorithms (jump_detector, 3f_liquidity, optimal_entry, funding_reversion, surprise_signal) generate directional signals. Not all signals are equally profitable — market conditions (entropy, toxicity, spread) modulate success rates. Meta-labeling learns which conditions favor profitable trades, filtering out low-probability signals.

This is a **precision filter**, not a directional model. It answers: "Given that a base signal fired, is NOW a good time to act?"

---

## Triple-Barrier Labeling (De Prado)

For each bar `t` where at least one base signal magnitude > 0.01:

```
entry_price = midprice_mean[t]
upper_barrier = entry * (1 + profit_target_bps / 10000)
lower_barrier = entry * (1 - stop_loss_bps / 10000)
time_barrier = t + max_holding_bars

label = 1   if upper_barrier hit first
      = 0   if lower_barrier hit first
      = 1{price[time_barrier] > entry}  if neither hit
```

Default parameters: `profit_target_bps=5.0`, `stop_loss_bps=10.0`, `max_holding_bars=100`.

---

## 10 State Features (Non-Directional)

| # | Column | Description | Why |
|---|--------|-------------|-----|
| 1 | `ent_tick_1m_mean` | Tick entropy | Market complexity |
| 2 | `ent_rate_of_change_5s_mean` | Entropy change rate | Regime transition speed |
| 3 | `toxic_vpin_10_mean` | Short-window VPIN | Immediate toxicity |
| 4 | `toxic_index_mean` | Composite toxicity | Adverse selection risk |
| 5 | `conc_hhi_last` | HHI concentration | Order flow concentration |
| 6 | `whale_directional_agreement_last` | Whale consensus | Smart money alignment |
| 7 | `vol_returns_5m_mean` | Return volatility | Opportunity vs risk |
| 8 | `vol_ratio_short_long_last` | Vol regime ratio | Volatility regime change |
| 9 | `regime_clarity_last` | Regime clarity | Confidence in current state |
| 10 | `raw_spread_bps_mean` | Spread in bps | Execution cost proxy |

All features are non-directional (sign-invariant or symmetric). This prevents the meta-label model from learning direction, which would conflict with base signal direction.

---

## Anti-Leakage Protocol

**Purged K-Fold Cross-Validation:**

```
For fold k with test set T_k:
  embargo_zone = {i : |i - t| <= embargo for any t in T_k}
  train_set = all indices NOT in embargo_zone
```

Default: K=5, embargo=100 bars. Each test bar is separated from all train bars by at least 100 bars on both sides. This prevents temporal leakage where overlapping triple-barrier windows contaminate train/test boundaries.

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `model_path` | `models/meta_labeling` | `[meta_labeling].model_path` | Model directory |
| `meta_threshold` | 0.55 | `[meta_labeling].meta_threshold` | Min P(success) for sizing |
| `profit_target_bps` | 5.0 | `[meta_labeling].profit_target_bps` | Upper barrier |
| `stop_loss_bps` | 10.0 | `[meta_labeling].stop_loss_bps` | Lower barrier |
| `max_holding_bars` | 100 | `[meta_labeling].max_holding_bars` | Time barrier |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_meta_probability` | [0, 1] | 50 | P(trade success) |
| `alg_meta_side` | {-1, 0, 1} | 50 | Direction from strongest base signal |
| `alg_meta_size` | [0, 1] | 50 | Position size = P * gate |

---

## References

- De Prado, M.L. (2018). *Advances in Financial Machine Learning*, Chapter 3: Meta-Labeling.
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*, Chapter 7: Cross-Validation in Finance.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/meta_labeling.py` |
| Preprocessing | `scripts/build_meta_training_data.py` |
| Training | `scripts/train_meta_labeling.py` |
| Config | `config/algorithms.toml` -> `[meta_labeling]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` -> `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` -> `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_meta_labeling_unit.py` |

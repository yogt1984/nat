# Specification: Regime State Machine (#4)

Classifies market microstructure into 6 discrete regimes using feature-based scoring with manual thresholds. Acts as a gating mechanism for downstream algorithms: momentum in trending regimes, mean-reversion in ranging, no trade in volatile noise.

---

## 6-State Table

| State | Name | Conditions | Trading Rule |
|-------|------|-----------|-------------|
| 0 | ACCUMULATION | accum_score > 0.70, whale > 0, ent < 0.60 | LONG bias |
| 1 | DISTRIBUTION | distrib_score > 0.70, whale < 0, ent < 0.60 | SHORT bias |
| 2 | TRENDING_UP | hurst > 0.55, whale > 0, ent < 0.30 | Momentum continuation (long) |
| 3 | TRENDING_DOWN | hurst > 0.55, whale < 0, ent < 0.30 | Momentum continuation (short) |
| 4 | RANGING | ent > 0.60, hurst < 0.45, \|whale\| < 500 | Mean reversion |
| 5 | VOLATILE_NOISE | vol > 2x median, vpin > 0.80, ent > 0.70 | NO TRADE |

---

## Scoring Formula

For each state s with conditions C_s:

```
score_s(t) = sum_{c in C_s} indicator(c is met)
state(t) = argmax_s { score_s(t) }
confidence(t) = max(scores) / (sum(scores) + epsilon)
```

Tie-breaking: highest state index wins (VOLATILE_NOISE preferred — conservative).

---

## Hysteresis

State is held for `min_duration` bars before a transition is allowed. This prevents rapid flickering between states on noisy data. When a transition occurs, `regime_age` resets to 0.

**Transition risk** decays exponentially with regime age:

```
transition_risk(t) = exp(-regime_age / 10)
```

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `accum_thresh` | 0.70 | `[regime_state_machine].accum_thresh` | Accumulation score threshold |
| `distrib_thresh` | 0.70 | `[regime_state_machine].distrib_thresh` | Distribution score threshold |
| `entropy_low` | 0.30 | `[regime_state_machine].entropy_low` | Low entropy boundary (trending) |
| `entropy_high` | 0.60 | `[regime_state_machine].entropy_high` | High entropy boundary (ranging) |
| `hurst_trend` | 0.55 | `[regime_state_machine].hurst_trend` | Hurst threshold for persistence |
| `vol_noise_mult` | 2.0 | `[regime_state_machine].vol_noise_mult` | Volatility spike multiplier |
| `min_duration` | 5 | `[regime_state_machine].min_duration` | Minimum bars per regime |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_rsm_regime` | {0..5} | 20 | Discrete regime label |
| `alg_rsm_confidence` | [0, 1] | 20 | Score confidence |
| `alg_rsm_transition_risk` | [0, 1] | 20 | Risk of regime change (decays with age) |
| `alg_rsm_trade_allowed` | {0, 1} | 0 | 0 in VOLATILE_NOISE, 1 otherwise |

---

## Input Features

| Column | Description |
|--------|-------------|
| `vol_returns_5m_last` | 5-minute return volatility |
| `trend_hurst_300_mean` | Hurst exponent (persistence measure) |
| `ent_tick_1m_mean` | 1-minute tick entropy |
| `whale_net_flow_4h_sum` | Net large-order flow (4h sum) |
| `toxic_vpin_50_mean` | Volume-synchronized PIN |
| `regime_accumulation_score_mean` | Accumulation regime score |

---

## References

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Wyckoff, R.D. (1931). *The Richard D. Wyckoff Method of Trading and Investing in Stocks*.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/regime_state_machine.py` |
| Config | `config/algorithms.toml` → `[regime_state_machine]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` → `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` → `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_regime_sm_unit.py` |
| Integration tests | `scripts/algorithms/tests/test_regime_sm_integration.py` |

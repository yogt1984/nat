# EAMM — Entropy-Adaptive Market Making

## Name

**EAMM** (Entropy-Adaptive Market Maker): a market making system that dynamically adjusts quoting parameters as a function of the information regime, estimated via entropy of the order flow.

---

## 1. Mathematical Framework

### 1.1 The Market Maker's Problem

At each discrete time t, a market maker must choose an action:

```
a(t) = (delta(t), psi(t), phi(t))
```

where:
- **delta(t) in R+** : half-spread (distance from midprice to quote, in bps)
- **psi(t) in R+** : quote depth (quantity to post at bid and ask)
- **phi(t) in R** : skew (asymmetric offset, positive = ask further from mid)

The quotes are:

```
P_bid(t) = P_mid(t) - delta(t) + phi(t)
P_ask(t) = P_mid(t) + delta(t) + phi(t)
```

### 1.2 Fill Model

A quote is filled when a trade crosses it. Over the next h bars:

```
F_bid(t) = 1{ min_{s in [t, t+h]} P_trade(s)  <=  P_bid(t) }
F_ask(t) = 1{ max_{s in [t, t+h]} P_trade(s)  >=  P_ask(t) }
```

where P_trade(s) is the trade price at time s, and h is the holding horizon (quote lifetime).

### 1.3 PnL per Quote Cycle

The realized PnL of one quote cycle starting at t:

```
PnL(t) = F_bid(t) * F_ask(t) * 2 * delta(t)                              [round trip]
        + F_bid(t) * (1 - F_ask(t)) * (P_mid(t+h) - P_bid(t))            [bid fill only]
        + (1 - F_bid(t)) * F_ask(t) * (P_ask(t) - P_mid(t+h))            [ask fill only]
        + (1 - F_bid(t)) * (1 - F_ask(t)) * 0                            [no fill]
        - F_bid(t) * C_maker                                              [maker fee bid]
        - F_ask(t) * C_maker                                              [maker fee ask]
```

where C_maker is the per-side maker fee (0 on Hyperliquid).

Simplifying (with C_maker = 0):

```
PnL(t) = F_bid * F_ask * 2 * delta                                        [spread captured]
        + F_bid * (1 - F_ask) * (P_mid(t+h) - P_mid(t) + delta - phi)     [long inventory mark]
        + (1 - F_bid) * F_ask * (P_mid(t) - P_mid(t+h) + delta + phi)     [short inventory mark]
```

### 1.4 The Entropy-Adaptive Thesis

**Core claim:** The optimal half-spread delta*(t) is a function of the information regime I(t):

```
delta*(t) = f(I(t), sigma(t), q(t))
```

where:
- **I(t)** = information regime, estimated by entropy features
- **sigma(t)** = local volatility
- **q(t)** = current inventory

The theoretical justification comes from Glosten-Milgrom (1985):

```
S* = 2 * mu * (V_high - V_low) / ((1 - mu) + mu)
```

where S* is the optimal spread and mu is the probability of informed trading. Our entropy features estimate mu:

```
mu_hat(t) ~ g(H_tick(t), VPIN(t), H_perm(t))
```

Low entropy → high mu (informed, predictable flow) → widen spread.
High entropy → low mu (noise, random flow) → tighten spread.

### 1.5 Information State Vector

Define the information state:

```
c(t) = [c_1(t), ..., c_19(t)] in R^19
```

| Index | Feature | Category | What it measures |
|-------|---------|----------|-----------------|
| c_1 | H_tick_1s | entropy | Tick direction entropy, 1s |
| c_2 | H_tick_5s | entropy | Tick direction entropy, 5s |
| c_3 | H_tick_30s | entropy | Tick direction entropy, 30s |
| c_4 | H_tick_1m | entropy | Tick direction entropy, 1m |
| c_5 | H_perm_8 | entropy | Permutation entropy, 8 ticks |
| c_6 | H_perm_16 | entropy | Permutation entropy, 16 ticks |
| c_7 | H_perm_32 | entropy | Permutation entropy, 32 ticks |
| c_8 | VPIN_50 | toxicity | Vol-sync prob of informed trading |
| c_9 | toxic_index | toxicity | Composite toxicity score |
| c_10 | adverse_sel | toxicity | Adverse selection component |
| c_11 | sigma_1m | volatility | Realized vol, 60 ticks |
| c_12 | sigma_5m | volatility | Realized vol, 300 ticks |
| c_13 | lambda_flow | flow | Trade arrival intensity |
| c_14 | aggressor_5s | flow | Buy/sell aggressor ratio, 5s |
| c_15 | I_l1 | imbalance | L1 volume imbalance |
| c_16 | I_l5 | imbalance | L5 volume imbalance |
| c_17 | mom_60 | trend | Momentum, 60 ticks |
| c_18 | hurst_300 | trend | Hurst exponent, 300 ticks |
| c_19 | S_bps | raw | Current spread in bps |

### 1.6 Action Space

Discretize the half-spread into K levels:

```
A = {delta_1, delta_2, ..., delta_K}
```

Default levels (in bps):

```
A = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0}    K = 8
```

### 1.7 Supervised Learning Formulation

**Training data construction:**

For each timestamp t in the historical data and each candidate spread delta_k:

```
PnL(t, delta_k) = simulate_mm(P_mid(t), P_trades[t:t+h], delta_k)
```

**Optimal action label:**

```
y(t) = argmax_{k in {1,...,K}}  PnL(t, delta_k)
```

**Classification problem:**

```
Learn:  f: R^19 -> {1, ..., K}
Such that:  f(c(t)) ≈ y(t)
```

**Regression alternative (preferred):**

Instead of classifying into discrete levels, predict the optimal continuous spread:

```
Learn:  f: R^19 -> R+
Minimize:  E[ (f(c(t)) - delta*(t))^2 ]
```

where delta*(t) is estimated by interpolating PnL across candidate spreads:

```
delta*(t) = argmax_{delta in R+}  interpolate({(delta_k, PnL(t, delta_k))}_k)
```

### 1.8 Performance Metric: Entropy-Conditioned Sharpe

The primary evaluation metric is the Sharpe ratio of the EAMM strategy, conditioned on entropy regime:

```
Sharpe_k = E[PnL | regime = k] / std(PnL | regime = k) * sqrt(N_k)
```

The overall Sharpe is:

```
Sharpe_EAMM = E[PnL_EAMM] / std(PnL_EAMM) * sqrt(N / periods_per_year)
```

**Benchmark:** Fixed-spread market maker with delta = median(delta*(t)).

**Success criterion:** Sharpe_EAMM > Sharpe_fixed at 95% confidence via paired bootstrap test.

### 1.9 Inventory Penalty (Avellaneda-Stoikov Extension)

When the model runs live, inventory modifies the optimal spread:

```
delta_live(t) = f(c(t)) + gamma * q(t)^2 * sigma(t)^2
phi_live(t)   = -gamma * q(t) * sigma(t)^2
```

where:
- q(t) = current inventory (positive = long)
- gamma = risk aversion parameter (tuned from backtest)
- sigma(t) = vol_returns_1m

This penalizes holding inventory by widening spread and skewing quotes away from the inventory direction.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      OFFLINE (Python)                    │
│                                                          │
│  NAT Parquet Data ──→ MM Simulator ──→ Training Labels   │
│                            │                             │
│                            v                             │
│                      LightGBM/XGBoost                    │
│                            │                             │
│                            v                             │
│                   Trained Model f(c) -> delta*           │
│                            │                             │
│                            v                             │
│                   Walk-Forward Validation                 │
│                            │                             │
│                            v                             │
│                   Parameter Table Export                  │
│                   (regime -> theta)                       │
└────────────────────────────┬────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────┐
│                     ONLINE (Rust + FPGA)                 │
│                                                          │
│  Market Data ──→ NAT Feature Engine ──→ c(t) vector      │
│                                           │              │
│                        ┌──────────────────┘              │
│                        v                                 │
│                  Model Inference ──→ delta*(t), phi*(t)   │
│                        │                                 │
│                        v (PCIe MMIO)                     │
│                  FPGA Quote Engine ──→ Exchange           │
│                        │                                 │
│                        v                                 │
│                  PnL Tracker ──→ Online Parameter Update  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

| Module | Language | Input | Output |
|--------|----------|-------|--------|
| `eamm.simulator` | Python | Parquet data + spread levels | PnL matrix (N_rows x K_spreads) |
| `eamm.labels` | Python | PnL matrix | Optimal spread labels y(t) |
| `eamm.features` | Python | Parquet data | Context vectors c(t) |
| `eamm.train` | Python | c(t), y(t) | Trained model |
| `eamm.evaluate` | Python | Model + test data | Walk-forward Sharpe, regime-conditioned metrics |
| `eamm.backtest` | Python | Model + full data | Simulated equity curve, trade log |
| `eamm.export` | Python | Trained model | Parameter table (JSON/binary) for Rust/FPGA |

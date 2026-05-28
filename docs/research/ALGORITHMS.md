# Algorithm Catalogue

**Last updated:** 2026-05-23
**Test conditions:** Walk-forward OOS, 3-day training window, P20/P80 z-score entry, 100min horizon, 1.61 bps RT fee (Binance VIP9)
**OOS window:** 13 dates (2026-05-07 to 2026-05-23)
**Symbols:** BTC, ETH, SOL | **Bars:** 5min from 100ms ticks

---

## Tier 1 — Deployable (net positive all 3 symbols)

### 1. jump_detector — Total +23,199 bps

**Post-jump mean-reversion via Lee-Mykland nonparametric test.**
Detects jumps using bipower variation volatility, then enters mean-reversion after the jump dissipates. Low jump fraction = stable regime = long.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +1.03        | 1.6    | +1,722          | 54%      | -1,981         |
| ETH    | 1,678  | +6.47        | 6.2    | +10,861         | 62%      | -3,412         |
| SOL    | 1,678  | +6.33        | 6.2    | +10,616         | 69%      | -3,311         |

- **Primary feature:** `alg_post_jump_reversion`
- **Signal polarity:** low_long (low jump ratio = stable = long)
- **Source:** `scripts/algorithms/jump_detector.py`

---

### 2. 3f_liquidity — Total +16,028 bps

**Equal-weight z-score composite of spread + depth + VWAP deviation.**
The baseline MF liquidity signal from Experiment Report 1. Strongest on BTC.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 950    | +5.58        | 9.2    | +5,302          | 62%      | —              |
| ETH    | 915    | +7.83        | 7.8    | +7,162          | 62%      | —              |
| SOL    | 954    | +3.74        | 3.2    | +3,564          | 62%      | —              |

- **Primary feature:** z-score of `mf_spread_bps + mf_depth_imbalance + mf_vwap_deviation`
- **Source:** `scripts/alpha/paper_trader.py`

---

### 3. funding_reversion — Total +14,459 bps

**Funding rate z-score mean-reversion with saturation.**
Crypto-native signal. Enters opposite to extreme funding rates using `-sign(z) * min(|z|/z_entry, 3)/3`. Dominates ETH.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +0.26        | 0.4    | +429            | 38%      | -2,327         |
| ETH    | 1,678  | +6.12        | 6.1    | +10,265         | 54%      | -2,629         |
| SOL    | 1,678  | +2.24        | 1.7    | +3,766          | 54%      | -3,786         |

- **Primary feature:** `alg_funding_signal`
- **Signal polarity:** high_long
- **Source:** `scripts/algorithms/funding_reversion.py`

---

### 4. optimal_entry — Total +13,679 bps

**SPRT on Kalman OU-filtered L1 imbalance innovations.**
Sequential hypothesis test for entry timing. Accumulates log-likelihood ratio until decision boundaries are crossed.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +0.90        | 1.1    | +1,504          | 46%      | -2,327         |
| ETH    | 1,678  | +5.89        | 5.2    | +9,877          | 62%      | -3,645         |
| SOL    | 1,678  | +1.37        | 1.0    | +2,298          | 54%      | -4,762         |

- **Primary feature:** `alg_entry_signal`
- **Signal polarity:** high_long
- **Source:** `scripts/algorithms/optimal_entry.py`
- **Known bug:** `run_batch()` hardcodes `sigma_process=0.01` instead of using `self._sigma_process`

---

## Tier 2 — Symbol-specific alpha

### 5. surprise_signal — Total +3,505 bps

**Entropy regime transition detection via ROC z-score.**
Captures microstructure regime shifts through entropy rate-of-change. Strong on SOL, decent on ETH, fails on BTC.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 954    | -4.78        | -8.3   | -4,563          | 15%      | -1,316         |
| ETH    | 1,010  | +2.85        | 3.1    | +2,878          | 54%      | -2,493         |
| SOL    | 981    | +5.29        | 6.7    | +5,190          | 46%      | -518           |

- **Primary feature:** `alg_entropy_surprise`
- **Signal polarity:** low_long (ordering = long, disordering = short)
- **Source:** `scripts/algorithms/surprise_signal.py`

---

## Tier 3 — No edge after costs

All algorithms below are net negative in aggregate across the OOS window.

| # | Algorithm | Total (bps) | BTC Sharpe | ETH Sharpe | SOL Sharpe | Primary Feature |
|---|-----------|-------------|-----------|-----------|-----------|-----------------|
| 6 | oi_divergence | -1,721 | -5.3 | -5.7 | +2.1 | `alg_oi_price_divergence` |
| 7 | regime_gated | -1,748 | -2.4 | -0.4 | -0.0 | `alg_regime_gated_imbalance` |
| 8 | entropy_momentum | -2,600 | -6.4 | -0.2 | -2.7 | `alg_entropy_gated_momentum` |
| 9 | propagator | -4,118 | -2.4 | -1.2 | -3.9 | `alg_transient_impact` |
| 10 | hawkes_intensity | -5,443 | +0.8 | -2.6 | -4.3 | `alg_bid_ask_hawkes_imbalance` |
| 11 | trade_through | -5,739 | -5.1 | -4.2 | +0.2 | `alg_trade_through_imbalance` |
| 12 | weighted_ofi | -6,183 | -4.8 | -0.6 | -3.6 | `alg_weighted_ofi` |
| 14 | switching_ou | -6,230 | -3.5 | +0.7 | -6.0 | `alg_switching_ou_state` |
| 15 | vpin_regime | -7,331 | -4.7 | -1.7 | -3.5 | `alg_vpin_gated_imbalance` |
| 16 | kalman_imbalance | -7,517 | -2.4 | -0.0 | -7.2 | `alg_kalman_signal_strength` |
| 17 | bipower_jump | -32,079 | -14.0 | -9.7 | -9.8 | `alg_jump_ratio` |
| 18 | spread_decomp | -34,510 | -10.7 | -14.8 | -8.4 | `alg_adverse_component` |

All implementations in `scripts/algorithms/*.py`. Generic paper trader: `scripts/alpha/paper_trader_generic.py`.

---

## Portfolio Notes

- **Complementarity:** 3f_liquidity dominates BTC (Sharpe 9.2), jump_detector dominates ETH/SOL (Sharpe 6.2). Near-zero Spearman correlation between jump_detector and funding_reversion — ideal for blending.
- **Full mathematical derivations:** `reports/algo_mathematical_foundations.md`
- **Verification tests:** `scripts/algorithms/tests/test_winning_algos.py` (25 tests, all passing)
- **Detailed results:** `reports/experiment_report_2.md`, `reports/algo_paper_trade_comparison.json`

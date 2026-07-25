# NAT Project — Situation Analysis

**Date**: 2026-06-12
**Scope**: Full audit of `docs/` and `reports/` directories

---

## I. Deployable Algorithms (100min horizon, 1.61 bps RT)

| Algorithm | BTC Sharpe | ETH Sharpe | SOL Sharpe | Total bps | Edge Source |
|-----------|-----------|-----------|-----------|-----------|-------------|
| jump_detector | 1.6 | 6.2 | 6.2 | +23,199 | Lee-Mykland jump reversion |
| optimal_entry | 1.1 | 5.2 | — | +13,679 | SPRT on Kalman innovations |
| funding_reversion | 0.4 | 6.0 | 1.7 | +14,459 | Funding rate mean-reversion |
| 3f_liquidity | 12.1 | 4.3 | 1.8 | — | Spread+depth+VWAP composite |

**Complementarity**: Signal correlations <0.35 across pairs. 3f dominates BTC, jump/optimal dominate ETH/SOL. Portfolio combination should boost Sharpe.

### Hierarchical Combiner (Preliminary)

3-layer architecture separating directional bias (L1), entry timing (L2), and volatility sizing (L3).

| Symbol | OOS IC | Dir Accuracy | Cost-adj Sharpe | Horizon |
|--------|--------|-------------|-----------------|---------|
| BTC | +0.178 | 55.7% | +1.25 | 5h |
| ETH | +0.248 | 57.6% | +1.71 | 5h |
| SOL | +0.359 | 59.4% | +2.40 | 3.3h |

**Key innovation**: First algorithm that structurally addresses adverse selection — L2 fast signals gated by L1 slow direction. Only 2 days of data; needs 7+ days for statistical confidence.

### Deployable ML Model

- `mean_reversion_detector` (LightGBM): OOS AUC 0.55–0.58 across all symbols, z-score driven, entropy-gated.

### MF Liquidity Signal (Headline Result)

First profitable OOS strategy. Sharpe 12.1 (BTC), 4.3 (ETH) at Binance VIP9 costs. All 6 hypotheses confirmed: directional, long-biased, no time-of-day artifact, 3-feature beats 2, maker-viable.

---

## II. Failed Strategies

| Strategy | Sharpe | Verdict |
|----------|--------|---------|
| surprise_signal (walk-forward) | -6.23 | Positive static IC, negative in walk-forward |
| vol_gated_divergence | -8.52 | Completely fails |
| momentum_continuation (LogReg) | — | Classic overfit, OOS AUC 0.37–0.45 |
| weighted_ofi (rolling) | -14.6 | Consistently negative |
| sweep_taker | -20.5 | Taker costs dominate |
| sweep_maker | -5.7 | Maker side also negative |

**Alpha screen**: 551 features at 1-day horizon, 0 significant after FDR correction.

---

## III. Spannung Research Arc (Phases A–F)

1. **Grid search** (1,350 combos): Raw L1 imbalance IC=0.45, EWM smoothing hurts
2. **Causality confirmed**: Smooth lag decay, no look-ahead bias
3. **OOS stable**: IC 0.48 (IS) → 0.47 (OOS 24h) → 0.45 (OOS 48h) → 0.36 (3 weeks old)
4. **Cost reality**: Unprofitable at taker fees (0.17–0.37 bps edge vs 7 bps RT)
5. **Spectral breakthrough**: IC=0.45 entirely in ultra-low band (0.005–0.1 Hz, periods 10–200s). OU half-life 5–7s, dominant coherence at 68s
6. **Regime gate**: `ent_book_shape` lifts IC 0.45 → 0.55–0.67 (replicates cross-symbol)

**Viable path**: Kalman filter for ultra-low band + ent_book_shape gating + zero-fee pairs + market-making.

---

## IV. Cluster & Regime Analysis

- 54 vector-timeframe combos tested, 22 pass Q1 (clusters exist), 6 pass Q1+Q2 (stable)
- Best: entropy@5min (k=2, sil=0.436) and orderflow@5min/15min
- **Q3 predictive power**: Only orderflow@5min passes (KW p=3e-6, weak effect eta²=0.057)
- Verdict: Clusters exist geometrically but have minimal predictive power for returns

---

## V. Data Quality (Korrektur Tasks)

| Issue | Severity | Status |
|-------|----------|--------|
| K1: Docker volume mount bug | Critical | FIXED |
| K2: 56 dead features (23.5% NaN) | Moderate | OPEN — missing whale/liquidation/concentration |
| K3: regime_accumulation constant | Low | OPEN |
| K4: WebSocket gaps 10–12/hr | Low | MONITORING |
| K5: 6-day data gap (Jun 4–10) | Moderate | FIXED (watchdog) |
| K6: Historical gaps ~17 days | Low | ACCEPTED |

---

## VI. Architecture Debt

| Phase | Scope | Effort | Priority |
|-------|-------|--------|----------|
| Arch-p.1 | SQLite research store, atomic writes, schema versioning, symbol config, contract tests | ~25h | Immediate |
| Arch-p.2 | Unified Postgres data plane, event bus | Deferred | Trigger: >10 agents |
| Arch-p.3 | pyproject.toml, script reorg, split base.py, unify cost model, provenance | ~12h | Near-term |

---

## VII. Unimplemented Specifications

1. **EAMM** — Entropy-adaptive market making with 19-feature state vector and discrete spread levels
2. **Liquidation Heatmap Model** — 4-channel liquidation density tensor, 8 spatial features, logistic regression
3. **Website Spec** — Autonomous paper-reading → strategy-coding → backtesting → paper-trading platform
4. **Microstructure Agent** — 7-layer system with 5 hypothesis generators
5. **Cloud Agent** — Nightly Claude-powered research cycles with Telegram alerts

---

## VIII. Key Metrics

- **Features computed**: 191 (123 active, 56 dead/NaN, 12 constant)
- **Algorithms tested**: 17 in sweep, 4 deployable + 1 preliminary (hierarchical)
- **Data resolution**: 100ms, 3 symbols (BTC/ETH/SOL)
- **OOS dates**: 10–25 depending on analysis (need 30+)
- **Full IC scan**: 207 features; raw_spread_bps strongest at 15m (IC=0.139)

---

## IX. Open Questions

1. Can Kalman-filtered ultra-low band imbalance become tradeable on zero-fee pairs?
2. Will `ent_book_shape` gating sustain 20–45% IC lift on 4+ weeks of new data?
3. What is the optimal market-making refresh rate given 5–7s OU half-life?
4. Can whale/concentration features (K2) be populated via Hyperliquid position API?
5. Will mean_reversion_detector hold up on 30+ days of data?
6. Does the hierarchical combiner's L2/L3 add value over L1 alone (ablation pending)?

# PhD Way vs Quant Way — Tasks & Milestones

**Date**: 2026-06-12
**Scope**: Strategic roadmap for both paths, derived from existing documentation

---

## I. PhD Way — Academic Research Path

**Goal**: PhD at ETH Zürich or EPFL in spectral microstructure / quantitative finance
**Hard deadlines**: EPFL Jan 15 / Mar 31 2027 (Sep 2027 entry). ETH rolling.

### Phase P1 — Preprint Completion (Jun–Aug 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| P1.1 | Finalize SVD convolver preprint | LaTeX PDF with 6 significant kernels, turtle-soup/trap IC [0.20, 0.36], BH-FDR correction | Existing data | NOT STARTED |
| P1.2 | Add spectral microstructure section (Spannung Phase D results) | Frequency-localized IC, OU dynamics (H=0.43, τ½=5–7s), 68s coherence | P1.1 | NOT STARTED |
| P1.3 | Add regime-gated section (Phase E results) | ent_book_shape IC lift 0.45→0.55–0.67, Pareto frontier | P1.2 | NOT STARTED |
| P1.4 | Add cross-symbol validation (Phase F results) | Walk-forward across BTC/ETH/SOL, liquidity ordering | P1.3 | NOT STARTED |
| P1.5 | Implement Arch-p.3 provenance tracking | git_sha + data fingerprint for PhD-grade reproducibility | — | NOT STARTED |

**Milestone P1**: Camera-ready preprint with reproducible results. **Target: Aug 2026.**

### Phase P2 — Publication (Aug–Sep 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| P2.1 | Upload to SSRN | Permanent URL (ssrn.com/abstract=XXXXXXX) | P1 complete | NOT STARTED |
| P2.2 | Submit to SSRN e-journals | Capital Markets: Market Microstructure + Financial Engineering | P2.1 | NOT STARTED |
| P2.3 | Obtain arXiv endorsement | Endorsement in q-fin.TR or q-fin.ST | P2.1 + professor contact | NOT STARTED |
| P2.4 | Cross-post to arXiv | arXiv listing under q-fin.TR | P2.3 | NOT STARTED |

**Milestone P2**: Paper live on SSRN + arXiv. **Target: Sep 2026.**

### Phase P3 — Professor Outreach (Sep–Nov 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| P3.1 | Customize emails for Tier 1 targets | 5 personalized emails (Teichmann, Collin-Dufresne, Hugonnier, Malamud, Bölcskei) | P2.1 | NOT STARTED |
| P3.2 | Send Tier 1 emails with SSRN link + PDF | Sent emails, responses tracked | P3.1 | NOT STARTED |
| P3.3 | Follow up with Tier 2 targets | Emails to Acciaio, Cheridito, Bühlmann, Leippold, Zdeborova, Krzakala, Filipovic, Kuhn | P3.2 (stagger 2 weeks) | NOT STARTED |
| P3.4 | Explore co-supervision pairings | Pitch joint framing to: Teichmann+Bölcskei, Collin-Dufresne+Krzakala, Malamud+Zdeborova | Positive responses | NOT STARTED |
| P3.5 | Schedule video calls / visits | Meeting with interested professors | P3.2/P3.3 | NOT STARTED |

**Milestone P3**: At least 2 professors with active interest. **Target: Nov 2026.**

### Phase P4 — Application (Nov 2026 – Mar 2027)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| P4.1 | Prepare formal application package | CV, research statement, transcripts, references | P3 | NOT STARTED |
| P4.2 | Submit EPFL EDFI Round 1 | Application to doctoral program | P4.1, deadline Jan 15 2027 | NOT STARTED |
| P4.3 | Submit ETH application | Application (rolling, but aim Jan–Feb 2027) | P4.1 | NOT STARTED |
| P4.4 | Submit EPFL EDFI Round 2 (backup) | Application if Round 1 unsuccessful | P4.1, deadline Mar 31 2027 | NOT STARTED |
| P4.5 | Interviews and admission decisions | Offer letter | P4.2/P4.3 | NOT STARTED |

**Milestone P4**: PhD offer accepted. **Target: Apr–May 2027.**

---

## II. Quant Way — Production Trading Path

**Goal**: Live trading with validated algorithms on Hyperliquid perps
**Hard constraint**: No live capital until paper trading passes G8 (14+ days)

### Phase Q1 — Foundation & Data Quality (Jun–Jul 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| Q1.1 | Fix K2: populate 56 dead features (whale/liquidation/concentration) | NaN rate < 5% on all 191 features | nan_wiring docs | IN PROGRESS |
| Q1.2 | Accumulate 7+ days continuous data | 2000+ bars per symbol for hierarchical combiner revalidation | K1 fixed, watchdog running | IN PROGRESS (target Jun 17) |
| Q1.3 | Implement Arch-p.1: SQLite research store | Atomic writes, schema versioning, contract tests (~25h) | — | NOT STARTED |
| Q1.4 | Implement Arch-p.3: pyproject.toml, cost model unification | Single cost source of truth, provenance tracking (~12h) | — | NOT STARTED |
| Q1.5 | Clear K3: fix regime_accumulation constant feature | Non-constant regime signal | K2 investigation | OPEN |
| Q1.6 | Monitor K4: WebSocket gap rate | Confirm gaps < 5/hr sustained | Watchdog | MONITORING |

**Milestone Q1**: Clean data pipeline, all features live, Arch-p.1 complete. **Target: Jul 2026.**

### Phase Q2 — Validation & Signal Strengthening (Jul–Aug 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| Q2.1 | Revalidate hierarchical combiner on 7+ days | OOS IC, Sharpe with proper walk-forward folds | Q1.2 (7 days data) | NOT STARTED |
| Q2.2 | Ablation study: L1 alone vs full hierarchical | Quantify L2/L3 marginal contribution | Q2.1 | NOT STARTED |
| Q2.3 | Run ROADMAP Step 1: full alpha screen with FDR | `alpha_screen.json` — ranked features with IC, adjusted p-values | Q1.1 (clean features) | NOT STARTED |
| Q2.4 | Run ROADMAP Step 2–3: feature combination + cost-aware sizing | Combined signal with trade filter, G2/G3 gates | Q2.3 pass G1 | NOT STARTED |
| Q2.5 | Run ROADMAP Step 4: walk-forward validation + deflated Sharpe | OOS Sharpe, OOS/IS ratio, deflated Sharpe p-value | Q2.4 pass G2/G3 | NOT STARTED |
| Q2.6 | Test Spannung Kalman filter on zero-fee pairs | Filtered ultra-low band signal with live cost model | Q1.2 | NOT STARTED |
| Q2.7 | Cross-validate horizons (30min, 1h, 2h, 5h, 12h) per symbol | IC-maximizing horizon map | Q2.1 | NOT STARTED |
| Q2.8 | Evaluate portfolio combination of 4 deployable algorithms | Risk-parity weighted portfolio Sharpe | Q1.1 | NOT STARTED |

**Milestone Q2**: Algorithms revalidated on 30+ days, pipeline gates G1–G4 passed. **Target: Aug 2026.**

### Phase Q3 — Paper Trading (Aug–Sep 2026)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| Q3.1 | Deploy paper trader for top algorithms | `data/paper_trades/YYYY-MM-DD.json` logs | Q2.5 pass G4 | NOT STARTED |
| Q3.2 | Run ROADMAP Steps 5–6: regime conditioning + multi-freq integration | Regime-conditioned signal, macro filter (G5/G6) | Q2.5, cluster validation | NOT STARTED |
| Q3.3 | Run ROADMAP Step 7: portfolio assembly | Risk-parity weights, drawdown controls (G7) | Q3.2 | NOT STARTED |
| Q3.4 | 14-day paper trading with daily reconciliation | Paper Sharpe, IC decay monitoring, infra stability | Q3.1 | NOT STARTED |
| Q3.5 | Measure actual conditional IC on fills | Fill-conditional IC vs assumed costs | Q3.4 (fill logs) | NOT STARTED |
| Q3.6 | Build kill switch infrastructure | Auto-halt on daily >1%, weekly >2%, monthly >5% drawdown | Q3.1 | NOT STARTED |

**Milestone Q3**: 14 days paper trading, G8 pass, fill data validates cost model. **Target: Sep 2026.**

### Phase Q4 — Live Deployment (Oct 2026 →)

| # | Task | Deliverable | Depends On | Status |
|---|------|-------------|------------|--------|
| Q4.1 | Week 1–2: deploy at 1% capital, maker orders only | Live P&L, execution quality metrics | Q3.4 pass G8 | NOT STARTED |
| Q4.2 | Week 3–4: scale to 5% if paper match holds | Confirmed backtest-to-live ratio | Q4.1 success | NOT STARTED |
| Q4.3 | Month 2–3: scale to 10% | Sustained Sharpe, no kill switch triggers | Q4.2 success | NOT STARTED |
| Q4.4 | Month 4+: scale to 25% max | Full deployment, ongoing monitoring | Q4.3 success | NOT STARTED |
| Q4.5 | Continuous IC decay monitoring | Rolling 7-day IC vs training IC, alert on 50% decay | Q4.1 | NOT STARTED |
| Q4.6 | Monthly pipeline re-run from Step 1 | Detect feature decay, retrain if needed | Q4.1 | NOT STARTED |

**Milestone Q4**: Profitable live trading at 5%+ capital. **Target: Nov 2026.**

---

## III. Synergies — Where the Paths Reinforce Each Other

| PhD Task | Quant Task | Shared Output |
|----------|------------|---------------|
| P1.5 Arch-p.3 provenance | Q1.4 Arch-p.3 cost unification | Same work — reproducibility serves both publication and production |
| P1.2–P1.4 Spannung sections | Q2.6 Kalman filter on zero-fee pairs | Spectral findings inform both the paper and the trading strategy |
| P1.1 SVD convolver preprint | Q2.3 Alpha screen with FDR | FDR methodology is identical — academic rigor = production rigor |
| P3.4 Co-supervision pitches | Q2.8 Portfolio combination | Cross-asset information flow is both a research question and a trading edge |
| P4.1 Research statement | Q3.5 Conditional fill analysis | Live fill data strengthens the academic narrative ("I deployed this") |

**Key insight**: The paths are not competing for resources — they compete for *time allocation*. The infrastructure, data, and methodology are shared. The divergence is in how results are packaged: LaTeX vs live capital.

---

## IV. Decision Points

### D1: After Phase Q2 (Aug 2026) — "Is there a trading business here?"

- **If G1–G4 all pass**: Quant path is viable. Continue to paper trading.
- **If G1 fails** (no features survive FDR): The alpha may not be tradeable at current costs. PhD path becomes primary — the research has academic value even without profitability.
- **Action**: This is not a fork — both paths continue. But capital allocation of *time* shifts.

### D2: After Phase P3 (Nov 2026) — "Do professors want this?"

- **If 2+ professors respond positively**: PhD path is viable. Prepare applications.
- **If zero interest**: Research framing needs work, or the findings aren't novel enough for academia. Double down on quant path.
- **Action**: If both D1 and D2 are positive, pursue both. If only one is positive, prioritize that path.

### D3: After Phase Q3 (Sep 2026) — "Does paper match backtest?"

- **If paper Sharpe within 2x of backtest**: Deploy live.
- **If paper is much worse**: Execution model is wrong. Pause live deployment, investigate adverse selection. This data is still useful for the PhD (Section P1 can include "why backtests overstate real performance").
- **Action**: Negative paper trading results strengthen the academic case for adverse selection research.

### D4: PhD Offer (Apr–May 2027) — "Take it or not?"

- **If live trading is profitable at 10%+ capital**: PhD becomes optional or part-time.
- **If live trading is marginal or failed**: PhD is the higher-value path.
- **If both work**: ETH/EPFL PhDs are compatible with independent trading — this is not a binary choice if the professor allows it.

---

## V. Critical Path — What to Do First (Jun 2026)

Regardless of which path dominates long-term, the next 4 weeks have a clear priority order:

| Priority | Task | Why | Effort |
|----------|------|-----|--------|
| 1 | Q1.2 — Accumulate 7+ days clean data | Blocks both hierarchical revalidation (quant) and stronger results section (PhD) | Passive (running) |
| 2 | Q1.1 — Fix K2 dead features | 56 dead features weaken both the trading signal base and academic reproducibility | ~8h |
| 3 | Q1.3 — Arch-p.1 SQLite research store | Blocks systematic alpha screening (Q2.3) and provides research provenance (P1.5) | ~25h |
| 4 | P1.1 — Start preprint draft | Long lead time to publication; can be written while data accumulates | ~40h writing |
| 5 | Q2.1 — Revalidate hierarchical combiner | First real test of the most promising new algorithm; informs both paths | ~4h (after data) |

**The single highest-leverage action right now is letting the data pipeline run undisturbed while starting the preprint draft.** Both paths need more data, and both paths need the preprint.

---

## VI. Timeline Overview

```
Jun 2026  ├── Q1: Foundation (K2 fix, data accumulation, Arch-p.1)
          ├── P1: Start preprint draft
          │
Jul 2026  ├── Q2: Validation (hierarchical revalidation, alpha screen, ablation)
          ├── P1: Complete preprint (spectral + regime + cross-symbol sections)
          │
Aug 2026  ├── Q2→Q3: Pipeline gates G1–G4, begin paper trading
          ├── P1→P2: SSRN upload, seek arXiv endorsement
          ├── D1: "Is there a trading business here?"
          │
Sep 2026  ├── Q3: Paper trading (14 days), fill analysis, kill switches
          ├── P2→P3: Tier 1 professor outreach (5 emails)
          │
Oct 2026  ├── Q4: Live at 1% capital (if G8 pass)
          ├── P3: Tier 2 outreach, co-supervision exploration
          │
Nov 2026  ├── Q4: Scale to 5% (if Week 1–2 holds)
          ├── P3→P4: Begin application prep
          ├── D2: "Do professors want this?"
          │
Dec 2026  ├── Q4: Scale to 10%
          ├── P4: Finalize application package
          │
Jan 2027  ├── P4.2: EPFL Round 1 submission (deadline Jan 15)
          ├── P4.3: ETH submission
          │
Mar 2027  ├── P4.4: EPFL Round 2 (deadline Mar 31, if needed)
          │
Apr 2027  ├── D4: PhD offer decision
```

---

## VII. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Alpha decays before live deployment | Quant path fails | Paper trade sooner with existing 4 algorithms while hierarchical accumulates data |
| No professor responds | PhD path stalls | Broaden beyond ETH/EPFL: Imperial, Oxford, TU Delft, KTH all have microstructure groups |
| Data pipeline breaks during accumulation | Both paths delayed | Watchdog already running (K5 fix), monitor K4 gaps |
| Preprint reveals proprietary edge | Quant path weakened | Publish spectral characterization (academic value) but withhold exact trading rules (regime thresholds, entry logic) |
| Arch-p.1 takes longer than 25h | Foundation phase delayed | Can start alpha screen on existing SQLite/Parquet without full research store |
| Live adverse selection worse than modeled | Quant returns disappoint | This is expected — the hierarchical combiner was designed specifically for this. Budget for 2x cost degradation. |

# Algorithm Candidates from the Literature — HF and LF

**Date:** 2026-06-12
**Purpose:** Literature-grounded candidates for new algorithms/generators, mapped to NAT's
constraints. Companion to `algorithms_report.md` (what exists) and `plan.md` §3.3 (agent
horizon partition: micro s–min, MF min–1h, macro 1h–24h, daily 1–7d).

---

## 0. Selection Constraints (why most papers don't make this list)

Every candidate below was filtered against NAT reality:

| Constraint | Value | Consequence |
|------------|-------|-------------|
| Costs | Hyperliquid taker ~3.5 bps, maker ~0.2 bps; sweeps validated at 1.61 bps RT | Taker-side HF signals need >4 bps edge per RT — almost nothing survives (sweep_taker: -20.5 Sharpe). Maker-side and longer-horizon designs preferred |
| Data | 100ms book/trade features, 3 symbols, 22 good days | No tick-by-tick event stream (100ms snapshots); deep-learning LOB models (DeepLOB etc.) excluded — insufficient data and wrong data shape |
| K2 | 82/236 features NaN (whale, liquidation, concentration, GMM, heatmap) | Liquidation/whale candidates marked **K2-gated** — design now, run after Q1.2 |
| Failures | weighted_ofi (-14.6), surprise_signal walk-forward (-6.23), momentum_continuation (overfit) | Single-level rolling OFI is dead; entropy *standalone* direction is dead (entropy as *gate* works — Phase E); LogReg on micro features overfits |
| Wins | jump_detector, optimal_entry, funding_reversion, 3f_liquidity; ent_book_shape gating (+22% IC) | New candidates should be complementary (corr <0.35 with these), not variations of them |

**Lesson encoded from the literature + our own failures:** signals that *explain
contemporaneous* price moves (OFI, imbalance) mostly do not *forecast* net of costs at taker
fees. The viable designs use them as (a) maker-side quote placement inputs, (b) gates/filters
on other signals, or (c) inputs at longer aggregation horizons.

---

## I. High-Frequency Candidates (micro agent: seconds–minutes; MF agent: minutes–1h)

### HF1 — Microprice Fair-Value Deviation (Stoikov 2018)

- **Literature:** Stoikov, *The Micro-Price: A High Frequency Estimator of Future Prices*
  ([SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)). The microprice
  is the limit of expected future mid-prices conditional on (imbalance, spread) — a martingale
  estimator, empirically better than mid or weighted-mid for short-term prediction.
- **Mechanism:** Estimate the adjustment function `g(imbalance, spread)` from our own 100ms
  data (Markov chain on discretized imbalance×spread states). Signal = `microprice − mid`.
  When the deviation exceeds a threshold, the mid is stale and will move toward the microprice.
- **NAT inputs:** `imbalance_qty_l1`-family, `raw_spread_bps`, bid/ask L1 sizes — all live (base features).
- **Why it can clear costs where weighted_ofi didn't:** This is not a regression on flow; it's
  a state-conditional expectation calibrated per symbol, used primarily as the **fair-value
  anchor for maker orders** (entry: post bid when microprice >> mid). Edge is captured via the
  maker rebate path (0.2 bps), not taker crossings.
- **Horizon / agent:** 1–60s; micro agent. Natural L2 (entry timing) input for the hierarchical combiner.
- **Effort:** ~6h (estimation script + `MicrostructureAlgorithm` with `alg_microprice_dev`).
- **Verification:** Conditional forward-return curves by deviation decile must be monotone;
  compare IC vs raw `imbalance_qty_l1` (the Spannung baseline, IC 0.45 in-band) — must add, not duplicate.

### HF2 — Integrated Multi-Level OFI (Cont, Cucuringu & Zhang 2023)

- **Literature:** *Cross-Impact of Order Flow Imbalance in Equity Markets*
  ([arXiv 2112.13213](https://arxiv.org/abs/2112.13213), Quantitative Finance 2023). Key
  results: (1) OFI integrated across the top ~10 book levels via PCA explains price impact far
  better than best-level OFI; (2) once multi-level OFI is integrated, **cross-asset impact terms
  add almost nothing** contemporaneously, but (3) cross-asset OFI *does* help in **forecasting**.
- **Why our weighted_ofi failed and this is different:** ours was a single-/fixed-weight,
  rolling-regression forecast — the design the paper shows is dominated. The integrated variant
  (PCA weights re-estimated per regime) plus *forecast* horizon at 1–5 min aggregation is the
  configuration with documented predictive power.
- **NAT inputs:** flow features + multi-level book (we compute L1–L10 depth features); cross-symbol
  variant uses all 3 symbols' OFIs jointly (lasso, per the paper).
- **Horizon / agent:** 1–5 min forecast; MF agent generator (`mf_integrated_ofi`).
- **Cost viability:** marginal as taker; frame as directional *bias* layer (L1-style) gating
  other entries, or maker-side.
- **Effort:** ~8h. **Priority caveat:** highest scientific value is the *post-mortem* — it
  explains a documented failure with a documented fix.

### HF3 — Hawkes Intensity Imbalance (state-dependent)

- **Literature:** state-dependent Hawkes processes for LOBs (Morariu-Patrichi & Pakkanen,
  [Quantitative Finance 2021](https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1983199));
  Hawkes-based crypto LOB forecasting ([arXiv 2312.16190](https://arxiv.org/abs/2312.16190)).
  Excitation magnitude/duration depends on spread and queue-imbalance state; buy/sell intensity
  differential forecasts short-term direction.
- **Mechanism:** We already compute 3 Hawkes trade-intensity features (base category 3.14).
  The candidate: bivariate (buy/sell) Hawkes with state-dependent baseline; signal =
  `(λ_buy − λ_sell) / (λ_buy + λ_sell)` with the *projected* intensity over the next 10–60s,
  gated by `ent_book_shape` regime (our Phase E result).
- **NAT inputs:** trade buffer timestamps/sides (live), spread, imbalance, existing `hawkes_*`.
- **Horizon / agent:** 10–60s; micro agent.
- **Why complementary:** jump_detector reacts *after* a jump; Hawkes intensity ramps detect
  activity clustering *before* — the pair covers both sides of the event.
- **Effort:** ~10h (EM/MLE fit per symbol-day, online intensity update).

### HF4 — VPIN Toxicity Gate + Jump Anticipation (Easley, López de Prado & O'Hara 2012)

- **Literature:** VPIN ([*Flow Toxicity and Liquidity*, RFS 2012](https://www.sciencedirect.com/science/article/abs/pii/S1044028318302679));
  crypto-specific: *Bitcoin wild moves: order flow toxicity and price jumps*
  ([RIBAF 2026](https://www.sciencedirect.com/science/article/pii/S0275531925004192)) — VPIN
  significantly predicts future price jumps in BTC, with persistence in both VPIN and jump size.
- **Mechanism:** Volume-synchronized buy/sell imbalance over volume buckets. **Not a
  directional signal** — a *condition*: high VPIN → widen quotes / suppress mean-reversion
  entries (adverse selection imminent), and arm jump_detector with higher prior.
- **NAT inputs:** toxicity category (10 features, live) already approximates parts of this;
  the candidate formalizes volume-bucketed VPIN and wires it as a **gate** into existing
  deployable algorithms — the same architecture as ent_book_shape gating, which is our single
  best-validated IC amplifier.
- **Horizon / agent:** gate at all horizons; implement as shared filter, hypothesis via micro agent.
- **Effort:** ~5h. Cheap, high prior: it upgrades 4 existing winners rather than adding a 5th sibling.

### HF5 — Inventory-Aware Market Making (Avellaneda–Stoikov + Guéant–Lehalle closed form)

- **Literature:** Avellaneda & Stoikov 2008; Guéant, Lehalle & Fernandez-Tapia closed-form
  solutions; practical syntheses combine the AS reservation price with OFI/microprice inputs
  (e.g. [Hummingbot technical guide](https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/),
  [AS+OFI analysis](https://www.quantlabsnet.com/post/ultra-low-latency-high-frequency-market-making-a-comprehensive-analysis-of-the-avellaneda-stoikov-f)).
- **Mechanism:** Quote bid/ask around a reservation price = microprice (HF1) shifted by
  inventory penalty; spread width from volatility + VPIN (HF4); quote refresh rate matched to
  the measured OU half-life of book imbalance (5–7s, Spannung Phase D); only quote in
  favorable `ent_book_shape` regimes (Phase E).
- **Why this is *the* Spannung endgame:** Phase D concluded the 0.2–0.4 bps in-band edge is
  20× too small for taker fees but viable for maker flow. This candidate is the documented
  "viable path" (situation_analysis §III) assembled from components we have already validated
  — it is less a new algorithm than the integration target for HF1+HF4+Kalman (Q2.6).
- **Horizon / agent:** continuous; separate execution-layer strategy, *not* an agent hypothesis.
  Needs the kill-switch daemon (plan §3.7) before any live quoting.
- **Effort:** ~3 days simulation-first (`emulation before hardware` rule applies: backtest the
  quote engine on recorded 100ms book states before touching the API).

### HF6 — Cross-Symbol Lead-Lag (Hayashi–Yoshida / Huth–Abergel)

- **Literature:** Hayashi & Yoshida 2005 (asynchronous correlation estimator); Huth & Abergel,
  *High-frequency lead/lag relationships*
  ([arXiv 1111.7103](https://arxiv.org/pdf/1111.7103), JEF 2014) — higher-volume assets lead;
  lags up to ~15s documented between BTC venues/assets
  ([NTU study](https://irep.ntu.ac.uk/id/eprint/35549/1/13112_Dao.pdf)).
- **Mechanism:** Estimate the HY cross-correlation function BTC→ETH and BTC→SOL on 100ms
  returns; if the lag peak is stable and >200ms, trade the lagger on the leader's move,
  maker-side on the lagger's book.
- **NAT inputs:** per-symbol mid returns (live). The 3 dead `cross_symbol` features become
  this candidate's outputs rather than blockers (computed in Python first).
- **Horizon / agent:** sub-second to seconds; micro agent (`cross_asset` generator exists as template).
- **Honest prior:** within a single venue (Hyperliquid) for the top-3 pairs, HFT arbitrageurs
  likely keep this near-dead; the cheap experiment is the HY scan itself (~3h) — implement the
  strategy only if a stable >200ms lag survives.

---

## II. Lower-Frequency Candidates (macro agent: 1h–24h; daily agent: 1–7d)

### LF1 — Funding-Settlement Window Effects

- **Literature:** intraday funding-spread seasonality tied to settlement mechanics — systematic
  post-settlement divergence windows documented
  ([MDPI perp microstructure 2026](https://www.mdpi.com/2227-7072/14/5/103)); intraday
  volume/volatility seasonality peaking 16:00–17:00 UTC
  ([tea-time study](https://link.springer.com/article/10.1007/s11156-024-01304-1),
  [arXiv 2109.12142](https://arxiv.org/pdf/2109.12142)).
- **NAT-specific hypothesis:** Hyperliquid pays funding **hourly** (8h premium prorated), while
  Binance/OKX/Bybit settle at fixed 8h marks (00/08/16 UTC). Candidate: test whether (a) HL
  prices show systematic drift into/out of the *other venues'* settlement marks (cross-venue
  arbitrage flow is forced through time), and (b) funding-premium dislocations around those
  marks mean-revert. This extends the already-deployable `funding_reversion` (100min horizon)
  with event-time conditioning instead of competing with it.
- **NAT inputs:** context features (funding rate, mark/oracle premium — live), time-of-day.
- **Horizon / agent:** 1–8h; macro agent generator (`macro/funding_meanrev.py` is the template).
- **Effort:** ~6h. High prior: funding edges are our best-replicated family (H-suite + funding_reversion).

### LF2 — OI-Positioning Extremes (Hong–Yogo logic, crypto evidence)

- **Literature:** open-interest movements predict futures returns (Hong & Yogo 2012, commodities);
  crypto: speculative net-positioning predicts returns
  ([JBEF 2023](https://www.sciencedirect.com/science/article/abs/pii/S2214635023000266));
  long/short OI ratio extremes + elevated funding → reversal probability
  ([BIS WP 1087 crypto carry](https://www.bis.org/publ/work1087.pdf) for the funding-premium structure).
- **Mechanism:** z-score of ΔOI interacted with funding level and price trend: OI expansion +
  extreme funding + extended price = crowded leverage → fade; OI expansion + neutral funding =
  genuine flow → follow. This is the *positioning* complement to LF1's *price-of-leverage* signal.
- **NAT inputs:** `oi`/ΔOI in context features (live), funding, trend features.
- **Horizon / agent:** 4h–48h; macro/daily agents.
- **Effort:** ~6h.

### LF3 — Liquidation-Cascade Reversion (**K2-gated**)

- **Literature:** cascade mechanics and forecastability: *Slippage-at-Risk for perp exchanges*
  ([arXiv 2603.09164](https://arxiv.org/pdf/2603.09164)); Oct 10–11 2025 $19B cascade event study
  ([arXiv 2605.10400](https://arxiv.org/html/2605.10400v1)). Forced liquidations are
  *uninformed* flow → overshoot and revert; cascade onset is predictable from leverage
  clustering near price (our liquidation heatmap features, currently NaN).
- **Mechanism:** two-stage: (1) cascade detection (liquidation intensity spike + depth
  withdrawal + Hawkes activity from HF3), (2) entry on exhaustion (intensity decay) for the
  reversion leg. Our confirmed H3 (cascade prediction) is the in-house evidence.
- **NAT inputs:** liquidation_risk (13) + heatmap (8) features — **dead until Q1.2**; depth/
  resilience features (live) allow a degraded depth-only prototype now.
- **Horizon / agent:** minutes–hours; MF/macro agents. Large, rare edges — complements the
  always-on algorithms exactly where they're weakest (regime breaks).
- **Effort:** ~8h after K2; ~4h for the depth-only prototype.

### LF4 — Volume-Weighted Time-Series Momentum (daily)

- **Literature:** TSM (Moskowitz, Ooi & Pedersen 2012); crypto: daily/weekly TSM significant —
  1σ daily BTC return → +0.33% next day (Liu & Tsyvinski,
  [NBER w24877](https://www.nber.org/system/files/working_papers/w24877/w24877.pdf));
  volume-weighted TSM variant with strong evidence
  ([Huang, Sangiorgi & Urquhart 2024, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389));
  realistic-assumptions check ([Han, Kang & Ryu, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565));
  decay caveat: early-years predictability has weakened
  ([Grobys 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/ijfe.70036)).
- **Mechanism:** sign of trailing 1–7d volume-weighted return, vol-scaled position, skip-day to
  dodge 1-day reversal contamination. The literature is mature enough to pin exact spec.
- **NAT inputs:** daily bars from candle daemon (plan §3.4) + `data/macro/` (365d, unused).
- **Horizon / agent:** 1–7d; this **is** the `daily/momentum.py` generator from plan §3.3 —
  the literature anchor it was missing.
- **Effort:** ~4h on top of candle daemon.

### LF5 — Weekend-Effect Conditioning (daily)

- **Literature:** weekend momentum returns exceed weekday, higher Sharpe, lower drawdowns,
  especially alts ([ACR 2025](https://acr-journal.com/article/the-weekend-effect-in-crypto-momentum-does-momentum-change-when-markets-never-sleep--1514/));
  NYSE-open/closed return decomposition for BTC
  ([QuantPedia summary](https://quantpedia.com/are-there-seasonal-intraday-or-overnight-anomalies-in-bitcoin/)).
- **Mechanism:** not standalone — a *conditioning layer*: LF4 momentum weights ×k on weekends /
  TradFi-closed windows; thinner books (documented liquidity seasonality) also argue for wider
  maker spreads in HF5 on weekends. Cheap to test, mirrors our proven gate architecture.
- **Horizon / agent:** daily agent; ~2h inside LF4.

### LF6 — HAR-RV Volatility Forecasting for Sizing (Corsi 2009)

- **Literature:** HAR-RV (Corsi 2009); for crypto, HAR beats GARCH and often ML for 1-day-ahead
  RV ([arXiv 2511.20105](https://arxiv.org/html/2511.20105v1),
  [arXiv 2404.04962](https://arxiv.org/html/2404.04962v1)); volatility-managed portfolios
  (Moreira & Muir 2017) for the sizing rule.
- **Mechanism:** **not directional** — daily RV forecast from daily/weekly/monthly RV
  components (we have 100ms data → excellent RV estimates). Output feeds: L3 of the
  hierarchical combiner, `meta_portfolio.py` risk-parity weights (plan §3.6), and kill-switch
  context. Honest, near-guaranteed value: vol forecasting works; alpha forecasting mostly doesn't.
- **NAT inputs:** volatility features (live) aggregated daily.
- **Horizon / agent:** daily; infrastructure algorithm (like regime_state_machine), not a signal.
- **Effort:** ~4h.

### LF7 — Conditional Funding Carry (daily, low priority)

- **Literature:** crypto carry ([BIS WP 1087](https://www.bis.org/publ/work1087.pdf);
  [CMU Crypto Carry Trade](https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf)):
  historically Sharpe 6+, **decayed to negative by 2025** as Ethena-style products crowded it.
- **Why still listed:** classic delta-neutral carry needs a spot leg we don't trade — but the
  *conditional* version (harvest funding only when LF2 positioning and LF6 vol forecasts say
  the crowded-unwind risk is low) is exactly what the literature says is left of this trade.
  Directional-lite: hold the side that *receives* funding when expected drift ≈ 0.
- **Horizon / agent:** 1–7d; daily agent. Effort ~6h. Run last.

---

## III. Priority & Sequencing

Scored on: data available now (K2), cost realism, complementarity to the 4 deployed winners,
literature strength, effort.

| Rank | Candidate | Agent | Available now? | Effort | Rationale |
|------|-----------|-------|----------------|--------|-----------|
| 1 | HF4 VPIN gate | shared | Yes | ~5h | Upgrades 4 existing winners; gate architecture already validated (ent_book_shape) |
| 2 | LF1 funding-settlement windows | macro | Yes | ~6h | Best-replicated edge family in-house; clean event-time hypothesis |
| 3 | HF1 microprice deviation | micro | Yes | ~6h | Maker-side economics; feeds HF5 and hierarchical L2 |
| 4 | LF4+LF5 VW-TSM + weekend | daily | After candle daemon | ~6h | The literature anchor for the planned daily generators |
| 5 | LF6 HAR-RV sizing | infra | Yes | ~4h | Near-certain value, no alpha claim; feeds portfolio + kill switch |
| 6 | LF2 OI-positioning | macro/daily | Yes | ~6h | Complements LF1; uses live context features |
| 7 | HF2 integrated OFI | MF | Yes | ~8h | Redeems weighted_ofi with the documented fix; bias layer only |
| 8 | HF3 Hawkes intensity | micro | Yes | ~10h | Pre-event complement to jump_detector |
| 9 | LF3 liquidation cascade | MF/macro | **K2-gated** | ~8h | Highest ceiling, blocked on Q1.2; depth-only prototype possible |
| 10 | HF5 AS market making | execution | Sim-first | ~3d | The Spannung endgame; integrate HF1+HF4+Q2.6 Kalman; needs kill switch |
| 11 | HF6 lead-lag scan | micro | Yes | ~3h scan | Run the cheap HY scan; implement only if lag >200ms survives |
| 12 | LF7 conditional carry | daily | Yes | ~6h | Literature says the naive trade is dead; only the conditional variant |

**Fit with the roadmap:** ranks 1–3 are implementable during the Jun 13–17 accumulation window
(pure Python on existing parquet, zero ingestor contact). Ranks 4–6 align with plan.md Phase 4
(daily agent + candle daemon). Each candidate enters as a **hypothesis through the agent
gates** (IC → cost → temporal/symbol replication → FDR), not as a pre-trusted algorithm —
the 5-gate protocol exists precisely so that listed ≠ deployed.

**For the preprint (P-path):** HF3 (Hawkes), HF5 (OU-driven quote refresh), and LF3 (cascade
microstructure) are the candidates with academic novelty in the crypto-perp setting; positive
*or negative* replications of HF2 (cross-impact in crypto) extend Cont-Cucuringu-Zhang to a
new asset class — publishable either way.

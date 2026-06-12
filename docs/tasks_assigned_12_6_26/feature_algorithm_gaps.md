# Feature & Algorithm Gaps — What's Missing and Worth Adding

**Date:** 2026-06-12
**Derived from:** `algorithm_classification.md` (empty matrix cells), `algorithm_candidates_literature.md` (candidate input requirements), `features_report.md` (what the 236-vector computes)
**Constraint context:** ingestor frozen until Jun 17 (Q1.1 accumulation streak); 236-vector schema changes are expensive (every downstream consumer); K2 fix in progress (position tracker)

---

## 1. Missing Features

### 1.1 Computable in Python today (from existing parquet — zero ingestor risk)

| # | Feature group | What | Unblocks | Effort |
|---|--------------|------|----------|--------|
| F1 | **Settlement-clock** | Time-to-next Binance/OKX 8h funding mark (00/08/16 UTC), hour-of-day sin/cos, weekend flag, cumulative funding 8h/24h | CAND-2 (LF1 funding-settlement), CAND-4b (LF5 weekend) | ~2h — pure timestamp arithmetic |
| F2 | **Microprice deviation** | `microprice − mid` via Stoikov state-conditional expectation over (imbalance, spread) states; all inputs (L1 sizes, `raw_spread_bps`) already live | CAND-3 (HF1); better fair-value anchor than mid for every maker-side design | ~4h incl. per-symbol estimation |
| F3 | **Multi-level OFI** | Differenced book-change *flow* between consecutive 100ms snapshots, per level L1–L10, plus PCA-integrated OFI (Cont–Cucuringu–Zhang) | CAND-7 (HF2). Note: existing `imbalance_*` features are **level snapshots**, not flow — `weighted_ofi`'s failure never tested the integrated variant | ~6h |
| F4 | **HAR-RV components** | Daily/weekly/monthly realized-vol aggregates from 100ms returns | CAND-5 (LF6 sizing); `meta_portfolio.py` risk parity; kill-switch context | ~3h |
| F5 | **Realized higher moments** | Rolling skewness/kurtosis at 5min–1h windows; signed-jump asymmetry | Conditioning for jump_detector; tail features for daily agent | ~2h |

**Design rule:** every new feature is computed in **Python from parquet first**, IC-validated, and only promoted into the Rust ingestor if it earns a place. Schema changes to the feature-vector contract (names_all/count_all/to_vec) propagate to every parquet reader, algorithm, and ML model — promotion must be earned, not default.

### 1.2 Requiring ingestor (Rust) changes — post-Jun 17 only

| # | Feature group | What | Why blocked / notes |
|---|--------------|------|---------------------|
| F6 | **Cross-symbol** | Lead-lag estimates, rolling beta to BTC, relative funding spread. The 3 dead `cross_symbol` columns are **not K2** — they're an architecture gap: symbols run in isolated tokio tasks with no shared state | Biggest *structural* feature gap. Needs a shared `MarketState` read path or a cross-symbol aggregator task; design carefully (the `biased` select loop must not block). Unblocks CAND-11 (HF6) |
| F7 | **Bivariate Hawkes** | Separate `λ_buy`, `λ_sell` intensities + branching ratio (instability/criticality measure) | Current 3 Hawkes features are aggregates. Unblocks CAND-8 (HF3); branching ratio doubles as a regime feature |
| F8 | **K2 completion** | Whale flow, liquidation risk, concentration (82 cols) | Already in progress (position tracker, `nan_wiring/`); listed for completeness — unblocks meta_labeling, cascade_probability at full strength, CAND-9 (LF3) |

### 1.3 New external data source (highest leverage, new infrastructure)

| # | Source | Features enabled | Notes |
|---|--------|------------------|-------|
| F9 | **Binance reference feed** (spot + perp mid for BTC/ETH/SOL) | True cross-venue basis, cross-venue lead-lag, settlement-arb pressure, relative funding | NAT is Hyperliquid-only; an entire feature family is structurally impossible without one external price reference. One WebSocket client following the `ws/client.rs` pattern. Post-Jun 17; run as a separate process first (don't touch the ingestor) |

---

## 2. Missing Algorithms

From `algorithm_classification.md` §4.2 — the empty matrix cells that are *genuine* gaps (not structural), beyond the 12 candidates already specced:

### A1 — Relative-Value Pairs (the missing 9th logic family)

ETH/BTC and SOL/BTC **ratio mean-reversion** at macro/daily horizons: cointegration test on log-ratio, OU half-life estimation, z-score entry on the spread, market-neutral two-leg execution. Nothing in the 26 implemented algorithms or 12 candidates trades *relative* prices — yet with 3 correlated symbols it is the most natural unexploited family, and it diversifies by construction (one leg always short). Inputs: per-symbol mids (live). Agent: macro/daily. Effort ~6h.

### A2 — Macro/Daily Mean-Reversion

Premium/basis multi-day reversion and range-reversion after trend exhaustion. The mean-reversion row of the matrix is empty above MF, and the funding family (best-replicated edge in-house) is logically adjacent. Inputs: `ctx_premium_bps`, trend features, F1 settlement-clock. Agent: macro/daily generators. Effort ~5h.

### A3 — Volatility Squeeze / Breakout (MF–macro)

Vol-ratio compression → expansion momentum. `vol_ratio_short_long` is computed but no algorithm consumes it directionally. Complements jump_detector, which requires the jump to have already happened; the squeeze fires *before*. Effort ~4h.

### A4 — Queue-Value Execution Model

Expected value of a resting limit order as f(queue position, book state, microprice deviation). Not a signal — the execution layer that converts the tick-column graveyard (10 Tier-3 algorithms with real native-horizon IC) into maker economics. Sits between HF1 and HF5 in ambition; prerequisite thinking for HF5 (AS market making). Inputs: `micro_queue_position_bid`, F2 microprice. Effort ~8h, simulation-first.

### A5 — Daily HMM (no action — trigger already set)

Latent state at daily horizon. Already specced with a 60-day data trigger (~Aug 1, `ml_algorithms.txt`). Listed so the cell isn't re-invented; it fills itself if accumulation holds.

---

## 3. Priority & Sequencing

| Rank | Item | Type | When | Rationale |
|------|------|------|------|-----------|
| 1 | F1 settlement-clock | Feature (Py) | Now | ~2h, zero risk, unblocks the top-3 candidate LF1 |
| 2 | F3 + F2 OFI & microprice | Feature (Py) | Now | Unblocks HF1/HF2; validates in Python before any Rust promotion |
| 3 | A1 relative-value pairs | Algorithm | Now | Fills a whole missing logic family using live features only |
| 4 | F4 HAR-RV | Feature (Py) | Now | Near-certain value: feeds sizing/risk regardless of alpha outcomes |
| 5 | A3 vol squeeze | Algorithm | Now | Cheap, consumes an orphaned live feature |
| 6 | A2 macro mean-reversion | Algorithm | After F1 | Needs settlement-clock features |
| 7 | F6 cross-symbol state | Feature (Rust) | **Post-Jun 17** | Biggest structural unlock; ingestor architecture change |
| 8 | F9 Binance reference feed | Infrastructure | **Post-Jun 17** | New process, opens basis/lead-lag families |
| 9 | F7 bivariate Hawkes | Feature (Rust) | Post-Jun 17 | Unblocks HF3 |
| 10 | A4 queue-value model | Algorithm | With HF5 planning | Execution-layer; pairs with kill-switch + sim-first rule |

**Interaction with the roadmap:** ranks 1–5 are pure Python on existing parquet — implementable during the accumulation freeze without touching the ingestor or schema. Each new algorithm enters through the agent 5-gate protocol as a hypothesis, same as the literature candidates. F6/F7/F9 should be batched into a single post-Jun 17 ingestor change window (one restart, one schema migration) rather than three separate disruptions.

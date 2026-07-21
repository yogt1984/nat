# Institutional Algorithms — Survey & NAT Coverage Audit

**Last updated:** 2026-07-21

A survey of algorithms that institutional quant desks (HFT market-making, stat-arb,
execution, systematic ML) actually run, mapped to the canonical papers behind each idea and
cross-referenced against NAT's current implementation.

This document is the **"what the field does" + gap audit** view. It complements:

- [`ALGORITHMS.md`](ALGORITHMS.md) — catalogue of NAT's *own implemented* algorithms with OOS backtest results.
- [`PAPERS_IDEAS.md`](PAPERS_IDEAS.md) — the reading bibliography for the entropy/regime agent swarm.

**Coverage legend:**

- `HAVE` — implemented in `scripts/algorithms/` (or as a feature category).
- `PARTIAL` — a related idea exists, but the specific institutional estimator does not.
- `GAP` — not in NAT; a candidate worth analyzing.

---

## 1. Market microstructure & LOB dynamics (HFT / market-making desks)

| Algorithm / idea | Canonical paper(s) | NAT |
|---|---|---|
| **Order-flow imbalance → price impact** (workhorse of every MM desk) | Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*; Cont & de Larrard (2013), *Price dynamics in a Markovian LOB* | `HAVE` — `weighted_ofi` |
| **Transient market impact / propagator model** | Bouchaud, Gefen, Potters & Wyart (2004); Bouchaud, Farmer & Lillo (2009), *How markets slowly digest changes in supply and demand* | `HAVE` — `propagator` |
| **Hawkes self-exciting order flow** | Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*; Bacry & Muzy (2014); Lu & Abergel (2018) | `HAVE` — `hawkes_intensity`, `cascade_probability` |
| **Queue-reactive model** (state-dependent LOB, MM inventory) | Huang, Lehalle & Rosenbaum (2015), *Simulating and analyzing order book data with a queue-reactive model* | `GAP` — strong fit for L2 perp data |
| **VPIN / flow toxicity** | Easley, López de Prado & O'Hara (2012), *Flow Toxicity and Liquidity in a High-Frequency World* | `HAVE` — `vpin_regime` |
| **Kyle's λ / PIN — adverse selection & price discovery** | Kyle (1985), *Continuous Auctions and Insider Trading*; Easley et al. (1996), PIN | `PARTIAL` — VPIN yes; no explicit Kyle-λ estimator |
| **Information share / who leads price discovery** (spot vs perp, BTC vs alts) | Hasbrouck (1995), *One security, many markets*; Gonzalo & Granger (1995) | `GAP` — natural for cross-symbol lead-lag |
| **Microprice** (fair value beyond mid) | Stoikov (2018), *The micro-price: a high-frequency estimator of future prices* | `GAP` — cheap, high-value; directly attacks the fill-conditional IC collapse |

**Priority:** *microprice* and *queue-reactive* — both directly attack the "IC ≈ 0.45 mid collapses to
~0.03 under realistic fills" constraint.

---

## 2. Optimal execution & market-making control

The part most quant shops treat as crown jewels. This is NAT's biggest structural gap and is the
framework to analyze *before* the paper-trading → live transition (currently pre-G8).

| Algorithm | Canonical paper | NAT |
|---|---|---|
| **Almgren–Chriss optimal execution** | Almgren & Chriss (2000), *Optimal execution of portfolio transactions* | `GAP` — baseline for any live sizing / TWAP |
| **Avellaneda–Stoikov optimal market making** | Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book* | `GAP` — reference MM inventory model |
| **Guéant–Lehalle–Fernandez-Tapia MM** | Guéant, Lehalle & Fernandez-Tapia (2013), *Dealing with the inventory risk* | `GAP` |
| **Cartea–Jaimungal stochastic-control suite** | Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading* (book); Cartea & Jaimungal (2016), order-flow signals in execution | `GAP` — the modern institutional bible |
| **SPRT / sequential detection for entries** | Wald (1945), *Sequential Analysis* | `HAVE` — `optimal_entry` |

---

## 3. Volatility, jumps & regime (stat-arb / vol desks)

| Algorithm | Canonical paper | NAT |
|---|---|---|
| **Lee–Mykland jump test** | Lee & Mykland (2008), *Jumps in Financial Markets* | `HAVE` — `jump_detector` |
| **Bipower variation jump separation** | Barndorff-Nielsen & Shephard (2004, 2006) | `HAVE` — `bipower_jump` |
| **Realized kernels / noise-robust RV** | Barndorff-Nielsen, Hansen, Lunde & Shephard (2008) | `GAP` — matters for 100ms noise |
| **Rough volatility** | Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough* | `GAP` — crypto vol is strongly rough; good for macro-horizon agent |
| **HAR-RV forecasting** | Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility* | `GAP` — trivial to add, strong multi-horizon baseline |
| **Regime switching (HMM / Markov-switching)** | Hamilton (1989); Ang & Bekaert (2002) | `HAVE` — `regime_state_machine`, GMM, `switching_ou` |
| **OU mean-reversion / pairs** | Avellaneda & Lee (2010), *Statistical arbitrage in the US equities market* | `HAVE` — `switching_ou`, `mean_reversion_detector` |

**Quick wins:** *HAR-RV* and *realized kernels* for the MF/macro agents. *Rough vol* is the
research-grade one worth a preprint.

---

## 4. ML / meta-strategy layer (López de Prado school)

What systematic funds standardized on.

| Algorithm | Canonical paper | NAT |
|---|---|---|
| **Triple-barrier + meta-labeling** | López de Prado (2018), *Advances in Financial Machine Learning*, ch. 3 | `HAVE` — `meta_labeling` |
| **Deflated Sharpe / PBO (overfit control)** | Bailey & López de Prado (2014); Bailey et al. (2016), PBO | `HAVE` — DSR gate (G4) |
| **Fractional differentiation** (stationary-yet-memory features) | López de Prado (2018), ch. 5 | `GAP` — cheap; helps every downstream model |
| **Deep LOB (CNN/LSTM on raw book)** | Zhang, Zohren & Roberts (2019), *DeepLOB* | `GAP` — L2 data already available |
| **Attention / Transformer LOB** | Wallbridge (2020); Tran et al. (2021), *Temporal attention-augmented BiLSTM* | `GAP` |
| **Nearest-neighbor / analog forecasting** | empirical (Lorenz-analog lineage) | `HAVE` — `knn_retrieval` |

---

## 5. Crypto-perp–specific (funding, basis, liquidation)

NAT's actual instrument.

| Idea | Source | NAT |
|---|---|---|
| **Funding-rate carry / basis reversion** | Practitioner lit; academic: Alexander et al. (2020+) on perpetual basis; exchange funding mechanics | `HAVE` — `funding_reversion`, `funding_settlement` |
| **Liquidation cascade prediction** | Empirical crypto microstructure (limited academic; cf. NAT hypothesis H3) | `HAVE` — `cascade_probability` |
| **Cross-exchange / spot-perp lead-lag** | Hasbrouck (1995) info-share applied cross-venue; Makarov & Schoar (2020), *Trading and arbitrage in cryptocurrency markets* | `PARTIAL` — cross-symbol features exist; no formal info-share estimator |

---

## Recommended shortlist (five things worth analyzing next)

1. **Stoikov (2018) microprice** — directly targets the fill-collapse problem, small to implement.
2. **Cartea, Jaimungal & Penalva (2015)** — the institutional signals→execution framework missing before live.
3. **Huang, Lehalle & Rosenbaum (2015) queue-reactive** — best use of untapped L2 depth.
4. **Gatheral–Jaisson–Rosenbaum (2018) rough vol** — research-grade, preprint-worthy for the macro agent.
5. **Zhang–Zohren–Roberts (2019) DeepLOB** — the raw book it needs is already streamed.

The two lowest-risk / highest-value additions that fit the existing `MicrostructureAlgorithm` ABC are
**microprice** and **HAR-RV** — both are natural next `scripts/algorithms/` units (planted test first,
per methodology).

# Microstructure Research Agent — Autonomous Alpha Discovery & Deployment

**Date**: 2026-05-15
**Status**: Concept — specification stage

## Motivation

NAT already has ~100 terminal commands across 15 categories: data ingestion, feature
profiling, spectral analysis, regime screening, alpha pipeline, backtesting, model
training, paper trading, and deployment readiness checks. The Spannung research arc
(Phases A-F) demonstrated that chaining these tools in a disciplined sequence —
grid search -> causality check -> OOS validation -> cost backtest -> spectral analysis
-> regime screening -> cross-symbol walk-forward — can discover genuine, replicating
microstructure signals. But each step was manually orchestrated.

The idea: **wrap the existing NAT toolchain in an autonomous agent that continuously
ingests data, generates research hypotheses, runs experiments, validates findings,
assembles signals into portfolios, paper trades, and ultimately deploys live algorithms
— with human oversight at critical gates.**

## What Already Exists

| Capability | NAT Command | Status |
|-----------|-------------|--------|
| 24/7 data ingestion (100ms, 191 features) | `nat start` | Production |
| Feature profiling + walk-forward | `nat profile scalp --forward-test` | Production |
| Spectral analysis (PSD, coherence, band IC) | `nat spannung spectral` | Production |
| Regime screening (quintile, Pareto, persistence) | `nat spannung regime` | Production |
| Signal grid search | `nat spannung` | Production |
| Cost-aware backtest | `nat spannung backtest` | Production |
| Horizon sweep | `nat spannung horizon` | Production |
| Alpha combination + sizing | `nat alpha combine/size` | Production |
| Walk-forward validation | `nat alpha validate` | Production |
| Regime conditioning | `nat alpha regime` | Production |
| Multi-frequency integration | `nat alpha multi-freq` | Partial |
| Portfolio assembly | `nat alpha portfolio` | Partial |
| Paper trading simulation | `nat alpha paper` | Partial |
| Deployment readiness | `nat alpha deploy` | Partial |
| GMM regime clustering | `nat profile`, `nat cluster` | Production |
| HMM fitting | `nat cluster hmm-fit` | Production |
| Skeptical validation (20+ tests) | `nat validate skeptical` | Production |
| Experiment tracking | `nat experiment workflow` | Production |
| Signal existence testing | `nat signal test` | Production |
| EAMM market-making | `nat eamm run/backtest` | Production |
| Live API server | `nat api start` | Production |

**~80% of the execution layer exists.** What's missing is the decision layer — the
brain that decides *what* to test, *when* to promote, and *how* to combine.

## Architecture — Seven Layers

```
Layer 7: Live Execution        ── Hyperliquid orders, risk management, kill switch
Layer 6: Promotion Gate        ── Paper → live graduation criteria
Layer 5: Paper Trading         ── Shadow execution, realized vs predicted IC tracking
Layer 4: Signal Assembly       ── Signal registry, portfolio optimization, sizing
Layer 3: Experiment Runner     ── Automated test execution, replication protocol
Layer 2: Hypothesis Engine     ── The "brain" — generates testable claims
Layer 1: Data Fabric           ── Continuous ingestion, metadata, manifest
```

### Layer 1 — Continuous Data Fabric

The Rust ingestor already runs 24/7 at 100ms resolution, producing 191 features per
symbol across BTC/ETH/SOL. Additions needed:

- **Data manifest** (`data/manifest.json`): auto-updated index of every parquet file
  with start/end timestamps, row counts, symbols, market context (avg volatility,
  avg spread, volume quantile, any exchange events)
- **Quality monitor**: extends `nat data validate` to run continuously, flagging
  corrupted files (6 corrupted parquets found in 2026-05-12 data), gaps, schema
  drift, frozen features
- **Market context tagger**: annotates each hour with regime labels from the GMM
  (via `nat cluster gmm`), enabling the hypothesis engine to reason about "test
  this in volatile regimes" vs "test this in quiet regimes"

### Layer 2 — Hypothesis Engine

The core intellectual contribution. A priority queue of testable claims, each with:

```python
@dataclass
class Hypothesis:
    id: str                          # unique identifier
    claim: str                       # human-readable statement
    category: str                    # signal, regime, spectral, cross-asset, cost
    generator: str                   # which generator produced it
    priority: float                  # expected information gain
    required_data: DataRequirement   # min hours, symbols, regime coverage
    test_protocol: TestProtocol      # which nat commands to run, thresholds
    status: str                      # queued, running, passed, failed, replicated
    parent_id: str | None            # hypothesis that spawned this one
    created: datetime
    results: dict | None
```

**See "Hypothesis Generation Fabric" section below for the generation methodology.**

### Layer 3 — Experiment Runner

Picks the highest-priority hypothesis from the queue, checks data availability against
the manifest, and executes the test protocol using existing NAT tools.

**Replication protocol** — every finding must pass three gates before advancing:
1. **Discovery gate**: walk-forward on the discovery date (5-fold, KEEP verdict)
2. **Temporal replication**: OOS on at least 2 other dates with IC within 70% of IS
3. **Cross-symbol replication**: KEEP verdict on at least 2 of 3 symbols

Failed hypotheses are logged with structured failure reasons:
- `insufficient_data` — not enough bars/hours to test
- `no_effect` — IC below threshold, no statistical significance
- `no_replication` — IS effect doesn't hold OOS or cross-symbol
- `no_persistence` — regime duration too short (<5s) for execution
- `cost_killed` — signal exists but net edge negative after costs

This failure taxonomy prevents re-testing dead ends and informs the hypothesis
generators about which directions are unproductive.

### Layer 4 — Signal Assembly

Validated signals get registered in a **signal registry**:

```python
@dataclass
class RegisteredSignal:
    name: str                       # e.g., "imbalance_l1_ent_gated"
    features: list[str]             # input features
    regime_gate: str | None         # e.g., "ent_book_shape < P40"
    spectral_band: tuple | None     # e.g., (0.005, 0.1) Hz
    extraction: str                 # "raw", "kalman", "bandpass"
    horizon_s: float                # optimal forward horizon in seconds
    expected_ic: float              # walk-forward OOS IC
    expected_ir: float              # IC information ratio
    decay_halflife_s: float         # signal decay from lag analysis
    correlation_with: dict          # pairwise correlation with other registered signals
    symbols: list[str]              # which symbols it works on
    capacity_usd: float | None      # estimated capacity before impact
    discovery_date: str
    last_validated: str
```

The **portfolio optimizer** (extending `nat alpha portfolio`) combines uncorrelated
signals subject to:
- Maximum correlation between any two signals < 0.5
- Per-symbol position limit
- Total portfolio risk budget
- Cost-aware sizing via `nat alpha size`

### Layer 5 — Paper Trading

Extends `nat alpha paper` into a live shadow trader:

- Connects to Hyperliquid WebSocket (read-only) or testnet
- Executes the assembled portfolio in simulated real-time
- Tracks: fill rate assumptions, slippage model accuracy, realized IC vs predicted IC,
  regime gate firing rate, actual vs expected turnover

**Anomaly detector**: monitors realized metrics in rolling windows:
- If realized IC drops >2 std below expected for >1 hour → pause signal, trigger diagnostic
- If regime gate firing rate deviates >50% from backtest → recalibrate thresholds
- If cross-signal correlation spikes (regime-driven) → reduce portfolio exposure

### Layer 6 — Promotion Gate

A signal graduates from paper to live when ALL of:
1. Paper Sharpe > 1.5 for >= 7 consecutive days
2. Realized IC within 80% of backtest IC
3. Maximum drawdown < 2% of allocated capital
4. No anomaly flags in last 48 hours
5. Human approval (initially — can be automated later)

**Graduated deployment**:
- Day 1-3: 10% target size
- Day 4-7: 25% target size (if metrics hold)
- Day 8-14: 50% target size
- Day 15+: 100% target size
- Automatic de-escalation: drop one tier if any metric breaches for >4 hours

### Layer 7 — Live Execution

Connects to Hyperliquid via the existing `api` crate (Axum on port 3000).

**Market-making mode** (primary, for imbalance-based signals):
- Post limit orders leaned by Kalman-filtered imbalance signal
- Refresh at OU half-life (~7s, from Phase D)
- Widen spread when `ent_book_shape` is high (noisy regime — Phase E)
- Tighten spread when `ent_book_shape` is low (informed regime — signal is strong)
- Target zero-fee pairs to eliminate taker cost

**Directional mode** (secondary, for longer-horizon signals):
- Take positions when regime gate fires AND signal exceeds threshold
- Hold for signal-specific horizon
- Exit on signal reversal or time expiry

**Risk manager**:
- Per-symbol position limits (notional and as % of daily volume)
- Cross-symbol correlation limits (don't be 3x long BTC via correlated signals)
- Daily loss limit → reduce size by 50%; weekly loss limit → pause all trading
- Kill switch: manual override via `nat agent stop` or Telegram command

---

## Hypothesis Generation Fabric

This is the hardest and most valuable component — the "brain" that turns NAT from
a toolbox into an autonomous researcher. The key insight: **hypothesis generation is
itself a structured search problem** that can be decomposed into generators with
different exploration strategies.

### Generator Architecture

```
                    +-----------------------+
                    |   Hypothesis Queue    |
                    |   (priority-ordered)  |
                    +-----------+-----------+
                                ^
                                |
          +---------------------+---------------------+
          |          |          |          |           |
     Systematic  Spectral  Regime     Cross-Asset  Failure
     Screening   Anomaly   Transition  Probing     Recycler
     Generator   Detector  Generator   Generator   Generator
```

### Generator 1: Systematic Screening

**What it does**: Exhaustive pairwise search of feature x condition x threshold.

This is the generalized version of `nat spannung regime`. For every feature F in the
191-feature vector:
- Test F as a **directional signal** (Spearman IC with forward returns at 1s, 5s, 30s, 60s)
- Test F as a **regime gate** for every known directional signal (does conditioning on
  F improve IC of the existing signal?)
- Test at quintile thresholds (P20/P40/P60/P80), both directions (< and >)

**Priority scoring**: new feature x condition pairs that haven't been tested get high
priority. Pairs that are uncorrelated with existing findings get bonus priority
(exploring orthogonal directions is more valuable than refining known effects).

**Output**: single-factor results feed into multi-factor combination search (same as
Phase E's Pareto frontier, but running continuously as new data arrives).

**Estimated search space**: 191 features x 17 gate features x 8 thresholds x 4 horizons
= ~104k tests per date. At ~50ms per test, ~90 minutes per full sweep. Run nightly.

### Generator 2: Spectral Anomaly Detector

**What it does**: Monitors the frequency-domain characteristics of all signals for
structural changes.

- Run `nat spannung spectral` daily on new data
- Compare today's PSD slope, Hurst exponent, coherence peaks, and band-filtered ICs
  against the historical distribution
- Flag anomalies: new coherence peaks (a new frequency at which imbalance predicts
  returns), PSD slope changes (signal becoming more/less persistent), band IC shifts
  (predictive power migrating to a different frequency)

**Hypothesis generation**: "Coherence at 0.05 Hz increased from 0.08 to 0.22 over
the last 3 days — test whether a bandpass filter at 0.04-0.06 Hz produces tradeable
IC." Or: "Hurst exponent dropped from 0.43 to 0.35 — the signal is becoming more
mean-reverting, re-optimize the OU half-life and market-making refresh rate."

**Key insight**: spectral characteristics are slow-moving (Phase D showed remarkable
stability across dates), so genuine changes are high-information events that should
trigger immediate investigation.

### Generator 3: Regime Transition Detector

**What it does**: Uses the GMM/HMM models (`nat cluster gmm`, `nat cluster hmm-fit`)
to detect when the market transitions between regimes, then tests whether signal
quality changes at the transition.

- Run HMM daily, track the state sequence
- At each regime transition: measure signal IC in the 30 minutes before and after
- Generate hypotheses: "IC increases by 15% in the first 5 minutes after transitioning
  from state 2 (low-vol) to state 3 (high-vol) — test whether transition-gated
  trading is profitable"

**Intuition**: regime transitions are moments of maximum information asymmetry.
Informed traders reposition before the market catches up. The imbalance signal should
be strongest at these transition points.

### Generator 4: Cross-Asset Probing

**What it does**: Tests whether signals in one asset predict returns in another.

- For each validated signal S on symbol A: compute S(t) and correlate with forward
  returns of symbol B at lags 0, 100ms, 500ms, 1s, 5s, 30s
- Focus on lead-lag at the spectral timescales identified in Phase D (the 0.015 Hz
  coherence frequency = 68s cycles)
- Test both direct (BTC imbalance → ETH returns) and inverse (when BTC and ETH
  imbalance diverge, which one leads?)

**Hypothesis generation**: "BTC imbalance leads SOL returns by 2.3s at 0.015 Hz with
coherence 0.18 — test whether a delayed-SOL strategy using BTC imbalance as the signal
produces positive net edge on zero-fee SOL pairs."

**Priority**: cross-asset signals are valuable because they are capacity-additive
(trading SOL based on BTC signal doesn't consume BTC capacity).

### Generator 5: Failure Recycler

**What it does**: Re-examines failed hypotheses when conditions change.

- Maintains a graveyard of failed hypotheses with their failure reasons
- When new data arrives that changes the failure condition, re-queue the hypothesis:
  - `insufficient_data` → re-queue when enough new data accumulates
  - `no_replication` → re-queue when a new date's data arrives (maybe the original
    OOS date was anomalous)
  - `cost_killed` → re-queue when a new zero-fee pair becomes available or when a
    complementary signal is found that could be combined to clear costs
  - `no_persistence` → re-queue if a new regime gate is discovered that might
    extend episode duration

**Intuition**: a hypothesis that failed on 2 dates might succeed on 5 dates. A signal
that was too weak alone might become viable when combined with a newly discovered
orthogonal signal. The failure graveyard is a source of "almost-worked" ideas that
deserve periodic re-examination.

### Priority Scoring

Each generator produces hypotheses. They compete for the experiment runner's time
based on a priority score:

```
priority = (expected_ic_gain * novelty * data_readiness) / estimated_runtime

where:
  expected_ic_gain  = prior estimate of IC improvement (from generator heuristics)
  novelty           = 1 / (1 + correlation with existing validated signals)
  data_readiness    = fraction of required data already available
  estimated_runtime = wall-clock minutes to run the test protocol
```

High-novelty, low-cost hypotheses run first. Expensive hypotheses (e.g., full
cross-asset spectral analysis across 3 symbols) run during off-peak hours.

### Meta-Learning: Generator Performance Tracking

Track each generator's hit rate (fraction of hypotheses that pass all three
replication gates). Over time, allocate more hypothesis budget to generators
with higher hit rates. This is a simple multi-armed bandit:

```
generator_weight[g] = (successes[g] + 1) / (attempts[g] + 2)  # Beta prior
```

If the spectral anomaly detector has a 30% hit rate while systematic screening
has 2%, the agent should spend more time on spectral anomalies. The weights
update monthly.

---

## Implementation Roadmap

### Phase 1 — Signal Registry + Automated Replication (1-2 weeks)
- Build the signal registry schema
- Automate the 3-gate replication protocol (walk-forward + temporal OOS + cross-symbol)
- Wire existing `nat spannung` + `nat profile scalp` results into the registry
- Register the imbalance signal variants from Phases A-F as the first entries

### Phase 2 — Hypothesis Engine MVP (2-3 weeks)
- Implement the hypothesis queue data structure
- Build Generator 1 (systematic screening) as a nightly cron job
- Build Generator 5 (failure recycler) as a weekly cron job
- Add experiment runner daemon that pops from queue and runs `nat` commands

### Phase 3 — Spectral + Regime Generators (1-2 weeks)
- Build Generator 2 (spectral anomaly detector) — daily PSD/coherence comparison
- Build Generator 3 (regime transition detector) — HMM state change analysis
- Both feed into the hypothesis queue

### Phase 4 — Paper Trading Loop (2-3 weeks)
- Extend `nat alpha paper` to run continuously against live feed
- Build anomaly detector for realized vs predicted metrics
- Implement the promotion gate criteria

### Phase 5 — Cross-Asset + Live Execution (2-4 weeks)
- Build Generator 4 (cross-asset probing)
- Connect to Hyperliquid execution via `api` crate
- Build risk manager and kill switch
- Graduated deployment logic

### Phase 6 — Meta-Learning + Full Autonomy (ongoing)
- Generator performance tracking and budget allocation
- Human oversight dashboard (Telegram alerts for promotion decisions)
- Continuous refinement of priority scoring

---

## Initial AI Agent Specification

### Core Loop

```
while True:
    # 1. Check data fabric
    manifest = update_manifest()
    new_data = manifest.since(last_run)

    # 2. Generate hypotheses
    for generator in [systematic, spectral, regime, cross_asset, recycler]:
        new_hypotheses = generator.generate(
            existing_signals=registry.all(),
            available_data=manifest,
            failed_hypotheses=graveyard,
        )
        queue.push_all(new_hypotheses)

    # 3. Run experiments
    while queue.has_runnable(manifest):
        hypothesis = queue.pop()
        result = experiment_runner.run(hypothesis)

        if result.passed_discovery_gate():
            result = experiment_runner.replicate(hypothesis, manifest)

            if result.passed_all_gates():
                registry.register(result.to_signal())
                portfolio.reoptimize(registry)
            else:
                graveyard.add(hypothesis, result.failure_reason)
        else:
            graveyard.add(hypothesis, result.failure_reason)

    # 4. Paper trade
    paper_trader.update(portfolio.current())
    metrics = paper_trader.collect_metrics()

    # 5. Promote or demote
    for signal in registry.paper_trading():
        if promotion_gate.check(signal, metrics):
            live_executor.promote(signal)
        elif anomaly_detector.flagged(signal, metrics):
            live_executor.demote(signal)

    # 6. Sleep until next cycle
    sleep(cycle_interval)  # 1h for hypothesis generation, real-time for execution
```

### Configuration

```toml
[agent]
cycle_interval_hours = 1
max_hypotheses_per_cycle = 50
max_experiments_per_cycle = 10

[replication]
min_oos_dates = 2
min_symbols = 2
ic_retention_threshold = 0.7    # OOS IC must be >= 70% of IS IC

[promotion]
min_paper_days = 7
min_paper_sharpe = 1.5
max_drawdown_pct = 2.0
ic_realized_threshold = 0.8    # realized IC >= 80% of backtest IC

[risk]
max_position_per_symbol_usd = 10000
max_daily_loss_pct = 1.0
max_weekly_loss_pct = 3.0
kill_switch_loss_pct = 5.0

[generators]
systematic_enabled = true
systematic_schedule = "0 2 * * *"    # nightly at 2am
spectral_enabled = true
spectral_schedule = "0 3 * * *"      # nightly at 3am
regime_enabled = true
cross_asset_enabled = true
recycler_schedule = "0 4 * * 0"      # weekly Sunday 4am

[generators.priority]
novelty_weight = 0.4
ic_gain_weight = 0.3
data_readiness_weight = 0.2
runtime_penalty_weight = 0.1
```

### CLI Interface

```bash
nat agent start              # Start the autonomous agent daemon
nat agent stop               # Graceful shutdown (finish current experiment)
nat agent status             # Queue depth, running experiments, paper P&L, live P&L
nat agent queue              # Show hypothesis queue with priorities
nat agent registry           # Show signal registry (validated signals)
nat agent graveyard          # Show failed hypotheses with reasons
nat agent promote <signal>   # Manual promotion override
nat agent demote <signal>    # Manual demotion
nat agent pause              # Pause all live trading, keep research running
nat agent report             # Generate full agent status report
nat agent log                # Tail agent log
```

### Telegram Alerts

- **Discovery**: "New signal found: `flow_aggressor_ratio_5s` gated by `ent_tick_5s<P20`, OOS IC=0.12, cross-symbol KEEP on BTC/ETH"
- **Promotion ready**: "Signal `imbalance_l1_ent_gated` passed 7-day paper trial: Sharpe=2.1, realized IC=0.41 (backtest: 0.45). Approve promotion? /yes /no"
- **Anomaly**: "Signal `imbalance_l1_ent_gated` realized IC dropped to 0.18 (expected: 0.41) over last 2h. Auto-paused. Investigate? /diagnose"
- **P&L**: Daily summary: "Agent P&L: +12.3 bps (paper), +4.1 bps (live). 3 signals active, 2 paper, 47 hypotheses queued."

---

## Connection to Spannung Research

The Spannung arc (Phases A-F) is the proof-of-concept for this agent. The research
sequence that a human executed manually:

```
Grid search → Causality check → OOS validation → Cost backtest → Bar aggregation test
→ Spectral analysis → Regime screening → Cross-symbol walk-forward
```

...is exactly the experiment runner's replication protocol. The key findings that
would already be in the signal registry:

| Signal | Gate | Band | IC | Horizon | Status |
|--------|------|------|-----|---------|--------|
| imbalance_qty_l1 | none | raw | 0.45 | 5s | validated, not tradeable at cost |
| imbalance_qty_l1 | ent_book_shape<P40 | ultra_low | 0.55 | 5s | validated, paper candidate |
| imbalance_qty_l1 | ent_book_shape<P20 + toxic_vpin_50>P80 | ultra_low | 0.64 | 5s | validated, too short-lived |

The agent would have generated these findings automatically via Generator 1
(systematic screening) and Generator 2 (spectral anomaly detector), then tested
them through the replication protocol, and currently be paper-trading the middle
row on zero-fee pairs.

## Open Questions

- How to handle non-stationarity? Microstructure signals decay as more participants
  discover them. The agent needs a "signal aging" model.
- Capacity estimation: how much capital can each signal absorb before market impact
  erodes the edge?
- Multi-agent coordination: if multiple instances run on different machines, how to
  prevent redundant research and conflicting positions?
- Compute budget: spectral analysis and walk-forward validation are CPU-intensive.
  How to prioritize when compute is constrained?
- Regulatory considerations: does autonomous trading on Hyperliquid require any
  compliance framework?

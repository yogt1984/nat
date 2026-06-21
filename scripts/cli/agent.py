"""`nat agent` / `mf-agent` / `macro-agent` / `meta-agent` / `it-engine` — the
autonomous research-agent families (they share `_agent_run`/`_agent_stop` and the
*_SCRIPT constants)."""

from __future__ import annotations

import argparse
import os
import textwrap

from cli.common import (
    ROOT, G, Y, _json, _banner, _py, _sh, _p, _sym, _json_mode, _output,
)


# ── Agent commands ───────────────────────────────────────────────────────────

AGENT_SCRIPT = "scripts/agent/daemon.py"


def _agent_run(script, verb, banner=None):
    """Run a research-agent daemon subcommand (the common passthrough)."""
    if banner:
        _banner(banner)
    _py(f"{script} {verb}")


def _agent_stop(daemon_match, label):
    """SIGTERM a running agent daemon, matched by its launch command line."""
    import signal as sig
    r = _sh(f"pgrep -f '{daemon_match}'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to {label} (pid {pid})")
    else:
        _p("–", Y, f"No running {label} found")


def cmd_agent_start(args):  _agent_run(AGENT_SCRIPT, "start", "Starting NAT Agent daemon")
def cmd_agent_stop(args):   _agent_stop("agent/daemon.py start", "agent")
def cmd_agent_once(args):   _agent_run(AGENT_SCRIPT, "once", "Running single agent cycle")

def cmd_agent_status(args):
    if _json_mode(args):
        p = ROOT / "data" / "agent" / "agent_state.json"
        if p.exists():
            print(p.read_text())
        else:
            _output({"error": "No agent state file", "path": str(p)}, args)
        return
    _py(f"{AGENT_SCRIPT} status")

def cmd_agent_queue(args):
    if _json_mode(args):
        p = ROOT / "data" / "agent" / "hypotheses.json"
        if p.exists():
            print(p.read_text())
        else:
            _output({"hypotheses": []}, args)
        return
    _py(f"{AGENT_SCRIPT} queue")

def cmd_agent_registry(args):
    if _json_mode(args):
        p = ROOT / "data" / "agent" / "registry.json"
        if p.exists():
            print(p.read_text())
        else:
            _output({"signals": []}, args)
        return
    _py(f"{AGENT_SCRIPT} registry")

def cmd_agent_graveyard(args):
    if _json_mode(args):
        p = ROOT / "data" / "agent" / "graveyard.json"
        if p.exists():
            print(p.read_text())
        else:
            _output({"failed": []}, args)
        return
    _py(f"{AGENT_SCRIPT} graveyard")

def cmd_agent_dashboard(args):
    port = getattr(args, 'port', 8060)
    _py(f"scripts/agent_dashboard.py --port {port}")

def cmd_agent_report(args):
    if _json_mode(args):
        # Aggregate all agent JSON state files
        agent_dir = ROOT / "data" / "agent"
        result = {}
        for name in ["agent_state", "registry", "graveyard", "hypotheses"]:
            p = agent_dir / f"{name}.json"
            if p.exists():
                try:
                    result[name] = _json.loads(p.read_text())
                except _json.JSONDecodeError:
                    result[name] = {"error": "invalid JSON"}
            else:
                result[name] = None
        _output(result, args)
        return
    _py(f"{AGENT_SCRIPT} report")


# ── Medium-frequency agent commands ──────────────────────────────────────────

MF_AGENT_SCRIPT = "scripts/agent/mf_daemon.py"

def cmd_mf_agent_start(args):     _agent_run(MF_AGENT_SCRIPT, "start", "Starting Medium-Frequency Agent daemon")
def cmd_mf_agent_stop(args):      _agent_stop("agent/mf_daemon.py start", "MF agent")
def cmd_mf_agent_once(args):      _agent_run(MF_AGENT_SCRIPT, "once", "Running single MF agent cycle")
def cmd_mf_agent_status(args):    _agent_run(MF_AGENT_SCRIPT, "status")
def cmd_mf_agent_queue(args):     _agent_run(MF_AGENT_SCRIPT, "queue")
def cmd_mf_agent_registry(args):  _agent_run(MF_AGENT_SCRIPT, "registry")
def cmd_mf_agent_graveyard(args): _agent_run(MF_AGENT_SCRIPT, "graveyard")
def cmd_mf_agent_report(args):    _agent_run(MF_AGENT_SCRIPT, "report")


# ── Macro agent commands ─────────────────────────────────────────────────────

MACRO_AGENT_SCRIPT = "scripts/agent/macro_daemon.py"

def cmd_macro_agent_start(args):     _agent_run(MACRO_AGENT_SCRIPT, "start", "Starting Macro Agent daemon")
def cmd_macro_agent_stop(args):      _agent_stop("agent/macro_daemon.py start", "macro agent")
def cmd_macro_agent_once(args):      _agent_run(MACRO_AGENT_SCRIPT, "once", "Running single macro agent cycle")
def cmd_macro_agent_status(args):    _agent_run(MACRO_AGENT_SCRIPT, "status")
def cmd_macro_agent_queue(args):     _agent_run(MACRO_AGENT_SCRIPT, "queue")
def cmd_macro_agent_registry(args):  _agent_run(MACRO_AGENT_SCRIPT, "registry")
def cmd_macro_agent_graveyard(args): _agent_run(MACRO_AGENT_SCRIPT, "graveyard")
def cmd_macro_agent_report(args):    _agent_run(MACRO_AGENT_SCRIPT, "report")


# ── Meta-Agent (orchestrator) commands ────────────────────────────────────────

META_AGENT_SCRIPT = "scripts/agent/meta_daemon.py"

def cmd_meta_agent_start(args):       _agent_run(META_AGENT_SCRIPT, "start", "Starting Meta-Agent orchestrator")
def cmd_meta_agent_stop(args):        _agent_stop("agent/meta_daemon.py start", "meta-agent")
def cmd_meta_agent_once(args):        _agent_run(META_AGENT_SCRIPT, "once", "Running single meta-agent cycle")
def cmd_meta_agent_status(args):      _agent_run(META_AGENT_SCRIPT, "status")
def cmd_meta_agent_portfolio(args):   _agent_run(META_AGENT_SCRIPT, "portfolio")
def cmd_meta_agent_correlation(args): _agent_run(META_AGENT_SCRIPT, "correlation")
def cmd_meta_agent_budget(args):      _agent_run(META_AGENT_SCRIPT, "budget")
def cmd_meta_agent_report(args):      _agent_run(META_AGENT_SCRIPT, "report")


# ── IT Engine commands ──────────────────────────────────────────────────────

IT_ENGINE_SCRIPT = "scripts/it_engine/daemon.py"

def cmd_it_engine_start(args):
    sym = _sym(args)
    _banner(f"Starting IT Engine for {sym}")
    cmd = f"-m scripts.it_engine.daemon start --symbol {sym}"
    if getattr(args, 'offline', False):
        cmd += " --offline"
    data = getattr(args, 'data_dir', None)
    if data:
        cmd += f" --data-dir {data}"
    if getattr(args, 'dry_run', False):
        cmd += " --dry-run"
    if getattr(args, 'verbose', False):
        cmd += " -v"
    _py(cmd)

def cmd_it_engine_stop(args):
    sym = _sym(args)
    import signal as sig
    r = _sh(f"pgrep -f 'it_engine.daemon start.*{sym}'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to IT engine (pid {pid})")
    else:
        _p("–", Y, "No running IT engine found")

def cmd_it_engine_status(args):
    sym = _sym(args)
    top = getattr(args, 'top', 10)
    _py(f"-m scripts.it_engine.daemon status --symbol {sym} --top {top}")


def register(sub):
    # ── agent ──
    ag_p = sub.add_parser('agent', help='Autonomous research agent')
    ag_p.set_defaults(func=cmd_agent_status)
    agsub = ag_p.add_subparsers(dest='subcmd')
    agsub.add_parser('start', help='Launch agent daemon').set_defaults(func=cmd_agent_start)
    agsub.add_parser('stop', help='Graceful shutdown (SIGTERM)').set_defaults(func=cmd_agent_stop)
    ag_once = agsub.add_parser('once', help='Run single cycle (testing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Runs one autonomous research cycle: generate → execute → validate.

        5-Gate Protocol (per hypothesis):
          G1  Discovery:   |IC| ≥ 0.10 at any horizon, and dIC ≥ 0.05
                           (delta IC = regime-gated IC minus baseline IC)
          G2  Temporal:    IC stable across 3+ time sub-periods
          G3  Symbol:      IC replicates on ≥ 2 of {BTC, ETH, SOL}
          G4  Incremental: dIC ≥ 0.05 over existing best signal
          G5  Cost:        |IC| × σ > spread_bps (signal exceeds trading cost)

        FDR control: Benjamini-Hochberg at q = 0.05 across all tested hypotheses.
        Promoted signals are added to the registry for production use.

        Example:
          nat agent once
        """))
    ag_once.set_defaults(func=cmd_agent_once)
    agsub.add_parser('status', help='Current state & stats').set_defaults(func=cmd_agent_status)
    agsub.add_parser('queue', help='Queued hypotheses by priority').set_defaults(func=cmd_agent_queue)
    agsub.add_parser('registry', help='Validated signals').set_defaults(func=cmd_agent_registry)
    agsub.add_parser('graveyard', help='Failed hypotheses').set_defaults(func=cmd_agent_graveyard)
    agsub.add_parser('report', help='Full summary report').set_defaults(func=cmd_agent_report)
    dash_p = agsub.add_parser('dashboard', help='Launch web dashboard (port 8060)')
    dash_p.add_argument('--port', type=int, default=8060, help='Dashboard port (default: 8060)')
    dash_p.set_defaults(func=cmd_agent_dashboard)

    # ── mf-agent ──
    mf_p = sub.add_parser('mf-agent',
        help='Medium-frequency research agent (1min-1h)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Autonomous alpha discovery at 5-minute bar resolution.',
        epilog=textwrap.dedent("""\
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        MEDIUM-FREQUENCY RESEARCH AGENT
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        OVERVIEW
          Discovers alpha signals from 100ms tick data resampled to 5-minute
          bars. Runs a generate-test-validate loop every 2 hours with strict
          budget limits (8 experiments, 90-minute wall clock).

        LIFECYCLE (per cycle)
          MANIFEST → GENERATE → EXECUTE → FDR → MONITOR → SLEEP

          1. MANIFEST   Scan data/features/ for available date directories.
          2. GENERATE   Run 3 hypothesis generators (see below). Each produces
                        candidate signals: (feature, regime_gate, horizon).
          3. EXECUTE    Pop up to 8 hypotheses by priority. Each runs the
                        4-gate protocol (see below). Signals that pass all
                        gates are registered.
          4. FDR        Benjamini-Hochberg correction across all tested
                        hypotheses in this cycle (q = 0.05). Signals that
                        pass gates but fail FDR are retracted from registry.
          5. MONITOR    For each registered signal, compute rolling IC on the
                        latest 2 dates. If IC drops below 50% of discovery IC
                        for 14 consecutive days, auto-retire the signal.
          6. SLEEP      Wait cycle_interval_s (default: 7200s = 2h).

        ──────────────────────────────────────────────────────────────────────
        4-GATE PROTOCOL
        ──────────────────────────────────────────────────────────────────────

        Each hypothesis H = (signal_feature, regime_gate, horizon) must pass
        all 4 gates sequentially. Failure at any gate → graveyard.

        GATE 0: DISCOVERY (IC + dIC)
          Run: nat profile scalp --symbol BTC --data <latest_date>
               --timeframe 5min --forward-test
          Extract IC = Spearman(signal, fwd_return) from profiler report.

          Pass condition:
            |IC| >= min_ic                                   (default: 0.08)

          If regime_gate is specified, also check:
            dIC = IC_gated - IC_ungated >= min_dIC           (default: 0.03)

          The gate must ADD value beyond the raw signal.

          P-value for FDR:
            Under H0 (no predictive power), IC ~ N(0, 1/sqrt(n)).
            z = |IC| * sqrt(n),  p = erfc(z / sqrt(2))     (two-sided)

          Adaptive threshold: once registry has signals, min_ic is raised to
            max(0.08, median(registry_ICs) * 0.8)
          This prevents dilution — later signals must beat existing quality.

        GATE 1: TEMPORAL REPLICATION
          Re-run Gate 0 on 2 additional dates (earliest available).
          Pass condition:
            n_pass >= min_oos_dates                           (default: 2)
          Tests whether the signal persists across different market regimes.

        GATE 2: SYMBOL REPLICATION
          Re-run Gate 0 on ETH and SOL (same date as discovery).
          Pass condition:
            n_symbols_passed >= min_symbols - 1               (default: 1)
          At least 2 of {BTC, ETH, SOL} must show the effect.

        GATE 3: CORRELATION DEDUP
          Compute Spearman rank correlation between the candidate signal
          (regime-gated) and every existing registry signal (regime-gated),
          using the latest date's tick data.
          Pass condition:
            max(|corr(candidate, existing_i)|) < max_corr    (default: 0.70)
          Prevents registering redundant signals that measure the same thing.

        ──────────────────────────────────────────────────────────────────────
        GENERATORS
        ──────────────────────────────────────────────────────────────────────

        1. MOMENTUM  (prefix: HYP-MOM)
           Signals:  trend_momentum_300       — OLS slope of midprice over 5min
                     trend_momentum_r2_300    — R^2 of that regression
                     trend_hurst_300          — Hurst exponent (H > 0.5 = trend)
                     trend_ma_crossover_norm  — (EMA_10 - EMA_50) / sigma
           Gates:    ent_tick_1m < P{20,40,60,80}           (low entropy)
                     vol_ratio_short_long < P{20,40,60,80}  (stable vol)
                     derived_regime_type_score > P{20,40,60,80} (breakout)
           Generates: 4 signals x 3 gates x 4 thresholds = 48 hypotheses
           Priority boost: +0.20 for R2, +0.15 for Hurst

        2. VOL_BREAKOUT  (prefix: HYP-VBK)
           Signals:  vol_ratio_short_long  — sigma_1min / sigma_5min
                     vol_zscore            — (sigma - mu) / sigma_long
           Regimes:  continuation  — high vol breakout follows through
                     reversion     — vol spike mean-reverts
           Gates:    ent_tick_1m < P{20,40}     (directional, not noise)
                     ent_tick_5s < P{20,40}
           Generates: 2 signals x 2 regimes x 2 gates x 2 thresholds = 16
           Priority boost: +0.10 for continuation, +0.10 for vol_ratio

        3. FLOW_CLUSTER  (prefix: HYP-FCL)
           Signals:  imbalance_qty_l5         — L5 book imbalance (bar agg)
                     flow_aggressor_ratio_5s  — buy_vol / total_vol
                     flow_volume_5s           — total 5s volume per bar
           Gates:    illiq_composite > P{40,60,80}   (high impact = informed)
                     toxic_vpin_50 > P{40,60,80}     (toxic flow)
                     illiq_kyle_100 > P{40,60,80}    (Kyle's lambda)
           Generates: 3 signals x 3 gates x 3 thresholds = 27 hypotheses
           Priority boost: +0.15 for imbalance, +0.10 for VPIN gate

        Total per cycle: up to 91 hypotheses generated, 8 executed.

        ──────────────────────────────────────────────────────────────────────
        DATA PIPELINE
        ──────────────────────────────────────────────────────────────────────

        100ms ticks (209 features) → aggregate_bars(df, "5min") → 5min bars

        Aggregation rules per feature category:
          Price (midprice, spread):  OHLC (open, high, low, close, mean)
          Flow/volume counts:        sum
          Entropy:                   mean, std, slope (OLS over bar)
          Whale flow:                sum (cumulative)
          Everything else:           mean, std, last

        Result: ~602 columns per bar. After variance filtering: ~429 usable.
        ~12 bars/hour, ~288 bars/day per symbol.

        Forward returns for IC computation:
          fwd_ret_k = (midprice[t+k] - midprice[t]) / midprice[t]
          Horizons: k = 1 (5min), 2 (10min), 5 (25min), 10 (50min)

        ──────────────────────────────────────────────────────────────────────
        IC DECAY MONITORING
        ──────────────────────────────────────────────────────────────────────

        For each registered signal, each cycle computes:
          rolling_IC = Spearman(signal, fwd_return) on latest 2 dates

        Decay detection:
          threshold = expected_ic * ic_decay_ratio            (default: 0.5)
          If rolling_IC < threshold for consecutive_days_limit (default: 14)
          consecutive cycles → signal status = "retired".
          If IC recovers above threshold → decay counter resets to 0.

        IC history: last 30 observations retained per signal.

        ──────────────────────────────────────────────────────────────────────
        FDR CONTROL (Benjamini-Hochberg)
        ──────────────────────────────────────────────────────────────────────

        After all experiments in a cycle, collect p-values from IC gates.
        Sort ascending: p_(1) <= p_(2) <= ... <= p_(m).

        Find largest k such that p_(k) <= (k/m) * q,  where q = 0.05.
        Reject all hypotheses with p > p_(k).

        This controls the expected false discovery rate: E[FDP] <= q.
        Signals that passed all 4 gates but fail FDR are retracted.

        ──────────────────────────────────────────────────────────────────────
        STATE & STORAGE
        ──────────────────────────────────────────────────────────────────────

        All state is in data/agent_mf/ (separate from microstructure agent):
          agent_state.json   — phase, cycle count, counters, history
          hypotheses.json    — priority queue + graveyard
          registry.json      — validated signals with IC history
          generator_stats.json — per-generator attempt/success counts

        ──────────────────────────────────────────────────────────────────────
        CONFIGURATION  (config/agent.toml [agent_mf])
        ──────────────────────────────────────────────────────────────────────

          cycle_interval_s = 7200         # 2 hours between cycles
          max_experiments_per_cycle = 8   # budget per cycle
          max_cycle_runtime_s = 5400      # 90-minute wall clock limit
          timeframe = "5min"              # bar resolution

          [agent_mf.gates]
          min_ic = 0.08                   # IC gate floor
          min_dIC = 0.03                  # regime gate must add this much
          fdr_q = 0.05                    # Benjamini-Hochberg q
          min_oos_dates = 2               # temporal replication dates
          min_symbols = 2                 # symbol replication count

          [agent_mf.decay]
          ic_decay_ratio = 0.5            # retire below 50% of discovery IC
          consecutive_days_limit = 14     # days below threshold before retire

        ──────────────────────────────────────────────────────────────────────
        EXAMPLES
        ──────────────────────────────────────────────────────────────────────

          nat mf-agent once       # run one discovery cycle (testing)
          nat mf-agent start      # launch daemon (cycles every 2h)
          nat mf-agent status     # phase, counters, generator hit rates
          nat mf-agent registry   # list validated signals with IC
          nat mf-agent queue      # pending hypotheses by priority
          nat mf-agent graveyard  # failed hypotheses with failure reason
          nat mf-agent report     # full summary (registry + graveyard + stats)
          nat mf-agent stop       # SIGTERM to running daemon
        """))
    mf_p.set_defaults(func=cmd_mf_agent_status)
    mfsub = mf_p.add_subparsers(dest='subcmd')
    mfsub.add_parser('start', help='Launch MF agent daemon (cycles every 2h)').set_defaults(func=cmd_mf_agent_start)
    mfsub.add_parser('stop', help='Send SIGTERM to running daemon').set_defaults(func=cmd_mf_agent_stop)
    mfsub.add_parser('once', help='Run single discovery cycle (for testing)').set_defaults(func=cmd_mf_agent_once)
    mfsub.add_parser('status', help='Phase, cycle count, generator hit rates').set_defaults(func=cmd_mf_agent_status)
    mfsub.add_parser('queue', help='Pending hypotheses sorted by priority').set_defaults(func=cmd_mf_agent_queue)
    mfsub.add_parser('registry', help='Validated signals with IC and status').set_defaults(func=cmd_mf_agent_registry)
    mfsub.add_parser('graveyard', help='Failed hypotheses with failure reasons').set_defaults(func=cmd_mf_agent_graveyard)
    mfsub.add_parser('report', help='Full summary: registry + graveyard + generator stats').set_defaults(func=cmd_mf_agent_report)

    # ── macro-agent ──
    ma_p = sub.add_parser('macro-agent', help='Macro research agent (1h-24h)')
    ma_p.set_defaults(func=cmd_macro_agent_status)
    masub = ma_p.add_subparsers(dest='subcmd')
    masub.add_parser('start', help='Launch macro agent daemon').set_defaults(func=cmd_macro_agent_start)
    masub.add_parser('stop', help='Graceful shutdown (SIGTERM)').set_defaults(func=cmd_macro_agent_stop)
    masub.add_parser('once', help='Run single cycle (testing)').set_defaults(func=cmd_macro_agent_once)
    masub.add_parser('status', help='Current state & stats').set_defaults(func=cmd_macro_agent_status)
    masub.add_parser('queue', help='Queued hypotheses by priority').set_defaults(func=cmd_macro_agent_queue)
    masub.add_parser('registry', help='Validated signals').set_defaults(func=cmd_macro_agent_registry)
    masub.add_parser('graveyard', help='Failed hypotheses').set_defaults(func=cmd_macro_agent_graveyard)
    masub.add_parser('report', help='Full summary report').set_defaults(func=cmd_macro_agent_report)

    # ── meta-agent ──
    me_p = sub.add_parser('meta-agent', help='Meta-agent orchestrator (cross-agent)')
    me_p.set_defaults(func=cmd_meta_agent_status)
    mesub = me_p.add_subparsers(dest='subcmd')
    mesub.add_parser('start', help='Launch meta-agent daemon').set_defaults(func=cmd_meta_agent_start)
    mesub.add_parser('stop', help='Graceful shutdown (SIGTERM)').set_defaults(func=cmd_meta_agent_stop)
    mesub.add_parser('once', help='Run single orchestration cycle').set_defaults(func=cmd_meta_agent_once)
    mesub.add_parser('status', help='Current state & budgets').set_defaults(func=cmd_meta_agent_status)
    mesub.add_parser('portfolio', help='Signal portfolio').set_defaults(func=cmd_meta_agent_portfolio)
    mesub.add_parser('correlation', help='Cross-agent correlation matrix').set_defaults(func=cmd_meta_agent_correlation)
    mesub.add_parser('budget', help='Agent budget allocation').set_defaults(func=cmd_meta_agent_budget)
    mesub.add_parser('report', help='Full orchestrator report').set_defaults(func=cmd_meta_agent_report)

    # ── it-engine ──
    it_p = sub.add_parser('it-engine',
        help='Information-theoretic alpha discovery engine',
        description='Continuous IT-based feature analysis and alpha discovery.')
    it_p.set_defaults(func=cmd_it_engine_status)
    itsub = it_p.add_subparsers(dest='subcmd')
    it_start = itsub.add_parser('start', help='Start IT engine')
    it_start.add_argument('--symbol', '-s', default='BTC', help='Symbol (default: BTC)')
    it_start.add_argument('--offline', action='store_true', help='Run on parquet files')
    it_start.add_argument('--data-dir', default='data/features', help='Parquet data dir')
    it_start.add_argument('--dry-run', action='store_true', help='Run 1 cycle and exit')
    it_start.add_argument('-v', '--verbose', action='store_true')
    it_start.set_defaults(func=cmd_it_engine_start)
    it_stop = itsub.add_parser('stop', help='Stop IT engine')
    it_stop.add_argument('--symbol', '-s', default='BTC')
    it_stop.set_defaults(func=cmd_it_engine_stop)
    it_status = itsub.add_parser('status', help='Show IT engine status')
    it_status.add_argument('--symbol', '-s', default='BTC')
    it_status.add_argument('--top', type=int, default=10, help='Top N features')
    it_status.set_defaults(func=cmd_it_engine_status)


__all__ = [
    "AGENT_SCRIPT", "MF_AGENT_SCRIPT", "MACRO_AGENT_SCRIPT", "META_AGENT_SCRIPT",
    "IT_ENGINE_SCRIPT", "_agent_run", "_agent_stop",
    "cmd_agent_start", "cmd_agent_stop", "cmd_agent_once", "cmd_agent_status",
    "cmd_agent_queue", "cmd_agent_registry", "cmd_agent_graveyard",
    "cmd_agent_dashboard", "cmd_agent_report",
    "cmd_mf_agent_start", "cmd_mf_agent_stop", "cmd_mf_agent_once",
    "cmd_mf_agent_status", "cmd_mf_agent_queue", "cmd_mf_agent_registry",
    "cmd_mf_agent_graveyard", "cmd_mf_agent_report",
    "cmd_macro_agent_start", "cmd_macro_agent_stop", "cmd_macro_agent_once",
    "cmd_macro_agent_status", "cmd_macro_agent_queue", "cmd_macro_agent_registry",
    "cmd_macro_agent_graveyard", "cmd_macro_agent_report",
    "cmd_meta_agent_start", "cmd_meta_agent_stop", "cmd_meta_agent_once",
    "cmd_meta_agent_status", "cmd_meta_agent_portfolio", "cmd_meta_agent_correlation",
    "cmd_meta_agent_budget", "cmd_meta_agent_report",
    "cmd_it_engine_start", "cmd_it_engine_stop", "cmd_it_engine_status",
    "register",
]

# `nat` — CLI Reference

> **Generated** by `scripts/ops/gen_commands_doc.py` from the live argparse tree
> (`nat --json commands`). **Do not edit by hand** — regenerate instead. This file was
> hand-maintained until 2026-08-07, by which point 26 command groups were missing and the
> headline count was stale by 80. A reference that disagrees with the CLI is worse than no
> reference, because it is trusted.
>
> Maturity tags (`[PRELIM]` etc.) appear where a group declares one; surfacing them for
> *every* command lands with NAT9. Absence of a tag is not a claim of maturity.


**345 commands** across **73 groups**.

| Group | Commands |
|---|---|
| [`15m`](#15m) | 4 |
| [`agent`](#agent) | 10 |
| [`alg`](#alg) | 5 |
| [`alg1`](#alg1) | 3 |
| [`algorithm`](#algorithm) | 5 |
| [`alpha`](#alpha) | 14 |
| [`api`](#api) | 4 |
| [`audit`](#audit) | 3 |
| [`backtest`](#backtest) | 9 |
| [`bridge`](#bridge) | 5 |
| [`build`](#build) | 7 |
| [`cluster`](#cluster) | 7 |
| [`commands`](#commands) | 1 |
| [`config`](#config) | 5 |
| [`daily`](#daily) | 1 |
| [`dashboard`](#dashboard) | 1 |
| [`data`](#data) | 5 |
| [`deploy`](#deploy) | 4 |
| [`discovery`](#discovery) | 5 |
| [`docker`](#docker) | 8 |
| [`doctor`](#doctor) | 1 |
| [`eamm`](#eamm) | 4 |
| [`evolve`](#evolve) | 6 |
| [`exp`](#exp) | 9 |
| [`experiment`](#experiment) | 7 |
| [`fetch`](#fetch) | 2 |
| [`gap`](#gap) | 7 |
| [`gauntlet`](#gauntlet) | 5 |
| [`health`](#health) | 1 |
| [`help`](#help) | 1 |
| [`ing`](#ing) | 5 |
| [`it-engine`](#it-engine) | 4 |
| [`kalman`](#kalman) | 3 |
| [`lifecycle`](#lifecycle) | 7 |
| [`log`](#log) | 3 |
| [`macro`](#macro) | 1 |
| [`macro-agent`](#macro-agent) | 9 |
| [`mesh`](#mesh) | 1 |
| [`meta-agent`](#meta-agent) | 9 |
| [`metrics`](#metrics) | 4 |
| [`mf-agent`](#mf-agent) | 9 |
| [`model`](#model) | 7 |
| [`monitor`](#monitor) | 2 |
| [`nightly`](#nightly) | 3 |
| [`oos`](#oos) | 1 |
| [`oos30`](#oos30) | 1 |
| [`package`](#package) | 2 |
| [`pipeline`](#pipeline) | 7 |
| [`process`](#process) | 6 |
| [`profile`](#profile) | 2 |
| [`promotion`](#promotion) | 5 |
| [`report`](#report) | 4 |
| [`reports`](#reports) | 3 |
| [`risk`](#risk) | 5 |
| [`run`](#run) | 4 |
| [`scan`](#scan) | 1 |
| [`screen`](#screen) | 1 |
| [`service`](#service) | 5 |
| [`signal`](#signal) | 3 |
| [`spannung`](#spannung) | 5 |
| [`start`](#start) | 1 |
| [`status`](#status) | 1 |
| [`stop`](#stop) | 1 |
| [`swarm`](#swarm) | 6 |
| [`test`](#test) | 23 |
| [`tournament`](#tournament) | 9 |
| [`trade`](#trade) | 2 |
| [`validate`](#validate) | 3 |
| [`visualize`](#visualize) | 8 |
| [`viz`](#viz) | 8 |
| [`viz3d`](#viz3d) | 1 |
| [`wallets`](#wallets) | 4 |
| [`xs`](#xs) | 7 |


## 15m

| Command | Description |
|---|---|
| `nat 15m` | 15-minute experiment (ingest → analyze → report) |
| `nat 15m offline` | Analyze existing data (no ingestion) |
| `nat 15m test` | Run unit tests |
| `nat 15m viz` | Visualize latest experiment |

## agent

| Command | Description |
|---|---|
| `nat agent` | Autonomous research agent |
| `nat agent dashboard` | Launch web dashboard (port 8060) |
| `nat agent graveyard` | Failed hypotheses |
| `nat agent once` | Run single cycle (testing) |
| `nat agent queue` | Queued hypotheses by priority |
| `nat agent registry` | Validated signals |
| `nat agent report` | Full summary report |
| `nat agent start` | Launch agent daemon |
| `nat agent status` | Current state & stats |
| `nat agent stop` | Graceful shutdown (SIGTERM) |

## alg

| Command | Description |
|---|---|
| `nat alg` |  |
| `nat alg config` | Show algorithm configuration (from algorithms.toml) |
| `nat alg evaluate` | Run IC/drift evaluation |
| `nat alg list` | List registered algorithms + lifecycle state + IC |
| `nat alg ls` |  |

## alg1

| Command | Description |
|---|---|
| `nat alg1` | MF 3-feature liquidity signal (100min) |
| `nat alg1 live` | LIVE orders on Hyperliquid (requires HL_PRIVATE_KEY) |
| `nat alg1 paper` | Paper trader batch + watch |

## algorithm

| Command | Description |
|---|---|
| `nat algorithm` | Microstructure algorithm evaluation |
| `nat algorithm config` | Show algorithm configuration (from algorithms.toml) |
| `nat algorithm evaluate` | Run IC/drift evaluation |
| `nat algorithm list` | List registered algorithms + lifecycle state + IC |
| `nat algorithm ls` |  |

## alpha

| Command | Description |
|---|---|
| `nat alpha` | Alpha research pipeline (Steps 2-9) |
| `nat alpha combine` | Feature combination (Step 2) |
| `nat alpha deploy` | Deployment status & readiness (Step 9) |
| `nat alpha multi-freq` | Multi-frequency integration (Step 6) |
| `nat alpha paper` | Paper trading simulation (Step 8) |
| `nat alpha pipeline-gates` | Detailed gate report with metrics |
| `nat alpha pipeline-resume` | Resume alpha pipeline from last phase |
| `nat alpha pipeline-start` | Start fresh alpha pipeline run (all 9 steps) |
| `nat alpha pipeline-status` | Alpha pipeline state + gate verdicts |
| `nat alpha pipeline-step` | Run single alpha pipeline step (1-9) |
| `nat alpha portfolio` | Portfolio assembly (Step 7) |
| `nat alpha regime` | Regime conditioning (Step 5) |
| `nat alpha size` | Cost-aware position sizing (Step 3) |
| `nat alpha validate` | Walk-forward validation (Step 4) |

## api

| Command | Description |
|---|---|
| `nat api` | API & alert services |
| `nat api alerts` | Telegram alerts |
| `nat api serve-all` | Full stack in tmux |
| `nat api start` | Start API server |

## audit

| Command | Description |
|---|---|
| `nat audit` | Backtest audit and parameter sweep tools |
| `nat audit aggregate` | Aggregate walk-forward backtest results |
| `nat audit sweep` | Systematic parameter sweep across symbols/timeframes |

## backtest

| Command | Description |
|---|---|
| `nat backtest` | Backtesting |
| `nat backtest algorithm` | Backtest using algorithm features as signals |
| `nat backtest funding` | Funding rate reversion backtest |
| `nat backtest list` | List strategies |
| `nat backtest ml` | ML predictions backtest |
| `nat backtest ml-quantile` | ML quantile thresholds |
| `nat backtest ml-tracked` | ML with tracking |
| `nat backtest ml-validate` | ML walk-forward |
| `nat backtest validate` | Walk-forward validation |

## bridge

| Command | Description |
|---|---|
| `nat bridge` | Signal bridge daemon — executes LIVE signals under risk gating (T17) |
| `nat bridge once` | Run a single execution cycle |
| `nat bridge start` | Run the bridge daemon (foreground) |
| `nat bridge status` | Mode / halt state / LIVE signals |
| `nat bridge stop` | Graceful shutdown (SIGTERM) |

## build

| Command | Description |
|---|---|
| `nat build` | Build & dev tools (default: release) |
| `nat build api` | API server |
| `nat build check` | Check code |
| `nat build clean` | Remove artifacts |
| `nat build debug` | Debug binary |
| `nat build fmt` | Format code |
| `nat build lint` | Run clippy |

## cluster

| Command | Description |
|---|---|
| `nat cluster` | Cluster analysis |
| `nat cluster all` | Analyze all symbols |
| `nat cluster analyze` | Analyze cluster quality |
| `nat cluster explore` | Exploratory clustering (PCA/UMAP/t-SNE) |
| `nat cluster gmm` | Analyze with GMM |
| `nat cluster hmm-fit` | Fit HMM on feature data (Baum-Welch) |
| `nat cluster quality` | Q3 predictive quality test |

## commands

| Command | Description |
|---|---|
| `nat commands` | List all commands with descriptions |

## config

| Command | Description |
|---|---|
| `nat config` | Configuration inspection |
| `nat config get` | Get specific config value (file.section.key) |
| `nat config paths` | Show resolved data/config/log/report locations |
| `nat config show` | Full config dump (all TOML files merged) |
| `nat config validate` | Check all config files for syntax errors |

## daily

| Command | Description |
|---|---|
| `nat daily` | Daily 6-hour OOS snapshot for winning algorithms |

## dashboard

| Command | Description |
|---|---|
| `nat dashboard` | Start dashboard |

## data

| Command | Description |
|---|---|
| `nat data` | Data stats & validation |
| `nat data explore` | Launch Jupyter |
| `nat data ls` | List individual parquet files (path/size/rows/mtime) |
| `nat data schema` | Scan parquet schema |
| `nat data validate` | Validate parquet data (a directory or a single file) |

## deploy

| Command | Description |
|---|---|
| `nat deploy` | Deploy ingestor to remote host |
| `nat deploy cloud` | Deploy a redundant ingestor to a cloud box via the .deb + systemd (T0b) |
| `nat deploy rollback` | Rollback to previous binary |
| `nat deploy status` | Check remote ingestor status |

## discovery

| Command | Description |
|---|---|
| `nat discovery` | Alpha discovery orchestrator |
| `nat discovery once` | Run single sweep |
| `nat discovery start` | Launch discovery orchestrator |
| `nat discovery status` | Current state |
| `nat discovery stop` | Stop orchestrator |

## docker

| Command | Description |
|---|---|
| `nat docker` | Docker operations |
| `nat docker build` | Build images (nat docker build [--verbose] [services...]) |
| `nat docker down` | Stop services |
| `nat docker logs` | View logs (nat docker logs [services...]) |
| `nat docker ps` | Show running services |
| `nat docker smoke` | Quick health check of running stack |
| `nat docker stack` | Build + start + verify full stack |
| `nat docker up` | Start services (nat docker up [services...]) |

## doctor

| Command | Description |
|---|---|
| `nat doctor` | Ingestion preflight (data-dir ownership, binary, disk) |

## eamm

| Command | Description |
|---|---|
| `nat eamm` | EAMM market making |
| `nat eamm backtest` | Stateful backtest |
| `nat eamm regime` | Regime analysis |
| `nat eamm run` | Full EAMM pipeline |

## evolve

| Command | Description |
|---|---|
| `nat evolve` | Evolutionary config optimization (Optuna) |
| `nat evolve best` | Show best configs |
| `nat evolve export` | Export best config as TOML |
| `nat evolve pareto` | Pareto front (NSGA-II) |
| `nat evolve start` | Start evolutionary optimization |
| `nat evolve status` | Show study status |

## exp

| Command | Description |
|---|---|
| `nat exp` | Experiment runner |
| `nat exp analyze` | End-of-experiment |
| `nat exp check` | Daily validation |
| `nat exp dashboard` | Show dashboard URL |
| `nat exp midweek` | Full validation |
| `nat exp start` | Start ingestor in tmux |
| `nat exp status` | Health + data stats |
| `nat exp stop` | Stop ingestor |
| `nat exp tunnel` | Cloudflare tunnel |

## experiment

| Command | Description |
|---|---|
| `nat experiment` | Experiment tracking |
| `nat experiment best` | Find best |
| `nat experiment compare` | Compare experiments |
| `nat experiment get` | Get experiment details |
| `nat experiment list` | List experiments |
| `nat experiment snapshot` | Create dataset snapshot |
| `nat experiment workflow` | Full ML workflow |

## fetch

| Command | Description |
|---|---|
| `nat fetch` | Fetch historical data from exchanges |
| `nat fetch candles` | Fetch OHLCV candles from Hyperliquid |

## gap

| Command | Description |
|---|---|
| `nat gap` | Data-gap alert daemon (Telegram page on ingestion stall, T0b) |
| `nat gap check` | One-shot freshness check (exit 1 if gapping) |
| `nat gap start` | Run the gap-alert daemon (foreground) |
| `nat gap status` | Show current gap state |
| `nat gap stop` | Graceful shutdown (SIGTERM) |
| `nat gap test` | Send a REAL test page via Telegram (exit 1 unless delivered) |
| `nat gap watchdog` | Install cron watchdog (auto-restart every 5 min) |

## gauntlet

| Command | Description |
|---|---|
| `nat gauntlet` | Multi-day OOS sweep across all algorithms |
| `nat gauntlet report` | Print the latest gauntlet report |
| `nat gauntlet report_all` | Merge all gauntlet runs into combined summary |
| `nat gauntlet run` | Start the sweep |
| `nat gauntlet stop` | Stop running gauntlet, print partial results |

## health

| Command | Description |
|---|---|
| `nat health` | Comprehensive system health check (all components) |

## help

| Command | Description |
|---|---|
| `nat help` | Show full help |

## ing

| Command | Description |
|---|---|
| `nat ing` | Ingestor control (start/stop/status/log) |
| `nat ing log` | Tail the latest ingestor log |
| `nat ing start` | Start ingestor locally |
| `nat ing status` | Ingestor + data health |
| `nat ing stop` | Stop ingestor + daemons |

## it-engine

| Command | Description |
|---|---|
| `nat it-engine` | Information-theoretic alpha discovery engine |
| `nat it-engine start` | Start IT engine |
| `nat it-engine status` | Show IT engine status |
| `nat it-engine stop` | Stop IT engine |

## kalman

| Command | Description |
|---|---|
| `nat kalman` | Kalman filter research (OU filter IC + drift analysis) |
| `nat kalman analysis` | Phase 1: Kalman filter IC analysis |
| `nat kalman drift` | Phase 2: Drift analysis (latency-aware) |

## lifecycle

| Command | Description |
|---|---|
| `nat lifecycle` | Signal promotion lifecycle (DISCOVERED→LIVE→RETIRED) |
| `nat lifecycle approve` | APPROVAL_PENDING → LIVE (human gate) |
| `nat lifecycle history` | Transition history for one signal |
| `nat lifecycle list` | List signals |
| `nat lifecycle reject` | Reject a pre-LIVE signal |
| `nat lifecycle seed` | Seed deployable winners (idempotent) |
| `nat lifecycle status` | Count signals by state |

## log

| Command | Description |
|---|---|
| `nat log` | Tail ingestor log |
| `nat log agent` | Tail agent daemon log |
| `nat log list` | List all log files with sizes and dates |

## macro

| Command | Description |
|---|---|
| `nat macro` | Daily macro signals |

## macro-agent

| Command | Description |
|---|---|
| `nat macro-agent` | Macro research agent (1h-24h) |
| `nat macro-agent graveyard` | Failed hypotheses |
| `nat macro-agent once` | Run single cycle (testing) |
| `nat macro-agent queue` | Queued hypotheses by priority |
| `nat macro-agent registry` | Validated signals |
| `nat macro-agent report` | Full summary report |
| `nat macro-agent start` | Launch macro agent daemon |
| `nat macro-agent status` | Current state & stats |
| `nat macro-agent stop` | Graceful shutdown (SIGTERM) |

## mesh

| Command | Description |
|---|---|
| `nat mesh` |  |

## meta-agent

| Command | Description |
|---|---|
| `nat meta-agent` | Meta-agent orchestrator (cross-agent) |
| `nat meta-agent budget` | Agent budget allocation |
| `nat meta-agent correlation` | Cross-agent correlation matrix |
| `nat meta-agent once` | Run single orchestration cycle |
| `nat meta-agent portfolio` | Signal portfolio |
| `nat meta-agent report` | Full orchestrator report |
| `nat meta-agent start` | Launch meta-agent daemon |
| `nat meta-agent status` | Current state & budgets |
| `nat meta-agent stop` | Graceful shutdown (SIGTERM) |

## metrics

| Command | Description |
|---|---|
| `nat metrics` | Metric catalogue (IC, MI, TE, Sharpe, …) |
| `nat metrics list` |  |
| `nat metrics ls` | List all metrics (name/category/definition) |
| `nat metrics show` | Show one metric (formula + estimator docstring) |

## mf-agent

| Command | Description |
|---|---|
| `nat mf-agent` | Medium-frequency research agent (1min-1h) |
| `nat mf-agent graveyard` | Failed hypotheses with failure reasons |
| `nat mf-agent once` | Run single discovery cycle (for testing) |
| `nat mf-agent queue` | Pending hypotheses sorted by priority |
| `nat mf-agent registry` | Validated signals with IC and status |
| `nat mf-agent report` | Full summary: registry + graveyard + generator stats |
| `nat mf-agent start` | Launch MF agent daemon (cycles every 2h) |
| `nat mf-agent status` | Phase, cycle count, generator hit rates |
| `nat mf-agent stop` | Send SIGTERM to running daemon |

## model

| Command | Description |
|---|---|
| `nat model` | Model training & serving |
| `nat model list` | List models |
| `nat model score` | Score data |
| `nat model serve` | Model serving API |
| `nat model train` | Train baseline model |
| `nat model train-gmm` | Train GMM classifier |
| `nat model train-hier` | Train hierarchical signal combiner |

## monitor

| Command | Description |
|---|---|
| `nat monitor` | Live feature probe: stream computed features (no ingestion) |
| `nat monitor tui` | Legacy rich dashboard (Redis health/agent/features tabs) |

## nightly

| Command | Description |
|---|---|
| `nat nightly` | Overnight feature stats + algo performance report |
| `nat nightly open` | Open latest nightly HTML report |
| `nat nightly report` | Print latest nightly summary |

## oos

| Command | Description |
|---|---|
| `nat oos` | Longitudinal OOS validation over a trailing window of gauntlet P&L |

## oos30

| Command | Description |
|---|---|
| `nat oos30` | 30-day OOS validation for winning algorithms |

## package

| Command | Description |
|---|---|
| `nat package` | Build distributables (.deb) |
| `nat package deb` | Build the nat .deb (packaging/build_deb.sh) |

## pipeline

| Command | Description |
|---|---|
| `nat pipeline` | Automated pipeline |
| `nat pipeline analyze` | Analyze existing data |
| `nat pipeline dashboard` | Pipeline dashboard |
| `nat pipeline resume` | Resume from saved state |
| `nat pipeline start` | Start pipeline |
| `nat pipeline status` | Show state |
| `nat pipeline stop` | Stop pipeline |

## process

| Command | Description |
|---|---|
| `nat process` | Analytical processes (IC sweep, MI/TE, spectral, ML importance) |
| `nat process list` | List registered processes |
| `nat process results` | List past runs from the nat.db index |
| `nat process run` | Run a process on feature data |
| `nat process show` | Show one run record |
| `nat process standing` | Standing (recurring) evaluations: list \| audit \| run <name> |

## profile

| Command | Description |
|---|---|
| `nat profile` | Profiling |
| `nat profile scalp` | Scalping feature profiler (--symbol, --top, --forward-test) |

## promotion

| Command | Description |
|---|---|
| `nat promotion` | Signal promotion daemon (lifecycle automation, T14) |
| `nat promotion once` | Run a single promotion cycle |
| `nat promotion start` | Run the promotion daemon (foreground) |
| `nat promotion status` | Signals by state + clean-day guard |
| `nat promotion stop` | Graceful shutdown (SIGTERM) |

## report

| Command | Description |
|---|---|
| `nat report` | Experiment reports (generate / ls) |
| `nat report generate` | Generate the full experiment report |
| `nat report list` |  |
| `nat report ls` | List experiment-outcome artifacts |

## reports

| Command | Description |
|---|---|
| `nat reports` | Report management |
| `nat reports latest` | Most recent report per category |
| `nat reports show` | Print report content |

## risk

| Command | Description |
|---|---|
| `nat risk` | Kill-switch daemon + halt control (T16) |
| `nat risk resume` | Clear a halt (kill_strategy refused; halt_review/halt need --confirm) |
| `nat risk start` | Run the kill-switch daemon (foreground) |
| `nat risk status` | Show current halt state |
| `nat risk stop` | Graceful shutdown (SIGTERM) |

## run

| Command | Description |
|---|---|
| `nat run` | Run in foreground (default: ingestor) |
| `nat run serve` | Ingestor + dashboard |
| `nat run show` | Real-time features |
| `nat run tunnel` | Cloudflare tunnel |

## scan

| Command | Description |
|---|---|
| `nat scan` | Scalp edge scanner |

## screen

| Command | Description |
|---|---|
| `nat screen` | Alpha screening |

## service

| Command | Description |
|---|---|
| `nat service` | systemd --user supervision: reboot-proof ingestor + gap daemon |
| `nat service install` | Install + enable units (replaces tmux+cron; brief ingestor restart) |
| `nat service restart` | Restart a unit |
| `nat service status` | Unit active/enabled + linger state |
| `nat service uninstall` | Remove units, restore the tmux+cron path |

## signal

| Command | Description |
|---|---|
| `nat signal` | Signal testing |
| `nat signal test` | Signal existence test |
| `nat signal test-all` | Full symbol sweep |

## spannung

| Command | Description |
|---|---|
| `nat spannung` | Spannung signal grid search |
| `nat spannung backtest` | Cost-aware backtest + regime gating |
| `nat spannung horizon` | Longer-horizon sweep (30s–15min bars) |
| `nat spannung regime` | Systematic regime condition screener |
| `nat spannung spectral` | Spectral analysis (PSD, coherence, ACF, band IC) |

## start

| Command | Description |
|---|---|
| `nat start` | Start ingestor + watchdog + dashboard |

## status

| Command | Description |
|---|---|
| `nat status` | Health check |

## stop

| Command | Description |
|---|---|
| `nat stop` | Stop everything |

## swarm

| Command | Description |
|---|---|
| `nat swarm` | Parameter sweep optimization |
| `nat swarm best` | Export best config as TOML |
| `nat swarm generate` | Generate configs only (no evaluation) |
| `nat swarm results` | Show top configs ranked by Sharpe |
| `nat swarm run` | Generate configs and evaluate in parallel |
| `nat swarm status` | Show swarm run status |

## test

| Command | Description |
|---|---|
| `nat test` | Testing (default: all Rust unit tests) |
| `nat test 15m` | Capture 15m of live data then visualize |
| `nat test 1m` | Capture 1m of live data then visualize |
| `nat test 5m` | Capture 5m of live data then visualize |
| `nat test api` | Test API endpoints |
| `nat test backtest` | Backtest tests |
| `nat test cluster` | Cluster quality tests |
| `nat test dashboard` | Dashboard tests |
| `nat test eamm` | EAMM tests |
| `nat test hypotheses` | Hypothesis tests |
| `nat test integration` | Integration tests |
| `nat test pipeline` | Pipeline tests |
| `nat test pipeline-runner` | Pipeline runner tests |
| `nat test process` | Process framework tests (synthetic + real-data smoke) |
| `nat test redis` | Test Redis |
| `nat test regression` | Run algorithms and compare against baseline |
| `nat test scan` | Scalp edge scanner tests |
| `nat test serving` | Model serving tests |
| `nat test snapshot` | Capture data snapshot + save algorithm baseline |
| `nat test unit` | Rust unit tests |
| `nat test validate` | Live API validations |
| `nat test verbose` | Tests with --nocapture |
| `nat test viz` | Scanner visualization tests |

## tournament

| Command | Description |
|---|---|
| `nat tournament` | Continuous algorithm testing engine |
| `nat tournament compare` | Head-to-head comparison of two algorithms |
| `nat tournament history` | Per-day history for one algorithm |
| `nat tournament rankings` | Current leaderboard |
| `nat tournament report` | Generate markdown report |
| `nat tournament run` | Run a single evaluation cycle |
| `nat tournament start` | Start background daemon |
| `nat tournament status` | Show daemon state and DB stats |
| `nat tournament stop` | Stop daemon |

## trade

| Command | Description |
|---|---|
| `nat trade` | Paper trade visualization |
| `nat trade viz` | Visualize paper trades (snapshot PNG) |

## validate

| Command | Description |
|---|---|
| `nat validate` | Validation suites |
| `nat validate regression` | Skeptical regression signal test battery (10 tests) |
| `nat validate skeptical` | 20+ statistical tests before investment |

## visualize

| Command | Description |
|---|---|
| `nat visualize` | Visualization suite |
| `nat visualize all` | All visualizations |
| `nat visualize cluster` | Cluster exploration plots (PCA/UMAP/t-SNE) |
| `nat visualize data` | Data quality plots (8-10) |
| `nat visualize hierarchy` | Hierarchical profiling plots (Phase 8) |
| `nat visualize profile` | Cluster profiling plots |
| `nat visualize scan` | Scanner plots (1-7) |
| `nat visualize skeptical` | Skeptical validation diagnostic plots |

## viz

| Command | Description |
|---|---|
| `nat viz` | Terminal-first visualization (features/algorithm/paper) |
| `nat viz algorithm` | Algorithm signal timeline, IC, entry/exit, P&L proxy |
| `nat viz features` | Per-feature overview (value/z/NaN%/IC/sparkline) |
| `nat viz file` | Curated snapshot of one parquet file → show → delete |
| `nat viz paper` | Approval evidence: P&L, IC decay, G8 scorecard (NAT6) |
| `nat viz portfolio` | Portfolio P&L / exposure / correlation / risk (NAT7) |
| `nat viz predictability` | PROC-8 surface: combo×horizon×label×regime MI, FDR-corrected |
| `nat viz render` | Paged PNG viewer at 1m/5m/15m (overview, or page INDEX) |

## viz3d

| Command | Description |
|---|---|
| `nat viz3d` | Interactive 3D feature-surface-over-time (Plotly HTML) |

## wallets

| Command | Description |
|---|---|
| `nat wallets` | On-chain wallet layer (WP-1..5) [PRELIM] |
| `nat wallets panel` | Accrual status of the position panel (WP-2 clock) [PRELIM] |
| `nat wallets positions` | One position sweep across the roster (WP-2) [PRELIM] |
| `nat wallets roster` | Derive the wallet roster from the leaderboard (WP-1) [PRELIM] |

## xs

| Command | Description |
|---|---|
| `nat xs` | Class-3 cross-sectional layer [PRELIM] |
| `nat xs capacity` | Tradability curve (XS-5) [PRELIM] |
| `nat xs ledger` | Program multiple-testing ledger (PROC-13) |
| `nat xs persistence` | Rank autocorrelation half-life (XS-4) [PRELIM] |
| `nat xs rank` | Rank-IC vs relative forward returns (XS-3) [PRELIM] |
| `nat xs trajectory` | Rotation t-stat trajectory (XS-10) [PRELIM] |
| `nat xs universe` | Candle archive + L2 sampler coverage [PRELIM] |

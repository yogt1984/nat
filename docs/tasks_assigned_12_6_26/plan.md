# NAT Action Plan — Sequential Implementation & Test Order

**Date:** 2026-06-12 (rev. 3 — condensed to sequential task list, ≤200 lines)
**Sources:** `Q/*.md`, `P/*.md`, `nat_cli_tasks/*.md`, `data_inventory.md`, `situation_analysis.md`,
`feature_algorithm_gaps.md`, `algorithm_classification.md`, `algorithm_candidates_literature.md`,
`phd_vs_quant_roadmap.md`. Detailed network design: git history of this file (rev. 2).

## Hard Constraints (violating any of these is a bug in execution, not a judgment call)

1. **Jun 11–17 accumulation streak (Q1.1):** zero ingestor contact until the 7-day streak completes.
   Docker work targets new services *by name* — never a bare `docker compose up`.
2. **Gates are imported, not invented:** VALIDATED = G4 (walk-forward + deflated Sharpe), paper exit = G8
   (14 days, 5 criteria), kill-switch thresholds = ROADMAP Step 9. No new thresholds.
3. **All costs via `load_costs()`** from `config/costs.toml` — OOS, paper, and live must agree.
4. **No live capital** before G8 passes and the kill-switch daemon is healthy.

---

## Stage A — Jun 12–17 (pure Python, zero ingestor risk)

### T1. Verify NaN-wiring fixes on live data (Q1.2 / K2)
Whale flow, position tracker, wallet discovery were wired in commit 38aa7e1. Confirm the 82 dead
columns start filling: run feature NaN-rate check on today's parquet; start the 48h concentration
viability clock (01_concentration doc — verdict: viable if 20+ wallets / >5% OI coverage, else keep NaN).
**Test:** NaN% report shows whale/liquidation columns < 100% NaN on Jun 12+ dates; no schema change (236 cols).

### T2. Shared foundations: costs + provenance (Q1.4 + P1.5, ~1 day)
Create `scripts/costs.py` (`load_costs(exchange)` reading `config/costs.toml`) and `scripts/provenance.py`
(`get_provenance()` git SHA + `data_fingerprint()` SHA-256 of parquet inputs). Complete Hyperliquid/Binance
entries in `config/costs.toml`. Integrate provenance into `research_output.py` and walk-forward results.
**Test:** fingerprint deterministic (same parquet set → same hash); `load_costs("hyperliquid")` reproduces
current `nat oos30` numbers exactly (regression); `nat test agent` passes.

### T3. Signal lifecycle core (~2 days)
Add `signal_lifecycle` + `lifecycle_history` tables to `nat.db` via the existing `_run_migrations()`
discipline in `scripts/data/state.py` (coordinate with Q1.3 — one migration framework only). Create
`scripts/signal_lifecycle.py` API: discover/validate/start_paper/request_approval/approve/reject/retire,
provenance stamped on every insert and transition. States: DISCOVERED → VALIDATED → PAPER_TRADING →
APPROVAL_PENDING → LIVE → MONITORING → RETIRED (+ REJECTED), transitions enforced.
**Test:** scripted end-to-end transition walk passes; illegal transition raises; every row carries git_sha.

### T4. Lifecycle CLI + seeding (~0.5 day)
Add `nat lifecycle status|list|approve|reject|history`. `approve` prints the G8 scorecard + OOS metrics +
provenance before confirming (sole human gate). Seed: jump_detector, optimal_entry, funding_reversion,
3f_liquidity at VALIDATED; hierarchical_combiner + mean_reversion_detector at DISCOVERED.
**Test:** `nat lifecycle status` shows 4 VALIDATED / 2 DISCOVERED; `nat lifecycle history <id>` shows seed
transition with git_sha.

### T5. Agent → lifecycle integration (~0.5 day)
In `scripts/agent/base.py:register_signal()` (after line 324), insert into lifecycle as DISCOVERED
(~10 lines, non-fatal on failure).
**Test:** `nat agent once` → new signal appears in `nat lifecycle list --state DISCOVERED`.

### T6. CLI quick wins: NAT1 + NAT2 (~5h)
NAT1: `nat help --grep <term>` — case-insensitive search over 251 command names + help text.
NAT2: curated group-level help for all 20+ groups (`nat alpha` prints scoped help, not argparse usage).
New groups from T4 (`lifecycle`, `risk`) follow the NAT2 pattern from day one.
**Test:** `nat help --grep kalman` finds kalman commands; `nat alpha` shows 9-step pipeline help;
`nat alpha combine -h` unchanged.

### T7. Viz foundation: NAT3 (~6h)
`scripts/viz/common.py` (load_features/load_algorithm_signals/load_paper_trades, theme) and
`scripts/viz/terminal.py` (sparkline, ic_color, live_refresh). Register `nat viz` group.
Blocks NAT4–NAT8 — do before any viz command.
**Test:** `load_features('BTC', hours=1)` returns DataFrame; `sparkline([1,2,3])` returns Unicode; no new deps.

### T8. Feature gaps F1–F5 (~17h total, Python only)
In priority order: F1 settlement-clock (2h, unlocks LF1), F2 microprice deviation (4h), F3 integrated
multi-level OFI (6h), F4 HAR-RV (3h), F5 realized moments (2h). Each follows the algorithm-feature
contract (`alg_` prefix, NaN-safe, registered, configurable via `config/algorithms.toml`).
**Test per feature:** `pytest scripts/tests/test_algorithm_smoke.py -k <name>` + spot-check values on
real parquet (smoke-test rule) before commit.

### T9. New algorithm candidates, literature priority order (~25h, interleave with T8)
1. **HF4 VPIN gate** (5h) — flow-toxicity filter upgrading all 4 deployed winners.
2. **LF1 funding-settlement** (6h) — needs F1; best-replicated family in the literature.
3. **HF1 microprice fair-value** (6h) — needs F2; maker-side anchor.
4. **HF2 integrated OFI** (8h) — needs F3; redeems failed weighted_ofi.
Defer: HF3 Hawkes (10h), LF3 liquidation cascade (gated on T1 whale data), A1 ETH/SOL ratio (6h).
**Test per algo:** smoke test + `nat algorithm evaluate --algorithm <name> --symbol BTC`; walk-forward
OOS before any deployable claim (momentum_continuation overfit is the cautionary tale).

### T10. Decision-trace viz: NAT4 + NAT5 (~14h)
`nat viz features` (per-feature value/z-score/NaN%/IC/sparkline, `--alive-only`, `--live`) — directly
verifies T1. `nat viz algorithm <name>` (signal timeline, entry/exit markers, features at trigger, P&L).
**Test:** `nat viz features --symbol BTC --alive-only` < 3s; `nat viz algorithm jump_detector --symbol BTC`
renders 4h trace; both support `--json` and `--output <png>`.

---

## Stage B — Jun 17–24 (streak complete → first real validation data)

### T11. Q2.1 hierarchical revalidation + Q2.2 alpha screen with FDR
Rerun hierarchical_combiner on the 7-day clean dataset (its 2-day monotone-IC folds are unverified).
Run the alpha screen with BH-FDR (G1) across the 154 live features + new T8 features.
**Test:** G1 gate report generated; combiner transitions DISCOVERED → VALIDATED in lifecycle *only* if
G4 criteria pass on ≥7 clean days; otherwise REJECTED with reason recorded.

### T12. Dockerize existing agents (~2 days)
`docker/Dockerfile.agent` (shared Python image) + compose services: agent-micro, agent-mf, agent-macro,
meta-agent. Named-service invocations only.
**Test:** `docker compose up agent-micro` completes a research cycle; meta-agent allocates budget;
ingestor container untouched (verify streak via daily data check).

### T13. Daily agent + candle fetcher (~3 days)
`get_candles()` on HyperliquidClient (~20 lines); `scripts/candle_daemon.py` (~80 lines, 1m/5m/15m/1h
candles → `data/candles/`); `DailyAgent` thin subclass at 1–7d horizons (strictly above macro's 1h–24h),
3 generators (momentum, mean-reversion/funding-cycle, cross-asset using `data/macro/`). Generators
validate feature availability at init (skip still-dead columns). Register in meta-agent + compose.
**Test:** candle parquet grows over 10 min; `nat daily-agent once` completes a cycle. Expect gate passes
to be rare before ~Aug 1 (data sufficiency) — that is correct behavior, do not loosen gates.

---

## Stage C — Jun 24 – Jul 8 (automation + risk layer)

### T14. Promotion daemon (~3 days)
`scripts/promotion_daemon.py`: poll 300s; data-sufficiency guard (≥7 clean days) before any G4 run;
DISCOVERED→VALIDATED via `oos_validator.py` subprocess (G4, load_costs); VALIDATED→PAPER_TRADING via
`paper_trader_generic.py` subprocess; PAPER→APPROVAL_PENDING after **14 days** on all 5 G8 criteria;
LIVE decay/kill checks → RETIRED. `[promotion]` config section; reconcile `[agent.promotion].paper_days=7`
vs G8=14. Compose service + `nat promotion status`.
**Test:** seeded VALIDATED signal auto-starts paper; insufficient-data case refuses OOS with logged reason;
subprocess timeout (600s) skips cleanly; provenance stamped at each transition.

### T15. Approval-evidence viz: NAT6 + NAT7 (~12h — must land before first APPROVAL_PENDING)
`nat viz paper` (cumulative P&L, IC decay, G8 checklist PASS/FAIL) and `nat viz portfolio` (4 tabs:
P&L / exposure / correlation / risk).
**Test:** `nat viz paper` shows G8 scorecard on live paper data; `nat viz portfolio --tab 3` shows
cross-algo correlation < 0.35 for the seeded winners; graceful no-data message.

### T16. Kill-switch daemon (Q3.6, ~1.5 days — ships BEFORE bridge)
`scripts/risk/kill_switch.py`: poll PnL 60s; thresholds — daily loss >1% → halt_24h, weekly DD >2% →
halt_review, monthly DD >5% → kill_strategy, IC<0 for 5d → halt. Writes `data/risk/halt_state.json`;
Telegram alert <60s; Prometheus metrics; `kill_strategy` also retires the signal in lifecycle.
CLI: `nat risk status|resume [--confirm]` (refuses to clear kill_strategy without pipeline re-run).
Compose service with healthcheck.
**Test:** synthetic PnL breach for each of the 4 thresholds triggers correctly; Telegram fires; resume
rules enforced — all verified during paper, before any live capital.

### T17. Signal bridge daemon mode (~1.5 days)
Add `run_daemon()` to `signal_bridge.py`: reads LIVE signals from lifecycle; checks halt_state.json
before every cycle (cannot be skipped); portfolio sizing via `meta_portfolio.py` risk parity (never
independent sizing); fill logging to `data/execution/fills_*.jsonl` for fill-conditional IC (Q3.5);
daily P&L rollup. Compose: bridge `depends_on` kill-switch healthy.
**Test:** `nat lifecycle approve <id>` → bridge picks signal up in dry-run; synthetic halt → bridge
skips cycle; fills JSONL populates.

---

## Stage D — Jul 8 onward (observability, paper window, polish)

### T18. Monitoring + E2E (~3 days)
Prometheus metrics on all Python services; 3 Grafana dashboards (lifecycle funnel, paper performance,
live P&L); lifecycle tab in agent dashboard; full `docker compose up`; cloud deployment doc.
**Test:** all services healthy; E2E: ingestion → discovery → OOS → paper → approve → dry-run trade.

### T19. 14-day paper window (Q3.2 / G8) — calendar-bound, starts as soon as T14 promotes
Network runs unattended; daily reconciliation. First APPROVAL_PENDING expected ~Aug.
**Test:** G8 scorecard via `nat viz paper`; D3 check: paper Sharpe within 2× backtest.

### T20. CLI polish: NAT8 + NAT9, then NAT10 last
NAT8 (~4h): `nat viz spectral|regime|correlation` wrappers. NAT9 (~3h): maturity tags
[LIVE/PROVEN/BETA/PRELIM/SPEC] on all commands + `nat visualize`/`trade-viz` aliases. NAT10 (~12h):
split the 5,113-line `nat` into `scripts/cli/` modules — only after NAT1–9 stabilize.
**Test:** NAT9 — every command tagged, `nat commands --json` has `maturity`; NAT10 — command count
identical, all `nat test*` suites pass, main script <400 lines.

### T21. Live deployment (Q4.1) — Sep–Oct, human-gated
First `nat lifecycle approve` → LIVE at 1% capital, maker-only; tier scale-up 1%→25% over 4+ months
per Q4 tier gates. Requires T16 healthy + G8 passed. **Test:** fill-conditional IC ratio ≥0.8 vs paper.

---

## Parallel P-track (writing, not code — fills analyst time while data accumulates)

- **P1 preprint** (Jun–Aug, ~40h): SVD kernels → spectral section (IC 0.45 ultra-low band) → regime
  gating (0.45→0.67) → cross-symbol validation → assembly with provenance appendix (uses T2).
  Check: LaTeX compiles, all IC/Hurst/OU values match source analyses, BH-FDR applied.
- **P2 SSRN + arXiv** (Sep, ~4h): SSRN upload + e-journals; arXiv q-fin.TR endorsement + cross-post.
- **P3 professor outreach** (Sep–Nov, ~12h): 5 Tier-1 (ETH/EPFL) then 8 Tier-2 staggered 2 weeks;
  target ≥2 active by Nov. Gate for P4.
- **P4 applications** (Dec–Mar, ~20h): EPFL EDFI Round 1 by Jan 15 2027; ETH Jan–Feb; offer Apr–May.

## Calendar

| When | Quant/infra | Milestone |
|------|-------------|-----------|
| Jun 12–17 | T1–T10 (Stage A) | Streak completes Jun 17 — touch nothing on su-35 |
| Jun 17–24 | T11–T13 | Combiner verdict; ~Jun 20: 30 good dates → OOS30 feasible |
| Jun 24–Jul 8 | T14–T17 | Promotion + risk layer live |
| Jul 8–Aug | T18–T19, P1 | First G8 windows; D1 decision (Aug) |
| Sep–Oct | T20–T21, P2–P3 | First LIVE at 1% (Q4.1); D3 paper-vs-backtest |
| Nov–Mar 2027 | scale-up, P4 | D2 professor interest (Nov); PhD offer target Apr–May 2027 |

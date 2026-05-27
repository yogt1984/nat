# Research Conventions

Rules and workflows for development on NAT. Read alongside `CLAUDE.md` (architecture) and `FEATURES.md` (feature manifest).

---

## Git Workflow

- **Branching:** `feat/<name>`, `fix/<name>`, `refactor/<name>` off master
- **Commits:** conventional commits — `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
- **Merge:** `git merge --no-ff feat/<name>` to master with descriptive merge message
- **No PRs:** solo repo, direct merge workflow
- **Push both:** feature branch + master after merge

## Verification Requirements

Every behavior change must include a verification step before committing:

| Change type | Verification |
|-------------|-------------|
| Rust ingestor | `cargo test --package ing` + `make validate` (live API) |
| Python algorithm | `pytest scripts/tests/test_algorithm_smoke.py -k <name>` |
| Python agent/daemon | `pytest scripts/tests/` + smoke on real parquet (`--dry-run` on one day of `data/features/`) |
| Alpha pipeline step | `nat alpha_pipeline_step N` on real data |
| API endpoint | `make test_api` + `make test_integration` |
| Frontend component | `cd web && npm test` |
| Config change | `nat config validate` |

**Real data before commit:** Unit tests with synthetic data are necessary but not sufficient. Always run on real parquet from `data/features/` before committing new subsystems. Zero-variance features, NaN-heavy columns, and degenerate matrices only surface on real data.

## Research Quality Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Sharpe ratio | > 1.5 | Deployable (paper trading eligible) |
| Spearman IC | > 0.10 | Worth investigating |
| IC (gated) | > 0.15 | Worth hypothesis testing |
| dIC (gated - ungated) | > 0.05 | Gate adds value |
| Gross return/trade | > 0.1 bps | Survives cost gate |
| Max correlation (ρ) | < 0.7 | Not redundant with registry |
| FDR q-value | 0.05 | Benjamini-Hochberg threshold |
| IC decay ratio | < 0.5 for 14 days | Auto-retire signal |
| Paper-to-live Sharpe | > 1.5 (7-day) | Promotion threshold |
| Paper-to-live IC | > 0.8 (realized/predicted) | Promotion threshold |
| Max drawdown | < 2% | Promotion threshold |

## Fee Models

| Venue | Taker (bps) | Maker (bps) | Use case |
|-------|-------------|-------------|----------|
| Binance VIP9 | 1.61 | 0.20 | Paper trading default |
| Binance VIP0 | 3.50 | — | Conservative backtest |
| Hyperliquid | 7.00 | 0.30 | Robustness check |

Default for paper trading: 1.61 bps round-trip (Binance VIP9 taker). Backtest engine uses `CostModel` from `scripts/utils/cost_model.py` — always use this, never hardcode fees.

## CLI Usage

- **Production ingestion:** `nat start` (tmux + watchdog + dashboard), not `make run` (foreground only)
- **Monitoring:** `nat log`, `nat status`, `nat dashboard`
- **Research:** `nat spannung`, `nat profile scalp`, `nat scan`
- **Agents:** `nat agent start/stop/once/status/report`
- **OOS validation:** `nat oos30` (runs all 5 winning algorithms)
- **Direct scripts:** only for development/debugging. Production uses `nat` CLI.

## Machine Topology

| Machine | Role | Runs |
|---------|------|------|
| **su-35** | Data collection | `nat start` (Rust ingestor), writes to `data/features/` |
| **research** | Analysis | Python agents, `nat oos30`, backtests, dashboards |

Data flows su-35 → research machine via shared storage or rsync. Never run heavy Python analysis on su-35 — it's dedicated to low-latency ingestion.

## Algorithm Development Checklist

When adding a new microstructure algorithm:

1. Create `scripts/algorithms/<name>.py`
2. Implement `MicrostructureAlgorithm` ABC (see contract below)
3. Include docstring with mathematical formulation
4. Add `@register` decorator
5. Add parameters to `config/algorithms.toml`
6. Verify: `pytest scripts/tests/test_algorithm_smoke.py -k <name>`
7. Evaluate: `nat algorithm evaluate --algorithm <name> --symbol BTC`
8. If promising (IC > 0.10): add to `paper_trader_generic.py` ALGO_CONFIG

## Paper Trading Protocol

- **Bar resolution:** 5min (300s)
- **Horizon:** 20 bars (100min)
- **Training window:** 3-day rolling
- **Entry:** P20/P80 percentile z-score thresholds
- **Exit:** fixed horizon (no early exit)
- **Walk-forward:** train on W days, test on day W+1, slide forward
- **Minimum data:** 12 bars/date (60 min) to include a date

## Naming Conventions

- **Features:** `category_name` (e.g., `raw_midprice`, `ent_book_shape`, `illiq_kyle_100`)
- **Algorithm features:** `alg_` prefix (e.g., `alg_jump_statistic`, `alg_entry_signal`)
- **Config sections:** `[agent]`, `[agent_mf]`, `[agent_macro]`, `[meta_agent]`
- **State files:** `*_state.json` in `data/` subdirectories
- **Reports:** `reports/<category>/<name>_{symbol}.json`

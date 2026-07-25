# `in_progress/` — Information-Content Index

Classification of the 89 docs in this folder by **information content**, read epistemically:
how much a file reduces uncertainty about an *external truth* (the data, the market, the code)
vs. how much is a record of *intent* (prescriptive, carries no discovery). Three axes:

- **Tier** — `A` Empirical findings (posterior-updating) · `B` Reference/theory (timeless method) ·
  `C` Design/architecture spec (blueprint) · `D` Directive/task-spec (prescriptive intent).
  Information content is highest at A, lowest at D.
- **Track** — `Method` (IT engine, convolver, IC research) · `Q` (production/trading) ·
  `P` (academic/publication) · `Infra` (cloud, CLI, tooling).
- **Lifecycle** — `Realized` (measured/built/known) · `Gated-Jun17` (blocked on the 7-day data
  streak) · `Forward` (forward build, not yet done).

**IT punchline:** joint information ≪ Σ per-file. Tier D is highly redundant — `phd_vs_quant_roadmap`
→ `P/`+`Q/` → `plan.md` re-encode the same decisions; `convolver_implementation.txt` ≈
`…_tasks.md` ≈ `convolver_implementation/01–14`. As signals, **rho_bar is high in D, low in A/B**,
so the effective count of *independent* documents N_eff ≪ 89. The genuinely uncorrelated
information lives in **A (findings)** and **B (references)**; the rest is breadth-via-restatement.

## Summary

| Tier | Files | Meaning |
|------|-------|---------|
| A — Empirical findings | 10 | Tell you something new about the world |
| B — Reference / theory | 9 | How to measure (incl. 3 LaTeX build artifacts) |
| C — Design / architecture | 33 | Blueprints for systems to build |
| D — Directive / task-specs | 37 | Plans & instructions (high redundancy) |
| **Total** | **89** | |

| Track | Files |
|-------|-------|
| Research-method | 32 |
| Q-production | 22 |
| Infra-tooling | 28 |
| P-academic | 5 |
| P+Q (cross) | 2 |

---

## Tier A — Empirical findings (10) · all Realized

| File | Track | Content |
|------|-------|---------|
| `research/new/9_6/full_ic_scan_report.md` | Method | Full IC scan, 236 feats × 3 symbols × 6 horizons; 8 independent signal axes. |
| `research/new/9_6/ic_horizon_analysis.md` | Method | IC-vs-horizon decay (1s–15m); order-book imbalance IC≈0.45 @1–5s. |
| `research/new/9_6/ic_validation_report.md` | Method | Robustness 7/8 checks; **key finding: conditional IC→0 under mid-cross fills (adverse selection is structural).** |
| `research/new/10_6/hierarchical_combiner_report.md` | Q | 3-layer combiner OOS IC +0.18/0.25/0.36; directional gating works (honest 2-day caveat). |
| `tasks_assigned_12_6_26/features_report.md` | Method | 236-feature catalogue + IC scan (21 categories). |
| `tasks_assigned_12_6_26/algorithms_report.md` | Q | 26-algorithm catalogue + OOS walk-forward results, tiers. |
| `tasks_assigned_12_6_26/algorithm_classification.md` | Q | 26 algos × logic-family × horizon matrix (+12 lit candidates). |
| `tasks_assigned_12_6_26/situation_analysis.md` | Q | Deployable algos vs failed strategies + data-quality audit (2-day). |
| `tasks_assigned_12_6_26/data_inventory.md` | Q | Parquet storage audit: 9.4 GB, 20 gaps, 7-day target by Jun 17. |
| `korrektur_tasks.md` | Infra | 6 audited ingestor defects (volume-mount data loss, 56 dead feats, WS gaps). Findings-driven. |

## Tier B — Reference / theory (9) · all Realized

| File | Track | Content |
|------|-------|---------|
| `research/it_engine_mathematical_foundations.md` | Method | KSG MI/CMI/TE/interaction-info, greedy selection, rate-distortion cost gate (derivations, 34 KB). |
| `IC_related/ic_reference.tex` | Method | Canonical IC reference — source (definitions, variants, 5-gate protocol, thresholds). |
| `IC_related/ic_reference.txt` | Method | Plaintext extract of the IC reference. |
| `IC_related/ic_reference.pdf` | Method | Compiled IC reference. |
| `research/new/convolver_data_analysis.txt` | Method | Data-volume/regime-coverage arithmetic (60s feasible @month; regime-robust needs 6–24 mo). |
| `tasks_assigned_12_6_26/process_mi_targets_derivatives.md` | Method | MI lineage + target taxonomy (theory + process refinements). |
| `IC_related/ic_reference.aux` | Method | LaTeX build artifact — *no information content*. |
| `IC_related/ic_reference.log` | Method | LaTeX build log — *no information content*. |
| `IC_related/ic_reference.toc` | Method | LaTeX table-of-contents artifact — *no information content*. |

## Tier C — Design / architecture specs (33)

*Process framework (Method):*

| File | Track | Lifecycle | Content |
|------|-------|-----------|---------|
| `tasks_assigned_12_6_26/process_concept.md` | Method | Realized | Process as 3rd first-class citizen; Stage 1+2 implemented (contract/registry/CLI). |
| `tasks_assigned_12_6_26/process_signal_design.md` | Method | Forward | 9-process set S1–S9 for uncorrelated, info-theoretic signal extraction. |

*Convolver pipeline — OHLCV→SVD pattern discovery, 14 stage specs (Method, Forward; redundant with the Tier-D convolver build checklists):*

| File | Content |
|------|---------|
| `convolver_implementation/01_pipeline_overview.md` | Stage 0–8 schematic + anti-overfit architecture. |
| `convolver_implementation/02_candle_decomposition.md` | Body/wick/volume decomposition. |
| `convolver_implementation/03_atr_normalization.md` | ATR normalization. |
| `convolver_implementation/04_event_breakouts.md` | Breakout event detection. |
| `convolver_implementation/05_event_turtle_soup.md` | Turtle-soup event detection. |
| `convolver_implementation/06_event_traps.md` | Trap event detection (rate-limiting stage). |
| `convolver_implementation/07_matrix_assembly.md` | Pattern matrix assembly. |
| `convolver_implementation/08_svd_decomposition.md` | SVD decomposition. |
| `convolver_implementation/09_ic_gate.md` | IC gate. |
| `convolver_implementation/10_bh_fdr_correction.md` | BH–FDR correction. |
| `convolver_implementation/11_kernel_persistence.md` | Kernel library persistence. |
| `convolver_implementation/12_analytical_basis.md` | Analytical-basis alignment. |
| `convolver_implementation/13_walk_forward_validation.md` | Walk-forward validation. |
| `convolver_implementation/14_online_scoring.md` | Online scoring. |

*Cloud/swarm infra — Tier1-3 build specs (Infra, Realized/built):*

| File | Content |
|------|---------|
| `cloud_deployment/0_overview.md` | Three-tier cloud architecture overview (Observe→Swarm→Evolve). |
| `cloud_deployment/1_3_production_hardening.md` | Caddy HTTPS + Postgres + backups (Tier 1). |
| `cloud_deployment/1_4_testing_verification.md` | Docker/endpoint smoke tests (Tier 1). |
| `cloud_deployment/2_1_shared_ingestor.md` | 1 ingestor → N evaluators via shared Parquet. |
| `cloud_deployment/2_2_config_generator.md` | 35D param-space TOML generator. |
| `cloud_deployment/2_3_evaluator_worker.md` | Evaluator worker (load→algos→fitness→SQLite). |
| `cloud_deployment/2_4_swarm_orchestrator.md` | Swarm orchestration + ranking + Grafana. |
| `cloud_deployment/3_1_optuna_setup.md` | Optuna CMA-ES/TPE backend. |
| `cloud_deployment/3_2_fitness_function.md` | Multi-objective fitness (Sharpe/DD/IC), walk-forward. |
| `cloud_deployment/3_3_guard_rails.md` | Anti-overfit guard rails (deflated Sharpe, IS/OOS). |

*NaN feature wiring — activate 56 dead features (Infra):*

| File | Lifecycle | Content |
|------|-----------|---------|
| `nan_wiring/01_whale_flow_trade_classification.md` | Forward | Whale-flow trade classification (12 feats). |
| `nan_wiring/02_position_tracker_config.md` | Forward | Shared position-tracker config + state. |
| `nan_wiring/03_spawn_tracker_wire_features.md` | Forward | Wire tracker → 40 feats (liquidation + concentration). |
| `nan_wiring/04_wallet_discovery.md` | Forward | Auto wallet discovery from trade stream. |
| `nan_wiring/05_concentration_viability.md` | Gated-Jun17 | Go/no-go for concentration feats (needs 48h OI coverage). |

*Gap / literature (Q, Forward):*

| File | Content |
|------|---------|
| `tasks_assigned_12_6_26/feature_algorithm_gaps.md` | Missing features (F1–F9) + algos (A1–A5), priority-ranked. |
| `tasks_assigned_12_6_26/algorithm_candidates_literature.md` | 12 literature algo candidates (HF1–6, LF1–6) specs. |

## Tier D — Directive / task-specs & plans (37)

*Master plans & meta (high cross-redundancy):*

| File | Track | Lifecycle | Content |
|------|-------|-----------|---------|
| `tasks_assigned_12_6_26/plan.md` | P+Q | Forward | T0–T21 master sequencer reconciling P/Q under the data constraint. |
| `tasks_assigned_12_6_26/phd_vs_quant_roadmap.md` | P+Q | Forward | Dual-track milestone map (P1–P4 / Q1–Q4) with deadlines. |
| `research/new/9_6/project_state_report.md` | Q | Realized | Platform maturity/state (120K LOC; execution feasibility = the blocker). |
| `tasks_assigned_12_6_26/01_concentration_viability_assessment.md` | Q | Gated-Jun17 | Concentration gate decision matrix (pending data). |
| `tasks_assigned_12_6_26/nat_cli_improvement_plan.md` | Infra | Forward | Parent plan for the NAT1–10 CLI/viz tasks. |
| `test_plan.md` | Infra | Forward | Manual command/UI validation checklist (260 cmds, 7 viz types). |

*P-track — academic (P-academic, Forward):*

| File | Content |
|------|---------|
| `tasks_assigned_12_6_26/P/P1_1_preprint_completion.md` | Preprint completion (SVD convolver + spectral + regime + cross-symbol). |
| `tasks_assigned_12_6_26/P/P1_2_provenance_tracking.md` | Result provenance / reproducibility tracking. |
| `tasks_assigned_12_6_26/P/P2_1_publication_pipeline.md` | SSRN / arXiv / journal submission pipeline. |
| `tasks_assigned_12_6_26/P/P3_1_professor_outreach.md` | 13-target professor outreach. |
| `tasks_assigned_12_6_26/P/P4_1_application_submission.md` | PhD application submission (ETH / EPFL). |

*Q-track — production pipeline, gated G1–G8 (Q-production):*

| File | Lifecycle | Content |
|------|-----------|---------|
| `tasks_assigned_12_6_26/Q/Q1_1_data_accumulation.md` | Gated-Jun17 | 7-day data streak by Jun 17. |
| `tasks_assigned_12_6_26/Q/Q1_2_fix_dead_features.md` | Forward | Fix 56 dead features (position tracker) — see `nan_wiring/`. |
| `tasks_assigned_12_6_26/Q/Q1_3_sqlite_research_store.md` | Forward | SQLite research store. |
| `tasks_assigned_12_6_26/Q/Q1_4_cost_model_unification.md` | Forward | Unify cost model (single source of truth). |
| `tasks_assigned_12_6_26/Q/Q2_1_hierarchical_revalidation.md` | Gated-Jun17 | Revalidate hierarchical combiner on clean data. |
| `tasks_assigned_12_6_26/Q/Q2_2_alpha_screen_fdr.md` | Forward | Alpha screen with FDR. |
| `tasks_assigned_12_6_26/Q/Q2_3_portfolio_combination.md` | Forward | Portfolio combination. |
| `tasks_assigned_12_6_26/Q/Q2_4_signal_combination_sizing.md` | Forward | Signal combination + position sizing. |
| `tasks_assigned_12_6_26/Q/Q2_5_spannung_kalman_filter.md` | Forward | Spannung Kalman filter. |
| `tasks_assigned_12_6_26/Q/Q3_1_kill_switch_infrastructure.md` | Forward | Kill-switch daemon. |
| `tasks_assigned_12_6_26/Q/Q3_2_paper_trading_deployment.md` | Gated-Jun17 | Paper trading (needs 14-day runtime). |
| `tasks_assigned_12_6_26/Q/Q3_3_regime_multifreq_portfolio.md` | Forward | Regime + multi-freq + portfolio. |
| `tasks_assigned_12_6_26/Q/Q4_1_live_deployment_scaleup.md` | Forward | Live deployment scale-up (post-G8). |

*CLI/viz tasks (Infra, Forward) — expansions of `nat_cli_improvement_plan.md`:*

| File | Content |
|------|---------|
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT1_command_search.md` | Command search (`nat help --grep`). |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT2_group_level_help.md` | Curated group-level help. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT3_viz_library_foundation.md` | Viz library foundation. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT4_viz_features.md` | Feature visualization. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT5_viz_algorithm.md` | Algorithm visualization. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT6_viz_paper_trading.md` | Paper-trading visualization. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT7_viz_portfolio.md` | Portfolio visualization. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT8_viz_spectral_regime_correlation.md` | Spectral/regime correlation viz. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT9_maturity_tags_compat.md` | Maturity-tag compatibility. |
| `tasks_assigned_12_6_26/nat_cli_tasks/NAT10_script_modularization.md` | Script modularization. |

*Build checklists (Method) — redundant with the Tier-C convolver specs:*

| File | Lifecycle | Content |
|------|-----------|---------|
| `research/new/ml_implementation_plan.txt` | Gated-Jun17 | Gated ML algo rollout (Waves 1–4) on 30-day tick history. |
| `research/new/convolver_implementation.txt` | Forward | Convolver 4-step execution roadmap (bar_agg fix → re-discovery). |
| `research/new/convolver_implementation_tasks.md` | Forward | Convolver 15-step task checklist with commands + gates. |

---

*Generated 2026-06-14. The single binding finding across Tier A: signal IC is real (≈0.45) but
collapses under realistic fills — execution, not signal discovery, is the blocker. Everything in
Tiers C/D is downstream of resolving that and of the Jun-17 data-streak gate.*

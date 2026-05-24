# NAT Documentation

Quantitative research platform for extracting alpha signals from Hyperliquid perpetual futures.

## Directory Structure

```
docs/
├── architecture/       System design, specifications, user manual
├── research/           Algorithm catalogue, experiment methodology, papers
├── specs/              Design proposals and technical specifications
└── archive/            Historical docs (out_of_date + in_progress from pre-May)
```

## Architecture

| Document | Description |
|----------|-------------|
| [PHASE1_ALGORITHM.md](architecture/PHASE1_ALGORITHM.md) | Phase 1 algorithm design and feature pipeline |
| [EAMM_SPEC.md](architecture/EAMM_SPEC.md) | EAMM specification |
| [USER_MANUAL.md](architecture/USER_MANUAL.md) | User-facing guide for running the system |

## Research

| Document | Description |
|----------|-------------|
| [ALGORITHMS.md](research/ALGORITHMS.md) | All 18 algorithms ranked by OOS performance |
| [ROADMAP.md](research/ROADMAP.md) | Research roadmap and strategic direction |
| [EXP_0_INTRODUCTIONS.md](research/EXP_0_INTRODUCTIONS.md) | Experiment 0 introduction |
| [liquidity_heatmap_model.md](research/liquidity_heatmap_model.md) | Liquidity heatmap model |
| [convolver_method.tex](research/convolver_method.tex) | Academic paper (TeX source + PDF) |

## Specs

| Document | Description |
|----------|-------------|
| [PROFILING_MATHEMATICAL_FORMULATION.md](specs/PROFILING_MATHEMATICAL_FORMULATION.md) | Profiling math formulation |
| [PROFILING_REQUIREMENTS.md](specs/PROFILING_REQUIREMENTS.md) | Profiling requirements |
| [PROFILING_MATHEMATICS.md](specs/PROFILING_MATHEMATICS.md) | Profiling mathematics |
| [WEBSITE_SPEC.md](specs/WEBSITE_SPEC.md) | Website specification |
| [DASHBOARD_BUILD_PLAN.md](specs/DASHBOARD_BUILD_PLAN.md) | Dashboard build plan |
| [ALPHA_RESEARCH_PLAN.md](specs/ALPHA_RESEARCH_PLAN.md) | Alpha research plan |

## Related Directories

- `reports/` -- Experiment results, JSON data, paper trade comparisons
- `config/` -- Runtime configuration (ingestor, agent, pipeline, alpha)
- `FEATURES.md` -- Complete feature manifest (209 features, formulas, references)
- `CLAUDE.md` -- Project conventions for Claude Code

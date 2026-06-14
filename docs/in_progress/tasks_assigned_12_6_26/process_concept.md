# Process — Third First-Class Citizen of NAT

**Status:** Stage 1 + 2 implemented (`scripts/processes/`, branch `feat/process-framework`)
**Companion:** `plan.md` tasks T9.5 / T11b

## Definition

| Citizen | Question it answers | Contract |
|---------|--------------------|----------|
| **feature** | what is computed | ingestor columns / derived Python features |
| **algorithm** | how it trades | `MicrostructureAlgorithm` (scripts/algorithms/) |
| **process** | whether/where information about price action exists | `Process` (scripts/processes/) |

A **process** is an analytical description — statistical, signal-processing, or
ML — that identifies whether a feature, or a derivative/combination of
features, carries information about future price action. It formalizes what
previously lived as scattered one-off analyses (alpha screener IC, IT-engine
MI, spannung spectral, phase-1 LightGBM tests) behind one registry, one
result schema, one CLI, and one persistence path.

## Contract (`scripts/processes/base.py`)

Two kinds share the registry/CLI/persistence surface:

- **`EvaluationProcess`** — `evaluate(df, ctx) -> ProcessResult`: scores
  existing columns.
- **`TransformProcess`** — `transform(df, ctx) -> (derived_df, ProcessResult)`:
  produces NEW series (PCA components, triple-barrier labels) sharing the
  input index. Derived series are first-class evaluable inputs: the runner
  chains them into any evaluation process (`--score-with ic_horizon`), and
  label-type outputs replace forward returns as targets via `target_col`.

Conventions (mirroring algorithms):
- `@register` decorator; registry key == `name()` == section in
  `config/processes.toml`; constructor kwargs = `PARAMS` defaults < toml < CLI.
- `required_columns(available)` receives the parquet schema so processes
  pattern-select columns without loading data.
- `data_level` = `"bars"` (aggregated, default) or `"ticks"` (10 Hz, e.g.
  spectral). The runner feeds accordingly.
- `describe()` returns a machine-readable spec (params with defaults + docs)
  — the hook for agent-generated processes (Stage 3).

**NaN policy (K2):** every process partitions its columns through
`partition_usable_columns()` — dead/constant/thin columns are skipped with a
recorded reason (`all_nan`, `constant`, `n_valid=K<min`), never crash a run.

## Implemented processes

| Name | Kind | Description |
|------|------|-------------|
| `ic_horizon` | evaluation | Expanding Spearman IC x horizon sweep with explicit time dependence (expanding-IC curve, IC-decay half-life), BH-FDR. p-values use full-sample rho with overlap-corrected n_eff = n/h — deliberately NOT the screener's expansion-point t-test, which treats overlapping windows as replications and flags noise. |
| `mi_ksg` | evaluation | KSG mutual information vs forward returns; informative iff MI >= I_min(fee, sigma_r, kurtosis) (cost viability). Optional CMI conditioning + interaction info. Inputs are rank (copula) transformed — raw-scale KSG produces a spurious ~0.07-bit noise floor when feature and return scales differ (verified on planted data). |
| `transfer_entropy` | evaluation | Directed TE(feature -> 1-bar return), linear (recommended) or KSG, with reverse-direction control. |
| `spectral` | evaluation | Tick-level PSD / Hurst / OU half-life / frequency-band IC (wraps spannung). Informative = band IC + persistence (half-life >= horizon). |
| `ml_importance` | evaluation | LightGBM expanding walk-forward gain importance + confidence-filtered net PnL after costs; informative = top-k rank AND profitable strategy. |
| `triple_barrier` | transform | López de Prado 3-barrier labels (tb_label/tb_ret/tb_hit_bars), vol-scaled, past-only vol, NaN tail. |
| `pca_combo` | transform | Train-prefix PCA -> orthogonal pc_1..pc_k; holdout-only IC scoring; `summary.orthogonality` = max off-diag holdout correlation. |

## Result schema + persistence

`ProcessResult` (schema_version 1): run_id, process, kind, symbol, timeframe,
params, data{dir, dates, n_rows, n_bars, fingerprint}, provenance{git_sha,
dirty, generated_at}, features_tested, features_skipped[{feature, reason}],
findings[{feature, horizon, metric, value, threshold, p_value, p_adjusted,
informative, extras}], derived{columns, parquet, scored_by}|null,
summary{n_tested, n_informative, top, runtime_s, error}.

- Full JSON (authoritative): `data/research/processes/{run_id}.json`, atomic
  write-then-rename. Derived parquet: `data/research/processes/derived/`.
- Index row (queryable): `process_results` table in `nat.db` via the standard
  `_run_migrations()` discipline; the index write is best-effort and never
  loses the JSON.

**FDR scope:** BH correction is per-run. The index is a research log, not a
meta-gate — aggregating `informative` flags across runs inflates discoveries;
gates G1/G4 remain authoritative for promotion decisions.

## CLI

`nat process` (curated group help) / `list [--kind] [--json]` /
`run NAME --symbol ... [--start-date] [--features pfx,...] [--param k=v]
[--score-with ic_horizon]` / `results [--process] [--symbol]` /
`show RUN_ID`. Tests: `nat test process`.

## Stage 3 — extension points (spec only, not implemented)

1. **Prism (exogenous narrative/hype series).** `ProcessContext.extra_sources:
   dict[str, DataFrame]` already exists. A `FeatureSource` protocol —
   `load(symbol, start, end) -> DataFrame` with a UTC timestamp index — lets
   the runner `merge_asof` external float-vs-time series (keyword hype
   footprints) onto bars via a future `--source` flag. Processes operate on
   columns and never know their origin: a prism series is scored by
   ic_horizon/mi_ksg exactly like an ingestor feature.
2. **Literature-to-process pipeline.** `describe()` exposes a param schema an
   agent generator can target; `scripts/processes/generated/` is reserved
   (mirror of `algorithms/generated/`). Pipeline sketch: `arxiv_papers.ideas`
   -> generator emits a Process subclass -> mandatory planted-signal gate
   (`scripts/processes/synthetic.py` — library code precisely so generated
   processes are auto-tested: must flag `feat_signal`, must NOT flag
   `feat_noise`/shuffled) -> real-data run -> human review.

## Verification

`nat test process` — 48 tests: planted-signal contracts per process,
no-lookahead checks for transforms (PCA holdout perturbation, triple-barrier
vs naive reference), persistence round-trip + migration idempotence, and a
real-parquet smoke (latest day, dead K2 columns skipped with reasons,
runtime budget). Data caveat: with < ~7 clean days, gates rarely pass and
holdout segments can fall below minimum observations — correct conservatism,
not a bug (see plan.md T13 caveat).

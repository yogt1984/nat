# Contract: Process

**Lives in** `scripts/processes/` · **registry** `@register` · **kind** `evaluation` | `transform`

A **process** answers *"where does information about future price action live?"* — it measures
predictive structure (IC, mutual information, transfer entropy, spectral power, ML importance, …).
It is **not** an algorithm: it never decides how to trade, only whether/where signal exists.
(Third first-class citizen, alongside feature and algorithm.)

## Signature

```python
from processes.base import EvaluationProcess, ProcessContext, ProcessResult, Finding
from processes.registry import register

@register
class MyProcess(EvaluationProcess):          # or TransformProcess
    """One-line what-it-measures (used by `describe()` + agent generators).

    Longer docstring: the estimator, its assumptions, references.
    """
    PARAMS = {"horizon_bars": (24, "lookahead in bars")}   # {name: (default, doc)}; unknown kwargs raise

    def name(self) -> str:
        return "my_process"                  # == registry key == config/processes.toml section

    # required_columns(available) is inherited; override only to pattern-select.

    def evaluate(self, df, ctx: ProcessContext) -> ProcessResult:
        findings = [Finding(feature=c, horizon="h1", metric="ic_mean",
                            value=v, informative=abs(v) > thr) for c, v in ...]
        return ProcessResult(run_id=..., process=self.name(), kind=self.kind,
                             symbol=ctx.symbol, timeframe=ctx.timeframe,
                             params=self.params, findings=findings).finalize(runtime_s)
```

- **EvaluationProcess** → implement `evaluate(df, ctx) -> ProcessResult`.
- **TransformProcess** → implement `transform(...)` returning derived columns (e.g. `pca_combo`,
  `triple_barrier`); register the derived parquet via `ProcessResult.derived`.
- `ctx.costs` is `config/costs.toml` via `utils.costs.load_costs()` — **never hardcode fees**.
- `data_level` (`"bars"` | `"ticks"`) tells the runner what to feed.

## Output contract — `ProcessResult` schema_version 1

`run_id, process, kind, symbol, timeframe, params, schema_version=1, data, provenance,
features_tested, features_skipped, findings:[Finding], derived, summary`.
`Finding = (feature, horizon, metric, value, threshold?, p_value?, p_adjusted?, informative,
extras)`. `.finalize()` fills `summary` (`n_tested, n_informative, top[5], runtime_s, error`).

> **Do not add or rename fields without bumping `schema_version`** — `scripts/data/state.py`
> (`process_results` table) and the future `/api/research/*` consumers depend on this shape.

## Tests

```bash
# 1. Planted (synthetic, known answer) — FIRST, before any real parquet:
#    processes/synthetic.py → make_planted_frame(ic=0.15, horizon=4), make_test_context()
nat test process                                   # 48 synthetic contracts + real-data smoke
pytest scripts/tests/test_process_<area>.py -q     # your unit's planted test

# 2. Real-parquet smoke (latest day), before commit:
nat process run my_process --symbol BTC            # dead/unavailable cols skipped w/ reason, never crash
nat process list                                   # confirm @register fired
nat process results                                # confirm row persisted to process_results
```

## Definition of Done (+ the shared 7 in [`README.md`](README.md))

- [ ] Planted test green **before** any real-data use; estimator validated on a known-MI frame.
- [ ] `@register` fires → appears in `nat process list`.
- [ ] `config/processes.toml` section name == `name()`; unknown `PARAMS` raise `TypeError`.
- [ ] `evaluate`/`transform` returns `ProcessResult(schema_version=1)`; `features_skipped` carries a
      reason for every dead column (no crash on K-columns).
- [ ] Real smoke on latest parquet: `nat process run …` clean; row in `process_results`.
- [ ] Maturity tag assigned (PRELIM on merge) · feat branch · `merge --no-ff` · CI green.

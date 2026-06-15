# Contract: Algorithm

**Lives in** `scripts/algorithms/` · **registry** `@register` · **base** `MicrostructureAlgorithm`

An **algorithm** turns features into a tradeable signal — *how* to act on information a process found.
Output feature names must start with `alg_`.

## Signature

```python
from algorithms.base import MicrostructureAlgorithm, AlgorithmFeature
from algorithms.registry import register

@register
class MyAlgorithm(MicrostructureAlgorithm):
    """Math formulation + references (required)."""

    def name(self) -> str: ...                              # unique id
    def alg_features(self) -> list[AlgorithmFeature]: ...   # output descriptors (alg_ prefix)
    def required_columns(self) -> list[str]: ...            # input feature names
    def step(self, tick: dict[str, float]) -> dict[str, float]: ...  # one tick → exactly alg_features() keys
    def reset(self) -> None: ...                            # clear internal state
    # def warmup(self) -> int: ...        # first N run_batch rows are NaN-blanked automatically
    # def run_batch(self, df) -> pd.DataFrame: ...  # override for vectorized numpy/pandas path
```

**Rules**
- `step()` returns **exactly** the keys from `alg_features()` — no more, no less.
- **NaN in → NaN out:** if any required column is NaN, return NaN for all outputs (don't crash, don't impute).
- Parameters configurable via `config/algorithms.toml` (section == `name()`).
- `run_batch()` default iterates `step()`; override with vectorized code for speed.

## Tests

```bash
# 1. Planted-IC (synthetic, known edge) — FIRST:
pytest scripts/tests/test_algorithm_smoke.py -k <name>

# 2. Real-data evaluation, before commit:
nat algorithm evaluate --algorithm <name> --symbol BTC    # IC + drift on real parquet
nat algorithm list                                        # confirm @register fired

# 3. OOS performance (after it earns BETA):
nat oos30           # 30-day OOS for winners        nat gauntlet run --last 7   # multi-day sweep
```

## Definition of Done (+ the shared 7 in [`README.md`](README.md))

- [ ] Planted-IC test green **before** real data.
- [ ] `@register` fires → `nat algorithm list`; `step()` keys == `alg_features()`; `alg_` prefix.
- [ ] NaN-in→NaN-out verified; warmup rows blanked.
- [ ] `config/algorithms.toml` section wired; docstring has the math + references.
- [ ] Real smoke: `nat algorithm evaluate …` clean on the latest day.
- [ ] Maturity tag (PRELIM on merge) · feat branch · `merge --no-ff` · CI green.

> Performance gates (BETA→PROVEN) are **imported, not invented**: G4 = walk-forward + deflated
> Sharpe (OOS Sharpe>0.5, OOS/IS>0.7, deflated p<0.05, maxDD<5%, ≥30 trades, PF>1.2). Never add a
> new threshold. Slow algos (e.g. `cascade_probability`, ~70 min/date) belong on the gauntlet's
> exclude list, not in the nightly default.

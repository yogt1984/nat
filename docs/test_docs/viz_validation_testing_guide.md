# Testing Guide — Parquet Viz & Validation Tooling

How to test the viz/validation tooling shipped this cycle, mapped to the commits that
introduced it. Spec: [`docs/requirements/parquet_viz_validation.md`](../requirements/parquet_viz_validation.md).
Full manual checklist: [`docs/in_progress/test_plan.md`](../in_progress/test_plan.md) → **Section A**.

All commands run from the repo root (`/home/onat/nat`).

## The commits (on `master`)

| Commit | What it shipped | Test area |
|--------|-----------------|-----------|
| `1f59aed` | `feat(viz): nat data validate <path>` single-file validation | **B** |
| `e6170d3` | `feat(viz): nat viz render` paged PNG viewer (+ `15m viz` auto-open) | **A**, **D** |
| `779d34d` | `feat(viz): nat viz3d / nat mesh` interactive feature-surface | **C** |
| `0ffb663` | `feat(viz): wire --features into nat viz render` | **A** (features) |
| `4c74126`, `b55d58d`, `16a59db` | requirements spec (proposed → implemented) | reference |
| `cf871f1` | README quickstart + CLI reference | reference |
| `6c639db` | manual test plan — new Section A | the checklist |

## 0. Fastest confidence — automated planted tests (~6s)

```bash
pytest scripts/tests/test_validate_data_file.py \
       scripts/tests/test_viz_render_pagination.py \
       scripts/tests/test_viz_render_features.py \
       scripts/tests/test_viz_mesh.py -q
# expect: 24 passed
```

## Manual smoke (real parquet)

Pick a real recent file first:

```bash
F=$(ls -t data/features/*/*.parquet | head -1); echo "$F"
```

### B. `nat data validate` — single-file validation (`1f59aed`)

```bash
nat data validate "$F"; echo "exit=$?"          # PASS/WARN/FAIL; on prod data FAIL+exit1 is EXPECTED
                                                #   (dead optional cols → NaN Ratio; real gaps → Continuity)
nat data validate "$F" --json | python3 -c "import json,sys;print(json.load(sys.stdin)['verdict'])"
nat data validate                               # no-arg = directory mode (unchanged)
nat data validate /tmp/nope.parquet; echo $?    # clean error, exit 1, no traceback
```
Verdict tiers: hard checks (integrity/continuity/NaN/sequence/presence) → **FAIL**; any other failing
check → **WARN**; else **PASS**. Single-file mode exits nonzero only on FAIL.

### A. `nat viz render` — paged PNG viewer (`e6170d3`, `0ffb663`)

Writes PNGs to `reports/figures/snapshots/`. `--tf` = granularity/page width; an optional **1-based
index** pages the day (data-relative: page 1 = first available tick).

```bash
nat viz render --tf 15m --symbol BTC            # overview: whole day, all-features curated panels
nat viz render --tf 5m 1  --symbol BTC          # page 1 = first 5 min   (filename …_w01_…)
nat viz render --tf 5m 2  --symbol BTC          # page 2 = 5–10 min
nat viz render --tf 5m 9999 --symbol BTC; echo $?       # "out of range; N pages", exit 1
nat viz render --tf 5m 1 --features flow --symbol BTC   # per-feature panel grid (filename feat_…)
nat viz render --tf 5m 1 --features raw_midprice,raw_spread --symbol BTC
nat viz render --tf 5m 1 --features nope_xyz --symbol BTC; echo $?   # "matched no columns", exit 1
nat viz render --tf 15m --open                  # opens the PNG (or prints path if headless)
ls -la reports/figures/snapshots/
```
A final partial window is flagged "— partial" in the title (expected). `--features` accepts a
category / named vector / comma-list / `all`; large categories are capped to top-N by variance
(`--max-features`, default 16).

### C. `nat viz3d` / `nat mesh` — interactive 3D feature-surface (`779d34d`)

Writes self-contained HTML to `reports/figures/mesh/`.

```bash
nat viz3d --tf 15m --symbol BTC                 # writes <sym>_<tf>_<date>.html
nat mesh  --tf 5m 2 --features entropy --symbol BTC   # alias; page 2, entropy scoped
nat viz3d --tf 5m 9999 --symbol BTC; echo $?    # out-of-range, exit 1
grep -c "Plotly.newPlot" reports/figures/mesh/*.html   # ≥1 → self-contained / offline
xdg-open reports/figures/mesh/*.html            # rotate/zoom: x=time, y=features, z=value
```
If `plotly` is absent → clean message ("plotly is required … pip install plotly"), exit 1 (not a
traceback).

### D. `nat 15m viz` — 15-minute snapshot auto-open (`e6170d3`)

```bash
nat 15m viz --symbol BTC --no-open              # renders latest-experiment snapshot, no auto-open
nat 15m viz --symbol BTC                        # same, but auto-opens the PNG(s)
```

## Notes

- The latest day (e.g. `data/features/2026-06-16/`) can be ~400 MB, so `viz render`/`viz3d` load the
  whole day (~20–40 s). For a faster check, point `--date` at a one-hour subset:

  ```bash
  mkdir -p data/features/_smoke && cp "$F" data/features/_smoke/
  nat viz render --tf 5m 1 --symbol BTC --date _smoke
  nat viz3d --tf 15m --symbol BTC --date _smoke
  rm -rf data/features/_smoke                   # cleanup — don't leave it under data/features
  ```

- `reports/figures/*` and the `_smoke` dir are gitignored, so generated artifacts won't dirty the tree.

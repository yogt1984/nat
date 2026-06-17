# Contract: Viz (a `nat` visualization command)

**Lives in** `nat` (`cmd_viz_<x>(args)`) · **registry** `build_parser()` → `nat viz <x>`
· **base** the shared libs in `scripts/viz/` + `scripts/cluster_pipeline/`

A **viz command** turns *stored* features / signals into something a human reads — a terminal
view, a PNG, or an interactive HTML surface. It is **read-only** on `data/` and writes only under
`reports/`. The north-star is **ergonomics: smart defaults, minimal flags** — `nat viz <x>` must
do something useful with *zero* required arguments; flags only *override*.

> Scope: this covers the **nat-embedded** (CLI) side of the viz citizen. The web `viz-component`
> (`web/src/components/`) stays P3-deferred.

## Signature

```python
def cmd_viz_<x>(args) -> int:          # 0 ok · nonzero on no-data / bad-arg. NEVER raise to the user.
    date = getattr(args, "date", None) or _latest_feature_day(args)   # smart default: latest day
    sym  = _sym(args)                                                 # smart default: BTC
    if not date:
        _p("x", R, f"No feature data under {_data(args)}"); return 1
    df = load_parquet(f"{_data(args)}/{date}", symbols=[sym], columns=cols, max_memory_mb=500)
    # … build output (terminal | PNG | HTML); skip all-NaN cols with a reason …
    if getattr(args, "open_after", False):
        from viz.open_helper import open_path; open_path(out)         # --open
    print(f"  Saved: {out}")
    return 0

# registration (in build_parser, under the `nat viz` group):
vzx = vizsub.add_parser("<x>", help="…")
vzx.add_argument("-s", "--symbol", default="BTC"); vzx.add_argument("--date", default=None)
vzx.add_argument("--open", dest="open_after", action="store_true")
vzx.set_defaults(func=cmd_viz_<x>)
```

**Rules (the ergonomic contract — the heart of this unit)**

- **Every flag optional, smart defaults.** `--date` → latest available day (`_latest_feature_day`);
  `-s/--symbol` → `_sym` (BTC); a sensible default view. `nat viz <x>` alone must produce a useful
  result. *Progressive disclosure:* flags exist only to override.
- **Consistent flag grammar — same name, same meaning, everywhere:**
  `-s/--symbol`, `--date YYYY-MM-DD`, `--tf {1m,5m,15m}` + optional positional `INDEX` (the
  data-relative page, `viz/pager.py`), `--features <category|vector|csv|all>`
  (`viz/feature_select.py`), `--open`, `--json`, `--output PATH`.
- **Return an int exit code** — the dispatcher propagates it. **Clean one-line errors, never a
  traceback**: nonzero on no-data / out-of-range page / unknown `--features` / missing date.
- **NaN-unavailable, don't crash.** All-NaN (dead optional) columns render as an explicit
  "unavailable" panel/cell — never an exception. (As today: whale/liq/concentration are NaN until
  the wired ingestor deploys.)
- **Read-only on `data/`; write only under `reports/`.** Filter by `--symbol`/`columns` early
  (`max_memory_mb`) so a single-symbol day stays well under ~500 MB.
- **Theme + naming.** PNGs use the dark `STYLE`/`COLORS` from `viz/features.py`,
  `dpi=150, bbox_inches="tight"`; deterministic output paths (only the timestamp varies).

**Output modalities** — pick the *lightest* that answers the question:

| Modality | Use | How |
|---|---|---|
| **terminal-first** | the fast inner loop, no files | `viz/terminal.py` sparklines / tables / ANSI |
| **PNG** | a shareable static snapshot | matplotlib via `viz/*` → `reports/figures/<group>/`, `--open` |
| **interactive HTML** | rotate/zoom / explore | Plotly `include_plotlyjs=True` (offline) → `reports/figures/mesh/`, `--open` |

**Reuse map (do NOT rebuild):**

| Need | Module |
|------|--------|
| load parquet (symbols / date / cols / mem cap) | `scripts/cluster_pipeline/loader.py::load_parquet` |
| latest-date / N-hour load | `scripts/swarm/parquet_reader.py::read_evaluation_data` |
| resample 1m/5m/15m bars | `scripts/cluster_pipeline/preprocess.py::aggregate_bars` (`TIMEFRAMES`) |
| data-relative `INDEX` page math | `scripts/viz/pager.py::window_bounds` / `window_edges` |
| `--open` opener (xdg-open/open/start) | `scripts/viz/open_helper.py::open_path` / `open_all` |
| `--features` selection + variance cap | `scripts/viz/feature_select.py::select_features` / `cap_by_variance` |
| themed plotters / per-feature panels | `scripts/viz/{features,correlations,distributions,events}.py`; terminal: `viz/terminal.py` |
| feature categories / quality | `scripts/data/schema.py` (`BASE_FEATURES`/`OPTIONAL_FEATURES`, `validate_columns`/`validate_quality`) |
| CLI helpers | `nat`: `_sym` · `_data` · `_latest_feature_day` · `_json_mode` · `_banner` |

## Tests

```bash
# 1. Planted first (red) — synthetic parquet, known pattern + injected NaN/gap. BEFORE real data:
pytest scripts/tests/test_viz_<x>.py -q
#    assert: output produced & correct shape (PNG non-empty / HTML surface dims / terminal rows);
#    all-NaN columns handled; bad-arg / empty / out-of-range → NONZERO exit.
#    models: test_viz_render_pagination.py · test_viz_render_features.py · test_viz_mesh.py

# 2. Real-parquet smoke, before commit (use a 1-hour subset for speed):
nat viz <x> --date <day>            # clean output on the latest day; --open opens it

# 3. Dispatch / no-regression:
nat viz <x> -h                      # parses & routes; existing viz commands unchanged
```

## Definition of Done (+ the shared 7 in [`README.md`](README.md))

- [ ] Planted test green **before** real data; error paths exit nonzero (no traceback).
- [ ] Registered under `nat viz` via `set_defaults(func=cmd_viz_<x>)` **returning an int**.
- [ ] Smart-default flags wired (`--date`→latest, `--symbol`→BTC); consistent flag grammar.
- [ ] All-NaN / empty / out-of-range handled gracefully; read-only on `data/`, writes under `reports/`.
- [ ] Real smoke clean on the latest day; PNG uses the dark theme + `dpi=150`.
- [ ] Maturity tag (**SPEC → PRELIM** on merge) · feat branch · `merge --no-ff` · CI green.

> Gates/thresholds are **imported, not invented**; costs only via `load_costs()`. A viz command
> *shows* numbers — it must never silently recompute a fee, slippage, or a promotion threshold.
> Reference exemplars that already meet this contract: `nat viz render` / `nat viz3d` (smart
> defaults, `--open`, pager, clean exit codes); `nat data validate` (verdict + `--json` + exit codes).

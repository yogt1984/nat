# Tasks — May 8, 2026

Improvement backlog generated from a full project scan. Items are grouped by
category and ordered by severity within each group.

---

## Category 1: Bugs (crash-level)

### 1.1 `cmd_visualize_hierarchy` passes args as cwd

**File:** `nat`, line ~487  
**Severity:** Critical  
**Description:**  
`_py(script_path, args_string)` passes the CLI arguments string as the `cwd`
keyword argument instead of appending them to the script invocation. This means
`nat visualize hierarchy` silently does nothing useful — it runs the script in
a nonexistent directory or the wrong directory without the intended flags.

**Root cause:** `_py()` signature is `_py(script_args, cwd=None)`. The second
positional argument is interpreted as `cwd`, not as extra arguments.

**Fix:** Build a single string: `_py(f"{script_path} --data {data} --output {output} --vector {vector} --timeframe {tf}")`.

**Acceptance criteria:**
- [ ] `nat visualize hierarchy --data data/features --output /tmp/test` produces PNG files in the output directory
- [ ] No Python traceback or silent failure
- [ ] Existing tests (`pytest scripts/tests/`) still pass

---

### 1.2 `cmd_cluster_hmm` uses numpy without import

**File:** `nat`, lines ~919, 921, 956  
**Severity:** High  
**Description:**  
The `cmd_cluster_hmm` function references `np.isfinite`, `np.nanmedian`, and
`np.array2string`, but `numpy` is never imported in the `nat` CLI script.
Running `nat cluster hmm-fit` crashes immediately with
`NameError: name 'np' is not defined`.

**Fix:** Add `import numpy as np` at the top of `cmd_cluster_hmm` (lazy import
to avoid loading numpy for every `nat` invocation).

**Acceptance criteria:**
- [ ] `nat cluster hmm-fit --help` does not crash
- [ ] `nat cluster hmm-fit --symbol BTC --data data/features` runs without NameError (may fail on missing data, but not on import)
- [ ] No numpy import at module top level (keep startup fast)

---

### 1.3 Pydantic v2 incompatibility in model_serving.py

**File:** `scripts/model_serving.py`, lines ~98, 112  
**Severity:** High  
**Description:**  
`requirements-serving.txt` pins `pydantic>=2.0.0`, but `model_serving.py`
uses the `@validator` decorator which was removed in pydantic v2 (replaced
by `@field_validator` with `@classmethod`). Running `nat model serve` crashes
with `ImportError` or `PydanticUserError` on any pydantic 2.x install.

**Fix:** Replace `@validator` with `@field_validator` and add `@classmethod`
decorator. Update field access from `values` dict to `info.data` pattern.

**Acceptance criteria:**
- [ ] `nat model serve --help` works without import errors
- [ ] `python -c "from model_serving import app"` succeeds with pydantic 2.x installed
- [ ] `pytest scripts/tests/test_model_serving.py` passes (if it exists; create smoke test if not)

---

## Category 2: Broken Dependencies

### 2.1 `tomli` missing from requirements.txt

**File:** `requirements.txt`  
**Severity:** High  
**Description:**  
Multiple scripts (`pipeline_runner.py`, `scalping_profiler.py`, `dashboard.py`,
`scalp_edge_scanner.py`) use a `try: import tomllib / except: import tomli`
shim for Python <3.11 compatibility. `tomli` is the fallback package but is
not listed in `requirements.txt`. Users on Python 3.10 get
`ModuleNotFoundError: No module named 'tomli'`.

**Fix:** Add `tomli>=2.0.0; python_version < "3.11"` to `requirements.txt`.

**Acceptance criteria:**
- [ ] `pip install -r requirements.txt` succeeds on Python 3.10 and 3.12
- [ ] `python -c "import tomli"` works on Python 3.10 after install
- [ ] On Python 3.11+ the tomli package is not installed (conditional marker)

---

### 2.2 fastapi/uvicorn/pydantic commented out in requirements.txt

**File:** `requirements.txt`, lines ~49-51  
**Severity:** Medium  
**Description:**  
`nat model serve` invokes `model_serving.py` which requires `fastapi`,
`uvicorn`, and `pydantic`. These are listed in a separate
`scripts/requirements-serving.txt` but are commented out in the main
`requirements.txt`. Anyone running `nat model serve` without separately
reading the docs gets import errors.

**Fix:** Either uncomment in `requirements.txt` with an `# optional: serving`
comment, or add a clear error message in `nat model serve` that tells the user
to `pip install -r scripts/requirements-serving.txt`.

**Acceptance criteria:**
- [ ] Running `nat model serve` without serving deps gives a clear, actionable error message (not a raw traceback)
- [ ] OR: `pip install -r requirements.txt` installs serving deps

---

### 2.3 `dash` commented out in requirements.txt

**File:** `requirements.txt`, line ~68  
**Severity:** Medium  
**Description:**  
`nat pipeline dashboard` invokes `scripts/dashboard.py` which imports `dash`.
The dependency is commented out. Same issue as 2.2.

**Fix:** Same approach as 2.2 — either uncomment or add a clear error message.

**Acceptance criteria:**
- [ ] `nat pipeline dashboard` without dash installed gives a clear error (not raw ImportError)
- [ ] OR: `pip install -r requirements.txt` includes `dash`

---

### 2.4 polars version range too loose

**File:** `requirements.txt`, line ~38  
**Severity:** Medium  
**Description:**  
`polars>=0.20.0` allows installs of polars versions with breaking API changes
(the `apply` to `map_elements` rename, `select` API changes between 0.x and
1.x). Multiple scripts use polars 1.x API patterns.

**Fix:** Tighten to `polars>=1.0.0`.

**Acceptance criteria:**
- [ ] `pip install -r requirements.txt` installs polars 1.x
- [ ] `nat cluster explore --help` works (uses polars heavily)
- [ ] `pytest scripts/tests/test_cluster_loader.py` passes

---

## Category 3: Test Coverage

### 3.1 Alpha pipeline scripts (Steps 2-9) have zero tests

**Directory:** `scripts/alpha/`  
**Severity:** High  
**Description:**  
All 8 alpha pipeline scripts lack test files:
- `combiner.py` (Step 2 — feature combination)
- `position.py` (Step 3 — cost-aware sizing)
- `adapter.py` (Step 4 — walk-forward adapter)
- `regime_filter.py` (Step 5 — regime conditioning)
- `multi_freq.py` (Step 6 — multi-frequency integration)
- `portfolio.py` (Step 7 — portfolio assembly)
- `paper_trader.py` (Step 8 — paper trading)
- `deployer.py` (Step 9 — deployment framework)

These are the core alpha research pipeline and cover sizing logic, quality
gates (G2-G9), kill switches, and position limits — all high-value, zero
coverage.

**Fix:** Create `scripts/tests/test_alpha_*.py` for each module. Tests should
use synthetic data and verify core functions (quality gate logic, signal
combination, kill switch evaluation, position limit computation).

**Acceptance criteria:**
- [ ] 8 new test files in `scripts/tests/`
- [ ] Each file tests at least the main pipeline function and quality gate logic
- [ ] All tests pass with `pytest scripts/tests/test_alpha_*.py`
- [ ] No real data or network access required

---

### 3.2 Alpha screener has no tests

**File:** `scripts/alpha/screener.py`  
**Severity:** Medium  
**Description:**  
The alpha screener (Benjamini-Hochberg FDR + IC computation) is the entry
point for the entire alpha pipeline. No test file exists.

**Fix:** Create `scripts/tests/test_alpha_screener.py` with synthetic feature
data testing IC computation, FDR correction, and ranking logic.

**Acceptance criteria:**
- [ ] `scripts/tests/test_alpha_screener.py` exists
- [ ] Tests IC computation on known-correlation synthetic data
- [ ] Tests FDR correction produces expected number of rejections
- [ ] All tests pass

---

### 3.3 Macro data layer has no tests

**File:** `scripts/data/macro.py`  
**Severity:** Medium  
**Description:**  
The macro data layer fetches and aggregates candle data. Contains two bare
`except Exception` blocks (lines ~106, 144) that silently swallow errors. No
tests exist.

**Fix:** Create `scripts/tests/test_macro.py` testing the aggregation logic
with synthetic candle data. Replace bare `except Exception` with specific
exception types or at minimum add logging.

**Acceptance criteria:**
- [ ] `scripts/tests/test_macro.py` exists and passes
- [ ] Bare `except Exception` blocks replaced with specific handling or logging
- [ ] SMA crossover computation tested on known data

---

## Category 4: CI/CD

### 4.1 No GitHub Actions workflow

**Directory:** `.github/` (does not exist)  
**Severity:** Medium  
**Description:**  
There are 38 Python test files and a Rust unit test suite, but none run
automatically on push or PR. Regressions are only caught manually.

**Fix:** Create `.github/workflows/ci.yml` with two jobs:
1. **Rust:** `cargo test --package ing` (requires Cargo workspace)
2. **Python:** `pip install -r requirements.txt && pytest scripts/tests/ -x --ignore=scripts/tests/test_integration_profiling.py`

Skip integration tests (they need real data). Skip model serving tests
(optional deps).

**Acceptance criteria:**
- [ ] `.github/workflows/ci.yml` exists
- [ ] Push to `master` triggers both Rust and Python test jobs
- [ ] Green status badge added to README or CLAUDE.md (optional)
- [ ] Integration and serving tests are excluded from CI

---

## Category 5: Configuration

### 5.1 Redis URL hardcoded in 4 places

**Files:**  
- `rust/ing/src/redis_publisher.rs` (lines ~138, 151)  
- `rust/api/src/config.rs` (lines ~17, 32)  
- `rust/api/src/bin/alert_service.rs` (line ~40)

**Severity:** Medium  
**Description:**  
The Redis connection URL `redis://127.0.0.1:6379` is hardcoded as a fallback
in 4 places across 2 crates. The only override is via `REDIS_URL` env var.
The `config/ing.toml` has no `[redis]` section. On the second machine (su-35)
this requires setting the env var every time.

**Fix:** Add `[redis] url = "redis://127.0.0.1:6379"` to `config/ing.toml`.
Read it in `Config` struct. Keep env var override as highest priority.

**Acceptance criteria:**
- [ ] `config/ing.toml` has a `[redis]` section with `url` key
- [ ] Rust `Config` struct parses the redis URL from toml
- [ ] `REDIS_URL` env var still overrides the toml value
- [ ] No hardcoded Redis URLs remain in source (single const or config only)

---

## Category 6: Performance

### 6.1 `cmd_status` calls stat() twice per parquet file

**File:** `nat`, lines ~169-171  
**Severity:** Low  
**Description:**  
```python
total_size = sum(f.stat().st_size for f in parquets if f.stat().st_size > 0)
```
Each parquet file gets `stat()` called twice — once for the filter, once for
the sum. With hundreds of parquet files this doubles the syscall count.

**Fix:** Cache stat results: `stats = [(f, f.stat()) for f in parquets]` then
filter and sum from the cache.

**Acceptance criteria:**
- [ ] Each file is stat'd exactly once
- [ ] `nat status` output unchanged
- [ ] No performance regression on large data directories

---

## Category 7: UX / Documentation

### 7.1 Two dashboards with no explanation

**Files:** `nat` (lines ~563, 578, 1189)  
**Severity:** Low  
**Description:**  
The Rust embedded dashboard (port 8080, `ING_DASHBOARD_ENABLED=true`) and the
Python Dash dashboard (port 8050, `nat dashboard`) are completely separate
systems. `nat run serve` announces 8080; `nat start` mentions 8050;
`nat run tunnel` tunnels 8080; `nat exp tunnel` tunnels 8050. Nothing explains
the distinction.

**Fix:** Add a note in `nat help` output and in the man page clarifying:
- Port 8080: Rust real-time feature dashboard (embedded in ingestor)
- Port 8050: Python analysis dashboard (Dash/Plotly, pipeline status)

**Acceptance criteria:**
- [ ] `nat help` mentions both dashboards and their ports
- [ ] `man nat` COMMANDS section clarifies the distinction
- [ ] No code changes needed beyond help text

---

## Category 8: Security

### 8.1 Telegram alert service fails silently without credentials

**File:** `rust/api/src/bin/alert_service.rs`, line ~40  
**Severity:** Low  
**Description:**  
If `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are not set, the alert service
starts but never sends alerts, with no warning at startup. This is confusing
when debugging missing alerts.

**Fix:** Add a startup log line: if either env var is missing, log a warning
like `"TELEGRAM_BOT_TOKEN not set — alerts disabled"`.

**Acceptance criteria:**
- [ ] Starting alert service without Telegram env vars logs a clear warning
- [ ] Service still starts (graceful degradation, not a crash)
- [ ] With env vars set, behavior unchanged

---

## Summary

| # | Item | Category | Severity | Est. effort |
|---|------|----------|----------|-------------|
| 1.1 | visualize hierarchy args bug | Bug | Critical | 5 min |
| 1.2 | hmm numpy import | Bug | High | 2 min |
| 1.3 | pydantic v2 compat | Bug | High | 15 min |
| 2.1 | tomli in requirements | Deps | High | 2 min |
| 2.2 | fastapi/uvicorn commented out | Deps | Medium | 10 min |
| 2.3 | dash commented out | Deps | Medium | 5 min |
| 2.4 | polars version range | Deps | Medium | 2 min |
| 3.1 | alpha pipeline tests (x8) | Tests | High | 2-3 hrs |
| 3.2 | alpha screener tests | Tests | Medium | 30 min |
| 3.3 | macro data layer tests | Tests | Medium | 30 min |
| 4.1 | GitHub Actions CI | CI/CD | Medium | 30 min |
| 5.1 | Redis URL in config | Config | Medium | 30 min |
| 6.1 | double stat() in status | Perf | Low | 5 min |
| 7.1 | two dashboards docs | UX | Low | 10 min |
| 8.1 | telegram silent fail | Security | Low | 10 min |

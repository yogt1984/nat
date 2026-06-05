# 02 — Data Sufficiency Checks

Source: `docs/research/new/ml_implementation_plan.txt`, Section 1 (lines 226–325)

Must pass before any ML model training. Run after Wave 0 infrastructure.

---

### Task 1: Create data sufficiency check script

**Read first:** `scripts/cluster_pipeline/loader.py` (for `load_parquet`), `scripts/cluster_pipeline/preprocess.py` (for `aggregate_bars`)

**Create:** `scripts/check_data_sufficiency.py` (~120 LOC)

CLI script with `--data-dir` and `--symbol` arguments. It must:

1. Load parquet files for the symbol via `load_parquet(data_dir, symbols=[symbol])`
2. Aggregate to 5-min bars via `aggregate_bars(df, timeframe="5min")`
3. Print and check 4 thresholds:

| Check | Threshold | Action on fail |
|-------|-----------|----------------|
| Bar count | >= 4,000 | Print FAIL, exit 1 |
| Label balance | 40% <= pos_rate <= 60% | Print WARNING (don't block) |
| Feature NaN rate | < 5% per feature | Print FAIL for each bad feature |
| Walk-forward fold | >= 500 bars per fold (4 folds) | Print FAIL, suggest fewer folds |

4. The feature list to check NaN rates against:
   `ent_tick_1m`, `trend_hurst_300`, `toxic_vpin_50`, `whale_net_flow_4h`,
   `vol_returns_5m`, `regime_accumulation_score`, `imbalance_qty_l1`
   (check both raw and `_mean` suffixed versions)
5. Print summary: `DATA SUFFICIENT` or `DATA INSUFFICIENT` with details.
6. Support `--json` flag for machine-readable output.

```bash
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features --json
```

---

### Task 2: Write unit tests for data sufficiency checks

**Create:** `scripts/tests/test_data_sufficiency.py` (~100 LOC)

Use synthetic DataFrames (no real parquet needed). Tests:

```python
def test_bar_count_pass():
    """DataFrame with 5000 rows passes bar count check."""

def test_bar_count_fail():
    """DataFrame with 3000 rows fails bar count check."""

def test_label_balance_pass():
    """Forward returns with 48% positive rate passes."""

def test_label_balance_warn():
    """Forward returns with 35% positive rate warns."""

def test_nan_rate_pass():
    """Feature column with 2% NaN passes."""

def test_nan_rate_fail():
    """Feature column with 10% NaN fails."""

def test_fold_size_pass():
    """8000 bars / 4 folds = 2000 per fold passes."""

def test_fold_size_fail():
    """1500 bars / 4 folds = 375 per fold fails."""

def test_json_output_structure():
    """JSON output has keys: bar_count, label_balance, nan_rates, fold_sizes, sufficient."""
```

Extract the checking logic into testable functions in `check_data_sufficiency.py` (e.g., `check_bar_count(n_bars)`, `check_nan_rates(df, features)`) so tests don't need I/O.

**Verification:** `cd scripts && python -m pytest tests/test_data_sufficiency.py -v`

---

### Task 3: Document data sufficiency requirements

**Create:** `docs/research/new/ml_specs/DATA_REQUIREMENTS.md` (~60 lines)

Content:
1. Table of 4 thresholds with rationale (samples-per-parameter for logistic, early-stopping needs for LightGBM)
2. Model complexity table: which models are safe/risky/impossible given bar count ranges
3. How to run the check: `python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features`
4. What to do if checks fail (wait for more data, reduce folds, drop features)

---

### Task 4: Update README.md testing section

**Modify:** `README.md`

Find the `## Testing` section. Add a new entry after the existing test commands:

```markdown
# Data sufficiency (run before ML training)
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features
```

**Verification:** `grep "check_data_sufficiency" README.md` returns 1 match.

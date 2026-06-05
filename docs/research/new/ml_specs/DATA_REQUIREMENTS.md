# Data Requirements for ML Training

Minimum data thresholds that must be met before training any ML algorithm.

---

## Threshold Table

| Check | Threshold | Rationale | On Failure |
|-------|-----------|-----------|------------|
| Bar count | >= 4,000 (5-min bars) | ~14 days of market data. LogReg with 7 features needs ~70 samples minimum; 4000 provides walk-forward headroom | FAIL — wait for more data |
| Label balance | 40–60% positive | Extreme imbalance biases classifiers toward majority class and inflates accuracy | WARNING — check forward return horizon |
| Feature NaN rate | < 5% per feature | High NaN rates reduce effective training samples and can bias imputation | FAIL — investigate feature computation |
| Walk-forward fold size | >= 500 bars per fold (4 folds) | Each fold needs enough samples for stable Sharpe estimation; 500 bars ≈ 1.7 days | FAIL — reduce folds or wait for data |

---

## Model Complexity vs Data

| Bar Count | LogReg (7 features) | LightGBM (15 leaves) | LightGBM (31 leaves) | KNN (buffer) |
|-----------|---------------------|----------------------|----------------------|--------------|
| < 2,000 | Risky | Impossible | Impossible | Insufficient |
| 2,000–4,000 | Safe | Risky | Impossible | Marginal |
| 4,000–8,000 | Safe | Safe | Risky | Safe |
| > 8,000 | Safe | Safe | Safe | Safe |

Rule of thumb: need ~200 samples per effective parameter. LogReg with 7 features = 8 parameters → ~1600 minimum. LightGBM with `num_leaves=15` has ~15 leaf values × features → needs more data.

---

## Running the Check

```bash
# Single symbol
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features

# JSON output (for scripts)
python scripts/check_data_sufficiency.py --symbol BTC --data-dir data/features --json
```

Exit code 0 = sufficient, 1 = insufficient.

---

## What to Do If Checks Fail

| Failure | Remediation |
|---------|-------------|
| Bar count too low | Wait for more ingestion time. At 3 symbols × 5-min bars, 1 day ≈ 288 bars/symbol |
| Label imbalance | Try a different forward return horizon (10, 20, 40 bars). Check for data gaps |
| High NaN rate | Check if the feature category is enabled in `config/ing.toml`. Look for upstream computation errors in Rust features |
| Fold size too small | Reduce from 4 folds to 3 or 2. Accept wider confidence intervals on OOS metrics |

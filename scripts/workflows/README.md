# NAT Workflow Scripts

Automated workflows for validation, testing, and analysis.

## Available Workflows

### 1. Complete Validation (`validate_all.sh`)

Runs full validation pipeline after collecting 2+ weeks of data.

**Steps:**
1. Validate data quality
2. Train GMM clustering
3. Analyze cluster quality
4. Run hypothesis tests
5. Generate summary report

**Usage:**
```bash
chmod +x scripts/workflows/validate_all.sh
./scripts/workflows/validate_all.sh
```

**Time:** 30-90 minutes (depending on data volume)

## Workflow Guide

### After Data Collection (2+ weeks)

```bash
# 1. Run complete validation
./scripts/workflows/validate_all.sh

# 2. Review results
cat output/cluster_analysis/quality_report.json
cat output/hypothesis_tests/results.json

# 3. Make decision
# - If 3+ hypotheses pass → Proceed to Phase 2 (HMM + strategies)
# - If 1-2 pass → Iterate configuration
# - If 0 pass → Rethink approach
```

### For Quick Iteration

```bash
# Just train and analyze clusters
make train_gmm_auto
make analyze_clusters_gmm

# Just run hypothesis tests
make test_hypotheses
```

### For Specific Analysis

```bash
# Validate recent data only
make validate_data_recent HOURS=48

# Analyze specific model
make analyze_clusters_gmm MODEL=models/regime_gmm.json
```

## Output Directories

All workflows write to:
- `output/cluster_analysis/` - Cluster quality metrics and visualizations
- `output/hypothesis_tests/` - Hypothesis test results
- `logs/` - Execution logs

## Next Steps

See `docs/NEXT_STEPS_ROADMAP.md` for complete roadmap and next actions.

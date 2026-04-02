#!/bin/bash
# Complete validation workflow
# Run this after collecting 2+ weeks of data

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          NAT VALIDATION WORKFLOW - PHASE 1                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Step 1: Validate Data Quality
echo ""
echo "[Step 1/6] Validating data quality..."
make validate_data

# Check data volume
DATA_HOURS=$(python -c "
import polars as pl
from pathlib import Path
files = list(Path('./data/features').glob('*.parquet'))
if not files:
    print(0)
else:
    df = pl.concat([pl.read_parquet(f) for f in files])
    hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    print(int(hours))
" 2>/dev/null || echo "0")

if [ "$DATA_HOURS" -lt 336 ]; then
    echo "⚠️  Warning: Only $DATA_HOURS hours of data (need 336+ hours / 2 weeks)"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborting. Collect more data first."
        exit 1
    fi
fi

echo "✅ Data quality validated: $DATA_HOURS hours available"

# Step 2: Train GMM Clustering
echo ""
echo "[Step 2/6] Training GMM regime classifier..."
make train_gmm_auto

echo "✅ GMM trained"

# Step 3: Analyze Cluster Quality
echo ""
echo "[Step 3/6] Analyzing cluster quality..."
make analyze_clusters_gmm

echo "✅ Cluster analysis complete"

# Step 4: Test Hypothesis
echo ""
echo "[Step 4/6] Running hypothesis tests (this may take 30-60 minutes)..."
make test_hypotheses || echo "⚠️ Some hypothesis tests failed or not implemented"

# Step 5: Generate Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          VALIDATION WORKFLOW COMPLETE                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Check results:"
echo "   - Cluster analysis: output/cluster_analysis/"
echo "   - Hypothesis tests: output/hypothesis_tests/"
echo ""
echo "🎯 Next steps:"
echo "   1. Review cluster quality metrics"
echo "   2. Check which hypotheses passed"
echo "   3. If 3+ hypotheses pass → Implement HMM + strategies"
echo "   4. If <3 pass → Iterate config or collect more data"

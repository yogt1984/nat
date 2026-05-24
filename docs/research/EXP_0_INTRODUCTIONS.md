# EXP-0: Introduction — One-Week Data Collection & Validation

## Goal

Collect 7 days of continuous Hyperliquid microstructure data (BTC, ETH, SOL at 100ms),
validate quality, and run the first real profiling experiment.

**Start date**: 2026-05-05
**End date**: 2026-05-12
**Machine**: su-35 (this machine)

## Expected Output

- ~18M rows of tick data (~2.6M/day * 7 days)
- ~672 fifteen-minute bars per symbol
- Validated parquet files in `data/features/`
- First profiling result with quality gate verdict (GO/PIVOT/COLLECT/DROP)

---

## Day 0: Setup & Start Collection (Today)

### Terminal 1: Your work terminal (keep using normally)

```bash
# Pull latest code
cd ~/nat
git pull origin master

# Build the release binary (takes ~2 min with LTO)
make release

# Verify the binary exists
ls -lh rust/target/release/ing
```

### Terminal 2: Open a new terminal for the ingestor

Open a new terminal tab/window (Ctrl+Shift+T in most terminals), then:

```bash
# Start a tmux session so the ingestor survives if you close the terminal
tmux new-session -s ingestor

# Inside tmux: start the ingestor
cd ~/nat
make run

# You should see output like:
#   Running ingestor...
#   [INFO] Connected to wss://api.hyperliquid.xyz/ws
#   [INFO] Subscribed: BTC, ETH, SOL
#   [INFO] Writing to ../data/features/2026-05-05/...

# DETACH from tmux (ingestor keeps running in background):
# Press: Ctrl+B, then D
```

You're done. Go back to Terminal 1 and use your computer normally.
The ingestor runs in the background writing ~30 rows/sec to parquet files.

---

## Day 1-6: Monitor (Once Daily, ~2 Minutes)

Each day, open Terminal 1 and run:

```bash
cd ~/nat

# Quick health check: is the ingestor still running?
pgrep -f "target/release/ing" && echo "RUNNING" || echo "DEAD - restart needed"

# Check data size and file count
du -sh data/features/
find data/features/ -name "*.parquet" | wc -l

# Check today's files are growing
ls -lh data/features/$(date +%Y-%m-%d)/

# Validate data quality on the last 24h
make validate_data_recent HOURS=24
```

### If the ingestor died (you see "DEAD"):

```bash
# Reattach to tmux to see what happened
tmux attach -t ingestor

# If the session is gone, create a new one
tmux new-session -s ingestor
cd ~/nat
make run
# Ctrl+B, then D to detach
```

### If your machine rebooted:

```bash
# tmux session is gone after reboot, start fresh
tmux new-session -s ingestor
cd ~/nat
make run
# Ctrl+B, then D to detach
```

---

## Day 3: Mid-Week Validation Checkpoint

After 3 days (~7.8M rows), run a deeper check:

```bash
cd ~/nat

# Full validation (all files, all checks)
make validate_data

# Schema scan (verify all 191 features present, no drift)
make scan_schema

# Quick row count
python3 -c "
from scripts.cluster_pipeline.loader import load_parquet
df = load_parquet('data/features')
print(f'Total rows: {len(df):,}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
print(f'Symbols: {df[\"symbol\"].nunique() if \"symbol\" in df.columns else \"N/A\"}')
"
```

**What to look for:**
- Total rows should be ~7-8M after 3 days
- NaN ratio should be < 1% for base features (categories 1-10)
- No continuity gaps > 5 seconds
- All 3 symbols present (BTC, ETH, SOL)

**If rows are much lower than expected**: the WebSocket may be dropping.
Check `tmux attach -t ingestor` for error messages.

---

## Day 7: End of Experiment

### Step 1: Stop the ingestor

```bash
# Reattach to tmux
tmux attach -t ingestor

# Stop the ingestor: Ctrl+C
# Exit tmux: type 'exit'
```

### Step 2: Final validation

```bash
cd ~/nat

# Full validation
make validate_data

# Schema scan
make scan_schema

# Comprehensive data summary
python3 -c "
from scripts.cluster_pipeline.loader import load_parquet
df = load_parquet('data/features')
print(f'Total rows:    {len(df):,}')
print(f'Date range:    {df.index.min()} to {df.index.max()}')
print(f'NaN ratio:     {df.isna().mean().mean():.4f}')
print(f'File size:     ', end='')
import subprocess
subprocess.run(['du', '-sh', 'data/features/'])
"
```

### Step 3: Run profiling pipeline

```bash
cd ~/nat

# Run the full profiling analysis
make pipeline_analyze

# Or run it manually with more control:
python3 -c "
from scripts.cluster_pipeline.loader import load_parquet
from scripts.cluster_pipeline.preprocess import aggregate_bars
from scripts.cluster_pipeline.derivatives import generate_derivatives
from scripts.cluster_pipeline.hierarchy import profile

# Load all data
df = load_parquet('data/features')
print(f'Loaded {len(df):,} rows')

# Aggregate to 15-min bars
bars = aggregate_bars(df, bar_minutes=15)
print(f'Aggregated to {len(bars)} bars')

# Generate derivatives
deriv = generate_derivatives(bars, vector='entropy', include_spectral=True)
print(f'Derivatives: {deriv.derivatives.shape[1]} features, warmup={deriv.warmup_rows}')

# Profile (this is the main experiment)
result = profile(deriv.derivatives, warmup_rows=deriv.warmup_rows)
print(f'Macro k={result.macro.k}, early_exit={result.macro.early_exit}')
print(f'Quality: silhouette={result.macro.quality.silhouette:.3f}')
print(f'Stability: ARI={result.macro.stability.mean_ari:.3f}')
print(f'Training stats: {result.training_stats}')
"
```

### Step 4: Run validation gates

```bash
python3 -c "
from scripts.cluster_pipeline.loader import load_parquet
from scripts.cluster_pipeline.preprocess import aggregate_bars
from scripts.cluster_pipeline.derivatives import generate_derivatives
from scripts.cluster_pipeline.hierarchy import profile
from scripts.cluster_pipeline.validate import validate
import numpy as np

df = load_parquet('data/features')
bars = aggregate_bars(df, bar_minutes=15)
deriv = generate_derivatives(bars, vector='entropy', include_spectral=True)
result = profile(deriv.derivatives, warmup_rows=deriv.warmup_rows)

# Need prices for Q2 (predictive quality)
prices = bars['raw_midprice_mean'].values if 'raw_midprice_mean' in bars.columns else np.ones(len(bars))

verdict = validate(result, prices)
print(verdict.summary)
print(f'Overall: {verdict.overall}')
print(f'Per-state: {verdict.per_state_verdicts}')
"
```

### Step 5: Open the cluster analysis notebook (optional, interactive)

```bash
cd ~/nat
jupyter notebook notebooks/cluster_analysis.ipynb
```

---

## Tmux Cheat Sheet

```
tmux new-session -s ingestor     Create a new named session
tmux attach -t ingestor          Reattach to a running session
tmux ls                          List all sessions
Ctrl+B, then D                   Detach (session keeps running)
Ctrl+B, then [                   Scroll mode (q to exit)
Ctrl+C                           Stop the running program
exit                             Kill the session
```

---

## Decision After Day 7

| Verdict | Meaning | Next Action |
|---------|---------|-------------|
| **GO** | Clusters are real, predictive, and tradeable | Start EXP-1: regime-conditioned strategies |
| **PIVOT** | Clusters exist but aren't tradeable yet | Try different bar sizes (5min, 1h), different vectors |
| **COLLECT** | Not enough data or signal too weak | Extend to 14-day collection |
| **DROP** | No structure found | Try different feature subsets or abandon clustering approach |

---

## Troubleshooting

**"make run" fails with "No such file or directory"**
→ Run `make release` first. The binary needs to be built.

**Ingestor starts but no data appears**
→ Check internet connectivity: `curl -s https://api.hyperliquid.xyz/info -d '{"type":"meta"}' | head -c 100`
→ Check disk space: `df -h .`

**validate_data shows high NaN ratio**
→ First few hours always have high NaN (features need warmup). Check again after 6+ hours.

**"Permission denied" on parquet files**
→ `chmod -R u+rw data/features/`

**Machine is slow while ingestor runs**
→ The ingestor uses <5% CPU. If slow, check other processes: `htop`

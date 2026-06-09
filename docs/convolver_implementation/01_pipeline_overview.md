# Step 1: Pipeline Overview

## Context

The convolver discovers data-driven pattern kernels from OHLCV candle data via
event-aligned SVD. This is the entry point — understand the full flow before
implementing any stage.

## Data Ingestion

Historical OHLCV data is fetched via CCXT from Hyperliquid directly:

```python
import ccxt
hl = ccxt.hyperliquid()
candles = hl.fetch_ohlcv("BTC/USDC:USDC", timeframe="1m", limit=1000, params={"since": since_ms})
# Returns: [[timestamp, O, H, L, C, V], ...]
```

Why CCXT: unified API, handles pagination and rate limits, Hyperliquid supported
natively, returns OHLCV arrays directly. Same venue as production (not Binance
proxies) so market microstructure matches.

Pagination: loop `fetch_ohlcv()` with `since` parameter to backfill months of
60s candles. Minimum data: 50K candles (~35 days) for sample size, 6+ months
for regime coverage (see `docs/research/new/convolver_data_analysis.txt`).

## Pipeline Stages

```
[0] CCXT fetch_ohlcv             Hyperliquid historical 1m candles via REST
    |
[1] OHLCV decomposition          4 channels: body, upper_wick, lower_wick, volume
[2] Event detection              6 boolean masks (breakout/turtle/trap x bull/bear)
[3] ATR normalization            center by mean, scale by ATR (shape-preserving)
[4] Matrix assembly              stack W-candle windows around events -> X in R^{n x W}
[5] Thin SVD                     X = U S V^T, right singular vectors V_k are shapes
[6] IC gate                      keep components whose loadings predict forward returns
[7] BH-FDR correction            control false discovery rate across all tests
[8] Kernel library output        .npz (arrays) + .json (metadata)
```

Supporting stages (run during discovery but not in the core linear flow):
- Walk-forward validation: 60/40 temporal split, IC decay ratio (step 13)
- Rolling SVD stability: cosine similarity of shapes across rolling windows (step 13)
- Analytical basis alignment: interpretability diagnostic (step 12)

Separate runtime component:
- Online scoring: cosine similarity matched filter on 100ms ticks (step 14)

One run per (symbol, candle_freq) pair. All event types and channels processed
in a single pass.

## Key Files

| File | Role |
|------|------|
| `scripts/analysis/convolver_discovery.py` | Offline discovery (stages 0-8 + validation) |
| `scripts/algorithms/convolver_kernels.py` | Shared math, data structures, persistence |
| `scripts/algorithms/convolver.py` | Online scoring algorithm (production) |
| `docs/research/convolver_method.tex` | Full mathematical specification |

## Anti-Overfitting Architecture

Three independent safeguards:

| Layer | Mechanism | Failure Mode |
|-------|-----------|-------------|
| SVD | Return-blind decomposition | Data snooping |
| BH-FDR | Multiple testing correction at q=0.05 | False discovery |
| Walk-forward | 60/40 OOS IC validation | In-sample overfitting |

## Verification

```bash
# Full discovery run (BTC, 60s candles)
python scripts/analysis/convolver_discovery.py \
    --data-dir data/features --symbol BTC --candle-freq 60s --window 20 --save

# Check outputs exist
ls models/convolver_kernels.{npz,json}
cat models/convolver_kernels.json | python -m json.tool
```

Expected: kernel library with 6+ kernels, all turtle/trap types (no breakouts).

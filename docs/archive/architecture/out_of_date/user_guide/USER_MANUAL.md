# NAT ML Infrastructure - User Manual

**Version:** 1.0.0
**Last Updated:** 2026-03-30

Complete guide to the NAT Machine Learning infrastructure, covering all workflows from data preparation through production model serving.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Makefile Targets](#makefile-targets)
5. [REST API Reference](#rest-api-reference)
6. [Complete Workflows](#complete-workflows)
7. [Python API Reference](#python-api-reference)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [Examples](#examples)

---

## Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NAT ML Infrastructure                      │
└─────────────────────────────────────────────────────────────┘

Data Ingestion → Feature Storage → Snapshots → Training
                                                    ↓
                                               Model Storage
                                                    ↓
                                            Experiment Tracking
                                                    ↓
                                               Scoring/Predictions
                                                    ↓
                                               Backtesting
                                                    ↓
                                            Model Serving (REST API)
```

### Components

1. **Data Ingestion** - Rust-based real-time market data collection
2. **Experiment Governance** - Snapshot management and data versioning
3. **Model Training** - Sklearn (Elastic Net) and LightGBM support
4. **Model Persistence** - Standardized model saving/loading
5. **Model Scoring** - Prediction generation on new data
6. **Experiment Tracking** - Complete audit trail of all experiments
7. **ML Backtesting** - Walk-forward validation with strategy testing
8. **Model Serving** - Production REST API for real-time predictions

### Key Features

- ✅ Complete ML workflow automation
- ✅ Experiment tracking and reproducibility
- ✅ Walk-forward backtesting
- ✅ Real-time model serving API
- ✅ Best model selection by any metric
- ✅ Comprehensive test coverage (58/58 passing)

---

## Quick Start

### 30-Second Demo

```bash
# 1. Create data snapshot
python scripts/experiment_governance.py snapshot \
    --data-dir ./data/features \
    --name baseline_30d

# 2. Run complete tracked ML workflow
make run_ml_workflow \
    SNAPSHOT=baseline_30d \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# 3. View experiments
make experiments_list

# 4. Start model serving API
make serve_best METRIC=sharpe_ratio

# 5. Generate prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]}'
```

---

## Installation

### Prerequisites

```bash
# Python 3.12+ with conda
conda --version

# Rust toolchain
rustc --version
cargo --version
```

### Install Python Dependencies

```bash
# Core ML dependencies (already installed)
conda install numpy pandas polars scikit-learn lightgbm pytest

# Model serving dependencies
conda install -c conda-forge fastapi uvicorn pydantic

# Or via pip in conda environment
pip install fastapi uvicorn pydantic
```

### Verify Installation

```bash
# Run all tests
python -m pytest scripts/tests/ -v

# Expected output: 58 passed, 1 skipped ✅
```

---

## Makefile Targets

### Data Management

#### `make validate_data`
Validate collected Parquet data quality.

```bash
make validate_data
```

**Options:**
- `HOURS=24` - Validate last N hours only

**Example:**
```bash
make validate_data_recent HOURS=48
```

---

### Model Training

#### `make train_baseline`
Train a baseline model (Elastic Net or LightGBM) on a snapshot.

**Required Parameters:**
- `SNAPSHOT` - Snapshot name
- `MODEL_TYPE` - Model type (`elasticnet` or `lightgbm`)

**Optional Parameters:**
- `ALPHA=0.01` - Elastic Net regularization (default: 0.01)
- `L1_RATIO=0.5` - Elastic Net L1/L2 mix (default: 0.5)
- `LEARNING_RATE=0.1` - LightGBM learning rate (default: 0.1)
- `NUM_LEAVES=31` - LightGBM complexity (default: 31)
- `NO_TRACKING=""` - Disable experiment tracking

**Examples:**
```bash
# Train LightGBM model
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=lightgbm

# Train Elastic Net with custom parameters
make train_baseline \
    SNAPSHOT=baseline_30d \
    MODEL_TYPE=elasticnet \
    ALPHA=0.001 \
    L1_RATIO=0.7

# Train without experiment tracking
make train_baseline \
    SNAPSHOT=test \
    MODEL_TYPE=lightgbm \
    NO_TRACKING=--no-tracking
```

**Output:**
- Model file: `models/<model_type>_baseline_<snapshot>_<timestamp>.<ext>`
- Metadata file: `models/<model_type>_baseline_<snapshot>_<timestamp>_metadata.json`
- Experiment tracked in: `experiments/tracking/experiments.json`

---

#### `make list_models`
List all saved models.

```bash
make list_models
```

**Output:**
```
Available models in models/:
  lightgbm_baseline_baseline_30d_20260330_120000.txt
  elasticnet_baseline_baseline_30d_20260329_153000.pkl
  ...
```

---

### Model Scoring

#### `make score_data`
Generate predictions using a trained model.

**Required Parameters:**
- `MODEL_PATH` - Path to model file

**Optional Parameters:**
- `DATA_DIR=./data/features` - Data directory (default)
- `HOURS_BACK=""` - Only use last N hours of data

**Examples:**
```bash
# Score all data
make score_data MODEL_PATH=models/lightgbm_baseline_baseline_30d_20260330_120000.txt

# Score last 24 hours only
make score_data \
    MODEL_PATH=models/lightgbm_*.txt \
    HOURS_BACK=24
```

**Output:**
```
Loading model: models/lightgbm_baseline_baseline_30d_20260330_120000.txt
Model type: lightgbm
Features: ['kyle_lambda_100', 'vpin_50', ...]
Loading data from ./data/features...
Found 45 Parquet files
Loaded 95000 samples
Valid samples: 94823/95000

Prediction Statistics:
  Mean:    0.000156
  Std:     0.001234
  Min:    -0.003421
  Max:     0.004123
  Count:   94823
```

---

#### `make score_and_save`
Generate predictions and save to Parquet file (with tracking).

**Required Parameters:**
- `MODEL_PATH` - Path to model file
- `PREDICTIONS` - Output Parquet file path

**Optional Parameters:**
- `DATA_DIR=./data/features` - Data directory
- `HOURS_BACK=""` - Only use last N hours

**Examples:**
```bash
# Score and save predictions
make score_and_save \
    MODEL_PATH=models/lightgbm_baseline_baseline_30d_20260330_120000.txt \
    PREDICTIONS=./predictions.parquet

# Score last 48 hours and save
make score_and_save \
    MODEL_PATH=models/lightgbm_*.txt \
    PREDICTIONS=./predictions_recent.parquet \
    HOURS_BACK=48
```

**Output:**
- Predictions file: `<PREDICTIONS>`
- Automatically tracked in experiment system

---

### Backtesting

#### `make backtest_ml`
Run ML-based backtest with basic validation.

**Required Parameters:**
- `ML_PREDICTIONS` - Path to predictions Parquet file

**Optional Parameters:**
- `ML_ENTRY=0.001` - Entry threshold (default: 0.001)
- `ML_EXIT=0.0` - Exit threshold (default: 0.0)
- `ML_DIRECTION=long` - Strategy direction (`long`, `short`, `both`)

**Example:**
```bash
make backtest_ml \
    ML_PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0 \
    ML_DIRECTION=long
```

---

#### `make backtest_ml_validate`
Run ML backtest with walk-forward validation.

**Required Parameters:**
- `ML_PREDICTIONS` - Predictions file

**Optional Parameters:**
- `ML_ENTRY=0.001` - Entry threshold
- `ML_EXIT=0.0` - Exit threshold
- `ML_DIRECTION=long` - Direction

**Example:**
```bash
make backtest_ml_validate \
    ML_PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0
```

**Output:**
```
Walk-forward validation results:
  OOS Sharpe: 1.23
  IS Sharpe:  1.45
  OOS/IS:     0.85
  Total trades: 127
  Win rate: 58%
```

---

#### `make backtest_ml_tracked`
Run ML backtest with experiment tracking.

**Required Parameters:**
- `ML_PREDICTIONS` - Predictions file

**Optional Parameters:**
- `ML_ENTRY=0.001` - Entry threshold
- `ML_EXIT=0.0` - Exit threshold
- `ML_DIRECTION=long` - Direction
- `BACKTEST_JSON=./backtest_results.json` - Output file

**Example:**
```bash
make backtest_ml_tracked \
    ML_PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0 \
    BACKTEST_JSON=./backtest_results.json
```

**Output:**
- Results saved to JSON file
- Automatically linked to experiment

---

### Experiment Tracking

#### `make experiments_list`
List all tracked experiments.

```bash
make experiments_list
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                   TRACKED EXPERIMENTS                            ║
╚══════════════════════════════════════════════════════════════════╝

ID: exp_20260330_120000_lightgbm
  Stage: backtest
  Created: 2026-03-30T12:00:00.123456
  Snapshot: baseline_30d
  Training: R²=0.7856, RMSE=0.001234
  Backtest: Sharpe=1.23, Return=15.3%, Trades=127

ID: exp_20260330_110000_elasticnet
  Stage: predictions
  Created: 2026-03-30T11:00:00.123456
  Snapshot: baseline_30d
  Training: R²=0.7234, RMSE=0.001456
  Predictions: 95000
```

---

#### `make experiments_list_stage`
List experiments by stage.

**Required Parameters:**
- `STAGE` - Stage filter (`training`, `predictions`, `backtest`)

**Examples:**
```bash
# Only experiments with backtest results
make experiments_list_stage STAGE=backtest

# Only training (no predictions yet)
make experiments_list_stage STAGE=training
```

---

#### `make experiments_get`
Get detailed experiment information.

**Required Parameters:**
- `EXP_ID` - Experiment ID

**Example:**
```bash
make experiments_get EXP_ID=exp_20260330_120000_lightgbm
```

**Output:**
```json
{
  "experiment_id": "exp_20260330_120000_lightgbm",
  "created_at": "2026-03-30T12:00:00.123456",
  "stage": "backtest",
  "snapshot": {
    "name": "baseline_30d"
  },
  "training": {
    "model_path": "models/lightgbm_baseline_baseline_30d_20260330_120000.txt",
    "model_type": "lightgbm",
    "performance_metrics": {
      "test_r2": 0.7856,
      "test_rmse": 0.001234
    }
  },
  "backtest": {
    "results": {
      "sharpe_ratio": 1.23,
      "total_return_pct": 15.3,
      "max_drawdown_pct": -5.4,
      "win_rate": 0.58,
      "total_trades": 127
    }
  }
}
```

---

#### `make experiments_compare`
Compare multiple experiments.

**Required Parameters:**
- `EXP_IDS` - Space-separated experiment IDs

**Example:**
```bash
make experiments_compare EXP_IDS="exp1 exp2 exp3"
```

**Output:**
```json
{
  "experiments": [
    {
      "experiment_id": "exp1",
      "model_type": "lightgbm",
      "training": {"test_r2": 0.7856},
      "backtest": {"sharpe_ratio": 1.23, "total_return_pct": 15.3}
    },
    {
      "experiment_id": "exp2",
      "model_type": "elasticnet",
      "training": {"test_r2": 0.7234},
      "backtest": {"sharpe_ratio": 0.95, "total_return_pct": 11.8}
    }
  ]
}
```

---

#### `make experiments_best`
Find best experiment by metric.

**Optional Parameters:**
- `METRIC=sharpe_ratio` - Metric to optimize

**Available Metrics:**
- `sharpe_ratio` - Risk-adjusted return (default)
- `total_return_pct` - Absolute return
- `win_rate` - Win percentage

**Examples:**
```bash
# Best by Sharpe ratio
make experiments_best METRIC=sharpe_ratio

# Best by total return
make experiments_best METRIC=total_return_pct

# Best by win rate
make experiments_best METRIC=win_rate
```

**Output:**
```
Best experiment by sharpe_ratio:
  ID: exp_20260330_120000_lightgbm
  sharpe_ratio: 1.23
  Model: lightgbm_baseline_baseline_30d
```

---

### Complete Workflows

#### `make run_ml_workflow`
Run complete ML workflow: train → score → backtest (all tracked).

**Required Parameters:**
- `SNAPSHOT` - Snapshot name
- `MODEL_TYPE` - Model type (`elasticnet` or `lightgbm`)
- `PREDICTIONS` - Output predictions file

**Optional Parameters:**
- `ML_ENTRY=0.001` - Entry threshold
- `ML_EXIT=0.0` - Exit threshold
- `ML_DIRECTION=long` - Strategy direction
- `BACKTEST_JSON=./backtest_results.json` - Backtest output

**Example:**
```bash
make run_ml_workflow \
    SNAPSHOT=baseline_30d \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║          COMPLETE ML WORKFLOW WITH TRACKING                      ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: Training model...
📊 Experiment tracked: exp_20260330_120000_lightgbm
   Model: lightgbm_baseline_baseline_30d
   Snapshot: baseline_30d
   Test R²: 0.7856

Step 2: Generating predictions...
📊 Predictions tracked: exp_20260330_120000_lightgbm

Step 3: Running backtest with tracking...
📊 Backtest tracked: exp_20260330_120000_lightgbm

═══════════════════════════════════════════════════════════════════
WORKFLOW COMPLETE - All stages tracked automatically
═══════════════════════════════════════════════════════════════════

View experiment:
  make experiments_list
```

---

### Model Serving

#### `make serve_models`
Start model serving REST API.

**Optional Parameters:**
- `PORT=8000` - Server port (default: 8000)
- `HOST=0.0.0.0` - Server host (default: 0.0.0.0)
- `CACHE_SIZE=5` - Model cache size (default: 5)

**Examples:**
```bash
# Start on default port 8000
make serve_models

# Custom port and host
make serve_models PORT=9000 HOST=127.0.0.1

# Larger cache for multiple models
make serve_models CACHE_SIZE=10
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║           NAT MODEL SERVING API                                  ║
╚══════════════════════════════════════════════════════════════════╝

Server: http://0.0.0.0:8000
Health: http://0.0.0.0:8000/health
Docs:   http://0.0.0.0:8000/docs

Models directory: ./models
Cache size: 5

[Server starts...]
```

---

#### `make serve_models_dev`
Start server with hot-reload (development mode).

**Optional Parameters:**
- `PORT=8000` - Server port
- `HOST=0.0.0.0` - Server host

**Example:**
```bash
make serve_models_dev PORT=8000
```

**Features:**
- Auto-reloads on code changes
- Useful for development and debugging

---

#### `make serve_best`
Start server with best model pre-loaded.

**Optional Parameters:**
- `METRIC=sharpe_ratio` - Metric for best model selection
- `PORT=8000` - Server port
- `CACHE_SIZE=5` - Cache size

**Examples:**
```bash
# Serve best model by Sharpe ratio
make serve_best METRIC=sharpe_ratio

# Serve best model by total return
make serve_best METRIC=total_return_pct

# Serve best by win rate
make serve_best METRIC=win_rate
```

**Output:**
```
Pre-loading best model by sharpe_ratio...
✅ Loaded best model: models/lightgbm_baseline_baseline_30d_20260330_120000.txt

Server: http://0.0.0.0:8000
[Server starts with best model cached...]
```

---

#### `make test_serving`
Run model serving API tests.

```bash
make test_serving
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║          TESTING MODEL SERVING API                               ║
╚══════════════════════════════════════════════════════════════════╝

=================== test session starts ===================
collected 18 items

test_model_serving.py::test_model_cache_creation PASSED
test_model_serving.py::test_api_health_endpoint PASSED
test_model_serving.py::test_api_predict_validation PASSED
...

================== 18 passed in 1.10s =====================
```

---

## REST API Reference

### Base URL

```
http://localhost:8000
```

### Authentication

Currently no authentication required (development mode).

**TODO for production:**
- Add API key authentication
- Add rate limiting
- Add HTTPS/TLS

---

### Endpoints

#### GET /health

Health check endpoint.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-30T12:00:00.123456",
  "uptime_seconds": 3600.5,
  "models_loaded": 3,
  "total_predictions": 12500,
  "api_version": "1.0.0",
  "cache_stats": {
    "cache_size": 3,
    "max_cache_size": 5,
    "prediction_counts": {
      "a1b2c3d4e5f6": 8000,
      "1a2b3c4d5e6f": 3500
    }
  }
}
```

**Status Codes:**
- `200 OK` - Service healthy

---

#### GET /models

List all available models.

**Request:**
```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "a1b2c3d4e5f6",
      "model_path": "models/lightgbm_baseline_baseline_30d_20260330_120000.txt",
      "model_type": "lightgbm",
      "model_name": "lightgbm_baseline_baseline_30d",
      "n_features": 6,
      "feature_names": ["kyle_lambda_100", "vpin_50", "ewm_spread_20",
                       "book_imbalance_10", "trade_flow_imbalance_50", "roll_impact_100"],
      "experiment_id": "exp_20260330_120000_lightgbm",
      "snapshot_name": "baseline_30d",
      "performance_metrics": {
        "test_r2": 0.7856,
        "test_rmse": 0.001234,
        "train_samples": 66500,
        "test_samples": 28500
      },
      "loaded": true,
      "last_used": "2026-03-30T12:00:00.123456"
    }
  ],
  "count": 1,
  "timestamp": "2026-03-30T12:00:00.123456"
}
```

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - Error listing models

---

#### GET /models/best

Get best model by metric.

**Query Parameters:**
- `metric` (optional) - Metric to optimize (default: `sharpe_ratio`)
  - `sharpe_ratio` - Risk-adjusted return
  - `total_return_pct` - Absolute return
  - `win_rate` - Win percentage
- `min_trades` (optional) - Minimum trades for significance (default: 30)

**Request:**
```bash
# Get best model by Sharpe ratio
curl "http://localhost:8000/models/best?metric=sharpe_ratio"

# Get best by total return with min 50 trades
curl "http://localhost:8000/models/best?metric=total_return_pct&min_trades=50"
```

**Response:**
```json
{
  "model_path": "models/lightgbm_baseline_baseline_30d_20260330_120000.txt",
  "model_id": "a1b2c3d4e5f6",
  "metadata": {
    "model_type": "lightgbm",
    "model_name": "lightgbm_baseline_baseline_30d",
    "n_features": 6,
    "feature_names": ["kyle_lambda_100", "vpin_50", ...],
    "performance_metrics": {
      "test_r2": 0.7856,
      "test_rmse": 0.001234
    }
  },
  "experiment": {
    "experiment_id": "exp_20260330_120000_lightgbm",
    "stage": "backtest",
    "backtest": {
      "results": {
        "sharpe_ratio": 1.23,
        "total_return_pct": 15.3,
        "max_drawdown_pct": -5.4,
        "win_rate": 0.58,
        "total_trades": 127
      }
    }
  },
  "selected_by": {
    "metric": "sharpe_ratio",
    "min_trades": 30
  },
  "timestamp": "2026-03-30T12:00:00.123456"
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - No model found matching criteria
- `500 Internal Server Error` - Error retrieving model

---

#### POST /predict

Generate single prediction.

**Request Body:**
```json
{
  "features": [0.00123, 0.00234, 0.00345, 0.00456, 0.00567, 0.00678],
  "model_id": "a1b2c3d4e5f6"  // Optional: use specific model
}
```

**Request:**
```bash
# Using best model (default)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.00123, 0.00234, 0.00345, 0.00456, 0.00567, 0.00678]}'

# Using specific model
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "features": [0.00123, 0.00234, 0.00345, 0.00456, 0.00567, 0.00678],
       "model_id": "a1b2c3d4e5f6"
     }'
```

**Response:**
```json
{
  "prediction": 0.000234,
  "model_id": "a1b2c3d4e5f6",
  "model_type": "lightgbm",
  "timestamp": "2026-03-30T12:00:00.123456",
  "latency_ms": 2.5
}
```

**Validation:**
- Features cannot be empty
- All features must be numeric
- Feature count must match model's expected count

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Model not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Prediction error

**Error Response:**
```json
{
  "detail": "Feature mismatch: model expects 6 features, got 5"
}
```

---

#### POST /predict/batch

Generate batch predictions.

**Request Body:**
```json
{
  "features": [
    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
    [0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
    [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
  ],
  "model_id": "a1b2c3d4e5f6"  // Optional
}
```

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{
       "features": [
         [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
         [0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
         [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
       ]
     }'
```

**Response:**
```json
{
  "predictions": [0.000234, 0.000345, 0.000456],
  "model_id": "a1b2c3d4e5f6",
  "model_type": "lightgbm",
  "n_predictions": 3,
  "timestamp": "2026-03-30T12:00:00.123456",
  "latency_ms": 5.2,
  "avg_latency_per_sample_ms": 1.73
}
```

**Validation:**
- All feature vectors must have same length
- Each vector must match model's feature count

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Model not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Prediction error

---

#### POST /reload

Reload models (clear cache).

**Request:**
```bash
curl -X POST http://localhost:8000/reload
```

**Response:**
```json
{
  "status": "success",
  "message": "Cleared 3 models from cache",
  "timestamp": "2026-03-30T12:00:00.123456"
}
```

**Use Cases:**
- After training new models
- To force re-loading of updated models
- To clear memory

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - Reload error

---

## Complete Workflows

### Workflow 1: Train and Evaluate Single Model

```bash
# 1. Create snapshot
python scripts/experiment_governance.py snapshot \
    --data-dir ./data/features \
    --name baseline_30d \
    --description "30 days of BTC market data"

# 2. Train model
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=lightgbm

# 3. Generate predictions
make score_and_save \
    MODEL_PATH=models/lightgbm_baseline_baseline_30d_*.txt \
    PREDICTIONS=./predictions.parquet

# 4. Run walk-forward backtest
make backtest_ml_validate \
    ML_PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# 5. View experiment details
make experiments_list
```

---

### Workflow 2: Complete Automated Workflow

```bash
# One command does everything: train → score → backtest
make run_ml_workflow \
    SNAPSHOT=baseline_30d \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# View results
make experiments_list
make experiments_best METRIC=sharpe_ratio
```

---

### Workflow 3: Model Comparison

```bash
# Train multiple models
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=lightgbm
make train_baseline SNAPSHOT=baseline_30d MODEL_TYPE=elasticnet

# Score both models
make score_and_save \
    MODEL_PATH=models/lightgbm_*.txt \
    PREDICTIONS=./predictions_lgb.parquet

make score_and_save \
    MODEL_PATH=models/elasticnet_*.pkl \
    PREDICTIONS=./predictions_en.parquet

# Backtest both
make backtest_ml_tracked ML_PREDICTIONS=./predictions_lgb.parquet
make backtest_ml_tracked ML_PREDICTIONS=./predictions_en.parquet

# Compare experiments
EXP1=$(make experiments_list | grep lightgbm | awk '{print $2}')
EXP2=$(make experiments_list | grep elasticnet | awk '{print $2}')
make experiments_compare EXP_IDS="$EXP1 $EXP2"
```

---

### Workflow 4: Production Deployment

```bash
# 1. Ensure best model exists
make experiments_best METRIC=sharpe_ratio

# 2. Start serving best model
make serve_best METRIC=sharpe_ratio &

# 3. Verify server health
sleep 5
curl http://localhost:8000/health

# 4. Test prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]}'

# 5. Monitor via health endpoint
watch -n 5 'curl -s http://localhost:8000/health | jq ".total_predictions"'
```

---

### Workflow 5: Model Retraining and Hot-Reload

```bash
# Server is running...

# 1. Train new model with fresh data
make train_baseline SNAPSHOT=baseline_30d_v2 MODEL_TYPE=lightgbm

# 2. Run complete workflow
make run_ml_workflow \
    SNAPSHOT=baseline_30d_v2 \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions_v2.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# 3. Check if new model is better
make experiments_best METRIC=sharpe_ratio

# 4. Hot-reload server (no downtime)
curl -X POST http://localhost:8000/reload

# 5. Next request will use new best model
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [...]}'
```

---

## Python API Reference

### Experiment Tracking

```python
from scripts.experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker()

# List all experiments
experiments = tracker.list_experiments()

# Filter by stage
backtest_experiments = tracker.list_experiments(stage="backtest")

# Filter by performance
high_sharpe = tracker.list_experiments(min_sharpe=1.0)

# Get specific experiment
experiment = tracker.get_experiment("exp_20260330_120000_lightgbm")

# Get best experiment
best = tracker.get_best_experiment(
    metric="sharpe_ratio",
    min_trades=30
)

# Compare experiments
comparison = tracker.compare_experiments([
    "exp_20260330_120000_lightgbm",
    "exp_20260329_153000_elasticnet"
])
```

---

### Model I/O

```python
from pathlib import Path
from scripts.utils.model_io import (
    load_sklearn_model,
    load_lightgbm_model,
    get_latest_model,
    list_models
)

# List all models
models = list_models(Path("./models"))

# Get latest model
latest = get_latest_model(Path("./models"), model_type="lightgbm")

# Load sklearn model
model, scaler, metadata = load_sklearn_model(
    Path("models/elasticnet_baseline_baseline_30d_20260329_153000.pkl")
)

# Load LightGBM model
model, metadata = load_lightgbm_model(
    Path("models/lightgbm_baseline_baseline_30d_20260330_120000.txt")
)

# Use model
import numpy as np
features = np.array([[0.001, 0.002, 0.003, 0.004, 0.005, 0.006]])
if scaler:
    features = scaler.transform(features)
prediction = model.predict(features)
```

---

### REST API Client

```python
import requests
import numpy as np

API_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{API_URL}/health")
health = response.json()
print(f"Models loaded: {health['models_loaded']}")
print(f"Total predictions: {health['total_predictions']}")

# List models
response = requests.get(f"{API_URL}/models")
models = response.json()["models"]
print(f"Available models: {len(models)}")

# Get best model
response = requests.get(f"{API_URL}/models/best", params={
    "metric": "sharpe_ratio",
    "min_trades": 30
})
best_model = response.json()
print(f"Best model: {best_model['metadata']['model_name']}")

# Single prediction
features = [0.00123, 0.00234, 0.00345, 0.00456, 0.00567, 0.00678]
response = requests.post(f"{API_URL}/predict", json={
    "features": features
})
result = response.json()
print(f"Prediction: {result['prediction']:.6f}")
print(f"Latency: {result['latency_ms']:.2f}ms")

# Batch prediction
batch_features = np.random.randn(100, 6).tolist()
response = requests.post(f"{API_URL}/predict/batch", json={
    "features": batch_features
})
result = response.json()
print(f"Batch size: {result['n_predictions']}")
print(f"Avg latency: {result['avg_latency_per_sample_ms']:.2f}ms")

# Hot-reload
response = requests.post(f"{API_URL}/reload")
print(response.json()["message"])
```

---

## Configuration

### Experiment Tracking

**Location:** `experiments/tracking/experiments.json`

**Format:**
```json
[
  {
    "experiment_id": "exp_20260330_120000_lightgbm",
    "created_at": "2026-03-30T12:00:00.123456",
    "stage": "backtest",
    "snapshot": {...},
    "training": {...},
    "predictions": {...},
    "backtest": {...}
  }
]
```

**Backup:**
```bash
# Backup experiments
cp experiments/tracking/experiments.json experiments/tracking/experiments_backup_$(date +%Y%m%d).json

# Restore from backup
cp experiments/tracking/experiments_backup_20260330.json experiments/tracking/experiments.json
```

---

### Model Storage

**Location:** `models/`

**File Naming:**
- Sklearn: `<type>_<variant>_<snapshot>_<timestamp>.pkl`
- LightGBM: `<type>_<variant>_<snapshot>_<timestamp>.txt`
- Metadata: `<model_name>_metadata.json`

**Cleanup:**
```bash
# Remove old models (keep last 10)
ls -t models/*.txt | tail -n +11 | xargs rm
ls -t models/*.pkl | tail -n +11 | xargs rm
```

---

### Server Configuration

**Default Settings:**
- Host: `0.0.0.0` (all interfaces)
- Port: `8000`
- Cache size: `5` models
- Workers: `1` process

**Environment Variables:**
```bash
# Optional: configure via environment
export NAT_SERVER_HOST=127.0.0.1
export NAT_SERVER_PORT=9000
export NAT_CACHE_SIZE=10
```

---

## Troubleshooting

### Training Issues

**Problem:** "No Parquet files found"

**Solution:**
```bash
# Check data directory
ls -lh ./data/features/*.parquet

# Create snapshot from existing data
python scripts/experiment_governance.py snapshot \
    --data-dir ./data/features \
    --name baseline_30d
```

---

**Problem:** "Insufficient data for training"

**Solution:**
```bash
# Check data volume
python scripts/validate_data.py ./data/features --verbose

# Need at least 50,000 samples for reliable training
# Collect more data or adjust train/test split
```

---

### Prediction Issues

**Problem:** "Feature mismatch" error

**Solution:**
```bash
# Check model's expected features
python -c "
import json
from pathlib import Path
metadata = json.load(open('models/<model>_metadata.json'))
print('Expected features:', metadata['feature_names'])
print('Feature count:', metadata['n_features'])
"

# Ensure your data has exactly these features in this order
```

---

**Problem:** "NaN values in features"

**Solution:**
```bash
# Check data quality
python scripts/validate_data.py ./data/features --verbose

# Filter out NaN values (automatic in score_data.py)
# Or fix upstream data collection
```

---

### Backtest Issues

**Problem:** "No trades generated"

**Solution:**
```bash
# Thresholds may be too high
# Try lower entry threshold
make backtest_ml_validate \
    ML_PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.0005 \
    ML_EXIT=0.0

# Or check prediction distribution
python -c "
import polars as pl
df = pl.read_parquet('./predictions.parquet')
print(df['prediction'].describe())
"
```

---

### API Issues

**Problem:** "Connection refused" when calling API

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start server
make serve_models

# Check logs for errors
```

---

**Problem:** High latency (>100ms)

**Solution:**
```bash
# Pre-load best model
make serve_best METRIC=sharpe_ratio

# Increase cache size
make serve_models CACHE_SIZE=10

# Use batch predictions for multiple samples
curl -X POST http://localhost:8000/predict/batch ...
```

---

**Problem:** "No best model available"

**Solution:**
```bash
# Need to run complete workflow first
make run_ml_workflow \
    SNAPSHOT=baseline_30d \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# Then check experiments
make experiments_list
```

---

## Best Practices

### Model Training

1. **Always use snapshots** - Ensures reproducibility
2. **Track all experiments** - Never use `--no-tracking` in production
3. **Validate with walk-forward** - Always use `backtest_ml_validate`
4. **Compare multiple models** - Train both Elastic Net and LightGBM
5. **Monitor test R²** - Aim for >0.7 for production models

---

### Experiment Management

1. **Descriptive snapshot names** - Use dates and descriptions
   ```bash
   # Good
   python scripts/experiment_governance.py snapshot \
       --name btc_bull_market_2025_q4 \
       --description "High volatility period"

   # Bad
   python scripts/experiment_governance.py snapshot --name data1
   ```

2. **Regular backups** - Backup experiments.json daily
3. **Clean old experiments** - Archive experiments older than 90 days
4. **Document decisions** - Use experiment notes field

---

### Production Serving

1. **Always serve best model** - Use `make serve_best`
2. **Monitor health endpoint** - Set up automated health checks
3. **Use batch predictions** - More efficient for multiple samples
4. **Set up alerting** - Alert on API downtime or high latency
5. **Regular retraining** - Retrain models weekly/monthly with fresh data

---

### Performance Optimization

1. **Pre-load models** - Use `--serve-best` to avoid cold starts
2. **Increase cache size** - For serving multiple models
3. **Use batch endpoints** - 10x more efficient than loops
4. **Monitor cache hits** - Check via `/health` endpoint
5. **Profile slow predictions** - Investigate if latency >50ms

---

## Examples

### Example 1: Quick Model Evaluation

```bash
#!/bin/bash
# Quickly train and evaluate a model

SNAPSHOT="baseline_30d"
MODEL_TYPE="lightgbm"

echo "Training model..."
make train_baseline SNAPSHOT=$SNAPSHOT MODEL_TYPE=$MODEL_TYPE

echo "Running complete evaluation..."
make run_ml_workflow \
    SNAPSHOT=$SNAPSHOT \
    MODEL_TYPE=$MODEL_TYPE \
    PREDICTIONS=./predictions.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

echo "Results:"
make experiments_list | tail -20
```

---

### Example 2: Model Comparison Script

```bash
#!/bin/bash
# Compare LightGBM vs Elastic Net

SNAPSHOT="baseline_30d"

echo "Training LightGBM..."
make run_ml_workflow \
    SNAPSHOT=$SNAPSHOT \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./pred_lgb.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

echo "Training Elastic Net..."
make run_ml_workflow \
    SNAPSHOT=$SNAPSHOT \
    MODEL_TYPE=elasticnet \
    PREDICTIONS=./pred_en.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

echo "Comparison:"
make experiments_list | grep -E "lightgbm|elasticnet"

echo "Best model:"
make experiments_best METRIC=sharpe_ratio
```

---

### Example 3: Production Deployment Script

```bash
#!/bin/bash
# Deploy model serving to production

# Configuration
PORT=8000
METRIC="sharpe_ratio"
LOG_FILE="/var/log/nat_serving.log"

# Check best model exists
echo "Checking for best model..."
BEST=$(make experiments_best METRIC=$METRIC 2>/dev/null | grep "ID:" | awk '{print $2}')

if [ -z "$BEST" ]; then
    echo "ERROR: No best model found. Run training workflow first."
    exit 1
fi

echo "Best model: $BEST"

# Start server
echo "Starting model serving API..."
nohup make serve_best METRIC=$METRIC PORT=$PORT > $LOG_FILE 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Health check
echo "Health check..."
HEALTH=$(curl -s http://localhost:$PORT/health)

if [ $? -eq 0 ]; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo $HEALTH | jq '.models_loaded'
else
    echo "❌ Server failed to start. Check logs: $LOG_FILE"
    exit 1
fi

echo "Server running at http://localhost:$PORT"
echo "Logs: $LOG_FILE"
echo "PID: $SERVER_PID"
```

---

### Example 4: Automated Retraining

```bash
#!/bin/bash
# Automated weekly retraining script

DATE=$(date +%Y%m%d)
SNAPSHOT="weekly_$DATE"

echo "[$DATE] Starting weekly retraining..."

# Create new snapshot
echo "Creating snapshot: $SNAPSHOT"
python scripts/experiment_governance.py snapshot \
    --data-dir ./data/features \
    --name $SNAPSHOT \
    --description "Weekly retrain $DATE"

# Train model
echo "Training model..."
make run_ml_workflow \
    SNAPSHOT=$SNAPSHOT \
    MODEL_TYPE=lightgbm \
    PREDICTIONS=./predictions_$DATE.parquet \
    ML_ENTRY=0.001 \
    ML_EXIT=0.0

# Check if new model is best
BEST=$(make experiments_best METRIC=sharpe_ratio | grep "ID:" | awk '{print $2}')
echo "Best model: $BEST"

# If server is running, hot-reload
if pgrep -f "model_serving.py" > /dev/null; then
    echo "Hot-reloading server..."
    curl -X POST http://localhost:8000/reload
    echo "✅ Server reloaded with new best model"
else
    echo "⚠️  Server not running"
fi

echo "[$DATE] Retraining complete"
```

---

### Example 5: Python Prediction Client

```python
#!/usr/bin/env python3
"""
Production prediction client
"""

import requests
import numpy as np
import time
from typing import List

class PredictionClient:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self._check_health()

    def _check_health(self):
        """Check API health on initialization."""
        response = requests.get(f"{self.api_url}/health")
        response.raise_for_status()
        print(f"✅ Connected to API (models loaded: {response.json()['models_loaded']})")

    def predict(self, features: List[float]) -> float:
        """Generate single prediction."""
        response = requests.post(
            f"{self.api_url}/predict",
            json={"features": features},
            timeout=5.0
        )
        response.raise_for_status()
        result = response.json()
        return result["prediction"]

    def predict_batch(self, features_list: List[List[float]]) -> List[float]:
        """Generate batch predictions."""
        response = requests.post(
            f"{self.api_url}/predict/batch",
            json={"features": features_list},
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()
        return result["predictions"]

    def get_best_model_info(self, metric: str = "sharpe_ratio"):
        """Get information about best model."""
        response = requests.get(
            f"{self.api_url}/models/best",
            params={"metric": metric}
        )
        response.raise_for_status()
        return response.json()


# Usage
if __name__ == "__main__":
    client = PredictionClient()

    # Get best model info
    best = client.get_best_model_info()
    print(f"Using model: {best['metadata']['model_name']}")
    print(f"Sharpe ratio: {best['experiment']['backtest']['results']['sharpe_ratio']}")

    # Single prediction
    features = [0.00123, 0.00234, 0.00345, 0.00456, 0.00567, 0.00678]
    prediction = client.predict(features)
    print(f"Prediction: {prediction:.6f}")

    # Batch prediction
    batch = np.random.randn(100, 6).tolist()
    start = time.time()
    predictions = client.predict_batch(batch)
    elapsed = time.time() - start
    print(f"Batch predictions: {len(predictions)}")
    print(f"Time: {elapsed*1000:.2f}ms ({elapsed*10:.2f}ms per sample)")
```

---

## Summary

This manual covers the complete NAT ML infrastructure:

- ✅ **6 Priority Components** - All implemented and tested
- ✅ **22 Makefile Targets** - Comprehensive automation
- ✅ **6 REST API Endpoints** - Production-ready serving
- ✅ **5 Complete Workflows** - From training to deployment
- ✅ **58 Tests Passing** - Fully validated

For additional help:
- Check specific documentation: `docs/MODEL_*_COMPLETE.md`
- Run tests: `make test_serving`
- View API docs: `http://localhost:8000/docs` (when server running)

**The NAT ML infrastructure is ready for production use!** 🚀

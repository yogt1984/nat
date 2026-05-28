# NAT Project Makefile
# Hyperliquid Market Data Ingestor

# Python interpreter — prefer 'python' (often conda/venv) over system 'python3'.
# Override with: make PYTHON=/usr/bin/python3
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null || echo python3)

# Shared variables
DATA ?= ./data/features
SYMBOL ?= BTC
HOURS ?= 24
FREQ ?= 1

# Backtest variables
STRATEGY ?= whale_flow_simple
ML_PREDICTIONS ?= ./predictions.parquet
ML_ENTRY ?= 0.001
ML_EXIT ?= 0.0
ML_DIRECTION ?= long
ML_ENTRY_Q ?= 0.75
ML_EXIT_Q ?= 0.50
BACKTEST_JSON ?= ./backtest_results.json
PREDICTIONS ?= ./predictions.parquet

# Model variables
SNAPSHOT ?= baseline_30d
MODEL_TYPE ?= elasticnet
MODEL_DIR ?= ./models
MODEL_PATH ?= models/latest.pkl

# Serving variables
PORT ?= 8000
HOST ?= 0.0.0.0
CACHE_SIZE ?= 5
METRIC ?= sharpe_ratio

# Config paths
PIPELINE_CONFIG ?= config/pipeline.toml
ALPHA_CONFIG ?= config/alpha.toml
DISCOVERY_CONFIG ?= config/discovery.toml

# Dashboard ports
DASHBOARD_PORT ?= 8050
AGENT_DASHBOARD_PORT ?= 8060

# Signal test variables
HORIZON ?= 3000
SPREAD_BPS ?= 1.0
REMOVE_LEAKY ?= 0

# Experiment tracking
STAGE ?= backtest
EXP_ID ?=
EXP_IDS ?=

# 15-min smoke test
SMOKE_DATA ?= ./data/features/$(shell date -u +%Y-%m-%d)
SMOKE_OUTPUT ?= ./reports/smoke_test
WINDOW ?=

# Trade visualization
TRADE_DATE ?=
TRADE_SYMBOL ?= BTC

# EAMM
EAMM_SYMBOL ?= BTC
EAMM_HORIZON ?= 3000
EAMM_MODE ?= regression
EAMM_GAMMA ?= 0.1
EAMM_QMAX ?= 1.0

# Include modular targets
include make/build.mk
include make/test.mk
include make/deploy.mk
include make/pipeline.mk
include make/alpha.mk
include make/experiment.mk

# =============================================================================
# HELP
# =============================================================================
.PHONY: help

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          NAT Project - Hyperliquid Market Data Ingestor          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [target] [OPTIONS]"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " RUNNING"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  run               Run ingestor (release build)"
	@echo "  run_and_serve     Run ingestor + dashboard at localhost:8080"
	@echo "  tunnel            Expose dashboard via cloudflare tunnel"
	@echo "  show              Show real-time features (SYMBOL=BTC FREQ=1)"
	@echo "  show_fast         Show features at 10 Hz"
	@echo "  show_hft          Show features at 50 Hz"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " DATA ANALYSIS"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  validate_data          Validate all collected Parquet data"
	@echo "  validate_data_recent   Validate last N hours (HOURS=24)"
	@echo "  explore                Launch Jupyter notebook for exploration"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " TESTING"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  test                    Run all unit tests"
	@echo "  test_verbose            Run tests with output"
	@echo "  test_hypotheses         Run H1-H5 hypothesis tests (DATA=./data/features)"
	@echo "  test_redis              Test Redis connection"
	@echo "  test_integration        Run full integration test suite"
	@echo "  test_agent              Run agent tests"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " ALPHA PIPELINE"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  alpha_pipeline        Start: screen -> combine -> ... -> deploy"
	@echo "  alpha_pipeline_resume Resume from saved state"
	@echo "  alpha_pipeline_force  Resume, forcing past a failed gate"
	@echo "  alpha_pipeline_status Show state"
	@echo "  alpha_pipeline_gates  Show gate verdicts"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " BACKTESTING"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  backtest              Run backtest (STRATEGY=... SYMBOL=BTC)"
	@echo "  backtest_validate     Walk-forward validation"
	@echo "  signal_test           Phase 1 signal existence test"
	@echo "  test_oos30            30-day OOS validation (5 algos)"
	@echo "  oos_validate          OOS validation (4 algos)"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " EXPERIMENTS"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  exp_start             Start ingestor in tmux"
	@echo "  exp_stop              Stop ingestor"
	@echo "  exp_status            Check health + data stats"
	@echo "  exp_check             Daily validation (HOURS=24)"
	@echo "  exp_analyze           Stop, validate, profile, quality gates"
	@echo "  15m                   Full 15-min experiment"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " AGENTS & SERVICES"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  agent_start           Start research agent daemon"
	@echo "  agent_stop            Stop agent"
	@echo "  agent_status          Show agent state"
	@echo "  api                   Start REST/WS API server"
	@echo "  alerts                Start Telegram alert service"
	@echo "  serve_all             Full stack (ingestor + API + alerts)"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " BUILD & DEV"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  build             Debug build"
	@echo "  release           Release build"
	@echo "  clean             Remove build artifacts"
	@echo "  fmt               Format code"
	@echo "  lint              Run clippy"
	@echo "  setup-python      pip install -e scripts/"
	@echo "  validate-config   Validate TOML config files"
	@echo "  help              Show this help"
	@echo ""

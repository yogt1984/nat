# NAT Project Makefile
# Hyperliquid Market Data Ingestor

.PHONY: all run run_and_serve tunnel test test_verbose test_hypotheses build release clean validate validate_all validate_api validate_positions validate_whales validate_entropy validate_data validate_data_recent show show_fast show_hft explore help fmt lint check api test_api test_redis test_integration alerts serve_all docker_build docker_up docker_down docker_logs train_gmm train_gmm_auto test_cluster_quality test_cluster_quality_cov analyze_clusters analyze_clusters_gmm analyze_all_symbols train_baseline list_models score_data score_and_save

# Default target: run the main ingestor
all: run

# =============================================================================
# MAIN TARGETS
# =============================================================================

# Build debug version
build:
	@echo "Building debug version..."
	cd rust && cargo build --bin ing

# Build release version (all binaries)
release:
	@echo "Building release version..."
	cd rust && cargo build --release --bin ing --bin validate_api --bin validate_positions --bin validate_whales --bin validate_entropy --bin show_features --bin test_hypotheses

# Run the main ingestor (requires config/ing.toml)
run: build
	@echo "Running ingestor..."
	cd rust && cargo run --bin ing -- ../config/ing.toml

# Run the ingestor with dashboard enabled
run_and_serve: release
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           STARTING INGESTOR WITH LIVE DASHBOARD                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Dashboard: http://localhost:8080"
	@echo ""
	@echo "To expose to internet, run in another terminal:"
	@echo "  make tunnel"
	@echo ""
	cd rust && ING_DASHBOARD_ENABLED=true ./target/release/ing ../config/ing.toml

# Expose dashboard to internet via cloudflare tunnel
tunnel:
	@echo "Starting cloudflare tunnel to localhost:8080..."
	@echo "Press Ctrl+C to stop"
	cloudflared tunnel --url http://localhost:8080

# =============================================================================
# DATA VALIDATION
# =============================================================================

# Validate collected data quality
validate_data:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                 VALIDATING COLLECTED DATA                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	python scripts/validate_data.py ./data/features --verbose

# Validate last N hours of data (default: 24)
HOURS ?= 24
validate_data_recent:
	@echo "Validating last $(HOURS) hours of data..."
	python scripts/validate_data.py ./data/features --hours $(HOURS) --verbose

# Launch Jupyter notebook for feature exploration
explore:
	@echo "Launching feature exploration notebook..."
	jupyter notebook notebooks/explore_features.ipynb

# =============================================================================
# TESTING
# =============================================================================

# Run all unit tests
test:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    RUNNING ALL UNIT TESTS                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd rust && cargo test --package ing
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    ALL UNIT TESTS PASSED                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# Run tests with verbose output
test_verbose:
	@echo "Running all tests (verbose)..."
	cd rust && cargo test --package ing -- --nocapture

# =============================================================================
# API VALIDATION (Skeptical Live API Tests)
# =============================================================================

# Run ALL validations (skeptical tests against live Hyperliquid API)
validate: release
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           RUNNING ALL VALIDATIONS (SKEPTICAL MODE)               ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "[1/4] API Connection Validation..."
	@cd rust && ./target/release/validate_api
	@echo ""
	@echo "[2/4] Position Tracking Validation..."
	@cd rust && ./target/release/validate_positions
	@echo ""
	@echo "[3/4] Whale Identification Validation..."
	@cd rust && ./target/release/validate_whales
	@echo ""
	@echo "[4/4] Tick Entropy Validation..."
	@cd rust && ./target/release/validate_entropy
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                  ALL VALIDATIONS COMPLETE                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# Alias for validate
validate_all: validate

# Individual validation targets
validate_api: release
	@echo "Running API connection validation..."
	cd rust && ./target/release/validate_api

validate_positions: release
	@echo "Running position tracking validation..."
	cd rust && ./target/release/validate_positions

validate_whales: release
	@echo "Running whale identification validation..."
	cd rust && ./target/release/validate_whales

validate_entropy: release
	@echo "Running tick entropy feature validation..."
	cd rust && ./target/release/validate_entropy

# =============================================================================
# REAL-TIME MONITORING
# =============================================================================

# Show real-time features (no file output, terminal only)
SYMBOL ?= BTC
FREQ ?= 1
show: release
	@echo "Starting real-time feature display..."
	@echo "  Symbol: $(SYMBOL)"
	@echo "  Frequency: $(FREQ) Hz"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd rust && ./target/release/show_features $(SYMBOL) $(FREQ)

# Quick frequency presets
show_fast: FREQ=10
show_fast: show

show_hft: FREQ=50
show_hft: show

# =============================================================================
# API SERVER
# =============================================================================

# Start API server
api: release_api
	@echo "Starting NAT API server..."
	@echo "  REST API: http://localhost:3000"
	@echo "  WebSocket: ws://localhost:3000/ws/stream/:symbol"
	@echo ""
	cd rust && ./target/release/nat-api

# Build API server (release)
release_api:
	@echo "Building API server..."
	cd rust && cargo build --release --bin nat-api

# Test API endpoints (requires running API server)
test_api:
	@echo "Testing API endpoints..."
	bash scripts/test_api.sh

# Test Redis connection and subscriptions
test_redis:
	@echo "Testing Redis connection..."
	@redis-cli ping && echo "✓ Redis is running" || (echo "✗ Redis not running" && exit 1)
	@echo ""
	@echo "Checking cached symbols..."
	@redis-cli KEYS "nat:latest:*" || true
	@echo ""
	@echo "To subscribe to features: redis-cli SUBSCRIBE nat:features:BTC"
	@echo "To subscribe to alerts:   redis-cli SUBSCRIBE nat:alerts"

# Run full integration test
test_integration:
	@echo "Running integration tests..."
	bash scripts/test_integration.sh

# =============================================================================
# TELEGRAM ALERTS
# =============================================================================

# Start Telegram alert service
alerts: release_api
	@echo "Starting Telegram Alert Service..."
	@echo ""
	@echo "Required environment variables:"
	@echo "  TELEGRAM_BOT_TOKEN - Bot token from @BotFather"
	@echo "  TELEGRAM_CHAT_ID   - Your chat ID"
	@echo ""
	cd rust && ./target/release/alert-service

# =============================================================================
# FULL STACK
# =============================================================================

# Start all services (ingestor + API + alerts)
# Requires: Redis running, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
serve_all: release release_api
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              STARTING FULL NAT STACK                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Services:"
	@echo "  - Ingestor:  Publishing features to Redis"
	@echo "  - API:       http://localhost:3000"
	@echo "  - Alerts:    Sending to Telegram"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Redis running on localhost:6379"
	@echo "  - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set"
	@echo ""
	@echo "Starting in tmux session 'nat'..."
	@echo "  (Use 'tmux attach -t nat' to reconnect)"
	@echo ""
	@tmux kill-session -t nat 2>/dev/null || true
	@tmux new-session -d -s nat -n ingestor 'cd rust && ./target/release/ing ../config/ing.toml; read'
	@tmux new-window -t nat -n api 'cd rust && ./target/release/nat-api; read'
	@tmux new-window -t nat -n alerts 'cd rust && ./target/release/alert-service; read'
	@tmux attach -t nat

# =============================================================================
# DOCKER
# =============================================================================

# Build all Docker images
docker_build:
	@echo "Building Docker images..."
	docker-compose build

# Start all services with Docker
docker_up:
	@echo "Starting NAT stack with Docker..."
	docker-compose up -d
	@echo ""
	@echo "Services:"
	@echo "  - API: http://localhost:3000"
	@echo "  - Redis: localhost:6379"
	@echo ""
	@echo "View logs: make docker_logs"

# Stop all Docker services
docker_down:
	@echo "Stopping NAT stack..."
	docker-compose down

# View Docker logs
docker_logs:
	docker-compose logs -f

# =============================================================================
# REGIME MODEL TRAINING
# =============================================================================

# Train GMM regime classifier on collected data
train_gmm:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║               TRAINING GMM REGIME CLASSIFIER                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p models
	python scripts/train_regime_gmm.py --data-dir $(DATA) --output-dir models

# Train with auto BIC selection
train_gmm_auto:
	@echo "Training GMM with auto-selected components..."
	@mkdir -p models
	python scripts/train_regime_gmm.py --data-dir $(DATA) --output-dir models --auto-select

# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

# Run hypothesis tests on collected data
DATA ?= ./data/features
test_hypotheses: release
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║         RUNNING HYPOTHESIS TESTING ON COLLECTED DATA             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	cd rust && ./target/release/test_hypotheses ../$(DATA)

# =============================================================================
# BACKTESTING
# =============================================================================

# Run backtest on collected data
STRATEGY ?= whale_flow_simple
backtest:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    RUNNING BACKTEST                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Strategy: $(STRATEGY)"
	@echo "Data: $(DATA)"
	@echo ""
	python scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) --strategy $(STRATEGY)

# Run walk-forward validation (recommended)
backtest_validate:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              WALK-FORWARD VALIDATION                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Strategy: $(STRATEGY)"
	@echo "Data: $(DATA)"
	@echo ""
	python scripts/run_backtest.py --data-dir $(DATA) --symbol $(SYMBOL) --strategy $(STRATEGY) --walk-forward

# List available strategies
backtest_list:
	@python scripts/run_backtest.py --list-strategies

# Run backtest tests
test_backtest:
	@echo "Running backtest tests..."
	cd scripts && python -m pytest backtest/tests/ -v

# Run backtest tests with coverage
test_backtest_cov:
	@echo "Running backtest tests with coverage..."
	cd scripts && python -m pytest backtest/tests/ -v --cov=backtest --cov-report=term-missing

# =============================================================================
# CLUSTER QUALITY MEASUREMENT
# =============================================================================

# Run cluster quality metrics tests
test_cluster_quality:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          TESTING CLUSTER QUALITY METRICS                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	python -m pytest scripts/cluster_quality/tests/ -v

# Run cluster quality tests with coverage
test_cluster_quality_cov:
	@echo "Running cluster quality tests with coverage..."
	python -m pytest scripts/cluster_quality/tests/ -v --cov=cluster_quality --cov-report=term-missing

# Analyze cluster quality on collected data
SYMBOL ?= BTC
DATA ?= ./data/features
analyze_clusters:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              ANALYZING CLUSTER QUALITY                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Symbol: $(SYMBOL)"
	@echo "Data: $(DATA)"
	@echo ""
	python scripts/analyze_clusters.py --data-dir $(DATA) --symbol $(SYMBOL) --hours $(HOURS)

# Analyze with trained GMM model
analyze_clusters_gmm:
	@echo "Analyzing with trained GMM model..."
	python scripts/analyze_clusters.py --data-dir $(DATA) --symbol $(SYMBOL) --model models/regime_gmm.json

# Analyze all symbols
analyze_all_symbols:
	@for sym in BTC ETH SOL; do \
		echo "Analyzing $$sym..."; \
		python scripts/analyze_clusters.py --data-dir $(DATA) --symbol $$sym --output reports/cluster_$$sym.txt; \
	done

# =============================================================================
# BASELINE MODEL TRAINING
# =============================================================================

# Train baseline models
SNAPSHOT ?= baseline_30d
MODEL_TYPE ?= elasticnet
MODEL_DIR ?= ./models
train_baseline:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              TRAINING BASELINE MODEL                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Snapshot: $(SNAPSHOT)"
	@echo "Model: $(MODEL_TYPE)"
	@echo "Output: $(MODEL_DIR)"
	@echo ""
	@mkdir -p $(MODEL_DIR)
	python scripts/train_baseline.py --snapshot $(SNAPSHOT) --model $(MODEL_TYPE) --output-dir $(MODEL_DIR)

# List saved models
list_models:
	@echo "Listing saved models..."
	python scripts/list_models.py --model-dir $(MODEL_DIR)

# Score data with trained model
MODEL_PATH ?= models/latest.pkl
score_data:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║              SCORING DATA WITH TRAINED MODEL                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Model: $(MODEL_PATH)"
	@echo "Data: $(DATA)"
	@echo ""
	python scripts/score_data.py --model $(MODEL_PATH) --data $(DATA) --evaluate

# Score and save predictions
PREDICTIONS ?= ./predictions.parquet
score_and_save:
	@echo "Scoring and saving predictions..."
	python scripts/score_data.py --model $(MODEL_PATH) --data $(DATA) --output $(PREDICTIONS) --evaluate

# =============================================================================
# DEVELOPMENT TOOLS
# =============================================================================

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd rust && cargo clean

# Format code
fmt:
	@echo "Formatting code..."
	cd rust && cargo fmt

# Run clippy linter
lint:
	@echo "Running clippy..."
	cd rust && cargo clippy -- -D warnings

# Check without building
check:
	@echo "Checking code..."
	cd rust && cargo check

# =============================================================================
# HELP
# =============================================================================

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
	@echo "  run               Run ingestor (debug build)"
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
	@echo "  test_backtest           Run backtest unit tests"
	@echo "  test_cluster_quality    Run cluster quality metrics tests"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " REGIME MODEL"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  train_gmm              Train GMM regime classifier (DATA=./data/features)"
	@echo "  train_gmm_auto         Train with auto-selected components via BIC"
	@echo "  analyze_clusters       Analyze cluster quality (SYMBOL=BTC HOURS=24)"
	@echo "  analyze_clusters_gmm   Analyze with trained GMM model"
	@echo "  analyze_all_symbols    Analyze BTC, ETH, SOL clusters"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " BASELINE MODELS (ML)"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  train_baseline         Train baseline ML model (SNAPSHOT=baseline_30d MODEL_TYPE=elasticnet)"
	@echo "  list_models            List all saved models with metrics"
	@echo "  score_data             Score data with trained model (MODEL_PATH=models/*.pkl)"
	@echo "  score_and_save         Score and save predictions to file"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " BACKTESTING"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  backtest          Run backtest (STRATEGY=whale_flow_simple SYMBOL=BTC)"
	@echo "  backtest_validate Run walk-forward validation (recommended)"
	@echo "  backtest_list     List available strategies"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " API VALIDATION (Live Hyperliquid)"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  validate          Run all API validations"
	@echo "  validate_all      Alias for validate"
	@echo "  validate_api      Test API connection"
	@echo "  validate_positions Test position tracking"
	@echo "  validate_whales   Test whale identification"
	@echo "  validate_entropy  Test entropy features"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " API & ALERTS"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  api               Start API server (REST + WebSocket)"
	@echo "  alerts            Start Telegram alert service"
	@echo "  serve_all         Start full stack (ingestor + API + alerts)"
	@echo "  release_api       Build API server (release)"
	@echo "  test_api          Test API endpoints (requires running server)"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " BUILD"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  build             Build debug version"
	@echo "  release           Build optimized release version"
	@echo "  clean             Remove build artifacts"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " DOCKER"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  docker_build      Build all Docker images"
	@echo "  docker_up         Start all services with Docker"
	@echo "  docker_down       Stop all Docker services"
	@echo "  docker_logs       View Docker logs"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " DEVELOPMENT"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  fmt               Format code with rustfmt"
	@echo "  lint              Run clippy linter"
	@echo "  check             Check code without building"
	@echo "  help              Show this help"
	@echo ""

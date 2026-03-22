# NAT Project Makefile
# Hyperliquid Market Data Ingestor

.PHONY: all run run_and_serve tunnel test test_verbose test_hypotheses build release clean validate validate_all validate_api validate_positions validate_whales validate_entropy validate_data validate_data_recent show show_fast show_hft explore help fmt lint check

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
	@echo "  test              Run all unit tests"
	@echo "  test_verbose      Run tests with output"
	@echo "  test_hypotheses   Run H1-H5 hypothesis tests (DATA=./data/features)"
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
	@echo " BUILD"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  build             Build debug version"
	@echo "  release           Build optimized release version"
	@echo "  clean             Remove build artifacts"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────────"
	@echo " DEVELOPMENT"
	@echo "───────────────────────────────────────────────────────────────────"
	@echo "  fmt               Format code with rustfmt"
	@echo "  lint              Run clippy linter"
	@echo "  check             Check code without building"
	@echo "  help              Show this help"
	@echo ""

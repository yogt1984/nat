# NAT Project Makefile
# Hyperliquid Market Data Ingestor

.PHONY: all run test build release clean validate validate-all validate-api validate-positions validate-whales validate-entropy show help fmt lint check

# Default target: run the main ingestor
all: run

# Build debug version
build:
	@echo "Building debug version..."
	cd rust && cargo build --bin ing

# Build release version (all binaries)
release:
	@echo "Building release version..."
	cd rust && cargo build --release --bin ing --bin validate_api --bin validate_positions --bin validate_whales --bin validate_entropy --bin show_features

# Run the main ingestor (requires config/ing.toml)
run: build
	@echo "Running ingestor..."
	cd rust && cargo run --bin ing -- ../config/ing.toml

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
test-verbose:
	@echo "Running all tests (verbose)..."
	cd rust && cargo test --package ing -- --nocapture

# =============================================================================
# VALIDATION (Skeptical Live API Tests)
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
validate-all: validate

# Individual validation targets
validate-api: release
	@echo "Running API connection validation..."
	cd rust && ./target/release/validate_api

validate-positions: release
	@echo "Running position tracking validation..."
	cd rust && ./target/release/validate_positions

validate-whales: release
	@echo "Running whale identification validation..."
	cd rust && ./target/release/validate_whales

validate-entropy: release
	@echo "Running tick entropy feature validation..."
	cd rust && ./target/release/validate_entropy

# =============================================================================
# REAL-TIME MONITORING
# =============================================================================

# Show real-time features (no file output, terminal only)
# Usage: make show [SYMBOL=BTC]
SYMBOL ?= BTC
show: release
	@echo "Starting real-time feature display for $(SYMBOL)..."
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd rust && ./target/release/show_features $(SYMBOL)

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

# Show help
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║          NAT Project - Hyperliquid Market Data Ingestor          ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "MAIN TARGETS:"
	@echo "  all             - Build and run the ingestor (default)"
	@echo "  run             - Run the main ingestor"
	@echo "  show            - Show real-time features (make show SYMBOL=ETH)"
	@echo ""
	@echo "TESTING:"
	@echo "  test            - Run all unit tests"
	@echo "  test-verbose    - Run tests with output"
	@echo "  validate        - Run ALL validations (live API tests)"
	@echo "  validate-api    - Run API connection validation only"
	@echo "  validate-positions - Run position tracking validation"
	@echo "  validate-whales - Run whale identification validation"
	@echo "  validate-entropy - Run tick entropy validation"
	@echo ""
	@echo "BUILD:"
	@echo "  build           - Build debug version"
	@echo "  release         - Build release version (all binaries)"
	@echo ""
	@echo "DEVELOPMENT:"
	@echo "  clean           - Clean build artifacts"
	@echo "  fmt             - Format code"
	@echo "  lint            - Run clippy linter"
	@echo "  check           - Check code without building"
	@echo "  help            - Show this help"

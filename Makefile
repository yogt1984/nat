# NAT Project Makefile
# Hyperliquid Market Data Ingestor

.PHONY: all run test build release clean validate help

# Default target: run the main ingestor
all: run

# Build debug version
build:
	@echo "Building debug version..."
	cd rust && cargo build --bin ing

# Build release version
release:
	@echo "Building release version..."
	cd rust && cargo build --release --bin ing --bin validate_api

# Run the main ingestor (requires config/ing.toml)
run: build
	@echo "Running ingestor..."
	cd rust && cargo run --bin ing -- ../config/ing.toml

# Run all tests
test:
	@echo "Running all tests..."
	cd rust && cargo test

# Run tests with verbose output
test-verbose:
	@echo "Running all tests (verbose)..."
	cd rust && cargo test -- --nocapture

# Run API validation (skeptical test)
validate: release
	@echo "Running Hyperliquid API validation..."
	cd rust && ./target/release/validate_api

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

# Show help
help:
	@echo "NAT Project - Hyperliquid Market Data Ingestor"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build and run the ingestor (default)"
	@echo "  run       - Run the main ingestor"
	@echo "  test      - Run all tests"
	@echo "  test-verbose - Run tests with output"
	@echo "  validate  - Run API validation (skeptical test)"
	@echo "  build     - Build debug version"
	@echo "  release   - Build release version"
	@echo "  clean     - Clean build artifacts"
	@echo "  fmt       - Format code"
	@echo "  lint      - Run clippy linter"
	@echo "  check     - Check code without building"
	@echo "  help      - Show this help"

# Build, release, dev tools
.PHONY: build release release_api clean fmt lint check setup-python

build:
	@echo "Building debug version..."
	cd rust && cargo build --bin ing

release:
	@echo "Building release version..."
	cd rust && cargo build --release --bin ing --bin validate_api --bin validate_positions --bin validate_whales --bin validate_entropy --bin show_features --bin test_hypotheses

release_api:
	@echo "Building API server..."
	cd rust && cargo build --release --bin nat-api

clean:
	@echo "Cleaning build artifacts..."
	cd rust && cargo clean

fmt:
	@echo "Formatting code..."
	cd rust && cargo fmt

lint:
	@echo "Running clippy..."
	cd rust && cargo clippy -- -D warnings

check:
	@echo "Checking code..."
	cd rust && cargo check

setup-python:
	pip install -e scripts/

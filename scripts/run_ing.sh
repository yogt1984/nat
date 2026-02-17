#!/bin/bash
# Run the ingestor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-$PROJECT_DIR/config/ing.toml}"

echo "Starting ING with config: $CONFIG_FILE"

cd "$PROJECT_DIR"

# Build in release mode
cargo build --release -p ing

# Run
RUST_LOG=info ./target/release/ing "$CONFIG_FILE"

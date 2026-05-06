#!/bin/bash
# Install the `nat` command so it's available from anywhere.
#
# Usage:
#     ./install.sh
#
# After install, just type `nat` from any directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NAT_BIN="$SCRIPT_DIR/nat"
LINK_DIR="$HOME/.local/bin"

# Create bin directory if needed
mkdir -p "$LINK_DIR"

# Create symlink
ln -sf "$NAT_BIN" "$LINK_DIR/nat"

# Check if LINK_DIR is in PATH
if echo "$PATH" | tr ':' '\n' | grep -q "$LINK_DIR"; then
    echo "  ✓  nat installed to $LINK_DIR/nat"
    echo "  ✓  Already in PATH"
else
    # Add to PATH via bashrc
    echo "" >> "$HOME/.bashrc"
    echo "# NAT research tool" >> "$HOME/.bashrc"
    echo "export PATH=\"$LINK_DIR:\$PATH\"" >> "$HOME/.bashrc"
    echo "  ✓  nat installed to $LINK_DIR/nat"
    echo "  ✓  Added $LINK_DIR to PATH in ~/.bashrc"
    echo "  →  Run: source ~/.bashrc (or open a new terminal)"
fi

echo ""
echo "  Try: nat help"
echo ""

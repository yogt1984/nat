#!/usr/bin/env bash
# build_deb.sh — stage and build the nat .deb package.
#
#   bash packaging/build_deb.sh [--no-build]
#
# Produces  dist/nat_<version>_amd64.deb
# Layout installed on the target:
#   /usr/lib/nat/{nat,scripts,rust/target/release/*}   (program)
#   /etc/nat/*.toml                                     (config)
#   /usr/bin/nat -> /usr/lib/nat/nat                    (postinst symlink)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# dpkg policy: version must start with a digit and use '-' only as the
# upstream/revision separator. Normalize git-describe into 0.1.0+<alnum> form.
RAW="$(git describe --tags --always --dirty 2>/dev/null | sed 's/^v//' || echo 0.1.0)"
VERSION="0.1.0+$(printf '%s' "$RAW" | tr -cd '[:alnum:].')"
ARCH="amd64"
PKG="nat_${VERSION}_${ARCH}"
STAGE="build/deb/${PKG}"
DIST="dist"

# Release Rust binaries to ship (only those that exist are copied).
BINS="ing api natviz3d validate_api validate_positions validate_whales validate_entropy show_features test_hypotheses"

if [ "${1:-}" != "--no-build" ]; then
    echo ">> building release binaries"
    ( cd rust && cargo build --release )
fi

echo ">> staging $STAGE"
rm -rf "build/deb"
mkdir -p "$STAGE/DEBIAN" \
         "$STAGE/usr/lib/nat/rust/target/release" \
         "$STAGE/usr/lib/nat/scripts" \
         "$STAGE/etc/nat"

# Program: the CLI script + the Python analysis stack.
install -m 0755 nat "$STAGE/usr/lib/nat/nat"
cp -r scripts/. "$STAGE/usr/lib/nat/scripts/"

# Rust release binaries.
for b in $BINS; do
    if [ -f "rust/target/release/$b" ]; then
        install -m 0755 "rust/target/release/$b" "$STAGE/usr/lib/nat/rust/target/release/$b"
    else
        echo "   (skip missing binary: $b)"
    fi
done

# Config → /etc/nat. nat_paths resolves config from /etc/nat directly in
# installed mode (no symlink needed); symbols.toml ships alongside it.
cp config/*.toml "$STAGE/etc/nat/" 2>/dev/null || true

# Control + maintainer scripts.
sed "s/@VERSION@/${VERSION}/" packaging/deb/control.in > "$STAGE/DEBIAN/control"
install -m 0755 packaging/deb/postinst "$STAGE/DEBIAN/postinst"
install -m 0755 packaging/deb/prerm    "$STAGE/DEBIAN/prerm"

echo ">> building package"
mkdir -p "$DIST"
dpkg-deb --build --root-owner-group "$STAGE" "$DIST/${PKG}.deb"

echo ">> done: $DIST/${PKG}.deb"
dpkg-deb --info "$DIST/${PKG}.deb" | sed 's/^/   /'

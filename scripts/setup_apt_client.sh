#!/usr/bin/env bash
# setup_apt_client.sh — point apt at the private, password-protected NAT repo.
#
#   sudo REPO_URL=https://repo.example.com/nat \
#        REPO_USER=alice REPO_PASS=secret \
#        PUBKEY_URL=https://repo.example.com/nat/nat-archive-keyring.gpg \
#        bash scripts/setup_apt_client.sh
#
# After this, `sudo apt update && sudo apt install nat` works only on this machine
# (credentials live in /etc/apt/auth.conf.d/, mode 600, never in the sources line).
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL, e.g. https://repo.example.com/nat}"
REPO_USER="${REPO_USER:?set REPO_USER}"
REPO_PASS="${REPO_PASS:?set REPO_PASS}"
PUBKEY_URL="${PUBKEY_URL:?set PUBKEY_URL (the repo signing public key)}"
SUITE="${SUITE:-stable}"
COMPONENT="${COMPONENT:-main}"

HOST="$(printf '%s' "$REPO_URL" | sed -E 's#^https?://([^/]+).*#\1#')"
KEYRING="/etc/apt/keyrings/nat-archive-keyring.gpg"

echo ">> installing signing key -> $KEYRING"
install -d -m 0755 /etc/apt/keyrings
curl -fsSL "$PUBKEY_URL" | gpg --dearmor -o "$KEYRING"
chmod 0644 "$KEYRING"

echo ">> writing /etc/apt/sources.list.d/nat.list"
cat > /etc/apt/sources.list.d/nat.list <<EOF
deb [signed-by=$KEYRING] $REPO_URL $SUITE $COMPONENT
EOF

echo ">> writing /etc/apt/auth.conf.d/nat.conf (mode 600)"
install -d -m 0700 /etc/apt/auth.conf.d
cat > /etc/apt/auth.conf.d/nat.conf <<EOF
machine $HOST
login $REPO_USER
password $REPO_PASS
EOF
chmod 0600 /etc/apt/auth.conf.d/nat.conf

echo ">> apt update"
apt-get update

echo ">> ready:  sudo apt install nat"

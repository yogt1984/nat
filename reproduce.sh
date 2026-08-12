#!/usr/bin/env bash
# DOCS-4 — one command, three headline figures, from data frozen in this repo.
#
#   ./reproduce.sh
#
# Needs: python3 with pandas, pyarrow, matplotlib (pip install pandas pyarrow matplotlib).
# Uses the project venv when present. Reads only reproduce/slice/ and config/costs.toml;
# writes reproduce/out/. No network, no live venue state, no other data.
set -euo pipefail
cd "$(dirname "$0")"

PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

exec "$PY" scripts/repro/make_figures.py "$@"

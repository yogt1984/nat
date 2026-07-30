#!/usr/bin/env bash
# COST-3 guardrail: fail if a hardcoded fee literal is assigned anywhere outside the
# cost single-source-of-truth. All fees must come from utils.costs (config/costs.toml)
# so a fee change ripples everywhere and no backtest silently uses a wrong/zero fee.
#
# Allowlisted: the cost-model modules that legitimately define numbers, and tests.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

# A fee-bearing variable assigned a *pure numeric literal* (e.g. `taker_fee_bps = 3.5`,
# `fee_bps=0`). The RHS-terminator check ([#,)]|EOL) means a config-derived expression
# like `= taker_bps()` or `= 2 * maker_bps()` or `= -cfg.maker_fee_bps` is NOT flagged.
PATTERN='(fee_bps|taker_fee_bps|maker_fee_bps|taker_cost_bps|maker_cost_bps|cost_bps|cost_per_trade|taker_fee|maker_fee)[[:space:]]*=[[:space:]]*-?[0-9][0-9._eE+-]*[[:space:]]*([#,)]|$)'

# Allowlist: the cost-model modules (which legitimately define numbers) and all tests.
hits=$(grep -rnE "$PATTERN" scripts/ --include='*.py' \
  | grep -vE 'scripts/utils/costs\.py|scripts/backtest/costs\.py|/tests?/|/test_[^/]+\.py:') || true

if [ -n "$hits" ]; then
  echo "COST guardrail FAILED — hardcoded fee literal(s); route through utils.costs (load_costs):"
  echo "$hits"
  exit 1
fi

# COST-5 guardrail: the VIP9 (Binance) tier must never be a DEFAULT — only an explicit
# opt-in choice. Defaulting to it produced the 5/5 false "deployable winners" refuted by
# the Q4 kill gate (docs/research/FINDINGS.md §4.6). Catches: argparse defaults, getattr
# fallbacks, and module-level DEFAULT_*/FEE_* constants built from the vip9 helper.
VIP9_PATTERN='default[[:space:]]*=[[:space:]]*["'\'']binance_vip9["'\'']|getattr\([^)]*["'\'']binance_vip9["'\'']\)|^(FEE_BPS|DEFAULT_COST|DEFAULT_FEE[A-Za-z_]*)[[:space:]]*=.*vip9'
vip9_hits=$(grep -rnE "$VIP9_PATTERN" scripts/ --include='*.py' \
  | grep -vE 'scripts/utils/costs\.py|scripts/backtest/costs\.py|/tests?/|/test_[^/]+\.py:') || true

if [ -n "$vip9_hits" ]; then
  echo "COST guardrail FAILED — VIP9 tier used as a DEFAULT (wrong venue; Hyperliquid SSOT ~11bps is the default, vip9 is opt-in only):"
  echo "$vip9_hits"
  exit 1
fi
echo "OK: no hardcoded fee literals outside the cost SSOT; no VIP9 defaults."

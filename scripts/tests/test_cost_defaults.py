"""COST-4/COST-5 guard — no evaluation harness may DEFAULT to the Binance VIP9 tier.

The wrong-venue cost default (binance_vip9 = 1.61 bps RT on a platform that trades
Hyperliquid at ~11 bps all-in) produced the entire false "deployable winners" tier —
caught 2026-05-27, recurred through harness defaults in the 2026-06-12 sweep, confirmed
5/5 by the Q4 kill gate (FINDINGS §4.6). This test makes the recurrence structurally
impossible:

  (a) source scan: no argparse default, getattr fallback, module-level DEFAULT_*/FEE_*
      constant, or default_* property may reference the VIP9 tier. VIP9 stays available
      as an EXPLICIT opt-in choice only.
  (b) positive checks: the harness defaults resolve to the Hyperliquid SSOT
      (config/costs.toml via load_costs(): 7.0 bps RT taker + 2.0 bps/side slippage).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]

# Files allowed to mention vip9 freely: the SSOT accessors and the preset registry
# (definitions), never harness defaults.
EXEMPT = {"utils/costs.py", "backtest/costs.py", "tests"}

FORBIDDEN = [
    ("argparse default", re.compile(r"default\s*=\s*['\"]binance_vip9['\"]")),
    ("getattr fallback", re.compile(r"getattr\([^)]*['\"]binance_vip9['\"]\s*\)")),
    ("module-level default constant",
     re.compile(r"^(FEE_BPS|DEFAULT_COST|DEFAULT_FEE\w*)\s*=.*vip9", re.M)),
    ("default_* property returns vip9",
     re.compile(r"def default_\w+.*?return\s+self\.binance_vip9\w*", re.S)),
]


def _scan_files():
    for path in sorted(SCRIPTS.rglob("*.py")):
        rel = path.relative_to(SCRIPTS).as_posix()
        if any(rel == e or rel.startswith(e + "/") or f"/{e}/" in f"/{rel}" for e in EXEMPT):
            continue
        yield rel, path.read_text(errors="replace")


class TestNoVip9Defaults:
    def test_no_harness_defaults_to_vip9(self):
        violations = []
        for rel, text in _scan_files():
            for label, pat in FORBIDDEN:
                if pat.search(text):
                    violations.append(f"{rel}: {label}")
        assert not violations, (
            "VIP9 (1.61 bps, Binance) used as a DEFAULT — the venue is Hyperliquid "
            "(~11 bps all-in). Make it an explicit opt-in choice instead:\n  "
            + "\n  ".join(violations)
        )


class TestDefaultsResolveToSsot:
    def test_realistic_rt_helper_matches_config(self):
        from utils.costs import load_costs, realistic_taker_rt_bps
        hl = load_costs()["hyperliquid"]
        expected = hl["round_trip_taker_bps"] + 2.0 * hl["slippage_bps"]
        assert realistic_taker_rt_bps() == pytest.approx(expected)
        assert realistic_taker_rt_bps() >= 7.0          # never the 1.61 fantasy

    def test_paper_trader_daily_default_cost_is_ssot(self):
        import sys
        sys.path.insert(0, str(SCRIPTS))
        from alpha.paper_trader_daily import DEFAULT_COST
        from utils.costs import realistic_taker_rt_bps
        assert DEFAULT_COST.round_trip_cost_bps == pytest.approx(realistic_taker_rt_bps())

    def test_paper_trader_surprise_fee_is_ssot(self):
        import sys
        sys.path.insert(0, str(SCRIPTS))
        from alpha.paper_trader_surprise import FEE_BPS
        from utils.costs import realistic_taker_rt_bps
        assert FEE_BPS == pytest.approx(realistic_taker_rt_bps())

    def test_it_engine_default_fee_is_hyperliquid(self):
        from it_engine.config import CostConfig
        cfg = CostConfig()
        assert cfg.default_fee_rt_bps == pytest.approx(cfg.hyperliquid_rt_bps)


class TestVip9StillExplicitlyAvailable:
    """Opt-in must survive — comparisons against the VIP9 tier are legitimate research."""

    def test_vip9_accessor_still_exists(self):
        from utils.costs import binance_vip9_rt_bps
        assert binance_vip9_rt_bps() > 0

    def test_daily_still_offers_vip9_as_choice(self):
        text = (SCRIPTS / "alpha" / "paper_trader_daily.py").read_text()
        assert '"binance_vip9"' in text                 # in choices/mapping, not default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

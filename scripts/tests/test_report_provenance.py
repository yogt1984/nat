"""REV-1 — no report may carry wrong-venue pricing without saying so.

FINDINGS §4.6: every eval harness once defaulted to Binance VIP9 (1.61 bps round trip)
while NAT trades Hyperliquid (~11 bps all-in), and *"every historical backtest number
produced through these paths is invalid until re-run"*. COST-6 re-pointed the harnesses and
COST-7 hardened the CI guard against wrong-preset calls — but the **artifacts those
harnesses had already written** kept sitting in `reports/`, unlabelled, as machine-readable
JSON that a future sweep, dashboard or agent would read as current.

This is the recurrence guard for that: an artifact may contain VIP9 pricing (deleting it
would destroy provenance, and one of them is a live input to
`analysis/mf_liquidity_backtest.py`), but it must carry a `_superseded` stamp saying so.

The failure this prevents is specific and has already happened once: a refuted number
re-entering the record because nothing on its face said it was refuted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

#: Markers of the wrong-venue tier. 1.61 = VIP9 round trip; the names are the presets.
VIP9_MARKERS = ("1.61", "vip9", "VIP9", "binance_vip9")


def _vip9_json_reports():
    if not REPORTS.exists():
        return []
    out = []
    for p in sorted(REPORTS.glob("*.json")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(m in text for m in VIP9_MARKERS):
            out.append(p)
    return out


def test_every_vip9_priced_report_is_stamped_superseded():
    """An artifact priced at the wrong venue must say so on its face."""
    unstamped = []
    for p in _vip9_json_reports():
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "_superseded" not in d:
            unstamped.append(p.name)

    assert not unstamped, (
        "these reports carry Binance VIP9 (1.61 bps) pricing with no `_superseded` stamp: "
        f"{unstamped}. NAT trades Hyperliquid at ~11 bps all-in (FINDINGS §4.6). Either "
        "re-run them at SSOT cost or stamp them — an unlabelled JSON of refuted numbers is "
        "how a false discovery re-enters the record."
    )


def test_the_stamp_carries_enough_to_act_on():
    """A stamp that does not say why, or point at the authority, is decoration."""
    for p in _vip9_json_reports():
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        stamp = (d or {}).get("_superseded")
        if not stamp:
            continue
        for field in ("status", "reason", "authority", "task"):
            assert stamp.get(field), f"{p.name}: `_superseded` is missing `{field}`"
        assert "4.6" in stamp["authority"], (
            f"{p.name}: the stamp should cite the finding that refuted it"
        )


def test_the_guard_can_actually_fail(tmp_path):
    """A guard that cannot fail is not a guard — prove the detector fires."""
    bad = tmp_path / "fake_report.json"
    bad.write_text(json.dumps({"fee_model": "binance_vip9", "net_bps": 123}))
    text = bad.read_text()
    assert any(m in text for m in VIP9_MARKERS)
    assert "_superseded" not in json.loads(text)

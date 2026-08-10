"""WP-1 — the wallet roster: derived, never hardcoded.

Step 1 of `docs/specs/wallet_positioning.md`. The roster is the list of wallets whose
positions the collector will poll, and it is the thing standing between NAT and the
deterministic-liquidation family: `state/mod.rs:127` initialises `position_state` to `None`,
so with no roster there are no positions, and the 13 liquidation + 15 concentration features
are NaN by construction (the K2 dead columns).

Four properties carry it, and the first two are the spec's non-negotiables:

  1. **Derived, never hardcoded.** A frozen list rots the moment the cohort turns over — and
     cohort turnover is itself one of the hypotheses under test (spec §3, failure mode 1). A
     source scan asserts no literal address survives in the module.

  2. **Addresses are validated before they can reach a path or an API call.** The XS-1 lesson:
     names become filenames. An address is a 0x-prefixed 40-hex string or it is rejected.

  3. **A malformed payload raises rather than returning a short roster.** A silently truncated
     roster narrows every downstream cohort claim, and would do so invisibly — the same shape
     as the XS-1 sweep that reported `ok=177` while collecting 4 % of the requested span.

  4. **Retry transport faults, never schema faults.** A 429 should back off; a payload whose
     shape changed should fail loudly rather than be retried into the same wall.

Hermetic: the venue call is injected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data.wallet_roster import (  # noqa: E402
    ADDRESS_RE,
    WalletRef,
    fetch_roster,
    is_valid_address,
    parse_leaderboard,
)

A1 = "0x" + "a1" * 20
A2 = "0x" + "b2" * 20
A3 = "0x" + "c3" * 20


def _row(addr, value=1_000_000.0, month_pnl=5000.0, month_vlm=2_000_000.0):
    return {
        "ethAddress": addr,
        "accountValue": str(value),
        "displayName": None,
        "windowPerformances": [
            ["day", {"pnl": "10", "roi": "0.001", "vlm": "1000"}],
            ["week", {"pnl": "100", "roi": "0.01", "vlm": "10000"}],
            ["month", {"pnl": str(month_pnl), "roi": "0.05", "vlm": str(month_vlm)}],
            ["allTime", {"pnl": "9999", "roi": "0.9", "vlm": "99999"}],
        ],
    }


def _payload(*rows):
    return {"leaderboardRows": list(rows)}


# ── 1. derived, never hardcoded ──────────────────────────────────────────

def test_module_contains_no_literal_wallet_address():
    """The spec's non-negotiable. A pinned list rots the moment the cohort turns over."""
    import re
    src = (SCRIPTS / "data" / "wallet_roster.py").read_text()
    # allow the regex itself and test-shaped placeholders; forbid real 40-hex literals
    literals = re.findall(r"0x[0-9a-fA-F]{40}", src)
    assert not literals, f"hardcoded address(es) in the module: {literals}"


def test_roster_is_ordered_by_a_stated_key_not_payload_order():
    rows = [_row(A1, value=10.0), _row(A2, value=999.0), _row(A3, value=500.0)]
    out = fetch_roster(info_fn=lambda: _payload(*rows), limit=3)
    assert [w.address for w in out] == [A2, A3, A1]


def test_limit_selects_the_top_not_an_arbitrary_slice():
    rows = [_row(A1, value=10.0), _row(A2, value=999.0), _row(A3, value=500.0)]
    out = fetch_roster(info_fn=lambda: _payload(*rows), limit=2)
    assert [w.address for w in out] == [A2, A3]


def test_ordering_is_deterministic_under_ties():
    rows = [_row(A3, value=100.0), _row(A1, value=100.0), _row(A2, value=100.0)]
    a = [w.address for w in fetch_roster(info_fn=lambda: _payload(*rows))]
    b = [w.address for w in fetch_roster(info_fn=lambda: _payload(*rows))]
    assert a == b == sorted([A1, A2, A3])


def test_duplicates_across_the_payload_collapse():
    out = fetch_roster(info_fn=lambda: _payload(_row(A1), _row(A1), _row(A2)))
    assert sorted(w.address for w in out) == sorted([A1, A2])


# ── 2. addresses validated before they can reach a path or an API ────────

@pytest.mark.parametrize("bad", [
    "not-an-address", "0x123", "0x" + "z" * 40, "../../etc/passwd",
    "0x" + "a" * 41, "", None, 12345, "0X" + "a" * 40 + "/../x",
])
def test_invalid_addresses_are_rejected(bad):
    assert not is_valid_address(bad)


def test_valid_address_shape_accepted_either_case():
    assert is_valid_address("0x" + "a" * 40)
    assert is_valid_address("0x" + "A" * 40)
    assert ADDRESS_RE.match("0x" + "0123456789abcdef" * 2 + "01234567")


def test_a_bad_address_in_the_payload_is_dropped_not_propagated():
    """One malformed row must not poison the roster, and must not reach a filesystem path."""
    rows = [_row(A1), _row("../../etc/passwd"), _row("0xdeadbeef")]
    out = fetch_roster(info_fn=lambda: _payload(*rows))
    assert [w.address for w in out] == [A1]


# ── 3. malformed payloads raise; they never return a short roster ────────

@pytest.mark.parametrize("payload", [
    {}, {"wrongKey": []}, {"leaderboardRows": []}, [], None, "nope",
])
def test_malformed_payload_raises(payload):
    with pytest.raises((ValueError, TypeError)):
        fetch_roster(info_fn=lambda: payload)


def test_a_row_that_is_not_an_object_raises():
    with pytest.raises((ValueError, TypeError)):
        parse_leaderboard({"leaderboardRows": [_row(A1), "not-a-row"]})


def test_all_rows_invalid_raises_rather_than_returning_empty():
    """An empty roster is never a valid result — it is a changed payload shape."""
    with pytest.raises(ValueError):
        fetch_roster(info_fn=lambda: _payload(_row("bad"), _row("alsobad")))


# ── 4. retry transport faults, never schema faults ───────────────────────

def test_transport_fault_is_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset")
        return _payload(_row(A1))

    out = fetch_roster(info_fn=flaky, retries=3, backoff=0.0)
    assert [w.address for w in out] == [A1] and calls["n"] == 3


def test_schema_fault_is_not_retried():
    """A 429 backs off; a payload whose shape changed must fail loudly and once."""
    calls = {"n": 0}

    def bad_schema():
        calls["n"] += 1
        return {"wrongKey": []}

    with pytest.raises(ValueError):
        fetch_roster(info_fn=bad_schema, retries=5, backoff=0.0)
    assert calls["n"] == 1, "a schema error was retried into the same wall"


def test_retries_are_bounded():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise OSError("down")

    with pytest.raises(OSError):
        fetch_roster(info_fn=always_fails, retries=2, backoff=0.0)
    assert calls["n"] == 3          # initial + 2


# ── provenance ───────────────────────────────────────────────────────────

def test_every_ref_carries_its_provenance():
    out = fetch_roster(info_fn=lambda: _payload(_row(A1, value=42.0, month_vlm=7.0)))
    w = out[0]
    assert isinstance(w, WalletRef)
    assert w.source == "leaderboard"
    assert w.account_value == 42.0
    assert w.notional_seen == 7.0
    assert w.first_seen


def test_a_profitability_filter_selects_on_the_stated_window():
    rows = [_row(A1, month_pnl=500.0), _row(A2, month_pnl=-500.0)]
    out = fetch_roster(info_fn=lambda: _payload(*rows), min_window_pnl=0.0, window="month")
    assert [w.address for w in out] == [A1]


# ── the defect the live smoke found ──────────────────────────────────────

def test_a_volume_floor_excludes_custodial_accounts():
    """The live leaderboard's largest accounts are vaults and the bridge, not traders.

    The top entry holds $14.1bn with ZERO traded volume, and `min_window_pnl >= 0` admits it
    because its P&L is exactly zero. A wallet that has not traded is not a trader whatever its
    balance. Synthetic fixtures could not have surfaced this — the real-data smoke did.
    """
    vault = _row(A1, value=14_000_000_000.0, month_pnl=0.0, month_vlm=0.0)
    trader = _row(A2, value=1_000_000.0, month_pnl=50_000.0, month_vlm=9_000_000.0)
    out = fetch_roster(info_fn=lambda: _payload(vault, trader), min_notional_seen=1.0)
    assert [w.address for w in out] == [A2]


def test_rank_by_selects_a_different_top():
    big_balance = _row(A1, value=1e9, month_vlm=1.0)
    big_volume = _row(A2, value=1.0, month_vlm=1e9)
    by_value = fetch_roster(info_fn=lambda: _payload(big_balance, big_volume))
    by_vlm = fetch_roster(info_fn=lambda: _payload(big_balance, big_volume),
                          rank_by="notional_seen")
    assert by_value[0].address == A1
    assert by_vlm[0].address == A2


def test_unknown_rank_by_raises():
    with pytest.raises(ValueError):
        fetch_roster(info_fn=lambda: _payload(_row(A1)), rank_by="nonsense")

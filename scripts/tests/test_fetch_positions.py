"""WP-2 — position snapshot collector. The failure rows are the unit.

Step 2 of `docs/specs/wallet_positioning.md`. This collector starts a clock that nothing else
can start: WP-5 needs ≥90 days of accrual, so every uncollected day is permanently lost and the
verdict date slips one-for-one. That makes *correctness under partial failure* the whole design
problem — a sweep that quietly drops what it could not reach produces a file that looks complete.

Contract encoded here:
  (a) **a wallet that fails is written, not dropped.** XS-8 appends failures to a report and
      `continue`s, so `aggregate_l2` cannot distinguish "frequently unreachable" from "less
      history". Here a wallet that stops responding must never look like a wallet that closed
      its positions — that distinction is the entire point of the unit;
  (b) **the arithmetic closes** — every requested wallet lands in exactly one of
      ok / empty / failed / rejected, asserted as a sum. A sweep that silently loses a wallet
      is the XS-1 failure (`ok=177` while collecting 4 % of what it asked for);
  (c) **signed size survives** — a short must not become a long. Sign is the direction of the
      whole family-5 hypothesis;
  (d) **`liquidation_price=None` stays null.** The first live position sampled had
      `liquidationPx: null`; coerced to 0.0 it reads as "liquidation at zero", and every
      downstream cascade-distance calculation silently becomes distance-to-zero;
  (e) truncation is reported, delay is applied, addresses are validated before they can reach a
      path or a URL (the XS-1 lesson: names become filenames).

Every test is offline via an injected fetcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data import fetch_positions as fp  # noqa: E402

LONG = "0x" + "a" * 40
SHORT = "0x" + "b" * 40
FLAT = "0x" + "c" * 40
DEAD = "0x" + "d" * 40


def _state(positions, account_value="1000.0", time_ms=1_700_000_000_000) -> dict:
    """A clearinghouseState payload in the venue's real shape (verified live 2026-08-10)."""
    return {
        "marginSummary": {"accountValue": account_value, "totalNtlPos": "500.0",
                          "totalRawUsd": "900.0", "totalMarginUsed": "100.0"},
        "crossMaintenanceMarginUsed": "50.0",
        "withdrawable": "800.0",
        "assetPositions": [{"type": "oneWay", "position": p} for p in positions],
        "time": time_ms,
    }


def _pos(coin="BTC", szi="1.5", liq="60000.0", entry="63851.5") -> dict:
    return {
        "coin": coin, "szi": szi,
        "leverage": {"type": "cross", "value": 3},
        "entryPx": entry, "positionValue": "320536.68",
        "unrealizedPnl": "-88.05", "returnOnEquity": "-0.0008",
        "liquidationPx": liq, "marginUsed": "106845.56", "maxLeverage": 40,
        "cumFunding": {"allTime": "-23135.72", "sinceOpen": "5.27", "sinceChange": "0.0"},
    }


def _fetcher(mapping):
    """Injected fetch_fn. A value that is an Exception is raised, mimicking an unreachable RPC."""
    def fetch(wallet):
        v = mapping[wallet]
        if isinstance(v, Exception):
            raise v
        return v
    return fetch


# ── (a) failures are written, not dropped ────────────────────────────────────────

def test_failed_wallet_is_written_with_status_failed():
    rows, report = fp.sweep_positions(
        [LONG, DEAD],
        fetch_fn=_fetcher({LONG: _state([_pos()]), DEAD: OSError("connection reset")}),
        delay=0,
    )
    by_wallet = {r["wallet"]: r for r in rows}
    assert DEAD in by_wallet, "an unreachable wallet must appear in the data, not only the report"
    assert by_wallet[DEAD]["status"] == fp.PositionStatus.FAILED
    assert "connection reset" in (by_wallet[DEAD]["error"] or "")
    assert report["failed"] == 1


def test_empty_account_is_distinguishable_from_a_failed_one():
    rows, _ = fp.sweep_positions(
        [FLAT, DEAD],
        fetch_fn=_fetcher({FLAT: _state([]), DEAD: OSError("boom")}),
        delay=0,
    )
    status = {r["wallet"]: r["status"] for r in rows}
    assert status[FLAT] == fp.PositionStatus.EMPTY
    assert status[DEAD] == fp.PositionStatus.FAILED
    assert status[FLAT] != status[DEAD], "flat and unreachable must never collapse"


def test_one_failure_does_not_abort_the_sweep():
    wallets = [DEAD, LONG, SHORT]
    rows, report = fp.sweep_positions(
        wallets,
        fetch_fn=_fetcher({
            DEAD: OSError("down"),
            LONG: _state([_pos(coin="BTC", szi="1.5")]),
            SHORT: _state([_pos(coin="ETH", szi="-2.0")]),
        }),
        delay=0,
    )
    assert {r["wallet"] for r in rows} == set(wallets), "the sweep stopped at the first failure"
    assert report["ok"] == 2


# ── (b) the arithmetic closes ────────────────────────────────────────────────────

def test_every_requested_wallet_is_accounted_for_exactly_once():
    wallets = [LONG, FLAT, DEAD, "not-an-address"]
    _, report = fp.sweep_positions(
        wallets,
        fetch_fn=_fetcher({LONG: _state([_pos()]), FLAT: _state([]), DEAD: OSError("x")}),
        delay=0,
    )
    total = report["ok"] + report["empty"] + report["failed"] + report["rejected"]
    assert report["n_requested"] == len(wallets)
    assert total == len(wallets), f"wallets lost or double-counted: {report}"


def test_invalid_address_is_rejected_before_any_fetch():
    seen = []

    def fetch(wallet):
        seen.append(wallet)
        return _state([])

    _, report = fp.sweep_positions(
        ["0xnope", "", None, "../../etc/passwd", LONG], fetch_fn=fetch, delay=0)
    assert report["rejected"] == 4
    assert seen == [LONG], "a malformed address reached the fetcher"


# ── (c) signed size ──────────────────────────────────────────────────────────────

def test_short_position_keeps_its_negative_size():
    rows = fp.parse_clearinghouse_state(_state([_pos(coin="ETH", szi="-2.5")]), SHORT)
    assert len(rows) == 1
    assert rows[0]["size"] == -2.5, "a short became a long"
    assert rows[0]["coin"] == "ETH"


def test_multiple_coins_produce_one_row_each():
    rows = fp.parse_clearinghouse_state(
        _state([_pos(coin="BTC", szi="1.0"), _pos(coin="ETH", szi="-2.0"),
                _pos(coin="SOL", szi="3.0")]), LONG)
    assert len(rows) == 3
    assert {r["coin"] for r in rows} == {"BTC", "ETH", "SOL"}
    assert all(r["wallet"] == LONG for r in rows)


# ── (d) null liquidation price ───────────────────────────────────────────────────

def test_null_liquidation_price_is_preserved_not_zeroed():
    """The first live position sampled had liquidationPx: null. Zero is a different claim."""
    rows = fp.parse_clearinghouse_state(_state([_pos(liq=None)]), LONG)
    assert rows[0]["liquidation_price"] is None, \
        "null liquidation price coerced to a number — every cascade distance is now wrong"


def test_present_liquidation_price_is_a_float():
    rows = fp.parse_clearinghouse_state(_state([_pos(liq="60000.0")]), LONG)
    assert rows[0]["liquidation_price"] == 60000.0


# ── (e) truncation, delay, schema, idempotency ───────────────────────────────────

def test_truncation_is_reported_never_silent():
    wallets = [LONG, SHORT, FLAT]
    _, report = fp.sweep_positions(
        wallets,
        fetch_fn=_fetcher({w: _state([]) for w in wallets}),
        delay=0, max_wallets=2,
    )
    assert report["truncated"] == 1
    assert report["n_requested"] == 2
    assert report["n_supplied"] == 3


def test_delay_is_applied_between_wallets(monkeypatch):
    slept = []
    monkeypatch.setattr(fp.time, "sleep", lambda s: slept.append(s))
    wallets = [LONG, SHORT, FLAT]
    fp.sweep_positions(wallets, fetch_fn=_fetcher({w: _state([]) for w in wallets}), delay=0.2)
    assert slept == [0.2, 0.2], "delay must apply between wallets, not before the first"


def test_account_level_fields_land_on_every_row():
    rows = fp.parse_clearinghouse_state(
        _state([_pos(coin="BTC"), _pos(coin="ETH")], account_value="1234.5"), LONG)
    assert all(r["account_value"] == 1234.5 for r in rows)
    assert all(r["total_margin_used"] == 100.0 for r in rows)


def test_malformed_payload_raises_rather_than_returning_a_short_roster():
    for bad in [None, [], "nope", {"assetPositions": "not-a-list"}]:
        with pytest.raises((TypeError, ValueError)):
            fp.parse_clearinghouse_state(bad, LONG)


def test_row_key_is_unique_and_parsing_is_deterministic():
    payload = _state([_pos(coin="BTC"), _pos(coin="ETH")])
    a = fp.parse_clearinghouse_state(payload, LONG, ts_ms=111)
    b = fp.parse_clearinghouse_state(payload, LONG, ts_ms=111)
    assert a == b, "parsing is not deterministic"
    keys = [(r["ts_ms"], r["wallet"], r["coin"]) for r in a]
    assert len(keys) == len(set(keys)), "duplicate (ts_ms, wallet, coin) within one sweep"


def test_every_row_carries_the_full_schema():
    rows, _ = fp.sweep_positions(
        [LONG, FLAT, DEAD],
        fetch_fn=_fetcher({LONG: _state([_pos()]), FLAT: _state([]), DEAD: OSError("x")}),
        delay=0,
    )
    for r in rows:
        assert set(r) == set(fp.POSITION_COLUMNS), \
            f"row schema drift for status={r['status']}: {set(r) ^ set(fp.POSITION_COLUMNS)}"


def test_snapshot_round_trips_through_parquet(tmp_path):
    """Sign, nulls and status must survive the file, not just the parser."""
    rows, _ = fp.sweep_positions(
        [SHORT, DEAD],
        fetch_fn=_fetcher({SHORT: _state([_pos(coin="ETH", szi="-2.5", liq=None)]),
                           DEAD: OSError("x")}),
        delay=0,
    )
    path = fp.write_snapshot(rows, tmp_path, ts_ms=1_700_000_000_000)
    assert path is not None and path.exists()

    import pandas as pd
    df = pd.read_parquet(path)
    short_row = df[df["coin"] == "ETH"].iloc[0]
    assert short_row["size"] == -2.5
    assert pd.isna(short_row["liquidation_price"]), "null liquidation price did not survive"
    assert (df["status"] == "failed").sum() == 1, "the failed row did not survive the file"
    # Plain strings, not an Enum repr like "PositionStatus.OK" — a reader filtering on
    # status must not have to know this module's types.
    assert set(df["status"]) <= {"ok", "empty", "failed"}, f"status not stringified: {set(df['status'])}"


def test_write_snapshot_is_append_only(tmp_path):
    rows, _ = fp.sweep_positions([FLAT], fetch_fn=_fetcher({FLAT: _state([])}), delay=0)
    p1 = fp.write_snapshot(rows, tmp_path, ts_ms=1_700_000_000_000)
    p2 = fp.write_snapshot(rows, tmp_path, ts_ms=1_700_000_060_000)
    assert p1 != p2, "a second sweep overwrote the first"
    assert p1.parent == p2.parent


def test_empty_rows_writes_nothing(tmp_path):
    assert fp.write_snapshot([], tmp_path) is None

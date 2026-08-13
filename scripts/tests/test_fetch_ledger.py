"""WP-3 Part B — non-funding ledger backfill. The classifier is the unit.

Realised P&L is `Δaccount_value − Δ uPnL − net_flows`. Measured on 20,065 real WP-2 intervals,
the residual `Δav − ΔuPnL` has a 99th percentile of 0.43 of account value and **6.6 % of
intervals move account value by >2 % net of uPnL** — so inferring flows from snapshots alone
would file 6.6 % of the panel as unattributable. `userNonFundingLedgerUpdates` gives them
exactly, which is why this collector exists.

**The risk is not fetching, it is classification.** Only some delta types move the *perp*
account, and the payloads say so in different ways (verified live 2026-08-13):

    deposit             {usdc}                                   -> +usdc
    withdraw            {usdc, fee}                              -> -usdc
    subAccountTransfer  {usdc, user, destination}                -> sign by direction
    send                {usdcValue, user, destination,
                         sourceDex, destinationDex}              -> sign by dex side
    spotTransfer        {usdcValue, user, destination}           -> spot-side, no perp effect
    cStakingTransfer    {token: HYPE, amount, isDeposit}         -> not USDC, no perp effect
    gossipPriorityGasAuction {token: HYPE, amount}               -> not USDC, no perp effect
    spotGenesis         {token, amount}                          -> spot-side, no perp effect

`send` is the subtle one: `sourceDex` is `'spot'` or `''` (perp), so a **self**-send from spot
to perp is a real perp inflow — one observed at 1,968,806 USDC. A flat sign map would get it
backwards or miss it.

Contract encoded here:
  (a) each known type is signed correctly, from the payload, in both directions;
  (b) types with **no** perp effect contribute exactly 0 and are recorded as *known-zero*, not
      as unknown — the two must not be conflated;
  (c) **an unrecognised type is flagged, never silently zeroed.** A new venue type contributing
      a silent 0 would re-contaminate realised P&L, which is the failure this collector exists
      to remove;
  (d) paging is complete and deduped, and cannot spin;
  (e) transport faults retry, schema faults do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data import fetch_ledger as fl  # noqa: E402

ME = "0x" + "a" * 40
OTHER = "0x" + "b" * 40
HOUR = 3_600_000


def _e(t_ms: int, delta: dict, h: str | None = None) -> dict:
    return {"time": t_ms, "hash": h or f"0x{t_ms:064x}", "delta": delta}


# ── (a) known types, signed from the payload ─────────────────────────────────────

def test_deposit_is_a_positive_perp_flow():
    assert fl.perp_flow_usdc(_e(0, {"type": "deposit", "usdc": "29.0"}), ME) == 29.0


def test_withdraw_is_a_negative_perp_flow():
    assert fl.perp_flow_usdc(
        _e(0, {"type": "withdraw", "usdc": "122.0", "fee": "1.0"}), ME) == -122.0


def test_subaccount_transfer_is_signed_by_direction():
    out = _e(0, {"type": "subAccountTransfer", "usdc": "1.0",
                 "user": ME, "destination": OTHER})
    inn = _e(0, {"type": "subAccountTransfer", "usdc": "1.0",
                 "user": OTHER, "destination": ME})
    assert fl.perp_flow_usdc(out, ME) == -1.0
    assert fl.perp_flow_usdc(inn, ME) == +1.0


def test_send_spot_to_perp_is_an_inflow_even_when_self_addressed():
    """The observed 1,968,806 USDC case: user == destination, sourceDex 'spot' -> perp."""
    e = _e(0, {"type": "send", "token": "USDC", "usdcValue": "1968806.12",
               "user": ME, "destination": ME, "sourceDex": "spot", "destinationDex": ""})
    assert fl.perp_flow_usdc(e, ME) == pytest.approx(+1968806.12)


def test_send_perp_to_spot_is_an_outflow():
    e = _e(0, {"type": "send", "token": "USDC", "usdcValue": "500.0",
               "user": ME, "destination": ME, "sourceDex": "", "destinationDex": "spot"})
    assert fl.perp_flow_usdc(e, ME) == -500.0


def test_send_perp_to_another_user_is_an_outflow():
    e = _e(0, {"type": "send", "token": "USDC", "usdcValue": "10.0",
               "user": ME, "destination": OTHER, "sourceDex": "", "destinationDex": ""})
    assert fl.perp_flow_usdc(e, ME) == -10.0


def test_a_non_usdc_send_moves_no_usdc_into_the_perp_account():
    e = _e(0, {"type": "send", "token": "HYPE", "amount": "5.0", "usdcValue": "0.0",
               "user": ME, "destination": OTHER, "sourceDex": "", "destinationDex": ""})
    assert fl.perp_flow_usdc(e, ME) == 0.0


# ── (b) known-zero is not unknown ────────────────────────────────────────────────

@pytest.mark.parametrize("delta", [
    {"type": "spotTransfer", "token": "USDC", "usdcValue": "50.0",
     "user": ME, "destination": OTHER},
    {"type": "cStakingTransfer", "token": "HYPE", "amount": "2.0", "isDeposit": True},
    {"type": "gossipPriorityGasAuction", "token": "HYPE", "amount": "0.85"},
    {"type": "spotGenesis", "token": "MAX", "amount": "0.0"},
], ids=["spotTransfer", "cStakingTransfer", "gasAuction", "spotGenesis"])
def test_types_with_no_perp_effect_are_known_zero(delta):
    assert fl.perp_flow_usdc(_e(0, delta), ME) == 0.0
    assert fl.is_known_type(delta["type"]), \
        f"{delta['type']} contributes 0 but must be KNOWN-zero, not unknown"


# ── (c) the decisive one: unknown types are flagged, never silently zeroed ───────

def test_an_unrecognised_type_is_flagged_not_zeroed():
    e = _e(0, {"type": "someNewVenueThing", "usdc": "1000.0"})
    assert not fl.is_known_type("someNewVenueThing")
    with pytest.raises(fl.UnknownDeltaType):
        fl.perp_flow_usdc(e, ME)


def test_net_flow_reports_unknown_types_rather_than_dropping_them():
    entries = [
        _e(1 * HOUR, {"type": "deposit", "usdc": "100.0"}),
        _e(2 * HOUR, {"type": "someNewVenueThing", "usdc": "9999.0"}),
        _e(3 * HOUR, {"type": "withdraw", "usdc": "40.0", "fee": "0.0"}),
    ]
    net, report = fl.net_perp_flow(entries, ME, 0, 10 * HOUR)
    assert net == pytest.approx(60.0), "known flows must still net correctly"
    assert report["unknown_types"] == {"someNewVenueThing": 1}
    assert report["unknown_count"] == 1, "an unknown type vanished from the report"


def test_window_bounds_are_respected():
    entries = [_e(1 * HOUR, {"type": "deposit", "usdc": "10.0"}),
               _e(5 * HOUR, {"type": "deposit", "usdc": "20.0"}),
               _e(9 * HOUR, {"type": "deposit", "usdc": "40.0"})]
    net, _ = fl.net_perp_flow(entries, ME, 2 * HOUR, 6 * HOUR)
    assert net == 20.0


# ── (d) paging ───────────────────────────────────────────────────────────────────

def _paged(total: int, page: int = None):
    page = page or fl.PAGE_LIMIT
    rows = [_e(i * HOUR, {"type": "deposit", "usdc": "1.0"}) for i in range(total)]

    def fetch(wallet, s, e):
        return [r for r in rows if s <= r["time"] <= e][:page]
    return fetch


def test_paging_covers_the_window_without_duplicates():
    n = fl.PAGE_LIMIT + 250
    got = fl.fetch_ledger_history(ME, 0, n * HOUR, fetch_fn=_paged(n), delay=0)
    assert len(got) == n
    keys = [(r["time"], r["hash"]) for r in got]
    assert len(keys) == len(set(keys)), "a page boundary duplicated an entry"
    assert [r["time"] for r in got] == sorted(r["time"] for r in got)


def test_a_non_advancing_venue_terminates():
    stuck = [_e(0, {"type": "deposit", "usdc": "1.0"})] * 5
    calls = {"n": 0}

    def fetch(wallet, s, e):
        calls["n"] += 1
        return stuck

    fl.fetch_ledger_history(ME, 0, 10_000 * HOUR, fetch_fn=fetch, delay=0)
    assert calls["n"] < 50, "the paging loop did not terminate"


def test_empty_history_is_empty_not_an_error():
    assert fl.fetch_ledger_history(ME, 0, HOUR, fetch_fn=lambda w, s, e: [], delay=0) == []


# ── (e) retry policy ─────────────────────────────────────────────────────────────

def test_transport_fault_retries_and_schema_fault_does_not(monkeypatch):
    monkeypatch.setattr(fl.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(wallet, s, e):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("HTTP Error 429: Too Many Requests")
        return [_e(0, {"type": "deposit", "usdc": "1.0"})]

    assert len(fl.fetch_ledger_history(ME, 0, HOUR, fetch_fn=flaky, delay=0)) == 1
    assert calls["n"] == 2

    with pytest.raises(KeyError):
        fl.fetch_ledger_history(ME, 0, HOUR, delay=0,
                                fetch_fn=lambda w, s, e: (_ for _ in ()).throw(KeyError("delta")))


# ── sweep + storage ──────────────────────────────────────────────────────────────

def test_one_wallet_failure_does_not_abort_and_arithmetic_closes(tmp_path):
    def fetch(wallet, s, e):
        if wallet == OTHER:
            raise OSError("down")
        return [_e(0, {"type": "deposit", "usdc": "1.0"})]

    _, report = fl.backfill([ME, OTHER, "not-an-address"], 0, HOUR,
                            out_dir=tmp_path, fetch_fn=fetch, delay=0, retries=1)
    assert report["ok"] + report["failed"] + report["rejected"] == report["n_requested"] == 3
    assert report["rejected"] == 1 and report["failed"] == 1


def test_rerun_extends_rather_than_rewrites(tmp_path):
    import pandas as pd
    first = [_e(i * HOUR, {"type": "deposit", "usdc": "1.0"}) for i in range(10)]
    fl.write_wallet(first, tmp_path, ME)
    overlap = [_e(i * HOUR, {"type": "deposit", "usdc": "1.0"}) for i in range(5, 15)]
    path = fl.write_wallet(overlap, tmp_path, ME)
    df = pd.read_parquet(path)
    assert len(df) == 15
    assert not df.duplicated(subset=["time", "hash"]).any()


def test_hostile_wallet_never_reaches_the_filesystem(tmp_path):
    with pytest.raises(ValueError):
        fl.write_wallet([_e(0, {"type": "deposit", "usdc": "1.0"})], tmp_path, "../evil")
    assert list(tmp_path.iterdir()) == []


# ── types surfaced by the first real universe backfill (2026-08-13) ──────────────

def test_account_class_transfer_is_signed_by_toPerp():
    """The dominant flow: 1,574 of the first backfill's entries, and initially unknown.

    A design that zeroed unknown types silently would have dropped this one and corrupted
    realised P&L for most wallets without a symptom.
    """
    into = _e(0, {"type": "accountClassTransfer", "toPerp": True, "usdc": "255650.63"})
    out = _e(0, {"type": "accountClassTransfer", "toPerp": False, "usdc": "255650.63"})
    assert fl.perp_flow_usdc(into, ME) == pytest.approx(+255650.63)
    assert fl.perp_flow_usdc(out, ME) == pytest.approx(-255650.63)


def test_internal_transfer_is_signed_by_direction():
    inn = _e(0, {"type": "internalTransfer", "usdc": "5.0",
                 "user": OTHER, "destination": ME, "fee": "0.0"})
    out = _e(0, {"type": "internalTransfer", "usdc": "5.0",
                 "user": ME, "destination": OTHER, "fee": "0.0"})
    assert fl.perp_flow_usdc(inn, ME) == +5.0
    assert fl.perp_flow_usdc(out, ME) == -5.0


def test_liquidation_is_zero_flow_because_it_is_a_trade_not_a_transfer():
    """Subtracting a liquidation as a 'flow' would erase the very loss the ranking must see."""
    e = _e(0, {"type": "liquidation", "accountValue": "60571.97",
               "leverageType": "Isolated", "liquidatedNtlPos": "2570352.68",
               "liquidatedPositions": [{"coin": "xyz:SKHX", "szi": "2675.731"}]})
    assert fl.is_known_type("liquidation")
    assert fl.perp_flow_usdc(e, ME) == 0.0


@pytest.mark.parametrize("t", fl.UNRESOLVED_TYPES)
def test_unresolved_types_stay_flagged_rather_than_guessed(t):
    """Their perp effect is not established, and a guessed sign is worse than a flag.

    Reconciliation against the WP-2 panel is not yet possible (ledger 90d, panel 4d), so these
    must keep raising until it is.
    """
    assert not fl.is_known_type(t)
    with pytest.raises(fl.UnknownDeltaType):
        fl.perp_flow_usdc(_e(0, {"type": t, "amount": "1.0", "token": "USDC"}), ME)

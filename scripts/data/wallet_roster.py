"""WP-1 — the wallet roster: derived, never hardcoded.

Step 1 of `docs/specs/wallet_positioning.md`, and the unblocker for the K2 dead columns.
`rust/ing/src/state/mod.rs:127` initialises `position_state` to `None`; with no roster there
are no positions, so the 13 liquidation and 15 concentration features are NaN **by
construction** rather than by defect. This module produces the list those pollers need.

**Why this family is worth unblocking** (`research/MECHANISM_FAMILIES.md`, family 5):
Hyperliquid is a fully on-chain perp DEX, so account positions, entry prices, leverage and
therefore **liquidation prices** are public. A perp liquidation is not Coval & Stafford's fire
sale — their seller *chooses* to sell under redemption pressure, whereas here an engine executes
automatically at a price anyone can compute in advance. The equity literature has to *infer*
forced flow from returns; this venue lets you observe it.

**Source, verified live 2026-08-10 rather than assumed** (the spec insists on this, and it was
right to): the venue's `info` endpoint returns **422** for a `leaderboard` request. The
leaderboard lives on a *different host* — `stats-data.hyperliquid.xyz/Mainnet/leaderboard` —
returning `{"leaderboardRows": [...]}`, 41,475 wallets / ~34 MB in about a second. Each row
carries `ethAddress`, `accountValue` and `windowPerformances` (day / week / month / allTime,
each with pnl, roi and vlm), which is also the P&L history cohort construction (WP-3) needs.

**Non-negotiables, both enforced by test:**

* **Derived, never hardcoded.** A pinned list rots the moment the cohort turns over — and
  cohort turnover is one of the hypotheses under test (spec §3, failure mode 1). A source scan
  asserts no literal address survives here.
* **Addresses are validated before they can reach a path or an API call.** Names become
  filenames (the XS-1 lesson), so anything that is not a 0x-prefixed 40-hex string is dropped
  at parse time, upstream of every consumer.

A malformed payload **raises** rather than returning a short roster: a silently truncated
roster narrows every downstream cohort claim invisibly, which is the shape of the XS-1 sweep
that reported `ok=177` while collecting 4 % of what it asked for.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = ["ADDRESS_RE", "WalletRef", "is_valid_address", "parse_leaderboard",
           "fetch_roster", "LEADERBOARD_URL"]

#: The leaderboard is NOT on the info endpoint — that returns 422. Verified live 2026-08-10.
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

#: A wallet address reaches filesystem paths and API URLs, so it is validated before use.
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

DEFAULT_WINDOW = "month"


def is_valid_address(addr) -> bool:
    """True only for a 0x-prefixed 40-hex string. Everything else is rejected."""
    return isinstance(addr, str) and bool(ADDRESS_RE.match(addr))


@dataclass(frozen=True)
class WalletRef:
    """One roster entry, carrying its provenance."""
    address: str
    source: str
    first_seen: str
    account_value: float = 0.0
    #: traded notional over the selection window — the spec's `notional_seen`
    notional_seen: float = 0.0
    window_pnl: float = 0.0
    extras: dict = field(default_factory=dict)


def _window(row: dict, window: str) -> dict:
    for entry in row.get("windowPerformances") or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == window:
            return entry[1] if isinstance(entry[1], dict) else {}
    return {}


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_leaderboard(payload, window: str = DEFAULT_WINDOW,
                      source: str = "leaderboard") -> list[WalletRef]:
    """Turn a leaderboard payload into validated `WalletRef`s.

    Raises on a payload whose *shape* is wrong. Individual rows with unusable addresses are
    dropped — one malformed row must not poison the roster — but a payload where **nothing**
    validates raises too, because an empty roster is never a legitimate result: it means the
    schema moved.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"leaderboard payload must be an object, got {type(payload).__name__}")
    rows = payload.get("leaderboardRows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("leaderboard payload has no non-empty 'leaderboardRows'")

    now = datetime.now(timezone.utc).isoformat()
    # Keyed by address so a wallet appearing twice collapses to one entry. Resolution is by
    # account_value rather than payload order, so the result does not depend on how the venue
    # happened to serialise its rows.
    seen: dict[str, WalletRef] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"leaderboard row is not an object: {str(row)[:60]!r}")
        addr = row.get("ethAddress")
        if not is_valid_address(addr):
            continue                       # dropped upstream of any path or API use
        w = _window(row, window)
        ref = WalletRef(
            address=addr, source=source, first_seen=now,
            account_value=_f(row.get("accountValue")),
            notional_seen=_f(w.get("vlm")),
            window_pnl=_f(w.get("pnl")),
            extras={"roi": _f(w.get("roi")), "window": window},
        )
        prev = seen.get(addr)
        if prev is None or ref.account_value > prev.account_value:
            seen[addr] = ref

    out = list(seen.values())
    if not out:
        raise ValueError(
            f"no row in a {len(rows)}-row leaderboard carried a valid address — "
            "the payload shape has probably changed"
        )
    return out


def _default_info_fn() -> dict:
    req = urllib.request.Request(LEADERBOARD_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def fetch_roster(info_fn=None, *, window: str = DEFAULT_WINDOW, limit: int | None = None,
                 min_account_value: float | None = None, min_window_pnl: float | None = None,
                 min_notional_seen: float | None = None, rank_by: str = "account_value",
                 retries: int = 2, backoff: float = 1.0) -> list[WalletRef]:
    """Fetch and rank the roster. `info_fn` is injected so the suite runs offline.

    **Transport faults retry; schema faults do not.** A 429 or a reset connection deserves a
    backoff — a payload whose shape changed deserves to fail loudly and once, rather than be
    retried into the same wall. (This is the failure that killed the XS-8 sampler at startup.)

    Ordering is by `rank_by` descending, then address ascending, so ties break
    deterministically and `limit` selects the top rather than an arbitrary slice.

    **`min_notional_seen` is not optional in practice.** The live leaderboard's largest
    accounts are vaults, the bridge and protocol treasuries: the top entry holds $14.1 bn with
    **zero** traded volume in the window. Ranked on balance alone the roster is a list of
    custodial accounts, and a `min_window_pnl >= 0` filter admits them all because their P&L is
    exactly zero. A wallet that has not traded is not a trader whatever its balance — so a
    volume floor is what makes this a *trader* roster. Found by the real-data smoke; the
    synthetic fixtures could not have shown it.
    """
    fn = info_fn or _default_info_fn

    last: Exception | None = None
    payload = None
    for attempt in range(retries + 1):
        try:
            payload = fn()
            break
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc                                  # transport / transport-shaped only
            if attempt < retries and backoff:
                time.sleep(backoff * (2 ** attempt))
    else:
        raise last if last else OSError("roster fetch failed")

    refs = parse_leaderboard(payload, window=window)     # schema errors propagate, unretried

    if min_account_value is not None:
        refs = [r for r in refs if r.account_value >= min_account_value]
    if min_window_pnl is not None:
        refs = [r for r in refs if r.window_pnl >= min_window_pnl]
    if min_notional_seen is not None:
        refs = [r for r in refs if r.notional_seen >= min_notional_seen]

    key = {"account_value": lambda r: r.account_value,
           "notional_seen": lambda r: r.notional_seen,
           "window_pnl": lambda r: r.window_pnl}.get(rank_by)
    if key is None:
        raise ValueError(f"unknown rank_by {rank_by!r}; use account_value|notional_seen|window_pnl")
    refs.sort(key=lambda r: (-key(r), r.address))
    return refs[:limit] if limit else refs

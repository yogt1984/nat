"""
Hyperliquid Exchange Client

Read-only info operations (no auth) + authenticated order placement.
Follows the same pattern as scripts/polymarket/client.py.

Info endpoints (public, no auth):
  - get_meta()           → asset metadata (name, sz_decimals, max_leverage)
  - get_midprices()      → current mid prices for all assets
  - get_positions(wallet) → open positions for a wallet
  - get_fills(wallet)    → recent trade fills
  - get_open_orders(wallet) → pending orders

Exchange endpoints (require private key):
  - place_order(symbol, is_buy, price, size, ...)
  - cancel_order(symbol, oid)
  - cancel_all(symbol)

Credentials:
  Set env var: HL_PRIVATE_KEY (hex-encoded private key, with or without 0x prefix)

Usage:
    from execution.hyperliquid_client import HyperliquidClient

    # Read-only
    client = HyperliquidClient.readonly()
    meta = client.get_meta()
    prices = client.get_midprices()

    # Authenticated
    client = HyperliquidClient.from_env()
    result = client.place_order("BTC", is_buy=True, price=67000.0, size=0.01)

    # Dry-run (logs orders, doesn't execute)
    client = HyperliquidClient.from_env(dry_run=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import msgpack
import requests
from eth_account import Account

try:
    from eth_account.messages import encode_structured_data as _encode_typed
except ImportError:
    from eth_account.messages import encode_typed_data as _encode_typed

log = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
TIMEOUT = 30

# EIP-712 domain for Hyperliquid mainnet
EIP712_DOMAIN = {
    "name": "Exchange",
    "version": "1",
    "chainId": 42161,  # Arbitrum One
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

# Coin name → asset index mapping (populated from get_meta)
_ASSET_INDEX_CACHE: dict[str, int] = {}


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class AssetMeta:
    name: str
    index: int
    sz_decimals: int
    max_leverage: float | None = None


@dataclass
class Position:
    coin: str
    size: float  # signed (negative = short)
    entry_price: float
    position_value: float
    unrealized_pnl: float
    liquidation_price: float | None
    leverage: float
    margin_used: float


@dataclass
class Fill:
    coin: str
    price: float
    size: float
    side: str  # "B" or "A"
    time_ms: int
    fee: float
    closed_pnl: float
    oid: int


@dataclass
class OrderResult:
    success: bool
    status: str  # "ok", "error", "dry_run"
    order_id: int | None = None
    error: str | None = None
    raw: dict | None = None


# ── Client ───────────────────────────────────────────────────────────────

class HyperliquidClient:

    def __init__(
        self,
        private_key: str | None = None,
        dry_run: bool = False,
    ):
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._private_key = private_key
        self._account = Account.from_key(private_key) if private_key else None
        self._wallet = self._account.address if self._account else None
        self._dry_run = dry_run
        self._authenticated = private_key is not None
        self._meta_cache: list[AssetMeta] | None = None

    @classmethod
    def readonly(cls) -> HyperliquidClient:
        return cls(private_key=None, dry_run=False)

    @classmethod
    def from_env(cls, dry_run: bool = False) -> HyperliquidClient:
        pk = os.environ.get("HL_PRIVATE_KEY")
        if not pk:
            raise EnvironmentError(
                "HL_PRIVATE_KEY not set. Export your hex private key."
            )
        if not pk.startswith("0x"):
            pk = "0x" + pk
        return cls(private_key=pk, dry_run=dry_run)

    @property
    def wallet(self) -> str | None:
        return self._wallet

    def _require_auth(self):
        if not self._authenticated:
            raise PermissionError(
                "This operation requires authentication. "
                "Use HyperliquidClient.from_env() or pass a private key."
            )

    # ── Info endpoints (public) ──────────────────────────────────────────

    def _info_request(self, payload: dict) -> dict | list:
        resp = self._session.post(INFO_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_meta(self) -> list[AssetMeta]:
        """Fetch asset metadata. Caches result."""
        if self._meta_cache is not None:
            return self._meta_cache

        data = self._info_request({"type": "meta"})
        assets = []
        for i, u in enumerate(data.get("universe", [])):
            asset = AssetMeta(
                name=u["name"],
                index=i,
                sz_decimals=u.get("szDecimals", 0),
                max_leverage=u.get("maxLeverage"),
            )
            assets.append(asset)
            _ASSET_INDEX_CACHE[u["name"]] = i

        self._meta_cache = assets
        return assets

    def get_asset_index(self, symbol: str) -> int:
        """Get the numeric asset index for a symbol (e.g., 'BTC' → 0)."""
        if symbol not in _ASSET_INDEX_CACHE:
            self.get_meta()
        if symbol not in _ASSET_INDEX_CACHE:
            raise ValueError(f"Unknown symbol: {symbol}")
        return _ASSET_INDEX_CACHE[symbol]

    def get_midprices(self) -> dict[str, float]:
        """Get current mid prices for all assets."""
        data = self._info_request({"type": "allMids"})
        return {k: float(v) for k, v in data.items()}

    def get_positions(self, wallet: str | None = None) -> list[Position]:
        """Get open positions for a wallet."""
        wallet = wallet or self._wallet
        if not wallet:
            raise ValueError("No wallet address. Authenticate or pass wallet.")

        data = self._info_request({
            "type": "clearinghouseState",
            "user": wallet,
        })
        positions = []
        for ap in data.get("assetPositions", []):
            p = ap.get("position", {})
            size = float(p.get("szi", "0"))
            if abs(size) < 1e-12:
                continue
            positions.append(Position(
                coin=p.get("coin", ""),
                size=size,
                entry_price=float(p.get("entryPx", "0") or "0"),
                position_value=float(p.get("positionValue", "0")),
                unrealized_pnl=float(p.get("unrealizedPnl", "0")),
                liquidation_price=float(p["liquidationPx"]) if p.get("liquidationPx") else None,
                leverage=float(p.get("leverage", {}).get("value", 1)),
                margin_used=float(p.get("marginUsed", "0") or "0"),
            ))
        return positions

    def get_account_value(self, wallet: str | None = None) -> float:
        """Get total account value in USD."""
        wallet = wallet or self._wallet
        if not wallet:
            raise ValueError("No wallet address.")
        data = self._info_request({
            "type": "clearinghouseState",
            "user": wallet,
        })
        return float(data.get("marginSummary", {}).get("accountValue", "0"))

    def get_fills(self, wallet: str | None = None) -> list[Fill]:
        """Get recent trade fills."""
        wallet = wallet or self._wallet
        if not wallet:
            raise ValueError("No wallet address.")

        data = self._info_request({
            "type": "userFills",
            "user": wallet,
        })
        fills = []
        for f in data:
            fills.append(Fill(
                coin=f["coin"],
                price=float(f["px"]),
                size=float(f["sz"]),
                side=f["side"],
                time_ms=int(f["time"]),
                fee=float(f["fee"]),
                closed_pnl=float(f["closedPnl"]),
                oid=int(f["oid"]),
            ))
        return fills

    def get_open_orders(self, wallet: str | None = None) -> list[dict]:
        """Get open orders for a wallet."""
        wallet = wallet or self._wallet
        if not wallet:
            raise ValueError("No wallet address.")
        return self._info_request({
            "type": "openOrders",
            "user": wallet,
        })

    # ── EIP-712 signing ──────────────────────────────────────────────────

    def _sign_action(self, action: dict, nonce: int) -> dict:
        """Sign an exchange action using EIP-712 phantom agent."""
        # Serialize action to msgpack and hash
        action_bytes = msgpack.packb(action, use_bin_type=True)
        action_hash = hashlib.sha256(action_bytes).digest()

        # Construct EIP-712 typed data
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
            },
            "primaryType": "Agent",
            "domain": EIP712_DOMAIN,
            "message": {
                "source": "a",  # "a" = API source
                "connectionId": action_hash,
            },
        }

        signable = _encode_typed(typed_data)
        signed = self._account.sign_message(signable)

        return {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }

    def _exchange_request(self, action: dict, nonce: int | None = None) -> dict:
        """Send a signed exchange request."""
        self._require_auth()

        if nonce is None:
            nonce = int(time.time() * 1000)

        if self._dry_run:
            log.info(f"[DRY RUN] action={json.dumps(action)}, nonce={nonce}")
            return {"status": "ok", "response": {"type": "dry_run"}}

        signature = self._sign_action(action, nonce)

        payload = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }

        resp = self._session.post(EXCHANGE_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    # ── Exchange endpoints (authenticated) ───────────────────────────────

    def place_order(
        self,
        symbol: str,
        is_buy: bool,
        price: float,
        size: float,
        reduce_only: bool = False,
        order_type: str = "limit",
        tif: str = "Gtc",
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            symbol: Asset name (e.g., "BTC", "ETH")
            is_buy: True for buy/long, False for sell/short
            price: Limit price in USD
            size: Order size in asset units (e.g., 0.01 BTC)
            reduce_only: Only reduce existing position
            order_type: "limit" (default) or "market"
            tif: Time-in-force: "Gtc", "Ioc", "Alo" (add liquidity only / maker)
        """
        asset_idx = self.get_asset_index(symbol)

        # Round size to sz_decimals
        meta = self.get_meta()
        sz_dec = meta[asset_idx].sz_decimals
        size = round(size, sz_dec)

        if order_type == "market":
            ot = {"market": {}}
        else:
            ot = {"limit": {"tif": tif}}

        order = {
            "a": asset_idx,
            "b": is_buy,
            "p": str(price),
            "s": str(size),
            "r": reduce_only,
            "t": ot,
        }

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na",
        }

        side_str = "BUY" if is_buy else "SELL"
        log.info(f"Order: {side_str} {size} {symbol} @ {price} ({tif})")

        try:
            result = self._exchange_request(action)
            resp_data = result.get("response", {})
            resp_type = resp_data.get("type", "")

            if resp_type == "dry_run":
                return OrderResult(
                    success=True, status="dry_run",
                    raw={"action": action},
                )

            if resp_type == "order":
                statuses = resp_data.get("data", {}).get("statuses", [])
                if statuses and "resting" in statuses[0]:
                    oid = statuses[0]["resting"]["oid"]
                    return OrderResult(success=True, status="ok", order_id=oid, raw=result)
                elif statuses and "filled" in statuses[0]:
                    oid = statuses[0]["filled"]["oid"]
                    return OrderResult(success=True, status="filled", order_id=oid, raw=result)
                elif statuses and "error" in statuses[0]:
                    return OrderResult(
                        success=False, status="error",
                        error=statuses[0]["error"], raw=result,
                    )

            return OrderResult(success=True, status="ok", raw=result)

        except Exception as e:
            return OrderResult(success=False, status="error", error=str(e))

    def cancel_order(self, symbol: str, oid: int) -> OrderResult:
        """Cancel an order by order ID."""
        asset_idx = self.get_asset_index(symbol)

        action = {
            "type": "cancel",
            "cancels": [{"a": asset_idx, "o": oid}],
        }

        log.info(f"Cancel: {symbol} oid={oid}")

        try:
            result = self._exchange_request(action)
            return OrderResult(success=True, status="ok", raw=result)
        except Exception as e:
            return OrderResult(success=False, status="error", error=str(e))

    def cancel_all(self, symbol: str | None = None) -> list[OrderResult]:
        """Cancel all open orders, optionally filtered by symbol."""
        orders = self.get_open_orders()
        results = []
        for o in orders:
            coin = o.get("coin", "")
            if symbol and coin != symbol:
                continue
            oid = o.get("oid")
            if oid:
                results.append(self.cancel_order(coin, int(oid)))
        return results

    # ── Convenience ──────────────────────────────────────────────────────

    def place_maker_order(
        self,
        symbol: str,
        is_buy: bool,
        price: float,
        size: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a post-only (maker) limit order. Rejects if it would take."""
        return self.place_order(
            symbol, is_buy, price, size,
            reduce_only=reduce_only,
            order_type="limit",
            tif="Alo",  # Add Liquidity Only
        )

    def __repr__(self):
        mode = "dry_run" if self._dry_run else ("auth" if self._authenticated else "readonly")
        wallet = self._wallet[:10] + "..." if self._wallet else "none"
        return f"HyperliquidClient(mode={mode}, wallet={wallet})"

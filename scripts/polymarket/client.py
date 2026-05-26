"""
Polymarket CLOB Client

Thin wrapper around py-clob-client that handles:
  - Credential management (env vars or config file)
  - Read-only mode (no wallet needed for market data)
  - Order placement with fee calculation
  - Position tracking

Credentials:
  Set env vars: PK (private key), CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE
  Or call derive_credentials() once to generate API keys from wallet.

Usage:
    from polymarket.client import PolymarketClient

    # Read-only (no auth needed)
    client = PolymarketClient()
    book = client.get_order_book(token_id)
    markets = client.get_all_markets()

    # Authenticated (for trading)
    client = PolymarketClient.from_env()
    client.place_limit_order(token_id, side="BUY", price=0.55, size=100)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
)

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: str = ""
    status: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


class PolymarketClient:
    """Polymarket CLOB API client."""

    def __init__(
        self,
        host: str = CLOB_HOST,
        chain_id: int = CHAIN_ID,
        private_key: str | None = None,
        creds: ApiCreds | None = None,
    ):
        kwargs = {"host": host, "chain_id": chain_id}
        if private_key:
            kwargs["key"] = private_key
        if creds:
            kwargs["creds"] = creds

        self._client = ClobClient(**kwargs)
        self._authenticated = private_key is not None and creds is not None
        self._readonly = not self._authenticated

    @classmethod
    def from_env(cls) -> "PolymarketClient":
        """
        Create authenticated client from environment variables.

        Required env vars:
          PK              - Ethereum private key (hex, with or without 0x prefix)
          CLOB_API_KEY    - API key from derive_credentials()
          CLOB_SECRET     - API secret
          CLOB_PASS_PHRASE - API passphrase
        """
        pk = os.environ.get("PK")
        if not pk:
            raise EnvironmentError(
                "PK env var not set. Set your Polygon wallet private key."
            )

        api_key = os.environ.get("CLOB_API_KEY", "")
        api_secret = os.environ.get("CLOB_SECRET", "")
        api_passphrase = os.environ.get("CLOB_PASS_PHRASE", "")

        creds = None
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )

        client = cls(private_key=pk, creds=creds)

        # If no API creds, derive them
        if creds is None:
            logger.info("No API creds found, deriving from wallet...")
            client.derive_credentials()

        return client

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PolymarketClient":
        """
        Create client from a JSON config file.

        Config format:
        {
            "private_key": "0x...",
            "api_key": "...",
            "api_secret": "...",
            "api_passphrase": "..."
        }
        """
        with open(config_path) as f:
            cfg = json.load(f)

        creds = None
        if cfg.get("api_key"):
            creds = ApiCreds(
                api_key=cfg["api_key"],
                api_secret=cfg["api_secret"],
                api_passphrase=cfg["api_passphrase"],
            )

        return cls(private_key=cfg.get("private_key"), creds=creds)

    @classmethod
    def readonly(cls) -> "PolymarketClient":
        """Create a read-only client (no auth, market data only)."""
        return cls()

    def derive_credentials(self) -> ApiCreds:
        """
        Derive API credentials from wallet. Only needs to be done once.
        Stores credentials and returns them.
        """
        creds = self._client.create_or_derive_api_creds()
        self._client.set_api_creds(creds)
        self._authenticated = True
        self._readonly = False
        logger.info("API credentials derived successfully")
        return creds

    # ------------------------------------------------------------------
    # Market data (no auth required)
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the CLOB API is reachable."""
        try:
            resp = self._client.get_ok()
            return resp == "OK"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_server_time(self) -> int:
        """Get server time (unix timestamp)."""
        return self._client.get_server_time()

    def get_order_book(self, token_id: str) -> dict:
        """
        Get order book for a token.

        Returns dict with 'bids', 'asks' (lists of {price, size}),
        'market', 'asset_id', 'hash'.
        """
        book = self._client.get_order_book(token_id)
        return {
            "market": book.market,
            "asset_id": book.asset_id,
            "bids": [{"price": float(b.price), "size": float(b.size)} for b in book.bids],
            "asks": [{"price": float(a.price), "size": float(a.size)} for a in book.asks],
            "hash": book.hash,
        }

    def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        try:
            mid = self._client.get_midpoint(token_id)
            return float(mid) if mid else None
        except Exception:
            return None

    def get_spread(self, token_id: str) -> dict | None:
        """Get bid-ask spread for a token."""
        try:
            return self._client.get_spread(token_id)
        except Exception:
            return None

    def get_last_trade_price(self, token_id: str) -> float | None:
        """Get last trade price."""
        try:
            p = self._client.get_last_trade_price(token_id)
            return float(p) if p else None
        except Exception:
            return None

    def get_tick_size(self, token_id: str) -> str:
        """Get minimum tick size for a token."""
        return self._client.get_tick_size(token_id)

    def get_fee_rate(self, token_id: str) -> int:
        """Get fee rate in bps for a token."""
        return self._client.get_fee_rate_bps(token_id)

    def get_all_markets(self) -> list[dict]:
        """
        Fetch all markets from the CLOB API (paginated).
        Returns list of market dicts.
        """
        markets = []
        cursor = "MA=="  # Base64 of "0"
        end_cursor = "LTE="  # Base64 of "-1"

        while cursor != end_cursor:
            try:
                resp = self._client.get_markets(next_cursor=cursor)
            except Exception as e:
                logger.error(f"Failed to fetch markets at cursor {cursor}: {e}")
                break

            if isinstance(resp, dict):
                data = resp.get("data", [])
                cursor = resp.get("next_cursor", end_cursor)
            elif isinstance(resp, list):
                data = resp
                cursor = end_cursor
            else:
                break

            markets.extend(data)

            if cursor == end_cursor:
                break

            time.sleep(0.1)  # Rate limit courtesy

        logger.info(f"Fetched {len(markets)} markets from CLOB")
        return markets

    def get_market(self, condition_id: str) -> dict:
        """Get a single market by condition_id."""
        return self._client.get_market(condition_id)

    # ------------------------------------------------------------------
    # Order management (auth required)
    # ------------------------------------------------------------------

    def _require_auth(self):
        if self._readonly:
            raise PermissionError(
                "Client is read-only. Use PolymarketClient.from_env() for trading."
            )

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
        tick_size: str = "0.01",
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            token_id: The outcome token to trade
            side: "BUY" or "SELL"
            price: Limit price (0 to 1)
            size: Number of shares
            order_type: GTC, GTD, FOK, or FAK
            tick_size: Minimum tick size for this market
        """
        self._require_auth()

        ot = getattr(OrderType, order_type, OrderType.GTC)

        args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )
        options = PartialCreateOrderOptions(tick_size=tick_size)

        try:
            resp = self._client.create_and_post_order(args, options)
            if isinstance(resp, dict):
                return OrderResult(
                    success=resp.get("success", False),
                    order_id=resp.get("orderID", resp.get("order_id", "")),
                    status=resp.get("status", ""),
                    raw=resp,
                )
            return OrderResult(success=True, raw={"response": str(resp)})
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def place_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
        tick_size: str = "0.01",
    ) -> OrderResult:
        """
        Place a market order (fill-or-kill).

        Args:
            token_id: The outcome token to trade
            side: "BUY" or "SELL"
            amount: USDC amount to spend (for BUY) or shares to sell (for SELL)
            tick_size: Minimum tick size for this market
        """
        self._require_auth()

        args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=side,
        )
        options = PartialCreateOrderOptions(tick_size=tick_size)

        try:
            resp = self._client.create_market_order(args, options)
            if isinstance(resp, dict):
                return OrderResult(
                    success=resp.get("success", False),
                    order_id=resp.get("orderID", resp.get("order_id", "")),
                    status=resp.get("status", ""),
                    raw=resp,
                )
            return OrderResult(success=True, raw={"response": str(resp)})
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        self._require_auth()
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def cancel_all(self) -> bool:
        """Cancel all open orders."""
        self._require_auth()
        try:
            self._client.cancel_all()
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False

    def get_orders(self) -> list[dict]:
        """Get open orders."""
        self._require_auth()
        try:
            return self._client.get_orders()
        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []

    def get_trades(self) -> list[dict]:
        """Get recent trades."""
        self._require_auth()
        try:
            return self._client.get_trades()
        except Exception as e:
            logger.error(f"Get trades failed: {e}")
            return []

    def get_balance(self, asset_type: str = "COLLATERAL") -> dict:
        """Get balance and allowance info."""
        self._require_auth()
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            at = AssetType.COLLATERAL if asset_type == "COLLATERAL" else AssetType.CONDITIONAL
            params = BalanceAllowanceParams(asset_type=at)
            return self._client.get_balance_allowance(params)
        except Exception as e:
            logger.error(f"Get balance failed: {e}")
            return {}

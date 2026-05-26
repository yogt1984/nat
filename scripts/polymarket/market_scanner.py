"""
Polymarket Market Scanner

Discovers and filters BTC-related prediction markets from Polymarket.
Uses the Gamma API for market discovery, then enriches with CLOB data
(order book depth, last trade price, tick size).

Market types detected:
  - Price target: "Will BTC be above $X by date?"
  - Range: "Will BTC stay between $X and $Y?"
  - Event: "Will BTC reach all-time high?"
  - Directional: "Will BTC go up/down this week?"

Usage:
    from polymarket.market_scanner import MarketScanner
    scanner = MarketScanner()
    markets = scanner.scan_btc_markets()
    for m in markets:
        print(f"{m.question}: YES={m.yes_price:.2f}, liquidity=${m.liquidity:,.0f}")

CLI:
    cd scripts && python -m polymarket.market_scanner
    cd scripts && python -m polymarket.market_scanner --min-liquidity 1000 --json
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Keywords for finding crypto/BTC markets
BTC_KEYWORDS = [
    "bitcoin", "btc", "₿",
]

CRYPTO_KEYWORDS = BTC_KEYWORDS + [
    "ethereum", "eth", "solana", "sol", "crypto",
]


@dataclass
class MarketInfo:
    """Parsed Polymarket market with trading-relevant fields."""
    # Identity
    condition_id: str
    question: str
    description: str = ""
    slug: str = ""

    # Tokens
    token_id_yes: str = ""
    token_id_no: str = ""

    # Prices
    yes_price: float = 0.0    # Current YES token price (= implied probability)
    no_price: float = 0.0
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    spread: float = 0.0

    # Liquidity
    liquidity: float = 0.0    # Total liquidity in USD
    volume_24h: float = 0.0
    volume_total: float = 0.0

    # Timing
    end_date: str = ""
    end_timestamp: int = 0
    hours_to_settlement: float = 0.0

    # Parsed fields
    market_type: str = ""     # "price_target", "range", "directional", "event"
    strike_price: float = 0.0  # For price target markets
    strike_above: bool = True  # True = "above $X", False = "below $X"

    # Trading
    tick_size: str = "0.01"
    neg_risk: bool = False
    active: bool = True

    # Book depth (enriched from CLOB)
    book_depth_yes: float = 0.0  # Total size on book (both sides)
    book_depth_no: float = 0.0

    @property
    def implied_prob(self) -> float:
        return self.yes_price

    @property
    def is_btc(self) -> bool:
        q = self.question.lower()
        return any(kw in q for kw in BTC_KEYWORDS)


class MarketScanner:
    """Discover and filter Polymarket markets."""

    def __init__(self, gamma_url: str = GAMMA_API, clob_url: str = CLOB_API):
        self._gamma = gamma_url
        self._clob = clob_url
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "NAT-Scanner/1.0",
            "Accept": "application/json",
        })

    def _gamma_get(self, path: str, params: dict | None = None) -> list | dict:
        """Make a GET request to the Gamma API."""
        url = f"{self._gamma}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Gamma API error: {e}")
            return []

    def _clob_get(self, path: str, params: dict | None = None) -> dict | list:
        """Make a GET request to the CLOB API."""
        url = f"{self._clob}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"CLOB API error: {e}")
            return {}

    def fetch_gamma_markets(
        self,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
        tag: str | None = None,
    ) -> list[dict]:
        """
        Fetch markets from Gamma API.

        The Gamma API supports filtering by tag, active/closed status.
        Returns raw market dicts.
        """
        params = {"limit": limit, "active": str(active).lower()}
        if closed:
            params["closed"] = "true"
        if tag:
            params["tag"] = tag

        # Try tag-based search first for crypto markets
        all_markets = []

        # Fetch with different filters
        for search_tag in ["crypto", "bitcoin", None]:
            p = dict(params)
            if search_tag:
                p["tag"] = search_tag
            else:
                # Generic fetch — get a large batch
                p["limit"] = 500

            data = self._gamma_get("/markets", params=p)
            if isinstance(data, list):
                all_markets.extend(data)
            elif isinstance(data, dict) and "data" in data:
                all_markets.extend(data["data"])

            time.sleep(0.2)

        # Deduplicate by condition_id
        seen = set()
        unique = []
        for m in all_markets:
            cid = m.get("condition_id", m.get("conditionId", ""))
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(m)

        logger.info(f"Fetched {len(unique)} unique markets from Gamma API")
        return unique

    def fetch_clob_markets(self) -> list[dict]:
        """
        Fetch markets from the CLOB API (paginated).
        Returns raw market dicts with trading info.
        """
        markets = []
        cursor = "MA=="
        end_cursor = "LTE="
        max_pages = 50

        for _ in range(max_pages):
            data = self._clob_get("/markets", params={"next_cursor": cursor})
            if isinstance(data, dict):
                batch = data.get("data", [])
                cursor = data.get("next_cursor", end_cursor)
            elif isinstance(data, list):
                batch = data
                cursor = end_cursor
            else:
                break

            markets.extend(batch)
            if cursor == end_cursor:
                break
            time.sleep(0.1)

        logger.info(f"Fetched {len(markets)} markets from CLOB API")
        return markets

    def parse_market(self, raw: dict) -> MarketInfo:
        """Parse a raw Gamma/CLOB market dict into MarketInfo."""
        question = raw.get("question", raw.get("title", ""))
        description = raw.get("description", "")
        condition_id = raw.get("condition_id", raw.get("conditionId", ""))
        slug = raw.get("slug", raw.get("market_slug", ""))

        # Extract token IDs
        tokens = raw.get("tokens", [])
        token_yes = ""
        token_no = ""
        yes_price = 0.0
        no_price = 0.0

        if isinstance(tokens, list):
            for t in tokens:
                outcome = t.get("outcome", "").upper()
                tid = t.get("token_id", "")
                price = float(t.get("price", 0) or 0)
                if outcome == "YES":
                    token_yes = tid
                    yes_price = price
                elif outcome == "NO":
                    token_no = tid
                    no_price = price

        # Fallback prices from other fields
        if yes_price == 0:
            yes_price = float(raw.get("outcomePrices", [0, 0])[0] or 0) if isinstance(raw.get("outcomePrices"), list) else 0
        if no_price == 0 and yes_price > 0:
            no_price = 1 - yes_price

        # Liquidity and volume
        liquidity = float(raw.get("liquidity", 0) or 0)
        volume_24h = float(raw.get("volume24hr", raw.get("volume_24h", 0)) or 0)
        volume_total = float(raw.get("volume", raw.get("volumeNum", 0)) or 0)

        # Timing
        end_date = raw.get("end_date_iso", raw.get("endDate", ""))
        end_ts = 0
        hours_to_settle = 0.0
        if end_date:
            try:
                if isinstance(end_date, str):
                    dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    end_ts = int(dt.timestamp())
                    hours_to_settle = max(0, (dt - datetime.now(timezone.utc)).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass

        # Active status
        active = raw.get("active", True)
        if isinstance(active, str):
            active = active.lower() == "true"

        # Parse market type and strike
        market_type, strike, above = self._parse_question(question)

        # Neg risk
        neg_risk = raw.get("neg_risk", raw.get("negRisk", False))
        if isinstance(neg_risk, str):
            neg_risk = neg_risk.lower() == "true"

        return MarketInfo(
            condition_id=condition_id,
            question=question,
            description=description,
            slug=slug,
            token_id_yes=token_yes,
            token_id_no=token_no,
            yes_price=yes_price,
            no_price=no_price,
            liquidity=liquidity,
            volume_24h=volume_24h,
            volume_total=volume_total,
            end_date=end_date,
            end_timestamp=end_ts,
            hours_to_settlement=hours_to_settle,
            market_type=market_type,
            strike_price=strike,
            strike_above=above,
            active=active,
            neg_risk=neg_risk,
        )

    def _parse_question(self, question: str) -> tuple[str, float, bool]:
        """
        Parse a market question to extract type and strike.

        Returns (market_type, strike_price, is_above).
        """
        q = question.lower().strip()

        # Price target: "Will BTC be above $108,000 by Friday?"
        # Patterns: "above $X", "over $X", "reach $X", "hit $X", "exceed $X"
        above_patterns = [
            r"(?:above|over|reach|hit|exceed|surpass|top)\s*\$?([\d,]+\.?\d*)\s*k?",
            r"\$?([\d,]+\.?\d*)\s*k?\s*(?:or\s+)?(?:above|higher|more)",
        ]
        for pat in above_patterns:
            m = re.search(pat, q)
            if m:
                strike = float(m.group(1).replace(",", ""))
                # Handle "k" suffix: $120k → $120,000
                after = q[m.end()-1:m.end()+1].lower()
                if "k" in after and strike < 10000:
                    strike *= 1000
                return "price_target", strike, True

        # Below patterns: "below $X", "under $X", "drop below $X", "fall below $X"
        below_patterns = [
            r"(?:below|under|drop\s+(?:below|under)|fall\s+(?:below|under))\s*\$?([\d,]+\.?\d*)\s*k?",
            r"\$?([\d,]+\.?\d*)\s*k?\s*(?:or\s+)?(?:below|lower|less)",
        ]
        for pat in below_patterns:
            m = re.search(pat, q)
            if m:
                strike = float(m.group(1).replace(",", ""))
                after = q[m.end()-1:m.end()+1].lower()
                if "k" in after and strike < 10000:
                    strike *= 1000
                return "price_target", strike, False

        # Range: "between $X and $Y"
        range_pat = r"between\s*\$?([\d,]+\.?\d*)\s*and\s*\$?([\d,]+\.?\d*)"
        m = re.search(range_pat, q)
        if m:
            lo = float(m.group(1).replace(",", ""))
            hi = float(m.group(2).replace(",", ""))
            mid = (lo + hi) / 2
            return "range", mid, True

        # Dollar amount mentioned (generic price reference)
        dollar_pat = r"\$\s*([\d,]+\.?\d*)\s*k?"
        m = re.search(dollar_pat, q)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "k" in q[m.end():m.end()+2]:
                val *= 1000
            if val > 1000:  # Likely a BTC price
                return "price_target", val, True

        # Directional: "go up", "go down", "increase", "decrease"
        if any(w in q for w in ["go up", "increase", "rise", "higher", "bullish", "rally"]):
            return "directional", 0.0, True
        if any(w in q for w in ["go down", "decrease", "fall", "lower", "bearish", "crash", "drop"]):
            return "directional", 0.0, False

        # Event: ATH, specific events
        if any(w in q for w in ["all-time high", "ath", "new high", "record"]):
            return "event", 0.0, True

        return "unknown", 0.0, True

    def enrich_with_clob(self, market: MarketInfo) -> MarketInfo:
        """Add order book data from CLOB API."""
        if not market.token_id_yes:
            return market

        try:
            # Get tick size
            ts = self._clob_get("/tick-size", {"token_id": market.token_id_yes})
            if isinstance(ts, dict) and "minimum_tick_size" in ts:
                market.tick_size = ts["minimum_tick_size"]

            # Get order book for YES token
            book = self._clob_get("/book", {"token_id": market.token_id_yes})
            if isinstance(book, dict):
                bids = book.get("bids", [])
                asks = book.get("asks", [])

                if bids:
                    market.yes_bid = float(bids[0].get("price", 0))
                if asks:
                    market.yes_ask = float(asks[0].get("price", 0))
                if market.yes_bid > 0 and market.yes_ask > 0:
                    market.spread = market.yes_ask - market.yes_bid

                market.book_depth_yes = sum(float(o.get("size", 0)) for o in bids + asks)

            time.sleep(0.1)  # Rate limit

            # Get book for NO token
            if market.token_id_no:
                book_no = self._clob_get("/book", {"token_id": market.token_id_no})
                if isinstance(book_no, dict):
                    all_orders = book_no.get("bids", []) + book_no.get("asks", [])
                    market.book_depth_no = sum(float(o.get("size", 0)) for o in all_orders)
                time.sleep(0.1)

        except Exception as e:
            logger.warning(f"CLOB enrichment failed for {market.condition_id}: {e}")

        return market

    def scan_btc_markets(
        self,
        min_liquidity: float = 0,
        min_volume: float = 0,
        max_hours_to_settlement: float = 0,
        enrich: bool = True,
    ) -> list[MarketInfo]:
        """
        Find all active BTC prediction markets.

        Args:
            min_liquidity: Minimum liquidity in USD
            min_volume: Minimum 24h volume in USD
            max_hours_to_settlement: Only markets settling within N hours (0 = no filter)
            enrich: Whether to fetch CLOB order book data (slower but more complete)
        """
        raw_markets = self.fetch_gamma_markets()

        btc_markets = []
        for raw in raw_markets:
            market = self.parse_market(raw)

            # Filter: BTC only
            if not market.is_btc:
                continue

            # Filter: active
            if not market.active:
                continue

            # Filter: liquidity
            if min_liquidity > 0 and market.liquidity < min_liquidity:
                continue

            # Filter: volume
            if min_volume > 0 and market.volume_24h < min_volume:
                continue

            # Filter: time to settlement
            if max_hours_to_settlement > 0 and market.hours_to_settlement > max_hours_to_settlement:
                continue

            # Enrich with order book data
            if enrich and market.token_id_yes:
                market = self.enrich_with_clob(market)

            btc_markets.append(market)

        # Sort by liquidity descending
        btc_markets.sort(key=lambda m: m.liquidity, reverse=True)

        logger.info(f"Found {len(btc_markets)} BTC markets "
                    f"(from {len(raw_markets)} total)")
        return btc_markets

    def scan_all_crypto(
        self,
        min_liquidity: float = 100,
        enrich: bool = False,
    ) -> list[MarketInfo]:
        """Find all crypto-related prediction markets."""
        raw_markets = self.fetch_gamma_markets()

        crypto = []
        for raw in raw_markets:
            market = self.parse_market(raw)
            q = market.question.lower()
            if not any(kw in q for kw in CRYPTO_KEYWORDS):
                continue
            if not market.active:
                continue
            if min_liquidity > 0 and market.liquidity < min_liquidity:
                continue
            if enrich and market.token_id_yes:
                market = self.enrich_with_clob(market)
            crypto.append(market)

        crypto.sort(key=lambda m: m.liquidity, reverse=True)
        return crypto

    def print_markets(self, markets: list[MarketInfo]):
        """Pretty-print a list of markets."""
        if not markets:
            print("No markets found.")
            return

        print(f"\n{'='*90}")
        print(f"Found {len(markets)} markets")
        print(f"{'='*90}")

        for i, m in enumerate(markets, 1):
            settle_str = f"{m.hours_to_settlement:.1f}h" if m.hours_to_settlement > 0 else "?"
            liq_str = f"${m.liquidity:,.0f}" if m.liquidity > 0 else "n/a"
            vol_str = f"${m.volume_24h:,.0f}" if m.volume_24h > 0 else "n/a"

            print(f"\n  [{i}] {m.question}")
            print(f"      Type: {m.market_type}"
                  f"{f' | Strike: ${m.strike_price:,.0f}' if m.strike_price > 0 else ''}"
                  f" | {'Above' if m.strike_above else 'Below'}")
            print(f"      YES: {m.yes_price:.3f}"
                  f" | Bid: {m.yes_bid:.3f} | Ask: {m.yes_ask:.3f}"
                  f" | Spread: {m.spread:.4f}")
            print(f"      Liquidity: {liq_str} | 24h Vol: {vol_str}"
                  f" | Book depth: {m.book_depth_yes:.0f}")
            print(f"      Settles in: {settle_str}"
                  f" | Condition: {m.condition_id[:16]}..."
                  f" | Tick: {m.tick_size}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scan Polymarket for BTC markets")
    parser.add_argument("--min-liquidity", type=float, default=0,
                        help="Minimum liquidity in USD")
    parser.add_argument("--min-volume", type=float, default=0,
                        help="Minimum 24h volume in USD")
    parser.add_argument("--max-hours", type=float, default=0,
                        help="Max hours to settlement (0 = no filter)")
    parser.add_argument("--all-crypto", action="store_true",
                        help="Scan all crypto markets, not just BTC")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip CLOB enrichment (faster)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    scanner = MarketScanner()

    if args.all_crypto:
        markets = scanner.scan_all_crypto(
            min_liquidity=args.min_liquidity,
            enrich=not args.no_enrich,
        )
    else:
        markets = scanner.scan_btc_markets(
            min_liquidity=args.min_liquidity,
            min_volume=args.min_volume,
            max_hours_to_settlement=args.max_hours,
            enrich=not args.no_enrich,
        )

    if args.json:
        output = [asdict(m) for m in markets]
        print(json.dumps(output, indent=2))
    else:
        scanner.print_markets(markets)
        if markets:
            # Summary stats
            total_liq = sum(m.liquidity for m in markets)
            total_vol = sum(m.volume_24h for m in markets)
            n_price = sum(1 for m in markets if m.market_type == "price_target")
            n_dir = sum(1 for m in markets if m.market_type == "directional")
            n_event = sum(1 for m in markets if m.market_type == "event")
            print(f"\n{'='*90}")
            print(f"Summary:")
            print(f"  Total markets: {len(markets)}")
            print(f"  Total liquidity: ${total_liq:,.0f}")
            print(f"  Total 24h volume: ${total_vol:,.0f}")
            print(f"  Price target: {n_price} | Directional: {n_dir} | Event: {n_event}")


if __name__ == "__main__":
    main()

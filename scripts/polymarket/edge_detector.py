"""
Polymarket Edge Detector

Compares model probability vs Polymarket market price.
When divergence exceeds threshold (adjusted for fees), generates trade signals.

Polymarket fee structure (crypto category):
  taker_fee = 1.8% × p × (1 - p) × shares
  maker_fee = 0%

At p=0.50: max fee = 0.45% per share
At p=0.90: fee = 0.162% per share
At p=0.95: fee = 0.0855% per share
At p=0.99: fee = 0.0178% per share

Key insight: near-certain outcomes have near-zero fees.
When your model says P=0.92 but market says P=0.85, the edge is
7 cents per share with <0.2% fee — a 35:1 edge-to-fee ratio.

Usage:
    from polymarket.edge_detector import EdgeDetector
    detector = EdgeDetector(min_edge_cents=3.0)
    signals = detector.scan(model, markets, current_features)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


CRYPTO_FEE_RATE = 0.018  # 1.8% for crypto category


@dataclasses.dataclass
class TradeSignal:
    """A detected edge opportunity on Polymarket."""
    market_id: str
    market_question: str
    token_id: str
    side: str          # "BUY_YES" or "BUY_NO"
    model_prob: float  # Our model's probability
    market_prob: float # Current market price
    edge_cents: float  # Edge per $1 share (cents)
    fee_cents: float   # Expected fee per share (cents)
    net_edge_cents: float  # Edge after fees
    kelly_fraction: float  # Optimal fraction to bet (Kelly criterion)
    confidence: float  # Model confidence (0-1)
    horizon_minutes: float

    @property
    def edge_to_fee_ratio(self) -> float:
        return self.edge_cents / (self.fee_cents + 1e-6)


def polymarket_fee(price: float, shares: float = 1.0) -> float:
    """
    Calculate Polymarket taker fee for crypto markets.
    fee = shares × fee_rate × p × (1 - p)
    """
    return shares * CRYPTO_FEE_RATE * price * (1 - price)


def kelly_criterion(model_prob: float, market_price: float) -> float:
    """
    Kelly fraction for binary bet.
    f* = (p × b - q) / b
    where p = model prob, q = 1-p, b = odds = (1-market_price)/market_price

    If we BUY YES at price `market_price`:
      Win: pay market_price, receive 1.0 → profit = 1 - market_price
      Lose: pay market_price, receive 0 → loss = market_price
      b = (1 - market_price) / market_price
    """
    p = model_prob
    q = 1 - p
    if market_price <= 0 or market_price >= 1:
        return 0.0

    b = (1 - market_price) / market_price
    f = (p * b - q) / b
    return max(0.0, min(f, 0.25))  # Cap at 25% of bankroll


class EdgeDetector:
    """Scan Polymarket markets for mispricings vs model."""

    def __init__(
        self,
        min_edge_cents: float = 2.0,
        min_edge_to_fee: float = 3.0,
        min_confidence: float = 0.05,
        max_kelly: float = 0.15,
    ):
        self._min_edge = min_edge_cents
        self._min_ratio = min_edge_to_fee
        self._min_confidence = min_confidence
        self._max_kelly = max_kelly

    def evaluate_market(
        self,
        model_prob: float,
        market_price: float,
        market_id: str = "",
        market_question: str = "",
        token_id_yes: str = "",
        token_id_no: str = "",
        confidence: float = 0.0,
        horizon_minutes: float = 5.0,
    ) -> Optional[TradeSignal]:
        """
        Check if model disagrees with market enough to trade.

        If model_prob > market_price: BUY YES (model thinks YES is more likely)
        If model_prob < market_price: BUY NO (model thinks NO is more likely)
        """
        edge = model_prob - market_price  # Positive = YES is underpriced

        if abs(edge) * 100 < self._min_edge:
            return None

        if confidence < self._min_confidence:
            return None

        if edge > 0:
            # BUY YES at market_price
            side = "BUY_YES"
            token_id = token_id_yes
            entry_price = market_price
            fee = polymarket_fee(market_price) * 100  # In cents
            edge_cents = edge * 100
        else:
            # BUY NO at (1 - market_price)
            side = "BUY_NO"
            token_id = token_id_no
            entry_price = 1 - market_price
            fee = polymarket_fee(1 - market_price) * 100
            edge_cents = abs(edge) * 100

        net_edge = edge_cents - fee

        if net_edge < self._min_edge * 0.5:
            return None

        if edge_cents / (fee + 1e-6) < self._min_ratio:
            return None

        kelly = kelly_criterion(
            model_prob if side == "BUY_YES" else (1 - model_prob),
            entry_price,
        )

        if kelly > self._max_kelly:
            kelly = self._max_kelly

        return TradeSignal(
            market_id=market_id,
            market_question=market_question,
            token_id=token_id,
            side=side,
            model_prob=model_prob,
            market_prob=market_price,
            edge_cents=edge_cents,
            fee_cents=fee,
            net_edge_cents=net_edge,
            kelly_fraction=kelly,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
        )

    def scan_btc_markets(
        self,
        model,
        markets: list[dict],
        current_features: dict[str, float],
        current_btc_price: float,
    ) -> list[TradeSignal]:
        """
        Scan a list of Polymarket BTC markets for opportunities.

        Each market dict has:
          - market_id, question, token_id_yes, token_id_no
          - strike: the BTC price threshold
          - horizon_minutes: time to settlement
          - market_price: current YES token price (= implied probability)
        """
        signals = []

        for mkt in markets:
            strike = mkt.get("strike")
            horizon = mkt.get("horizon_minutes", 5)
            market_price = mkt.get("market_price")

            if not all([strike, market_price]):
                continue

            prob, confidence = model.predict_with_confidence(
                current_features, current_btc_price, strike, horizon
            )

            signal = self.evaluate_market(
                model_prob=prob,
                market_price=market_price,
                market_id=mkt.get("market_id", ""),
                market_question=mkt.get("question", ""),
                token_id_yes=mkt.get("token_id_yes", ""),
                token_id_no=mkt.get("token_id_no", ""),
                confidence=confidence,
                horizon_minutes=horizon,
            )

            if signal:
                signals.append(signal)

        # Sort by net edge descending
        signals.sort(key=lambda s: s.net_edge_cents, reverse=True)
        return signals

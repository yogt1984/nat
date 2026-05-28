"""
Cost Model for NAT Backtester

Realistic transaction cost modeling for Hyperliquid perpetuals.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CostModel:
    """
    Transaction cost model.

    Attributes
    ----------
    fee_bps : float
        Trading fee in basis points (1 bp = 0.01%)
        Hyperliquid: ~2.5 bps maker, ~5 bps taker
    slippage_bps : float
        Expected slippage in basis points
        Depends on order size and liquidity
    funding_enabled : bool
        Whether to apply funding rate costs
        (requires funding_rate column in data)

    Notes
    -----
    Hyperliquid fee structure (as of 2026):
    - Maker: 0.02% (2 bps) - adds liquidity
    - Taker: 0.05% (5 bps) - removes liquidity

    For backtesting, assume taker fees (conservative).
    """

    fee_bps: float = 5.0  # Taker fee
    slippage_bps: float = 2.0  # Conservative slippage estimate
    funding_enabled: bool = False

    def __post_init__(self):
        if self.fee_bps < 0:
            raise ValueError(f"fee_bps must be non-negative, got {self.fee_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be non-negative, got {self.slippage_bps}")

    @property
    def one_way_cost_bps(self) -> float:
        """Total cost for a single trade (entry OR exit)."""
        return self.fee_bps + self.slippage_bps

    @property
    def one_way_cost_fraction(self) -> float:
        """One-way cost as a decimal fraction."""
        return self.one_way_cost_bps / 10000

    @property
    def round_trip_cost_bps(self) -> float:
        """Total cost for entry + exit."""
        return 2 * self.one_way_cost_bps

    @property
    def round_trip_cost_fraction(self) -> float:
        """Round-trip cost as a decimal fraction."""
        return self.round_trip_cost_bps / 10000

    def apply_entry_cost(
        self,
        price: float,
        direction: Literal["long", "short"],
    ) -> float:
        """
        Adjust entry price for transaction costs.

        For long: you pay MORE than the quoted price
        For short: you receive LESS than the quoted price

        Parameters
        ----------
        price : float
            Quoted market price
        direction : str
            "long" or "short"

        Returns
        -------
        float
            Effective entry price after costs
        """
        cost_mult = self.one_way_cost_fraction

        if direction == "long":
            # Buying: pay more
            return price * (1 + cost_mult)
        else:
            # Selling short: effective entry is lower
            return price * (1 - cost_mult)

    def apply_exit_cost(
        self,
        price: float,
        direction: Literal["long", "short"],
    ) -> float:
        """
        Adjust exit price for transaction costs.

        For long exit (sell): you receive LESS than the quoted price
        For short exit (buy to cover): you pay MORE than the quoted price

        Parameters
        ----------
        price : float
            Quoted market price
        direction : str
            "long" or "short"

        Returns
        -------
        float
            Effective exit price after costs
        """
        cost_mult = self.one_way_cost_fraction

        if direction == "long":
            # Selling to close long: receive less
            return price * (1 - cost_mult)
        else:
            # Buying to close short: pay more
            return price * (1 + cost_mult)

    def compute_funding_cost(
        self,
        holding_hours: float,
        funding_rate_bps: float,
        funding_interval_hours: float = 8.0,
    ) -> float:
        """
        Compute accumulated funding cost for a position.

        Parameters
        ----------
        holding_hours : float
            Duration of the position in hours
        funding_rate_bps : float
            Average funding rate in basis points (per interval)
        funding_interval_hours : float
            Funding settlement interval (default 8h for Hyperliquid)

        Returns
        -------
        float
            Funding cost as percentage. Positive = cost for longs when
            funding is positive (longs pay shorts).
        """
        if not self.funding_enabled or funding_interval_hours <= 0:
            return 0.0
        n_settlements = holding_hours / funding_interval_hours
        return funding_rate_bps / 100 * n_settlements

    def compute_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: Literal["long", "short"],
        include_costs: bool = True,
        funding_cost_pct: float = 0.0,
    ) -> float:
        """
        Compute P&L percentage for a trade.

        Parameters
        ----------
        entry_price : float
            Raw entry price (before costs)
        exit_price : float
            Raw exit price (before costs)
        direction : str
            "long" or "short"
        include_costs : bool
            Whether to include transaction costs
        funding_cost_pct : float
            Funding cost as percentage (from compute_funding_cost).
            Positive = cost for longs, income for shorts.

        Returns
        -------
        float
            P&L as percentage (e.g., 2.5 means +2.5%)
        """
        if include_costs:
            eff_entry = self.apply_entry_cost(entry_price, direction)
            eff_exit = self.apply_exit_cost(exit_price, direction)
        else:
            eff_entry = entry_price
            eff_exit = exit_price

        if direction == "long":
            pnl = (eff_exit / eff_entry - 1) * 100
            pnl -= funding_cost_pct  # longs pay positive funding
        else:
            pnl = (1 - eff_exit / eff_entry) * 100
            pnl += funding_cost_pct  # shorts receive positive funding
        return pnl

    def breakeven_move_pct(self) -> float:
        """
        Minimum price move needed to break even after costs.

        Returns
        -------
        float
            Required move as percentage
        """
        return self.round_trip_cost_bps / 100

    def __repr__(self) -> str:
        return (
            f"CostModel(fee={self.fee_bps}bps, "
            f"slippage={self.slippage_bps}bps, "
            f"round_trip={self.round_trip_cost_bps}bps)"
        )


# =============================================================================
# PRESET COST MODELS
# =============================================================================


def hyperliquid_taker() -> CostModel:
    """Hyperliquid taker fees with conservative slippage."""
    return CostModel(fee_bps=3.5, slippage_bps=2.0)


def hyperliquid_maker() -> CostModel:
    """Hyperliquid maker fees (optimistic - assumes limit orders fill)."""
    return CostModel(fee_bps=0.2, slippage_bps=0.5)


def conservative() -> CostModel:
    """Conservative cost model for skeptical backtesting."""
    return CostModel(fee_bps=7.5, slippage_bps=5.0)


def zero_cost() -> CostModel:
    """Zero cost model - only for debugging, not real backtests."""
    return CostModel(fee_bps=0.0, slippage_bps=0.0)

"""
Cost Model for NAT Backtester

Realistic transaction cost modeling for Hyperliquid perpetuals. Fees come from the
single source of truth (config/costs.toml via utils.costs) — never hardcode them here.
"""

from dataclasses import dataclass, field
from typing import Literal

#: A maker REBATE larger than this is a typo, not a fee tier (the venue's best rung is
#: 0.3 bps). Keeps the non-negative guard's intent while allowing a real rebate through.
MAX_REBATE_BPS = 5.0

try:
    from utils.costs import funding_interval_hours, maker_bps, taker_bps
except ImportError:  # pragma: no cover - path bootstrap for direct/script imports
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.costs import funding_interval_hours, maker_bps, taker_bps


@dataclass
class CostModel:
    """
    Transaction cost model.

    Attributes
    ----------
    fee_bps : float
        Trading fee in basis points (1 bp = 0.01%). Defaults to the config taker fee
        (config/costs.toml `hyperliquid.taker_bps`) — the single source of truth.
    slippage_bps : float
        Expected slippage in basis points. Depends on order size and liquidity.
    funding_enabled : bool
        Whether to apply funding accrual (requires funding_rate column in data).
        DEFAULTS TO TRUE (COST-9): funding is a forced payment at every settlement
        while a position is held — omitting it mis-states every held-position
        result, with the sign of the error set by which side was held. Opt out
        explicitly (or via the "zero" preset), never by accident.
    funding_interval_hours : float
        Hours between funding settlements. Sourced from config/costs.toml via
        utils.costs (1 h on Hyperliquid, venue-verified) — never hardcoded.

    Notes
    -----
    Hyperliquid fees live in config/costs.toml (taker 3.5 bps, maker 0.2 bps rebate as
    of 2026). This model defaults to the taker fee — always sourced from that file, so a
    fee change ripples everywhere. Slippage is a modeling assumption (no config key yet).
    """

    fee_bps: float = field(default_factory=taker_bps)  # config taker fee (SSOT)
    slippage_bps: float = 2.0  # Conservative slippage estimate
    funding_enabled: bool = True
    funding_interval_hours: float = field(default_factory=funding_interval_hours)
    fill_probability: float = 1.0  # Probability an order fills (1.0 = always)

    def __post_init__(self):
        # fee_bps is a COST, so a NEGATIVE value is a rebate (the venue pays you). That is
        # a real tier, not a typo — but only within the range a venue actually offers, so
        # the guard is relaxed to a floor rather than removed (COST-8).
        if self.fee_bps < -MAX_REBATE_BPS:
            raise ValueError(
                f"fee_bps {self.fee_bps} implies a rebate larger than {MAX_REBATE_BPS} bps "
                "— the venue's best rung is 0.3 bps, so this is a typo")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be non-negative, got {self.slippage_bps}")
        if not 0.0 < self.fill_probability <= 1.0:
            raise ValueError(f"fill_probability must be in (0, 1], got {self.fill_probability}")
        if self.funding_interval_hours <= 0:
            raise ValueError(
                f"funding_interval_hours must be positive, got {self.funding_interval_hours}")

    @classmethod
    def from_config(cls, path: str | None = None,
                    venue: str = "hyperliquid",
                    role: str = "taker") -> "CostModel":
        """Build from the shared cost SSOT (config/costs.toml via utils.costs).

        Parameters
        ----------
        path : accepted for backwards-compat but ignored — utils.costs owns resolution.
        venue : "hyperliquid" or "binance"
        role : "taker" or "maker"
        """
        from utils.costs import load_costs, maker_bps, taker_bps

        if venue == "hyperliquid":
            # utils.costs uses POSITIVE = rebate earned; CostModel uses POSITIVE = cost.
            # The negation is the whole point — booking a rebate as a charge is a 2x error
            # on the maker leg (COST-8).
            fee = -maker_bps() if role == "maker" else taker_bps()
        else:
            section = load_costs().get(venue, {})
            fee = section.get(f"{role}_bps", taker_bps())
        return cls(fee_bps=fee)

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
        funding_interval_hours: float | None = None,
    ) -> float:
        """
        Compute accumulated funding cost for a position.

        Parameters
        ----------
        holding_hours : float
            Duration of the position in hours
        funding_rate_bps : float
            Average funding rate in basis points (per settlement interval)
        funding_interval_hours : float, optional
            Funding settlement interval. Defaults to the model's SSOT-sourced
            value (1 h on Hyperliquid, venue-verified — COST-9). The old
            hardcoded 8.0 understated accrual 8x.

        Returns
        -------
        float
            Funding cost as percentage. Positive = cost for longs when
            funding is positive (longs pay shorts).
        """
        if funding_interval_hours is None:
            funding_interval_hours = self.funding_interval_hours
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
        parts = [
            f"fee={self.fee_bps}bps",
            f"slippage={self.slippage_bps}bps",
            f"round_trip={self.round_trip_cost_bps}bps",
        ]
        if self.fill_probability < 1.0:
            parts.append(f"fill={self.fill_probability:.0%}")
        return f"CostModel({', '.join(parts)})"


# =============================================================================
# PRESET COST MODELS
# =============================================================================


def hyperliquid_taker() -> CostModel:
    """Hyperliquid taker fees with conservative slippage."""
    return CostModel.from_config(role="taker")


def _maker_fee_bps() -> float:
    """Active maker rate as a COST (negative = rebate earned), from the SSOT ladder.

    Sourced from `utils.costs.maker_bps()` so the `[hyperliquid_maker_tiers]` rung in
    `config/costs.toml` ripples here instead of requiring an edit — the rate §4.11 names
    the most load-bearing unvalidated assumption in the stack must never be a literal.
    """
    from utils.costs import maker_bps as _maker_bps
    return -_maker_bps()


def hyperliquid_maker() -> CostModel:
    """Hyperliquid maker rate (SSOT tier) with a realistic fill rate (~40%)."""
    return CostModel(fee_bps=_maker_fee_bps(), slippage_bps=0.5, fill_probability=0.4)


def hyperliquid_maker_conservative() -> CostModel:
    """Same rate, conservative fill rate (~30%)."""
    return CostModel(fee_bps=_maker_fee_bps(), slippage_bps=0.5, fill_probability=0.3)


def conservative() -> CostModel:
    """Conservative cost model for skeptical backtesting."""
    return CostModel(fee_bps=7.5, slippage_bps=5.0)


def zero_cost() -> CostModel:
    """Zero cost model - only for debugging, not real backtests.

    Funding is off too: "zero" means genuinely cost-free, not fee-free-but-funded.
    """
    return CostModel(fee_bps=0.0, slippage_bps=0.0, funding_enabled=False)


# Named presets selectable via a `--cost-model` flag.
_COST_PRESETS = {
    "taker": hyperliquid_taker,
    "maker": hyperliquid_maker,
    "conservative": conservative,
    "zero": zero_cost,  # explicit opt-in only — never an accidental fallback
}


def cost_model_from_name(name: str) -> CostModel:
    """Resolve a ``--cost-model`` preset name to a CostModel.

    Unknown names raise ValueError (COST-2): a mistyped or empty preset must fail loudly,
    never silently produce a cost-free backtest. Zero cost requires the explicit "zero".
    """
    try:
        return _COST_PRESETS[name]()
    except KeyError:
        raise ValueError(
            f"unknown --cost-model {name!r}; choose one of {sorted(_COST_PRESETS)}"
        ) from None

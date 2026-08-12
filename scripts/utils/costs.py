"""Load trading costs from config/costs.toml (single source of truth).

Fee tiers (X-1) are SSOT state, never an assumption. The HYPE-staking discount ladder
lives in ``[hyperliquid_staked]``; the *active* tier is ``config.tier``, overridable
per-experiment with the ``NAT_FEE_TIER`` env var (``fee_tier_override`` is the
programmatic form). Two rules are structural, not stylistic:

  * the discount multiplies FEES PAID — a maker *rebate* is income and is untouched
    (venue-documented: staking discounts do not apply to maker rebates);
  * an unknown tier raises. A silent fallback to "no discount" would be benign, but
    a silent fallback in the other direction is how the VIP9 tier produced five false
    winners (FINDINGS §4.6), so tier resolution fails loudly in both directions.

Every experiment that prices a tier must stamp ``tier_summary()`` into its artifact.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    import nat_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nat_paths

_COSTS_PATH = nat_paths.config_dir() / "costs.toml"


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_costs() -> dict:
    """Return the full costs.toml dict. Falls back to defaults if file missing."""
    if not _COSTS_PATH.exists():
        return {
            "hyperliquid": {"taker_bps": 3.5, "maker_bps": 0.2, "round_trip_taker_bps": 7.0},
            "binance": {"vip9_round_trip_bps": 1.61},
        }
    return _load_toml(_COSTS_PATH)


# ── HYPE-staking fee tier (X-1) ──────────────────────────────────────────────────
#: Fallback ladder, used only when config/costs.toml is missing entirely.
_DEFAULT_DISCOUNTS = {"none": 0.0, "wood": 0.05, "bronze": 0.10, "silver": 0.15,
                      "gold": 0.20, "platinum": 0.30, "diamond": 0.40}


def _staked_section() -> dict:
    return load_costs().get("hyperliquid_staked", {})


def fee_tier() -> str:
    """The active HYPE-staking tier: ``NAT_FEE_TIER`` env, else config, else "none"."""
    return os.environ.get("NAT_FEE_TIER") or _staked_section().get("tier", "none")


def rebate_discount_applies() -> bool:
    """Whether the discount also shrinks the maker rebate (pessimistic sensitivity).

    FALSE per the venue docs — staking discounts do not apply to maker rebates.
    """
    env = os.environ.get("NAT_FEE_REBATE_DISCOUNT")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(_staked_section().get("rebate_discount_applies", False))


def staking_discount(tier: str | None = None) -> float:
    """Fractional fee discount for ``tier`` (default: the active tier). Raises on unknown."""
    name = (tier if tier is not None else fee_tier()).strip().lower()
    table = _staked_section().get("discounts") or _DEFAULT_DISCOUNTS
    if name not in table:
        raise ValueError(
            f"unknown fee tier {name!r}; choose one of {sorted(table)} "
            "(config/costs.toml [hyperliquid_staked.discounts])"
        )
    return float(table[name])


def apply_fee_discount(fee_bps: float, tier: str | None = None) -> float:
    """Apply the staking discount to a fee. Negative fees (rebates) pass through."""
    if fee_bps <= 0:
        return float(fee_bps)
    return float(fee_bps) * (1.0 - staking_discount(tier))


@contextmanager
def fee_tier_override(tier: str, rebate_discount: bool | None = None):
    """Price a block of code at ``tier`` (test/experiment scope only, never a default)."""
    saved = {k: os.environ.get(k) for k in ("NAT_FEE_TIER", "NAT_FEE_REBATE_DISCOUNT")}
    os.environ["NAT_FEE_TIER"] = tier
    if rebate_discount is not None:
        os.environ["NAT_FEE_REBATE_DISCOUNT"] = "1" if rebate_discount else "0"
    try:
        staking_discount(tier)          # fail fast on a bad tier name
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def tier_summary() -> dict:
    """Self-describing cost stamp — every experiment artifact must carry this."""
    return {
        "tier": fee_tier(),
        "discount": staking_discount(),
        "rebate_discount_applies": rebate_discount_applies(),
        "taker_bps": taker_bps(),
        "maker_bps": maker_bps(),
        "round_trip_taker_bps": round_trip_taker_bps(),
        "slippage_bps": slippage_bps(),
        "realistic_taker_rt_bps": realistic_taker_rt_bps(),
        "funding_interval_hours": funding_interval_hours(),
    }


def taker_bps() -> float:
    """Hyperliquid one-way taker fee in bps, net of the active staking discount."""
    base = load_costs().get("hyperliquid", {}).get("taker_bps", 3.5)
    return apply_fee_discount(base)


def maker_tier() -> str:
    """Active perp maker tier: ``NAT_MAKER_TIER`` env, else config, else the SSOT value."""
    return (os.environ.get("NAT_MAKER_TIER")
            or load_costs().get("hyperliquid_maker_tiers", {}).get("tier", "rebate_t2"))


def maker_tier_bps(tier: str | None = None) -> float:
    """Maker rate for ``tier`` in bps, POSITIVE = rebate earned, NEGATIVE = fee paid.

    Falls back to ``hyperliquid.maker_bps`` when the ladder is absent (old configs).
    """
    table = load_costs().get("hyperliquid_maker_tiers", {}).get("rates_bps")
    if not table:
        return float(load_costs().get("hyperliquid", {}).get("maker_bps", 0.2))
    name = (tier if tier is not None else maker_tier()).strip()
    if name not in table:
        raise ValueError(
            f"unknown maker tier {name!r}; choose one of {sorted(table)} "
            "(config/costs.toml [hyperliquid_maker_tiers.rates_bps])"
        )
    return float(table[name])


@contextmanager
def maker_tier_override(tier: str):
    """Price a block of code at a given maker tier (experiment scope only)."""
    saved = os.environ.get("NAT_MAKER_TIER")
    os.environ["NAT_MAKER_TIER"] = tier
    try:
        maker_tier_bps(tier)          # fail fast on a bad tier name
        yield
    finally:
        if saved is None:
            os.environ.pop("NAT_MAKER_TIER", None)
        else:
            os.environ["NAT_MAKER_TIER"] = saved


def maker_bps() -> float:
    """Hyperliquid one-way maker rate in bps, at the active maker tier.

    POSITIVE = rebate earned, NEGATIVE = fee paid. The default tier reproduces the
    historical SSOT value (+0.2 bps) exactly, so no pre-COST-5 number moves.

    The staking discount does not touch a rebate unless ``rebate_discount_applies`` is
    explicitly on; it DOES discount a positive maker fee (a negative rate here), which
    is the venue-documented behaviour.
    """
    rate = maker_tier_bps()
    if rate < 0:                                   # a fee: the staking discount applies
        return -apply_fee_discount(-rate)
    if rebate_discount_applies():                  # explicit pessimistic sensitivity
        return rate * (1.0 - staking_discount())
    return rate


def round_trip_taker_bps() -> float:
    """Hyperliquid round-trip taker fee in bps, net of the active staking discount."""
    base = load_costs().get("hyperliquid", {}).get("round_trip_taker_bps", 7.0)
    return apply_fee_discount(base)


def slippage_bps() -> float:
    """Hyperliquid one-way slippage assumption in bps."""
    return load_costs().get("hyperliquid", {}).get("slippage_bps", 2.0)


def funding_interval_hours() -> float:
    """Hours between perp funding settlements on Hyperliquid.

    1 h, a venue fact verified live 2026-08-12 (`fundingHistory` spacing;
    `assetCtx.funding` is the hourly rate) — see config/costs.toml (COST-9).
    """
    return float(load_costs().get("hyperliquid", {}).get("funding_interval_hours", 1.0))


def realistic_taker_rt_bps() -> float:
    """The honest all-in round-trip taker cost on the venue NAT trades:
    Hyperliquid RT fee + slippage both ways (~11 bps). THE default for every
    evaluation harness — the VIP9 tier below is explicit-opt-in only
    (Q4 kill gate, FINDINGS §4.6: defaulting to it created 5/5 false winners)."""
    return round_trip_taker_bps() + 2.0 * slippage_bps()


def binance_vip9_rt_bps() -> float:
    """Binance VIP9 round-trip fee in bps. EXPLICIT OPT-IN comparison tier only —
    never a harness default (wrong venue; see realistic_taker_rt_bps)."""
    return load_costs().get("binance", {}).get("vip9_round_trip_bps", 1.61)

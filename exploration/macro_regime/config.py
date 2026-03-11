"""
Macro Regime Detection Configuration

This module defines the indicators, thresholds, and scoring weights
for macro-level regime detection (business cycle / liquidity cycle).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json


class MacroRegime(Enum):
    """Macro regime classifications"""
    EXPANSION = "expansion"           # Risk-on, full exposure
    LATE_CYCLE = "late_cycle"         # Cautious, reduce leverage
    CONTRACTION = "contraction"       # Risk-off, defensive
    EARLY_RECOVERY = "early_recovery" # Accumulation phase
    UNCERTAIN = "uncertain"           # Mixed signals, reduce size


class CryptoRegime(Enum):
    """Crypto-specific regime overlay"""
    BULL_EARLY = "bull_early"         # Accumulation, BTC dominance rising
    BULL_MID = "bull_mid"             # Altcoin rotation beginning
    BULL_LATE = "bull_late"           # Euphoria, OTHERS outperforming
    BEAR_EARLY = "bear_early"         # Distribution, smart money exiting
    BEAR_MID = "bear_mid"             # Capitulation
    BEAR_LATE = "bear_late"           # Accumulation by long-term holders


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator"""
    name: str
    description: str

    # Thresholds for regime scoring
    expansion_threshold: float      # Above this = expansion signal
    contraction_threshold: float    # Below this = contraction signal

    # Direction: 1 = higher is expansion, -1 = higher is contraction
    direction: int = 1

    # Weight in composite score (0-1)
    weight: float = 1.0

    # Data source
    source: str = "manual"  # manual, fred, cryptoquant, custom
    ticker: Optional[str] = None

    # Transformation
    use_yoy_change: bool = False    # Use year-over-year change
    use_mom_change: bool = False    # Use month-over-month change
    use_zscore: bool = False        # Normalize to z-score
    lookback_periods: int = 12      # For z-score calculation


@dataclass
class MacroRegimeConfig:
    """Full configuration for macro regime detection"""

    # ==========================================================================
    # BUSINESS CYCLE INDICATORS
    # ==========================================================================

    business_cycle_indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {

        # Manufacturing PMI - Leading indicator
        "ism_pmi": IndicatorConfig(
            name="ISM Manufacturing PMI",
            description="Institute for Supply Management Manufacturing Index. >50 = expansion, <50 = contraction",
            expansion_threshold=52.0,      # Solidly expanding
            contraction_threshold=48.0,    # Solidly contracting
            direction=1,
            weight=1.5,                    # High weight - leading indicator
            source="fred",
            ticker="MANEMP",               # Or use ISM directly
        ),

        # Services PMI
        "ism_services": IndicatorConfig(
            name="ISM Services PMI",
            description="Services sector activity. Larger portion of US GDP than manufacturing",
            expansion_threshold=52.0,
            contraction_threshold=48.0,
            direction=1,
            weight=1.2,
            source="fred",
            ticker="NMFCI",
        ),

        # Chicago PMI - Regional but leading
        "chicago_pmi": IndicatorConfig(
            name="Chicago PMI",
            description="Regional manufacturing. Often leads national ISM",
            expansion_threshold=52.0,
            contraction_threshold=48.0,
            direction=1,
            weight=0.8,
            source="manual",
        ),

        # New Orders - Inventories Spread (Leading)
        "new_orders_inventories": IndicatorConfig(
            name="ISM New Orders - Inventories",
            description="Spread between new orders and inventories. Positive = restocking cycle",
            expansion_threshold=5.0,       # Strong restocking
            contraction_threshold=-5.0,    # Destocking
            direction=1,
            weight=1.3,                    # Very leading
            source="manual",
        ),
    })

    # ==========================================================================
    # LIQUIDITY & FINANCIAL CONDITIONS
    # ==========================================================================

    liquidity_indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {

        # DXY - Dollar strength (inverse)
        "dxy": IndicatorConfig(
            name="US Dollar Index",
            description="Dollar strength. Weaker dollar = more liquidity, risk-on",
            expansion_threshold=100.0,     # Below 100 = weak dollar = good
            contraction_threshold=105.0,   # Above 105 = strong dollar = bad
            direction=-1,                  # INVERSE: lower is better
            weight=1.2,
            source="manual",
        ),

        # US 10Y Yield
        "us10y": IndicatorConfig(
            name="US 10-Year Treasury Yield",
            description="Long-term rates. Context-dependent interpretation",
            expansion_threshold=4.0,       # Below 4% generally supportive
            contraction_threshold=5.0,     # Above 5% restrictive
            direction=-1,                  # Lower is generally better for risk
            weight=1.0,
            source="fred",
            ticker="DGS10",
        ),

        # 10Y-2Y Spread (Yield Curve)
        "yield_curve_10y2y": IndicatorConfig(
            name="10Y-2Y Treasury Spread",
            description="Yield curve slope. Inversion predicts recession",
            expansion_threshold=0.5,       # Steep curve = healthy
            contraction_threshold=-0.2,    # Inverted = warning
            direction=1,
            weight=1.0,
            source="fred",
            ticker="T10Y2Y",
        ),

        # Credit Spreads (HY - IG or HY - Treasury)
        "credit_spreads": IndicatorConfig(
            name="High Yield Credit Spread",
            description="HY OAS spread. Wider = stress, tighter = risk-on",
            expansion_threshold=350,       # Below 350bps = healthy
            contraction_threshold=500,     # Above 500bps = stress
            direction=-1,                  # Lower is better
            weight=1.3,
            source="fred",
            ticker="BAMLH0A0HYM2",
        ),
    })

    # ==========================================================================
    # REAL ECONOMY INDICATORS
    # ==========================================================================

    real_economy_indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {

        # Copper/Gold Ratio - Risk appetite proxy
        "copper_gold": IndicatorConfig(
            name="Copper/Gold Ratio",
            description="Industrial vs safe haven. Rising = growth optimism",
            expansion_threshold=0.0,       # Use z-score: >0 = expansion
            contraction_threshold=0.0,     # <0 = contraction
            direction=1,
            weight=1.1,
            use_zscore=True,
            lookback_periods=52,           # 1 year weekly
            source="manual",
        ),

        # Initial Jobless Claims (inverse)
        "jobless_claims": IndicatorConfig(
            name="Initial Jobless Claims",
            description="Labor market health. Lower = stronger economy",
            expansion_threshold=220000,    # Below 220k = strong
            contraction_threshold=280000,  # Above 280k = weakening
            direction=-1,                  # Lower is better
            weight=1.0,
            source="fred",
            ticker="ICSA",
        ),

        # Retail Sales MoM
        "retail_sales": IndicatorConfig(
            name="Retail Sales MoM%",
            description="Consumer spending momentum",
            expansion_threshold=0.3,       # >0.3% MoM = strong
            contraction_threshold=-0.2,    # <-0.2% = weak
            direction=1,
            weight=0.8,
            use_mom_change=True,
            source="fred",
            ticker="RSXFS",
        ),
    })

    # ==========================================================================
    # CRYPTO-SPECIFIC INDICATORS
    # ==========================================================================

    crypto_indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {

        # ETH/BTC - Risk appetite within crypto
        "eth_btc": IndicatorConfig(
            name="ETH/BTC Ratio",
            description="Risk appetite within crypto. Rising = altcoin season approaching",
            expansion_threshold=0.0,       # Z-score based
            contraction_threshold=0.0,
            direction=1,
            weight=1.0,
            use_zscore=True,
            lookback_periods=90,           # 90 days
            source="manual",
        ),

        # OTHERS/BTC (Total3/BTC or similar)
        "others_btc": IndicatorConfig(
            name="OTHERS/BTC (Altcoin Index)",
            description="Altcoin performance vs BTC. ATL historically precedes alt surge",
            expansion_threshold=0.0,       # Z-score
            contraction_threshold=0.0,
            direction=1,
            weight=0.9,
            use_zscore=True,
            lookback_periods=90,
            source="manual",
        ),

        # BTC Dominance (inverse for alt exposure)
        "btc_dominance": IndicatorConfig(
            name="BTC Dominance %",
            description="BTC share of total crypto market cap. Lower = alt rotation",
            expansion_threshold=45.0,      # Below 45% = alts rotating
            contraction_threshold=55.0,    # Above 55% = BTC dominance
            direction=-1,                  # Lower = more alt opportunity
            weight=0.8,
            source="manual",
        ),
    })

    # ==========================================================================
    # ON-CHAIN INDICATORS (CryptoQuant)
    # ==========================================================================

    onchain_indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {

        # MVRV Ratio
        "mvrv": IndicatorConfig(
            name="MVRV Ratio",
            description="Market Value to Realized Value. >3 = overheated, <1 = undervalued",
            expansion_threshold=1.5,       # Healthy bull range
            contraction_threshold=2.8,     # Approaching overheated
            direction=1,                   # Complex: mid-range is best
            weight=1.2,
            source="cryptoquant",
        ),

        # Exchange Whale Ratio
        "exchange_whale_ratio": IndicatorConfig(
            name="Exchange Whale Ratio",
            description="Large transactions to exchange. >0.5 = distribution risk",
            expansion_threshold=0.35,      # Below = accumulation
            contraction_threshold=0.50,    # Above = distribution
            direction=-1,                  # Lower is better
            weight=1.0,
            source="cryptoquant",
        ),

        # Funding Rates
        "funding_rate": IndicatorConfig(
            name="Perp Funding Rate",
            description="Perpetual futures funding. Extreme = sentiment overheated",
            expansion_threshold=0.01,      # Mildly positive = healthy
            contraction_threshold=0.05,    # Very high = overheated longs
            direction=1,                   # Complex interpretation
            weight=0.9,
            source="cryptoquant",
        ),

        # Exchange Reserve
        "exchange_reserve": IndicatorConfig(
            name="Exchange BTC Reserve",
            description="BTC held on exchanges. Declining = bullish (self-custody)",
            expansion_threshold=0.0,       # Z-score: declining = positive
            contraction_threshold=0.0,
            direction=-1,                  # Lower/declining is better
            use_zscore=True,
            lookback_periods=30,
            weight=1.0,
            source="cryptoquant",
        ),

        # Stablecoin Exchange Reserve
        "stablecoin_reserve": IndicatorConfig(
            name="Stablecoin Exchange Reserve",
            description="Dry powder on exchanges. Rising = buying pressure incoming",
            expansion_threshold=0.0,       # Z-score
            contraction_threshold=0.0,
            direction=1,                   # Rising is bullish
            use_zscore=True,
            lookback_periods=30,
            weight=0.8,
            source="cryptoquant",
        ),

        # Long-term Holder Supply
        "lth_supply": IndicatorConfig(
            name="Long-term Holder Supply",
            description="Coins held >155 days. Rising = accumulation, falling = distribution",
            expansion_threshold=0.0,       # Z-score of change
            contraction_threshold=0.0,
            direction=1,
            use_zscore=True,
            lookback_periods=30,
            weight=1.1,
            source="cryptoquant",
        ),

        # SOPR (Spent Output Profit Ratio)
        "sopr": IndicatorConfig(
            name="SOPR",
            description="Profit/loss of spent coins. <1 = capitulation (buying opp), >1.05 = profit taking",
            expansion_threshold=1.0,       # Above 1 = profits being realized
            contraction_threshold=0.98,    # Below 1 = losses realized (capitulation)
            direction=1,                   # Complex: extremes are informative
            weight=0.9,
            source="cryptoquant",
        ),
    })

    # ==========================================================================
    # SCORING CONFIGURATION
    # ==========================================================================

    # Regime thresholds (composite score ranges)
    regime_thresholds: Dict[str, tuple] = field(default_factory=lambda: {
        MacroRegime.EXPANSION.value: (0.3, 1.0),        # Score > 0.3
        MacroRegime.LATE_CYCLE.value: (0.1, 0.3),       # 0.1 to 0.3
        MacroRegime.UNCERTAIN.value: (-0.1, 0.1),       # -0.1 to 0.1
        MacroRegime.EARLY_RECOVERY.value: (-0.3, -0.1), # -0.3 to -0.1
        MacroRegime.CONTRACTION.value: (-1.0, -0.3),    # Score < -0.3
    })

    # Position sizing by regime
    position_sizing: Dict[str, float] = field(default_factory=lambda: {
        MacroRegime.EXPANSION.value: 1.0,        # Full size
        MacroRegime.LATE_CYCLE.value: 0.7,       # Reduce exposure
        MacroRegime.UNCERTAIN.value: 0.5,        # Half size
        MacroRegime.EARLY_RECOVERY.value: 0.6,   # Building positions
        MacroRegime.CONTRACTION.value: 0.25,     # Minimal exposure
    })

    # Category weights for composite score
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "business_cycle": 1.0,
        "liquidity": 1.2,      # Slightly higher - liquidity drives crypto
        "real_economy": 0.8,
        "crypto": 1.0,
        "onchain": 1.1,        # On-chain often leads price
    })

    # ==========================================================================
    # KILL SWITCH RULES
    # ==========================================================================

    kill_switches: Dict[str, dict] = field(default_factory=lambda: {
        "ism_pmi_crash": {
            "indicator": "ism_pmi",
            "condition": "below",
            "threshold": 45.0,
            "action": "reduce_50pct",
            "description": "ISM PMI below 45 = severe contraction",
        },
        "dxy_spike": {
            "indicator": "dxy",
            "condition": "above",
            "threshold": 110.0,
            "action": "reduce_30pct",
            "description": "Dollar strength above 110 = liquidity crisis",
        },
        "credit_blowout": {
            "indicator": "credit_spreads",
            "condition": "above",
            "threshold": 600,
            "action": "reduce_50pct",
            "description": "HY spreads above 600bps = credit stress",
        },
        "mvrv_extreme": {
            "indicator": "mvrv",
            "condition": "above",
            "threshold": 3.5,
            "action": "reduce_30pct",
            "description": "MVRV above 3.5 = historically overheated",
        },
        "whale_distribution": {
            "indicator": "exchange_whale_ratio",
            "condition": "above",
            "threshold": 0.55,
            "action": "reduce_20pct",
            "description": "Whale ratio above 0.55 = active distribution",
        },
    })

    def to_json(self) -> str:
        """Serialize config to JSON for persistence"""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, IndicatorConfig):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        data = {
            "business_cycle_indicators": serialize(self.business_cycle_indicators),
            "liquidity_indicators": serialize(self.liquidity_indicators),
            "real_economy_indicators": serialize(self.real_economy_indicators),
            "crypto_indicators": serialize(self.crypto_indicators),
            "onchain_indicators": serialize(self.onchain_indicators),
            "regime_thresholds": self.regime_thresholds,
            "position_sizing": self.position_sizing,
            "category_weights": self.category_weights,
            "kill_switches": self.kill_switches,
        }
        return json.dumps(data, indent=2)


# Default configuration instance
DEFAULT_CONFIG = MacroRegimeConfig()

"""
IT Engine configuration loader.

Reads from config/it_engine.toml with sensible defaults.
"""

import os
from dataclasses import dataclass, field

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from config_utils import load_symbols
except ImportError:
    from config_utils import load_symbols


def _load_costs_toml(path: str = "config/costs.toml") -> dict:
    """Load the shared costs config if it exists."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class CostConfig:
    binance_vip9_rt_bps: float = 1.61
    binance_vip0_rt_bps: float = 3.50
    hyperliquid_rt_bps: float = 7.00

    @classmethod
    def from_shared(cls, costs_path: str = "config/costs.toml") -> "CostConfig":
        """Load from shared costs.toml."""
        raw = _load_costs_toml(costs_path)
        hl = raw.get("hyperliquid", {})
        bn = raw.get("binance", {})
        return cls(
            binance_vip9_rt_bps=bn.get("vip9_round_trip_bps", 1.61),
            binance_vip0_rt_bps=bn.get("vip0_round_trip_bps", 3.50),
            hyperliquid_rt_bps=hl.get("round_trip_taker_bps", 7.00),
        )

    @property
    def default_fee_rt_bps(self) -> float:
        return self.binance_vip9_rt_bps


@dataclass
class ITEngineConfig:
    # Buffer
    buffer_size: int = 6000  # 10min at 100ms
    compute_interval_s: float = 5.0

    # KSG estimator
    ksg_k: int = 5

    # Horizons (tick-level)
    horizons: list[int] = field(default_factory=lambda: [10, 50, 500])

    # Stride divisor: stride = max(1, horizon // stride_divisor)
    # Breaks overlapping-return autocorrelation for MI estimation
    stride_divisor: int = 5

    # Bar horizons (medium-frequency)
    bar_horizons: list[str] = field(default_factory=lambda: ["5min", "25min", "50min"])

    # Greedy selector
    max_features_greedy: int = 10

    # Entropy features for conditioning
    entropy_conditioning: list[str] = field(default_factory=lambda: [
        "ent_tick_5s", "ent_book_shape_std", "ent_perm_8_std"
    ])

    # Symbols
    symbols: list[str] = field(default_factory=load_symbols)

    # Transfer entropy
    te_top_n: int = 20  # compute TE for top-N features by MI
    te_lag: int = 1
    te_order: int = 1
    te_method: str = "ksg"  # "ksg" (nonparametric) or "linear" (Gaussian AR)

    # CMI sample guard
    cmi_min_samples: int = 500  # minimum N for reliable CMI estimation (5D conditioning)
    cmi_max_z_dims: int = 5     # cap Z dimensionality to prevent curse of dimensionality

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_feature_key: str = "nat:features:{symbol}"
    redis_output_key: str = "nat:it:{symbol}"

    # State
    state_dir: str = "data/it_engine"

    # Costs — loaded from shared config/costs.toml, overridable in it_engine.toml
    costs: CostConfig = field(default_factory=lambda: CostConfig.from_shared())

    @classmethod
    def load(cls, path: str = "config/it_engine.toml") -> "ITEngineConfig":
        """Load config from TOML file, falling back to defaults."""
        if not os.path.exists(path):
            return cls()

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        section = raw.get("it_engine", {})
        costs_raw = section.pop("costs", {})

        config = cls()
        for key, val in section.items():
            if hasattr(config, key):
                setattr(config, key, val)

        # Local overrides on top of shared costs
        if costs_raw:
            for key, val in costs_raw.items():
                if hasattr(config.costs, key):
                    setattr(config.costs, key, val)

        # Environment overrides
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            config.redis_url = redis_url

        return config

# 2.2 Config Generator

## Status: NOT_STARTED

## Goal

Generate N TOML config variants from a base config + parameter ranges. Each
variant represents one point in the 35-dimensional parameter space.

## Prerequisites

- [2.1 Shared Ingestor](2_1_shared_ingestor.md) — architecture defined

## Parameter Space (~35 dimensions)

### Algorithm Thresholds (15 params)

| Parameter | Type | Range | Source |
|-----------|------|-------|--------|
| jump_detector.z_threshold | float | [2.0, 5.0] | `config/algorithms.toml` |
| jump_detector.window | int | [50, 500] | `config/algorithms.toml` |
| optimal_entry.sprt_upper | float | [1.0, 5.0] | `config/algorithms.toml` |
| optimal_entry.sprt_lower | float | [-5.0, -1.0] | `config/algorithms.toml` |
| optimal_entry.kalman_q | float | [1e-5, 1e-2] | `config/algorithms.toml` |
| funding_reversion.z_window | int | [100, 1000] | `config/algorithms.toml` |
| funding_reversion.z_threshold | float | [1.5, 4.0] | `config/algorithms.toml` |
| surprise_signal.entropy_window | int | [50, 500] | `config/algorithms.toml` |
| 3f_liquidity.flow_window | int | [100, 500] | `config/algorithms.toml` |
| 3f_liquidity.spread_weight | float | [0.1, 0.5] | `config/algorithms.toml` |
| ... (5 more per-algorithm params) | | | |

### Ensemble Parameters (5 params)

| Parameter | Type | Range | Source |
|-----------|------|-------|--------|
| ensemble.method | categorical | [equal_weight, ic_weight, regime_switch] | `config/algorithms.toml` |
| ensemble.ic_lookback | int | [100, 2000] | `config/algorithms.toml` |
| ensemble.regime_column | categorical | [regime_label, vol_regime] | `config/algorithms.toml` |
| ensemble.top_n | int | [3, 10] | `config/algorithms.toml` |
| ensemble.min_ic | float | [0.0, 0.05] | `config/algorithms.toml` |

### Trading Parameters (8 params)

| Parameter | Type | Range | Source |
|-----------|------|-------|--------|
| entry_percentile | float | [0.7, 0.95] | `config/algorithms.toml` |
| exit_percentile | float | [0.3, 0.6] | `config/algorithms.toml` |
| bar_seconds | int | [60, 600] | `config/algorithms.toml` |
| max_position_usd | float | [1000, 50000] | `config/algorithms.toml` |
| stop_loss_pct | float | [0.005, 0.03] | `config/algorithms.toml` |
| take_profit_pct | float | [0.005, 0.05] | `config/algorithms.toml` |
| cooldown_bars | int | [1, 10] | `config/algorithms.toml` |
| max_daily_trades | int | [10, 200] | `config/algorithms.toml` |

### Feature Selection (7 binary params)

| Feature Category | Type | Default |
|------------------|------|---------|
| whale_flow | bool | true |
| liquidation_risk | bool | true |
| concentration | bool | true |
| regime | bool | true |
| gmm_classification | bool | false |
| cross_symbol | bool | false |
| heatmap | bool | false |

## Implementation

**New file:** `scripts/swarm/config_generator.py`

```python
class ConfigGenerator:
    def __init__(self, base_config: str, param_ranges: dict):
        self.base = toml.load(base_config)
        self.ranges = param_ranges

    def generate_random(self, n: int) -> list[dict]:
        """Generate N random configs from parameter ranges."""

    def generate_grid(self, resolution: int) -> list[dict]:
        """Generate grid search configs (for small subspaces)."""

    def generate_from_optuna(self, trial) -> dict:
        """Generate config from Optuna trial (Tier 3 integration)."""

    def write_configs(self, configs: list[dict], output_dir: str):
        """Write configs as {output_dir}/config_{i}.toml"""
```

**Parameter ranges file:** `config/swarm_ranges.toml`

```toml
[jump_detector]
z_threshold = { min = 2.0, max = 5.0, type = "float" }
window = { min = 50, max = 500, type = "int" }

[ensemble]
method = { choices = ["equal_weight", "ic_weight", "regime_switch"], type = "categorical" }
```

## Verification

```bash
python scripts/swarm/config_generator.py \
  --base config/algorithms.toml \
  --ranges config/swarm_ranges.toml \
  --count 16 \
  --output data/swarm/configs/

ls data/swarm/configs/
# config_0.toml config_1.toml ... config_15.toml
```

## Files Created

- `scripts/swarm/config_generator.py`
- `config/swarm_ranges.toml`

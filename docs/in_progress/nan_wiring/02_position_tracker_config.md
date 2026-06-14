# 02 — Position Tracker Config and Shared State

## What

Add configuration for the existing `PositionTracker` and create a thread-safe
`SharedPositionState` that bridges the tracker task with per-symbol `MarketState`.

## Why

- `PositionTracker` (367 LOC in `ing/src/positions/tracker.rs`) is fully implemented
- It polls `clearinghouse_state(wallet)` via REST — public, no auth
- Produces `PositionSnapshot` (with liquidation_price, position_value, leverage, size)
  and `PositionDelta` (size_delta, change_type)
- But it's never started — no config section, no integration with main loop
- This step adds the config + shared state. Step 03 spawns and wires it.

## Changes

### 1. Config struct — `rust/ing/src/config.rs`

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct PositionTrackerTomlConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_poll_secs")]
    pub poll_interval_secs: u64,          // 60
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,   // 10
    #[serde(default)]
    pub initial_wallets: Vec<String>,
    #[serde(default = "default_true")]
    pub discover_from_trades: bool,       // true
    #[serde(default = "default_max_wallets")]
    pub max_tracked_wallets: usize,       // 50
}
```

Add to top-level `Config`:
```rust
#[serde(default)]
pub position_tracker: Option<PositionTrackerTomlConfig>,
```

### 2. Shared state — new file `rust/ing/src/positions/state.rs`

```rust
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use super::snapshot::{PositionSnapshot, PositionDelta};

pub struct SharedPositionState {
    /// Latest snapshots keyed by symbol -> vec of wallet snapshots
    snapshots: RwLock<HashMap<String, Vec<PositionSnapshot>>>,
    /// Recent deltas not yet consumed, keyed by symbol
    deltas: RwLock<HashMap<String, VecDeque<PositionDelta>>>,
}
```

Methods:
- `update(&self, snapshots: Vec<PositionSnapshot>, deltas: Vec<PositionDelta>)`
  Groups by symbol, replaces snapshots, appends deltas.
- `drain_deltas(&self, symbol: &str) -> Vec<PositionDelta>`
  Returns and clears deltas for one symbol.
- `get_liquidation_positions(&self, symbol: &str) -> Vec<LiquidationPosition>`
  Converts `PositionSnapshot` -> `LiquidationPosition` (direct field mapping).
- `get_concentration_positions(&self, symbol: &str) -> Vec<ConcentrationPosition>`
  Converts `PositionSnapshot` -> position value + whale flag.
- `snapshot_count(&self, symbol: &str) -> usize`
  For logging/health.

### 3. Register module — `rust/ing/src/positions/mod.rs`

Add:
```rust
mod state;
pub use state::*;
```

### 4. Config file — `config/ing.toml`

```toml
[position_tracker]
enabled = false
poll_interval_secs = 60
max_concurrent_requests = 10
discover_from_trades = true
max_tracked_wallets = 50
# initial_wallets = ["0xabc...", "0xdef..."]
```

## Verify

```bash
cd rust && cargo test -p ing
# Unit tests for SharedPositionState:
#   - insert snapshots, verify get_liquidation_positions conversion
#   - insert deltas, verify drain_deltas returns and clears
#   - verify snapshot_count
```

## Notes

- `PositionSnapshot` already has all fields needed by `LiquidationPosition`:
  `liquidation_price` (Option<f64>), `position_value`, `size` (sign = direction), `entry_price`
- `parking_lot::RwLock` is already a dependency (used in tracker.rs)
- `enabled = false` by default — opt-in until wallet discovery is validated

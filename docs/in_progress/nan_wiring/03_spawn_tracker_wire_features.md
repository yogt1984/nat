# 03 — Spawn Tracker and Wire All 40 Features

## What

Start `PositionTracker` as a tokio task in `main.rs`. Feed position data into
`MarketState` to produce liquidation_risk (13) and concentration (15) features,
and upgrade whale_flow (12) from trade-size heuristic to real position deltas.

## Why

- Step 01 gives us 12 whale_flow features from trade classification (heuristic)
- Step 02 gives us config + shared state
- This step wires them together: tracker -> shared state -> MarketState -> features
- Unlocks up to 40 features total (12 whale + 13 liq + 15 concentration)

## Changes

### 1. Spawn tracker — `rust/ing/src/main.rs`

Before the symbol ingestor loop (~line 153), if `config.position_tracker.enabled`:

```rust
let shared_pos_state = Arc::new(SharedPositionState::new());

// Create channels
let (snapshot_tx, mut snapshot_rx) = mpsc::channel(1000);
let (delta_tx, mut delta_rx) = mpsc::channel(1000);

// Build tracker
let tracker_config = PositionTrackerConfig {
    poll_interval_secs: pt_config.poll_interval_secs,
    symbols: config.symbols.clone(),
    max_concurrent_requests: pt_config.max_concurrent_requests,
};
let tracker = PositionTracker::new(tracker_config)?
    .with_snapshot_channel(snapshot_tx)
    .with_delta_channel(delta_tx);

// Add initial wallets
if !pt_config.initial_wallets.is_empty() {
    tracker.add_wallets(&pt_config.initial_wallets);
}

// Spawn tracker polling loop
tokio::spawn(async move { tracker.run().await });

// Spawn consumer: channels -> SharedPositionState
let sps = shared_pos_state.clone();
tokio::spawn(async move {
    loop {
        tokio::select! {
            Some(snapshot) = snapshot_rx.recv() => {
                sps.update_snapshot(snapshot);
            }
            Some(delta) = delta_rx.recv() => {
                sps.update_delta(delta);
            }
            else => break,
        }
    }
});
```

Pass `Arc<SharedPositionState>` to each `run_symbol_ingestor()`.

### 2. Wire into MarketState — `rust/ing/src/state/mod.rs`

Add field:
```rust
position_state: Option<Arc<SharedPositionState>>,
```

Add setter:
```rust
pub fn set_position_state(&mut self, state: Arc<SharedPositionState>) {
    self.position_state = Some(state);
}
```

In `compute_features()`, after the Phase 1 whale_flow block:

```rust
if let Some(ref pos_state) = self.position_state {
    // Upgrade whale flow with real position deltas
    let deltas = pos_state.drain_deltas(&self.symbol);
    if let Some(ref mut wfb) = self.whale_flow_buffer {
        for d in &deltas {
            wfb.add_change(WhalePositionChange {
                timestamp_ms: d.timestamp_ms,
                wallet: d.wallet.clone(),
                symbol: d.symbol.clone(),
                position_change_usd: d.size_delta * d.curr_entry_price,
                is_market_maker: false,
            });
        }
    }

    // Liquidation risk (13 features)
    let liq_positions = pos_state.get_liquidation_positions(&self.symbol);
    if !liq_positions.is_empty() {
        let price = self.order_book.midprice().unwrap_or(0.0);
        let oi = self.context.open_interest();
        features.liquidation_risk = Some(
            liquidation::compute(&liq_positions, price, oi, &LiquidationRiskConfig::default())
        );
    }

    // Concentration (15 features)
    let conc_positions = pos_state.get_concentration_positions(&self.symbol);
    if !conc_positions.is_empty() {
        let oi = self.context.open_interest();
        features.concentration = Some(
            self.concentration_buffer.compute(&conc_positions, oi)
        );
    }
}
```

Add `concentration_buffer: ConcentrationBuffer` field, initialized with default config.

### 3. Context accessors — `rust/ing/src/state/mod.rs`

If `context` or `order_book` fields are private, add getters:
```rust
fn midprice(&self) -> f64 { self.order_book.midprice().unwrap_or(0.0) }
fn open_interest(&self) -> f64 { self.context.open_interest() }
```

### 4. Graceful shutdown

The tracker runs in a spawned task. On SIGTERM (already handled in main.rs),
the tracker task is dropped automatically. No special shutdown needed — it's
a read-only polling loop with no state to flush.

## Verify

```bash
cd rust && cargo test -p ing -p ing-features
# Integration: run ingestor with position_tracker.enabled = true
# and at least 1 wallet in initial_wallets
# After first poll (60s), check parquet:
#   - liquidation_risk_above_1pct etc. should have real values
#   - concentration columns depend on having enough wallets
```

## Data Flow

```
PositionTracker::poll_all()
  -> clearinghouse_state(wallet) per wallet
  -> PositionSnapshot + PositionDelta
  -> mpsc channels
  -> consumer task -> SharedPositionState
  -> MarketState::compute_features() reads SharedPositionState
  -> features.whale_flow / liquidation_risk / concentration = Some(...)
  -> Parquet writer (real values instead of NaN)
```

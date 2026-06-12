# 01 — Whale Flow from Trade Classification (12 features)

## What

Classify large trades (>$100K USD) from the existing WebSocket trade stream as whale activity.
Feed them into the already-implemented `WhaleFlowBuffer` to produce 12 real-valued features
that are currently NaN-padded.

## Why

- All 12 whale_flow features have full Rust computation code (1109 LOC in `ing-features/src/whale_flow.rs`)
- They emit NaN because nothing feeds `WhalePositionChange` into the buffer
- Trade-size classification is a zero-cost heuristic: no new API calls, no wallet tracking
- Later upgraded with real position data in step 03

## Features Unlocked

```
whale_net_flow_1h, whale_net_flow_4h, whale_net_flow_24h,
whale_flow_normalized_1h, whale_flow_normalized_4h,
whale_flow_momentum, whale_flow_intensity, whale_flow_roc,
whale_buy_ratio, whale_directional_agreement,
active_whale_count, whale_total_activity
```

## Changes

### 1. Config — `rust/ing-types/src/config.rs`

Add to `FeaturesConfig`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct WhaleFlowTradeConfig {
    #[serde(default = "default_whale_threshold_usd")]
    pub whale_threshold_usd: f64,
}
fn default_whale_threshold_usd() -> f64 { 100_000.0 }

// In FeaturesConfig:
#[serde(default)]
pub whale_flow: Option<WhaleFlowTradeConfig>,
```

### 2. Buffer + classification — `rust/ing/src/state/mod.rs`

Add fields to `MarketState`:
```rust
whale_flow_buffer: Option<WhaleFlowBuffer>,
whale_threshold_usd: f64,
```

Initialize in `new_with_algorithms()` from config.

In `update()`, inside `WsMessage::Trades` arm, after `self.trade_buffer.add(trade)`:
```rust
let notional = trade.price() * trade.size();
if notional >= self.whale_threshold_usd {
    if let Some(ref mut wfb) = self.whale_flow_buffer {
        wfb.add_change(WhalePositionChange {
            timestamp_ms: trade.time as i64,
            wallet: format!("trade_{}", trade.tid),
            symbol: self.symbol.clone(),
            position_change_usd: notional * if trade.is_buy() { 1.0 } else { -1.0 },
            is_market_maker: false,
        });
    }
}
```

### 3. Wire compute — `rust/ing/src/state/mod.rs`

In `compute_features()`, after base feature computation:
```rust
if let Some(ref mut wfb) = self.whale_flow_buffer {
    if !wfb.is_empty() {
        features.whale_flow = Some(wfb.compute());
    }
}
```

### 4. Config file — `config/ing.toml`

```toml
[features.whale_flow]
whale_threshold_usd = 100000.0
```

## Verify

```bash
cd rust && cargo test -p ing -p ing-features
# Run ingestor 5 min, inspect parquet:
# 12 whale_flow columns should have real values after first large trade
```

## Limitations

- Trade size != position change (a $500K position may execute across many small fills)
- No market maker classification (all trades treated as non-MM)
- Upgraded to real position data in step 03

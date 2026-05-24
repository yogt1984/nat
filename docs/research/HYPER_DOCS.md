# Hyperliquid API Documentation for Feature Extraction

## Executive Summary: Why Hyperliquid?

**Your assumption is correct.** Hyperliquid is an excellent choice for this project due to:

| Factor | Assessment | Details |
|--------|------------|---------|
| **Transparency** | Excellent | Fully on-chain CLOB - every order, cancel, trade, liquidation verifiable |
| **Data Richness** | Excellent | L2/L4 book, trades, funding, liquidations, user flows all accessible |
| **Latency** | Good | ~200ms finality, 200k orders/sec throughput |
| **API Quality** | Good | WebSocket + REST, multiple SDKs (Python, Rust, Go, TS) |
| **Self-Hosting** | Excellent | Can run own order book server for L4 data |

**Key differentiator**: Unlike CEXs where the order book is a black box, Hyperliquid's on-chain CLOB means you can observe the *complete* market microstructure - not just aggregated L2 data.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYPERLIQUID DATA FLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│   │   HyperBFT   │────────▶│   HyperCore  │────────▶│  Public API  │   │
│   │  (Consensus) │         │   (Matching) │         │   (WebSocket)│   │
│   └──────────────┘         └──────────────┘         └──────────────┘   │
│         │                         │                         │           │
│         │                         │                         ▼           │
│         │                         │                  ┌──────────────┐   │
│         │                         │                  │   ING Agent  │   │
│         │                         │                  │   (Your App) │   │
│         │                         ▼                  └──────────────┘   │
│         │                  ┌──────────────┐                │           │
│         └─────────────────▶│  Order Book  │────────────────┘           │
│                            │   Server     │    (L4 data)               │
│                            │  (Self-host) │                             │
│                            └──────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### Base URLs

| Network | REST | WebSocket |
|---------|------|-----------|
| **Mainnet** | `https://api.hyperliquid.xyz/info` | `wss://api.hyperliquid.xyz/ws` |
| **Testnet** | `https://api.hyperliquid-testnet.xyz/info` | `wss://api.hyperliquid-testnet.xyz/ws` |

### Rate Limits (Public API)

- **Connections**: 100 simultaneous WebSocket connections
- **Subscriptions**: 1,000 total subscriptions per connection
- **REST**: Rate-limited per user (check `userRateLimit` endpoint)

---

## WebSocket Subscriptions (Real-Time Data)

### Market Data Channels

| Channel | Parameters | Data Structure | Update Frequency | Use Case |
|---------|------------|----------------|------------------|----------|
| `l2Book` | `coin`, `nSigFigs`?, `mantissa`? | `WsBook` | On change | Order book depth |
| `trades` | `coin` | `WsTrade[]` | On execution | Trade flow analysis |
| `bbo` | `coin` | `WsBbo` | On change | Best bid/offer tracking |
| `allMids` | `dex`? | `AllMids` | Periodic | Cross-market mid prices |
| `candle` | `coin`, `interval` | `Candle[]` | On close | OHLCV for spectral analysis |
| `activeAssetCtx` | `coin` | `WsActiveAssetCtx` | On change | Funding, open interest |

### User-Specific Channels (Requires Address)

| Channel | Parameters | Data Structure | Use Case |
|---------|------------|----------------|----------|
| `orderUpdates` | `user` | `WsOrder[]` | Order lifecycle tracking |
| `userFills` | `user`, `aggregateByTime`? | `WsUserFills` | Execution analysis |
| `userFundings` | `user` | `WsUserFundings` | Funding payment tracking |
| `clearinghouseState` | `user`, `dex`? | `ClearinghouseState` | Position/margin state |
| `openOrders` | `user`, `dex`? | `OpenOrders` | Active order tracking |
| `userNonFundingLedgerUpdates` | `user` | Polymorphic | Liquidations, deposits, etc. |

### Subscription Message Format

```json
{
  "method": "subscribe",
  "subscription": {
    "type": "l2Book",
    "coin": "BTC",
    "nSigFigs": 5
  }
}
```

---

## Data Structures (Feature Extraction Targets)

### WsBook (L2 Order Book)

```typescript
interface WsBook {
  coin: string;           // e.g., "BTC", "ETH"
  levels: [
    WsLevel[],            // Bids (descending price)
    WsLevel[]             // Asks (ascending price)
  ];
  time: number;           // Unix timestamp (ms)
}

interface WsLevel {
  px: string;             // Price (string for precision)
  sz: string;             // Size (string for precision)
  n: number;              // Number of orders at this level
}
```

**Extractable Features:**
- `RAW_LOB_BID_P{1-N}`, `RAW_LOB_ASK_P{1-N}` - Price levels
- `RAW_LOB_BID_Q{1-N}`, `RAW_LOB_ASK_Q{1-N}` - Quantities
- `RAW_LOB_BID_N{1-N}`, `RAW_LOB_ASK_N{1-N}` - Order counts (unique to L2!)
- `IMBALANCE_LOB_*` - All imbalance metrics
- `QUALITY_DEPTH_*` - Depth within price bands

### WsTrade (Trade Execution)

```typescript
interface WsTrade {
  coin: string;           // Asset
  side: "A" | "B";        // A = sell aggressor, B = buy aggressor
  px: string;             // Execution price
  sz: string;             // Execution size
  hash: string;           // Transaction hash (on-chain!)
  time: number;           // Timestamp (ms)
  tid: number;            // Trade ID (50-bit hash of oids)
  users?: [string, string]; // [maker_address, taker_address] - UNIQUE!
}
```

**Extractable Features:**
- `FLOW_TRADE_*` - Volume, count, VWAP, intensity
- `FLOW_AGGRESSOR_*` - Buy/sell aggressor ratio
- `FLOW_TOXICITY_*` - Order flow toxicity (VPIN-like)
- **Unique**: User address analysis for informed flow detection

### WsBbo (Best Bid/Offer)

```typescript
interface WsBbo {
  coin: string;
  time: number;
  bbo: [[string, string], [string, string]]; // [[bid_px, bid_sz], [ask_px, ask_sz]]
}
```

**Use**: Lightweight spread/microprice tracking without full book overhead.

### WsActiveAssetCtx (Asset Context)

```typescript
interface WsActiveAssetCtx {
  coin: string;
  ctx: {
    dayNtlVlm: string;      // 24h notional volume
    funding: string;        // Current funding rate
    openInterest: string;   // Total open interest
    oraclePx: string;       // Oracle price
    prevDayPx: string;      // Previous day close
    markPx: string;         // Mark price
    premium: string;        // Basis (mark - oracle)
  };
}
```

**Extractable Features:**
- `FUNDING_RATE_CURRENT` - Funding rate
- `BASIS_MARK_ORACLE` - Mark-oracle premium
- `OI_TOTAL`, `OI_CHANGE_*` - Open interest dynamics
- `VOLUME_24H` - Activity level

### ClearinghouseState (User Positions)

```typescript
interface ClearinghouseState {
  assetPositions: AssetPosition[];
  marginSummary: MarginSummary;
  crossMarginSummary: CrossMarginSummary;
  withdrawable: string;
}

interface AssetPosition {
  position: {
    coin: string;
    szi: string;            // Signed size (negative = short)
    leverage: {
      type: "cross" | "isolated";
      value: number;
    };
    entryPx: string;
    positionValue: string;
    unrealizedPnl: string;
    liquidationPx: string | null;
  };
}
```

---

## REST Info Endpoint (Snapshots & Historical)

### Key Queries for Feature Engineering

| Query Type | Purpose | Response |
|------------|---------|----------|
| `l2Book` | Book snapshot (up to 20 levels default) | `WsBook` equivalent |
| `candleSnapshot` | Historical OHLCV | Candle array |
| `userFillsByTime` | Historical fills with pagination | Fill array |
| `meta` | Asset metadata (tick size, lot size, etc.) | Universe info |
| `metaAndAssetCtxs` | Meta + live context (funding, OI) | Combined |
| `fundingHistory` | Historical funding rates | Funding array |
| `clearinghouseState` | Position snapshot | Margin/position state |

### Example: Candle Snapshot Request

```json
{
  "type": "candleSnapshot",
  "req": {
    "coin": "BTC",
    "interval": "1m",
    "startTime": 1704067200000,
    "endTime": 1704153600000
  }
}
```

---

## L4 Order Book (Self-Hosted Server)

### What L4 Provides (Beyond Public L2)

| Data Level | Public API | Self-Hosted L4 |
|------------|------------|----------------|
| Aggregated depth | Up to 20 levels | Up to 100 levels |
| Individual orders | No | Yes |
| Order diffs by block | No | Yes |
| User addresses on orders | No | Yes (with node) |
| Latency | ~50-100ms | ~10-20ms (local) |

### L4 Subscription

```json
{
  "method": "subscribe",
  "subscription": {
    "type": "l4book",
    "coin": "BTC"
  }
}
```

**Response Flow:**
1. Initial full book snapshot
2. Block-by-block order diffs (add, cancel, modify, fill)

**Unique Features Unlockable:**
- `ORDERFLOW_INDIVIDUAL_*` - Individual order size distribution
- `QUEUE_POSITION_*` - Queue dynamics at each level
- `CANCEL_RATE_*` - Cancellation patterns
- `INFORMED_FLOW_*` - Address-level flow analysis

### Running the Order Book Server

```bash
# 1. Run non-validating Hyperliquid node
git clone https://github.com/hyperliquid-dex/node
cd node && ./run_node.sh

# 2. Run order book server
git clone https://github.com/hyperliquid-dex/order_book_server
cd order_book_server && cargo run --release
```

---

## Information Content Summary

### What Can Be Extracted

| Category | Public API | With L4 Server | Entropy Relevance |
|----------|------------|----------------|-------------------|
| **Price Levels** | 20 levels | 100 levels | Higher resolution for book shape entropy |
| **Order Counts per Level** | Yes (`n` field) | Yes + individual | Critical for queue entropy |
| **Trade Flow** | Full | Full | Aggressor ratio, toxicity |
| **Funding Rate** | Yes | Yes | Regime indicator |
| **Open Interest** | Yes | Yes | Positioning entropy |
| **User Addresses** | On trades | On orders too | Informed trader detection |
| **Liquidations** | Via ledger updates | Yes | Forced flow detection |
| **Historical Candles** | Yes | Yes | Spectral analysis |

### Unique Hyperliquid Advantages for Your Project

1. **On-Chain Transparency**: Every order is verifiable - no hidden dark pools or internalization
2. **Order Count at Levels**: The `n` field in `WsLevel` reveals queue fragmentation (rare in CEX APIs)
3. **User Addresses**: Can potentially identify informed vs noise traders
4. **Self-Hosted L4**: Block-by-block diffs enable microstructure research impossible elsewhere
5. **Funding Rate**: Continuous funding (every 8h) provides regime signal
6. **No Maker Rebates**: Cleaner price discovery (all taker fees positive)

---

## Recommended Data Collection Strategy

### Phase 1: Public API (Immediate)

```
┌─────────────────────────────────────────────────┐
│              INITIAL DATA COLLECTION            │
├─────────────────────────────────────────────────┤
│                                                  │
│  Subscribe to (per asset):                       │
│  ├── l2Book (full depth, nSigFigs=5)            │
│  ├── trades                                      │
│  ├── bbo (lightweight backup)                   │
│  └── activeAssetCtx (funding, OI)               │
│                                                  │
│  Poll periodically:                              │
│  ├── candleSnapshot (backfill)                  │
│  └── fundingHistory (backfill)                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Phase 2: L4 Server (After Validation)

Deploy self-hosted order book server for:
- Individual order visibility
- Lower latency
- Block-level granularity

### Asset Selection

Start with high-liquidity perpetuals:
1. **BTC** - Most liquid, lowest noise
2. **ETH** - Second most liquid
3. **SOL** - High retail activity, different microstructure

---

## Code Sketch: Rust WebSocket Client

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde::{Deserialize, Serialize};
use futures_util::{SinkExt, StreamExt};

#[derive(Serialize)]
struct Subscription {
    method: String,
    subscription: SubscriptionParams,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum SubscriptionParams {
    #[serde(rename = "l2Book")]
    L2Book { coin: String },
    #[serde(rename = "trades")]
    Trades { coin: String },
    #[serde(rename = "activeAssetCtx")]
    ActiveAssetCtx { coin: String },
}

#[derive(Deserialize, Debug)]
struct WsMessage {
    channel: String,
    data: serde_json::Value,
}

async fn connect_hyperliquid(coins: Vec<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let url = "wss://api.hyperliquid.xyz/ws";
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to channels
    for coin in coins {
        let sub = Subscription {
            method: "subscribe".into(),
            subscription: SubscriptionParams::L2Book { coin: coin.into() },
        };
        write.send(Message::Text(serde_json::to_string(&sub)?)).await?;

        let sub = Subscription {
            method: "subscribe".into(),
            subscription: SubscriptionParams::Trades { coin: coin.into() },
        };
        write.send(Message::Text(serde_json::to_string(&sub)?)).await?;
    }

    // Process messages
    while let Some(msg) = read.next().await {
        match msg? {
            Message::Text(text) => {
                let parsed: WsMessage = serde_json::from_str(&text)?;
                // Route to feature extractors...
                println!("Channel: {}, Data: {:?}", parsed.channel, parsed.data);
            }
            _ => {}
        }
    }

    Ok(())
}
```

---

## References

- [Official WebSocket Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket)
- [Subscription Types](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
- [Info Endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [Order Book Server (GitHub)](https://github.com/hyperliquid-dex/order_book_server)
- [Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- [Rust SDK](https://docs.rs/hyperliquid/latest/hyperliquid/)
- [Technical Deep Dive (RockNBlock)](https://rocknblock.io/blog/how-does-hyperliquid-work-a-technical-deep-dive)
- [On-Chain Order Book Analysis (Medium)](https://medium.com/@gwrx2005/hyperliquid-on-chain-order-book-6df27cbce416)

//! Emission budget assertion: p99 feature computation latency must be < 80ms.
//!
//! The ingestor emits features every 100ms. This test verifies that the
//! stress-case computation (10-level book, 300 trades, warmed buffers)
//! stays within 80% of that budget, leaving 20ms for Parquet write + channel send.
//!
//! Run: `cd rust && cargo test --package ing-features -- emission_budget`

use std::time::{Duration, Instant};

use ing_features::FeatureComputer;
use ing_types::messages::{WsBook, WsLevel, WsTrade};
use ing_types::state::{MarketContext, OrderBook, TradeBuffer};
use ing_types::FeaturesConfig;
use ing_types::RingBuffer;

const ITERATIONS: usize = 1000;
const BUDGET_P99_MS: u128 = 80;

fn level(px: f64, sz: f64, n: u32) -> WsLevel {
    WsLevel {
        px: px.to_string(),
        sz: sz.to_string(),
        n,
    }
}

fn make_book(depth: usize, mid: f64) -> OrderBook {
    let half_spread = 0.5;
    let mut bids = Vec::with_capacity(depth);
    let mut asks = Vec::with_capacity(depth);
    for i in 0..depth {
        let offset = half_spread + i as f64 * 0.5;
        bids.push(level(mid - offset, 1.0 + i as f64 * 0.5, 3));
        asks.push(level(mid + offset, 1.0 + i as f64 * 0.5, 3));
    }
    let ws = WsBook {
        coin: "BTC".to_string(),
        levels: (bids, asks),
        time: 1717574400000,
    };
    let mut ob = OrderBook::new(depth);
    ob.update(&ws);
    ob
}

fn make_trades(n: usize, base_ts: u64) -> TradeBuffer {
    let mut tb = TradeBuffer::new(60);
    for i in 0..n {
        let trade = WsTrade {
            coin: "BTC".to_string(),
            side: if i % 3 == 0 {
                "A".to_string()
            } else {
                "B".to_string()
            },
            px: format!("{:.1}", 67000.0 + (i as f64 * 0.1).sin()),
            sz: format!("{:.4}", 0.01 + (i as f64 * 0.001)),
            hash: format!("0x{:016x}", i),
            time: base_ts + i as u64 * 100,
            tid: i as u64,
            users: None,
        };
        tb.add(trade);
    }
    tb
}

fn make_price_buffer(n: usize) -> RingBuffer<f64> {
    let mut buf = RingBuffer::new(n);
    let mut price = 67000.0;
    for i in 0..n {
        price += (i as f64 * 0.01).sin() * 0.5;
        buf.push(price);
    }
    buf
}

#[test]
fn emission_budget_p99_under_80ms() {
    let config = FeaturesConfig {
        emission_interval_ms: 100,
        trade_buffer_seconds: 60,
        book_levels: 10,
        price_buffer_size: 3000,
        gmm_model_path: None,
        whale_flow: None,
    };
    let ts = 1717574400000_u64;

    let ob = make_book(10, 67000.0);
    let tb = make_trades(300, ts);
    let ctx = MarketContext::new();
    let prices = make_price_buffer(3000);

    // Warm up internal buffers (entropy, spread, midprice, vol_1m)
    let mut computer = FeatureComputer::new(&config);
    for _ in 0..100 {
        computer.compute(&ob, &tb, &ctx, &prices);
    }

    // Measure ITERATIONS iterations
    let mut durations = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let features = computer.compute(&ob, &tb, &ctx, &prices);
        let _ = features.to_vec();
        durations.push(start.elapsed());
    }

    durations.sort();

    let p50 = durations[ITERATIONS / 2];
    let p95 = durations[ITERATIONS * 95 / 100];
    let p99 = durations[ITERATIONS * 99 / 100];

    eprintln!(
        "Feature computation latency (stress, {} iterations):",
        ITERATIONS
    );
    eprintln!("  p50: {:?}", p50);
    eprintln!("  p95: {:?}", p95);
    eprintln!("  p99: {:?}", p99);

    assert!(
        p99 < Duration::from_millis(BUDGET_P99_MS as u64),
        "p99 latency {:?} exceeds {}ms budget",
        p99,
        BUDGET_P99_MS,
    );
}

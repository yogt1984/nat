//! Benchmarks for feature computation under synthetic load.
//!
//! Three scenarios:
//! - Baseline: empty book, no trades
//! - Normal: 10-level book, 30 trades in buffer
//! - Stress: 10-level book, 300 trades, all buffers warmed
//!
//! Run: `cd rust && cargo bench --package ing`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ing_features::FeatureComputer;
use ing_types::messages::{WsBook, WsLevel, WsTrade};
use ing_types::state::{MarketContext, OrderBook, TradeBuffer};
use ing_types::FeaturesConfig;
use ing_types::RingBuffer;

/// Build a WsLevel from price and size
fn level(px: f64, sz: f64, n: u32) -> WsLevel {
    WsLevel {
        px: px.to_string(),
        sz: sz.to_string(),
        n,
    }
}

/// Build an order book with `depth` levels on each side
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

/// Fill a trade buffer with `n` trades
fn make_trades(n: usize, base_ts: u64) -> TradeBuffer {
    let mut tb = TradeBuffer::new(60);
    for i in 0..n {
        let trade = WsTrade {
            coin: "BTC".to_string(),
            side: if i % 3 == 0 { "A".to_string() } else { "B".to_string() },
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

/// Fill a price buffer with `n` synthetic mid-prices
fn make_price_buffer(n: usize) -> RingBuffer<f64> {
    let mut buf = RingBuffer::new(n);
    let mut price = 67000.0;
    for i in 0..n {
        price += (i as f64 * 0.01).sin() * 0.5;
        buf.push(price);
    }
    buf
}

fn bench_feature_computation(c: &mut Criterion) {
    let config = FeaturesConfig {
        emission_interval_ms: 100,
        trade_buffer_seconds: 60,
        book_levels: 10,
        price_buffer_size: 3000,
        gmm_model_path: None,
    };
    let ts = 1717574400000_u64;

    // --- Baseline: empty book, no trades ---
    let mut group = c.benchmark_group("feature_compute");
    group.bench_function("baseline_empty", |b| {
        let mut computer = FeatureComputer::new(&config);
        let ob = OrderBook::new(10);
        let tb = TradeBuffer::new(60);
        let ctx = MarketContext::new();
        let prices = RingBuffer::new(3000);
        b.iter(|| {
            let features = computer.compute(&ob, &tb, &ctx, &prices);
            black_box(features.to_vec());
        });
    });

    // --- Normal: 10-level book, 30 trades ---
    group.bench_function("normal_10lvl_30trades", |b| {
        let mut computer = FeatureComputer::new(&config);
        let ob = make_book(10, 67000.0);
        let tb = make_trades(30, ts);
        let ctx = MarketContext::new();
        let prices = make_price_buffer(600);
        b.iter(|| {
            let features = computer.compute(&ob, &tb, &ctx, &prices);
            black_box(features.to_vec());
        });
    });

    // --- Stress: 10-level book, 300 trades, warmed buffers ---
    group.bench_function("stress_10lvl_300trades", |b| {
        let ob = make_book(10, 67000.0);
        let tb = make_trades(300, ts);
        let ctx = MarketContext::new();
        let prices = make_price_buffer(3000);

        // Warm up the computer with 100 iterations to fill internal buffers
        let mut computer = FeatureComputer::new(&config);
        for _ in 0..100 {
            computer.compute(&ob, &tb, &ctx, &prices);
        }

        b.iter(|| {
            let features = computer.compute(&ob, &tb, &ctx, &prices);
            black_box(features.to_vec());
        });
    });
    group.finish();

    // --- to_vec serialization ---
    c.bench_function("features_to_vec", |b| {
        let mut computer = FeatureComputer::new(&config);
        let ob = make_book(10, 67000.0);
        let tb = make_trades(30, ts);
        let ctx = MarketContext::new();
        let prices = make_price_buffer(600);
        let features = computer.compute(&ob, &tb, &ctx, &prices);
        b.iter(|| {
            black_box(features.to_vec());
        });
    });
}

criterion_group!(benches, bench_feature_computation);
criterion_main!(benches);

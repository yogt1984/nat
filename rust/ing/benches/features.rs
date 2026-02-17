//! Benchmarks for feature computation

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_permutation_entropy(c: &mut Criterion) {
    // This would benchmark the permutation entropy computation
    // For now, just a placeholder

    c.bench_function("permutation_entropy_32", |b| {
        let data: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        b.iter(|| {
            // Would call permutation_entropy here
            black_box(&data);
        });
    });
}

fn bench_order_book_update(c: &mut Criterion) {
    c.bench_function("order_book_update", |b| {
        // Would benchmark order book update
        b.iter(|| {
            black_box(42);
        });
    });
}

criterion_group!(benches, bench_permutation_entropy, bench_order_book_update);
criterion_main!(benches);

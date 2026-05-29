//! Hawkes Process Trade Intensity Feature Extraction
//!
//! Models trade arrival as a self-exciting point process where each trade
//! temporarily increases the probability of subsequent trades (clustering).
//!
//! # Features (3 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Hawkes intensity** | λ(t) = μ + Σ α·exp(-β·(t-tᵢ)) | [0, +inf) | Higher = more clustered arrivals |
//! | **Hawkes baseline** | μ = long-term average trade rate | [0, +inf) | Trades/sec without clustering |
//! | **Branching ratio** | (λ - μ) / max(μ, ε) | [0, +inf) | >1 = self-sustaining cascade |
//!
//! # Algorithm
//!
//! Fixed-parameter Hawkes with α=0.5/sec, β=1.0/sec (standard HFT calibration).
//! Baseline μ estimated from 30s trade count. Intensity computed over all trades
//! in the 30s window with exponential decay kernel.
//!
//! # References
//!
//! - Bacry, Mastromatteo & Muzy (2015) — Hawkes processes in finance

use ing_types::TradeBuffer;

/// Hawkes process parameters (fixed calibration)
const ALPHA: f64 = 0.5; // Excitation amplitude per trade (per second)
const BETA: f64 = 1.0; // Decay rate (per second)
const WINDOW_SECONDS: u64 = 30;

/// Hawkes trade intensity features (3 features)
#[derive(Debug, Clone, Default)]
pub struct HawkesFeatures {
    /// Current instantaneous intensity λ(t)
    pub hawkes_intensity: f64,
    /// Baseline rate μ (trades per second)
    pub hawkes_baseline: f64,
    /// Branching ratio: (λ - μ) / μ — excitation above baseline
    pub branching_ratio: f64,
}

impl HawkesFeatures {
    pub fn count() -> usize {
        3
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "hawkes_intensity",
            "hawkes_baseline",
            "hawkes_branching_ratio",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.hawkes_intensity,
            self.hawkes_baseline,
            self.branching_ratio,
        ]
    }
}

/// Compute Hawkes intensity features from trade buffer
pub fn compute(trade_buffer: &TradeBuffer) -> HawkesFeatures {
    let trades = trade_buffer.trades_in_window(WINDOW_SECONDS);

    if trades.len() < 2 {
        return HawkesFeatures::default();
    }

    // Baseline: trades per second over the window
    let mu = trades.len() as f64 / WINDOW_SECONDS as f64;

    // Current time = latest trade timestamp
    let now = trades.last().map(|t| t.timestamp).unwrap_or(0);

    // Compute self-exciting component: Σ α·exp(-β·(t_now - t_i))
    // Timestamps are in milliseconds, convert to seconds for decay
    let excitation: f64 = trades
        .iter()
        .map(|trade| {
            let dt_sec = (now - trade.timestamp) as f64 / 1000.0;
            ALPHA * (-BETA * dt_sec).exp()
        })
        .sum();

    let lambda = mu + excitation;

    let branching_ratio = if mu > 1e-9 { excitation / mu } else { 0.0 };

    HawkesFeatures {
        hawkes_intensity: lambda,
        hawkes_baseline: mu,
        branching_ratio,
    }
}

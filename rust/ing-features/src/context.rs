//! Market Context Feature Extraction
//!
//! Hyperliquid-specific market metadata from the activeAssetCtx WebSocket channel.
//! These features capture macro-level conditions (funding, open interest, premium)
//! that drive regime transitions and inform position sizing.
//!
//! # Features (12 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Funding rate** | Current perp funding rate | (-inf, +inf) | >0 = longs pay shorts |
//! | **Funding z-score** | Funding vs historical distribution | (-inf, +inf) | |z| > 2 = extreme funding |
//! | **Open interest** | Total open contracts (USD) | [0, +inf) | Higher = more leveraged |
//! | **OI change (5m)** | Absolute OI change over ~5 min | (-inf, +inf) | >0 = positions opening |
//! | **OI change % (5m)** | OI percent change over ~5 min | (-inf, +inf) | Normalized by OI level |
//! | **Premium (bps)** | Mark price vs index price premium | (-inf, +inf) | >0 = perp trades above spot |
//! | **Volume 24h** | Rolling 24-hour traded volume | [0, +inf) | Absolute liquidity measure |
//! | **Volume ratio** | Current vs average 24h volume | [0, +inf) | >1 = above-average activity |
//! | **Mark-oracle divergence** | Mark price minus oracle price | (-inf, +inf) | Measures pricing dislocation |
//! | **Funding momentum 8h** | funding(t) - funding(t-8h) | (-inf, +inf) | Mean-reversion timing signal |
//! | **Funding acceleration** | d(momentum)/dt over 1h | (-inf, +inf) | Second derivative of funding |
//! | **OI momentum 1h** | OI % change over available buffer | (-inf, +inf) | Position flow momentum |
//!
//! Note: OI change uses a 60-sample lookback (~5 min at ~5s context updates).
//! Funding momentum uses 8h lookback (Hyperliquid settlement cycle).

use ing_types::MarketContext;

/// Context features (12 features)
#[derive(Debug, Clone, Default)]
pub struct ContextFeatures {
    /// Current funding rate
    pub funding_rate: f64,
    /// Funding rate z-score
    pub funding_zscore: f64,
    /// Total open interest
    pub open_interest: f64,
    /// OI change over 5 minutes
    pub oi_change_5m: f64,
    /// OI percent change over 5 minutes
    pub oi_change_pct_5m: f64,
    /// Premium in basis points
    pub premium_bps: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// Volume ratio vs average
    pub volume_ratio: f64,
    /// Mark-oracle divergence
    pub mark_oracle_divergence: f64,
    /// Funding rate momentum: funding(t) - funding(t - 8h)
    pub funding_momentum_8h: f64,
    /// Funding acceleration: d(momentum)/dt
    pub funding_acceleration: f64,
    /// OI momentum over available buffer (~5min)
    pub oi_momentum_1h: f64,
}

impl ContextFeatures {
    pub fn count() -> usize { 12 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "ctx_funding_rate",
            "ctx_funding_zscore",
            "ctx_open_interest",
            "ctx_oi_change_5m",
            "ctx_oi_change_pct_5m",
            "ctx_premium_bps",
            "ctx_volume_24h",
            "ctx_volume_ratio",
            "ctx_mark_oracle_divergence",
            "ctx_funding_momentum_8h",
            "ctx_funding_acceleration",
            "ctx_oi_momentum_1h",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.funding_rate,
            self.funding_zscore,
            self.open_interest,
            self.oi_change_5m,
            self.oi_change_pct_5m,
            self.premium_bps,
            self.volume_24h,
            self.volume_ratio,
            self.mark_oracle_divergence,
            self.funding_momentum_8h,
            self.funding_acceleration,
            self.oi_momentum_1h,
        ]
    }
}

/// Compute context features from market context
pub fn compute(market_context: &MarketContext) -> ContextFeatures {
    ContextFeatures {
        funding_rate: market_context.funding_rate(),
        funding_zscore: market_context.funding_zscore(),
        open_interest: market_context.open_interest(),
        oi_change_5m: market_context.oi_change(60),  // ~5min at 5s updates
        oi_change_pct_5m: market_context.oi_change_pct(60),
        premium_bps: market_context.premium_bps(),
        volume_24h: market_context.volume_24h(),
        volume_ratio: market_context.volume_ratio(),
        mark_oracle_divergence: market_context.mark_oracle_divergence(),
        funding_momentum_8h: market_context.funding_momentum_8h(),
        funding_acceleration: market_context.funding_acceleration(),
        oi_momentum_1h: market_context.oi_momentum_1h(),
    }
}

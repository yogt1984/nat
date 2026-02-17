//! Market context features

use crate::state::MarketContext;

/// Context features (9 features)
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
}

impl ContextFeatures {
    pub fn count() -> usize { 9 }

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
    }
}

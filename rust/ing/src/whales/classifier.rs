//! Whale classification logic
//!
//! Defines thresholds and rules for identifying whale wallets.

use serde::{Deserialize, Serialize};

/// Configuration for whale classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleConfig {
    /// Minimum position value (USD) to be considered a whale
    pub min_position_usd: f64,
    /// Minimum 30-day volume (USD) to be considered a whale
    pub min_volume_30d_usd: f64,
    /// Minimum number of trades to be considered active
    pub min_trades_30d: usize,
    /// Number of top whales to track for concentration
    pub top_whale_count: usize,
}

impl Default for WhaleConfig {
    fn default() -> Self {
        Self {
            min_position_usd: 500_000.0,      // $500K position
            min_volume_30d_usd: 10_000_000.0, // $10M 30-day volume
            min_trades_30d: 10,                // At least 10 trades
            top_whale_count: 10,               // Track top 10 for concentration
        }
    }
}

/// Classification tier for a wallet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WhaleTier {
    /// Not a whale (below thresholds)
    Retail,
    /// Small whale: $500K-$2M position or $10M-$50M volume
    SmallWhale,
    /// Medium whale: $2M-$10M position or $50M-$200M volume
    MediumWhale,
    /// Large whale: $10M+ position or $200M+ volume
    LargeWhale,
    /// Market maker: high volume, balanced long/short, low PnL variance
    MarketMaker,
}

impl WhaleTier {
    pub fn is_whale(&self) -> bool {
        !matches!(self, WhaleTier::Retail)
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            WhaleTier::Retail => "Retail",
            WhaleTier::SmallWhale => "Small Whale",
            WhaleTier::MediumWhale => "Medium Whale",
            WhaleTier::LargeWhale => "Large Whale",
            WhaleTier::MarketMaker => "Market Maker",
        }
    }
}

/// Wallet statistics used for classification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalletStats {
    /// Wallet address
    pub address: String,
    /// Current total position value (USD)
    pub current_position_usd: f64,
    /// Maximum position value seen (USD)
    pub max_position_usd: f64,
    /// 30-day trading volume (USD)
    pub volume_30d_usd: f64,
    /// Number of trades in last 30 days
    pub trades_30d: usize,
    /// Total realized PnL (USD)
    pub realized_pnl: f64,
    /// Current unrealized PnL (USD)
    pub unrealized_pnl: f64,
    /// Win rate (profitable trades / total trades)
    pub win_rate: f64,
    /// Long/short ratio (positive = net long, negative = net short)
    pub long_short_ratio: f64,
    /// First seen timestamp (ms)
    pub first_seen_ms: i64,
    /// Last active timestamp (ms)
    pub last_active_ms: i64,
    /// Number of unique symbols traded
    pub symbols_traded: usize,
}

impl WalletStats {
    /// Classify this wallet based on thresholds
    pub fn classify(&self, config: &WhaleConfig) -> WhaleTier {
        // Check if market maker (high volume, balanced positions, frequent trading)
        if self.is_likely_market_maker(config) {
            return WhaleTier::MarketMaker;
        }

        // Classify by position size
        let position_tier = self.classify_by_position();

        // Classify by volume
        let volume_tier = self.classify_by_volume(config);

        // Take the higher tier
        match (position_tier, volume_tier) {
            (WhaleTier::LargeWhale, _) | (_, WhaleTier::LargeWhale) => WhaleTier::LargeWhale,
            (WhaleTier::MediumWhale, _) | (_, WhaleTier::MediumWhale) => WhaleTier::MediumWhale,
            (WhaleTier::SmallWhale, _) | (_, WhaleTier::SmallWhale) => WhaleTier::SmallWhale,
            _ => WhaleTier::Retail,
        }
    }

    fn classify_by_position(&self) -> WhaleTier {
        let pos = self.max_position_usd;
        if pos >= 10_000_000.0 {
            WhaleTier::LargeWhale
        } else if pos >= 2_000_000.0 {
            WhaleTier::MediumWhale
        } else if pos >= 500_000.0 {
            WhaleTier::SmallWhale
        } else {
            WhaleTier::Retail
        }
    }

    fn classify_by_volume(&self, config: &WhaleConfig) -> WhaleTier {
        let vol = self.volume_30d_usd;
        if vol >= 200_000_000.0 {
            WhaleTier::LargeWhale
        } else if vol >= 50_000_000.0 {
            WhaleTier::MediumWhale
        } else if vol >= config.min_volume_30d_usd {
            WhaleTier::SmallWhale
        } else {
            WhaleTier::Retail
        }
    }

    fn is_likely_market_maker(&self, config: &WhaleConfig) -> bool {
        // Market makers typically have:
        // 1. Very high volume relative to position
        // 2. Balanced long/short (ratio close to 0)
        // 3. Many trades
        // 4. Low PnL variance (making money from spread, not direction)

        let high_volume = self.volume_30d_usd >= config.min_volume_30d_usd * 5.0;
        let balanced = self.long_short_ratio.abs() < 0.3;
        let frequent_trader = self.trades_30d >= 100;
        let volume_to_position = if self.max_position_usd > 0.0 {
            self.volume_30d_usd / self.max_position_usd
        } else {
            0.0
        };
        let high_turnover = volume_to_position > 50.0; // Turns over position 50x/month

        high_volume && balanced && frequent_trader && high_turnover
    }

    /// Check if this wallet qualifies as a whale under given config
    pub fn is_whale(&self, config: &WhaleConfig) -> bool {
        self.max_position_usd >= config.min_position_usd
            || self.volume_30d_usd >= config.min_volume_30d_usd
    }

    /// Calculate a "whale score" for ranking (0-100)
    pub fn whale_score(&self, config: &WhaleConfig) -> f64 {
        let position_score = (self.max_position_usd / config.min_position_usd).min(10.0) * 10.0;
        let volume_score = (self.volume_30d_usd / config.min_volume_30d_usd).min(10.0) * 10.0;

        // Weighted average: position matters more for "whale" status
        (position_score * 0.6 + volume_score * 0.4).min(100.0)
    }
}

/// Result of analyzing a wallet for whale status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleClassification {
    pub address: String,
    pub tier: WhaleTier,
    pub whale_score: f64,
    pub stats: WalletStats,
    pub classified_at_ms: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whale_classification_by_position() {
        let config = WhaleConfig::default();

        let mut stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 600_000.0, // $600K
            ..Default::default()
        };

        assert_eq!(stats.classify(&config), WhaleTier::SmallWhale);

        stats.max_position_usd = 3_000_000.0; // $3M
        assert_eq!(stats.classify(&config), WhaleTier::MediumWhale);

        stats.max_position_usd = 15_000_000.0; // $15M
        assert_eq!(stats.classify(&config), WhaleTier::LargeWhale);
    }

    #[test]
    fn test_whale_classification_by_volume() {
        let config = WhaleConfig::default();

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 100_000.0, // Below position threshold
            volume_30d_usd: 15_000_000.0, // Above volume threshold
            ..Default::default()
        };

        assert_eq!(stats.classify(&config), WhaleTier::SmallWhale);
    }

    #[test]
    fn test_retail_classification() {
        let config = WhaleConfig::default();

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 50_000.0,   // $50K - below threshold
            volume_30d_usd: 500_000.0,    // $500K - below threshold
            ..Default::default()
        };

        assert_eq!(stats.classify(&config), WhaleTier::Retail);
        assert!(!stats.is_whale(&config));
    }

    #[test]
    fn test_market_maker_detection() {
        let config = WhaleConfig::default();

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 1_000_000.0,       // $1M position
            volume_30d_usd: 100_000_000.0,       // $100M volume (100x position)
            trades_30d: 500,                      // Very active
            long_short_ratio: 0.1,               // Nearly balanced
            ..Default::default()
        };

        assert_eq!(stats.classify(&config), WhaleTier::MarketMaker);
    }

    #[test]
    fn test_whale_score() {
        let config = WhaleConfig::default();

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 1_000_000.0,  // 2x threshold
            volume_30d_usd: 20_000_000.0,   // 2x threshold
            ..Default::default()
        };

        let score = stats.whale_score(&config);
        assert!(score > 0.0 && score <= 100.0);
    }
}

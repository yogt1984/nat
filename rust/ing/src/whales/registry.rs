//! Whale registry
//!
//! Persistent storage and management of whale wallet data.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::{WhaleClassification, WhaleConfig, WalletStats, WhaleTier};

/// Registry of whale wallets
#[derive(Debug)]
pub struct WhaleRegistry {
    config: WhaleConfig,
    /// All known whales by address
    whales: Arc<RwLock<HashMap<String, WhaleClassification>>>,
    /// Historical snapshots of whale count by tier
    history: Arc<RwLock<Vec<WhaleSnapshot>>>,
    /// Last update timestamp
    last_update_ms: Arc<RwLock<i64>>,
}

/// Snapshot of whale population at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleSnapshot {
    pub timestamp_ms: i64,
    pub total_whales: usize,
    pub small_whales: usize,
    pub medium_whales: usize,
    pub large_whales: usize,
    pub market_makers: usize,
    pub total_whale_position_usd: f64,
    pub concentration_top10: f64,
}

impl WhaleRegistry {
    /// Create a new whale registry
    pub fn new(config: WhaleConfig) -> Self {
        Self {
            config,
            whales: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            last_update_ms: Arc::new(RwLock::new(0)),
        }
    }

    /// Update or add a whale to the registry
    pub fn update_whale(&self, stats: WalletStats) {
        let tier = stats.classify(&self.config);

        if tier == WhaleTier::Retail {
            // Remove from registry if they've fallen below threshold
            let mut whales = self.whales.write();
            if whales.remove(&stats.address).is_some() {
                debug!(address = %stats.address, "Wallet demoted from whale status");
            }
            return;
        }

        let classification = WhaleClassification {
            address: stats.address.clone(),
            tier,
            whale_score: stats.whale_score(&self.config),
            stats,
            classified_at_ms: Utc::now().timestamp_millis(),
        };

        let mut whales = self.whales.write();
        whales.insert(classification.address.clone(), classification);
    }

    /// Bulk update from wallet stats
    pub fn update_from_stats(&self, all_stats: &[WalletStats]) {
        for stats in all_stats {
            self.update_whale(stats.clone());
        }

        *self.last_update_ms.write() = Utc::now().timestamp_millis();

        let whale_count = self.whales.read().len();
        info!(whale_count, "Whale registry updated");
    }

    /// Get all whales
    pub fn get_all_whales(&self) -> Vec<WhaleClassification> {
        self.whales.read().values().cloned().collect()
    }

    /// Get whales by tier
    pub fn get_whales_by_tier(&self, tier: WhaleTier) -> Vec<WhaleClassification> {
        self.whales.read()
            .values()
            .filter(|w| w.tier == tier)
            .cloned()
            .collect()
    }

    /// Get top N whales by score
    pub fn get_top_whales(&self, n: usize) -> Vec<WhaleClassification> {
        let mut whales: Vec<_> = self.whales.read().values().cloned().collect();
        whales.sort_by(|a, b| b.whale_score.partial_cmp(&a.whale_score).unwrap());
        whales.truncate(n);
        whales
    }

    /// Get whale by address
    pub fn get_whale(&self, address: &str) -> Option<WhaleClassification> {
        self.whales.read().get(address).cloned()
    }

    /// Check if an address is a known whale
    pub fn is_whale(&self, address: &str) -> bool {
        self.whales.read().contains_key(address)
    }

    /// Get whale tier for an address (returns Retail if not found)
    pub fn get_tier(&self, address: &str) -> WhaleTier {
        self.whales.read()
            .get(address)
            .map(|w| w.tier)
            .unwrap_or(WhaleTier::Retail)
    }

    /// Get total whale count
    pub fn whale_count(&self) -> usize {
        self.whales.read().len()
    }

    /// Get count by tier
    pub fn count_by_tier(&self) -> HashMap<WhaleTier, usize> {
        let whales = self.whales.read();
        let mut counts = HashMap::new();

        for whale in whales.values() {
            *counts.entry(whale.tier).or_insert(0) += 1;
        }

        counts
    }

    /// Take a snapshot of current whale population
    pub fn take_snapshot(&self, total_oi: f64) -> WhaleSnapshot {
        let whales = self.whales.read();

        let mut small_whales = 0;
        let mut medium_whales = 0;
        let mut large_whales = 0;
        let mut market_makers = 0;
        let mut total_position = 0.0;

        // Collect positions for concentration calculation
        let mut positions: Vec<f64> = Vec::new();

        for whale in whales.values() {
            match whale.tier {
                WhaleTier::SmallWhale => small_whales += 1,
                WhaleTier::MediumWhale => medium_whales += 1,
                WhaleTier::LargeWhale => large_whales += 1,
                WhaleTier::MarketMaker => market_makers += 1,
                WhaleTier::Retail => {}
            }

            let pos = whale.stats.current_position_usd.abs();
            total_position += pos;
            positions.push(pos);
        }

        // Calculate top 10 concentration
        positions.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top10_position: f64 = positions.iter().take(10).sum();
        let concentration = if total_oi > 0.0 {
            top10_position / total_oi
        } else if total_position > 0.0 {
            top10_position / total_position
        } else {
            0.0
        };

        let snapshot = WhaleSnapshot {
            timestamp_ms: Utc::now().timestamp_millis(),
            total_whales: whales.len(),
            small_whales,
            medium_whales,
            large_whales,
            market_makers,
            total_whale_position_usd: total_position,
            concentration_top10: concentration,
        };

        // Store in history
        self.history.write().push(snapshot.clone());

        snapshot
    }

    /// Get snapshot history
    pub fn get_history(&self) -> Vec<WhaleSnapshot> {
        self.history.read().clone()
    }

    /// Get whale addresses (for position tracking)
    pub fn get_whale_addresses(&self) -> Vec<String> {
        self.whales.read().keys().cloned().collect()
    }

    /// Get last update time
    pub fn last_update_ms(&self) -> i64 {
        *self.last_update_ms.read()
    }

    /// Export registry to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let whales = self.get_all_whales();
        serde_json::to_string_pretty(&whales)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_update() {
        let config = WhaleConfig::default();
        let registry = WhaleRegistry::new(config);

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 1_000_000.0,
            current_position_usd: 800_000.0,
            ..Default::default()
        };

        registry.update_whale(stats);

        assert_eq!(registry.whale_count(), 1);
        assert!(registry.is_whale("0x123"));
    }

    #[test]
    fn test_registry_demotion() {
        let config = WhaleConfig::default();
        let registry = WhaleRegistry::new(config.clone());

        // Add as whale
        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 1_000_000.0,
            ..Default::default()
        };
        registry.update_whale(stats);
        assert!(registry.is_whale("0x123"));

        // Update with lower position - should be removed
        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 100_000.0,
            ..Default::default()
        };
        registry.update_whale(stats);
        assert!(!registry.is_whale("0x123"));
    }

    #[test]
    fn test_top_whales() {
        let config = WhaleConfig::default();
        let registry = WhaleRegistry::new(config);

        // Add several whales with different scores
        for i in 0..5 {
            let stats = WalletStats {
                address: format!("0x{}", i),
                max_position_usd: (i as f64 + 1.0) * 1_000_000.0,
                ..Default::default()
            };
            registry.update_whale(stats);
        }

        let top3 = registry.get_top_whales(3);
        assert_eq!(top3.len(), 3);
        // Highest position should be first
        assert_eq!(top3[0].address, "0x4");
    }

    #[test]
    fn test_snapshot() {
        let config = WhaleConfig::default();
        let registry = WhaleRegistry::new(config);

        let stats = WalletStats {
            address: "0x123".to_string(),
            max_position_usd: 1_000_000.0,
            current_position_usd: 800_000.0,
            ..Default::default()
        };
        registry.update_whale(stats);

        let snapshot = registry.take_snapshot(10_000_000.0);
        assert_eq!(snapshot.total_whales, 1);
        assert!(snapshot.concentration_top10 > 0.0);
    }
}

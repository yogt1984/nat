//! Wallet discovery from trade stream
//!
//! Accumulates wallet addresses seen in WsTrade.users, then periodically
//! promotes the most active ones to the PositionTracker.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::info;

/// Accumulates wallet addresses from trade stream and promotes top-N to tracking.
pub struct WalletDiscovery {
    /// Wallet address -> trade count
    counts: HashMap<String, usize>,
    /// Already-promoted wallets (avoid re-adding)
    promoted: HashSet<String>,
    /// Maximum wallets the tracker should hold
    max_wallets: usize,
    /// Whether we've logged the WsTrade.users availability status
    users_status_logged: bool,
}

impl WalletDiscovery {
    pub fn new(max_wallets: usize) -> Self {
        Self {
            counts: HashMap::new(),
            promoted: HashSet::new(),
            max_wallets,
            users_status_logged: false,
        }
    }

    /// Observe a trade with maker/taker addresses.
    /// Returns true on the first call (for diagnostic logging).
    pub fn observe_trade(&mut self, maker: &str, taker: &str) -> bool {
        *self.counts.entry(maker.to_string()).or_insert(0) += 1;
        *self.counts.entry(taker.to_string()).or_insert(0) += 1;

        if !self.users_status_logged {
            self.users_status_logged = true;
            return true; // signal: first observation
        }
        false
    }

    /// Returns top-N wallets not yet promoted, sorted by trade count descending.
    /// Also adds them to the promoted set.
    pub fn promote_top(&mut self, n: usize, tracker_wallets: &Arc<RwLock<Vec<String>>>) -> usize {
        let current_count = tracker_wallets.read().len();
        let budget = self.max_wallets.saturating_sub(current_count);
        let take = n.min(budget);

        if take == 0 {
            return 0;
        }

        let mut candidates: Vec<(&String, &usize)> = self
            .counts
            .iter()
            .filter(|(w, _)| !self.promoted.contains(*w))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(a.1));

        let new_wallets: Vec<String> = candidates
            .into_iter()
            .take(take)
            .map(|(w, _)| w.clone())
            .collect();

        if new_wallets.is_empty() {
            return 0;
        }

        let count = new_wallets.len();

        // Add to tracker
        {
            let mut w = tracker_wallets.write();
            for wallet in &new_wallets {
                if !w.contains(wallet) {
                    w.push(wallet.clone());
                }
            }
        }

        // Mark as promoted
        for w in &new_wallets {
            self.promoted.insert(w.clone());
        }

        info!(
            new = count,
            total_tracked = tracker_wallets.read().len(),
            total_seen = self.counts.len(),
            "Discovered and promoted whale wallets"
        );

        count
    }

    /// Number of unique wallets observed so far.
    pub fn unique_wallets_seen(&self) -> usize {
        self.counts.len()
    }

    /// Whether the users field has been observed at least once.
    pub fn users_field_available(&self) -> bool {
        self.users_status_logged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_and_promote() {
        let mut discovery = WalletDiscovery::new(10);

        // Observe trades
        assert!(discovery.observe_trade("0xAAA", "0xBBB")); // first = true
        assert!(!discovery.observe_trade("0xAAA", "0xCCC")); // subsequent = false
        assert!(!discovery.observe_trade("0xAAA", "0xDDD"));
        assert!(!discovery.observe_trade("0xBBB", "0xCCC"));

        assert_eq!(discovery.unique_wallets_seen(), 4);

        // Promote top 2
        let wallets = Arc::new(RwLock::new(Vec::new()));
        let promoted = discovery.promote_top(2, &wallets);
        assert_eq!(promoted, 2);
        assert_eq!(wallets.read().len(), 2);

        // 0xAAA has 3 trades, should be first
        assert!(wallets.read().contains(&"0xAAA".to_string()));

        // Promote again — same wallets shouldn't be re-added
        let promoted = discovery.promote_top(2, &wallets);
        assert_eq!(promoted, 2); // 0xCCC and 0xDDD
        assert_eq!(wallets.read().len(), 4);
    }

    #[test]
    fn test_max_wallets_cap() {
        let mut discovery = WalletDiscovery::new(3);

        for i in 0..10 {
            discovery.observe_trade(&format!("0x{:03}", i), &format!("0x{:03}", i + 100));
        }

        let wallets = Arc::new(RwLock::new(vec!["0xEXISTING".to_string()]));
        // Budget = 3 - 1 = 2
        let promoted = discovery.promote_top(5, &wallets);
        assert_eq!(promoted, 2);
        assert_eq!(wallets.read().len(), 3);

        // No more budget
        let promoted = discovery.promote_top(5, &wallets);
        assert_eq!(promoted, 0);
    }

    #[test]
    fn test_empty_discovery() {
        let mut discovery = WalletDiscovery::new(10);
        let wallets = Arc::new(RwLock::new(Vec::new()));
        assert_eq!(discovery.promote_top(5, &wallets), 0);
        assert!(!discovery.users_field_available());
    }
}

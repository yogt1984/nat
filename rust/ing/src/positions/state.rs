//! Shared position state
//!
//! Thread-safe bridge between the PositionTracker task and per-symbol MarketState.
//! The tracker writes snapshots/deltas; MarketState reads them during feature computation.

use std::collections::{HashMap, VecDeque};

use parking_lot::RwLock;

use super::snapshot::{PositionDelta, PositionSnapshot};
use ing_features::{ConcentrationPosition, LiquidationPosition};

/// Thread-safe shared state for position data across symbols.
pub struct SharedPositionState {
    /// Latest snapshots keyed by symbol -> vec of wallet snapshots
    snapshots: RwLock<HashMap<String, Vec<PositionSnapshot>>>,
    /// Recent deltas not yet consumed, keyed by symbol
    deltas: RwLock<HashMap<String, VecDeque<PositionDelta>>>,
}

impl SharedPositionState {
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
            deltas: RwLock::new(HashMap::new()),
        }
    }

    /// Update with a new position snapshot (replaces previous for same wallet+symbol).
    pub fn update_snapshot(&self, snapshot: PositionSnapshot) {
        let mut map = self.snapshots.write();
        let entry = map.entry(snapshot.symbol.clone()).or_default();

        // Replace existing snapshot for same wallet, or append
        if let Some(existing) = entry.iter_mut().find(|s| s.wallet == snapshot.wallet) {
            *existing = snapshot;
        } else {
            entry.push(snapshot);
        }
    }

    /// Update with a batch of snapshots for a single poll cycle.
    pub fn update_snapshots(&self, snapshots: Vec<PositionSnapshot>) {
        for s in snapshots {
            self.update_snapshot(s);
        }
    }

    /// Add a position delta.
    pub fn update_delta(&self, delta: PositionDelta) {
        let mut map = self.deltas.write();
        map.entry(delta.symbol.clone())
            .or_default()
            .push_back(delta);
    }

    /// Add a batch of deltas.
    pub fn update_deltas(&self, deltas: Vec<PositionDelta>) {
        for d in deltas {
            self.update_delta(d);
        }
    }

    /// Drain all pending deltas for a symbol. Returns and clears.
    pub fn drain_deltas(&self, symbol: &str) -> Vec<PositionDelta> {
        let mut map = self.deltas.write();
        match map.get_mut(symbol) {
            Some(deque) => deque.drain(..).collect(),
            None => Vec::new(),
        }
    }

    /// Convert current snapshots for a symbol into LiquidationPosition structs.
    pub fn get_liquidation_positions(&self, symbol: &str) -> Vec<LiquidationPosition> {
        let map = self.snapshots.read();
        match map.get(symbol) {
            Some(snapshots) => snapshots
                .iter()
                .filter_map(|s| {
                    s.liquidation_price.map(|liq_px| LiquidationPosition {
                        position_value_usd: s.position_value.abs(),
                        liquidation_price: liq_px,
                        is_long: s.size > 0.0,
                        entry_price: s.entry_price,
                    })
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Convert current snapshots for a symbol into concentration Position structs.
    /// Positions above `whale_threshold_usd` are classified as whale.
    pub fn get_concentration_positions(
        &self,
        symbol: &str,
        whale_threshold_usd: f64,
    ) -> Vec<ConcentrationPosition> {
        let map = self.snapshots.read();
        match map.get(symbol) {
            Some(snapshots) => snapshots
                .iter()
                .filter(|s| s.position_value.abs() > 0.0)
                .map(|s| ConcentrationPosition {
                    value_usd: s.position_value.abs(),
                    is_whale: s.position_value.abs() >= whale_threshold_usd,
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Number of tracked snapshots for a symbol.
    pub fn snapshot_count(&self, symbol: &str) -> usize {
        self.snapshots
            .read()
            .get(symbol)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Total number of tracked wallets across all symbols.
    pub fn total_wallet_count(&self) -> usize {
        self.snapshots.read().values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(
        wallet: &str,
        symbol: &str,
        size: f64,
        entry: f64,
        liq: Option<f64>,
    ) -> PositionSnapshot {
        PositionSnapshot {
            timestamp_ms: 1000,
            wallet: wallet.to_string(),
            symbol: symbol.to_string(),
            size,
            entry_price: entry,
            liquidation_price: liq,
            unrealized_pnl: 0.0,
            position_value: size * entry,
            leverage: 10.0,
            is_cross_margin: true,
        }
    }

    fn make_delta(wallet: &str, symbol: &str, prev: f64, curr: f64) -> PositionDelta {
        use super::super::snapshot::PositionChangeType;
        PositionDelta {
            timestamp_ms: 2000,
            wallet: wallet.to_string(),
            symbol: symbol.to_string(),
            prev_size: prev,
            curr_size: curr,
            size_delta: curr - prev,
            prev_entry_price: 50000.0,
            curr_entry_price: 50000.0,
            pnl_delta: 0.0,
            change_type: if curr > prev {
                PositionChangeType::IncreaseLong
            } else {
                PositionChangeType::DecreaseLong
            },
        }
    }

    #[test]
    fn test_snapshot_update_and_replace() {
        let state = SharedPositionState::new();

        state.update_snapshot(make_snapshot("0x1", "BTC", 1.0, 50000.0, Some(45000.0)));
        assert_eq!(state.snapshot_count("BTC"), 1);

        // Same wallet replaces
        state.update_snapshot(make_snapshot("0x1", "BTC", 2.0, 51000.0, Some(46000.0)));
        assert_eq!(state.snapshot_count("BTC"), 1);

        // Different wallet appends
        state.update_snapshot(make_snapshot("0x2", "BTC", 0.5, 50000.0, Some(44000.0)));
        assert_eq!(state.snapshot_count("BTC"), 2);
    }

    #[test]
    fn test_drain_deltas() {
        let state = SharedPositionState::new();

        state.update_delta(make_delta("0x1", "BTC", 1.0, 2.0));
        state.update_delta(make_delta("0x2", "BTC", 0.0, 1.0));
        state.update_delta(make_delta("0x3", "ETH", 5.0, 3.0));

        let btc_deltas = state.drain_deltas("BTC");
        assert_eq!(btc_deltas.len(), 2);

        // Drained — second call returns empty
        let btc_deltas2 = state.drain_deltas("BTC");
        assert!(btc_deltas2.is_empty());

        // ETH untouched
        let eth_deltas = state.drain_deltas("ETH");
        assert_eq!(eth_deltas.len(), 1);
    }

    #[test]
    fn test_get_liquidation_positions() {
        let state = SharedPositionState::new();

        // Long position with liquidation price
        state.update_snapshot(make_snapshot("0x1", "BTC", 1.0, 50000.0, Some(45000.0)));
        // Short position with liquidation price
        state.update_snapshot(make_snapshot("0x2", "BTC", -0.5, 50000.0, Some(55000.0)));
        // Position without liquidation price (cross margin edge case)
        state.update_snapshot(make_snapshot("0x3", "BTC", 0.3, 50000.0, None));

        let liq_positions = state.get_liquidation_positions("BTC");
        assert_eq!(liq_positions.len(), 2, "Should skip None liquidation_price");
        assert!(liq_positions[0].is_long);
        assert!(!liq_positions[1].is_long);
        assert!((liq_positions[0].liquidation_price - 45000.0).abs() < 0.01);
    }

    #[test]
    fn test_get_concentration_positions() {
        let state = SharedPositionState::new();

        // Whale: $50K position
        state.update_snapshot(make_snapshot("0x1", "BTC", 1.0, 50000.0, Some(45000.0)));
        // Retail: $5K position
        state.update_snapshot(make_snapshot("0x2", "BTC", 0.1, 50000.0, Some(44000.0)));

        let conc = state.get_concentration_positions("BTC", 10_000.0);
        assert_eq!(conc.len(), 2);

        let whale = conc.iter().find(|p| p.is_whale).unwrap();
        assert!((whale.value_usd - 50000.0).abs() < 0.01);

        let retail = conc.iter().find(|p| !p.is_whale).unwrap();
        assert!((retail.value_usd - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_symbol_returns_empty() {
        let state = SharedPositionState::new();
        assert!(state.get_liquidation_positions("BTC").is_empty());
        assert!(state.get_concentration_positions("BTC", 10_000.0).is_empty());
        assert!(state.drain_deltas("BTC").is_empty());
        assert_eq!(state.snapshot_count("BTC"), 0);
    }

    #[test]
    fn test_total_wallet_count() {
        let state = SharedPositionState::new();
        assert_eq!(state.total_wallet_count(), 0);

        state.update_snapshot(make_snapshot("0x1", "BTC", 1.0, 50000.0, None));
        state.update_snapshot(make_snapshot("0x2", "ETH", 5.0, 3000.0, None));
        assert_eq!(state.total_wallet_count(), 2);
    }
}

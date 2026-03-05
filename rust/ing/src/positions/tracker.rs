//! Position tracker
//!
//! Polls Hyperliquid API to track positions for a set of wallets.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use futures_util::future::join_all;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::rest::HyperliquidRestClient;
use super::snapshot::{PositionSnapshot, PositionDelta, WalletPositionStats};

/// Configuration for position tracking
#[derive(Debug, Clone)]
pub struct PositionTrackerConfig {
    /// Polling interval in seconds
    pub poll_interval_secs: u64,
    /// Symbols to track (empty = all)
    pub symbols: Vec<String>,
    /// Maximum concurrent API requests
    pub max_concurrent_requests: usize,
}

impl Default for PositionTrackerConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 60,
            symbols: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
            max_concurrent_requests: 10,
        }
    }
}

/// Position tracker that polls wallets for position updates
pub struct PositionTracker {
    config: PositionTrackerConfig,
    client: HyperliquidRestClient,
    /// Set of wallet addresses to track
    wallets: Arc<RwLock<Vec<String>>>,
    /// Last known positions per wallet
    last_positions: Arc<RwLock<HashMap<String, HashMap<String, PositionSnapshot>>>>,
    /// Channel to send position updates
    snapshot_tx: Option<mpsc::Sender<PositionSnapshot>>,
    /// Channel to send position deltas
    delta_tx: Option<mpsc::Sender<PositionDelta>>,
}

impl PositionTracker {
    /// Create a new position tracker
    pub fn new(config: PositionTrackerConfig) -> Result<Self> {
        let client = HyperliquidRestClient::new()?;

        Ok(Self {
            config,
            client,
            wallets: Arc::new(RwLock::new(Vec::new())),
            last_positions: Arc::new(RwLock::new(HashMap::new())),
            snapshot_tx: None,
            delta_tx: None,
        })
    }

    /// Set channel for position snapshots
    pub fn with_snapshot_channel(mut self, tx: mpsc::Sender<PositionSnapshot>) -> Self {
        self.snapshot_tx = Some(tx);
        self
    }

    /// Set channel for position deltas
    pub fn with_delta_channel(mut self, tx: mpsc::Sender<PositionDelta>) -> Self {
        self.delta_tx = Some(tx);
        self
    }

    /// Add wallets to track
    pub fn add_wallets(&self, wallets: &[String]) {
        let mut w = self.wallets.write();
        for wallet in wallets {
            if !w.contains(wallet) {
                w.push(wallet.clone());
            }
        }
        info!(count = w.len(), "Updated tracked wallets");
    }

    /// Remove wallets from tracking
    pub fn remove_wallets(&self, wallets: &[String]) {
        let mut w = self.wallets.write();
        w.retain(|addr| !wallets.contains(addr));
    }

    /// Get list of tracked wallets
    pub fn get_wallets(&self) -> Vec<String> {
        self.wallets.read().clone()
    }

    /// Poll positions for all tracked wallets
    pub async fn poll_all(&self) -> Result<Vec<PositionSnapshot>> {
        let wallets = self.wallets.read().clone();
        let timestamp_ms = Utc::now().timestamp_millis();

        let mut all_snapshots = Vec::new();
        let mut all_deltas = Vec::new();

        // Process wallets in batches to avoid overwhelming the API
        for chunk in wallets.chunks(self.config.max_concurrent_requests) {
            let futures: Vec<_> = chunk.iter()
                .map(|wallet| self.fetch_wallet_positions(wallet, timestamp_ms))
                .collect();

            let results = join_all(futures).await;

            for result in results {
                match result {
                    Ok((snapshots, deltas)) => {
                        all_snapshots.extend(snapshots);
                        all_deltas.extend(deltas);
                    }
                    Err(e) => {
                        warn!(?e, "Failed to fetch positions for wallet");
                    }
                }
            }
        }

        // Send snapshots to channel
        if let Some(tx) = &self.snapshot_tx {
            for snapshot in &all_snapshots {
                if tx.send(snapshot.clone()).await.is_err() {
                    warn!("Snapshot channel closed");
                    break;
                }
            }
        }

        // Send deltas to channel
        if let Some(tx) = &self.delta_tx {
            for delta in &all_deltas {
                if tx.send(delta.clone()).await.is_err() {
                    warn!("Delta channel closed");
                    break;
                }
            }
        }

        debug!(snapshots = all_snapshots.len(), deltas = all_deltas.len(), "Polled positions");

        Ok(all_snapshots)
    }

    /// Fetch positions for a single wallet
    async fn fetch_wallet_positions(
        &self,
        wallet: &str,
        timestamp_ms: i64,
    ) -> Result<(Vec<PositionSnapshot>, Vec<PositionDelta>)> {
        let state = self.client.get_clearinghouse_state(wallet).await?;

        let mut snapshots = Vec::new();
        let mut deltas = Vec::new();

        // Get previous positions for this wallet
        let prev_positions = {
            let positions = self.last_positions.read();
            positions.get(wallet).cloned().unwrap_or_default()
        };

        // Build current positions map
        let mut current_positions: HashMap<String, PositionSnapshot> = HashMap::new();

        for asset_pos in &state.asset_positions {
            let pos = &asset_pos.position;
            let symbol = pos.coin.clone();

            // Skip symbols not in our filter (if filter is non-empty)
            if !self.config.symbols.is_empty() && !self.config.symbols.contains(&symbol) {
                continue;
            }

            // Skip empty positions
            if pos.is_empty() {
                continue;
            }

            let snapshot = PositionSnapshot {
                timestamp_ms,
                wallet: wallet.to_string(),
                symbol: symbol.clone(),
                size: pos.size(),
                entry_price: pos.entry_price(),
                liquidation_price: pos.liquidation_price(),
                unrealized_pnl: pos.unrealized_pnl(),
                position_value: pos.position_value(),
                leverage: pos.leverage.value,
                is_cross_margin: pos.leverage.leverage_type == "cross",
            };

            // Compute delta if we have previous position
            if let Some(prev) = prev_positions.get(&symbol) {
                let delta = PositionDelta::from_snapshots(prev, &snapshot);
                if delta.change_type != super::snapshot::PositionChangeType::NoChange {
                    deltas.push(delta);
                }
            } else if !snapshot.is_empty() {
                // New position opened
                let empty_prev = PositionSnapshot {
                    timestamp_ms: timestamp_ms - 1,
                    wallet: wallet.to_string(),
                    symbol: symbol.clone(),
                    size: 0.0,
                    entry_price: 0.0,
                    liquidation_price: None,
                    unrealized_pnl: 0.0,
                    position_value: 0.0,
                    leverage: 1.0,
                    is_cross_margin: true,
                };
                deltas.push(PositionDelta::from_snapshots(&empty_prev, &snapshot));
            }

            snapshots.push(snapshot.clone());
            current_positions.insert(symbol, snapshot);
        }

        // Check for closed positions (positions that existed before but not now)
        for (symbol, prev) in &prev_positions {
            if !current_positions.contains_key(symbol) {
                // Position was closed
                let empty_curr = PositionSnapshot {
                    timestamp_ms,
                    wallet: wallet.to_string(),
                    symbol: symbol.clone(),
                    size: 0.0,
                    entry_price: 0.0,
                    liquidation_price: None,
                    unrealized_pnl: 0.0,
                    position_value: 0.0,
                    leverage: 1.0,
                    is_cross_margin: true,
                };
                deltas.push(PositionDelta::from_snapshots(prev, &empty_curr));
            }
        }

        // Update stored positions
        {
            let mut positions = self.last_positions.write();
            positions.insert(wallet.to_string(), current_positions);
        }

        Ok((snapshots, deltas))
    }

    /// Get wallet statistics from last poll
    pub fn get_wallet_stats(&self, wallet: &str) -> Option<WalletPositionStats> {
        let positions = self.last_positions.read();
        let wallet_positions = positions.get(wallet)?;

        if wallet_positions.is_empty() {
            return None;
        }

        let mut stats = WalletPositionStats {
            wallet: wallet.to_string(),
            ..Default::default()
        };

        for (symbol, snapshot) in wallet_positions {
            stats.position_count += 1;
            stats.total_position_value += snapshot.position_value.abs();
            stats.total_unrealized_pnl += snapshot.unrealized_pnl;

            if snapshot.position_value.abs() > stats.largest_position_value {
                stats.largest_position_value = snapshot.position_value.abs();
                stats.largest_position_symbol = Some(symbol.clone());
            }
        }

        Some(stats)
    }

    /// Run the position tracker polling loop
    pub async fn run(&self) -> Result<()> {
        info!(
            interval_secs = self.config.poll_interval_secs,
            "Starting position tracker"
        );

        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(self.config.poll_interval_secs)
        );

        loop {
            interval.tick().await;

            match self.poll_all().await {
                Ok(snapshots) => {
                    debug!(count = snapshots.len(), "Position poll complete");
                }
                Err(e) => {
                    error!(?e, "Position poll failed");
                }
            }
        }
    }
}

/// Discover wallets by monitoring trades
pub async fn discover_wallets_from_trades(
    trades_rx: &mut mpsc::Receiver<crate::ws::WsTrade>,
    min_trades: usize,
    max_wallets: usize,
) -> Vec<String> {
    use std::collections::HashMap;

    let mut wallet_counts: HashMap<String, usize> = HashMap::new();

    while let Some(trade) = trades_rx.recv().await {
        if let Some((maker, taker)) = &trade.users {
            *wallet_counts.entry(maker.clone()).or_insert(0) += 1;
            *wallet_counts.entry(taker.clone()).or_insert(0) += 1;
        }

        // Check if we have enough data
        let active_wallets: Vec<_> = wallet_counts.iter()
            .filter(|(_, count)| **count >= min_trades)
            .map(|(wallet, _)| wallet.clone())
            .take(max_wallets)
            .collect();

        if active_wallets.len() >= max_wallets {
            return active_wallets;
        }
    }

    // Return what we have
    let mut wallets: Vec<_> = wallet_counts.into_iter().collect();
    wallets.sort_by(|a, b| b.1.cmp(&a.1));
    wallets.into_iter()
        .take(max_wallets)
        .map(|(wallet, _)| wallet)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_tracker_config() {
        let config = PositionTrackerConfig::default();
        assert_eq!(config.poll_interval_secs, 60);
        assert!(config.symbols.contains(&"BTC".to_string()));
    }
}

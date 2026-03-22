//! Dashboard state management and snapshot types.

use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Log entry for streaming to dashboard
#[derive(Clone, Debug, Serialize)]
pub struct LogEntry {
    pub timestamp: i64,
    pub level: String,
    pub target: String,
    pub message: String,
}

/// Summary of key features for display
#[derive(Clone, Debug, Default, Serialize)]
pub struct FeaturesSummary {
    pub midprice: f64,
    pub spread_bps: f64,
    pub imbalance: f64,
    pub volatility_1m: f64,
    pub vpin: f64,
    pub whale_flow: f64,
    pub kyle_lambda: f64,
}

/// State snapshot for a single symbol
#[derive(Clone, Debug, Serialize)]
pub struct SymbolState {
    pub symbol: String,
    pub connected: bool,
    pub last_update_ms: i64,
    pub message_count: u64,
    pub feature_count: u64,
    pub features: FeaturesSummary,
}

impl SymbolState {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            connected: false,
            last_update_ms: 0,
            message_count: 0,
            feature_count: 0,
            features: FeaturesSummary::default(),
        }
    }
}

/// Full state snapshot sent to dashboard clients
#[derive(Clone, Debug, Serialize)]
pub struct StateSnapshot {
    pub timestamp: i64,
    pub symbols: Vec<SymbolState>,
    pub total_messages: u64,
    pub total_features: u64,
    pub uptime_secs: u64,
}

/// Shared state for the dashboard
pub struct DashboardState {
    /// Broadcast channel for log entries
    pub log_tx: broadcast::Sender<LogEntry>,
    /// Broadcast channel for state snapshots
    pub state_tx: broadcast::Sender<StateSnapshot>,
    /// Recent log buffer for backfill
    pub log_buffer: RwLock<Vec<LogEntry>>,
    /// Current state per symbol
    pub symbols: RwLock<HashMap<String, SymbolState>>,
    /// Start time for uptime calculation
    pub start_time: std::time::Instant,
    /// Configuration
    pub config: DashboardConfig,
}

/// Dashboard configuration
#[derive(Clone, Debug)]
pub struct DashboardConfig {
    /// Maximum logs to buffer for backfill
    pub log_buffer_size: usize,
    /// State update interval in milliseconds
    pub state_update_interval_ms: u64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            log_buffer_size: 500,
            state_update_interval_ms: 1000,
        }
    }
}

impl DashboardState {
    pub fn new() -> Self {
        Self::with_config(DashboardConfig::default())
    }

    pub fn with_config(config: DashboardConfig) -> Self {
        let (log_tx, _) = broadcast::channel(1000);
        let (state_tx, _) = broadcast::channel(100);

        Self {
            log_tx,
            state_tx,
            log_buffer: RwLock::new(Vec::with_capacity(config.log_buffer_size)),
            symbols: RwLock::new(HashMap::new()),
            start_time: std::time::Instant::now(),
            config,
        }
    }

    /// Add a log entry
    pub fn add_log(&self, entry: LogEntry) {
        // Add to buffer
        {
            let mut buffer = self.log_buffer.write();
            if buffer.len() >= self.config.log_buffer_size {
                buffer.remove(0);
            }
            buffer.push(entry.clone());
        }

        // Broadcast (ignore if no receivers)
        let _ = self.log_tx.send(entry);
    }

    /// Update symbol state
    pub fn update_symbol(&self, symbol: &str, update: impl FnOnce(&mut SymbolState)) {
        let mut symbols = self.symbols.write();
        let state = symbols
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolState::new(symbol.to_string()));
        update(state);
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> StateSnapshot {
        let symbols = self.symbols.read();
        let mut total_messages = 0;
        let mut total_features = 0;

        let symbol_states: Vec<SymbolState> = symbols
            .values()
            .map(|s| {
                total_messages += s.message_count;
                total_features += s.feature_count;
                s.clone()
            })
            .collect();

        StateSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis(),
            symbols: symbol_states,
            total_messages,
            total_features,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// Broadcast current state
    pub fn broadcast_state(&self) {
        let snapshot = self.snapshot();
        let _ = self.state_tx.send(snapshot);
    }

    /// Get log buffer for backfill
    pub fn get_log_backfill(&self) -> Vec<LogEntry> {
        self.log_buffer.read().clone()
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}

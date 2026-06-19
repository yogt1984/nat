//! Configuration management for the ingestor

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Root configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub general: GeneralConfig,
    pub websocket: WebSocketConfig,
    pub symbols: SymbolsConfig,
    pub features: FeaturesConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub dashboard: DashboardConfig,
    #[serde(default)]
    pub redis: RedisTomlConfig,
    #[serde(default)]
    pub algorithms: AlgorithmsConfig,
    #[serde(default)]
    pub trade_output: TradeOutputConfig,
    #[serde(default)]
    pub position_tracker: Option<PositionTrackerTomlConfig>,
}

fn default_poll_interval_secs() -> u64 {
    60
}
fn default_max_concurrent() -> usize {
    10
}
fn default_max_tracked_wallets() -> usize {
    50
}
fn default_whale_threshold_usd_tracker() -> f64 {
    500_000.0
}

/// Configuration for the position tracker (polls wallet positions via REST API)
#[derive(Debug, Clone, Deserialize)]
pub struct PositionTrackerTomlConfig {
    /// Enable position tracking (default: false)
    #[serde(default)]
    pub enabled: bool,
    /// Polling interval in seconds
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// Maximum concurrent API requests per poll cycle
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    /// Seed wallet addresses to track from startup
    #[serde(default)]
    pub initial_wallets: Vec<String>,
    /// Discover whale wallets from trade stream (requires WsTrade.users)
    #[serde(default)]
    pub discover_from_trades: bool,
    /// Maximum number of wallets to track
    #[serde(default = "default_max_tracked_wallets")]
    pub max_tracked_wallets: usize,
    /// Position value (USD) above which a wallet is classified as whale
    #[serde(default = "default_whale_threshold_usd_tracker")]
    pub whale_threshold_usd: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeneralConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
    pub data_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebSocketConfig {
    #[serde(default = "default_ws_url")]
    pub url: String,
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_ms: u64,
    #[serde(default = "default_max_reconnect_delay")]
    pub max_reconnect_delay_ms: u64,
    #[serde(default = "default_ping_interval")]
    pub ping_interval_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymbolsConfig {
    pub assets: Vec<String>,
    /// Path to config/symbols.toml. If set, overrides `assets`.
    #[serde(default)]
    pub symbols_file: Option<String>,
}

pub use ing_types::FeaturesConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_row_group_size")]
    pub row_group_size: usize,
    #[serde(default = "default_compression")]
    pub compression: String,
    #[serde(default = "default_rotate_interval")]
    pub rotate_interval: String,
    pub data_dir: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MetricsConfig {
    pub prometheus_addr: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DashboardConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_dashboard_addr")]
    pub addr: String,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            addr: default_dashboard_addr(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedisTomlConfig {
    #[serde(default = "default_redis_url")]
    pub url: String,
    #[serde(default = "default_redis_prefix")]
    pub channel_prefix: Option<String>,
    #[serde(default)]
    pub cache_ttl_seconds: Option<u64>,
    #[serde(default)]
    pub publish_interval_ms: Option<u64>,
}

impl Default for RedisTomlConfig {
    fn default() -> Self {
        Self {
            url: default_redis_url(),
            channel_prefix: None,
            cache_ttl_seconds: None,
            publish_interval_ms: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AlgorithmsConfig {
    #[serde(default)]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradeOutputConfig {
    #[serde(default)]
    pub enabled: bool,
    pub data_dir: Option<String>,
    #[serde(default = "default_trade_buffer_size")]
    pub buffer_size: usize,
    #[serde(default = "default_compression")]
    pub compression: String,
    #[serde(default = "default_rotate_interval")]
    pub rotate_interval: String,
}

impl Default for TradeOutputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            data_dir: None,
            buffer_size: default_trade_buffer_size(),
            compression: default_compression(),
            rotate_interval: default_rotate_interval(),
        }
    }
}

fn default_trade_buffer_size() -> usize {
    50_000
}

/// Read symbols list from an external TOML file (config/symbols.toml).
fn load_symbols_toml(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read symbols file: {:?}", path))?;
    let table: toml::Table = content
        .parse()
        .with_context(|| format!("Failed to parse symbols file: {:?}", path))?;
    let symbols = table
        .get("symbols")
        .and_then(|v| v.as_array())
        .with_context(|| "symbols file missing 'symbols' array")?;
    symbols
        .iter()
        .map(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .with_context(|| "symbols array must contain strings")
        })
        .collect()
}

// Default values
fn default_log_level() -> String {
    "info".to_string()
}
fn default_ws_url() -> String {
    "wss://api.hyperliquid.xyz/ws".to_string()
}
fn default_reconnect_delay() -> u64 {
    1000
}
fn default_max_reconnect_delay() -> u64 {
    30000
}
fn default_ping_interval() -> u64 {
    30000
}
fn default_format() -> String {
    "parquet".to_string()
}
fn default_row_group_size() -> usize {
    10000
}
fn default_compression() -> String {
    "zstd".to_string()
}
fn default_rotate_interval() -> String {
    "1h".to_string()
}

/// Parse a file-rotation interval like "30s", "5m", "10m", "1h", "1d" into a
/// `Duration`. Unrecognized, zero, or unit-less values fall back to 1 hour (the
/// historical hourly-rotation behavior) so a typo never disables rotation.
pub fn parse_rotate_interval(s: &str) -> std::time::Duration {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let n: u64 = num.parse().unwrap_or(0);
    let secs = match unit.trim().to_ascii_lowercase().as_str() {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86400,
        _ => 0,
    };
    if secs == 0 {
        std::time::Duration::from_secs(3600)
    } else {
        std::time::Duration::from_secs(secs)
    }
}

fn default_dashboard_addr() -> String {
    "0.0.0.0:8080".to_string()
}
fn default_redis_url() -> String {
    "redis://127.0.0.1:6379".to_string()
}
fn default_redis_prefix() -> Option<String> {
    None
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;

        let mut config: Config =
            toml::from_str(&content).with_context(|| "Failed to parse config file")?;

        // Load symbols from external file if symbols_file is set
        if let Some(ref symbols_file) = config.symbols.symbols_file {
            let symbols_path = path.parent().unwrap_or(Path::new(".")).join(symbols_file);
            match load_symbols_toml(&symbols_path) {
                Ok(symbols) => {
                    tracing::info!(
                        ?symbols_path,
                        n = symbols.len(),
                        "Loaded symbols from symbols_file"
                    );
                    config.symbols.assets = symbols;
                }
                Err(e) => {
                    tracing::warn!(?symbols_path, %e, "Failed to load symbols_file, using inline assets");
                }
            }
        }

        // Allow environment variable overrides
        if let Ok(val) = std::env::var("ING_DASHBOARD_ENABLED") {
            config.dashboard.enabled = val == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("ING_PROMETHEUS_ADDR") {
            config.metrics.prometheus_addr = Some(val);
        }

        config.validate()?;

        Ok(config)
    }

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        if self.symbols.assets.is_empty() {
            anyhow::bail!("At least one asset must be configured");
        }

        if self.features.emission_interval_ms == 0 {
            anyhow::bail!("emission_interval_ms must be > 0");
        }

        if self.features.book_levels == 0 || self.features.book_levels > 100 {
            anyhow::bail!("book_levels must be between 1 and 100");
        }

        Ok(())
    }

    /// Get the data directory, with fallback
    pub fn data_dir(&self) -> &str {
        self.output
            .data_dir
            .as_deref()
            .unwrap_or(&self.general.data_dir)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                log_level: default_log_level(),
                data_dir: "./data/features".to_string(),
            },
            websocket: WebSocketConfig {
                url: default_ws_url(),
                reconnect_delay_ms: default_reconnect_delay(),
                max_reconnect_delay_ms: default_max_reconnect_delay(),
                ping_interval_ms: default_ping_interval(),
            },
            symbols: SymbolsConfig {
                assets: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
                symbols_file: None,
            },
            features: FeaturesConfig {
                emission_interval_ms: 100,
                trade_buffer_seconds: 60,
                book_levels: 10,
                price_buffer_size: 1000,
                gmm_model_path: None,
                whale_flow: None,
            },
            output: OutputConfig {
                format: default_format(),
                row_group_size: default_row_group_size(),
                compression: default_compression(),
                rotate_interval: default_rotate_interval(),
                data_dir: None,
            },
            metrics: MetricsConfig::default(),
            dashboard: DashboardConfig::default(),
            redis: RedisTomlConfig::default(),
            algorithms: AlgorithmsConfig::default(),
            trade_output: TradeOutputConfig::default(),
            position_tracker: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_rotate_interval;
    use std::time::Duration;

    #[test]
    fn parse_rotate_interval_units() {
        assert_eq!(parse_rotate_interval("30s"), Duration::from_secs(30));
        assert_eq!(parse_rotate_interval("5m"), Duration::from_secs(300));
        assert_eq!(parse_rotate_interval("10m"), Duration::from_secs(600));
        assert_eq!(parse_rotate_interval("1h"), Duration::from_secs(3600));
        assert_eq!(parse_rotate_interval("2d"), Duration::from_secs(172_800));
        assert_eq!(parse_rotate_interval(" 5M "), Duration::from_secs(300)); // trim + case
    }

    #[test]
    fn parse_rotate_interval_falls_back_to_1h() {
        for bad in ["garbage", "0m", "", "5x", "h", "100"] {
            assert_eq!(
                parse_rotate_interval(bad),
                Duration::from_secs(3600),
                "input {bad:?} should fall back to 1h"
            );
        }
    }
}

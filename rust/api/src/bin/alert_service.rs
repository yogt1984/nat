//! Standalone Telegram Alert Service
//!
//! Subscribes to Redis channels and forwards alerts to Telegram.
//!
//! Configuration is loaded from `config/agent.toml` under `[alerts]`.
//! Environment variables `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
//! override config file values.
//!
//! # Usage
//!
//! ```bash
//! # Config-driven (credentials in agent.toml or env):
//! cargo run --bin alert-service
//!
//! # Env-override:
//! TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." cargo run --bin alert-service
//! ```

use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use nat_api::telegram::{run_multi_channel_alert_service, AlertConfig};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("nat_api=info,alert_service=info")),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting NAT Telegram Alert Service");

    let config = load_config();

    match (&config.telegram_bot_token, &config.telegram_chat_id) {
        (None, _) => {
            warn!("No Telegram bot token — alerts disabled. Set TELEGRAM_BOT_TOKEN or configure in agent.toml [alerts].");
        }
        (_, None) => {
            warn!("No Telegram chat ID — alerts disabled. Set TELEGRAM_CHAT_ID or configure in agent.toml [alerts].");
        }
        (Some(_), Some(chat_id)) => {
            info!("Telegram bot configured, will send to chat {}", chat_id);
        }
    }

    info!(redis_url = %config.redis_url, channels = ?config.channels, "Connecting to Redis");

    if let (Some(token), Some(chat_id)) = (config.telegram_bot_token.clone(), config.telegram_chat_id.clone()) {
        if let Err(e) = run_multi_channel_alert_service(
            &config.redis_url,
            token,
            chat_id,
            &config.channels,
            &config.research_events,
        ).await {
            error!("Alert service error: {}", e);
            std::process::exit(1);
        }
    } else {
        info!("Running in listen-only mode (no Telegram credentials).");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        }
    }
}

/// Load alert config from TOML, with env var overrides.
fn load_config() -> AlertConfig {
    let mut config = AlertConfig::default();

    // Try reading config file
    for path in ["config/agent.toml", "../config/agent.toml"] {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(toml) = content.parse::<toml::Table>() {
                if let Some(alerts) = toml.get("alerts").and_then(|v| v.as_table()) {
                    if let Some(token) = alerts.get("telegram_bot_token").and_then(|v| v.as_str()) {
                        if !token.is_empty() {
                            config.telegram_bot_token = Some(token.to_string());
                        }
                    }
                    if let Some(chat_id) = alerts.get("telegram_chat_id").and_then(|v| v.as_str()) {
                        if !chat_id.is_empty() {
                            config.telegram_chat_id = Some(chat_id.to_string());
                        }
                    }
                    if let Some(url) = alerts.get("redis_url").and_then(|v| v.as_str()) {
                        config.redis_url = url.to_string();
                    }
                    if let Some(channels) = alerts.get("channels").and_then(|v| v.as_array()) {
                        config.channels = channels
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }
                    if let Some(events) = alerts.get("research_events").and_then(|v| v.as_array()) {
                        config.research_events = events
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }
                }
            }
            break;
        }
    }

    // Env vars override config
    if let Ok(token) = std::env::var("TELEGRAM_BOT_TOKEN") {
        config.telegram_bot_token = Some(token);
    }
    if let Ok(chat_id) = std::env::var("TELEGRAM_CHAT_ID") {
        config.telegram_chat_id = Some(chat_id);
    }
    if let Ok(url) = std::env::var("REDIS_URL") {
        config.redis_url = url;
    }

    config
}

//! Standalone Telegram Alert Service
//!
//! Subscribes to Redis alerts and forwards them to Telegram.
//!
//! # Environment Variables
//!
//! - `REDIS_URL`: Redis connection URL (default: redis://127.0.0.1:6379)
//! - `TELEGRAM_BOT_TOKEN`: Telegram bot token from @BotFather (required)
//! - `TELEGRAM_CHAT_ID`: Telegram chat ID to send alerts to (required)
//!
//! # Usage
//!
//! ```bash
//! export TELEGRAM_BOT_TOKEN="your_token_here"
//! export TELEGRAM_CHAT_ID="your_chat_id"
//! cargo run --bin alert-service
//! ```

use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// Import from the library crate
use nat_api::telegram::run_alert_service;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("nat_api=info,alert_service=info")),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting NAT Telegram Alert Service");

    // Load Redis URL: env var > config file > hardcoded default
    let toml_redis_url = std::fs::read_to_string("../config/ing.toml")
        .ok()
        .and_then(|content| {
            content.lines()
                .skip_while(|l| !l.starts_with("[redis]"))
                .skip(1)
                .find(|l| l.starts_with("url"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"').to_string())
        });
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| toml_redis_url.unwrap_or_else(|| "redis://127.0.0.1:6379".to_string()));

    let telegram_token = std::env::var("TELEGRAM_BOT_TOKEN").ok();
    let telegram_chat_id = std::env::var("TELEGRAM_CHAT_ID").ok();

    match (&telegram_token, &telegram_chat_id) {
        (None, _) => {
            tracing::warn!("TELEGRAM_BOT_TOKEN not set — alerts disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable.");
        }
        (_, None) => {
            tracing::warn!("TELEGRAM_CHAT_ID not set — alerts disabled. Set TELEGRAM_CHAT_ID to enable.");
        }
        (Some(_), Some(chat_id)) => {
            info!("Telegram bot configured, will send alerts to chat {}", chat_id);
        }
    }

    info!(redis_url = %redis_url, "Connecting to Redis");

    if let (Some(token), Some(chat_id)) = (telegram_token, telegram_chat_id) {
        // Run with Telegram alerts enabled
        if let Err(e) = run_alert_service(&redis_url, token, chat_id).await {
            error!("Alert service error: {}", e);
            std::process::exit(1);
        }
    } else {
        // Run in listen-only mode (no alerts sent)
        info!("Running in listen-only mode (no Telegram credentials). Subscribing to Redis for monitoring.");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        }
    }
}

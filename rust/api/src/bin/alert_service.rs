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

    // Load configuration from environment
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let telegram_token = match std::env::var("TELEGRAM_BOT_TOKEN") {
        Ok(token) => token,
        Err(_) => {
            error!("TELEGRAM_BOT_TOKEN environment variable is required");
            error!("");
            error!("To get a bot token:");
            error!("  1. Message @BotFather on Telegram");
            error!("  2. Send /newbot");
            error!("  3. Follow the prompts to create your bot");
            error!("  4. Copy the token and set TELEGRAM_BOT_TOKEN");
            std::process::exit(1);
        }
    };

    let telegram_chat_id = match std::env::var("TELEGRAM_CHAT_ID") {
        Ok(chat_id) => chat_id,
        Err(_) => {
            error!("TELEGRAM_CHAT_ID environment variable is required");
            error!("");
            error!("To get your chat ID:");
            error!("  1. Start a chat with your bot");
            error!("  2. Send any message");
            error!("  3. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates");
            error!("  4. Find your chat.id in the response");
            std::process::exit(1);
        }
    };

    info!(redis_url = %redis_url, "Connecting to Redis");
    info!("Telegram bot configured, will send alerts to chat {}", telegram_chat_id);

    // Run the alert service
    if let Err(e) = run_alert_service(&redis_url, telegram_token, telegram_chat_id).await {
        error!("Alert service error: {}", e);
        std::process::exit(1);
    }
}

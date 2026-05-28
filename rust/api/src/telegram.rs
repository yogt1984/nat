//! Telegram Alert Service
//!
//! Subscribes to Redis channels (market alerts + research events) and sends
//! formatted messages to Telegram.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Alert service configuration (loaded from TOML + env overrides).
pub struct AlertConfig {
    pub telegram_bot_token: Option<String>,
    pub telegram_chat_id: Option<String>,
    pub redis_url: String,
    pub channels: Vec<String>,
    pub research_events: Vec<String>,
    pub alert_log_path: Option<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            telegram_bot_token: None,
            telegram_chat_id: None,
            redis_url: "redis://127.0.0.1:6379".to_string(),
            channels: vec!["nat:alerts".to_string()],
            research_events: vec![
                "hypothesis_registered".to_string(),
                "cycle_completed".to_string(),
            ],
            alert_log_path: Some("data/alerts/alerts.jsonl".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Telegram bot
// ---------------------------------------------------------------------------

/// Telegram bot client
pub struct TelegramBot {
    client: Client,
    token: String,
    chat_id: String,
}

#[derive(Serialize)]
struct SendMessageRequest<'a> {
    chat_id: &'a str,
    text: &'a str,
    parse_mode: &'a str,
}

impl TelegramBot {
    pub fn new(token: String, chat_id: String) -> Self {
        Self {
            client: Client::new(),
            token,
            chat_id,
        }
    }

    pub async fn send_alert(&self, alert: &AlertMessage) -> Result<(), reqwest::Error> {
        let text = format_alert_message(alert);
        self.send_raw(&text).await
    }

    pub async fn send_message(&self, text: &str) -> Result<(), reqwest::Error> {
        self.send_raw(text).await
    }

    async fn send_raw(&self, text: &str) -> Result<(), reqwest::Error> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        let request = SendMessageRequest {
            chat_id: &self.chat_id,
            text,
            parse_mode: "HTML",
        };

        let response = self.client.post(&url).json(&request).send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!("Telegram API error: {} - {}", status, body);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// File-based alert logger (fallback when Telegram is unavailable)
// ---------------------------------------------------------------------------

/// Appends alert JSON lines to a local file as a durable fallback.
pub struct AlertLogger {
    path: PathBuf,
}

impl AlertLogger {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        info!("Alert file logger: {}", path.display());
        Ok(Self { path })
    }

    /// Append a JSON value as a single line (atomic via O_APPEND).
    pub fn log(&self, value: &serde_json::Value) {
        let line = match serde_json::to_string(value) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to serialize alert for file log: {}", e);
                return;
            }
        };
        match OpenOptions::new().create(true).append(true).open(&self.path) {
            Ok(mut f) => {
                if let Err(e) = writeln!(f, "{}", line) {
                    warn!("Failed to write alert to {}: {}", self.path.display(), e);
                }
            }
            Err(e) => {
                warn!("Failed to open alert log {}: {}", self.path.display(), e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Alert messages (market alerts from ingestor)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertMessage {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub data: serde_json::Value,
}

fn format_alert_message(alert: &AlertMessage) -> String {
    let emoji = match alert.severity.as_str() {
        "Critical" => "\u{1f6a8}",
        "Warning" => "\u{26a0}\u{fe0f}",
        "Info" => "\u{2139}\u{fe0f}",
        _ => "\u{1f4ca}",
    };

    let type_emoji = match alert.alert_type.as_str() {
        "WhaleAccumulation" => "\u{1f40b}\u{2b06}\u{fe0f}",
        "WhaleDistribution" => "\u{1f40b}\u{2b07}\u{fe0f}",
        "LiquidationCluster" => "\u{1f4a5}",
        "RegimeChange" => "\u{1f504}",
        "EntropyDrop" => "\u{1f3af}",
        "ConcentrationSpike" => "\u{1f4ca}",
        _ => "\u{1f4cc}",
    };

    let timestamp = chrono::DateTime::from_timestamp_millis(alert.timestamp_ms as i64)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let data_str = serde_json::to_string_pretty(&alert.data).unwrap_or_default();
    let data_truncated = if data_str.len() > 500 {
        format!("{}...", &data_str[..500])
    } else {
        data_str
    };

    format!(
        "{emoji} <b>NAT Alert: {type_emoji} {alert_type}</b>\n\n\
        <b>Symbol:</b> {symbol}\n\
        <b>Severity:</b> {severity}\n\
        <b>Message:</b> {message}\n\n\
        <b>Details:</b>\n<pre>{data}</pre>\n\n\
        <i>{timestamp}</i>",
        emoji = emoji,
        type_emoji = type_emoji,
        alert_type = alert.alert_type,
        symbol = alert.symbol,
        severity = alert.severity,
        message = alert.message,
        data = data_truncated,
        timestamp = timestamp,
    )
}

// ---------------------------------------------------------------------------
// Research event formatting
// ---------------------------------------------------------------------------

fn format_research_event(event: &serde_json::Value) -> Option<String> {
    let event_type = event.get("event").and_then(|v| v.as_str())?;

    match event_type {
        "hypothesis_registered" => {
            let id = event.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let agent = event.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
            let generator = event.get("generator").and_then(|v| v.as_str()).unwrap_or("?");
            let claim = event.get("claim").and_then(|v| v.as_str()).unwrap_or("");
            let claim_short = if claim.len() > 120 { &claim[..120] } else { claim };
            Some(format!(
                "\u{2705} <b>Signal Registered</b>\n\n\
                <b>ID:</b> <code>{id}</code>\n\
                <b>Agent:</b> {agent}\n\
                <b>Generator:</b> {generator}\n\
                <b>Claim:</b> {claim_short}",
            ))
        }
        "cycle_completed" => {
            let agent = event.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
            let cycle_id = event.get("cycle_id").and_then(|v| v.as_str()).unwrap_or("?");
            let tested = event.get("n_tested").and_then(|v| v.as_u64()).unwrap_or(0);
            let registered = event.get("n_registered").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(format!(
                "\u{1f501} <b>Cycle Complete</b>\n\n\
                <b>Agent:</b> {agent}\n\
                <b>Cycle:</b> <code>{cycle_id}</code>\n\
                <b>Tested:</b> {tested}  |  <b>Registered:</b> {registered}",
            ))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Service runners
// ---------------------------------------------------------------------------

/// Legacy single-channel runner (backward compat).
pub async fn run_alert_service(
    redis_url: &str,
    telegram_token: String,
    telegram_chat_id: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run_multi_channel_alert_service(
        redis_url,
        telegram_token,
        telegram_chat_id,
        &["nat:alerts".to_string()],
        &["hypothesis_registered".to_string(), "cycle_completed".to_string()],
    ).await
}

/// Multi-channel alert service — subscribes to market alerts (Pub/Sub) + research events (Stream).
///
/// Market alerts use Pub/Sub (fire-and-forget, ephemeral).
/// Research events use Redis Streams (reliable delivery, replay on restart).
/// All alerts are always logged to a local JSONL file as a durable fallback.
pub async fn run_multi_channel_alert_service(
    redis_url: &str,
    telegram_token: String,
    telegram_chat_id: String,
    channels: &[String],
    research_events: &[String],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run_alert_service_with_log(
        redis_url,
        telegram_token,
        telegram_chat_id,
        channels,
        research_events,
        None,
    )
    .await
}

/// Full alert service with optional file-based logging.
pub async fn run_alert_service_with_log(
    redis_url: &str,
    telegram_token: String,
    telegram_chat_id: String,
    channels: &[String],
    research_events: &[String],
    alert_log_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let file_logger = match alert_log_path {
        Some(path) => match AlertLogger::new(path) {
            Ok(logger) => Some(logger),
            Err(e) => {
                warn!("Could not initialize alert file logger: {}", e);
                None
            }
        },
        None => None,
    };
    let client = redis::Client::open(redis_url)?;
    let conn = redis::aio::ConnectionManager::new(client.clone()).await?;

    // Subscribe to market alert channels via Pub/Sub
    let alert_channels: Vec<&String> = channels
        .iter()
        .filter(|c| *c != "nat:research:events")
        .collect();
    let mut pubsub = client.get_async_pubsub().await?;
    for channel in &alert_channels {
        pubsub.subscribe(channel.as_str()).await?;
        info!("Subscribed to Redis Pub/Sub channel: {}", channel);
    }

    // Set up consumer group for research stream
    let research_stream = "nat:research:stream";
    let research_group = "nat-alerts";
    let consumer_name = "alert-service";
    {
        let mut c = conn.clone();
        let result: redis::RedisResult<()> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(research_stream)
            .arg(research_group)
            .arg("$")
            .arg("MKSTREAM")
            .query_async(&mut c)
            .await;
        match result {
            Ok(_) | Err(_) => {} // ignore BUSYGROUP
        }
    }
    info!("Listening on research stream: {}", research_stream);

    let bot = TelegramBot::new(telegram_token, telegram_chat_id);

    bot.send_message(&format!(
        "\u{1f7e2} <b>NAT Alert Service Started</b>\n\nAlerts: {}\nResearch: stream",
        alert_channels.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    ))
    .await
    .ok();

    let allowed_events: std::collections::HashSet<&str> =
        research_events.iter().map(|s| s.as_str()).collect();

    let mut alert_stream = pubsub.on_message();

    loop {
        tokio::select! {
            // Market alerts via Pub/Sub
            msg = alert_stream.next() => {
                let Some(msg) = msg else { break };
                let payload: String = msg.get_payload().unwrap_or_default();
                match serde_json::from_str::<serde_json::Value>(&payload) {
                    Ok(alert_json) => {
                        // Always log to file first (durable fallback)
                        if let Some(ref logger) = file_logger {
                            logger.log(&alert_json);
                        }
                        let alert = AlertMessage {
                            timestamp_ms: alert_json["timestamp_ms"].as_u64().unwrap_or(0),
                            symbol: alert_json["symbol"]
                                .as_str()
                                .unwrap_or("UNKNOWN")
                                .to_string(),
                            alert_type: extract_alert_type(&alert_json["alert_type"]),
                            severity: extract_severity(&alert_json["severity"]),
                            message: alert_json["message"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                            data: alert_json["data"].clone(),
                        };
                        if let Err(e) = bot.send_alert(&alert).await {
                            error!("Failed to send Telegram alert: {}", e);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse alert JSON: {} - {}", e, &payload[..payload.len().min(200)]);
                    }
                }
            }
            // Research events via Redis Stream
            _ = async {
                let mut c = conn.clone();
                let result: redis::RedisResult<redis::Value> = redis::cmd("XREADGROUP")
                    .arg("GROUP").arg(research_group).arg(consumer_name)
                    .arg("COUNT").arg(10)
                    .arg("BLOCK").arg(2000)
                    .arg("STREAMS").arg(research_stream).arg(">")
                    .query_async(&mut c)
                    .await;
                match result {
                    Ok(value) => {
                        let messages = crate::redis_client::parse_xread_response(value);
                        let mut ack_ids = Vec::new();
                        for (id, payload) in &messages {
                            match serde_json::from_str::<serde_json::Value>(payload) {
                                Ok(event) => {
                                    let event_type = event.get("event").and_then(|v| v.as_str()).unwrap_or("");
                                    if !allowed_events.is_empty() && !allowed_events.contains(event_type) {
                                        ack_ids.push(id.clone());
                                        continue;
                                    }
                                    // Always log to file first
                                    if let Some(ref logger) = file_logger {
                                        logger.log(&event);
                                    }
                                    if let Some(text) = format_research_event(&event) {
                                        if let Err(e) = bot.send_message(&text).await {
                                            error!("Failed to send research event to Telegram: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to parse research event: {}", e);
                                }
                            }
                            ack_ids.push(id.clone());
                        }
                        // ACK all processed
                        if !ack_ids.is_empty() {
                            let mut cmd = redis::cmd("XACK");
                            cmd.arg(research_stream).arg(research_group);
                            for id in &ack_ids {
                                cmd.arg(id.as_str());
                            }
                            let _: redis::RedisResult<()> = cmd.query_async(&mut c).await;
                        }
                    }
                    Err(e) => {
                        warn!("XREADGROUP error in alert service: {}", e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            } => {}
        }
    }

    Ok(())
}

/// Extract alert type from JSON value (handles enum serialization)
fn extract_alert_type(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else if let Some(obj) = value.as_object() {
        obj.keys().next().cloned().unwrap_or_else(|| "Unknown".to_string())
    } else {
        "Unknown".to_string()
    }
}

fn extract_severity(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else {
        "Info".to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_alert_message() {
        let alert = AlertMessage {
            timestamp_ms: 1711000000000,
            symbol: "BTC".to_string(),
            alert_type: "WhaleAccumulation".to_string(),
            severity: "Warning".to_string(),
            message: "Whale accumulation detected".to_string(),
            data: serde_json::json!({
                "flow_1h": 1000000,
                "zscore": 2.5
            }),
        };

        let formatted = format_alert_message(&alert);
        assert!(formatted.contains("BTC"));
        assert!(formatted.contains("Warning"));
        assert!(formatted.contains("WhaleAccumulation"));
    }

    #[test]
    fn test_extract_alert_type() {
        let value = serde_json::json!("WhaleAccumulation");
        assert_eq!(extract_alert_type(&value), "WhaleAccumulation");

        let value = serde_json::json!({"Custom": "test"});
        assert_eq!(extract_alert_type(&value), "Custom");
    }

    #[test]
    fn test_format_research_hypothesis_registered() {
        let event = serde_json::json!({
            "event": "hypothesis_registered",
            "id": "HYP-SYS-042",
            "agent": "microstructure",
            "generator": "systematic",
            "claim": "Spread compression predicts short-term momentum"
        });
        let text = format_research_event(&event).unwrap();
        assert!(text.contains("Signal Registered"));
        assert!(text.contains("HYP-SYS-042"));
        assert!(text.contains("microstructure"));
        assert!(text.contains("systematic"));
    }

    #[test]
    fn test_format_research_cycle_completed() {
        let event = serde_json::json!({
            "event": "cycle_completed",
            "agent": "medium_freq",
            "cycle_id": "CYC-MF-012",
            "n_tested": 8,
            "n_registered": 2
        });
        let text = format_research_event(&event).unwrap();
        assert!(text.contains("Cycle Complete"));
        assert!(text.contains("medium_freq"));
        assert!(text.contains("8"));
        assert!(text.contains("2"));
    }

    #[test]
    fn test_format_research_unknown_event_returns_none() {
        let event = serde_json::json!({
            "event": "gate_passed",
            "hypothesis_id": "H1"
        });
        assert!(format_research_event(&event).is_none());
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert!(config.telegram_bot_token.is_none());
        assert_eq!(config.channels.len(), 1);
        assert_eq!(config.channels[0], "nat:alerts");
        assert_eq!(config.research_events.len(), 2);
        assert_eq!(
            config.alert_log_path.as_deref(),
            Some("data/alerts/alerts.jsonl")
        );
    }

    #[test]
    fn test_alert_logger_write() {
        let dir = std::env::temp_dir().join("nat_test_alerts");
        let path = dir.join("test.jsonl");
        let _ = fs::remove_file(&path);

        let logger = AlertLogger::new(path.to_str().unwrap()).unwrap();
        let alert = serde_json::json!({"type": "test", "symbol": "BTC"});
        logger.log(&alert);
        logger.log(&alert);

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("BTC"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }
}

//! Telegram Alert Service
//!
//! Subscribes to Redis alerts and sends formatted messages to Telegram.

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

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
    /// Create a new Telegram bot client
    pub fn new(token: String, chat_id: String) -> Self {
        Self {
            client: Client::new(),
            token,
            chat_id,
        }
    }

    /// Send an alert message to Telegram
    pub async fn send_alert(&self, alert: &AlertMessage) -> Result<(), reqwest::Error> {
        let text = format_alert_message(alert);

        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);

        let request = SendMessageRequest {
            chat_id: &self.chat_id,
            text: &text,
            parse_mode: "HTML",
        };

        let response = self.client.post(&url).json(&request).send().await?;

        if response.status().is_success() {
            info!(
                symbol = %alert.symbol,
                alert_type = %alert.alert_type,
                "Alert sent to Telegram"
            );
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!("Telegram API error: {} - {}", status, body);
        }

        Ok(())
    }

    /// Send a simple text message
    pub async fn send_message(&self, text: &str) -> Result<(), reqwest::Error> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);

        let request = SendMessageRequest {
            chat_id: &self.chat_id,
            text,
            parse_mode: "HTML",
        };

        self.client.post(&url).json(&request).send().await?;
        Ok(())
    }
}

/// Alert message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertMessage {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub data: serde_json::Value,
}

/// Format an alert message for Telegram with emojis and HTML
fn format_alert_message(alert: &AlertMessage) -> String {
    let emoji = match alert.severity.as_str() {
        "Critical" => "🚨",
        "Warning" => "⚠️",
        "Info" => "ℹ️",
        _ => "📊",
    };

    let type_emoji = match alert.alert_type.as_str() {
        "WhaleAccumulation" => "🐋⬆️",
        "WhaleDistribution" => "🐋⬇️",
        "LiquidationCluster" => "💥",
        "RegimeChange" => "🔄",
        "EntropyDrop" => "🎯",
        "ConcentrationSpike" => "📊",
        _ => "📌",
    };

    // Format timestamp
    let timestamp = chrono::DateTime::from_timestamp_millis(alert.timestamp_ms as i64)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    // Format data as pretty JSON (truncated if too long)
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

/// Run the alert service
///
/// Subscribes to Redis `nat:alerts` channel and forwards alerts to Telegram.
pub async fn run_alert_service(
    redis_url: &str,
    telegram_token: String,
    telegram_chat_id: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = redis::Client::open(redis_url)?;
    let mut pubsub = client.get_async_pubsub().await?;

    pubsub.subscribe("nat:alerts").await?;

    let bot = TelegramBot::new(telegram_token, telegram_chat_id);

    // Send startup message
    bot.send_message("🟢 <b>NAT Alert Service Started</b>\n\nListening for alerts...")
        .await
        .ok();

    info!("Alert service started, listening for alerts on nat:alerts...");

    let mut stream = pubsub.on_message();

    while let Some(msg) = stream.next().await {
        let payload: String = msg.get_payload().unwrap_or_default();

        match serde_json::from_str::<serde_json::Value>(&payload) {
            Ok(alert_json) => {
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
                warn!("Failed to parse alert JSON: {} - payload: {}", e, payload);
            }
        }
    }

    Ok(())
}

/// Extract alert type from JSON value (handles enum serialization)
fn extract_alert_type(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else if let Some(obj) = value.as_object() {
        // Handle {"Custom": "value"} format
        obj.keys().next().cloned().unwrap_or_else(|| "Unknown".to_string())
    } else {
        "Unknown".to_string()
    }
}

/// Extract severity from JSON value
fn extract_severity(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else {
        "Info".to_string()
    }
}

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
        assert!(formatted.contains("🐋⬆️"));
        assert!(formatted.contains("BTC"));
        assert!(formatted.contains("Warning"));
    }

    #[test]
    fn test_extract_alert_type() {
        // String format
        let value = serde_json::json!("WhaleAccumulation");
        assert_eq!(extract_alert_type(&value), "WhaleAccumulation");

        // Object format (Custom enum variant)
        let value = serde_json::json!({"Custom": "test"});
        assert_eq!(extract_alert_type(&value), "Custom");
    }
}

//! WebSocket client implementation for Hyperliquid

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async,
    tungstenite::Message,
    MaybeTlsStream,
    WebSocketStream,
};
use tracing::{debug, info, warn, error};

use crate::config::WebSocketConfig;
use super::messages::{parse_ws_message, SubscriptionRequest, WsMessage};

/// Hyperliquid WebSocket client with keepalive and stale detection
pub struct HyperliquidClient {
    config: WebSocketConfig,
    symbol: String,
    stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    reconnect_attempts: u32,
    first_message_logged: bool,
    connected_at: Option<std::time::Instant>,
    message_count: u64,
    last_message_at: Option<std::time::Instant>,
    last_ping_sent: Option<std::time::Instant>,
    pong_received: bool,
}

impl HyperliquidClient {
    /// Create a new client (does not connect yet)
    pub fn new(config: &WebSocketConfig, symbol: &str) -> Self {
        Self {
            config: config.clone(),
            symbol: symbol.to_string(),
            stream: None,
            reconnect_attempts: 0,
            first_message_logged: false,
            connected_at: None,
            message_count: 0,
            last_message_at: None,
            last_ping_sent: None,
            pong_received: true,
        }
    }

    /// Connect and subscribe to channels
    pub async fn connect(&mut self) -> Result<()> {
        info!(symbol = %self.symbol, url = %self.config.url, "Connecting to WebSocket");

        let (stream, response) = connect_async(&self.config.url)
            .await
            .context("Failed to connect to WebSocket")?;

        debug!(?response, "WebSocket connected");

        self.stream = Some(stream);
        self.reconnect_attempts = 0;
        self.connected_at = Some(std::time::Instant::now());
        self.last_message_at = Some(std::time::Instant::now());
        self.last_ping_sent = None;
        self.pong_received = true;
        self.first_message_logged = false;
        self.message_count = 0;

        // Subscribe to channels
        self.subscribe().await?;

        Ok(())
    }

    /// Subscribe to all required channels
    async fn subscribe(&mut self) -> Result<()> {
        let stream = self.stream.as_mut()
            .context("Not connected")?;

        // Subscribe to L2 book
        let l2_sub = SubscriptionRequest::l2_book(&self.symbol);
        let msg = serde_json::to_string(&l2_sub)?;
        stream.send(Message::Text(msg)).await?;
        debug!(symbol = %self.symbol, "Subscribed to l2Book");

        // Subscribe to trades
        let trades_sub = SubscriptionRequest::trades(&self.symbol);
        let msg = serde_json::to_string(&trades_sub)?;
        stream.send(Message::Text(msg)).await?;
        debug!(symbol = %self.symbol, "Subscribed to trades");

        // Subscribe to asset context
        let ctx_sub = SubscriptionRequest::active_asset_ctx(&self.symbol);
        let msg = serde_json::to_string(&ctx_sub)?;
        stream.send(Message::Text(msg)).await?;
        debug!(symbol = %self.symbol, "Subscribed to activeAssetCtx");

        info!(symbol = %self.symbol, "All subscriptions complete");

        Ok(())
    }

    /// Receive the next message
    pub async fn recv(&mut self) -> Result<Option<WsMessage>> {
        // Connect if not connected
        if self.stream.is_none() {
            self.connect().await?;
        }

        let stream = self.stream.as_mut()
            .context("Not connected")?;

        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                self.last_message_at = Some(std::time::Instant::now());
                match parse_ws_message(&text) {
                    Some(msg) => {
                        self.message_count += 1;
                        if !self.first_message_logged {
                            self.first_message_logged = true;
                            let elapsed = self.connected_at
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.0);
                            info!(
                                symbol = %self.symbol,
                                elapsed_s = format!("{:.1}", elapsed),
                                "First WebSocket data message received"
                            );
                        }
                        Ok(Some(msg))
                    }
                    None => {
                        debug!(text = %text, "Unparseable message");
                        Ok(None)
                    }
                }
            }
            Some(Ok(Message::Ping(data))) => {
                self.last_message_at = Some(std::time::Instant::now());
                stream.send(Message::Pong(data)).await?;
                Ok(None)
            }
            Some(Ok(Message::Pong(_))) => {
                self.last_message_at = Some(std::time::Instant::now());
                self.pong_received = true;
                debug!(symbol = %self.symbol, "Pong received");
                Ok(None)
            }
            Some(Ok(Message::Close(frame))) => {
                warn!(symbol = %self.symbol, ?frame, "WebSocket closed by server");
                self.stream = None;
                Ok(None)
            }
            Some(Ok(Message::Binary(_))) => {
                self.last_message_at = Some(std::time::Instant::now());
                Ok(None)
            }
            Some(Ok(Message::Frame(_))) => Ok(None),
            Some(Err(e)) => {
                error!(symbol = %self.symbol, ?e, "WebSocket error");
                self.stream = None;
                Err(e.into())
            }
            None => {
                warn!(symbol = %self.symbol, "WebSocket stream ended");
                self.stream = None;
                Ok(None)
            }
        }
    }

    /// Reconnect with exponential backoff
    pub async fn reconnect(&mut self) -> Result<()> {
        self.stream = None;
        self.reconnect_attempts += 1;

        // Calculate delay with exponential backoff
        let delay_ms = std::cmp::min(
            self.config.reconnect_delay_ms * (1 << self.reconnect_attempts.min(10)),
            self.config.max_reconnect_delay_ms,
        );

        warn!(
            symbol = %self.symbol,
            attempt = self.reconnect_attempts,
            delay_ms,
            "Reconnecting after delay"
        );

        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

        self.connect().await
    }

    /// Send a ping frame to keep the connection alive.
    /// Returns true if ping was sent, false if not connected.
    pub async fn send_ping(&mut self) -> Result<bool> {
        if let Some(ref mut stream) = self.stream {
            let payload = b"keepalive".to_vec();
            match stream.send(Message::Ping(payload)).await {
                Ok(()) => {
                    self.last_ping_sent = Some(std::time::Instant::now());
                    self.pong_received = false;
                    debug!(symbol = %self.symbol, "Ping sent");
                    Ok(true)
                }
                Err(e) => {
                    warn!(symbol = %self.symbol, ?e, "Failed to send ping");
                    self.stream = None;
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Check if the connection is stale (no messages received recently).
    /// A connection is stale if no data has arrived for 2x the ping interval.
    pub fn is_stale(&self) -> bool {
        if self.stream.is_none() {
            return true;
        }
        let stale_threshold = std::time::Duration::from_millis(self.config.ping_interval_ms * 3);
        match self.last_message_at {
            Some(t) => t.elapsed() > stale_threshold,
            None => {
                // Never received a message — stale if connected for > threshold
                self.connected_at
                    .map(|t| t.elapsed() > stale_threshold)
                    .unwrap_or(true)
            }
        }
    }

    /// Check if a sent ping timed out (no pong received within ping_interval).
    pub fn ping_timed_out(&self) -> bool {
        if self.pong_received {
            return false;
        }
        match self.last_ping_sent {
            Some(t) => t.elapsed() > std::time::Duration::from_millis(self.config.ping_interval_ms),
            None => false,
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Get the number of parsed data messages received since last connect
    pub fn message_count(&self) -> u64 {
        self.message_count
    }

    /// Get seconds elapsed since connection was established
    pub fn elapsed_since_connect(&self) -> f64 {
        self.connected_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Seconds since last message (any frame type)
    pub fn seconds_since_last_message(&self) -> f64 {
        self.last_message_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(self.elapsed_since_connect())
    }
}

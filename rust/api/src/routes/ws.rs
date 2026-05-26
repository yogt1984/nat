//! WebSocket routes for real-time streaming

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::state::AppState;

/// GET /ws/stream/:symbol
/// WebSocket endpoint for real-time feature streaming
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> impl IntoResponse {
    let symbol = symbol.to_uppercase();
    info!("WebSocket connection request for {}", symbol);

    ws.on_upgrade(move |socket| handle_feature_stream(socket, state, symbol))
}

async fn handle_feature_stream(socket: WebSocket, state: Arc<AppState>, symbol: String) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to Redis channel for this symbol
    let channel = format!("nat:features:{}", symbol);

    // Create a subscription
    let mut pubsub = match state.redis.subscribe(&channel).await {
        Ok(ps) => ps,
        Err(e) => {
            error!("Failed to subscribe to {}: {}", channel, e);
            let _ = sender
                .send(Message::Text(format!(
                    r#"{{"error": "Failed to subscribe: {}"}}"#,
                    e
                )))
                .await;
            return;
        }
    };

    info!("WebSocket connected and subscribed to {}", channel);

    // Handle incoming messages (for ping/pong or client commands)
    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => {
                    debug!("Client closed connection");
                    break;
                }
                Ok(Message::Ping(_)) => {
                    debug!("Received ping");
                }
                Ok(Message::Text(text)) => {
                    debug!("Received text message: {}", text);
                }
                Err(e) => {
                    debug!("WebSocket receive error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Forward Redis messages to WebSocket
    let mut stream = pubsub.on_message();
    loop {
        tokio::select! {
            msg = stream.next() => {
                match msg {
                    Some(msg) => {
                        let payload: String = msg.get_payload().unwrap_or_default();
                        if sender.send(Message::Text(payload)).await.is_err() {
                            debug!("Failed to send message, client disconnected");
                            break;
                        }
                    }
                    None => {
                        debug!("Redis subscription ended");
                        break;
                    }
                }
            }
            _ = &mut recv_task => {
                debug!("Receiver task ended");
                break;
            }
        }
    }

    info!("WebSocket disconnected from {}", symbol);
}

/// GET /ws/alerts
/// WebSocket endpoint for alert streaming
pub async fn alerts_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("WebSocket connection request for alerts");
    ws.on_upgrade(move |socket| handle_alerts_stream(socket, state))
}

async fn handle_alerts_stream(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    let channel = "nat:alerts";

    let mut pubsub = match state.redis.subscribe(channel).await {
        Ok(ps) => ps,
        Err(e) => {
            error!("Failed to subscribe to alerts: {}", e);
            let _ = sender
                .send(Message::Text(format!(
                    r#"{{"error": "Failed to subscribe: {}"}}"#,
                    e
                )))
                .await;
            return;
        }
    };

    info!("Alerts WebSocket connected");

    // Handle incoming messages
    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }
    });

    // Forward Redis messages to WebSocket
    let mut stream = pubsub.on_message();
    loop {
        tokio::select! {
            msg = stream.next() => {
                match msg {
                    Some(msg) => {
                        let payload: String = msg.get_payload().unwrap_or_default();
                        if sender.send(Message::Text(payload)).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
            _ = &mut recv_task => {
                break;
            }
        }
    }

    info!("Alerts WebSocket disconnected");
}

/// GET /ws/research
/// WebSocket endpoint for real-time research event streaming via Redis Streams.
/// Uses XREADGROUP with consumer group for reliable delivery and replay on reconnect.
/// Events: hypothesis_started, gate_passed, gate_failed, hypothesis_registered, cycle_completed
pub async fn research_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("WebSocket connection request for research events");
    ws.on_upgrade(move |socket| handle_research_stream(socket, state))
}

const RESEARCH_STREAM: &str = "nat:research:stream";
const RESEARCH_GROUP: &str = "nat-api";

async fn handle_research_stream(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Ensure consumer group exists (idempotent)
    if let Err(e) = state
        .redis
        .ensure_consumer_group(RESEARCH_STREAM, RESEARCH_GROUP)
        .await
    {
        error!("Failed to create consumer group: {}", e);
        let _ = sender
            .send(Message::Text(format!(
                r#"{{"error": "Failed to setup stream: {}"}}"#,
                e
            )))
            .await;
        return;
    }

    // Each WebSocket connection gets a unique consumer name
    let consumer = format!("ws-{}", uuid::Uuid::new_v4());
    info!("Research WebSocket connected (consumer: {})", consumer);

    let mut recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }
    });

    loop {
        tokio::select! {
            // Read from stream with 2s block timeout (allows checking recv_task)
            messages = state.redis.xread_group(
                RESEARCH_GROUP, &consumer, RESEARCH_STREAM, 10, 2000
            ) => {
                match messages {
                    Ok(msgs) if !msgs.is_empty() => {
                        let mut ack_ids = Vec::new();
                        for (id, payload) in &msgs {
                            // Extract correlation IDs for structured logging
                            if let Ok(event) = serde_json::from_str::<serde_json::Value>(payload) {
                                let event_type = event.get("event").and_then(|v| v.as_str()).unwrap_or("unknown");
                                let hypothesis_id = event.get("hypothesis_id")
                                    .or_else(|| event.get("id"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("-");
                                let cycle_id = event.get("cycle_id").and_then(|v| v.as_str()).unwrap_or("-");
                                debug!(
                                    event = event_type,
                                    hypothesis_id = hypothesis_id,
                                    cycle_id = cycle_id,
                                    "forwarding research event to WebSocket"
                                );
                            }
                            if sender.send(Message::Text(payload.clone())).await.is_err() {
                                return; // client disconnected
                            }
                            ack_ids.push(id.clone());
                        }
                        // ACK processed messages
                        let _ = state.redis.xack(RESEARCH_STREAM, RESEARCH_GROUP, &ack_ids).await;
                    }
                    Ok(_) => {} // empty (timeout), loop again
                    Err(e) => {
                        warn!("XREADGROUP error: {}, retrying...", e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
            _ = &mut recv_task => {
                break;
            }
        }
    }

    info!("Research WebSocket disconnected (consumer: {})", consumer);
}

//! WebSocket handlers for dashboard endpoints.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tracing::{debug, warn};

use super::state::DashboardState;

/// Handler for /ws/logs endpoint
pub async fn logs_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_logs_socket(socket, state))
}

/// Handle a logs WebSocket connection
async fn handle_logs_socket(socket: WebSocket, state: Arc<DashboardState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send backfill of recent logs
    let backfill = state.get_log_backfill();
    for log in backfill {
        if let Ok(json) = serde_json::to_string(&log) {
            if sender.send(Message::Text(json)).await.is_err() {
                return;
            }
        }
    }

    // Subscribe to new logs
    let mut rx = state.log_tx.subscribe();

    // Spawn task to handle incoming messages (for ping/pong)
    let mut ping_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    // Stream new logs until client disconnects or channel closes
    loop {
        tokio::select! {
            result = rx.recv() => {
                match result {
                    Ok(log) => {
                        if let Ok(json) = serde_json::to_string(&log) {
                            if sender.send(Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Log subscriber lagged by {} messages", n);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            _ = &mut ping_task => {
                break;
            }
        }
    }

    debug!("Logs WebSocket connection closed");
}

/// Handler for /ws/state endpoint
pub async fn state_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_state_socket(socket, state))
}

/// Handle a state WebSocket connection
async fn handle_state_socket(socket: WebSocket, state: Arc<DashboardState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send initial state
    let snapshot = state.snapshot();
    if let Ok(json) = serde_json::to_string(&snapshot) {
        if sender.send(Message::Text(json)).await.is_err() {
            return;
        }
    }

    // Subscribe to state updates
    let mut rx = state.state_tx.subscribe();

    // Spawn task to handle incoming messages
    let mut ping_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    // Stream state updates until client disconnects or channel closes
    loop {
        tokio::select! {
            result = rx.recv() => {
                match result {
                    Ok(snapshot) => {
                        if let Ok(json) = serde_json::to_string(&snapshot) {
                            if sender.send(Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("State subscriber lagged by {} updates", n);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            _ = &mut ping_task => {
                break;
            }
        }
    }

    debug!("State WebSocket connection closed");
}

/// Handler for /api/symbols endpoint
pub async fn symbols_handler(
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let symbols = state.symbols.read();
    let symbol_list: Vec<String> = symbols.keys().cloned().collect();
    axum::Json(symbol_list)
}

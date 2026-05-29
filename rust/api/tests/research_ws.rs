//! Integration tests for the /ws/research WebSocket endpoint.
//!
//! These tests require a running Redis instance. Skip with:
//!   cargo test --package nat-api -- --skip research_ws
//!
//! The test spawns a temporary Axum server on a random port, connects
//! a WebSocket client, publishes events via XADD to the research stream,
//! and asserts they arrive on the WebSocket within a timeout.

use axum::{routing::get, Router};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

use nat_api::config::ApiConfig;
use nat_api::redis_client::RedisClient;
use nat_api::routes;
use nat_api::state::AppState;

const RESEARCH_STREAM: &str = "nat:research:stream";

/// Spawn a test Axum server on a random port, return the base URL.
async fn spawn_test_server() -> Result<String, Box<dyn std::error::Error>> {
    let config = ApiConfig::from_env();
    let redis = RedisClient::new(&config.redis_url).await?;
    let state = Arc::new(AppState::new(redis, config));

    let app = Router::new()
        .route("/ws/research", get(routes::ws::research_websocket_handler))
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    Ok(format!("127.0.0.1:{}", addr.port()))
}

/// Get a Redis connection for publishing test events via XADD.
async fn get_redis_publisher() -> Result<redis::aio::ConnectionManager, Box<dyn std::error::Error>>
{
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let client = redis::Client::open(url)?;
    let conn = redis::aio::ConnectionManager::new(client).await?;
    Ok(conn)
}

/// Publish an event to the research stream via XADD.
async fn xadd_event(
    publisher: &mut redis::aio::ConnectionManager,
    event: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    redis::cmd("XADD")
        .arg(RESEARCH_STREAM)
        .arg("MAXLEN")
        .arg("~")
        .arg("10000")
        .arg("*")
        .arg("event")
        .arg(event.to_string())
        .query_async::<_, String>(publisher)
        .await?;
    Ok(())
}

#[tokio::test]
async fn test_ws_connect() {
    let addr = match spawn_test_server().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping test (Redis not available): {}", e);
            return;
        }
    };

    let url = format!("ws://{}/ws/research", addr);
    let (ws_stream, response) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("Failed to connect WebSocket");

    assert_eq!(response.status(), 101);

    // Clean close
    let (mut write, _read) = ws_stream.split();
    write
        .send(tokio_tungstenite::tungstenite::Message::Close(None))
        .await
        .ok();
}

#[tokio::test]
async fn test_hypothesis_started_event() {
    let addr = match spawn_test_server().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping test (Redis not available): {}", e);
            return;
        }
    };
    let mut publisher = match get_redis_publisher().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let url = format!("ws://{}/ws/research", addr);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("Failed to connect");
    let (_write, mut read) = ws_stream.split();

    // Allow consumer group setup to complete
    tokio::time::sleep(Duration::from_millis(200)).await;

    let event = serde_json::json!({
        "event": "hypothesis_started",
        "id": "h_test_001",
        "agent": "micro",
        "claim": "test claim"
    });
    xadd_event(&mut publisher, &event)
        .await
        .expect("Failed to XADD");

    let msg = tokio::time::timeout(Duration::from_secs(5), read.next())
        .await
        .expect("Timeout waiting for WS message")
        .expect("Stream ended")
        .expect("WS error");

    let text = msg.to_text().expect("Not text");
    let parsed: serde_json::Value = serde_json::from_str(text).expect("Not JSON");
    assert_eq!(parsed["event"], "hypothesis_started");
    assert_eq!(parsed["id"], "h_test_001");
    assert_eq!(parsed["agent"], "micro");
}

#[tokio::test]
async fn test_gate_passed_event() {
    let addr = match spawn_test_server().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping test (Redis not available): {}", e);
            return;
        }
    };
    let mut publisher = match get_redis_publisher().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let url = format!("ws://{}/ws/research", addr);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("Failed to connect");
    let (_write, mut read) = ws_stream.split();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let event = serde_json::json!({
        "event": "gate_passed",
        "id": "h_test_002",
        "gate": "G2_temporal",
        "msg": "2/2 dates passed"
    });
    xadd_event(&mut publisher, &event).await.unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), read.next())
        .await
        .expect("Timeout")
        .expect("Stream ended")
        .expect("WS error");

    let parsed: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    assert_eq!(parsed["event"], "gate_passed");
    assert_eq!(parsed["gate"], "G2_temporal");
}

#[tokio::test]
async fn test_gate_failed_event() {
    let addr = match spawn_test_server().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping test (Redis not available): {}", e);
            return;
        }
    };
    let mut publisher = match get_redis_publisher().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let url = format!("ws://{}/ws/research", addr);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("Failed to connect");
    let (_write, mut read) = ws_stream.split();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let event = serde_json::json!({
        "event": "gate_failed",
        "id": "h_test_003",
        "gate": "G3_symbol",
        "reason": "only 1/3 symbols passed"
    });
    xadd_event(&mut publisher, &event).await.unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), read.next())
        .await
        .expect("Timeout")
        .expect("Stream ended")
        .expect("WS error");

    let parsed: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    assert_eq!(parsed["event"], "gate_failed");
    assert_eq!(parsed["reason"], "only 1/3 symbols passed");
}

#[tokio::test]
async fn test_cycle_completed_event() {
    let addr = match spawn_test_server().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Skipping test (Redis not available): {}", e);
            return;
        }
    };
    let mut publisher = match get_redis_publisher().await {
        Ok(p) => p,
        Err(_) => return,
    };

    let url = format!("ws://{}/ws/research", addr);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("Failed to connect");
    let (_write, mut read) = ws_stream.split();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let event = serde_json::json!({
        "event": "cycle_completed",
        "agent": "micro",
        "tested": 8,
        "passed": 1,
        "cycle": 42
    });
    xadd_event(&mut publisher, &event).await.unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), read.next())
        .await
        .expect("Timeout")
        .expect("Stream ended")
        .expect("WS error");

    let parsed: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    assert_eq!(parsed["event"], "cycle_completed");
    assert_eq!(parsed["tested"], 8);
    assert_eq!(parsed["passed"], 1);
    assert_eq!(parsed["cycle"], 42);
}

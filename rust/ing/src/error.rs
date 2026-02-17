//! Error types for the ingestor

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IngError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Channel send error")]
    ChannelSend,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Connection closed")]
    ConnectionClosed,
}

pub type Result<T> = std::result::Result<T, IngError>;

//! Application State

use crate::config::ApiConfig;
use crate::redis_client::RedisClient;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tracing::warn;

/// Shared application state
pub struct AppState {
    pub redis: RedisClient,
    pub config: ApiConfig,
    /// SQLite connection for research data (WAL mode, read-only from API side).
    pub research_db: Option<Arc<TokioMutex<Connection>>>,
}

impl AppState {
    pub fn new(redis: RedisClient, config: ApiConfig) -> Self {
        let research_db = open_research_db(&config.research_db_path);
        Self {
            redis,
            config,
            research_db,
        }
    }
}

/// Open SQLite database in WAL mode (read-only access for the API).
/// Returns None if the database doesn't exist or can't be opened.
fn open_research_db(path: &str) -> Option<Arc<TokioMutex<Connection>>> {
    match Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(conn) => {
            // Ensure WAL mode for concurrent reads
            let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
            Some(Arc::new(TokioMutex::new(conn)))
        }
        Err(e) => {
            warn!("Could not open research DB at {}: {} — falling back to JSON", path, e);
            None
        }
    }
}

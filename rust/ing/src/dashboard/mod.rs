//! Real-time dashboard for monitoring ingestor state and logs.
//!
//! Provides WebSocket endpoints for streaming logs and market state
//! to a web-based dashboard.

mod handlers;
mod log_layer;
pub mod server;
pub mod state;

pub use log_layer::BroadcastLayer;
pub use server::run_dashboard_server;
pub use state::{DashboardState, StateSnapshot, SymbolState};

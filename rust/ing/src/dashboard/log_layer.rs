//! Tracing subscriber layer for broadcasting logs to dashboard.

use std::sync::Arc;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use super::state::{DashboardState, LogEntry};

/// A tracing layer that broadcasts log events to the dashboard.
pub struct BroadcastLayer {
    state: Arc<DashboardState>,
}

impl BroadcastLayer {
    pub fn new(state: Arc<DashboardState>) -> Self {
        Self { state }
    }
}

/// Visitor to extract message from tracing event
struct MessageVisitor {
    message: String,
    fields: Vec<(String, String)>,
}

impl MessageVisitor {
    fn new() -> Self {
        Self {
            message: String::new(),
            fields: Vec::new(),
        }
    }
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        } else {
            self.fields.push((field.name().to_string(), format!("{:?}", value)));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields.push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }
}

impl<S> Layer<S> for BroadcastLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let level = metadata.level().to_string();
        let target = metadata.target().to_string();

        // Extract message and fields
        let mut visitor = MessageVisitor::new();
        event.record(&mut visitor);

        // Build message with fields if message is empty
        let message = if visitor.message.is_empty() {
            visitor
                .fields
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ")
        } else if visitor.fields.is_empty() {
            visitor.message
        } else {
            format!(
                "{} {}",
                visitor.message,
                visitor
                    .fields
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };

        let entry = LogEntry {
            timestamp: chrono::Utc::now().timestamp_millis(),
            level,
            target,
            message,
        };

        self.state.add_log(entry);
    }
}

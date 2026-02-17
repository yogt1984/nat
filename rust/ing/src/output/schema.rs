//! Arrow schema definition for feature output

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use crate::features::Features;

/// Create the Arrow schema for feature output
pub fn create_schema() -> Arc<Schema> {
    let mut fields = vec![
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence_id", DataType::UInt64, false),
    ];

    // Add feature fields
    for name in Features::names() {
        fields.push(Field::new(name, DataType::Float64, false));
    }

    Arc::new(Schema::new(fields))
}

/// Get column names
pub fn column_names() -> Vec<String> {
    let mut names = vec![
        "timestamp_ns".to_string(),
        "symbol".to_string(),
        "sequence_id".to_string(),
    ];

    for name in Features::names() {
        names.push(name.to_string());
    }

    names
}

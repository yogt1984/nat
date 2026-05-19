//! Output module for writing features to Parquet

mod schema;
mod writer;

pub use schema::{create_schema, create_schema_with_alg_features};
pub use writer::ParquetWriter;

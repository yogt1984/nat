//! Output module for writing features to Parquet

mod schema;
mod writer;

pub use schema::create_schema;
pub use writer::ParquetWriter;

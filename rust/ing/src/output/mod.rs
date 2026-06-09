//! Output module for writing features and trades to Parquet

mod schema;
mod trade_writer;
mod writer;

pub use schema::{create_schema, create_schema_with_alg_features};
pub use trade_writer::{TradeParquetWriter, TradeRecord};
pub use writer::ParquetWriter;

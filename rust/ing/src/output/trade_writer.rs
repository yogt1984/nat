//! Parquet writer for raw trade output
//!
//! Persists individual trades (price, size, side, timestamp, tid) to Parquet
//! files in `data/trades/`, enabling event-driven fill simulation downstream.

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Timelike, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

use crate::config::TradeOutputConfig;

/// A single trade record to be written to Parquet.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub tid: u64,
    pub price: f64,
    pub size: f64,
    pub is_buy: bool,
}

/// Create the Parquet schema for raw trades.
fn create_trade_schema() -> Arc<Schema> {
    let fields = vec![
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("tid", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("size", DataType::Float64, false),
        Field::new("is_buy", DataType::Boolean, false),
    ];

    let mut metadata = HashMap::new();
    metadata.insert("schema_version".to_string(), "1".to_string());
    metadata.insert("data_type".to_string(), "raw_trades".to_string());

    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Buffer for accumulating trades before writing.
struct TradeBuffer {
    timestamps: Vec<i64>,
    symbols: Vec<String>,
    tids: Vec<u64>,
    prices: Vec<f64>,
    sizes: Vec<f64>,
    is_buy: Vec<bool>,
    capacity: usize,
}

impl TradeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            symbols: Vec::with_capacity(capacity),
            tids: Vec::with_capacity(capacity),
            prices: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
            is_buy: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, rec: &TradeRecord) {
        self.timestamps.push(rec.timestamp_ns);
        self.symbols.push(rec.symbol.clone());
        self.tids.push(rec.tid);
        self.prices.push(rec.price);
        self.sizes.push(rec.size);
        self.is_buy.push(rec.is_buy);
    }

    fn len(&self) -> usize {
        self.timestamps.len()
    }

    fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    fn clear(&mut self) {
        self.timestamps.clear();
        self.symbols.clear();
        self.tids.clear();
        self.prices.clear();
        self.sizes.clear();
        self.is_buy.clear();
    }

    fn to_record_batch(&self) -> Result<RecordBatch> {
        let schema = create_trade_schema();
        let mut columns: Vec<ArrayRef> = Vec::new();

        let mut ts_builder = Int64Builder::with_capacity(self.len());
        for ts in &self.timestamps {
            ts_builder.append_value(*ts);
        }
        columns.push(Arc::new(ts_builder.finish()));

        let mut sym_builder = StringBuilder::with_capacity(self.len(), self.len() * 5);
        for s in &self.symbols {
            sym_builder.append_value(s);
        }
        columns.push(Arc::new(sym_builder.finish()));

        let mut tid_builder = UInt64Builder::with_capacity(self.len());
        for tid in &self.tids {
            tid_builder.append_value(*tid);
        }
        columns.push(Arc::new(tid_builder.finish()));

        let mut px_builder = Float64Builder::with_capacity(self.len());
        for px in &self.prices {
            px_builder.append_value(*px);
        }
        columns.push(Arc::new(px_builder.finish()));

        let mut sz_builder = Float64Builder::with_capacity(self.len());
        for sz in &self.sizes {
            sz_builder.append_value(*sz);
        }
        columns.push(Arc::new(sz_builder.finish()));

        let mut buy_builder = BooleanBuilder::with_capacity(self.len());
        for b in &self.is_buy {
            buy_builder.append_value(*b);
        }
        columns.push(Arc::new(buy_builder.finish()));

        RecordBatch::try_new(schema, columns).context("Failed to create trade record batch")
    }
}

/// Parquet writer for raw trades with buffering and hourly file rotation.
pub struct TradeParquetWriter {
    data_dir: PathBuf,
    compression: String,
    buffer: TradeBuffer,
    current_file: Option<ArrowWriter<File>>,
    current_file_path: Option<PathBuf>,
    current_tmp_path: Option<PathBuf>,
    current_hour: Option<u32>,
    rows_written: usize,
}

impl TradeParquetWriter {
    /// Create a new trade writer.
    pub fn new(config: &TradeOutputConfig, general_data_dir: &str) -> Result<Self> {
        let data_dir = config
            .data_dir
            .as_deref()
            .unwrap_or(general_data_dir);

        fs::create_dir_all(data_dir).context("Failed to create trade data directory")?;

        Ok(Self {
            data_dir: PathBuf::from(data_dir),
            compression: config.compression.clone(),
            buffer: TradeBuffer::new(config.buffer_size),
            current_file: None,
            current_file_path: None,
            current_tmp_path: None,
            current_hour: None,
            rows_written: 0,
        })
    }

    /// Write a trade record.
    pub fn write(&mut self, rec: &TradeRecord) -> Result<()> {
        let now: DateTime<Utc> = DateTime::from_timestamp_nanos(rec.timestamp_ns);
        let hour = now.hour();

        if self.current_hour != Some(hour) {
            self.rotate_file(&now)?;
            self.current_hour = Some(hour);
        }

        self.buffer.push(rec);

        if self.buffer.is_full() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.len() == 0 {
            return Ok(());
        }

        // Disk space check: 6 columns × ~16 bytes average × 2× safety
        let estimated_bytes = (self.buffer.len() * 6 * 16) as u64;
        let min_required = estimated_bytes.saturating_mul(2);
        if let Ok(avail) = fs2::available_space(&self.data_dir) {
            if avail < min_required {
                error!(
                    available_bytes = avail,
                    required_bytes = min_required,
                    buffer_rows = self.buffer.len(),
                    "Trade writer: disk space too low — skipping flush"
                );
                return Ok(());
            }
        }

        let batch = self.buffer.to_record_batch()?;

        if let Some(writer) = &mut self.current_file {
            writer.write(&batch)?;
            writer.flush()?;
            self.rows_written += batch.num_rows();

            info!(
                rows_flushed = batch.num_rows(),
                rows_in_file = self.rows_written,
                "Trade buffer flushed"
            );
        }

        self.buffer.clear();
        Ok(())
    }

    fn rotate_file(&mut self, now: &DateTime<Utc>) -> Result<()> {
        self.close_current_file()?;

        let date_dir = self.data_dir.join(now.format("%Y-%m-%d").to_string());
        fs::create_dir_all(&date_dir)?;

        let filename = format!("{}.parquet", now.format("%Y%m%d_%H%M%S"));
        let file_path = date_dir.join(&filename);
        let tmp_path = date_dir.join(format!("{}.tmp", filename));

        info!(path = ?file_path, "Opening new trade Parquet file");

        let file = File::create(&tmp_path)?;
        let schema = create_trade_schema();

        let compression = match self.compression.as_str() {
            "zstd" => Compression::ZSTD(Default::default()),
            "snappy" => Compression::SNAPPY,
            "gzip" => Compression::GZIP(Default::default()),
            _ => Compression::UNCOMPRESSED,
        };

        let props = WriterProperties::builder()
            .set_compression(compression)
            .build();

        let writer = ArrowWriter::try_new(file, schema, Some(props))?;

        self.current_file = Some(writer);
        self.current_file_path = Some(file_path);
        self.current_tmp_path = Some(tmp_path);
        self.rows_written = 0;

        Ok(())
    }

    fn close_current_file(&mut self) -> Result<()> {
        self.flush_buffer()?;

        if let Some(writer) = self.current_file.take() {
            writer.close()?;

            if let (Some(tmp), Some(final_path)) =
                (self.current_tmp_path.take(), &self.current_file_path)
            {
                if tmp.exists() {
                    fs::rename(&tmp, final_path)?;
                    info!(path = ?final_path, rows = self.rows_written, "Closed trade Parquet file");
                }
            }
        } else {
            self.current_tmp_path = None;
        }

        self.current_file_path = None;
        Ok(())
    }

    /// Flush and close.
    pub fn flush(&mut self) -> Result<()> {
        self.close_current_file()
    }
}

impl Drop for TradeParquetWriter {
    fn drop(&mut self) {
        if let Err(e) = self.close_current_file() {
            tracing::error!(?e, "Error closing trade Parquet file in drop");
        }
    }
}

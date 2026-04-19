//! Parquet writer for feature output

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder, UInt64Builder};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc, Timelike};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, debug};

use crate::config::OutputConfig;
use crate::features::Features;
use crate::FeatureVector;
use super::schema::create_schema;

/// Parquet writer with buffering and file rotation
pub struct ParquetWriter {
    config: OutputConfig,
    data_dir: PathBuf,
    buffer: FeatureBuffer,
    current_file: Option<ArrowWriter<File>>,
    current_file_path: Option<PathBuf>,
    current_hour: Option<u32>,
    rows_written: usize,
}

/// Buffer for accumulating features before writing
struct FeatureBuffer {
    timestamps: Vec<i64>,
    symbols: Vec<String>,
    sequence_ids: Vec<u64>,
    features: Vec<Vec<f64>>,
    capacity: usize,
}

impl FeatureBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            symbols: Vec::with_capacity(capacity),
            sequence_ids: Vec::with_capacity(capacity),
            features: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, fv: &FeatureVector) {
        self.timestamps.push(fv.timestamp_ns);
        self.symbols.push(fv.symbol.clone());
        self.sequence_ids.push(fv.sequence_id);
        self.features.push(fv.features.to_vec());
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
        self.sequence_ids.clear();
        self.features.clear();
    }

    fn to_record_batch(&self) -> Result<RecordBatch> {
        let schema = create_schema();

        // Build arrays
        let mut columns: Vec<ArrayRef> = Vec::new();

        // Timestamp
        let mut ts_builder = Int64Builder::with_capacity(self.len());
        for ts in &self.timestamps {
            ts_builder.append_value(*ts);
        }
        columns.push(Arc::new(ts_builder.finish()));

        // Symbol
        let mut symbol_builder = StringBuilder::with_capacity(self.len(), self.len() * 5);
        for symbol in &self.symbols {
            symbol_builder.append_value(symbol);
        }
        columns.push(Arc::new(symbol_builder.finish()));

        // Sequence ID
        let mut seq_builder = UInt64Builder::with_capacity(self.len());
        for seq in &self.sequence_ids {
            seq_builder.append_value(*seq);
        }
        columns.push(Arc::new(seq_builder.finish()));

        // Feature columns
        let n_features = if self.features.is_empty() {
            Features::count()
        } else {
            self.features[0].len()
        };

        for i in 0..n_features {
            let mut builder = Float64Builder::with_capacity(self.len());
            for feature_vec in &self.features {
                builder.append_value(feature_vec.get(i).copied().unwrap_or(0.0));
            }
            columns.push(Arc::new(builder.finish()));
        }

        RecordBatch::try_new(schema, columns)
            .context("Failed to create record batch")
    }
}

impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn new(config: &OutputConfig, general_data_dir: &str) -> Result<Self> {
        let data_dir = config.data_dir.as_deref()
            .unwrap_or(general_data_dir);

        fs::create_dir_all(data_dir)
            .context("Failed to create data directory")?;

        Ok(Self {
            config: config.clone(),
            data_dir: PathBuf::from(data_dir),
            buffer: FeatureBuffer::new(config.row_group_size),
            current_file: None,
            current_file_path: None,
            current_hour: None,
            rows_written: 0,
        })
    }

    /// Write a feature vector
    pub fn write(&mut self, fv: &FeatureVector) -> Result<()> {
        // Check if we need to rotate files
        let now: DateTime<Utc> = DateTime::from_timestamp_nanos(fv.timestamp_ns);
        let hour = now.hour();

        if self.current_hour != Some(hour) {
            self.rotate_file(&now)?;
            self.current_hour = Some(hour);
        }

        // Add to buffer
        self.buffer.push(fv);

        // Flush if buffer is full
        if self.buffer.is_full() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Flush the buffer to disk
    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.len() == 0 {
            return Ok(());
        }

        let batch = self.buffer.to_record_batch()?;

        if let Some(writer) = &mut self.current_file {
            writer.write(&batch)?;
            self.rows_written += batch.num_rows();
            debug!(rows = batch.num_rows(), total = self.rows_written, "Wrote batch");
        }

        self.buffer.clear();
        Ok(())
    }

    /// Rotate to a new file
    fn rotate_file(&mut self, now: &DateTime<Utc>) -> Result<()> {
        // Close current file
        self.close_current_file()?;

        // Create new file path
        let date_dir = self.data_dir.join(now.format("%Y-%m-%d").to_string());
        fs::create_dir_all(&date_dir)?;

        let filename = format!("{}.parquet", now.format("%Y%m%d_%H%M%S"));
        let file_path = date_dir.join(filename);

        info!(path = ?file_path, "Opening new Parquet file");

        // Create writer
        let file = File::create(&file_path)?;
        let schema = create_schema();

        let compression = match self.config.compression.as_str() {
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
        self.rows_written = 0;

        Ok(())
    }

    /// Close the current file
    fn close_current_file(&mut self) -> Result<()> {
        // Flush remaining buffer
        self.flush_buffer()?;

        if let Some(writer) = self.current_file.take() {
            writer.close()?;
            if let Some(path) = &self.current_file_path {
                info!(path = ?path, rows = self.rows_written, "Closed Parquet file");
            }
        }

        self.current_file_path = None;
        Ok(())
    }

    /// Flush and close
    pub fn flush(&mut self) -> Result<()> {
        self.close_current_file()
    }
}

impl Drop for ParquetWriter {
    fn drop(&mut self) {
        if let Err(e) = self.close_current_file() {
            tracing::error!(?e, "Error closing Parquet file in drop");
        }
    }
}

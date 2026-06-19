//! Parquet writer for feature output

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder, UInt64Builder};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Timelike, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{error, info};

use crate::config::OutputConfig;
use crate::features::Features;
use crate::FeatureVector;

/// Parquet writer with buffering and file rotation
pub struct ParquetWriter {
    config: OutputConfig,
    data_dir: PathBuf,
    buffer: FeatureBuffer,
    current_file: Option<ArrowWriter<File>>,
    current_file_path: Option<PathBuf>,
    /// Temporary path for atomic writes; renamed to current_file_path on close.
    current_tmp_path: Option<PathBuf>,
    current_hour: Option<u32>,
    rows_written: usize,
    file_opened_at: Option<std::time::Instant>,
    last_progress_pct: usize,
    /// Algorithm feature names for extended schema
    alg_feature_names: Vec<&'static str>,
    /// Counter for disk-full flush skips
    disk_full_skips: AtomicU64,
    /// Max wall-clock age of an open file before rotation (parsed from
    /// `config.rotate_interval`). Bounds the data orphaned in a `.tmp` if the
    /// process dies before `close()`. Effective cap is ~1h via the hour trigger.
    rotate_interval: std::time::Duration,
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
        let mut row = fv.features.to_vec();
        row.extend_from_slice(&fv.alg_values);
        self.features.push(row);
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

    fn to_record_batch(&self, alg_feature_names: &[&str]) -> Result<RecordBatch> {
        let schema = super::schema::create_schema_with_alg_features(alg_feature_names);

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

        RecordBatch::try_new(schema, columns).context("Failed to create record batch")
    }
}

/// Decide whether to rotate the open Parquet file: on an hour change (preserves
/// the per-hour file layout) OR once the file has been open for `interval`
/// (bounds the data orphaned in a `.tmp` if the process dies before `close()`).
pub(crate) fn should_rotate(
    current_hour: Option<u32>,
    now_hour: u32,
    file_age: Option<std::time::Duration>,
    interval: std::time::Duration,
) -> bool {
    current_hour != Some(now_hour) || file_age.is_some_and(|age| age >= interval)
}

impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn new(config: &OutputConfig, general_data_dir: &str) -> Result<Self> {
        Self::new_with_alg_features(config, general_data_dir, Vec::new())
    }

    /// Create a new Parquet writer with algorithm feature columns
    pub fn new_with_alg_features(
        config: &OutputConfig,
        general_data_dir: &str,
        alg_feature_names: Vec<&'static str>,
    ) -> Result<Self> {
        let data_dir = config.data_dir.as_deref().unwrap_or(general_data_dir);

        fs::create_dir_all(data_dir).context("Failed to create data directory")?;

        Ok(Self {
            config: config.clone(),
            data_dir: PathBuf::from(data_dir),
            buffer: FeatureBuffer::new(config.row_group_size),
            current_file: None,
            current_file_path: None,
            current_tmp_path: None,
            current_hour: None,
            rows_written: 0,
            file_opened_at: None,
            last_progress_pct: 0,
            alg_feature_names,
            disk_full_skips: AtomicU64::new(0),
            rotate_interval: crate::config::parse_rotate_interval(&config.rotate_interval),
        })
    }

    /// Write a feature vector
    pub fn write(&mut self, fv: &FeatureVector) -> Result<()> {
        // Rotate on hour change, or once the file has been open for rotate_interval
        // (bounds data lost in an orphaned .tmp on an unclean stop).
        let now: DateTime<Utc> = DateTime::from_timestamp_nanos(fv.timestamp_ns);
        let hour = now.hour();
        let file_age = self.file_opened_at.map(|t| t.elapsed());

        if should_rotate(self.current_hour, hour, file_age, self.rotate_interval) {
            self.rotate_file(&now)?;
            self.current_hour = Some(hour);
        }

        // Add to buffer
        self.buffer.push(fv);

        // Log buffer progress at 25% milestones
        let pct = (self.buffer.len() * 100) / self.buffer.capacity;
        let milestone = (pct / 25) * 25;
        if milestone > 0 && milestone > self.last_progress_pct && milestone < 100 {
            let elapsed = self
                .file_opened_at
                .map(|t| format!("{:.0}s", t.elapsed().as_secs_f64()))
                .unwrap_or_else(|| "?".to_string());
            info!(
                buffer_pct = milestone,
                rows = self.buffer.len(),
                capacity = self.buffer.capacity,
                elapsed,
                "Writer buffer progress"
            );
            self.last_progress_pct = milestone;
        }

        // Flush if buffer is full
        if self.buffer.is_full() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Check available disk space at the data directory.
    /// Returns available bytes, or None if the check fails.
    fn available_disk_space(&self) -> Option<u64> {
        use fs2::available_space;
        available_space(&self.data_dir).ok()
    }

    /// Number of times flush was skipped due to low disk space
    pub fn disk_full_skip_count(&self) -> u64 {
        self.disk_full_skips.load(Ordering::Relaxed)
    }

    /// Flush the buffer to disk
    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffer.len() == 0 {
            return Ok(());
        }

        // Pre-flush disk space check: estimate batch size as ~8 bytes per f64 field per row
        let estimated_bytes = (self.buffer.len() * 220 * 8) as u64; // conservative: 220 columns × 8 bytes
        let min_required = estimated_bytes.saturating_mul(2); // 2× safety margin
        if let Some(avail) = self.available_disk_space() {
            if avail < min_required {
                let skips = self.disk_full_skips.fetch_add(1, Ordering::Relaxed) + 1;
                error!(
                    available_bytes = avail,
                    required_bytes = min_required,
                    buffer_rows = self.buffer.len(),
                    total_skips = skips,
                    "Disk space too low — skipping flush to preserve buffer"
                );
                // Don't clear buffer — data stays in memory for next attempt
                return Ok(());
            }
        }

        let batch = self.buffer.to_record_batch(&self.alg_feature_names)?;

        if let Some(writer) = &mut self.current_file {
            writer.write(&batch)?;
            // Force the row group to disk so the file is non-zero immediately
            writer.flush()?;
            self.rows_written += batch.num_rows();

            // Log flush with the size of the file actually being written (the
            // .tmp; the final .parquet does not exist until the rename at close).
            if let Some(ref tmp_path) = self.current_tmp_path {
                let file_size = fs::metadata(tmp_path).map(|m| m.len()).unwrap_or(0);
                let elapsed = self
                    .file_opened_at
                    .map(|t| format!("{:.0}s", t.elapsed().as_secs_f64()))
                    .unwrap_or_else(|| "?".to_string());
                info!(
                    rows_flushed = batch.num_rows(),
                    rows_in_file = self.rows_written,
                    file_size_bytes = file_size,
                    elapsed,
                    path = ?self.current_file_path,
                    "Buffer flushed to disk"
                );
            }
        }

        self.buffer.clear();
        self.last_progress_pct = 0;
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
        let file_path = date_dir.join(&filename);
        let tmp_path = date_dir.join(format!("{}.tmp", filename));

        info!(path = ?file_path, "Opening new Parquet file");

        // Write to .tmp file; will be renamed to final path on close
        let file = File::create(&tmp_path)?;
        let schema = super::schema::create_schema_with_alg_features(&self.alg_feature_names);

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
        self.current_tmp_path = Some(tmp_path);
        self.rows_written = 0;
        self.file_opened_at = Some(std::time::Instant::now());
        self.last_progress_pct = 0;

        Ok(())
    }

    /// Close the current file
    fn close_current_file(&mut self) -> Result<()> {
        // Flush remaining buffer
        self.flush_buffer()?;

        if let Some(writer) = self.current_file.take() {
            writer.close()?;

            // Atomic rename: .tmp → final path
            if let (Some(tmp), Some(final_path)) =
                (self.current_tmp_path.take(), &self.current_file_path)
            {
                if tmp.exists() {
                    fs::rename(&tmp, final_path)?;
                    info!(path = ?final_path, rows = self.rows_written, "Closed Parquet file");
                }
            } else if let Some(path) = &self.current_file_path {
                // Fallback: no tmp path (shouldn't happen, but be safe)
                info!(path = ?path, rows = self.rows_written, "Closed Parquet file");
            }
        } else {
            self.current_tmp_path = None;
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

#[cfg(test)]
mod tests {
    use super::{should_rotate, ParquetWriter};
    use std::time::Duration;

    const TEN_MIN: Duration = Duration::from_secs(600);

    #[test]
    fn rotates_on_first_write() {
        // No file open yet (current_hour None) -> open one regardless of age.
        assert!(should_rotate(None, 13, None, TEN_MIN));
    }

    #[test]
    fn rotates_on_hour_change() {
        assert!(should_rotate(Some(13), 14, Some(Duration::from_secs(5)), TEN_MIN));
    }

    #[test]
    fn rotates_when_interval_elapsed() {
        assert!(should_rotate(Some(13), 13, Some(Duration::from_secs(600)), TEN_MIN));
        assert!(should_rotate(Some(13), 13, Some(Duration::from_secs(900)), TEN_MIN));
    }

    #[test]
    fn holds_within_same_hour_and_interval() {
        assert!(!should_rotate(Some(13), 13, Some(Duration::from_secs(599)), TEN_MIN));
        assert!(!should_rotate(Some(13), 13, Some(Duration::ZERO), TEN_MIN));
    }

    /// Real-parquet smoke: a sub-hourly `rotate_interval` must finalize the open
    /// file (footer + atomic rename to `*.parquet`) on the elapsed trigger, not
    /// only on the hour boundary — so an unclean stop can orphan at most ~interval.
    /// Writes through the real ParquetWriter and reads the results back.
    #[test]
    fn subhourly_rotation_writes_readable_parquet() {
        use parquet::file::reader::FileReader;
        use std::fs::File;

        let dir = std::env::temp_dir().join(format!("ing_rot_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = crate::config::OutputConfig {
            format: "parquet".to_string(),
            row_group_size: 10,
            compression: "zstd".to_string(),
            rotate_interval: "1s".to_string(),
            data_dir: Some(dir.to_string_lossy().into_owned()),
        };
        let mk = |ts: i64| crate::FeatureVector {
            timestamp_ns: ts,
            symbol: "BTC".to_string(),
            sequence_id: 0,
            features: ing_features::Features::default(),
            alg_values: Vec::new(),
        };
        let base: i64 = 1_767_225_600_000_000_000; // 2026-01-01T00:00:00Z, one hour
        {
            let mut w = ParquetWriter::new(&cfg, dir.to_str().unwrap()).unwrap();
            for i in 0..12 {
                w.write(&mk(base + i)).unwrap(); // fills buffer -> flush to file1.tmp
            }
            std::thread::sleep(Duration::from_millis(1100));
            // Advance data-time by 2s so the rotated file's %H%M%S name differs
            // from file1's (real ingestion timestamps always advance; only this
            // synthetic same-instant test would otherwise collide the names).
            let later = base + 2_000_000_000;
            w.write(&mk(later)).unwrap(); // elapsed >= 1s -> rotate: file1 -> .parquet
            for i in 0..12 {
                w.write(&mk(later + i)).unwrap();
            }
            w.flush().unwrap(); // close file2 -> .parquet
        }

        let mut parquets = Vec::new();
        for date_dir in std::fs::read_dir(&dir).unwrap().flatten() {
            if date_dir.path().is_dir() {
                for f in std::fs::read_dir(date_dir.path()).unwrap().flatten() {
                    let p = f.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("parquet") {
                        parquets.push(p);
                    }
                }
            }
        }
        assert!(
            parquets.len() >= 2,
            "elapsed-trigger should have rotated mid-hour; got {} .parquet files",
            parquets.len()
        );
        let mut total_rows = 0i64;
        for p in &parquets {
            // SerializedFileReader reads the footer -> fails on a footer-less .tmp,
            // so this asserts each rotated file is a complete, readable parquet.
            let rdr = parquet::file::reader::SerializedFileReader::new(File::open(p).unwrap())
                .expect("rotated file must be a complete, readable parquet");
            total_rows += rdr.metadata().file_metadata().num_rows();
        }
        assert_eq!(total_rows, 25, "all written rows should be recoverable");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

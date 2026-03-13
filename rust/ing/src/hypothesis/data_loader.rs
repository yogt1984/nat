//! Parquet Data Loader for Hypothesis Testing
//!
//! Loads feature data from Parquet files produced by the ingestor
//! and converts to hypothesis test input format.

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, StringArray, TimestampNanosecondArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::h1_whale_flow::H1TestData;
use super::h2_entropy_whale::H2TestData;
use super::h3_liquidation_cascade::H3TestData;
use super::h4_concentration_vol::H4TestData;
use super::h5_persistence::FeatureRow;

/// H5 test data with features and price series
pub struct H5TestData {
    pub features: Vec<FeatureRow>,
    pub prices: Vec<f64>,
    pub volatility: Option<Vec<f64>>,
}

/// Load all Parquet files from a directory
pub fn load_all_data(data_dir: &Path) -> Result<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    let mut file_count = 0;

    let entries: Vec<_> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "parquet")
                .unwrap_or(false)
        })
        .collect();

    for entry in entries.iter() {
        let path = entry.path();
        file_count += 1;

        if file_count % 10 == 0 {
            println!("      Loaded {} files...", file_count);
        }

        let file = File::open(&path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("Failed to create reader for {}", path.display()))?;

        let mut reader = builder.build()?;

        while let Some(batch) = reader.next() {
            let batch = batch
                .with_context(|| format!("Failed to read batch from {}", path.display()))?;
            batches.push(batch);
        }
    }

    Ok(batches)
}

/// Extract a Float64 column from all batches
pub fn extract_f64_column(batches: &[RecordBatch], column_name: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();

    for batch in batches {
        if let Some(col) = batch.column_by_name(column_name) {
            if let Some(array) = col.as_any().downcast_ref::<Float64Array>() {
                for i in 0..array.len() {
                    if !array.is_null(i) {
                        values.push(array.value(i));
                    } else {
                        // Use NaN for null values to maintain alignment
                        values.push(f64::NAN);
                    }
                }
            } else {
                anyhow::bail!(
                    "Column '{}' exists but is not Float64 type",
                    column_name
                );
            }
        } else {
            anyhow::bail!("Column '{}' not found in Parquet data", column_name);
        }
    }

    Ok(values)
}

/// Extract timestamps
pub fn extract_timestamps(batches: &[RecordBatch]) -> Result<Vec<i64>> {
    let mut values = Vec::new();

    for batch in batches {
        if let Some(col) = batch.column_by_name("timestamp") {
            if let Some(array) = col.as_any().downcast_ref::<TimestampNanosecondArray>() {
                for i in 0..array.len() {
                    if !array.is_null(i) {
                        values.push(array.value(i));
                    } else {
                        values.push(0); // Use 0 for null timestamps
                    }
                }
            }
        }
    }

    if values.is_empty() {
        anyhow::bail!("No timestamps found in Parquet data");
    }

    Ok(values)
}

/// Extract symbols
pub fn extract_symbols(batches: &[RecordBatch]) -> Result<Vec<String>> {
    let mut values = Vec::new();

    for batch in batches {
        if let Some(col) = batch.column_by_name("symbol") {
            if let Some(array) = col.as_any().downcast_ref::<StringArray>() {
                for i in 0..array.len() {
                    if !array.is_null(i) {
                        values.push(array.value(i).to_string());
                    } else {
                        values.push(String::new());
                    }
                }
            }
        }
    }

    Ok(values)
}

/// Filter data by symbol
pub fn filter_by_symbol(
    timestamps: &[i64],
    symbols: &[String],
    data_cols: &HashMap<String, Vec<f64>>,
    target_symbol: &str,
) -> Result<(Vec<i64>, HashMap<String, Vec<f64>>)> {
    let mut filtered_timestamps = Vec::new();
    let mut filtered_data: HashMap<String, Vec<f64>> = HashMap::new();

    // Initialize filtered data vectors
    for key in data_cols.keys() {
        filtered_data.insert(key.clone(), Vec::new());
    }

    for (i, symbol) in symbols.iter().enumerate() {
        if symbol == target_symbol {
            filtered_timestamps.push(timestamps[i]);

            for (key, values) in data_cols.iter() {
                if let Some(filtered_vec) = filtered_data.get_mut(key) {
                    filtered_vec.push(values[i]);
                }
            }
        }
    }

    Ok((filtered_timestamps, filtered_data))
}

/// Load H1 test data (Whale Flow Predicts Returns)
pub fn load_h1_data(batches: &[RecordBatch]) -> Result<H1TestData> {
    println!("      Extracting features for H1...");

    let timestamps = extract_timestamps(batches)?;

    // Load whale flows at different windows
    let whale_flow_1h = extract_f64_column(batches, "whale_net_flow_1h")
        .or_else(|_| extract_f64_column(batches, "whale_flow_1h"))
        .context("Missing whale flow 1h column")?;

    let whale_flow_4h = extract_f64_column(batches, "whale_net_flow_4h")
        .or_else(|_| extract_f64_column(batches, "whale_flow_4h"))
        .context("Missing whale flow 4h column")?;

    let whale_flow_24h = extract_f64_column(batches, "whale_net_flow_24h")
        .or_else(|_| extract_f64_column(batches, "whale_flow_24h"))
        .context("Missing whale flow 24h column")?;

    // Load returns at different horizons
    let returns_1h = extract_f64_column(batches, "returns_1h")
        .or_else(|_| extract_f64_column(batches, "forward_returns_1h"))
        .context("Missing returns 1h column")?;

    let returns_4h = extract_f64_column(batches, "returns_4h")
        .or_else(|_| extract_f64_column(batches, "forward_returns_4h"))
        .context("Missing returns 4h column")?;

    let returns_24h = extract_f64_column(batches, "returns_24h")
        .or_else(|_| extract_f64_column(batches, "forward_returns_24h"))
        .context("Missing returns 24h column")?;

    // Filter out NaN values
    let valid_indices: Vec<usize> = (0..timestamps.len())
        .filter(|&i| {
            !whale_flow_1h[i].is_nan()
            && !whale_flow_4h[i].is_nan()
            && !whale_flow_24h[i].is_nan()
            && !returns_1h[i].is_nan()
            && !returns_4h[i].is_nan()
            && !returns_24h[i].is_nan()
            && timestamps[i] > 0
        })
        .collect();

    let filtered_whale_flow_1h: Vec<f64> = valid_indices.iter().map(|&i| whale_flow_1h[i]).collect();
    let filtered_whale_flow_4h: Vec<f64> = valid_indices.iter().map(|&i| whale_flow_4h[i]).collect();
    let filtered_whale_flow_24h: Vec<f64> = valid_indices.iter().map(|&i| whale_flow_24h[i]).collect();
    let filtered_returns_1h: Vec<f64> = valid_indices.iter().map(|&i| returns_1h[i]).collect();
    let filtered_returns_4h: Vec<f64> = valid_indices.iter().map(|&i| returns_4h[i]).collect();
    let filtered_returns_24h: Vec<f64> = valid_indices.iter().map(|&i| returns_24h[i]).collect();
    let filtered_timestamps: Vec<i64> = valid_indices.iter().map(|&i| timestamps[i]).collect();

    println!(
        "      ✓ Loaded {} valid samples for H1",
        filtered_timestamps.len()
    );

    Ok(H1TestData {
        whale_flow_1h: filtered_whale_flow_1h,
        whale_flow_4h: filtered_whale_flow_4h,
        whale_flow_24h: filtered_whale_flow_24h,
        returns_1h: filtered_returns_1h,
        returns_4h: filtered_returns_4h,
        returns_24h: filtered_returns_24h,
        timestamps_ms: filtered_timestamps,
    })
}

/// Load H2 test data (Entropy + Whale Interaction)
pub fn load_h2_data(batches: &[RecordBatch]) -> Result<H2TestData> {
    println!("      Extracting features for H2...");

    let timestamps = extract_timestamps(batches)?;

    let whale_flows = extract_f64_column(batches, "whale_net_flow_1h")
        .or_else(|_| extract_f64_column(batches, "whale_flow_1h"))
        .context("Missing whale flow column")?;

    let entropies = extract_f64_column(batches, "tick_entropy_1s")
        .or_else(|_| extract_f64_column(batches, "entropy_1s"))
        .context("Missing entropy column (tried: tick_entropy_1s, entropy_1s)")?;

    let returns = extract_f64_column(batches, "returns_1m")
        .context("Missing returns_1m column")?;

    // Filter out NaN values
    let valid_indices: Vec<usize> = (0..timestamps.len())
        .filter(|&i| {
            !whale_flows[i].is_nan()
                && !entropies[i].is_nan()
                && !returns[i].is_nan()
                && timestamps[i] > 0
        })
        .collect();

    let filtered_whale_flows: Vec<f64> = valid_indices.iter().map(|&i| whale_flows[i]).collect();
    let filtered_entropies: Vec<f64> = valid_indices.iter().map(|&i| entropies[i]).collect();
    let filtered_returns: Vec<f64> = valid_indices.iter().map(|&i| returns[i]).collect();
    let filtered_timestamps: Vec<i64> = valid_indices.iter().map(|&i| timestamps[i]).collect();

    println!(
        "      ✓ Loaded {} valid samples for H2",
        filtered_timestamps.len()
    );

    Ok(H2TestData {
        whale_flow: filtered_whale_flows,
        entropy: filtered_entropies,
        returns: filtered_returns,
    })
}

/// Load H3 test data (Liquidation Cascades)
pub fn load_h3_data(batches: &[RecordBatch]) -> Result<H3TestData> {
    println!("      Extracting features for H3...");

    let timestamps = extract_timestamps(batches)?;

    // Load liquidation risk above at different distance buckets [1%, 2%, 5%, 10%]
    let liq_above_1pct = extract_f64_column(batches, "liq_risk_above_1pct")
        .context("Missing liq_risk_above_1pct column")?;
    let liq_above_2pct = extract_f64_column(batches, "liq_risk_above_2pct")
        .context("Missing liq_risk_above_2pct column")?;
    let liq_above_5pct = extract_f64_column(batches, "liq_risk_above_5pct")
        .context("Missing liq_risk_above_5pct column")?;
    let liq_above_10pct = extract_f64_column(batches, "liq_risk_above_10pct")
        .context("Missing liq_risk_above_10pct column")?;

    // Load liquidation risk below at different distance buckets [1%, 2%, 5%, 10%]
    let liq_below_1pct = extract_f64_column(batches, "liq_risk_below_1pct")
        .context("Missing liq_risk_below_1pct column")?;
    let liq_below_2pct = extract_f64_column(batches, "liq_risk_below_2pct")
        .context("Missing liq_risk_below_2pct column")?;
    let liq_below_5pct = extract_f64_column(batches, "liq_risk_below_5pct")
        .context("Missing liq_risk_below_5pct column")?;
    let liq_below_10pct = extract_f64_column(batches, "liq_risk_below_10pct")
        .context("Missing liq_risk_below_10pct column")?;

    let prices = extract_f64_column(batches, "midprice")
        .or_else(|_| extract_f64_column(batches, "mid_price"))
        .context("Missing price column (tried: midprice, mid_price)")?;

    // Build liquidation arrays and filter NaN values
    let mut liquidation_above = Vec::new();
    let mut liquidation_below = Vec::new();
    let mut filtered_prices = Vec::new();
    let mut filtered_timestamps = Vec::new();

    for i in 0..timestamps.len() {
        // Skip if any value is NaN
        if liq_above_1pct[i].is_nan()
            || liq_above_2pct[i].is_nan()
            || liq_above_5pct[i].is_nan()
            || liq_above_10pct[i].is_nan()
            || liq_below_1pct[i].is_nan()
            || liq_below_2pct[i].is_nan()
            || liq_below_5pct[i].is_nan()
            || liq_below_10pct[i].is_nan()
            || prices[i].is_nan()
            || timestamps[i] == 0
        {
            continue;
        }

        liquidation_above.push([
            liq_above_1pct[i],
            liq_above_2pct[i],
            liq_above_5pct[i],
            liq_above_10pct[i],
        ]);

        liquidation_below.push([
            liq_below_1pct[i],
            liq_below_2pct[i],
            liq_below_5pct[i],
            liq_below_10pct[i],
        ]);

        filtered_prices.push(prices[i]);
        filtered_timestamps.push(timestamps[i]);
    }

    println!(
        "      ✓ Loaded {} valid samples for H3",
        filtered_timestamps.len()
    );

    Ok(H3TestData {
        liquidation_above,
        liquidation_below,
        prices: filtered_prices,
        timestamps_ms: filtered_timestamps,
    })
}

/// Load H4 test data (Concentration Predicts Volatility)
pub fn load_h4_data(batches: &[RecordBatch]) -> Result<H4TestData> {
    println!("      Extracting features for H4...");

    let timestamps = extract_timestamps(batches)?;

    let hhi = extract_f64_column(batches, "hhi")
        .context("Missing hhi column")?;

    let gini = extract_f64_column(batches, "gini_coefficient")
        .or_else(|_| extract_f64_column(batches, "gini"))
        .context("Missing gini coefficient column")?;

    let theil = extract_f64_column(batches, "theil_index")
        .or_else(|_| extract_f64_column(batches, "theil"))
        .context("Missing theil index column")?;

    let top10 = extract_f64_column(batches, "top10_concentration")
        .context("Missing top10_concentration column")?;

    let top20 = extract_f64_column(batches, "top20_concentration")
        .context("Missing top20_concentration column")?;

    let current_volatility = extract_f64_column(batches, "realized_vol_1m")
        .or_else(|_| extract_f64_column(batches, "volatility_1m"))
        .context("Missing realized volatility column")?;

    let prices = extract_f64_column(batches, "midprice")
        .or_else(|_| extract_f64_column(batches, "mid_price"))
        .context("Missing price column")?;

    // Filter out NaN values
    let valid_indices: Vec<usize> = (0..timestamps.len())
        .filter(|&i| {
            !hhi[i].is_nan()
                && !gini[i].is_nan()
                && !theil[i].is_nan()
                && !top10[i].is_nan()
                && !top20[i].is_nan()
                && !current_volatility[i].is_nan()
                && !prices[i].is_nan()
                && timestamps[i] > 0
        })
        .collect();

    let filtered_hhi: Vec<f64> = valid_indices.iter().map(|&i| hhi[i]).collect();
    let filtered_gini: Vec<f64> = valid_indices.iter().map(|&i| gini[i]).collect();
    let filtered_theil: Vec<f64> = valid_indices.iter().map(|&i| theil[i]).collect();
    let filtered_top10: Vec<f64> = valid_indices.iter().map(|&i| top10[i]).collect();
    let filtered_top20: Vec<f64> = valid_indices.iter().map(|&i| top20[i]).collect();
    let filtered_current_volatility: Vec<f64> =
        valid_indices.iter().map(|&i| current_volatility[i]).collect();
    let filtered_prices: Vec<f64> = valid_indices.iter().map(|&i| prices[i]).collect();
    let filtered_timestamps: Vec<i64> = valid_indices.iter().map(|&i| timestamps[i]).collect();

    println!(
        "      ✓ Loaded {} valid samples for H4",
        filtered_timestamps.len()
    );

    Ok(H4TestData {
        hhi: filtered_hhi,
        gini: filtered_gini,
        theil: filtered_theil,
        top10: filtered_top10,
        top20: filtered_top20,
        current_volatility: filtered_current_volatility,
        prices: filtered_prices,
        timestamps_ms: filtered_timestamps,
    })
}

/// Load H5 test data (Persistence Indicator)
pub fn load_h5_data(batches: &[RecordBatch]) -> Result<H5TestData> {
    println!("      Extracting features for H5...");

    // Load the 6 features required for persistence indicator
    let entropy = extract_f64_column(batches, "tick_entropy_1s")
        .or_else(|_| extract_f64_column(batches, "entropy_1s"))
        .context("Missing entropy column")?;

    let momentum = extract_f64_column(batches, "momentum_1m")
        .or_else(|_| extract_f64_column(batches, "momentum"))
        .context("Missing momentum column")?;

    let monotonicity = extract_f64_column(batches, "monotonicity_1m")
        .or_else(|_| extract_f64_column(batches, "monotonicity"))
        .context("Missing monotonicity column")?;

    let hurst = extract_f64_column(batches, "hurst_1m")
        .or_else(|_| extract_f64_column(batches, "hurst"))
        .context("Missing hurst column")?;

    let ofi = extract_f64_column(batches, "ofi_1s")
        .or_else(|_| extract_f64_column(batches, "ofi"))
        .context("Missing ofi column")?;

    let illiquidity = extract_f64_column(batches, "amihud_illiquidity_1m")
        .or_else(|_| extract_f64_column(batches, "illiquidity"))
        .context("Missing illiquidity column")?;

    // Load prices for return calculation
    let price_data = extract_f64_column(batches, "midprice")
        .or_else(|_| extract_f64_column(batches, "mid_price"))
        .context("Missing price column")?;

    // Load volatility (optional)
    let volatility_data = extract_f64_column(batches, "realized_vol_1m")
        .or_else(|_| extract_f64_column(batches, "volatility_1m"))
        .ok();

    // Build feature rows and aligned price series
    let mut features = Vec::new();
    let mut prices = Vec::new();
    let mut volatility = Vec::new();
    let n = entropy.len();

    for i in 0..n {
        // Skip if any required value is NaN
        if entropy[i].is_nan()
            || momentum[i].is_nan()
            || monotonicity[i].is_nan()
            || hurst[i].is_nan()
            || ofi[i].is_nan()
            || illiquidity[i].is_nan()
            || price_data[i].is_nan()
        {
            continue;
        }

        features.push(FeatureRow {
            entropy: entropy[i],
            momentum: momentum[i],
            monotonicity: monotonicity[i],
            hurst: hurst[i],
            ofi: ofi[i],
            illiquidity: illiquidity[i],
        });

        prices.push(price_data[i]);

        if let Some(ref vol_data) = volatility_data {
            if !vol_data[i].is_nan() {
                volatility.push(vol_data[i]);
            }
        }
    }

    println!("      ✓ Loaded {} valid samples for H5", features.len());

    Ok(H5TestData {
        features,
        prices,
        volatility: if volatility.is_empty() { None } else { Some(volatility) },
    })
}

/// Get data summary statistics
pub fn summarize_data(batches: &[RecordBatch]) -> Result<()> {
    println!("\n📊 Data Summary:");
    println!("   Batches loaded: {}", batches.len());

    if batches.is_empty() {
        println!("   ⚠️  No data loaded!");
        return Ok(());
    }

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("   Total rows: {}", total_rows);

    // Get column names from first batch
    if let Some(first_batch) = batches.first() {
        println!("   Columns: {}", first_batch.num_columns());
        println!("   Column names:");
        for (i, field) in first_batch.schema().fields().iter().enumerate() {
            if i < 10 {
                // Show first 10
                println!("      - {}: {:?}", field.name(), field.data_type());
            }
        }
        if first_batch.num_columns() > 10 {
            println!("      ... and {} more", first_batch.num_columns() - 10);
        }
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore] // Only run when data exists
    fn test_load_sample_data() {
        let data_dir = PathBuf::from("../../data/features");
        if !data_dir.exists() {
            eprintln!("Data directory does not exist, skipping test");
            return;
        }

        let batches = load_all_data(&data_dir).unwrap();
        assert!(!batches.is_empty(), "Should load at least one batch");

        let timestamps = extract_timestamps(&batches).unwrap();
        assert!(!timestamps.is_empty(), "Should extract timestamps");

        println!("Loaded {} batches with {} rows total",
                 batches.len(),
                 batches.iter().map(|b| b.num_rows()).sum::<usize>());
    }

    #[test]
    #[ignore]
    fn test_load_h1_data() {
        let data_dir = PathBuf::from("../../data/features");
        if !data_dir.exists() {
            eprintln!("Data directory does not exist, skipping test");
            return;
        }

        let batches = load_all_data(&data_dir).unwrap();
        let h1_data = load_h1_data(&batches).unwrap();

        assert!(!h1_data.whale_flows.is_empty(), "Should have whale flow data");
        assert_eq!(
            h1_data.whale_flows.len(),
            h1_data.returns.len(),
            "Arrays should be same length"
        );
        assert_eq!(
            h1_data.whale_flows.len(),
            h1_data.timestamps.len(),
            "Arrays should be same length"
        );

        println!("H1 data: {} samples", h1_data.whale_flows.len());
    }
}

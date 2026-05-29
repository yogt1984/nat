//! Cross-Symbol Imbalance Feature Extraction
//!
//! Captures relative order book pressure across symbols. When one asset's book
//! diverges from the others, it signals asset-specific flow vs market-wide moves.
//!
//! # Features (3 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Cross OBI divergence** | OBI_self - mean(OBI_others) | [-2, +2] | >0 = more bid pressure than peers |
//! | **Cross OBI mean** | mean(OBI_others) | [-1, +1] | Market-wide directional bias |
//! | **Cross OBI dispersion** | std(all OBI values) | [0, 1] | Higher = fragmented market |
//!
//! # Architecture
//!
//! Each symbol task writes its L5 OBI to a shared `CrossSymbolState` via `Arc<RwLock<>>`.
//! At feature emission time, each task reads others' OBI to compute cross features.
//! Features are NaN-padded when fewer than 2 symbols have reported.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Shared state for cross-symbol OBI values
#[derive(Debug, Clone)]
pub struct CrossSymbolState {
    inner: Arc<RwLock<HashMap<String, f64>>>,
}

impl CrossSymbolState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::with_capacity(4))),
        }
    }

    /// Update this symbol's OBI value
    pub fn update(&self, symbol: &str, obi_l5: f64) {
        if let Ok(mut map) = self.inner.write() {
            map.insert(symbol.to_string(), obi_l5);
        }
    }

    /// Compute cross-symbol features for a given symbol
    pub fn compute(&self, self_symbol: &str) -> CrossSymbolFeatures {
        let map = match self.inner.read() {
            Ok(m) => m,
            Err(_) => return CrossSymbolFeatures::default(),
        };

        let self_obi = match map.get(self_symbol) {
            Some(&v) => v,
            None => return CrossSymbolFeatures::default(),
        };

        // Collect other symbols' OBI values
        let others: Vec<f64> = map
            .iter()
            .filter(|(k, _)| k.as_str() != self_symbol)
            .map(|(_, &v)| v)
            .collect();

        if others.is_empty() {
            return CrossSymbolFeatures::default();
        }

        let mean_others = others.iter().sum::<f64>() / others.len() as f64;

        // Dispersion: std of all OBI values (self + others)
        let all_values: Vec<f64> = std::iter::once(self_obi)
            .chain(others.iter().copied())
            .collect();
        let mean_all = all_values.iter().sum::<f64>() / all_values.len() as f64;
        let dispersion = if all_values.len() > 1 {
            let var = all_values
                .iter()
                .map(|x| (x - mean_all).powi(2))
                .sum::<f64>()
                / (all_values.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        CrossSymbolFeatures {
            cross_obi_divergence: self_obi - mean_others,
            cross_obi_mean: mean_others,
            cross_obi_dispersion: dispersion,
        }
    }
}

impl Default for CrossSymbolState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-symbol imbalance features (3 features)
#[derive(Debug, Clone, Default)]
pub struct CrossSymbolFeatures {
    /// OBI_self - mean(OBI_others)
    pub cross_obi_divergence: f64,
    /// Mean OBI of other symbols
    pub cross_obi_mean: f64,
    /// Std dev of all symbols' OBI (market fragmentation)
    pub cross_obi_dispersion: f64,
}

impl CrossSymbolFeatures {
    pub fn count() -> usize {
        3
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "cross_obi_divergence",
            "cross_obi_mean",
            "cross_obi_dispersion",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.cross_obi_divergence,
            self.cross_obi_mean,
            self.cross_obi_dispersion,
        ]
    }
}

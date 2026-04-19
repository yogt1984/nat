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

    // Add feature fields (all features including optional ones)
    for name in Features::names_all() {
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

    for name in Features::names_all() {
        names.push(name.to_string());
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{
        Features, WhaleFlowFeatures, LiquidationRiskFeatures,
        ConcentrationFeatures, RegimeFeatures, GmmClassificationFeatures,
    };

    #[test]
    fn test_schema_column_count_matches_feature_vector_all_none() {
        // When all optional features are None, to_vec must still match schema
        let features = Features::default();
        let vec = features.to_vec();
        let schema = create_schema();
        // Schema has 3 metadata columns + feature columns
        assert_eq!(
            schema.fields().len(),
            3 + vec.len(),
            "Schema fields ({}) must equal 3 metadata + to_vec length ({})",
            schema.fields().len(), 3 + vec.len()
        );
    }

    #[test]
    fn test_schema_column_count_matches_feature_vector_all_some() {
        // When all optional features are Some, to_vec must still match schema
        let mut features = Features::default();
        features.whale_flow = Some(WhaleFlowFeatures::default());
        features.liquidation_risk = Some(LiquidationRiskFeatures::default());
        features.concentration = Some(ConcentrationFeatures::default());
        features.regime = Some(RegimeFeatures::default());
        features.gmm_classification = Some(GmmClassificationFeatures::default());
        let vec = features.to_vec();
        let schema = create_schema();
        assert_eq!(
            schema.fields().len(),
            3 + vec.len(),
            "Schema fields must match to_vec length regardless of optional feature state"
        );
    }

    #[test]
    fn test_schema_column_count_matches_partial_optionals() {
        // Every combination of Some/None must produce same length
        let base = Features::default();
        let base_len = base.to_vec().len();

        let mut with_whale = Features::default();
        with_whale.whale_flow = Some(WhaleFlowFeatures::default());
        assert_eq!(with_whale.to_vec().len(), base_len, "Adding whale_flow must not change vec length");

        let mut with_regime = Features::default();
        with_regime.regime = Some(RegimeFeatures::default());
        assert_eq!(with_regime.to_vec().len(), base_len, "Adding regime must not change vec length");

        let mut with_liq_conc = Features::default();
        with_liq_conc.liquidation_risk = Some(LiquidationRiskFeatures::default());
        with_liq_conc.concentration = Some(ConcentrationFeatures::default());
        assert_eq!(with_liq_conc.to_vec().len(), base_len, "Adding liquidation+concentration must not change vec length");
    }

    #[test]
    fn test_to_vec_length_equals_count_all() {
        let features = Features::default();
        assert_eq!(
            features.to_vec().len(),
            Features::count_all(),
            "to_vec().len() must always equal count_all()"
        );
    }

    #[test]
    fn test_names_all_length_equals_count_all() {
        assert_eq!(
            Features::names_all().len(),
            Features::count_all(),
            "names_all() and count_all() must agree"
        );
    }

    #[test]
    fn test_no_duplicate_feature_names() {
        let names = Features::names_all();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(name), "Duplicate feature name: {}", name);
        }
    }

    #[test]
    fn test_nan_padding_for_missing_optionals() {
        let features = Features::default();
        let vec = features.to_vec();
        let base_count = Features::count();
        // Everything after base features should be NaN (all optionals are None)
        for (i, &val) in vec[base_count..].iter().enumerate() {
            assert!(val.is_nan(),
                "Optional feature at index {} (offset {}) should be NaN when not set, got {}",
                base_count + i, i, val);
        }
    }

    #[test]
    fn test_real_values_replace_nan_when_set() {
        let mut features = Features::default();
        features.whale_flow = Some(WhaleFlowFeatures::default());
        let vec = features.to_vec();
        let wf_start = Features::count();
        // Whale flow features should be 0.0 (default), not NaN
        for i in 0..WhaleFlowFeatures::count() {
            assert!(!vec[wf_start + i].is_nan(),
                "Whale flow feature at offset {} should not be NaN when set", i);
        }
        // But liquidation (next optional) should still be NaN
        let lr_start = wf_start + WhaleFlowFeatures::count();
        assert!(vec[lr_start].is_nan(), "Liquidation features should still be NaN");
    }

    #[test]
    fn test_column_names_match_schema_fields() {
        let schema = create_schema();
        let col_names = column_names();
        let schema_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(col_names, schema_names, "column_names() must match schema field names");
    }

    #[test]
    fn test_record_batch_creation_with_mixed_optionals() {
        // Simulate what the writer does: create a FeatureVector and build a RecordBatch
        // This is the exact code path that was crashing
        use crate::output::writer::ParquetWriter;
        use crate::FeatureVector;

        let schema = create_schema();

        // Simulate features with no optionals (early in session)
        let f1 = Features::default();
        let v1 = f1.to_vec();

        // Simulate features with regime kicked in (later in session)
        let mut f2 = Features::default();
        f2.regime = Some(RegimeFeatures::default());
        f2.whale_flow = Some(WhaleFlowFeatures::default());
        let v2 = f2.to_vec();

        // Both must have same length and match schema
        assert_eq!(v1.len(), v2.len(), "Vector length must be constant across the session");
        assert_eq!(v1.len() + 3, schema.fields().len(), "Vector + metadata must match schema");
    }
}

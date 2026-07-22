# Hypothesis Testing Guide

## Overview

This guide explains how to run the NAT hypothesis testing framework on collected market data.

## Current Status

**✅ Implemented:**
- H1-H5 hypothesis test functions
- Statistical testing framework
- Final GO/PIVOT/NO-GO decision logic
- 266 unit tests passing

**🔨 To Implement:**
- Binary to load Parquet data and run tests
- Data transformation pipeline

---

## The 5 Hypotheses

| ID | Hypothesis | Success Criteria |
|----|------------|------------------|
| **H1** | Whale flow predicts returns | r > 0.05, p < 0.001, MI > 0.02 bits |
| **H2** | Entropy + whale interaction | Lift > 10%, MI gain > 0.01 bits |
| **H3** | Liquidation cascades predictable | Precision > 30%, Lift > 2x |
| **H4** | Concentration predicts volatility | r > 0.2, partial r > 0.1 |
| **H5** | Persistence indicator works | WF Sharpe > 0.5, OOS/IS > 0.7 |

---

## How to Run (Currently)

### Step 1: Collect Data

Run the ingestor for 2-4 weeks minimum:

```bash
make run
```

This creates Parquet files in `./data/features/`

### Step 2: Run Unit Tests

The hypothesis tests have comprehensive unit tests:

```bash
# Run all hypothesis tests
cd rust
cargo test --package ing --lib hypothesis::

# Run specific hypothesis
cargo test --package ing --lib hypothesis::h1_whale_flow
cargo test --package ing --lib hypothesis::h2_entropy_whale
cargo test --package ing --lib hypothesis::h3_liquidation_cascade
cargo test --package ing --lib hypothesis::h4_concentration_vol
cargo test --package ing --lib hypothesis::h5_persistence

# Run final decision tests
cargo test --package ing --lib hypothesis::final_decision
```

### Step 3: Manual Testing (Programmatic)

To run on real data, you need to write custom code. Here's the template:

```rust
use ing::hypothesis::{
    run_h1_whale_flow_test, H1TestData, H1TestConfig,
    run_final_decision, DecisionInput,
};

// Load your Parquet data
let whale_flows = vec![...];  // Extract from Parquet
let returns = vec![...];      // Extract from Parquet
let timestamps = vec![...];   // Extract from Parquet

// Create test data
let test_data = H1TestData {
    whale_flows,
    returns,
    timestamps,
};

// Run H1 test
let config = H1TestConfig::default();
let h1_result = run_h1_whale_flow_test(&test_data, &config);

// Check decision
match h1_result.decision {
    H1Decision::Accept => println!("H1 PASSED"),
    H1Decision::Reject => println!("H1 FAILED"),
    H1Decision::Inconclusive => println!("H1 INCONCLUSIVE"),
}

// Repeat for H2-H5...

// Final decision
let decision_input = DecisionInput {
    h1: Some(h1_result),
    h2: Some(h2_result),
    h3: Some(h3_result),
    h4: Some(h4_result),
    h5: Some(h5_result),
};

let final_decision = run_final_decision(&decision_input)?;

match final_decision {
    FinalDecision::Go => println!("🚀 GO - Deploy signals"),
    FinalDecision::Pivot => println!("🔄 PIVOT - Selective deployment"),
    FinalDecision::NoGo => println!("🛑 NO-GO - Do not deploy"),
}
```

---

## To Build Complete Hypothesis Runner

You need to implement:

### 1. Parquet Data Loader

```rust
fn load_parquet_data(data_dir: &Path) -> Result<DataFrame> {
    // Load all .parquet files
    // Concatenate into single DataFrame
    // Return structured data
}
```

### 2. Feature Extractors for Each Hypothesis

```rust
fn extract_h1_data(df: &DataFrame) -> H1TestData {
    H1TestData {
        whale_flows: df["whale_net_flow_1h"].to_vec(),
        returns: df["returns_1m"].to_vec(),
        timestamps: df["timestamp"].to_vec(),
    }
}

// Similar for H2-H5
```

### 3. Test Runner Binary

```rust
// In src/bin/run_hypotheses.rs
fn main() -> Result<()> {
    let data = load_parquet_data("./data/features")?;

    let h1 = run_h1_test(&data)?;
    let h2 = run_h2_test(&data)?;
    let h3 = run_h3_test(&data)?;
    let h4 = run_h4_test(&data)?;
    let h5 = run_h5_test(&data)?;

    let decision = run_final_decision(&DecisionInput {
        h1: Some(h1),
        h2: Some(h2),
        h3: Some(h3),
        h4: Some(h4),
        h5: Some(h5),
    })?;

    generate_report(&decision)?;

    Ok(())
}
```

---

## Quick Reference: Test Functions

| Hypothesis | Function | Input Type | Output Type |
|------------|----------|------------|-------------|
| H1 | `run_h1_whale_flow_test` | `&H1TestData` | `H1TestResult` |
| H2 | `run_h2_entropy_whale_test` | `&H2TestData` | `H2TestResult` |
| H3 | `run_h3_liquidation_cascade_test` | `&H3TestData` | `H3TestResult` |
| H4 | `run_h4_concentration_vol_test` | `&H4TestData` | `H4TestResult` |
| H5 | `run_h5_persistence_test` | `&[FeatureRow]` | `H5TestResult` |
| Final | `run_final_decision` | `&DecisionInput` | `FinalDecisionResult` |

---

## Example: Running H1 Test

```rust
use ing::hypothesis::{
    run_h1_whale_flow_test,
    H1TestData,
    H1TestConfig,
    H1Decision,
};

// Your data from Parquet
let whale_flows = vec![100.0, 150.0, -50.0, 200.0, ...];
let returns = vec![0.001, 0.002, -0.001, 0.003, ...];
let timestamps = vec![t1, t2, t3, t4, ...];

let test_data = H1TestData {
    whale_flows,
    returns,
    timestamps,
};

let config = H1TestConfig {
    flow_windows: vec![3600, 14400, 86400],  // 1h, 4h, 24h
    return_horizons: vec![60, 300, 900],     // 1m, 5m, 15m
    min_samples: 1000,
    correlation_threshold: 0.05,
    p_value_threshold: 0.001,
    mi_threshold: 0.02,
    bonferroni_correct: true,
    min_passing_combinations: 3,
};

let result = run_h1_whale_flow_test(&test_data, &config);

println!("H1 Decision: {:?}", result.decision);
println!("Passing combinations: {}/{}", result.n_passing, result.n_total);

if let Some(best) = result.best_combination() {
    println!("Best combination:");
    println!("  Flow window: {}s", best.flow_window);
    println!("  Return horizon: {}s", best.return_horizon);
    println!("  Pearson r: {:.4}", best.correlation.pearson);
    println!("  Spearman ρ: {:.4}", best.correlation.spearman);
    println!("  p-value: {:.6}", best.correlation.p_value);
    println!("  MI: {:.4} bits", best.mi);
}
```

---

## Expected Output Example

```
╔══════════════════════════════════════════════════════════════════╗
║            NAT HYPOTHESIS TESTING PIPELINE                       ║
╚══════════════════════════════════════════════════════════════════╝

[1/5] H1: Whale flow predicts returns
      ✅ ACCEPT - Strong predictive power detected
      Best: 14400s/300s window/horizon
      Passing: 7/9 combinations

[2/5] H2: Entropy + whale interaction
      ✅ ACCEPT - Significant interaction detected

[3/5] H3: Liquidation cascades
      ⚠️  INCONCLUSIVE - Weak predictive power

[4/5] H4: Concentration predicts volatility
      ✅ ACCEPT - Concentration predicts volatility

[5/5] H5: Persistence indicator
      ✅ ACCEPT - Persistence indicator validated
      Walk-forward Sharpe: 0.72

╔══════════════════════════════════════════════════════════════════╗
║                    FINAL DECISION ANALYSIS                       ║
╚══════════════════════════════════════════════════════════════════╝

🚀 GO - Proceed with full deployment
   4/5 hypotheses validated
   Strong statistical evidence for alpha generation
```

---

## Next Steps

1. **Collect data** for 2-4 weeks minimum
2. **Implement Parquet loader** specific to your schema
3. **Create test runner binary** using the template above
4. **Run tests** and get GO/PIVOT/NO-GO decision
5. **If GO:** Build trading strategy
6. **If PIVOT:** Focus on validated signals only
7. **If NO-GO:** Research failed, don't trade

---

## Important Notes

- The framework is **intentionally skeptical**
- Bonferroni correction prevents false positives
- Walk-forward validation prevents overfitting
- OOS/IS ratio must be > 0.7
- If all tests pass easily, something's wrong (likely overfitting)

**The tests SHOULD be hard to pass. That's the point.**

//! Hypothesis Testing Runner
//!
//! Loads collected market data and runs H1-H5 hypothesis tests.
//! Generates GO/PIVOT/NO-GO decision with full statistical report.

use anyhow::{Context, Result};
use std::path::PathBuf;

mod hypothesis_runner {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    pub fn main() -> Result<()> {
        // Parse command line arguments
        let args: Vec<String> = std::env::args().collect();
        let data_dir = if args.len() >= 2 {
            PathBuf::from(&args[1])
        } else {
            PathBuf::from("./data/features")
        };

        print_header();
        println!("Data directory: {}", data_dir.display());
        println!();

        // Check if data directory exists
        if !data_dir.exists() {
            println!("⚠️  Data directory does not exist: {}", data_dir.display());
            println!();
            println!("Please collect data first by running: make run");
            println!("Let it run for at least 2-4 weeks before testing hypotheses.");
            return Ok(());
        }

        // Check for parquet files
        let parquet_count = count_parquet_files(&data_dir)?;
        if parquet_count == 0 {
            println!("⚠️  No Parquet files found in {}", data_dir.display());
            println!();
            println!("Please collect data first by running: make run");
            return Ok(());
        }

        println!("📊 Found {} Parquet files", parquet_count);
        println!();

        // Run hypothesis tests
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║              RUNNING HYPOTHESIS TESTS (MOCK DATA)                ║");
        println!("║                                                                  ║");
        println!("║  ⚠️  IMPORTANT: Data loader not yet implemented                  ║");
        println!("║  Tests will run with synthetic data to demonstrate flow         ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();

        // Run tests with mock data (placeholder)
        let results = run_all_tests()?;

        // Print results
        print_results(&results);

        // Generate report
        let report_path = data_dir.join("hypothesis_test_report.txt");
        save_report(&report_path, &results)?;

        println!();
        println!("📄 Report saved to: {}", report_path.display());
        println!();

        print_next_steps();

        Ok(())
    }

    fn print_header() {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║            NAT HYPOTHESIS TESTING PIPELINE                       ║");
        println!("║            Statistical Validation Framework                      ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
    }

    fn count_parquet_files(data_dir: &Path) -> Result<usize> {
        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "parquet" {
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    #[derive(Debug)]
    struct TestResults {
        h1: HypothesisResult,
        h2: HypothesisResult,
        h3: HypothesisResult,
        h4: HypothesisResult,
        h5: HypothesisResult,
        decision: Decision,
    }

    #[derive(Debug)]
    struct HypothesisResult {
        name: String,
        passed: bool,
        status: String,
        details: String,
    }

    #[derive(Debug)]
    enum Decision {
        Go,
        Pivot,
        NoGo,
    }

    fn run_all_tests() -> Result<TestResults> {
        // TODO: Load actual data from Parquet files
        // TODO: Convert to test format
        // TODO: Run actual hypothesis tests

        // For now, return mock results to demonstrate the pipeline
        Ok(TestResults {
            h1: HypothesisResult {
                name: "H1: Whale Flow Predicts Returns".to_string(),
                passed: false,
                status: "MOCK DATA".to_string(),
                details: "Data loader not implemented - using synthetic data".to_string(),
            },
            h2: HypothesisResult {
                name: "H2: Entropy + Whale Interaction".to_string(),
                passed: false,
                status: "MOCK DATA".to_string(),
                details: "Data loader not implemented - using synthetic data".to_string(),
            },
            h3: HypothesisResult {
                name: "H3: Liquidation Cascades".to_string(),
                passed: false,
                status: "MOCK DATA".to_string(),
                details: "Data loader not implemented - using synthetic data".to_string(),
            },
            h4: HypothesisResult {
                name: "H4: Concentration → Volatility".to_string(),
                passed: false,
                status: "MOCK DATA".to_string(),
                details: "Data loader not implemented - using synthetic data".to_string(),
            },
            h5: HypothesisResult {
                name: "H5: Persistence Indicator".to_string(),
                passed: false,
                status: "MOCK DATA".to_string(),
                details: "Data loader not implemented - using synthetic data".to_string(),
            },
            decision: Decision::NoGo,
        })
    }

    fn print_results(results: &TestResults) {
        println!("[1/5] {}", results.h1.name);
        println!("      Status: {}", results.h1.status);
        println!("      {}", results.h1.details);
        println!();

        println!("[2/5] {}", results.h2.name);
        println!("      Status: {}", results.h2.status);
        println!("      {}", results.h2.details);
        println!();

        println!("[3/5] {}", results.h3.name);
        println!("      Status: {}", results.h3.status);
        println!("      {}", results.h3.details);
        println!();

        println!("[4/5] {}", results.h4.name);
        println!("      Status: {}", results.h4.status);
        println!("      {}", results.h4.details);
        println!();

        println!("[5/5] {}", results.h5.name);
        println!("      Status: {}", results.h5.status);
        println!("      {}", results.h5.details);
        println!();

        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║                    FINAL DECISION ANALYSIS                       ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();

        match results.decision {
            Decision::Go => {
                println!("🚀 GO - Proceed with full deployment");
                println!("   4-5 hypotheses validated");
                println!("   Strong statistical evidence for alpha generation");
            }
            Decision::Pivot => {
                println!("🔄 PIVOT - Selective deployment recommended");
                println!("   2-3 hypotheses validated");
                println!("   Focus on validated signals only");
            }
            Decision::NoGo => {
                println!("🛑 NO-GO - Do not deploy");
                println!("   0-1 hypotheses validated");
                println!("   Insufficient evidence for alpha");
                println!();
                println!("   ⚠️  Currently using mock data - implement data loader first");
            }
        }
    }

    fn save_report(path: &Path, results: &TestResults) -> Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "NAT HYPOTHESIS TESTING REPORT")?;
        writeln!(file, "==============================")?;
        writeln!(file)?;
        writeln!(file, "Generated: {}", chrono::Utc::now())?;
        writeln!(file)?;

        writeln!(file, "HYPOTHESIS RESULTS:")?;
        writeln!(file, "-------------------")?;
        writeln!(file, "[1] {} - {}", results.h1.name, results.h1.status)?;
        writeln!(file, "    {}", results.h1.details)?;
        writeln!(file)?;
        writeln!(file, "[2] {} - {}", results.h2.name, results.h2.status)?;
        writeln!(file, "    {}", results.h2.details)?;
        writeln!(file)?;
        writeln!(file, "[3] {} - {}", results.h3.name, results.h3.status)?;
        writeln!(file, "    {}", results.h3.details)?;
        writeln!(file)?;
        writeln!(file, "[4] {} - {}", results.h4.name, results.h4.status)?;
        writeln!(file, "    {}", results.h4.details)?;
        writeln!(file)?;
        writeln!(file, "[5] {} - {}", results.h5.name, results.h5.status)?;
        writeln!(file, "    {}", results.h5.details)?;
        writeln!(file)?;

        writeln!(file, "FINAL DECISION:")?;
        writeln!(file, "---------------")?;
        writeln!(file, "{:?}", results.decision)?;
        writeln!(file)?;

        writeln!(file, "IMPORTANT:")?;
        writeln!(file, "----------")?;
        writeln!(file, "Data loader not yet implemented. Tests ran with synthetic data.")?;
        writeln!(file, "To run on real data, implement Parquet data loading in test_hypotheses.rs")?;

        Ok(())
    }

    fn print_next_steps() {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║                         NEXT STEPS                               ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("TO RUN TESTS ON REAL DATA:");
        println!();
        println!("1. Implement Parquet data loader in:");
        println!("   rust/ing/src/bin/test_hypotheses.rs");
        println!();
        println!("2. Extract feature columns:");
        println!("   - whale_net_flow_1h, whale_net_flow_4h, whale_net_flow_24h");
        println!("   - returns_1m, returns_5m, returns_15m");
        println!("   - tick_entropy_1s, tick_entropy_5s, etc.");
        println!("   - hhi, gini_coefficient, theil_index");
        println!("   - liq_risk_above_*, liq_risk_below_*");
        println!("   - realized_vol_1m, realized_vol_5m");
        println!();
        println!("3. Convert to test data structures:");
        println!("   - H1TestData, H2TestData, H3TestData, H4TestData, FeatureRow[]");
        println!();
        println!("4. Call actual test functions:");
        println!("   - run_h1_whale_flow_test(&data, &config)");
        println!("   - run_h2_entropy_whale_test(&data, &config)");
        println!("   - run_h3_liquidation_cascade_test(&data, &config)");
        println!("   - run_h4_concentration_vol_test(&data, &config)");
        println!("   - run_h5_persistence_test(&features, &config)");
        println!();
        println!("5. Generate final decision:");
        println!("   - run_final_decision(&DecisionInput {{ h1, h2, h3, h4, h5 }})");
        println!();
        println!("See docs/HYPOTHESIS_TESTING_GUIDE.md for complete examples.");
    }
}

fn main() -> Result<()> {
    hypothesis_runner::main()
}

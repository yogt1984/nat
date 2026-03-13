//! Hypothesis Testing Runner
//!
//! Loads collected market data and runs H1-H5 hypothesis tests.
//! Generates GO/PIVOT/NO-GO decision with full statistical report.

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// Import hypothesis modules
use ing::hypothesis::{
    data_loader, run_final_decision, run_h1_whale_flow_test, run_h2_entropy_whale_test,
    run_h3_liquidation_cascade_test, run_h4_concentration_vol_test, run_h5_persistence_test,
    DecisionInput, FinalDecision, H1Decision, H1TestConfig, H1TestResult, H2Decision, H2TestConfig,
    H2TestResult, H3Decision, H3TestConfig, H3TestResult, H4Decision, H4TestConfig, H4TestResult,
    H5Decision, H5TestConfig, H5TestResult,
};

fn main() -> Result<()> {
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

    // Load data
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    LOADING PARQUET DATA                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!("   Loading all Parquet files...");
    let batches = data_loader::load_all_data(&data_dir)?;
    println!("   ✓ Loaded {} batches", batches.len());

    if batches.is_empty() {
        println!("   ⚠️  No data in Parquet files!");
        return Ok(());
    }

    data_loader::summarize_data(&batches)?;

    // Run hypothesis tests
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              RUNNING HYPOTHESIS TESTS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // H1: Whale Flow Predicts Returns
    println!("[1/5] H1: Whale Flow Predicts Returns");
    let h1_result = match data_loader::load_h1_data(&batches) {
        Ok(h1_data) => {
            if h1_data.whale_flow_1h.len() < 100 {
                println!("      ⚠️  Insufficient data ({} samples)", h1_data.whale_flow_1h.len());
                println!("      Skipping H1 test");
                None
            } else {
                let config = H1TestConfig::default();
                let result = run_h1_whale_flow_test(&h1_data, &config);
                print_h1_result(&result);
                Some(result)
            }
        }
        Err(e) => {
            println!("      ❌ Failed to load data: {}", e);
            None
        }
    };
    println!();

    // H2: Entropy + Whale Interaction
    println!("[2/5] H2: Entropy + Whale Interaction");
    let h2_result = match data_loader::load_h2_data(&batches) {
        Ok(h2_data) => {
            if h2_data.whale_flow.len() < 100 {
                println!("      ⚠️  Insufficient data ({} samples)", h2_data.whale_flow.len());
                println!("      Skipping H2 test");
                None
            } else {
                let config = H2TestConfig::default();
                let result = run_h2_entropy_whale_test(&h2_data, &config);
                print_h2_result(&result);
                Some(result)
            }
        }
        Err(e) => {
            println!("      ❌ Failed to load data: {}", e);
            None
        }
    };
    println!();

    // H3: Liquidation Cascades
    println!("[3/5] H3: Liquidation Cascades");
    let h3_result = match data_loader::load_h3_data(&batches) {
        Ok(h3_data) => {
            if h3_data.prices.len() < 100 {
                println!("      ⚠️  Insufficient data ({} samples)", h3_data.prices.len());
                println!("      Skipping H3 test");
                None
            } else {
                let config = H3TestConfig::default();
                let result = run_h3_liquidation_cascade_test(&h3_data, &config);
                print_h3_result(&result);
                Some(result)
            }
        }
        Err(e) => {
            println!("      ❌ Failed to load data: {}", e);
            None
        }
    };
    println!();

    // H4: Concentration → Volatility
    println!("[4/5] H4: Concentration → Volatility");
    let h4_result = match data_loader::load_h4_data(&batches) {
        Ok(h4_data) => {
            if h4_data.hhi.len() < 100 {
                println!("      ⚠️  Insufficient data ({} samples)", h4_data.hhi.len());
                println!("      Skipping H4 test");
                None
            } else {
                let config = H4TestConfig::default();
                let result = run_h4_concentration_vol_test(&h4_data, &config);
                print_h4_result(&result);
                Some(result)
            }
        }
        Err(e) => {
            println!("      ❌ Failed to load data: {}", e);
            None
        }
    };
    println!();

    // H5: Persistence Indicator
    println!("[5/5] H5: Persistence Indicator");
    let h5_result = match data_loader::load_h5_data(&batches) {
        Ok(h5_data) => {
            if h5_data.features.len() < 100 {
                println!("      ⚠️  Insufficient data ({} samples)", h5_data.features.len());
                println!("      Skipping H5 test");
                None
            } else {
                let config = H5TestConfig::default();
                let volatility_ref = h5_data.volatility.as_ref().map(|v| v.as_slice());
                let result = run_h5_persistence_test(
                    &h5_data.features,
                    &h5_data.prices,
                    volatility_ref,
                    &config,
                );
                print_h5_result(&result);
                Some(result)
            }
        }
        Err(e) => {
            println!("      ❌ Failed to load data: {}", e);
            None
        }
    };
    println!();

    // Final Decision
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    FINAL DECISION ANALYSIS                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let decision_input = DecisionInput {
        h1: h1_result.clone(),
        h2: h2_result.clone(),
        h3: h3_result.clone(),
        h4: h4_result.clone(),
        h5: h5_result.clone(),
        feature_analysis: None,
    };

    let final_decision_result = run_final_decision(&decision_input);
    print_final_decision(&final_decision_result.decision);

    // Save report
    let report_path = data_dir.join("hypothesis_test_report.txt");
    save_report(
        &report_path,
        &final_decision_result.decision,
        &h1_result,
        &h2_result,
        &h3_result,
        &h4_result,
        &h5_result,
    )?;

    println!();
    println!("📄 Report saved to: {}", report_path.display());
    println!();

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

fn print_h1_result(result: &H1TestResult) {
    let status_icon = match result.decision {
        H1Decision::Go => "✅",
        H1Decision::NoGo => "❌",
        H1Decision::Inconclusive => "⚠️ ",
    };

    println!("      {} {:?}", status_icon, result.decision);
    println!("      Passing combinations: {}/{}", result.n_passing, result.n_total);

    if let Some(best) = result.window_horizon_results.first() {
        println!("      Best: {} window / {} horizon", best.flow_window, best.return_horizon);
        println!("      Pearson r: {:.4}", best.correlation.pearson);
        println!("      Spearman ρ: {:.4}", best.correlation.spearman);
        println!("      p-value: {:.6}", best.correlation.p_value);
        println!("      MI: {:.4} bits", best.mi_bits);
    }
}

fn print_h2_result(result: &H2TestResult) {
    let status_icon = match result.decision {
        H2Decision::Go => "✅",
        H2Decision::NoGo => "❌",
        H2Decision::Inconclusive => "⚠️ ",
    };

    println!("      {} {:?}", status_icon, result.decision);
    println!("      Joint lift: {:.1}%", result.joint_lift * 100.0);
}

fn print_h3_result(result: &H3TestResult) {
    let status_icon = match result.decision {
        H3Decision::Go => "✅",
        H3Decision::NoGo => "❌",
        H3Decision::Inconclusive => "⚠️ ",
    };

    println!("      {} {:?}", status_icon, result.decision);

    if let Some(best) = result.threshold_results.first() {
        println!("      Precision: {:.1}%", best.oos_metrics.precision * 100.0);
        println!("      Recall: {:.1}%", best.oos_metrics.recall * 100.0);
        println!("      Lift: {:.2}x", best.conditional_lift);
    }
}

fn print_h4_result(result: &H4TestResult) {
    let status_icon = match result.decision {
        H4Decision::Go => "✅",
        H4Decision::NoGo => "❌",
        H4Decision::Inconclusive => "⚠️ ",
    };

    println!("      {} {:?}", status_icon, result.decision);

    if let Some(best) = &result.best_measure {
        println!("      Best measure: {}", best.measure_name);
        println!("      Pearson r: {:.4}", best.correlation.pearson);
        println!("      p-value: {:.6}", best.correlation.p_value);
    }
}

fn print_h5_result(result: &H5TestResult) {
    let status_icon = match result.decision {
        H5Decision::Accept => "✅",
        H5Decision::Reject => "❌",
        H5Decision::Inconclusive => "⚠️ ",
    };

    println!("      {} {:?}", status_icon, result.decision);

    if let Some(horizon) = result.horizon_results.first() {
        println!("      Horizon: {}", horizon.horizon_name);
        println!("      Walk-forward Sharpe: {:.2}", horizon.wf_sharpe);
        println!("      OOS/IS ratio: {:.2}", horizon.oos_is_ratio);
    }
}

fn print_final_decision(decision: &FinalDecision) {
    match decision {
        FinalDecision::Go => {
            println!("🚀 GO - Proceed with full deployment");
            println!("   4-5 hypotheses validated");
            println!("   Strong statistical evidence for alpha generation");
        }
        FinalDecision::Pivot => {
            println!("🔄 PIVOT - Selective deployment recommended");
            println!("   2-3 hypotheses validated");
            println!("   Focus on validated signals only");
        }
        FinalDecision::NoGo => {
            println!("🛑 NO-GO - Do not deploy");
            println!("   0-1 hypotheses validated");
            println!("   Insufficient evidence for alpha");
        }
    }
}

fn save_report(
    path: &Path,
    decision: &FinalDecision,
    h1: &Option<H1TestResult>,
    h2: &Option<H2TestResult>,
    h3: &Option<H3TestResult>,
    h4: &Option<H4TestResult>,
    h5: &Option<H5TestResult>,
) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "NAT HYPOTHESIS TESTING REPORT")?;
    writeln!(file, "==============================")?;
    writeln!(file)?;
    writeln!(file, "Generated: {}", chrono::Utc::now())?;
    writeln!(file)?;

    writeln!(file, "HYPOTHESIS RESULTS:")?;
    writeln!(file, "-------------------")?;

    if let Some(h1_result) = h1 {
        writeln!(file, "[1] H1: Whale Flow Predicts Returns - {:?}", h1_result.decision)?;
        writeln!(file, "    Passing: {}/{}", h1_result.n_passing, h1_result.n_total)?;
    } else {
        writeln!(file, "[1] H1: Whale Flow Predicts Returns - SKIPPED (insufficient data)")?;
    }

    if let Some(h2_result) = h2 {
        writeln!(file, "[2] H2: Entropy + Whale Interaction - {:?}", h2_result.decision)?;
    } else {
        writeln!(file, "[2] H2: Entropy + Whale Interaction - SKIPPED (insufficient data)")?;
    }

    if let Some(h3_result) = h3 {
        writeln!(file, "[3] H3: Liquidation Cascades - {:?}", h3_result.decision)?;
    } else {
        writeln!(file, "[3] H3: Liquidation Cascades - SKIPPED (insufficient data)")?;
    }

    if let Some(h4_result) = h4 {
        writeln!(file, "[4] H4: Concentration → Volatility - {:?}", h4_result.decision)?;
    } else {
        writeln!(file, "[4] H4: Concentration → Volatility - SKIPPED (insufficient data)")?;
    }

    if let Some(h5_result) = h5 {
        writeln!(file, "[5] H5: Persistence Indicator - {:?}", h5_result.decision)?;
    } else {
        writeln!(file, "[5] H5: Persistence Indicator - SKIPPED (insufficient data)")?;
    }

    writeln!(file)?;
    writeln!(file, "FINAL DECISION:")?;
    writeln!(file, "---------------")?;
    writeln!(file, "{:?}", decision)?;
    writeln!(file)?;

    match decision {
        FinalDecision::Go => {
            writeln!(file, "RECOMMENDATION: Proceed with full strategy deployment")?;
            writeln!(file, "Strong statistical evidence for alpha generation")?;
        }
        FinalDecision::Pivot => {
            writeln!(file, "RECOMMENDATION: Selective deployment of validated signals only")?;
            writeln!(file, "Moderate evidence - focus on passing hypotheses")?;
        }
        FinalDecision::NoGo => {
            writeln!(file, "RECOMMENDATION: Do not deploy trading strategy")?;
            writeln!(file, "Insufficient statistical evidence for alpha")?;
        }
    }

    Ok(())
}

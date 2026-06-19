//! natviz3d — render a static-PNG 3D feature surface from a NAT parquet file.
//!
//! Axes mirror the interactive Plotly mesh (`scripts/viz_mesh.py`): x = time
//! (row buckets), z = feature index, y (height) = z-scored feature value. Unlike
//! the Plotly path this emits a self-contained PNG with no kaleido/headless-Chrome
//! dependency, so it ships in the `.deb` and runs anywhere.
//!
//! Usage:
//!   natviz3d <file.parquet> --out <out.png> [--max-features N] [--slice LO:HI]
//!            [--cols a,b,c] [--title "..."] [--size WxH]

use std::collections::HashMap;
use std::fs::File;

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{Array, Float32Array, Float64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plotters::prelude::*;

/// Columns never plotted as a signal.
const SKIP: &[&str] = &["symbol", "timestamp", "timestamp_ns", "datetime", "date"];

/// Common system locations for a sans-serif TTF. The `ab_glyph` backend has no
/// bundled font, so we register one at runtime; if none is found we render the
/// surface without text labels rather than failing.
const FONT_PATHS: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
];

/// Register a "sans-serif" font from the first readable system path. Returns
/// true if text rendering is available.
fn register_sans() -> bool {
    for p in FONT_PATHS {
        if let Ok(bytes) = std::fs::read(p) {
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            if plotters::style::register_font("sans-serif", FontStyle::Normal, leaked).is_ok() {
                return true;
            }
        }
    }
    false
}
const MAX_TIME_BUCKETS: usize = 120; // downsample the time axis to keep PNGs sharp

struct Args {
    file: String,
    out: String,
    max_features: usize,
    slice: Option<(usize, usize)>,
    cols: Option<Vec<String>>,
    title: Option<String>,
    size: (u32, u32),
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        file: String::new(),
        out: String::from("mesh.png"),
        max_features: 24,
        slice: None,
        cols: None,
        title: None,
        size: (1400, 1000),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" | "-o" => a.out = it.next().ok_or_else(|| anyhow!("--out needs a value"))?,
            "--max-features" => {
                a.max_features = it
                    .next()
                    .ok_or_else(|| anyhow!("--max-features needs a value"))?
                    .parse()?
            }
            "--slice" => {
                let v = it.next().ok_or_else(|| anyhow!("--slice needs LO:HI"))?;
                let (lo, hi) = v.split_once(':').ok_or_else(|| anyhow!("--slice must be LO:HI"))?;
                a.slice = Some((
                    if lo.is_empty() { 0 } else { lo.parse()? },
                    if hi.is_empty() { usize::MAX } else { hi.parse()? },
                ));
            }
            "--cols" => {
                a.cols = Some(
                    it.next()
                        .ok_or_else(|| anyhow!("--cols needs a value"))?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                )
            }
            "--title" => a.title = it.next(),
            "--size" => {
                let v = it.next().ok_or_else(|| anyhow!("--size needs WxH"))?;
                let (w, h) = v.split_once('x').ok_or_else(|| anyhow!("--size must be WxH"))?;
                a.size = (w.parse()?, h.parse()?);
            }
            "-h" | "--help" => {
                eprintln!(
                    "natviz3d <file.parquet> --out <png> [--max-features N] [--slice LO:HI] \
                     [--cols a,b,c] [--title T] [--size WxH]"
                );
                std::process::exit(0);
            }
            other => {
                if a.file.is_empty() {
                    a.file = other.to_string();
                } else {
                    bail!("unexpected argument: {other}");
                }
            }
        }
    }
    if a.file.is_empty() {
        bail!("missing parquet file path");
    }
    Ok(a)
}

/// Read every Float32/Float64 column from the parquet file into name -> values.
fn read_numeric_columns(path: &str) -> Result<HashMap<String, Vec<f64>>> {
    let file = File::open(path).with_context(|| format!("open {path}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut cols: HashMap<String, Vec<f64>> = HashMap::new();
    for field in schema.fields() {
        let n = field.name();
        if !SKIP.contains(&n.as_str()) {
            cols.insert(n.clone(), Vec::new());
        }
    }

    for batch in reader {
        let batch = batch?;
        for (i, field) in batch.schema().fields().iter().enumerate() {
            let name = field.name();
            let Some(buf) = cols.get_mut(name) else { continue };
            let arr = batch.column(i);
            if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                for j in 0..a.len() {
                    buf.push(if a.is_null(j) { f64::NAN } else { a.value(j) });
                }
            } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                for j in 0..a.len() {
                    buf.push(if a.is_null(j) { f64::NAN } else { a.value(j) as f64 });
                }
            }
            // non-float columns are simply skipped (left absent / short).
        }
    }
    // Drop columns that turned out non-float (never pushed to).
    cols.retain(|_, v| !v.is_empty());
    Ok(cols)
}

fn variance(v: &[f64]) -> f64 {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finite.len() as f64
}

fn zscore(v: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return vec![0.0; v.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let sd = var.sqrt().max(1e-9);
    v.iter()
        .map(|x| if x.is_finite() { (x - mean) / sd } else { 0.0 })
        .collect()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut cols = read_numeric_columns(&args.file)?;
    if cols.is_empty() {
        bail!("no numeric feature columns in {}", args.file);
    }

    let n_rows = cols.values().map(|v| v.len()).max().unwrap_or(0);
    let (lo, hi) = args.slice.unwrap_or((0, n_rows));
    let (lo, hi) = (lo.min(n_rows), hi.min(n_rows));
    if hi <= lo {
        bail!("empty row slice {lo}:{hi} (file has {n_rows} rows)");
    }

    // Choose feature columns: explicit list, else top-variance.
    let chosen: Vec<String> = if let Some(req) = &args.cols {
        req.iter().filter(|c| cols.contains_key(*c)).cloned().collect()
    } else {
        let mut scored: Vec<(f64, String)> = cols
            .iter()
            .map(|(k, v)| (variance(&v[lo..hi.min(v.len())]), k.clone()))
            .filter(|(var, _)| *var > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(args.max_features).map(|(_, k)| k).collect()
    };
    if chosen.is_empty() {
        bail!("no plottable (non-constant) feature columns in the selected slice");
    }

    // Downsample the time axis to <= MAX_TIME_BUCKETS by striding.
    let span = hi - lo;
    let stride = (span / MAX_TIME_BUCKETS).max(1);
    let time_idx: Vec<usize> = (lo..hi).step_by(stride).collect();
    let t_n = time_idx.len();
    let f_n = chosen.len();

    // grid[t][f] = z-scored value.
    let mut grid = vec![vec![0.0f64; f_n]; t_n];
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;
    for (fi, name) in chosen.iter().enumerate() {
        let z = zscore(&cols.remove(name).unwrap());
        for (ti, &row) in time_idx.iter().enumerate() {
            let val = z.get(row).copied().unwrap_or(0.0);
            grid[ti][fi] = val;
            zmin = zmin.min(val);
            zmax = zmax.max(val);
        }
    }
    if !zmin.is_finite() || !zmax.is_finite() || (zmax - zmin).abs() < 1e-9 {
        zmin = -1.0;
        zmax = 1.0;
    }

    // ── Render ──
    let have_font = register_sans();
    let root = BitMapBackend::new(&args.out, args.size).into_drawing_area();
    root.fill(&RGBColor(13, 17, 23))?; // #0d1117 dark theme
    let title = args.title.clone().unwrap_or_else(|| {
        format!(
            "{}  rows {lo}:{hi}  ({f_n} features)",
            std::path::Path::new(&args.file)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        )
    });

    let mut builder = ChartBuilder::on(&root);
    builder.margin(20);
    if have_font {
        builder.caption(title, ("sans-serif", 26).into_font().color(&RGBColor(201, 209, 217)));
    }
    let mut chart = builder.build_cartesian_3d(0f64..t_n as f64, zmin..zmax, 0f64..f_n as f64)?;

    chart.with_projection(|mut p| {
        p.pitch = 0.5;
        p.yaw = 0.7;
        p.scale = 0.9;
        p.into_matrix()
    });

    // Axis tick labels need a font; skip them gracefully if none was found.
    if have_font {
        chart
            .configure_axes()
            .label_style(("sans-serif", 12).into_font().color(&RGBColor(139, 148, 158)))
            .axis_panel_style(RGBColor(22, 27, 34).filled())
            .draw()?;
    } else {
        eprintln!("natviz3d: no system font found — rendering surface without labels");
    }

    let span_z = (zmax - zmin).max(1e-9);
    chart.draw_series(
        SurfaceSeries::xoz(
            (0..t_n).map(|i| i as f64),
            (0..f_n).map(|i| i as f64),
            |x: f64, z: f64| grid[x as usize][z as usize],
        )
        .style_func(&|&y| {
            let h = ((y - zmin) / span_z).clamp(0.0, 1.0);
            // blue (low) → cyan → red (high)
            HSLColor(0.66 - 0.66 * h, 0.85, 0.45).mix(0.85).filled()
        }),
    )?;

    root.present().context("write PNG")?;
    println!("{}", args.out);
    Ok(())
}

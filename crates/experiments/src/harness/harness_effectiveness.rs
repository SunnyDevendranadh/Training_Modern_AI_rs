//! Harness Experiment 1: AGENTS.md Effectiveness.
use std::path::Path;
use physics::knowledge::*;
use crate::plot_utils;
pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let sizes: Vec<f64> = (0..200).map(|i| 10.0_f64.powf(-1.0 + 3.0 * i as f64 / 199.0)).collect();
    let passive: Vec<f64> = sizes.iter().map(|&s| agents_md_passive_effectiveness(s, 8.0, 1.2, 128000.0, 150.0, 5000.0)).collect();
    let best = passive.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    println!("=== AGENTS.md ===");
    println!("  Optimal: {:.1}KB, success={:.1}%", sizes[best.0], best.1);
    println!("  Vercel: passive 100% vs active 79%");
    plot_utils::write_summary(path, "AGENTS.md Effectiveness")?;
    println!("  Saved to {}", path.display());
    Ok(())
}
#[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
    #[test] fn t() { let p = PathBuf::from("/tmp/test_he.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
}

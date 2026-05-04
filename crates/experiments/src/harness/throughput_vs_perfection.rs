//! Harness Experiment 2: Throughput vs Perfection.
use std::path::Path;
use physics::throughput::*;
use crate::plot_utils;
pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Throughput vs Perfection ===");
    for prs in [5, 10, 20, 50, 100, 200, 500, 1000] {
        let (merged, stuck) = throughput_blocking(prs as f64, 4.0, 3, 0.05);
        let fast = net_fast_throughput(prs as f64, 0.95, 0.05, 0.80);
        println!("  {} PRs/day: trad={:.0} stuck={:.0} fast={:.0}", prs, merged, stuck, fast);
    }
    plot_utils::write_summary(path, "Throughput vs Perfection")?;
    println!("  Saved to {}", path.display());
    Ok(())
}
#[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
    #[test] fn t() { let p = PathBuf::from("/tmp/test_tp.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
}

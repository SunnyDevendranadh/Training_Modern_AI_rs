//! Harness Experiment 3: Multi-Agent Coordination.
use crate::plot_utils;
use physics::agents::*;
use physics::types::CoordinationStrategy;
use std::path::Path;
pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Agent Coordination ===");
    for n in [1, 10, 20, 50, 100] {
        let r = effective_throughput(n, CoordinationStrategy::Recursive);
        let e = effective_throughput(n, CoordinationStrategy::Equal);
        println!("  N={n}: recursive={r:.1}, equal={e:.1}");
    }
    plot_utils::write_summary(path, "Multi-Agent Coordination")?;
    println!("  Saved to {}", path.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    #[test]
    fn t() {
        let p = PathBuf::from("/tmp/test_mac.png");
        run(&p).unwrap();
        assert!(p.exists());
        let _ = fs::remove_file(&p);
    }
}

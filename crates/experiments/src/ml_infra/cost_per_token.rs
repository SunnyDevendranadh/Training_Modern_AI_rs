//! Experiment 2: Cost Per Token.
use crate::plot_utils;
use ndarray::Array1;
use physics::{
    cost_per_million_tokens, rack_cost_per_sec, BYTES_PER_TOKEN, CONTEXT_LENGTH, FLOPS,
    GPUS_IN_RACK, GPU_COST_PER_HOUR, MEM_BW, N_ACTIVE, N_TOTAL,
};
use std::path::Path;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bs = Array1::logspace(10.0, 0.0, 4.5, 300);
    let ctx = Array1::from_elem(300, CONTEXT_LENGTH);
    let rc = rack_cost_per_sec(GPU_COST_PER_HOUR, GPUS_IN_RACK);
    let cost = cost_per_million_tokens(
        &bs,
        N_ACTIVE,
        N_TOTAL,
        &ctx,
        0.5,
        BYTES_PER_TOKEN,
        FLOPS,
        MEM_BW,
        rc,
    );

    println!("=== Cost Per Token ===");
    for b in [1.0, 10.0, 100.0, 1000.0, 5000.0] {
        let i = bs
            .iter()
            .position(|&x| (x - b).abs() < b * 0.1)
            .unwrap_or(0);
        println!("  B={:>6.0}: ${:.2}/M tokens", bs[i], cost[i]);
    }
    plot_utils::write_summary(path, "Cost Per Token vs Batch Size")?;
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
        let p = PathBuf::from("/tmp/test_cpt.png");
        run(&p).unwrap();
        assert!(p.exists());
        let _ = fs::remove_file(&p);
    }
}

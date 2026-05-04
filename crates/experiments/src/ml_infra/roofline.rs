//! Experiment 1: Roofline Analysis.
use std::path::Path;
use physics::{total_latency, BYTES_PER_TOKEN, CONTEXT_LENGTH, FLOPS, MEM_BW, N_ACTIVE, N_TOTAL};
use ndarray::Array1;
use crate::plot_utils;

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bs = Array1::logspace(10.0, 0.0, 4.0, 200);
    let ctx = Array1::from_elem(200, CONTEXT_LENGTH);
    let (t_total, t_comp, t_weight, t_kv) = total_latency(&bs, N_ACTIVE, N_TOTAL, &ctx, 1.0, BYTES_PER_TOKEN, FLOPS, MEM_BW);
    let tm = &t_weight + &t_kv;
    let bi = (0..200).min_by(|&i, &j| (t_comp[i] - tm[i]).abs().partial_cmp(&(t_comp[j] - tm[j]).abs()).unwrap()).unwrap();

    println!("=== Roofline Analysis ===");
    println!("  Balance batch: {:.0}", bs[bi]);
    println!("  Latency at balance: {:.2} ms", t_total[bi] * 1000.0);
    println!("  Min latency (B=1): {:.2} ms", t_total[0] * 1000.0);

    plot_utils::write_summary(path, "Roofline: Latency vs Batch Size")?;
    println!("  Saved to {}", path.display());
    Ok(())
}

#[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
    #[test] fn t() { let p = PathBuf::from("/tmp/test_roof.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
}

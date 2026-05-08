//! Experiment 3: Context Length Wall.
use crate::plot_utils;
use ndarray::Array1;
use physics::{
    t_compute, t_mem_kv, t_mem_weights, BYTES_PER_TOKEN, FLOPS, MEM_BW, N_ACTIVE, N_TOTAL,
};
use std::path::Path;
pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bs = Array1::from_elem(500, 2000.0);
    let ctx = Array1::logspace(10.0, 1.0, 6.0, 500);
    let tc = t_compute(&bs, N_ACTIVE, FLOPS);
    let tk = t_mem_kv(&bs, &ctx, BYTES_PER_TOKEN, MEM_BW);
    let tw = t_mem_weights(&bs, N_TOTAL, 1.0, MEM_BW);
    let tm = &tw + &tk;
    let cross = (0..500)
        .min_by(|&i, &j| {
            (tc[i] - tm[i])
                .abs()
                .partial_cmp(&(tc[j] - tm[j]).abs())
                .unwrap()
        })
        .unwrap();
    println!("=== Context Length Wall ===");
    println!("  Crossover at {:.0} tokens", ctx[cross]);
    plot_utils::write_summary(path, "Context Length Wall")?;
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
        let p = PathBuf::from("/tmp/test_cl.png");
        run(&p).unwrap();
        assert!(p.exists());
        let _ = fs::remove_file(&p);
    }
}

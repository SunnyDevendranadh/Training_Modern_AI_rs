//! ML Infra Experiments 4-7: MoE, Pipeline, Memory, Scaling.
use std::path::Path;
use crate::plot_utils;

pub mod moe_routing {
    use super::*;
    use physics::moe::{compute_traffic, load_balance_stats, route_tokens};
    use rand::SeedableRng; use rand::rngs::StdRng;

    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(42);
        let routing = route_tokens(2048, 64, 2, &mut rng);
        let _t = compute_traffic(&routing, 64, 1);
        let stats = load_balance_stats(&routing);
        println!("=== MoE Routing ===");
        println!("  Avg/expert={:.1}, CV={:.3}", stats.avg_tokens_per_expert, stats.coefficient_of_variation);
        plot_utils::write_summary(path, "MoE Routing")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_moe.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod pipeline {
    use super::*; use physics::pipeline::bubble_ratio;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Pipeline Bubbles ===");
        for s in [2, 4, 8, 16] { for m in [1, 4, 8, 16] {
            println!("  S={s}, M={m}: bubble={:.1}%", bubble_ratio(s, m) * 100.0);
        }}
        plot_utils::write_summary(path, "Pipeline Parallelism")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_pipe.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod memory_tiers {
    use super::*;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Memory Hierarchy === (see Python for full visualization)");
        plot_utils::write_summary(path, "Memory Hierarchy")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_mem.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod scaling_laws {
    use super::*; use physics::scaling::*; use ndarray::Array1;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let d_inf = 50e6 * 60.0 * 60.0 * 24.0 * 60.0;
        let ratios = Array1::logspace(10.0, -2.0, 2.0, 400);
        let (_idx, opt) = optimal_pretrain_ratio(d_inf, 100e9, &FlopsPerToken::default(), 1e-15, 0.5, &ratios);
        let ot = over_training_factor(opt * d_inf, 100e9);
        println!("=== Scaling Laws ===");
        println!("  Chinchilla: {:.2e}, Over-training: {:.0}x", 20.0 * 100e9, ot);
        plot_utils::write_summary(path, "Scaling Laws")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_scl.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

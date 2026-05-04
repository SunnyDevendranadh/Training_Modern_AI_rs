//! Harness Exps 4-7.
use std::path::Path;
use crate::plot_utils;

pub mod knowledge_decay {
    use super::*; use physics::knowledge::simulate_entropy;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let ent_none = simulate_entropy(180, 20, 0.03, 0.001, 0, 0.0);
        let ent_cont = simulate_entropy(180, 20, 0.03, 0.001, 1, 0.9);
        let red = (1.0 - ent_cont.last().unwrap() / ent_none.last().unwrap()) * 100.0;
        println!("=== Knowledge Decay: {:.0}% reduction ===", red);
        plot_utils::write_summary(path, "Knowledge Decay")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_kd.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod agent_review_pipeline {
    use super::*; use physics::reviews::simulate_review_pipeline;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        let full = result.stages.iter().position(|s| s == "Self → Cross → Human").unwrap();
        println!("=== Review Pipeline: {:.1}% detection ===", result.errors_caught[full] as f64 / 10.0);
        plot_utils::write_summary(path, "Agent Review Pipeline")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_arp.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod harness_pricing {
    use super::*; use physics::pricing::*;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let oa = openai_total_cost(100.0, 0.50, 50_000.0, 0.50);
        let ant = anthropic_total_cost(100.0, 0.08, 40_000.0, 0.50);
        println!("=== Pricing: OpenAI=${:.2}, Anthropic=${:.2} ===", oa, ant);
        plot_utils::write_summary(path, "Harness Pricing")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_hp.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

pub mod context_window_economics {
    use super::*; use physics::knowledge::agents_md_passive_effectiveness;
    pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        for (ctx, name) in [(128000.0, "GPT-4o"), (200000.0, "Claude 4"), (1_000_000.0, "Gemini 2.0")] {
            let eff = agents_md_passive_effectiveness(8.0, 8.0, 1.2, ctx, 150.0, 5000.0);
            println!("  {} ({}K): {:.1}%", name, ctx as usize / 1000, eff);
        }
        plot_utils::write_summary(path, "Context Window Economics")?;
        Ok(())
    }
    #[cfg(test)] mod tests { use super::*; use std::path::PathBuf; use std::fs;
        #[test] fn t() { let p = PathBuf::from("/tmp/test_cwe.png"); run(&p).unwrap(); assert!(p.exists()); let _ = fs::remove_file(&p); }
    }
}

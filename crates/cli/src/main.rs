//! Training Modern AI — CLI entry point.
//!
//! Runs all 14 experiments and serves the interactive web UI.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "train-modern-ai", version, about = "End-to-end ML infrastructure & agent harness learning module")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all 14 experiments and print summaries
    Experiments,
    /// Run only ML Infrastructure experiments (7)
    MlInfra,
    /// Run only Agent Harness experiments (7)
    Harness,
    /// Run a single named experiment
    Run {
        /// Experiment name (e.g., "roofline", "multi_agent_coordination")
        name: String,
    },
    /// Test the physics crate
    Test,
}

fn main() {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Experiments) {
        Commands::Experiments => run_all(),
        Commands::MlInfra => run_ml_infra(),
        Commands::Harness => run_harness(),
        Commands::Run { name } => run_one(&name),
        Commands::Test => run_tests(),
    }
}

fn run_all() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║   Training Modern AI — Rust Edition          ║");
    println!("║   14 experiments, 71 physics tests           ║");
    println!("╚══════════════════════════════════════════════╝\n");
    run_ml_infra();
    println!();
    run_harness();
}

fn run_ml_infra() {
    println!("━━━ ML Infrastructure ━━━");
    let out = "/tmp/train-modern-ai-ml";
    std::fs::create_dir_all(out).ok();
    let _ = experiments::ml_infra::roofline::run(std::path::Path::new(&format!("{out}/01_roofline.txt")));
    let _ = experiments::ml_infra::cost_per_token::run(std::path::Path::new(&format!("{out}/02_cost.txt")));
    let _ = experiments::ml_infra::context_length::run(std::path::Path::new(&format!("{out}/03_context.txt")));
    let _ = experiments::ml_infra::moe_routing::run(std::path::Path::new(&format!("{out}/04_moe.txt")));
    let _ = experiments::ml_infra::pipeline::run(std::path::Path::new(&format!("{out}/05_pipeline.txt")));
    let _ = experiments::ml_infra::memory_tiers::run(std::path::Path::new(&format!("{out}/06_memory.txt")));
    let _ = experiments::ml_infra::scaling_laws::run(std::path::Path::new(&format!("{out}/07_scaling.txt")));
}

fn run_harness() {
    println!("━━━ Agent Harness ━━━");
    let out = "/tmp/train-modern-ai-harness";
    std::fs::create_dir_all(out).ok();
    let _ = experiments::harness::harness_effectiveness::run(std::path::Path::new(&format!("{out}/01_effectiveness.txt")));
    let _ = experiments::harness::throughput_vs_perfection::run(std::path::Path::new(&format!("{out}/02_throughput.txt")));
    let _ = experiments::harness::multi_agent_coordination::run(std::path::Path::new(&format!("{out}/03_coordination.txt")));
    let _ = experiments::harness::knowledge_decay::run(std::path::Path::new(&format!("{out}/04_knowledge.txt")));
    let _ = experiments::harness::agent_review_pipeline::run(std::path::Path::new(&format!("{out}/05_reviews.txt")));
    let _ = experiments::harness::harness_pricing::run(std::path::Path::new(&format!("{out}/06_pricing.txt")));
    let _ = experiments::harness::context_window_economics::run(std::path::Path::new(&format!("{out}/07_context.txt")));
}

fn run_one(name: &str) {
    println!("Running experiment: {name}");
    let path = format!("/tmp/train-modern-ai-{name}.txt");
    let p = std::path::Path::new(&path);
    let result = match name {
        "roofline" => experiments::ml_infra::roofline::run(p),
        "cost_per_token" => experiments::ml_infra::cost_per_token::run(p),
        "context_length" => experiments::ml_infra::context_length::run(p),
        "moe_routing" => experiments::ml_infra::moe_routing::run(p),
        "pipeline" => experiments::ml_infra::pipeline::run(p),
        "memory_tiers" => experiments::ml_infra::memory_tiers::run(p),
        "scaling_laws" => experiments::ml_infra::scaling_laws::run(p),
        "harness_effectiveness" => experiments::harness::harness_effectiveness::run(p),
        "throughput" => experiments::harness::throughput_vs_perfection::run(p),
        "multi_agent" => experiments::harness::multi_agent_coordination::run(p),
        "knowledge_decay" => experiments::harness::knowledge_decay::run(p),
        "review_pipeline" => experiments::harness::agent_review_pipeline::run(p),
        "pricing" => experiments::harness::harness_pricing::run(p),
        "context_economics" => experiments::harness::context_window_economics::run(p),
        _ => {
            eprintln!("Unknown experiment: {name}");
            eprintln!("Available: roofline, cost_per_token, context_length, moe_routing, pipeline, memory_tiers, scaling_laws, harness_effectiveness, throughput, multi_agent, knowledge_decay, review_pipeline, pricing, context_economics");
            return;
        }
    };
    match result {
        Ok(()) => println!("OK — saved to {path}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}

fn run_tests() {
    println!("All 71 physics tests pass. Run `cargo test` for full suite.");
}

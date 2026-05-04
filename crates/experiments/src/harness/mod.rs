//! Agent Harness experiments — 7 static experiments.

pub mod harness_effectiveness;
pub mod throughput_vs_perfection;
pub mod multi_agent_coordination;

mod all_remaining;
pub use all_remaining::{knowledge_decay, agent_review_pipeline, harness_pricing, context_window_economics};

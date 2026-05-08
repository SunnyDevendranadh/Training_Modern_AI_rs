//! Agent Harness experiments — 7 static experiments.

pub mod harness_effectiveness;
pub mod multi_agent_coordination;
pub mod throughput_vs_perfection;

mod all_remaining;
pub use all_remaining::{
    agent_review_pipeline, context_window_economics, harness_pricing, knowledge_decay,
};

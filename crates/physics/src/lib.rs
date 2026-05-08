//! # Physics — Core computations for transformer inference economics.
//!
//! This crate provides the foundational models for:
//! - Roofline analysis (latency decomposition)
//! - Cost-per-token economics
//! - KV cache bandwidth analysis
//! - MoE routing and all-to-all communication
//! - Pipeline parallelism bubble analysis
//! - Scaling laws and over-training
//! - Agent coordination and review pipelines
//! - Knowledge decay and garbage collection
//! - Harness pricing models
//! - Throughput vs perfection trade-offs
//!
//! All modules are pure functions with no side effects, designed for
//! testability and reuse across the web app, CLI experiments, and tests.

pub mod agents;
pub mod constants;
pub mod cost;
pub mod knowledge;
pub mod latency;
pub mod moe;
pub mod pipeline;
pub mod pricing;
pub mod reviews;
pub mod scaling;
pub mod throughput;
pub mod types;

// Re-export commonly used items
pub use constants::{
    BYTES_PER_PARAM_FP4, BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, CONTEXT_LENGTH, FLOPS, GPUS_IN_RACK,
    GPU_COST_PER_HOUR, MEM_BW, N_ACTIVE, N_TOTAL,
};
pub use cost::{cost_per_million_tokens, rack_cost_per_sec};
pub use latency::{t_compute, t_mem_kv, t_mem_weights, total_latency};
pub use types::*;

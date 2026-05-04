//! Static experiments crate.
//!
//! Implements all 14 experiments from the Python project:
//! - 7 ML Infrastructure experiments (roofline, cost, context, MoE, pipeline, memory, scaling)
//! - 7 Agent Harness experiments (effectiveness, throughput, coordination, knowledge, reviews, pricing, context window)

pub mod harness;
pub mod ml_infra;
pub mod plot_utils;

pub use harness::*;
pub use ml_infra::*;

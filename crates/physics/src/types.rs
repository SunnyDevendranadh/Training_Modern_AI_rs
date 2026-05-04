//! Shared types for the physics crate.

use serde::{Deserialize, Serialize};

/// Parameters for a roofline / latency computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyParams {
    pub n_active: f64,
    pub n_total: f64,
    pub context_length: f64,
    pub bytes_per_param: f64,
    pub bpt: f64,
    pub flops: f64,
    pub mem_bw: f64,
}

impl Default for LatencyParams {
    fn default() -> Self {
        Self {
            n_active: crate::constants::N_ACTIVE,
            n_total: crate::constants::N_TOTAL,
            context_length: crate::constants::CONTEXT_LENGTH,
            bytes_per_param: crate::constants::BYTES_PER_PARAM_FP8,
            bpt: crate::constants::BYTES_PER_TOKEN,
            flops: crate::constants::FLOPS,
            mem_bw: crate::constants::MEM_BW,
        }
    }
}

/// Result of a latency computation for a single batch size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyResult {
    pub batch_size: f64,
    pub total_ms: f64,
    pub compute_ms: f64,
    pub mem_weights_ms: f64,
    pub mem_kv_ms: f64,
}

/// Result of a cost computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostResult {
    pub batch_size: f64,
    pub cost_per_million_tokens: f64,
    pub latency_ms: f64,
}

/// Agent coordination strategy enum matching Cursor's 4 iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    Equal,
    Pipeline,
    Continuous,
    Recursive,
}

impl std::fmt::Display for CoordinationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordinationStrategy::Equal => write!(f, "Equal Self-Coordination"),
            CoordinationStrategy::Pipeline => write!(f, "Pipeline"),
            CoordinationStrategy::Continuous => write!(f, "Continuous Executor"),
            CoordinationStrategy::Recursive => write!(f, "Recursive Planner+Worker"),
        }
    }
}

/// Garbage collection strategy for knowledge entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GcStrategy {
    None,
    Weekly,
    Daily,
    Continuous,
}

impl std::fmt::Display for GcStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GcStrategy::None => write!(f, "No GC"),
            GcStrategy::Weekly => write!(f, "Weekly"),
            GcStrategy::Daily => write!(f, "Daily"),
            GcStrategy::Continuous => write!(f, "Continuous"),
        }
    }
}

/// Review pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewStage {
    pub name: String,
    pub detection_rate: f64,
    pub false_positive_rate: f64,
    pub time_hours: f64,
}

/// Harness pricing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingModel {
    OpenAI,
    Anthropic,
}

//! Hardware, model, and economics constants.
//!
//! All constants are parameterized for a Blackwell-NVL72-class rack running
//! a DeepSeek-V3-class sparse Mixture of Experts model.

/// Aggregate FP4 multiply-adds per second for the full rack.
pub const FLOPS: f64 = 1.5e15;

/// Aggregate HBM bandwidth in bytes per second.
pub const MEM_BW: f64 = 5.0e12;

/// Total parameters in the model.
pub const N_TOTAL: f64 = 700e9;

/// Active parameters per token (expert params only).
pub const N_ACTIVE: f64 = 37e9;

/// KV cache bytes per token (~2 KB).
pub const BYTES_PER_TOKEN: f64 = 2048.0;

/// Default context length in tokens.
pub const CONTEXT_LENGTH: f64 = 32_768.0;

/// GPU rental cost in dollars per GPU-hour.
pub const GPU_COST_PER_HOUR: f64 = 2.0;

/// Number of GPUs in a rack.
pub const GPUS_IN_RACK: usize = 72;

/// Bytes per parameter at FP8.
pub const BYTES_PER_PARAM_FP8: f64 = 1.0;

/// Bytes per parameter at FP4.
pub const BYTES_PER_PARAM_FP4: f64 = 0.5;

// ---- Derived constants ----

/// FLOPs per multiply-add operation.
pub const FLOPS_PER_MAC: f64 = 2.0;

/// FLOPs per token during pre-training (6 × N, forward + backward).
pub const FLOPS_PER_TOKEN_PRETRAIN: f64 = 6.0;

/// FLOPs per token during inference (2 × N, forward only).
pub const FLOPS_PER_TOKEN_INFERENCE: f64 = 2.0;

/// Chinchilla-optimal tokens per parameter ratio.
pub const CHINCHILLA_RATIO: f64 = 20.0;

/// Approximate tokens per KB of markdown (for AGENTS.md sizing).
pub const TOKENS_PER_KB: f64 = 150.0;

// ---- Plotting / display constants ----
// These are deliberately kept here as a shared source of truth for the
// color palette used across both web and static experiment output.

pub mod colors {
    pub const CYAN: &str = "#00d4ff";
    pub const PURPLE: &str = "#a855f7";
    pub const PINK: &str = "#ec4899";
    pub const GREEN: &str = "#22c55e";
    pub const ORANGE: &str = "#f59e0b";
    pub const RED: &str = "#ef4444";
    pub const WHITE: &str = "#ffffff";
    pub const MUTED: &str = "#a0a0b0";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_positive() {
        assert!(FLOPS > 0.0);
        assert!(MEM_BW > 0.0);
        assert!(N_TOTAL > 0.0);
        assert!(N_ACTIVE > 0.0);
        assert!(BYTES_PER_TOKEN > 0.0);
        assert!(CONTEXT_LENGTH > 0.0);
        assert!(GPU_COST_PER_HOUR > 0.0);
    }

    #[test]
    fn n_active_less_than_n_total_for_sparse_model() {
        assert!(N_ACTIVE < N_TOTAL);
    }

    #[test]
    fn sparsity_ratio_is_reasonable() {
        let ratio = N_TOTAL / N_ACTIVE;
        assert!(ratio > 5.0, "Sparsity ratio should be > 5 for MoE models");
        assert!(ratio < 100.0, "Sparsity ratio should be < 100");
    }

    #[test]
    fn flops_per_mac_is_two() {
        assert!((FLOPS_PER_MAC - 2.0).abs() < 1e-9);
    }
}

//! Latency primitives for the roofline model.
//!
//! Models three latency components:
//! - `t_compute`: time for matrix multiplications (linear in batch size)
//! - `t_mem_weights`: time to fetch model weights (independent of batch)
//! - `t_mem_kv`: time to fetch KV cache (linear in batch size and context)
//!
//! The total latency is the maximum of compute and total memory time,
//! following the roofline principle: total time = max(compute, memory).

use ndarray::Array1;

use crate::constants;

/// Compute time for the forward pass.
///
/// # Formula
/// `t_compute = (batch_size × N_active × 2) / FLOPS`
///
/// The factor of 2 accounts for multiply-add operations counting as 2 FLOPs.
///
/// # Arguments
/// * `batch_size` - Batch size (1 or array of batch sizes)
/// * `n_active` - Active parameters per token (default: N_ACTIVE)
/// * `flops` - Aggregate FLOPS (default: FLOPS)
///
/// # Returns
/// Latency in seconds.
pub fn t_compute(
    batch_size: &Array1<f64>,
    n_active: f64,
    flops: f64,
) -> Array1<f64> {
    batch_size.mapv(|bs| bs * n_active * constants::FLOPS_PER_MAC / flops)
}

/// Time to fetch model weights from HBM.
///
/// # Formula
/// `t_mem_weights = (N_total × bytes_per_param) / BW`
///
/// This is independent of batch size — every forward pass must read every
/// weight at least once (modulo expert sparsity).
///
/// # Returns
/// Latency in seconds (same value broadcast to array shape of batch_size).
pub fn t_mem_weights(
    batch_size: &Array1<f64>,
    n_total: f64,
    bytes_per_param: f64,
    mem_bw: f64,
) -> Array1<f64> {
    let val = n_total * bytes_per_param / mem_bw;
    Array1::from_elem(batch_size.len(), val)
}

/// Time to fetch the KV cache for all sequences.
///
/// # Formula
/// `t_mem_kv = (batch_size × context_length × bytes_per_token) / BW`
///
/// # Arguments
/// * `batch_size` - Batch size (or array)
/// * `context_length` - Context length in tokens (scalar or same-shaped array)
/// * `bpt` - Bytes per token in KV cache
/// * `mem_bw` - Memory bandwidth in bytes/sec
///
/// # Returns
/// Latency in seconds.
pub fn t_mem_kv(
    batch_size: &Array1<f64>,
    context_length: &Array1<f64>,
    bpt: f64,
    mem_bw: f64,
) -> Array1<f64> {
    batch_size * context_length * bpt / mem_bw
}

/// Total latency: `max(t_compute, t_mem_weights + t_mem_kv)`.
///
/// Returns a tuple of `(total, t_compute, t_mem_weights, t_mem_kv)`.
pub fn total_latency(
    batch_size: &Array1<f64>,
    n_active: f64,
    n_total: f64,
    context_length: &Array1<f64>,
    bytes_per_param: f64,
    bpt: f64,
    flops: f64,
    mem_bw: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let tc = t_compute(batch_size, n_active, flops);
    let tw = t_mem_weights(batch_size, n_total, bytes_per_param, mem_bw);
    let tk = t_mem_kv(batch_size, context_length, bpt, mem_bw);
    let tm = &tw + &tk;

    let mut total = Array1::zeros(batch_size.len());
    for i in 0..total.len() {
        total[i] = tc[i].max(tm[i]);
    }

    (total, tc, tw, tk)
}

/// Find the balance batch size where compute time ≈ memory time.
///
/// This is the "ridge point" of the roofline. Below it you're memory-bound;
/// above it you're compute-bound.
pub fn find_balance_point(
    batch_sizes: &Array1<f64>,
    n_active: f64,
    n_total: f64,
    context_length: &Array1<f64>,
    bytes_per_param: f64,
    bpt: f64,
    flops: f64,
    mem_bw: f64,
) -> (usize, f64, f64) {
    let (_, tc, tw, tk) = total_latency(
        batch_sizes, n_active, n_total, context_length,
        bytes_per_param, bpt, flops, mem_bw,
    );
    let tm = &tw + &tk;
    let diff = (&tc - &tm).mapv(|d| d.abs());
    let idx = diff
        .indexed_iter()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (idx, batch_sizes[idx], tc[idx].max(tm[idx]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use constants::{
        BYTES_PER_PARAM_FP4, BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, CONTEXT_LENGTH,
        FLOPS, MEM_BW, N_ACTIVE, N_TOTAL,
    };
    use ndarray::arr1;

    // Helper: create an array of batch sizes from single values
    fn bs(vals: &[f64]) -> Array1<f64> {
        arr1(vals).mapv(|v| v as f64)
    }

    fn ctx_vals(vals: &[f64]) -> Array1<f64> {
        arr1(vals).mapv(|v| v as f64)
    }

    #[test]
    fn t_compute_scales_linearly() {
        let b1 = bs(&[1.0]);
        let b100 = bs(&[100.0]);
        let t1 = t_compute(&b1, N_ACTIVE, FLOPS);
        let t100 = t_compute(&b100, N_ACTIVE, FLOPS);
        // t_compute(100) should be 100× t_compute(1)
        assert_relative_eq!(t100[0], t1[0] * 100.0, epsilon = 1e-12);
    }

    #[test]
    fn t_compute_is_reasonable() {
        let b = bs(&[2000.0]); // typical production batch
        let t = t_compute(&b, N_ACTIVE, FLOPS);
        // Should be ~50 µs for 2000 sequences with 37B active params at 1.5 PFLOPs
        assert!(t[0] > 1e-6, "compute time too small: {}", t[0]);
        assert!(t[0] < 0.1, "compute time too large: {}", t[0]);
    }

    #[test]
    fn t_mem_weights_is_constant_across_batch() {
        let b1 = bs(&[1.0]);
        let b1000 = bs(&[1000.0]);
        let t1 = t_mem_weights(&b1, N_TOTAL, BYTES_PER_PARAM_FP8, MEM_BW);
        let t1000 = t_mem_weights(&b1000, N_TOTAL, BYTES_PER_PARAM_FP8, MEM_BW);
        assert_relative_eq!(t1[0], t1000[0], epsilon = 1e-12);
    }

    #[test]
    fn t_mem_weights_is_roughly_140ms() {
        let b = bs(&[1.0]);
        let t = t_mem_weights(&b, N_TOTAL, 0.5, MEM_BW);
        // 700B × 0.5 bytes / 5 TB/s = 70 ms for FP4
        // 700B × 1.0 bytes / 5 TB/s = 140 ms for FP8
        assert!(t[0] * 1000.0 > 60.0, "too fast: {} ms", t[0] * 1000.0);
        assert!(t[0] * 1000.0 < 150.0, "too slow: {} ms", t[0] * 1000.0);
    }

    #[test]
    fn t_mem_kv_scales_with_batch_and_context() {
        let b = bs(&[1000.0]);
        let ctx_short = ctx_vals(&[1024.0]);
        let ctx_long = ctx_vals(&[32768.0]);

        let t_short = t_mem_kv(&b, &ctx_short, BYTES_PER_TOKEN, MEM_BW);
        let t_long = t_mem_kv(&b, &ctx_long, BYTES_PER_TOKEN, MEM_BW);

        // 32× longer context should give 32× longer KV time
        let ratio = t_long[0] / t_short[0];
        assert!((ratio - 32.0).abs() < 1.0, "ratio should be ~32, got {ratio}");
    }

    #[test]
    fn total_latency_at_batch_1_is_memory_bound() {
        let b = bs(&[1.0]);
        let ctx = ctx_vals(&[CONTEXT_LENGTH]);
        let (total, tc, tw, tk) = total_latency(
            &b, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
        );
        let tm = tw[0] + tk[0];
        // At batch=1, memory time should dominate
        assert!(tm > tc[0], "should be memory-bound at batch=1");
        assert_relative_eq!(total[0], tm, epsilon = 1e-12);
    }

    #[test]
    fn total_latency_at_large_batch_is_compute_bound() {
        let b = bs(&[100_000.0]);
        let ctx = ctx_vals(&[1024.0]); // short context
        let (_total, tc, tw, tk) = total_latency(
            &b, N_ACTIVE, N_TOTAL, &ctx,
            0.1, 256.0, FLOPS, MEM_BW / 10.0, // make memory very fast
        );
        // With these params, compute should dominate at large batch
        assert!(tc[0] > tw[0] + tk[0], "should be compute-bound at large batch");
    }

    #[test]
    fn find_balance_point_is_between_extremes() {
        let batch_sizes = Array1::logspace(10.0, 0.0, 4.0, 200);
        let ctx = Array1::from_elem(200, CONTEXT_LENGTH);

        let (idx, bs, lat) = find_balance_point(
            &batch_sizes, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
        );

        assert!(idx > 0, "balance point should not be at first element");
        assert!(idx < 199, "balance point should not be at last element");
        assert!(bs > 10.0, "balance batch should be > 10, got {bs}");
        assert!(bs < 50_000.0, "balance batch should be < 50k, got {bs}");
        assert!(lat > 0.0);
    }

    #[test]
    fn total_latency_is_at_least_max_of_components() {
        let b = bs(&[1.0, 10.0, 100.0, 1000.0, 10000.0]);
        let ctx = ctx_vals(&[CONTEXT_LENGTH; 5]);
        let (total, tc, tw, tk) = total_latency(
            &b, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
        );

        for i in 0..5 {
            let tm = tw[i] + tk[i];
            let expected_max = tc[i].max(tm);
            assert_relative_eq!(total[i], expected_max, epsilon = 1e-12);
        }
    }

    #[test]
    fn latency_consistency_across_defaults() {
        // Verify that using constants gives consistent, physically sensible results
        let b = bs(&[2000.0]);
        let ctx = ctx_vals(&[CONTEXT_LENGTH]);
        let (total, tc, tw, tk) = total_latency(
            &b, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
        );

        // With defaults, memory should dominate at 32K context / 2K batch
        let tm = tw[0] + tk[0];
        assert!(tm > tc[0], "default params should be memory-bound");
        assert!(total[0] * 1000.0 < 500.0, "latency unreasonably high");
        assert!(total[0] * 1000.0 > 10.0, "latency unreasonably low");
    }

    #[test]
    fn fp4_halves_weight_fetch_time_vs_fp8() {
        let b = bs(&[1.0]);
        let _ctx = ctx_vals(&[CONTEXT_LENGTH]);

        let tw_fp8 = t_mem_weights(&b, N_TOTAL, BYTES_PER_PARAM_FP8, MEM_BW);
        let tw_fp4 = t_mem_weights(&b, N_TOTAL, BYTES_PER_PARAM_FP4, MEM_BW);

        assert_relative_eq!(tw_fp8[0], tw_fp4[0] * 2.0, epsilon = 1e-12);
    }
}

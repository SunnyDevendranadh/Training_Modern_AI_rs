//! Cost and economics computations.
//!
//! Cost models: rack rental, cost per token, and the Pareto frontier
//! between latency and cost that drives API pricing tiers.

use ndarray::Array1;

use crate::constants;
use crate::latency::total_latency;

/// Rack cost per second.
///
/// # Formula
/// `(gpu_cost_per_hour × gpus) / 3600`
pub fn rack_cost_per_sec(gpu_cost_per_hour: f64, gpus: usize) -> f64 {
    gpu_cost_per_hour * gpus as f64 / 3600.0
}

/// Cost per million tokens.
///
/// # Formula
/// `(total_latency × rack_cost_per_sec) / batch_size × 1e6`
///
/// Cost is proportional to GPU time. Bigger batches amortize weight fetch
/// over more tokens, reducing per-token cost.
pub fn cost_per_million_tokens(
    batch_size: &Array1<f64>,
    n_active: f64,
    n_total: f64,
    context_length: &Array1<f64>,
    bytes_per_param: f64,
    bpt: f64,
    flops: f64,
    mem_bw: f64,
    rack_cost: f64,
) -> Array1<f64> {
    let (total, _, _, _) = total_latency(
        batch_size, n_active, n_total, context_length,
        bytes_per_param, bpt, flops, mem_bw,
    );
    total * rack_cost / batch_size * 1e6
}

/// Minimum achievable cost per million tokens (the compute floor).
///
/// At infinite batch size, cost approaches the compute cost alone.
pub fn compute_cost_floor(
    n_active: f64,
    flops: f64,
    rack_cost: f64,
) -> f64 {
    // At infinite batch: time per token = n_active * 2 / flops
    // cost = time * rack_cost
    n_active * constants::FLOPS_PER_MAC / flops * rack_cost * 1e6
}

/// Cost ratio between decode (batch_eff=1) and a prefill pass of given length.
///
/// This models why output tokens cost ~5× input tokens.
pub fn decode_prefill_cost_ratio(
    prefill_pass_length: f64,
    batch_size: f64,
    n_active: f64,
    n_total: f64,
    context_length: f64,
    bytes_per_param: f64,
    bpt: f64,
    flops: f64,
    mem_bw: f64,
    rack_cost: f64,
) -> f64 {
    let bs_decode = Array1::from_elem(1, batch_size);
    let cl_decode = Array1::from_elem(1, context_length);
    let cost_decode = cost_per_million_tokens(
        &bs_decode, n_active, n_total, &cl_decode,
        bytes_per_param, bpt, flops, mem_bw, rack_cost,
    );

    // Prefill: pass_length tokens processed in parallel, cost amortized
    let cost_prefill_per_token = cost_decode[0] / prefill_pass_length
        .min(bs_decode[0].max(1.0));

    cost_decode[0] / cost_prefill_per_token.max(1e-12)
}

/// Compute the Pareto frontier tuple (cost, latency) for a range of batch sizes.
///
/// Useful for plotting the "Fast Mode" vs "Slow Mode" trade-off.
pub fn pareto_curve(
    batch_sizes: &Array1<f64>,
    n_active: f64,
    n_total: f64,
    context_length: &Array1<f64>,
    bytes_per_param: f64,
    bpt: f64,
    flops: f64,
    mem_bw: f64,
    rack_cost: f64,
) -> (Array1<f64>, Array1<f64>) {
    let cost = cost_per_million_tokens(
        batch_sizes, n_active, n_total, context_length,
        bytes_per_param, bpt, flops, mem_bw, rack_cost,
    );
    let (total, _, _, _) = total_latency(
        batch_sizes, n_active, n_total, context_length,
        bytes_per_param, bpt, flops, mem_bw,
    );
    (cost, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use constants::{
        BYTES_PER_TOKEN, CONTEXT_LENGTH, FLOPS, GPU_COST_PER_HOUR, GPUS_IN_RACK,
        MEM_BW, N_ACTIVE, N_TOTAL,
    };
    use constants::BYTES_PER_PARAM_FP8;

    fn default_rack_cost() -> f64 {
        rack_cost_per_sec(GPU_COST_PER_HOUR, GPUS_IN_RACK)
    }

    #[test]
    fn rack_cost_per_sec_calculation() {
        let cost = rack_cost_per_sec(2.0, 72);
        // 2 * 72 / 3600 = 144 / 3600 = 0.04
        assert_relative_eq!(cost, 0.04, epsilon = 1e-9);
    }

    #[test]
    fn rack_cost_per_sec_with_different_rates() {
        let cost = rack_cost_per_sec(3.0, 64);
        assert_relative_eq!(cost, 3.0 * 64.0 / 3600.0, epsilon = 1e-9);
    }

    #[test]
    fn cost_decreases_with_larger_batches() {
        let bs_small = Array1::from_elem(1, 10.0);
        let bs_large = Array1::from_elem(1, 2000.0);
        let ctx = Array1::from_elem(1, CONTEXT_LENGTH);
        let rc = default_rack_cost();

        let cost_small = cost_per_million_tokens(
            &bs_small, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );
        let cost_large = cost_per_million_tokens(
            &bs_large, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );

        assert!(cost_small[0] > cost_large[0],
            "cost should decrease with larger batch: small={}, large={}",
            cost_small[0], cost_large[0]);
    }

    #[test]
    fn cost_at_batch_1_is_high() {
        let bs = Array1::from_elem(1, 1.0);
        let ctx = Array1::from_elem(1, CONTEXT_LENGTH);
        let rc = default_rack_cost();

        let cost = cost_per_million_tokens(
            &bs, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );

        // At batch=1, cost should be very high (no amortization)
        assert!(cost[0] > 1.0, "cost at batch=1 should be > $1/M tokens, got ${:.4}", cost[0]);
    }

    #[test]
    fn compute_cost_floor_positive() {
        let rc = default_rack_cost();
        let floor = compute_cost_floor(N_ACTIVE, FLOPS, rc);
        assert!(floor > 0.0);
        // Floor should be relatively small — raw compute is cheap
        assert!(floor < 10.0, "compute floor suspiciously high: ${:.6}", floor);
    }

    #[test]
    fn cost_never_below_compute_floor() {
        let bs = Array1::logspace(10.0, 0.0, 5.0, 200);
        let ctx = Array1::from_elem(200, 1024.0); // short context to approach floor
        let rc = default_rack_cost();
        let floor = compute_cost_floor(N_ACTIVE, FLOPS, rc);

        let cost = cost_per_million_tokens(
            &bs, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );

        let min_cost = cost.iter().cloned().fold(f64::INFINITY, f64::min);
        // Cost should be above floor (or very close, allowing for numerical issues)
        assert!(min_cost >= floor * 0.9,
            "cost floor violated: min={min_cost:.6}, floor={floor:.6}");
    }

    #[test]
    fn decode_prefill_ratio_greater_than_one() {
        let ratio = decode_prefill_cost_ratio(
            2048.0, 64.0, N_ACTIVE, N_TOTAL, CONTEXT_LENGTH,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
            default_rack_cost(),
        );
        // Decode should cost more per token than prefill
        assert!(ratio > 1.0, "decode should cost more than prefill per token");
    }

    #[test]
    fn decode_cost_ratio_in_reasonable_range() {
        let ratio = decode_prefill_cost_ratio(
            2048.0, 64.0, N_ACTIVE, N_TOTAL, CONTEXT_LENGTH,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW,
            default_rack_cost(),
        );
        // Typical API pricing is 3-50× for output vs input.
        // With our params, memory-bandwidth-bound decode gives a high ratio.
        assert!(ratio > 2.0, "ratio too small: {ratio}");
        assert!(ratio < 100.0, "ratio implausibly large: {ratio}");
    }

    #[test]
    fn pareto_curve_returns_matching_lengths() {
        let bs = Array1::from_vec(vec![1.0, 10.0, 100.0, 1000.0]);
        let ctx = Array1::from_elem(4, CONTEXT_LENGTH);
        let rc = default_rack_cost();

        let (cost, latency) = pareto_curve(
            &bs, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );
        assert_eq!(cost.len(), 4);
        assert_eq!(latency.len(), 4);
    }

    #[test]
    fn cost_scales_roughly_with_batch_inverse() {
        // At small batch where memory dominates, cost ~ 1/batch
        let bs1 = Array1::from_elem(1, 100.0);
        let bs2 = Array1::from_elem(1, 200.0);
        let ctx = Array1::from_elem(1, CONTEXT_LENGTH);
        let rc = default_rack_cost();

        let c1 = cost_per_million_tokens(
            &bs1, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );
        let c2 = cost_per_million_tokens(
            &bs2, N_ACTIVE, N_TOTAL, &ctx,
            BYTES_PER_PARAM_FP8, BYTES_PER_TOKEN, FLOPS, MEM_BW, rc,
        );

        let ratio = c1[0] / c2[0];
        // Should be roughly 2 (inverse of batch ratio) when memory-bound
        assert!(ratio > 1.5, "cost should scale inversely with batch at low batch");
        assert!(ratio < 3.0, "batch scaling seems off: ratio={}", ratio);
    }
}

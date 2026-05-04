//! Pipeline parallelism bubble analysis.
//!
//! Computes the pipeline bubble ratio for training (inference has no bubbles).

/// Compute the training pipeline bubble ratio.
///
/// # Formula
/// `bubble_ratio = (S - 1) / (S - 1 + M)`
///
/// where S = number of pipeline stages, M = number of micro-batches.
///
/// In inference there are no bubbles because the next batch starts the
/// moment a stage frees up. In training, all forward passes must complete
/// before backward can begin, creating idle GPU time.
pub fn bubble_ratio(num_stages: usize, num_micro_batches: usize) -> f64 {
    let s = num_stages as f64;
    let m = num_micro_batches as f64;
    (s - 1.0) / (s - 1.0 + m)
}

/// Effective throughput for inference pipeline (no bubbles).
///
/// With S stages and batch size B, the pipeline processes B tokens at
/// the rate of the slowest stage (since next batch starts immediately).
pub fn inference_throughput(
    batch_size: f64,
    _num_stages: usize,
    stage_time: f64,
) -> f64 {
    // Token throughput: B / stage_time per forward pass
    // No bubble — stages are continuously utilized
    batch_size / stage_time
}

/// Training throughput considering the bubble.
///
/// Only `(1 - bubble_ratio)` of the pipeline is actually productive.
pub fn training_throughput(
    batch_size: f64,
    num_stages: usize,
    num_micro_batches: usize,
    stage_time: f64,
) -> f64 {
    let effective = 1.0 - bubble_ratio(num_stages, num_micro_batches);
    batch_size / stage_time * effective
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_stage_has_no_bubble() {
        assert_relative_eq!(bubble_ratio(1, 1), 0.0, epsilon = 1e-9);
        assert_relative_eq!(bubble_ratio(1, 10), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn many_micro_batches_reduce_bubble() {
        let b1 = bubble_ratio(4, 1);
        let b10 = bubble_ratio(4, 10);
        let b100 = bubble_ratio(4, 100);
        assert!(b1 > b10);
        assert!(b10 > b100);
    }

    #[test]
    fn bubble_converges_to_zero_with_infinite_micro_batches() {
        let b = bubble_ratio(8, 10_000);
        assert!(b < 0.001);
    }

    #[test]
    fn classic_1f1b_bubble_rate() {
        // 4 stages, 4 micro-batches gives classic bubble
        let b = bubble_ratio(4, 4);
        // (4-1)/(4-1+4) = 3/7 ≈ 0.429
        assert_relative_eq!(b, 3.0 / 7.0, epsilon = 1e-9);
    }

    #[test]
    fn inference_throughput_equals_training_with_one_stage() {
        let tp = training_throughput(1000.0, 1, 4, 0.01);
        let ip = inference_throughput(1000.0, 1, 0.01);
        assert_relative_eq!(tp, ip, epsilon = 1e-9);
    }

    #[test]
    fn training_always_less_or_equal_throughput() {
        for s in 2..=8 {
            for mb in &[1, 2, 4, 8, 16] {
                let ip = inference_throughput(1000.0, s, 0.01);
                let tp = training_throughput(1000.0, s, *mb, 0.01);
                assert!(tp <= ip + 1e-9,
                    "training throughput ({tp}) should <= inference ({ip}) at S={s}, M={mb}");
            }
        }
    }
}

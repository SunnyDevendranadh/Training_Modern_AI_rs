//! Throughput vs perfection: merge strategy trade-offs.
//!
//! Models the trade-off between traditional blocking and minimally-blocking
//! (agent-era) merge strategies when agent PR output far exceeds human
//! review capacity.

/// Throughput under traditional blocking review.
///
/// Every PR must be human-reviewed before merge. Returns (merged_clean, stuck_in_queue).
pub fn throughput_blocking(
    prs_per_day: f64,
    review_hours_per_pr: f64,
    num_reviewers: usize,
    error_rate: f64,
) -> (f64, f64) {
    let max_reviewable = 24.0 / review_hours_per_pr;
    let human_capacity = num_reviewers as f64 * max_reviewable;
    let merged_clean = prs_per_day.min(human_capacity) * (1.0 - error_rate);
    let stuck = (prs_per_day - human_capacity).max(0.0);
    (merged_clean, stuck)
}

/// Throughput under minimally-blocking merge.
///
/// Auto-merge if tests pass, fix errors later. Returns (merged, auto_fixed, remaining_errors).
pub fn throughput_minimally_blocking(
    prs_per_day: f64,
    test_pass_rate: f64,
    error_rate: f64,
    auto_fix_success: f64,
) -> (f64, f64, f64) {
    let merged = prs_per_day * test_pass_rate;
    let errors = merged * error_rate;
    let auto_fixed = errors * auto_fix_success;
    let remaining = errors * (1.0 - auto_fix_success);
    (merged, auto_fixed, remaining)
}

/// Net healthy PRs for minimally-blocking strategy.
pub fn net_fast_throughput(
    prs_per_day: f64,
    test_pass_rate: f64,
    error_rate: f64,
    auto_fix_success: f64,
) -> f64 {
    let (merged, _, remaining) = throughput_minimally_blocking(
        prs_per_day, test_pass_rate, error_rate, auto_fix_success,
    );
    merged - remaining
}

/// Speedup: minimally-blocking vs traditional.
pub fn throughput_speedup(
    prs_per_day: f64,
    review_hours_per_pr: f64,
    num_reviewers: usize,
    error_rate: f64,
    test_pass_rate: f64,
    auto_fix_success: f64,
) -> f64 {
    let (traditional, _) = throughput_blocking(prs_per_day, review_hours_per_pr, num_reviewers, error_rate);
    let fast = net_fast_throughput(prs_per_day, test_pass_rate, error_rate, auto_fix_success);
    fast / traditional.max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn blocking_stuck_at_zero_when_under_capacity() {
        let (merged, stuck) = throughput_blocking(10.0, 4.0, 3, 0.05);
        assert!(stuck == 0.0);
        assert!(merged > 0.0);
    }

    #[test]
    fn blocking_bottlenecks_at_capacity() {
        // 3 reviewers × (24/4 = 6 max each) = 18 max
        let (merged, stuck) = throughput_blocking(100.0, 4.0, 3, 0.05);
        assert!(stuck > 0.0, "should have stuck PRs at 100/day");
        assert!(merged < 100.0);
    }

    #[test]
    fn minimally_blocking_scales_linearly() {
        let m10 = net_fast_throughput(10.0, 0.95, 0.05, 0.80);
        let m100 = net_fast_throughput(100.0, 0.95, 0.05, 0.80);
        // Should scale roughly 10×
        let ratio = m100 / m10;
        assert!(ratio > 9.0, "should scale ~linearly, got {ratio}");
    }

    #[test]
    fn speedup_grows_with_pr_volume() {
        let s10 = throughput_speedup(10.0, 4.0, 3, 0.05, 0.95, 0.80);
        let s1000 = throughput_speedup(1000.0, 4.0, 3, 0.05, 0.95, 0.80);
        assert!(s1000 > s10, "speedup should grow with PR volume");
    }

    #[test]
    fn zero_error_rate_means_no_remaining_errors() {
        let (_merged, auto_fixed, remaining) = throughput_minimally_blocking(100.0, 0.95, 0.0, 0.80);
        assert_relative_eq!(remaining, 0.0);
        assert_relative_eq!(auto_fixed, 0.0);
    }

    #[test]
    fn perfect_auto_fix_leaves_no_remaining() {
        let (_, _, remaining) = throughput_minimally_blocking(100.0, 0.95, 0.05, 1.0);
        assert_relative_eq!(remaining, 0.0);
    }
}

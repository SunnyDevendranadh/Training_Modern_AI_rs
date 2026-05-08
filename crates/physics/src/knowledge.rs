//! Knowledge decay and garbage collection simulation.
//!
//! Models how technical debt (entropy) accumulates in codebases
//! written by AI agents, and how different garbage collection
//! strategies control it.

/// Simulate entropy accumulation over `num_days` days.
///
/// # Arguments
/// * `num_days` - Number of days to simulate
/// * `prs_per_day` - Agent PRs generated per day
/// * `replication_rate` - Fraction of new code that replicates suboptimal patterns
/// * `natural_growth` - Daily entropy increase rate from existing patterns
/// * `gc_interval` - Days between garbage collection runs (0 = never)
/// * `gc_coverage` - Fraction of entropy caught per GC run
///
/// # Returns
/// Vector of entropy levels, one per day.
pub fn simulate_entropy(
    num_days: usize,
    prs_per_day: usize,
    replication_rate: f64,
    natural_growth: f64,
    gc_interval: usize,
    gc_coverage: f64,
) -> Vec<f64> {
    let mut entropy = vec![0.0; num_days];

    for day in 0..num_days {
        let prev = if day > 0 { entropy[day - 1] } else { 0.0 };

        // Daily entropy from new PRs
        let new_entropy = prs_per_day as f64 * replication_rate;

        // Growth from existing patterns reinforcing
        let growth = prev * natural_growth;

        entropy[day] = prev + new_entropy + growth;

        // Garbage collection
        if gc_interval > 0 && day > 0 && day % gc_interval == 0 {
            let cleaned = entropy[day] * gc_coverage;
            entropy[day] -= cleaned;
        }
    }

    entropy
}

/// Compute final entropy statistics.
pub struct EntropyStats {
    pub final_entropy: f64,
    pub peak_entropy: f64,
    pub entropy_per_pr: f64,
    pub reduction_vs_no_gc: f64,
}

pub fn entropy_stats(entropy: &[f64], no_gc_entropy: &[f64], total_prs: usize) -> EntropyStats {
    let final_val = *entropy.last().unwrap_or(&0.0);
    let peak = entropy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let per_pr = final_val / total_prs.max(1) as f64;
    let no_gc_final = *no_gc_entropy.last().unwrap_or(&1.0);
    let reduction = if no_gc_final > 0.0 {
        1.0 - final_val / no_gc_final
    } else {
        0.0
    };

    EntropyStats {
        final_entropy: final_val,
        peak_entropy: peak,
        entropy_per_pr: per_pr,
        reduction_vs_no_gc: reduction,
    }
}

/// AGENTS.md effectiveness model (Vercel: passive context vs active retrieval).
///
/// Returns success rate (0-100) for passive (auto-injected) context.
pub fn agents_md_passive_effectiveness(
    size_kb: f64,
    optimal_kb: f64,
    sigma: f64,
    context_window_tokens: f64,
    tokens_per_kb: f64,
    task_complexity_tokens: f64,
) -> f64 {
    if size_kb <= 0.0 {
        return 0.0;
    }

    let log_ratio = (size_kb / optimal_kb).ln();
    let passive_score = 100.0 * (-log_ratio * log_ratio / (2.0 * sigma * sigma)).exp();

    let tokens_used = size_kb * tokens_per_kb + task_complexity_tokens;
    let pollution_penalty = (1.0 - tokens_used / context_window_tokens).max(0.0);

    passive_score * pollution_penalty
}

/// AGENTS.md active (skills-based) effectiveness.
pub fn agents_md_active_effectiveness(size_kb: f64, optimal_kb: f64, sigma: f64) -> f64 {
    if size_kb <= 0.0 {
        return 40.0;
    }

    let log_ratio = (size_kb / optimal_kb).ln();
    79.0 * (-log_ratio * log_ratio / (2.0 * sigma * sigma)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn entropy_without_gc_grows_linearly() {
        let ent = simulate_entropy(100, 20, 0.03, 0.0, 0, 0.0);
        // Without growth and GC, entropy should be linear: day * new_per_day
        let expected_per_day = 20.0 * 0.03; // 0.6 per day
        for (day, &e) in ent.iter().enumerate() {
            assert_relative_eq!(e, (day + 1) as f64 * expected_per_day, epsilon = 1e-9);
        }
    }

    #[test]
    fn gc_reduces_entropy() {
        let ent_no_gc = simulate_entropy(30, 20, 0.03, 0.001, 0, 0.0);
        let ent_weekly = simulate_entropy(30, 20, 0.03, 0.001, 7, 0.6);

        // Weekly GC should have lower final entropy
        assert!(
            ent_weekly.last().unwrap() < ent_no_gc.last().unwrap(),
            "GC should reduce entropy"
        );
    }

    #[test]
    fn continuous_gc_best() {
        let strategies = [
            (simulate_entropy(90, 20, 0.03, 0.001, 0, 0.0), "none"),
            (simulate_entropy(90, 20, 0.03, 0.001, 7, 0.6), "weekly"),
            (simulate_entropy(90, 20, 0.03, 0.001, 1, 0.5), "daily"),
            (simulate_entropy(90, 20, 0.03, 0.001, 1, 0.9), "continuous"),
        ];

        let continuous = strategies[3].0.last().unwrap();
        for (ent, name) in &strategies[..3] {
            assert!(
                continuous < ent.last().unwrap(),
                "continuous should beat {name}"
            );
        }
    }

    #[test]
    fn agents_md_peaks_at_optimal_size() {
        let opt = agents_md_passive_effectiveness(8.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);
        let too_small = agents_md_passive_effectiveness(1.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);
        let too_large = agents_md_passive_effectiveness(50.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);

        assert!(opt > too_small, "8 KB should beat 1 KB");
        assert!(opt > too_large, "8 KB should beat 50 KB");
    }

    #[test]
    fn passive_beats_active_at_optimal() {
        let passive = agents_md_passive_effectiveness(8.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);
        let active = agents_md_active_effectiveness(10.0, 10.0, 1.5);

        // Vercel finding: passive hits 100%, active maxes at ~79%
        assert!(passive > active, "passive should beat active");
    }

    #[test]
    fn context_pollution_penalty_kicks_in() {
        let small = agents_md_passive_effectiveness(8.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);
        let huge = agents_md_passive_effectiveness(500.0, 8.0, 1.2, 128000.0, 150.0, 5000.0);

        // 500 KB should be severely polluted
        assert!(
            huge < small * 0.5,
            "500 KB should have heavy pollution penalty"
        );
    }
}

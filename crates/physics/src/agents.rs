//! Agent coordination models — Cursor's 4 multi-agent strategies.
//!
//! Based on Cursor's experimental results scaling from 1 to 100 agents:
//! 1. Equal Self-Coordination → catastrophic lock contention
//! 2. Pipeline (Planner-Exec-Worker-Judge) → bottlenecked by slowest stage
//! 3. Continuous Executor → role overload, pathological regression
//! 4. Recursive Planner+Worker → near-linear scaling ★

/// Coordination overhead for equal self-coordination.
///
/// Degrades rapidly: 20 agents → 1-3 effective throughput.
pub fn overhead_equal(num_agents: usize) -> f64 {
    let n = num_agents as f64;
    let lock_contention = 1.0 / (1.0 + 0.3 * n);
    let risk_aversion = (-0.05 * n).exp();
    lock_contention * risk_aversion
}

/// Coordination overhead for pipeline strategy.
///
/// Works better but saturates due to pipeline bottleneck.
pub fn overhead_pipeline(num_agents: usize) -> f64 {
    let n = num_agents as f64;
    let base = 0.7 * (1.0 - (-0.1 * n).exp()) + 0.3;
    let bottleneck = 1.0 / (1.0 + 0.01 * n);
    base * bottleneck
}

/// Coordination overhead for continuous executor.
///
/// Initially flexible, then regresses due to role overload.
pub fn overhead_continuous(num_agents: usize) -> f64 {
    let n = num_agents as f64;
    if n <= 10.0 {
        0.8 * (1.0 - (-0.3 * n).exp()) + 0.2
    } else {
        0.5 * (-0.02 * (n - 10.0)).exp()
    }
}

/// Coordination overhead for recursive planner+worker.
///
/// Near-linear scaling — the only strategy that scales.
pub fn overhead_recursive(num_agents: usize) -> f64 {
    let n = num_agents as f64;
    let base = 0.85;
    let tax = 1.0 / (1.0 + 0.005 * (n + 1.0).ln());
    base * tax
}

/// Effective throughput = N × overhead.
pub fn effective_throughput(
    num_agents: usize,
    strategy: crate::types::CoordinationStrategy,
) -> f64 {
    let n = num_agents as f64;
    let overhead = match strategy {
        crate::types::CoordinationStrategy::Equal => overhead_equal(num_agents),
        crate::types::CoordinationStrategy::Pipeline => overhead_pipeline(num_agents),
        crate::types::CoordinationStrategy::Continuous => overhead_continuous(num_agents),
        crate::types::CoordinationStrategy::Recursive => overhead_recursive(num_agents),
    };
    n * overhead
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CoordinationStrategy;

    #[test]
    fn single_agent_is_fully_effective() {
        // One agent working alone should be highly effective
        // (Pipeline has lower efficiency at N=1 — designed for multi-agent)
        for strategy in [CoordinationStrategy::Equal, CoordinationStrategy::Recursive] {
            let eff = effective_throughput(1, strategy);
            assert!(
                eff > 0.70,
                "{strategy}: single agent should be > 0.70 effective, got {eff}"
            );
        }
    }

    #[test]
    fn equal_strategy_collapses_at_20() {
        let eff = effective_throughput(20, CoordinationStrategy::Equal);
        // Should be around 1-3 effective agents (Cursor finding)
        assert!(eff < 5.0, "equal should collapse: {eff} effective at N=20");
        assert!(eff > 0.5, "equal should not go to zero: {eff} at N=20");
    }

    #[test]
    fn recursive_best_at_scale() {
        let eff_r = effective_throughput(100, CoordinationStrategy::Recursive);
        let eff_e = effective_throughput(100, CoordinationStrategy::Equal);
        let eff_p = effective_throughput(100, CoordinationStrategy::Pipeline);
        let eff_c = effective_throughput(100, CoordinationStrategy::Continuous);

        assert!(eff_r > eff_e, "recursive should beat equal");
        assert!(eff_r > eff_p, "recursive should beat pipeline");
        assert!(eff_r > eff_c, "recursive should beat continuous");
    }

    #[test]
    fn continuous_regresses_after_10() {
        let eff_10 = effective_throughput(10, CoordinationStrategy::Continuous);
        let eff_20 = effective_throughput(20, CoordinationStrategy::Continuous);
        // Beyond 10, effective throughput should regress
        assert!(
            eff_20 / 20.0 < eff_10 / 10.0,
            "continuous should have lower efficiency at 20 than 10"
        );
    }

    #[test]
    fn throughput_monotonic_for_recursive() {
        // Recursive should be monotonic (more agents → more throughput)
        let mut prev = effective_throughput(1, CoordinationStrategy::Recursive);
        for n in 2..=100 {
            let curr = effective_throughput(n, CoordinationStrategy::Recursive);
            assert!(
                curr > prev,
                "recursive should be monotonic: N={n} ({curr}) <= N={} ({prev})",
                n - 1
            );
            prev = curr;
        }
    }
}

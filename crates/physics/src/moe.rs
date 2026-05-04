//! MoE routing and all-to-all communication simulation.

use ndarray::Array2;
use rand::Rng;

/// Route tokens to experts.
///
/// Each token is randomly assigned to `top_k` distinct experts.
/// Returns a boolean routing matrix of shape `(num_tokens, num_experts)`.
pub fn route_tokens(
    num_tokens: usize,
    num_experts: usize,
    top_k: usize,
    rng: &mut impl Rng,
) -> Array2<i32> {
    let mut routing = Array2::zeros((num_tokens, num_experts));
    for i in 0..num_tokens {
        let mut chosen = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let expert = rng.gen_range(0..num_experts);
            if !chosen.contains(&expert) {
                chosen.push(expert);
            }
        }
        for &e in &chosen {
            routing[[i, e]] = 1;
        }
    }
    routing
}

/// Compute all-to-all traffic matrix from token routing.
///
/// Returns a matrix of shape `(num_gpus, num_gpus)` where entry `(i, j)`
/// is the number of tokens routed from GPU i to GPU j.
pub fn compute_traffic(
    routing: &Array2<i32>,
    num_gpus: usize,
    experts_per_gpu: usize,
) -> Array2<i32> {
    let num_tokens = routing.shape()[0];
    let num_experts = routing.shape()[1];
    let mut traffic = Array2::zeros((num_gpus, num_gpus));

    for token_idx in 0..num_tokens {
        let source_gpu = token_idx % num_gpus; // round-robin token distribution
        for expert_idx in 0..num_experts {
            if routing[[token_idx, expert_idx]] == 1 {
                let dest_gpu = expert_idx / experts_per_gpu;
                traffic[[source_gpu, dest_gpu]] += 1;
            }
        }
    }
    traffic
}

/// Compute load balance statistics for expert routing.
pub struct LoadBalanceStats {
    pub avg_tokens_per_expert: f64,
    pub max_tokens_per_expert: i32,
    pub min_tokens_per_expert: i32,
    pub coefficient_of_variation: f64,
}

pub fn load_balance_stats(routing: &Array2<i32>) -> LoadBalanceStats {
    let num_experts = routing.shape()[1];
    let expert_counts: Vec<f64> = (0..num_experts)
        .map(|e| routing.column(e).sum() as f64)
        .collect();

    let n = expert_counts.len() as f64;
    let avg = expert_counts.iter().sum::<f64>() / n;
    let max_val = expert_counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max) as i32;
    let min_val = expert_counts.iter().cloned().fold(f64::INFINITY, f64::min) as i32;
    let variance = expert_counts.iter().map(|&x| (x - avg).powi(2)).sum::<f64>() / n;
    let cv = variance.sqrt() / avg.max(1e-9);

    LoadBalanceStats {
        avg_tokens_per_expert: avg,
        max_tokens_per_expert: max_val,
        min_tokens_per_expert: min_val,
        coefficient_of_variation: cv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn route_tokens_gives_correct_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let routing = route_tokens(2048, 64, 2, &mut rng);
        assert_eq!(routing.shape(), &[2048, 64]);
    }

    #[test]
    fn each_token_gets_exactly_top_k_experts() {
        let mut rng = StdRng::seed_from_u64(42);
        let routing = route_tokens(100, 32, 2, &mut rng);

        for t in 0..100 {
            let sum: i32 = routing.row(t).sum();
            assert_eq!(sum, 2, "token {} should have exactly 2 experts", t);
        }
    }

    #[test]
    fn traffic_matrix_has_expected_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let routing = route_tokens(2048, 64, 2, &mut rng);
        let traffic = compute_traffic(&routing, 64, 1); // 1 expert per GPU
        assert_eq!(traffic.shape(), &[64, 64]);
    }

    #[test]
    fn load_balance_cv_between_zero_and_one() {
        let mut rng = StdRng::seed_from_u64(42);
        let routing = route_tokens(2048, 64, 2, &mut rng);
        let stats = load_balance_stats(&routing);

        assert!(stats.avg_tokens_per_expert > 0.0);
        assert!(stats.coefficient_of_variation >= 0.0);
        // With random routing and enough tokens, CV should be small
        assert!(stats.coefficient_of_variation < 0.5,
            "CV should be < 0.5 for uniform random, got {:.3}", stats.coefficient_of_variation);
    }

    #[test]
    fn routing_is_deterministic_with_same_seed() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let r1 = route_tokens(100, 32, 2, &mut rng1);
        let r2 = route_tokens(100, 32, 2, &mut rng2);

        for i in 0..100 {
            for j in 0..32 {
                assert_eq!(r1[[i, j]], r2[[i, j]]);
            }
        }
    }
}
